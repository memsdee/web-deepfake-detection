# inference.py  -- bản THẬT (DFD-FCG @3c0b840, ffg ViT-L/14)
import logging, os, shutil, subprocess, sys, time, uuid
from contextlib import contextmanager

import cv2
import torch

log = logging.getLogger("inference")

REPO = os.environ.get("DFD_REPO", "/kaggle/working/DFD-FCG")
CKPT = os.environ.get(
    "DFD_CKPT",
    "/kaggle/input/notebooks/nhalac/dfd-train-2/ckpt_run2/epoch=26-step=1377.ckpt",
)
FACE_FEAT = os.path.join(REPO, "misc/L14_real_semantic_patches_v4_2000.pickle")
MEAN_FACE = "misc/20words_mean_face.npy"

CLIP_SEC = 3.0
NUM_FRAMES = 10
MAX_CLIPS = 5          # 15 s -> 5 clip
FWD_CHUNK = 2          # số clip mỗi lần forward (tránh tràn VRAM)
STEP_TIMEOUT = 300     # giây cho mỗi bước tiền xử lý
TMP_ROOT = "/dev/shm/dfd_infer" if os.path.isdir("/dev/shm") else "/tmp/dfd_infer"


class NoFaceError(Exception):
    """Không tìm thấy khuôn mặt."""


class InferenceError(Exception):
    """Lỗi khác khi xử lý."""


@contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _run(cmd, env, stdin=None):
    r = subprocess.run(cmd, cwd=REPO, env=env, input=stdin, text=True,
                       capture_output=True, timeout=STEP_TIMEOUT)
    if r.returncode != 0:
        log.error("Lệnh lỗi %s\n%s", cmd[2], (r.stdout + r.stderr)[-1500:])
        raise InferenceError(f"preprocess failed: {cmd[2]}")
    return r.stdout + r.stderr


class InferenceEngine:
    def __init__(self, device: str):
        if not device.startswith("cuda"):
            raise InferenceError("Bản thật cần GPU (face-alignment hard-code cuda).")
        self.device = device
        self.gpu_index = device.split(":")[1] if ":" in device else "0"
        os.makedirs(TMP_ROOT, exist_ok=True)

        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        with _cwd(REPO):  # repo/CLIP có thể đọc file theo đường dẫn tương đối
            from src.model.clip.svl import FFGSynoVideoLearner
            self.model = FFGSynoVideoLearner.load_from_checkpoint(
                CKPT, map_location="cpu", face_feature_path=FACE_FEAT)
        self.model.to(device).eval()
        log.info("Đã nạp model trên %s", device)

    # ---------- tiền xử lý ----------
    def _preprocess(self, video_path, job):
        ext = os.path.splitext(video_path)[1].lower() or ".mp4"
        vdir = os.path.join(job, "videos")
        os.makedirs(vdir)
        os.symlink(os.path.abspath(video_path), os.path.join(vdir, "clip" + ext))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu_index, PYTHONUNBUFFERED="1")

        t0 = time.perf_counter()
        _run([sys.executable, "-u", "-m", "src.preprocess.fetch_landmark_bbox",
              f"--root-dir={job}", "--video-dir=videos", "--fdata-dir=frame_data",
              "--glob-exp=*", "--batch=16", "--max-res=800"], env, stdin="y\n")
        t_detect = time.perf_counter() - t0
        if not os.path.exists(os.path.join(job, "frame_data", "clip.pickle")):
            raise InferenceError("không tạo được frame_data")

        t0 = time.perf_counter()
        _run([sys.executable, "-m", "src.preprocess.crop_main_face",
              f"--root-dir={job}", "--video-dir=videos", "--fdata-dir=frame_data",
              "--crop-dir=cropped", "--glob-exp=*", "--crop-size=150",
              f"--mean-face={MEAN_FACE}", "--workers=0"], env)
        t_crop = time.perf_counter() - t0

        avi = os.path.join(job, "cropped", "videos", "clip.avi")
        # crop_main_face lỗi vẫn thoát mã 0: không có file = không có mặt / mặt theo dõi < 90% frame
        if not os.path.exists(avi):
            raise NoFaceError()
        return avi, t_detect, t_crop

    # ---------- lấy clip giống get_entity (không random speed) ----------
    @staticmethod
    def _load_clips(avi):
        cap = cv2.VideoCapture(avi)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        n = len(frames)
        if n == 0:
            raise NoFaceError()
        duration = n / fps
        n_clips = min(int(duration // CLIP_SEC), MAX_CLIPS)
        if n_clips == 0:
            if duration < 2.0:
                raise InferenceError("video quá ngắn")
            n_clips = 1
        samples = int(fps * CLIP_SEC)
        stride = ((samples - 1) / (NUM_FRAMES - 1)) / fps
        clips = []
        for c in range(n_clips):
            off = int(c * CLIP_SEC)
            idxs = [min(max(int(round((off + i * stride) * fps)), 0), n - 1)
                    for i in range(NUM_FRAMES)]
            clips.append(torch.stack(
                [torch.from_numpy(frames[i]).permute(2, 0, 1).contiguous() for i in idxs]))
        return clips, duration

    # ---------- suy luận ----------
    @torch.inference_mode()
    def _score(self, clips):
        x = torch.stack([self.model.transform(c) for c in clips])  # [N,10,3,224,224]
        probs = []
        for i in range(0, len(x), FWD_CHUNK):
            xb = x[i:i + FWD_CHUNK].to(self.device)
            masks = torch.ones(xb.shape[:2], dtype=torch.bool, device=self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model.evaluate(xb, masks=masks)
            probs += out["logits"].float().softmax(dim=-1)[:, 1].cpu().tolist()
        return probs

    def predict(self, video_path: str) -> dict:
        t_all = time.perf_counter()
        job = os.path.join(TMP_ROOT, uuid.uuid4().hex)
        try:
            avi, t_detect, t_crop = self._preprocess(video_path, job)
            t0 = time.perf_counter()
            clips, duration = self._load_clips(avi)
            probs = self._score(clips)
            t_model = time.perf_counter() - t0
        except (NoFaceError, InferenceError):
            raise
        except Exception as e:
            log.exception("predict lỗi")
            raise InferenceError(str(e))
        finally:
            shutil.rmtree(job, ignore_errors=True)  # không giữ bản sao video

        return {
            "prob_fake": round(sum(probs) / len(probs), 4),  # trung bình các clip
            "clips": [{"t_start": round(i * CLIP_SEC, 2),
                       "t_end": round(min((i + 1) * CLIP_SEC, duration), 2),
                       "prob_fake": round(p, 4)} for i, p in enumerate(probs)],
            "n_clips": len(probs),
            "timings": {"detect": round(t_detect, 2), "crop": round(t_crop, 2),
                        "model": round(t_model, 2),
                        "total": round(time.perf_counter() - t_all, 2)},
        }