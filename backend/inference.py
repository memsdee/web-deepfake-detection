# inference.py -- bản THẬT cho TALL-Swin (rainy-xu/TALL4Deepfake)
import logging, os, sys, time

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("inference")

REPO = os.environ.get("TALL_REPO", "/kaggle/working/TALL4Deepfake")
CKPT = os.environ.get("TALL_CKPT", "")

# ---- phải bằng lệnh train của bạn ----
DURATION = 4          # --duration
THUMBNAIL_ROWS = 2    # --thumbnail_rows
INPUT_SIZE = 112      # --input_size
HPE_TO_TOKEN = True   # --hpe_to_token
NUM_CLIPS = 8         # --num_clips lúc test
NUM_CLASSES = 2
FAKE_INDEX = 1

BOX_SCALE = 1.3       # crop vuông = max(w,h)*1.3 quanh tâm mặt (kiểu FaceForensics)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MIN_CLIP_RATIO = 0.5


class NoFaceError(Exception):
    """Không tìm thấy khuôn mặt."""


class InferenceError(Exception):
    """Lỗi khác khi xử lý."""


class InferenceEngine:
    def __init__(self, device: str):
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise InferenceError("Bản thật cần GPU.")
        self.device = device
        torch.cuda.set_device(torch.device(device))

        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

        self.model = self._build_model()
        self.model.to(device).eval()
        log.info("Đã nạp TALL trên %s", device)

    def _build_model(self):
        if not CKPT or not os.path.exists(CKPT):
            raise InferenceError(f"không thấy checkpoint: {CKPT!r}")
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        import my_models  # noqa: F401
        from timm.models import create_model

        model = create_model(
            "TALL_SWIN", pretrained=False,
            duration=DURATION, hpe_to_token=HPE_TO_TOKEN, thumbnail_rows=THUMBNAIL_ROWS,
            num_classes=NUM_CLASSES,
        )
        # weights_only=False: checkpoint do chính mình train, có kèm argparse.Namespace (PyTorch >= 2.6)
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            log.warning("checkpoint có key thừa: %s", unexpected[:5])
        if missing:
            raise InferenceError(f"checkpoint không khớp model, thiếu: {missing[:5]}")
        return model

    @staticmethod
    def _plan(n_frames):
        sample_pos = max(1, 1 + n_frames - DURATION)
        starts = np.linspace(0, sample_pos - 1, num=NUM_CLIPS, dtype=int).tolist()
        return starts, [[(s + i) % n_frames for i in range(DURATION)] for s in starts]

    @staticmethod
    def _read_frames(cap, need):
        got, pos, k = {}, 0, 0
        while k < len(need):
            if not cap.grab():
                break
            if pos == need[k]:
                ok, f = cap.retrieve()
                if ok:
                    got[pos] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                k += 1
            pos += 1
        return got

    def _face_box(self, rgb):
        boxes, _ = self.mtcnn.detect(Image.fromarray(rgb))
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        H, W = rgb.shape[:2]
        size = int(max(x2 - x1, y2 - y1) * BOX_SCALE)
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        x, y = max(int(cx - size // 2), 0), max(int(cy - size // 2), 0)
        size = min(W - x, size)
        size = min(H - y, size)
        return (x, y, size) if size > 8 else None

    @staticmethod
    def _prep(rgb, box):
        x, y, s = box
        img = Image.fromarray(rgb[y:y + s, x:x + s]).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
        return a.transpose(2, 0, 1)

    @torch.inference_mode()
    def _logits(self, clips):
        x = torch.from_numpy(np.stack(clips)).to(self.device)
        return self.model(x).float()

    def predict(self, video_path: str) -> dict:
        t_all = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        try:
            try:
                if not cap.isOpened():
                    raise InferenceError("không mở được video")
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if n < DURATION:
                    raise InferenceError("không đọc được số frame của video")
                duration = n / fps
                if duration < 2.0:
                    raise InferenceError("video quá ngắn")

                t0 = time.perf_counter()
                starts, plan = self._plan(n)
                frames = self._read_frames(cap, sorted({i for p in plan for i in p}))
                boxes = {i: self._face_box(f) for i, f in frames.items()}
                t_detect = time.perf_counter() - t0
            finally:
                cap.release()

            t0 = time.perf_counter()
            kept, tensors = [], []
            for ci, idxs in enumerate(plan):
                ok = [i for i in idxs if boxes.get(i) is not None]
                if not ok:
                    continue
                per = [self._prep(frames[i], boxes[i] if boxes.get(i) is not None
                                  else boxes[min(ok, key=lambda j: abs(j - i))]) for i in idxs]
                tensors.append(np.concatenate(per, axis=0))
                kept.append(ci)
            t_crop = time.perf_counter() - t0
            if len(kept) < MIN_CLIP_RATIO * NUM_CLIPS:
                raise NoFaceError()

            t0 = time.perf_counter()
            logits = self._logits(tensors)
            clip_p = logits.softmax(dim=-1)[:, FAKE_INDEX].cpu().tolist()
            prob = logits.mean(dim=0).softmax(dim=-1)[FAKE_INDEX].item()
            t_model = time.perf_counter() - t0
        except (NoFaceError, InferenceError):
            raise
        except Exception as e:
            log.exception("predict lỗi")
            raise InferenceError(str(e))

        seg = duration / NUM_CLIPS
        return {
            "prob_fake": round(prob, 4),
            "clips": [{"t_start": round(ci * seg, 2), "t_end": round((ci + 1) * seg, 2),
                       "prob_fake": round(p, 4)} for ci, p in zip(kept, clip_p)],
            "n_clips": len(kept),
            "timings": {"detect": round(t_detect, 2), "crop": round(t_crop, 2),
                        "model": round(t_model, 2),
                        "total": round(time.perf_counter() - t_all, 2)},
        }