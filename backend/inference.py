# inference.py -- bản THẬT cho TALL-Swin (rainy-xu/TALL4Deepfake)
import base64
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
VIS_SIZE = 224
STRIP_SIZE = 128
SMOOTH_STEPS = 4
SMOOTH_SIGMA = 0.15


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

    def _logits_and_saliency(self, clips):
        """Chạy model lấy logits sạch và tính bản đồ SmoothGrad saliency theo từng frame."""
        batch = np.stack(clips)
        x_clean = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(x_clean).float()

        try:
            self.model.zero_grad(set_to_none=True)
            grad_acc = torch.zeros_like(x_clean)
            for _ in range(SMOOTH_STEPS):
                with torch.enable_grad():
                    x_noisy = (x_clean + torch.randn_like(x_clean) * SMOOTH_SIGMA).detach().requires_grad_(True)
                    out = self.model(x_noisy).float()
                    out[:, FAKE_INDEX].sum().backward()
                    grad_acc += x_noisy.grad.detach().abs()
                self.model.zero_grad(set_to_none=True)
            # (B, 3*DURATION, H, W) -> (B, DURATION, H, W)
            b, _, h, w = grad_acc.shape
            sal = (grad_acc / SMOOTH_STEPS).view(b, DURATION, 3, h, w).mean(dim=2)
            return logits, sal.cpu().numpy()
        except Exception as e:
            log.warning("Không tính được SmoothGrad saliency (%s), bỏ qua heatmap", e)
            return logits, None
        finally:
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _encode_jpeg_b64(bgr: np.ndarray, quality: int = 85) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    @classmethod
    def _sal_to_jet_b64(cls, raw_sal: np.ndarray, size: int, quality: int = 85) -> str:
        hi = float(np.percentile(raw_sal, 98))
        norm_sal = np.clip(raw_sal / (hi + 1e-8), 0.0, 1.0).astype(np.float32)
        up_sal = cv2.resize(norm_sal, (size, size), interpolation=cv2.INTER_CUBIC)
        k = 21 if size >= 200 else 13
        smooth_sal = cv2.GaussianBlur(up_sal, (k, k), 0)
        max_v = float(smooth_sal.max())
        if max_v > 1e-8:
            smooth_sal = np.clip(smooth_sal / max_v, 0.0, 1.0)
        smooth_sal = np.power(smooth_sal, 0.85)
        jet_bgr = cv2.applyColorMap(np.uint8(smooth_sal * 255), cv2.COLORMAP_JET)
        return cls._encode_jpeg_b64(jet_bgr, quality=quality)

    @classmethod
    def _render_clip_evidence(cls, frames, boxes, idxs, sal_clip, fps: float) -> dict:
        """Tạo ảnh khuôn mặt + lớp heatmap + dải 4 frame liên tiếp cho 1 clip."""
        ok_ids = [f_idx for f_idx in idxs if boxes.get(f_idx) is not None]
        ok_pos = [k for k, f_idx in enumerate(idxs) if boxes.get(f_idx) is not None]
        if sal_clip is not None and ok_pos:
            energies = [float(sal_clip[k].mean()) for k in ok_pos]
            rep_k = ok_pos[int(np.argmax(energies))]
        elif ok_pos:
            rep_k = ok_pos[len(ok_pos) // 2]
        else:
            rep_k = 0

        rep_f = idxs[rep_k]
        rgb = frames[rep_f]
        H, W = rgb.shape[:2]
        bx, by, bs = boxes[rep_f] if boxes.get(rep_f) is not None else (0, 0, min(H, W))

        face_rgb = cv2.resize(rgb[by:by + bs, bx:bx + bs], (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_LINEAR)
        face_b64 = cls._encode_jpeg_b64(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), quality=85)

        heatmap_b64 = ""
        if sal_clip is not None:
            raw_sal = 0.65 * sal_clip[rep_k] + 0.35 * sal_clip.mean(axis=0)
            heatmap_b64 = cls._sal_to_jet_b64(raw_sal, VIS_SIZE, quality=85)

        frames_4 = []
        for k, f_idx in enumerate(idxs):
            f_rgb = frames[f_idx]
            fH, fW = f_rgb.shape[:2]
            if boxes.get(f_idx) is not None:
                fx, fy, fs = boxes[f_idx]
            elif ok_ids:
                nearest = min(ok_ids, key=lambda j: abs(j - f_idx))
                fx, fy, fs = boxes[nearest]
            else:
                fx, fy, fs = 0, 0, min(fH, fW)
            crop = cv2.resize(f_rgb[fy:fy + fs, fx:fx + fs], (STRIP_SIZE, STRIP_SIZE), interpolation=cv2.INTER_LINEAR)
            f_b64 = cls._encode_jpeg_b64(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), quality=80)
            hm_b64 = cls._sal_to_jet_b64(sal_clip[k], STRIP_SIZE, quality=80) if sal_clip is not None else f_b64
            frames_4.append({
                "t": round(f_idx / fps, 2),
                "face_b64": f_b64,
                "heatmap_b64": hm_b64,
            })

        return {
            "frame_time": round(rep_f / fps, 2),
            "face_box": [round(bx / W, 4), round(by / H, 4), round(bs / W, 4), round(bs / H, 4)],
            "face_b64": face_b64,
            "heatmap_b64": heatmap_b64 or face_b64,
            "frames": frames_4,
        }

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
            logits, sal = self._logits_and_saliency(tensors)
            clip_p = logits.softmax(dim=-1)[:, FAKE_INDEX].cpu().tolist()
            prob = logits.mean(dim=0).softmax(dim=-1)[FAKE_INDEX].item()
            t_model = time.perf_counter() - t0

            seg = duration / NUM_CLIPS
            clips_out = []
            for idx_k, (ci, p) in enumerate(zip(kept, clip_p)):
                ev = self._render_clip_evidence(
                    frames, boxes, plan[ci],
                    sal[idx_k] if sal is not None else None,
                    fps,
                )
                clips_out.append({
                    "t_start": round(ci * seg, 2),
                    "t_end": round((ci + 1) * seg, 2),
                    "prob_fake": round(p, 4),
                    **ev,
                })
        except (NoFaceError, InferenceError):
            raise
        except Exception as e:
            log.exception("predict lỗi")
            raise InferenceError(str(e))

        return {
            "prob_fake": round(prob, 4),
            "clips": clips_out,
            "n_clips": len(kept),
            "timings": {"detect": round(t_detect, 2), "crop": round(t_crop, 2),
                        "model": round(t_model, 2),
                        "total": round(time.perf_counter() - t_all, 2)},
        }