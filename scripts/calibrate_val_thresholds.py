# ===== HIỆU CHỈNH NGƯỠNG 3 VÙNG (LOW, HIGH) TRÊN TẬP VAL (1.050) & TEST (1.040) =====
# Phiên bản tối ưu tốc độ: DataLoader đa luồng (num_workers=4, batch_size=8) + FP16 autocast
# Thời gian chạy dự kiến trên Kaggle GPU T4: ~1 - 1.5 phút cho cả val + test!

import json, os, subprocess, sys, time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

W = "/kaggle/working"
BASE = "/kaggle/input/datasets/nhalac/tall-preprocessed-a"
MASTER = "/kaggle/input/datasets/nhalac/ffpp-master-747/master_747.csv"
CKPT = "/kaggle/input/notebooks/tuphu1/tall-train-5/tall_out/model_best.pth"
REPO = f"{W}/TALL4Deepfake"

DURATION = 4
THUMBNAIL_ROWS = 2
INPUT_SIZE = 112
NUM_CLIPS = 8
BATCH_VIDEOS = 8
NUM_WORKERS = 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

assert torch.cuda.is_available(), "Hãy bật Accelerator = GPU T4 trước khi chạy!"
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# 1) Clone TALL repo & cài gói tối thiểu
if not os.path.isdir(REPO):
    subprocess.run(f"git clone -q https://github.com/rainy-xu/TALL4Deepfake.git {REPO}", shell=True, check=True)
subprocess.run("pip install -q fvcore && pip install -q --no-deps timm==0.4.12", shell=True, check=True)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import my_models  # noqa: F401
from timm.models import create_model

# 2) Nạp model_best.pth đúng chuẩn backend/inference.py
model = create_model(
    "TALL_SWIN", pretrained=False,
    duration=DURATION, hpe_to_token=True, thumbnail_rows=THUMBNAIL_ROWS, num_classes=2,
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(device).eval()
print("Đã nạp model_best.pth thành công!", flush=True)


def prep_frame(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != INPUT_SIZE or rgb.shape[1] != INPUT_SIZE:
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD


class VideoClipDataset(Dataset):
    def __init__(self, df_split):
        self.items = [
            (row.method, os.path.splitext(row.filename)[0].split("/")[-1], 0 if row.method == "real" else 1)
            for row in df_split.itertuples(index=False)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        m, vid, label = self.items[idx]
        folder = f"{BASE}/{m}/{m}/{vid}/{vid}"
        jpgs = sorted(x for x in os.listdir(folder) if x.endswith(".jpg"))
        n = len(jpgs)
        sample_pos = max(1, 1 + n - DURATION)
        starts = np.linspace(0, sample_pos - 1, num=NUM_CLIPS, dtype=int).tolist()

        # Chỉ đọc mỗi khung hình duy nhất 1 lần (cache nếu bị trùng chỉ số)
        cache = {}
        clips = []
        for s in starts:
            per = []
            for k in range(DURATION):
                fi = (s + k) % n
                if fi not in cache:
                    cache[fi] = prep_frame(f"{folder}/{jpgs[fi]}")
                per.append(cache[fi])
            clips.append(np.concatenate(per, axis=0))  # (12, 112, 112)
        return torch.from_numpy(np.stack(clips, axis=0)), label


@torch.no_grad()
def predict_split(df_split, split_name):
    t0 = time.time()
    ds = VideoClipDataset(df_split)
    dl = DataLoader(ds, batch_size=BATCH_VIDEOS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    probs, labels = [], []
    done = 0
    for clips_b, y_b in dl:
        B, C, CH, H, W_ = clips_b.shape  # (B, 8, 12, 112, 112)
        x = clips_b.view(B * C, CH, H, W_).to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)  # (B*8, 2)
        logits = logits.float().view(B, C, 2)
        p_fake = logits.mean(dim=1).softmax(dim=-1)[:, 1].cpu().numpy()
        probs.append(p_fake)
        labels.append(y_b.numpy())
        done += B
        if done % 200 < BATCH_VIDEOS or done == len(ds):
            print(f"  [{split_name}] {done}/{len(ds)} ({time.time()-t0:.1f}s)", flush=True)
    return np.concatenate(probs), np.concatenate(labels)


df = pd.read_csv(MASTER, dtype=str)
val_p, val_y = predict_split(df[df["split"] == "val"], "val")
test_p, test_y = predict_split(df[df["split"] == "test"], "test")

# 3) Tính ROC, Youden's J và hiệu chỉnh cặp ngưỡng (LOW, HIGH) trên VAL
val_auc = roc_auc_score(val_y, val_p)
fpr, tpr, thr = roc_curve(val_y, val_p)
fnr = 1.0 - tpr
eer_idx = int(np.argmin(np.abs(fpr - fnr)))
eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

youden_idx = int(np.argmax(tpr - fpr))
t_youden = float(thr[youden_idx])

# Chow's Reject Option trên VAL:
# - LOW: ngưỡng lớn nhất đảm bảo FNR <= 5% (Recall >= 95% trên Fake) -> hạn chế bỏ sót video giả
# - HIGH: ngưỡng nhỏ nhất đảm bảo FPR <= 5% (Specificity >= 95% trên Real) -> hạn chế báo động nhầm
cand_low = thr[fnr <= 0.05]
t_low = float(np.max(cand_low)) if len(cand_low) else float(np.quantile(val_p[val_y == 1], 0.05))
cand_high = thr[fpr <= 0.05]
t_high = float(np.min(cand_high)) if len(cand_high) else float(np.quantile(val_p[val_y == 0], 0.95))

LOW = round(min(t_low, t_youden), 2)
HIGH = round(max(t_high, t_youden), 2)
if HIGH - LOW < 0.06:
    LOW = round(max(0.05, t_youden - 0.10), 2)
    HIGH = round(min(0.95, t_youden + 0.10), 2)


def eval_3zone(p, y, lo, hi):
    green = p < lo
    yellow = (p >= lo) & (p < hi)
    red = p >= hi
    decided = green | red
    pred_decided = np.where(red[decided], 1, 0)
    sel_acc = float((pred_decided == y[decided]).mean()) if decided.any() else 0.0
    fnr_green = float(((y == 1) & green).sum() / max((y == 1).sum(), 1))
    fpr_red = float(((y == 0) & red).sum() / max((y == 0).sum(), 1))
    acc_05 = float(((p >= 0.5).astype(int) == y).mean())
    acc_youden = float(((p >= t_youden).astype(int) == y).mean())
    return {
        "n": int(len(y)),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "acc_at_0.5": round(acc_05 * 100, 2),
        "acc_at_youden": round(acc_youden * 100, 2),
        "coverage_pct": round(float(decided.mean()) * 100, 2),
        "abstain_yellow_pct": round(float(yellow.mean()) * 100, 2),
        "selective_acc_pct": round(sel_acc * 100, 2),
        "fnr_in_green_pct": round(fnr_green * 100, 2),
        "fpr_in_red_pct": round(fpr_red * 100, 2),
        "real_quantiles": [round(float(x), 4) for x in np.quantile(p[y == 0], [0.05, 0.25, 0.5, 0.75, 0.95])],
        "fake_quantiles": [round(float(x), 4) for x in np.quantile(p[y == 1], [0.05, 0.25, 0.5, 0.75, 0.95])],
    }


summary = {
    "val_auc": round(float(val_auc), 4),
    "val_eer_pct": round(eer * 100, 2),
    "t_youden": round(t_youden, 4),
    "raw_t_low_fnr5": round(t_low, 4),
    "raw_t_high_fpr5": round(t_high, 4),
    "recommended_LOW": LOW,
    "recommended_HIGH": HIGH,
    "val_with_recommended": eval_3zone(val_p, val_y, LOW, HIGH),
    "test_with_recommended": eval_3zone(test_p, test_y, LOW, HIGH),
    "val_with_old_035_055": eval_3zone(val_p, val_y, 0.35, 0.55),
    "test_with_old_035_055": eval_3zone(test_p, test_y, 0.35, 0.55),
}

with open(f"{W}/val_calibration_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== KẾT QUẢ HIỆU CHỈNH NGƯỠNG (COPY ĐOẠN JSON NÀY) =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))

# 4) Vẽ biểu đồ phân phối xác suất + ROC để đưa vào báo cáo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=300)
fig.patch.set_facecolor("white")

bins = np.linspace(0, 1, 41)
ax1.hist(val_p[val_y == 0], bins=bins, density=True, alpha=0.65, color="#059669", label="Video thật (Real, n=105)")
ax1.hist(val_p[val_y == 1], bins=bins, density=True, alpha=0.55, color="#DC2626", label="Video giả (Fake, n=945)")
ax1.axvspan(LOW, HIGH, color="#F59E0B", alpha=0.22, label=f"Vùng trung gian [{LOW:.2f}, {HIGH:.2f})")
ax1.axvline(LOW, color="#059669", linestyle="--", lw=1.6)
ax1.axvline(HIGH, color="#DC2626", linestyle="--", lw=1.6)
ax1.set_title("(a) Phân phối p_fake trên tập kiểm định (1.050 video)", fontsize=10.5, fontweight="bold")
ax1.set_xlabel("Xác suất giả mạo p_fake", fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Mật độ xác suất", fontsize=9.5, fontweight="bold")
ax1.legend(fontsize=8.2, loc="upper center")
ax1.grid(True, linestyle="--", alpha=0.3)

ax2.plot(fpr, tpr, color="#1D4ED8", lw=2.0, label=f"ROC trên val (AUC = {val_auc:.4f}, EER = {eer*100:.2f}%)")
ax2.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", lw=1.2)
ax2.scatter([fpr[youden_idx]], [tpr[youden_idx]], color="#DC2626", s=55, zorder=5,
            label=f"Youden t* = {t_youden:.3f}")
ax2.set_title("(b) Đường cong ROC & điểm cắt tối ưu trên tập val", fontsize=10.5, fontweight="bold")
ax2.set_xlabel("False Positive Rate (FPR)", fontsize=9.5, fontweight="bold")
ax2.set_ylabel("True Positive Rate (TPR)", fontsize=9.5, fontweight="bold")
ax2.legend(fontsize=8.2, loc="lower right")
ax2.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout(pad=1.4)
plt.savefig(f"{W}/val_threshold_calibration.png", dpi=300, facecolor="white")
print("Đã lưu biểu đồ tại /kaggle/working/val_threshold_calibration.png")
