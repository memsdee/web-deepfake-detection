# ===== CẤU HÌNH =====
STOP_EPOCH = 30      # lần 2: 17 | lần 3: 24 | lần 4: 30
WARMUP     = 5       # KHÔNG đổi (phải giống lần 1)
PREV_DIR   = "auto"  # tự tìm checkpoint.pth của lần trước
# ====================
import os, sys, time, json, shutil, subprocess
import pandas as pd, torch

W = "/kaggle/working"
BASE = "/kaggle/input/datasets/nhalac/tall-preprocessed-a"
MASTER = "/kaggle/input/datasets/nhalac/ffpp-master-747/master_747.csv"
LISTS, OUT, REPO = f"{W}/lists", f"{W}/tall_out", f"{W}/TALL4Deepfake"
REAL_REP = 9

assert torch.cuda.is_available(), "Chưa bật GPU"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
os.makedirs(OUT, exist_ok=True); os.makedirs(LISTS, exist_ok=True)

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    return r

# 1) repo + môi trường
if not os.path.isdir(REPO):
    sh(f"cd {W} && git clone -q https://github.com/rainy-xu/TALL4Deepfake.git")
os.chdir(REPO)
sh("pip install -q fvcore && pip install -q --no-deps timm==0.4.12")
print(sh("pip list 2>/dev/null | grep -i -E '^(timm|fvcore) '").stdout, flush=True)

# 2) patch (thêm weights_only=False cho torch.load để resume được)
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if new in s: continue
        assert s.count(old) == 1, f"{path}: không khớp đúng 1 chỗ: {old[:60]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s)

patch("video_dataset_config.py", [
    ("'train_list_name': 'cdf_test_fold.txt'", "'train_list_name': 'train.txt'"),
    ("'val_list_name': 'cdf_test_fold.txt'",   "'val_list_name': 'val.txt'"),
    ("'test_list_name': 'cdf_test_fold.txt'",  "'test_list_name': 'test.txt'"),
])
patch("engine.py", [
    ("    metric_logger.update(loss=real_loss.item())\n",
     "    metric_logger.update(loss=real_loss.item())\n    metric_logger.update(auc=float(auc_score))\n"),
])
patch("main.py", [
    ('max_accuracy = max(max_accuracy, test_stats["acc1"])',
     'max_accuracy = max(max_accuracy, test_stats["auc"] * 100)'),
    ("checkpoint_paths = [output_dir / 'checkpoint{}.pth'.format(epoch)]",
     "checkpoint_paths = [output_dir / 'checkpoint.pth']"),
    ('if test_stats["acc1"] == max_accuracy:',
     'if test_stats["auc"] * 100 == max_accuracy:'),
    ("for epoch in range(args.start_epoch, args.epochs):",
     "for epoch in range(args.start_epoch, min(args.epochs, int(os.environ.get('STOP_EPOCH', 10**9)))):"),
    ("checkpoint = torch.load(args.resume, map_location='cpu')",
     "checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)"),
    ("checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')",
     "checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)"),
])
print("patch xong", flush=True)

# 3) resume: tìm checkpoint của lần trước (không đi vào /kaggle/input/datasets vì quá nhiều file)
prev = PREV_DIR
if prev == "auto":
    prev = ""
    for root in ("/kaggle/input/notebooks", "/kaggle/input"):
        for dp, dn, fn in os.walk(root):
            if dp.startswith("/kaggle/input/datasets"): dn[:] = []; continue
            if "checkpoint.pth" in fn: prev = dp; break
        if prev: break
if prev:
    print("tìm thấy lần chạy trước:", prev, flush=True)
    for f in ("checkpoint.pth", "model_best.pth", "log.txt", "train.log"):
        if os.path.exists(f"{prev}/{f}"): shutil.copy(f"{prev}/{f}", f"{OUT}/{f}")
    ck = torch.load(f"{OUT}/checkpoint.pth", map_location="cpu", weights_only=False)
    print(f"checkpoint OK: epoch đã xong = {ck['epoch']}, max_accuracy(AUC*100) = {ck['max_accuracy']:.2f}, "
          f"có model_ema = {'model_ema' in ck}", flush=True)
    assert ck["epoch"] + 1 < STOP_EPOCH, f"STOP_EPOCH={STOP_EPOCH} <= epoch kế tiếp {ck['epoch']+1}, sẽ không train gì"
    del ck
else:
    print("KHÔNG tìm thấy checkpoint trước, sẽ train từ đầu epoch 0!", flush=True)
    assert STOP_EPOCH <= 10, "Đang resume mà không có checkpoint, dừng lại để tránh train lại từ đầu"

# 4) file list: dùng lại của lần trước nếu có, không thì dựng
need = ("train.txt", "val.txt", "test.txt")
src_lists = os.path.join(os.path.dirname(prev), "lists") if prev else ""
if prev and all(os.path.exists(f"{src_lists}/{n}") for n in need):
    for n in need: shutil.copy(f"{src_lists}/{n}", f"{LISTS}/{n}")
    print("dùng lại file list của lần trước:", {n: sum(1 for _ in open(f"{LISTS}/{n}")) for n in need}, flush=True)
else:
    df = pd.read_csv(MASTER, dtype=str); rows = []
    for i, (m, fn, sp) in enumerate(zip(df["method"], df["filename"], df["split"])):
        vid = os.path.splitext(fn)[0].split("/")[-1]; rel = f"{m}/{m}/{vid}/{vid}"
        f = sorted(x for x in os.listdir(f"{BASE}/{rel}") if x.endswith(".jpg")); n = len(f)
        assert n > 0 and f[0] == "0001.jpg" and f[-1] == f"{n:04d}.jpg", f"lỗi frame: {rel}"
        rows.append((sp, f"{rel} 1 {n} {0 if m == 'real' else 1}"))
        if i % 1500 == 0: print(f"  list {i}/{len(df)}", flush=True)
    for sp in ("val", "test"):
        open(f"{LISTS}/{sp}.txt", "w").write("\n".join(l for s, l in rows if s == sp) + "\n")
    tr = [l for s, l in rows if s == "train"]
    real = [l for l in tr if l.endswith(" 0")]; fake = [l for l in tr if l.endswith(" 1")]
    open(f"{LISTS}/train.txt", "w").write("\n".join(fake + real * REAL_REP) + "\n")
    print(f"dựng list xong: real={len(real)}x{REAL_REP} fake={len(fake)}", flush=True)

# 5) train fp32 theo README
args = ["python", "-u", "main.py", "--dataset", "ffpp", "--data_dir", BASE, "--data_txt_dir", LISTS,
        "--input-size", "112", "--num_clips", "8", "--output_dir", OUT, "--opt", "adamw",
        "--lr", "1.5e-5", "--warmup-lr", "1.5e-8", "--min-lr", "1.5e-7", "--epochs", "30",
        "--sched", "cosine", "--duration", "4", "--batch-size", "4", "--thumbnail_rows", "2",
        "--disable_scaleup", "--cutout", "True", "--pretrained", "--warmup-epochs", str(WARMUP),
        "--no-amp", "--model", "TALL_SWIN", "--hpe_to_token", "--num_workers", "24"]
env = dict(os.environ, STOP_EPOCH=str(STOP_EPOCH), CUDA_VISIBLE_DEVICES="0")
t0 = time.time()
with open(f"{OUT}/train.log", "a") as logf:
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "FutureWarning" in line or line.startswith(("Namespace(", "  with torch.cuda", "  self._scaler")): continue
        print(line, end="", flush=True); logf.write(line); logf.flush()
p.wait()
print(f"\n>>> train xong sau {(time.time()-t0)/3600:.2f} giờ, exit code: {p.returncode}", flush=True)

# 6) bảng tóm tắt theo epoch (gồm cả các lần trước)
print("\nepoch | train_loss | test_acc1 | test_auc")
for l in open(f"{OUT}/log.txt"):
    try: d = json.loads(l)
    except Exception: continue
    print(f"{d.get('epoch')!s:>5} | {d.get('train_loss', float('nan')):.4f} | "
          f"{d.get('test_acc1', float('nan')):.2f} | {d.get('test_auc', float('nan')):.4f}")
print(subprocess.run(f"ls -la {OUT}", shell=True, capture_output=True, text=True).stdout)

