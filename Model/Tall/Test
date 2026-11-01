import os, re, json, glob, shutil, subprocess, importlib.util, time
import torch

W = "/kaggle/working"
BASE = "/kaggle/input/datasets/nhalac/tall-preprocessed-a"
REPO = f"{W}/TALL4Deepfake"
EVAL_LISTS, EVAL_OUT = f"{W}/lists_eval", f"{W}/tall_eval_out"
assert torch.cuda.is_available(), "Chưa bật GPU"

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    return r

# 1) tìm output train có epoch cuối lớn nhất (chỉ đọc log.txt, không quét dataset ảnh)
best_dir, best_ep = "", -1
for log in glob.glob("/kaggle/input/notebooks/*/*/tall_out/log.txt"):
    eps = []
    for l in open(log):
        if l.strip().startswith("{"):
            try: eps.append(json.loads(l)["epoch"])
            except Exception: pass
    print(os.path.dirname(os.path.dirname(log)), "| epoch cuối:", max(eps) if eps else None)
    if eps and max(eps) > best_ep:
        best_ep, best_dir = max(eps), os.path.dirname(os.path.dirname(log))
assert best_ep == 29, f"Chưa thấy output nào tới epoch 29 (cao nhất: {best_ep}). Kiểm tra Add Input."
CKPT = f"{best_dir}/tall_out/model_best.pth"
SRC_LISTS = f"{best_dir}/lists"
assert os.path.exists(CKPT), CKPT
for n in ("train.txt", "val.txt", "test.txt"):
    assert os.path.exists(f"{SRC_LISTS}/{n}"), f"Thiếu {SRC_LISTS}/{n}"
print("dùng:", best_dir)
for n in ("val.txt", "test.txt"):
    print(n, "=", sum(1 for _ in open(f"{SRC_LISTS}/{n}")), "video")

# 2) repo + môi trường
if not os.path.isdir(REPO):
    sh(f"cd {W} && git clone -q https://github.com/rainy-xu/TALL4Deepfake.git")
assert os.path.isdir(REPO), "git clone thất bại (kiểm tra Internet đã bật chưa)"
os.chdir(REPO)
sh("pip install -q fvcore && pip install -q --no-deps timm==0.4.12")

# 3) patch repo
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
    ("checkpoint = torch.load(args.resume, map_location='cpu')",
     "checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)"),
    ("checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')",
     "checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)"),
])

# 4) patch timm (ModelEma gọi torch.load không có weights_only=False)
timm_dir = os.path.dirname(importlib.util.find_spec("timm").origin)
ema_py = f"{timm_dir}/utils/model_ema.py"
s = open(ema_py).read()
old = "torch.load(checkpoint_path, map_location='cpu')"
new = "torch.load(checkpoint_path, map_location='cpu', weights_only=False)"
if new not in s:
    assert s.count(old) == 1, "Không khớp dòng torch.load trong timm"
    open(ema_py, "w").write(s.replace(old, new))
print("đã vá repo và timm", flush=True)

# 5) list eval: val.txt và test.txt đều = test
os.makedirs(EVAL_LISTS, exist_ok=True); os.makedirs(EVAL_OUT, exist_ok=True)
shutil.copy(f"{SRC_LISTS}/train.txt", f"{EVAL_LISTS}/train.txt")
shutil.copy(f"{SRC_LISTS}/test.txt", f"{EVAL_LISTS}/val.txt")
shutil.copy(f"{SRC_LISTS}/test.txt", f"{EVAL_LISTS}/test.txt")

# 6) eval trên test
args = ["python", "-u", "main.py", "--dataset", "ffpp", "--data_dir", BASE, "--data_txt_dir", EVAL_LISTS,
        "--input-size", "112", "--num_clips", "8", "--output_dir", EVAL_OUT, "--opt", "adamw",
        "--epochs", "30", "--sched", "cosine", "--duration", "4", "--batch-size", "4",
        "--thumbnail_rows", "2", "--disable_scaleup", "--cutout", "True", "--pretrained",
        "--no-amp", "--model", "TALL_SWIN", "--hpe_to_token", "--num_workers", "8",
        "--eval", "--resume", CKPT]
t0 = time.time()
pr = subprocess.Popen(args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                      env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"))
lines = []
for line in pr.stdout:
    if "FutureWarning" in line or line.startswith("Namespace("): continue
    lines.append(line); print(line, end="", flush=True)
pr.wait()
out = "".join(lines)
open(f"{W}/tall_eval.log", "w").write(out)
print(f"\nreturncode: {pr.returncode} | {(time.time()-t0)/60:.1f} phút")

# 7) lưu kết quả
m = re.search(r"Acc@1\s+([\d.]+)\s+AUC\s+([\d.]+)\s+loss\s+([\d.]+)", out)
if m:
    res = {"model": "TALL_SWIN", "split": "test", "n_videos": sum(1 for _ in open(f"{EVAL_LISTS}/test.txt")),
           "ckpt_source": best_dir, "acc1": float(m[1]), "auc": float(m[2]), "loss": float(m[3])}
    json.dump(res, open(f"{W}/tall_test_result.json", "w"), indent=2)
    print("\n[TALL TEST]", res)
else:
    print("\nKhông parse được kết quả, xem tall_eval.log")

# dọn file nặng nếu có (output_dir có thể chứa checkpoint)
for f in glob.glob(f"{EVAL_OUT}/*.pth"):
    os.remove(f)