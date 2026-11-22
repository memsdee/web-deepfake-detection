
WORKER = 2   # Colab 1: 0 | Colab 2: 1 | Colab 3: 2

import os, glob, time, shutil, subprocess, tarfile, zipfile
from google.colab import drive
drive.mount('/content/drive')


R = "/content/drive/MyDrive/grad project"
PRE = f"{R}/MODEL/DFD/PREPROCESSED"
BK = f"{PRE}/cropped"
LOG = f"{PRE}/step_b_log_w{WORKER}.txt"
W = f"/content/work{WORKER}"          # mỗi phiên Colab là máy riêng, tên này chỉ để dễ nhìn
REPO = "/content/DFD-FCG"
INTERVAL = 180
ALL_GROUPS = ["ffpp_real_c23", "ffpp_deepfakes_c23", "ffpp_faceswap_c23", "ffpp_faceshifter_c23",
              "df40_blendface", "df40_facedancer", "df40_fsgan", "df40_inswap", "df40_mobileswap", "df40_simswap"]
GROUPS = ALL_GROUPS
print("Colab", WORKER, "xử lý:", GROUPS)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ids(d, ext):
    return {os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f"{d}/*{ext}")}


def free_gb(p):
    return shutil.disk_usage(p).free / 1024**3


def cp(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def sync(g):
    n = 0
    for i in ids(f"{W}/cropped/frame_data/{g}", ".pickle"):
        avi = f"{W}/cropped/videos/{g}/{i}.avi"
        if not os.path.exists(avi):
            continue
        n += cp(avi, f"{BK}/videos/{g}/{i}.avi")                                          # avi trước
        cp(f"{W}/cropped/frame_data/{g}/{i}.pickle", f"{BK}/frame_data/{g}/{i}.pickle")   # pickle sau
    return n


def drive_done(g):
    return ids(f"{BK}/frame_data/{g}", ".pickle") & ids(f"{BK}/videos/{g}", ".avi")


def bar(k, n, w=30):
    f = int(w * k / n)
    return f"[{'#'*f}{'.'*(w-f)}] {k}/{n} ({100*k/n:.0f}%)"


# --- chuẩn bị ---
os.makedirs(BK, exist_ok=True)
os.makedirs(W, exist_ok=True)
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/aiiu-lab/DFD-FCG", REPO], check=True)
if len(glob.glob(f"{W}/frame_data/*/*.pickle")) < 7470:
    with tarfile.open(f"{PRE}/frame_data.tar") as tf:
        tf.extractall(W, filter="data")
log(f"[w{WORKER}] frame_data local: {len(glob.glob(f'{W}/frame_data/*/*.pickle'))} pickle | CPU {os.cpu_count()} | nhóm: {GROUPS}")


T0 = time.time()
for g in GROUPS:
    marker = f"{BK}/DONE_{g}.txt"
    if os.path.exists(marker):
        log(f"{g}: đã DONE, bỏ qua"); continue
    if free_gb("/content") < 12 or free_gb("/content/drive/MyDrive") < 10:
        log(f"DỪNG: hết dung lượng (local {free_gb('/content'):.1f} GB, Drive {free_gb('/content/drive/MyDrive'):.1f} GB)")
        break


    zpath = f"{R}/DATASET_747/{g.replace('_', '-')}.zip"
    vdir = f"{W}/videos/{g}"
    os.makedirs(vdir, exist_ok=True)
    done0 = drive_done(g)
    with zipfile.ZipFile(zpath) as z:
        allv = sorted(n for n in z.namelist() if n.endswith(".mp4"))
        need = [n for n in allv if os.path.splitext(n)[0] not in done0]
        for n in need:
            z.extract(n, vdir)
    total = len(allv)
    log(f"{g}: {total} video, đã backup {len(done0)}, cần crop {len(need)}")


    err = open(f"{W}/crop_err.txt", "w")
    p = subprocess.Popen(
        ["python", "-m", "src.preprocess.crop_main_face", f"--root-dir={W}", "--video-dir=videos",
         "--fdata-dir=frame_data", "--crop-dir=cropped", f"--glob-exp={g}/*", "--crop-size=150",
         "--mean-face=misc/20words_mean_face.npy", "--workers=2"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=err)
    tg, last = time.time(), 0
    while True:
        time.sleep(5)
        finished = p.poll() is not None
        if finished or time.time() - last >= INTERVAL:
            last = time.time()
            moved = sync(g)
            k = len(drive_done(g) | ids(f"{W}/cropped/frame_data/{g}", ".pickle"))
            rate = (k - len(done0)) / max(time.time() - tg, 1)
            eta = (total - k) / rate / 60 if rate > 0 else float("nan")
            log(f"{g} {bar(k, total)} | +{moved} lên Drive | {rate*60:.1f} video/phút | ETA nhóm ~{eta:.0f} phút | "
                f"local trống {free_gb('/content'):.0f} GB, Drive trống {free_gb('/content/drive/MyDrive'):.0f} GB")
        if finished:
            break
    err.close()
    if p.returncode != 0:
        log(f"{g}: LỖI returncode={p.returncode}\n" + open(f"{W}/crop_err.txt").read()[-800:])
        break


    sync(g)
    dd = drive_done(g)
    failed = sorted(set(os.path.splitext(os.path.basename(n))[0] for n in allv) - dd)
    with open(marker, "w") as f:
        f.write(f"ok={len(dd)} total={total} bị_bỏ={len(failed)}\n" + "\n".join(failed))
    log(f"{g}: XONG ok={len(dd)}/{total}, bị bỏ={len(failed)} {failed[:5]}")
    shutil.rmtree(vdir, ignore_errors=True)
    shutil.rmtree(f"{W}/cropped/videos/{g}", ignore_errors=True)
    shutil.rmtree(f"{W}/cropped/frame_data/{g}", ignore_errors=True)


log(f"Tổng thời gian lần chạy này: {(time.time()-T0)/3600:.2f} giờ")
----------------------------KAGLLE
# ============================================================
# 4a FINAL (Save & Run All): cài đặt + GPU check + import pickle cũ + fetch_landmark_bbox
# ============================================================
import os, re, shutil, time, subprocess, collections

# ---------------- TÙY CHỌN ----------------
MAX_VIDEOS_TOTAL     = None   # tổng video xử lý trong lần chạy này (None = không giới hạn)
MAX_VIDEOS_PER_LABEL = None   # tối đa video mỗi dataset (None = hết; ví dụ 300)
MAX_HOURS            = 1    # dừng êm sau X giờ để Kaggle kịp lưu output (đặt thấp hơn giới hạn session)
STALL_MIN            = 25     # quá X phút không có pickle mới thì coi là treo, dừng và in log
NUM_GPUS             = 2      # 1 = an toàn; 2 = chia video cho 2 GPU T4 chạy song song
ONLY    = r""                 # regex: chỉ chạy label khớp, ví dụ "deepfakes|fsgan"
EXCLUDE = r"^$"               # regex: bỏ label khớp
BATCH, MAXRES = 16, 800
PRINT_EVERY = 30
HINT = {"DF": "ffpp_deepfakes", "deepfakes": "ffpp_deepfakes", "real": "ffpp_real",
        "faceswap": "ffpp_faceswap", "faceshifter": "ffpp_faceshifter"}
# ------------------------------------------

T0 = time.time()
INPUT, OUT = "/kaggle/input", "/kaggle/working/frame_data"
REPO, LOGDIR = "/kaggle/working/DFD-FCG", "/kaggle/working/logs"
MARK = f"{REPO}/.installed_v3"
os.makedirs(LOGDIR, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=True)
stem = lambda f: f.split(".")[0]
P = lambda *a: print(*a, flush=True)

# ---- 1. Cài thư viện ----
if not os.path.exists(MARK):
    P("Đang cài thư viện (vài phút)...")
    sh("cd /kaggle/working && rm -rf DFD-FCG && git clone -q https://github.com/aiiu-lab/DFD-FCG.git")
    sh("pip uninstall -y torch torchvision torchaudio -q")
    sh("pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121 -q")
    sh("pip install aiohttp==3.9.3 albumentations==1.4.0 antlr4-python3-runtime==4.9.3 ftfy==6.1.3 hydra-core==1.3.2 lightning==2.2.0.post0 omegaconf==2.3.0 open-clip-torch==2.24.0 pytorch-lightning==2.2.0.post0 qudida==0.0.4 tensorboardx==2.6.2.2 timm==0.9.16 wandb==0.16.3 typeshed-client==2.5.1 -q")
    sh('pip install -U "jsonargparse[signatures]" -q')
    sh("pip install face-alignment==1.4.1 -q")
    sh("pip install numpy==1.26.4 opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80 av -q")
    sh("pip install numpy==1.26.4 -q")
    open(MARK, "w").close()

# ---- 2. Kiểm tra GPU (nếu không có CUDA thì dừng ngay, không chạy mù) ----
r = subprocess.run('python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"',
                   shell=True, capture_output=True, text=True)
P("torch / cuda / số GPU:", r.stdout.strip(), r.stderr.strip()[-300:])
if " True " not in f" {r.stdout.strip()} ":
    raise SystemExit("⚠ Torch không thấy CUDA -> dừng. Cần cài lại bản torch cu121.")
NUM_GPUS = max(1, min(NUM_GPUS, int(r.stdout.split()[-1])))

# ---- 3. Quét dataset ----
videos, other, fd_dirs = {}, collections.Counter(), []
for dp, dn, fn in os.walk(INPUT):
    if os.path.basename(dp) == "frame_data":
        fd_dirs.append(dp); dn[:] = []; continue
    if dp.startswith(INPUT + "/notebooks"): continue
    mp4 = [f for f in fn if f.lower().endswith(".mp4")]
    if mp4: videos[dp] = sorted(mp4)
    for f in fn:
        e = os.path.splitext(f)[1].lower()
        if e in (".avi", ".mov", ".mkv", ".jpg", ".png", ".webp"): other[e] += 1

def mk_label(dp):
    rel = os.path.relpath(dp, INPUT).split(os.sep)
    if rel[0] == "datasets" and len(rel) > 2: rel = rel[2:]
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(rel)).strip("_")

LABELS, VDIR = {}, {}
for dp, f in videos.items():
    l = mk_label(dp)
    if re.search(EXCLUDE, l): continue
    if ONLY and not re.search(ONLY, l, re.I): continue
    LABELS[l], VDIR[l] = f, dp
# deepfakes chạy trước (còn thiếu), các nhóm còn lại theo alphabet
LABELS = dict(sorted(LABELS.items(), key=lambda kv: (0 if "ffpp_deepfakes" in kv[0] else 1, kv[0])))

P(f"\n{len(LABELS)} nhóm video mp4:")
for l, v in LABELS.items(): P(f"  {len(v):6d}  {l}")
if other: P("⚠ File KHÔNG phải mp4:", dict(other))

# ---- 4. Import pickle từ các lần chạy trước ----
def target(lbl):
    if lbl in LABELS: return lbl
    h = HINT.get(lbl)
    if h:
        c = [l for l in LABELS if h in l.lower()]
        if len(c) == 1: return c[0]
    return None

n_copied = 0
for fd in fd_dirs:
    parent = os.path.basename(os.path.dirname(fd))
    pairs = []
    if parent == "c23":
        pairs.append((fd, target(os.path.basename(os.path.dirname(os.path.dirname(fd))))))
    else:
        for s in os.listdir(fd):
            if os.path.isdir(f"{fd}/{s}"): pairs.append((f"{fd}/{s}", target(s)))
    for src, tgt in pairs:
        if not tgt: P("⚠ bỏ qua (không map được):", src); continue
        os.makedirs(f"{OUT}/{tgt}", exist_ok=True)
        for f in os.listdir(src):
            d = f"{OUT}/{tgt}/{f}"
            if not os.path.exists(d): shutil.copy2(f"{src}/{f}", d); n_copied += 1
        P(f"  import {src} -> {tgt}")
P("Đã import", n_copied, "pickle\n")

# ---- 5. Lập kế hoạch ----
plan, left = {}, MAX_VIDEOS_TOTAL
for label, vids in LABELS.items():
    os.makedirs(f"{OUT}/{label}", exist_ok=True)
    done = {stem(f) for f in os.listdir(f"{OUT}/{label}")}
    todo = [v for v in vids if stem(v) not in done]
    if MAX_VIDEOS_PER_LABEL is not None: todo = todo[:MAX_VIDEOS_PER_LABEL]
    if left is not None: todo = todo[:left]; left -= len(todo)
    plan[label] = (len(done), todo)
    P(f"[{label}] đã xong {len(done)}/{len(vids)} | lần này chạy {len(todo)}")
total_all = sum(len(t) for _, t in plan.values())
P(f"\nTỔNG cần chạy lần này: {total_all} video | GPU dùng: {NUM_GPUS}\n")

# ---- 6. Chạy ----
t_start, finished = time.time(), 0

def report(n, label, extra=""):
    el = time.time() - t_start
    rate = n / el if n > 0 and el > 0 else 0
    eta = (total_all - n) / rate / 60 if rate else float("nan")
    pct = 100 * n / total_all if total_all else 100
    P(f"[{pct:5.1f}%] {n}/{total_all} vid | {label} | {rate*60:.1f} vid/phút | còn ~{eta:.0f} phút {extra}")

def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 600))
            t = f.read().decode(errors="ignore").replace("\r", "\n").strip().split("\n")
        return t[-1][:150] if t and t[0] else ""
    except Exception:
        return ""

def stop_time(): return MAX_HOURS is not None and time.time() - T0 > MAX_HOURS * 3600

for label, (base, todo) in plan.items():
    if not todo: continue
    if stop_time(): P("Hết quỹ thời gian, dừng êm."); break

    shutil.rmtree(f"/tmp/ffpp_source/{label}", ignore_errors=True)
    procs = []
    for g in range(NUM_GPUS):
        part = todo[g::NUM_GPUS]
        if not part: continue
        root = f"/tmp/ffpp_source/{label}/g{g}/c23"
        os.makedirs(f"{root}/videos")
        for v in part: os.symlink(f"{VDIR[label]}/{v}", f"{root}/videos/{v}")
        os.symlink(f"{OUT}/{label}", f"{root}/frame_data")
        cmd = ["python", "-u", "-m", "src.preprocess.fetch_landmark_bbox",
               f"--root-dir={root}", "--video-dir=videos", "--fdata-dir=frame_data",
               "--glob-exp=*", f"--batch={BATCH}", f"--max-res={MAXRES}"]
        logp = f"{LOGDIR}/{label}_g{g}.log"
        logf = open(logp, "w")
        ys = subprocess.Popen(["yes"], stdout=subprocess.PIPE)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        p = subprocess.Popen(cmd, cwd=REPO, stdin=ys.stdout, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((p, ys, logf, logp))

    last_print, last_progress, last_cnt, aborted = 0, time.time(), 0, False
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(2)
        cur = min(max(len(os.listdir(f"{OUT}/{label}")) - base, 0), len(todo))
        if cur != last_cnt: last_cnt, last_progress = cur, time.time()
        if time.time() - last_print >= PRINT_EVERY:
            last_print = time.time()
            report(finished + cur, label, "| log: " + last_line(procs[0][3]))
        if time.time() - last_progress > STALL_MIN * 60:
            P(f"⚠ [{label}] {STALL_MIN} phút không có pickle mới -> coi như treo"); aborted = True; break
        if stop_time():
            P("Hết quỹ thời gian, dừng êm."); aborted = True; break
    for p, ys, logf, logp in procs:
        if p.poll() is None: p.terminate()
        try: p.wait(timeout=30)
        except Exception: p.kill()
        ys.kill(); logf.close()

    cur = min(max(len(os.listdir(f"{OUT}/{label}")) - base, 0), len(todo))
    finished += cur
    report(finished, label, "(xong nhóm)")
    bad = [lp for p, _, _, lp in procs if p.returncode not in (0, None, -15)]
    if bad or (aborted and stop_time() is False):
        for lp in (bad or [procs[0][3]]):
            P(f"--- 25 dòng cuối {lp} ---"); sh(f"tail -25 {lp}")
        raise SystemExit("Dừng vì lỗi/treo ở trên")
    if aborted: break

# ---- 7. Tổng kết ----
P("\n===== TỔNG KẾT 4a =====")
tot_done = tot = 0
for label, vids in LABELS.items():
    d = len(os.listdir(f"{OUT}/{label}")) if os.path.exists(f"{OUT}/{label}") else 0
    tot_done += d; tot += len(vids)
    P(f"{d:6d}/{len(vids):<6d} {label}")
P(f"TOTAL {tot_done}/{tot} | thời gian: {(time.time()-T0)/60:.1f} phút")
