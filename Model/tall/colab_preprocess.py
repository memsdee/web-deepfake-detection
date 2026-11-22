=====
WORKER, N_WORKERS, LIMIT = 1, 4, None     # LIMIT=None để chạy thật
LOG_EVERY, SYNC_EVERY = 30, 180        # giây: log / backup lên Drive


!pip -q install facenet-pytorch --no-deps
from google.colab import drive; drive.mount('/content/drive')
import os, io, time, shutil, tarfile, zipfile, datetime, cv2, torch, numpy as np, pandas as pd
from facenet_pytorch import MTCNN


BASE   = "/content/drive/MyDrive/grad project"
OUT    = f"{BASE}/MODEL/TALL/PREPROCESSED"      # <method>/<video_id>.tar
LOGDIR = f"{OUT}/logs"
BUF    = f"/content/buf_w{WORKER}"              # bộ đệm cục bộ
SIZE, MARGIN, BATCH, DET_SHORT = 128, 0.3, 32, 480
ZN = {"real":"ffpp-real-c23","deepfakes":"ffpp-deepfakes-c23",
      "faceswap":"ffpp-faceswap-c23","faceshifter":"ffpp-faceshifter-c23"}
zpath = lambda m: f"{BASE}/DATASET_747/{ZN.get(m, 'df40-'+m)}.zip"
for d in (OUT, LOGDIR, BUF): os.makedirs(d, exist_ok=True)


m = pd.read_csv(f"{BASE}/DATASET/master_747.csv", dtype=str)
vids = m.sort_values(["method","filename"]).reset_index(drop=True)
vids = vids[vids.index % N_WORKERS == WORKER]
if LIMIT: vids = vids.groupby("method").head(1).head(LIMIT)
N = len(vids)


# ---------- log + backup ----------
S = dict(done=0, skip=0, fail=0, synced=0, cur="-", cur_n=0, cur_tot=0, times=[],
         last=0, sync=time.time(), vt=time.time())


def buffered():
    return sum(f.endswith(".tar") for _, _, fs in os.walk(BUF) for f in fs)


def heartbeat(force=False, final=False):
    now = time.time()
    if not force and now - S["last"] < LOG_EVERY: return
    S["last"] = now
    avg = sum(S["times"])/len(S["times"]) if S["times"] else 0
    left = N - S["done"] - S["skip"] - S["fail"]
    eta = str(datetime.timedelta(seconds=int(avg*left))) if avg else "?"
    fps = S["cur_n"]/max(now - S["vt"], 1e-6)
    line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | worker {WORKER} | "
            f"done {S['done']} skip {S['skip']} fail {S['fail']} / {N} | "
            f"chờ backup {buffered()}, đã lên Drive {S['synced']} | "
            f"cur {S['cur']} {S['cur_n']}/{S['cur_tot']} fr ({fps:.1f} fps) | ETA {eta}"
            + (" | FINISHED" if final else ""))
    print(line)
    try:
        open(f"{LOGDIR}/worker{WORKER}.log", "a").write(line + "\n")
        open(f"{LOGDIR}/worker{WORKER}_status.txt", "w").write(line + "\n")
    except Exception as e: print("log error:", e)


def sync(force=False):
    if not force and time.time() - S["sync"] < SYNC_EVERY: return
    S["sync"] = time.time()
    for root, _, files in os.walk(BUF):
        for f in files:
            if not f.endswith(".tar"): continue
            lp = os.path.join(root, f)
            dst = f"{OUT}/{os.path.relpath(lp, BUF)}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(lp, dst + ".part"); os.replace(dst + ".part", dst)
            os.remove(lp); S["synced"] += 1


# ---------- xử lý ----------
zc = {}
def get_zip(method):
    if method not in zc:
        z = zipfile.ZipFile(zpath(method))
        zc[method] = (z, {os.path.basename(n): n for n in z.namelist()
                          if n.endswith(".mp4") and "__MACOSX" not in n})
    return zc[method]


dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=dev, post_process=False)


def sq_box(b, w, h):
    x1, y1, x2, y2 = b
    cx, cy = (x1+x2)/2, (y1+y2)/2
    s = min(max(x2-x1, y2-y1)*(1+MARGIN), w, h)
    x1 = int(np.clip(cx-s/2, 0, w-s)); y1 = int(np.clip(cy-s/2, 0, h-s))
    return x1, y1, int(x1+s), int(y1+s)


def detect(frames):
    h, w = frames[0].shape[:2]; sc = min(1.0, DET_SHORT/min(h, w))
    small = [cv2.resize(f, None, fx=sc, fy=sc) for f in frames] if sc < 1 else frames
    boxes, _ = mtcnn.detect(np.stack(small))
    out = []
    for b in boxes:
        if b is None or len(b) == 0: out.append(None); continue
        big = max(b, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
        out.append(sq_box(big/sc, w, h))
    return out


def process(src, vid, tmp):
    cap = cv2.VideoCapture(src); tf = tarfile.open(tmp, "w")
    S["cur_tot"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); S["cur_n"] = 0
    last = None; pending = []
    def emit(fr, bx):
        x1, y1, x2, y2 = bx
        c = cv2.resize(fr[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        S["cur_n"] += 1
        ti = tarfile.TarInfo(f"{vid}/{S['cur_n']:04d}.jpg"); ti.size = len(buf)
        tf.addfile(ti, io.BytesIO(buf.tobytes()))
    done = False
    while not done:
        fr = []
        while len(fr) < BATCH:
            ok, f = cap.read()
            if not ok: done = True; break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if not fr: break
        for f, bx in zip(fr, detect(fr)):
            if bx is not None:
                last = bx
                for pf in pending: emit(pf, bx)
                pending = []; emit(f, bx)
            elif last is not None: emit(f, last)
            else: pending.append(f)
        heartbeat(); sync()
    cap.release(); tf.close()
    return S["cur_n"] if (S["cur_n"] > 0 and not pending) else 0


tmp = f"/content/tmp_w{WORKER}.tar"; src = f"/content/v_w{WORKER}.mp4"
heartbeat(force=True)
finished = False
try:
    for i, r in enumerate(vids.itertuples(), 1):
        vid = os.path.splitext(r.filename)[0]
        rel = f"{r.method}/{vid}.tar"
        S["cur"] = f"{r.method}/{vid}"; S["cur_n"] = 0; S["cur_tot"] = 0
        if os.path.exists(f"{OUT}/{rel}") or os.path.exists(f"{BUF}/{rel}"):
            S["skip"] += 1; heartbeat(); continue
        S["vt"] = t0 = time.time()
        z, idx = get_zip(r.method)
        with z.open(idx[r.filename]) as a, open(src, "wb") as b: shutil.copyfileobj(a, b)
        n = process(src, vid, tmp)
        if n == 0:
            S["fail"] += 1
            open(f"{LOGDIR}/failed_worker{WORKER}.txt", "a").write(f"{S['cur']}\n")
            heartbeat(force=True); continue
        os.makedirs(f"{BUF}/{r.method}", exist_ok=True)
        shutil.move(tmp, f"{BUF}/{rel}")
        S["done"] += 1; S["times"].append(time.time() - t0)
        sync(); heartbeat(force=True)
    finished = True
except BaseException as e:
    S["cur"] = f"INTERRUPTED ({type(e).__name__}) at {S['cur']}"
    raise
finally:
    sync(force=True)                      # backup lần cuối khi hết việc hoặc khi bạn bấm Stop
    heartbeat(force=True, final=finished)
