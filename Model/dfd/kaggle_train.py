# ============================================================
# DFD-FCG | Save & Run All | LẦN 2: epoch 11 -> 30 (resume từ ckpt_run1/last.ckpt)
# ============================================================
import os, sys, glob, re, time, shutil, subprocess, textwrap, yaml

MAX_EPOCHS = 30            # tổng tích lũy (lần 1 đã xong 10)
RUN_NAME   = "ckpt_run2"
PREV_RUN   = "ckpt_run1"
MAX_HOURS  = 10.5          # lưới an toàn cho riêng lần chạy này (giới hạn Kaggle là 12 giờ)

W    = "/kaggle/working"
REPO = f"{W}/DFD-FCG"
LOG  = f"{W}/train_{RUN_NAME}.log"
T0   = time.time()

def log(msg):
    line = f"[{(time.time()-T0)/60:6.1f} ph] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, executable="/bin/bash")
    if check and r.returncode != 0:
        raise RuntimeError(f"Lệnh lỗi ({r.returncode}): {cmd}\n{(r.stdout + r.stderr)[-1500:]}")
    return r.stdout.strip()

# ---------- 1. kiểm tra môi trường + tìm checkpoint lần 1 (DỪNG NGAY nếu không thấy) ----------
n_gpu = len([l for l in sh("nvidia-smi -L", check=False).splitlines() if l.startswith("GPU")])
if n_gpu != 2:
    raise SystemExit(f"Cần đúng 2 GPU (T4 x2), hiện thấy {n_gpu}. Đổi Accelerator rồi chạy lại.")

def find(default, tail):
    if os.path.exists(default):
        return default
    hits = glob.glob(f"/kaggle/input/*/{tail}") + glob.glob(f"/kaggle/input/*/*/{tail}")
    if hits:
        return hits[0]
    raise SystemExit(f"Không thấy {tail} trong /kaggle/input. Hãy Add Input dataset tương ứng.")

CSV  = find("/kaggle/input/datasets/nhalac/ffpp-master-747/master_747.csv", "ffpp-master-747/master_747.csv")
ROOT = find("/kaggle/input/datasets/nhalac/dfd-preprocessed-b", "dfd-preprocessed-b")

cands = (glob.glob(f"{W}/{PREV_RUN}/last.ckpt")
         + glob.glob(f"/kaggle/input/**/{PREV_RUN}/last.ckpt", recursive=True))
log(f"last.ckpt của lần 1 tìm thấy: {cands}")
if not cands:
    raise SystemExit(f"KHÔNG thấy {PREV_RUN}/last.ckpt ở {W} lẫn /kaggle/input. Dừng để không train lại từ đầu. "
                     "Gắn output của lần 1 làm input (Add Input > Notebook Output Files) rồi chạy lại.")
RESUME_CKPT = cands[0]

import torch
ck = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)
done = int(ck["epoch"]) + 1
log(f"checkpoint: epoch(đếm từ 0)={ck['epoch']} global_step={ck['global_step']} -> đã xong {done} epoch")
del ck
if done >= MAX_EPOCHS:
    raise SystemExit(f"Checkpoint đã xong {done} epoch >= MAX_EPOCHS={MAX_EPOCHS}, không còn gì để train.")
log(f"GPU={n_gpu} | sẽ train epoch {done + 1} -> {MAX_EPOCHS} ({MAX_EPOCHS - done} epoch) | RESUME={RESUME_CKPT}")

# ---------- 2. clone repo (ghim commit) + cài thư viện ----------
sh(f"rm -rf {REPO} && git clone -q https://github.com/aiiu-lab/DFD-FCG.git {REPO} && git -C {REPO} checkout -q 3c0b840")
log("đã clone repo @ " + sh(f"git -C {REPO} log -1 --format='%h %ad'"))
sh("pip install -q lightning 'jsonargparse[signatures]' open_clip_torch ftfy wandb 'albumentations>=1.4,<1.5' 2>&1")
log("đã cài thư viện")

# ---------- 3. ghi các file ----------
open(f"{REPO}/src/utility/cv2_video.py", "w").write(r'''
import cv2, torch, torchvision.io as tio

class VideoReader:
    def __init__(self, path, stream="video", num_threads=0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._idx = 0
        self._last = None

    def get_metadata(self):
        return {"video": {"fps": [self.fps], "duration": [self.n / self.fps]}}

    def seek(self, t, keyframes_only=False):
        self._idx = max(0, min(int(round(t * self.fps)), self.n - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._idx)

    def __iter__(self):
        return self

    def __next__(self):
        ok, f = self.cap.read()
        if ok:
            self._last = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        elif self._last is None:
            raise StopIteration
        data = torch.from_numpy(self._last).permute(2, 0, 1).contiguous()
        pts = self._idx / self.fps
        self._idx += 1
        return {"data": data, "pts": pts}

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass

tio.VideoReader = VideoReader
''')

open(f"{REPO}/src/dataset/ours.py", "w").write(r'''
import os, pickle, random, logging
import cv2
import pandas as pd
from os import path
from typing import List
from .ffpp import FFPP, FFPPSampleStrategy, FFPPAugmentation
from .base import DeepFakeDataModule

GROUP_DIRS = {
    'REAL': 'ffpp_real_c23', 'DF': 'ffpp_deepfakes_c23', 'FS': 'ffpp_faceswap_c23',
    'FSH': 'ffpp_faceshifter_c23', 'FSGAN': 'df40_fsgan', 'SIMSWAP': 'df40_simswap',
    'INSWAP': 'df40_inswap', 'BLENDFACE': 'df40_blendface',
    'MOBILESWAP': 'df40_mobileswap', 'FACEDANCER': 'df40_facedancer',
}
METHOD_TO_TYPE = {
    'real': 'REAL', 'deepfakes': 'DF', 'faceswap': 'FS', 'faceshifter': 'FSH',
    'fsgan': 'FSGAN', 'simswap': 'SIMSWAP', 'inswap': 'INSWAP',
    'blendface': 'BLENDFACE', 'mobileswap': 'MOBILESWAP', 'facedancer': 'FACEDANCER',
}


class OursFFPP(FFPP):
    TYPE_DIRS = {k: k for k in GROUP_DIRS}

    @property
    def cls_name(self):
        return "FFPP"

    def __init__(self, master_csv, video_root, meta_cache, *args, **kargs):
        self.master_csv = master_csv
        self.video_root = video_root
        self.meta_cache = meta_cache
        super().__init__(*args, **kargs)

    @staticmethod
    def _probe(p):
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            return (0.0, 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (fps, n)

    def _build_video_table(self):
        df = pd.read_csv(self.master_csv, dtype=str)
        df = df[df["split"] == self.split]
        cache = {}
        if path.exists(self.meta_cache):
            with open(self.meta_cache, "rb") as f:
                cache = pickle.load(f)
        dirty = False
        table = {t: {c: {} for c in self.compressions} for t in self.df_types}
        for row in df.itertuples():
            t = METHOD_TO_TYPE[row.method]
            if t not in table:
                continue
            name = row.filename[:-len(".mp4")]
            g = GROUP_DIRS[t]
            p = path.join(self.video_root, g, "videos", g, name + ".avi")
            if p not in cache:
                cache[p] = self._probe(p)
                dirty = True
            fps, n = cache[p]
            if n <= 0:
                logging.warning(f"unreadable video: {p}")
                continue
            for c in self.compressions:
                table[t][c][name] = {"fps": fps, "frames": n, "duration": n / fps, "path": p}
        if dirty:
            with open(self.meta_cache, "wb") as f:
                pickle.dump(cache, f)
        self.video_table = table

    def _build_video_list(self):
        self.video_list = []
        for t in self.df_types:
            for comp in self.compressions:
                comp_videos = []
                for idx in sorted(self.video_table[t][comp]):
                    clips = int(self.video_table[t][comp][idx]["duration"] // self.clip_duration)
                    if clips > 0:
                        comp_videos.append((t, comp, idx, min(clips, self.max_clips)))
                self.video_list += comp_videos[:int(len(comp_videos) * self.ratio)]

        random.Random(1019).shuffle(self.video_list)

        self.stack_video_clips = [0]
        self.real_clip_idx = {}
        self.fake_clip_idx = {'back': {}, 'fore': {}}
        for df_type, _, idx, i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
            interval = [self.stack_video_clips[-2], self.stack_video_clips[-1] - 1]
            if df_type == "REAL":
                self.real_clip_idx[idx] = interval
            else:
                back_idx, fore_idx = idx.split('_')
                self.fake_clip_idx['back'].setdefault(back_idx, []).append(interval)
                self.fake_clip_idx['fore'].setdefault(fore_idx, []).append(interval)
        self.stack_video_clips.pop(0)


class OursDataModule(DeepFakeDataModule):
    def __init__(
        self,
        master_csv: str,
        video_root: str,
        meta_cache: str = '/kaggle/working/meta_cache.pkl',
        df_types: List[str] = [],
        strategy: FFPPSampleStrategy = FFPPSampleStrategy.NORMAL,
        augmentations: List[FFPPAugmentation] = [FFPPAugmentation.NONE],
        force_random_speed: bool = None,
        *args,
        **kargs
    ):
        kargs.setdefault('vid_ext', '.avi')
        kargs.setdefault('data_dir', video_root)
        super().__init__(*args, **kargs)
        self.master_csv = master_csv
        self.video_root = video_root
        self.meta_cache = meta_cache
        self.df_types = [t for t in (df_types or list(GROUP_DIRS)) if t in GROUP_DIRS]
        self.strategy = strategy
        self.augmentations = FFPPAugmentation(sum(augmentations))
        self.force_random_speed = force_random_speed

    def affine_model(self, model):
        super().affine_model(model)
        self.n_px = model.n_px

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        from functools import partial
        data_cls = partial(
            OursFFPP,
            master_csv=self.master_csv, video_root=self.video_root, meta_cache=self.meta_cache,
            data_dir=self.data_dir, vid_ext=self.vid_ext,
            df_types=self.df_types, compressions=['c23'],
            num_frames=self.num_frames, clip_duration=self.clip_duration,
            transform=self.transform, n_px=self.n_px, ratio=self.ratio,
            force_random_speed=self.force_random_speed,
        )
        if stage == "fit":
            self._train_dataset = data_cls(split="train", pack=self.pack, strategy=self.strategy,
                                           augmentations=self.augmentations, max_clips=self.max_clips)
            self._val_dataset = data_cls(split="val", pack=self.pack, strategy=FFPPSampleStrategy.NORMAL,
                                         augmentations=FFPPAugmentation.NONE, max_clips=self.max_clips)
        elif stage == "test":
            self._test_dataset = data_cls(split="test", pack=self.pack, strategy=FFPPSampleStrategy.NORMAL,
                                          augmentations=FFPPAugmentation.NONE)
''')

# callback: tiến trình trong epoch + val metric + thông tin dataloader + đồng hồ an toàn riêng
open(f"{REPO}/src/utility/progress_cb.py", "w").write(r'''
import time
from lightning.pytorch.callbacks import Callback


class ProgressLog(Callback):
    def __init__(self, every=100, max_hours=None):
        self.every = every
        self.max_s = None if max_hours is None else float(max_hours) * 3600
        self.t0 = time.time()
        self.t_ep = time.time()
        self.guard_hit = False

    def _p(self, trainer, msg):
        if trainer.is_global_zero:
            print(msg, flush=True)

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
        try:
            it = trainer.train_dataloader
            it = getattr(it, "iterables", it)
            dl = list(it.values())[0] if isinstance(it, dict) else it
            self._p(trainer, f"[info] train: items={len(dl.dataset)} batch/GPU={dl.batch_size} "
                             f"batches={len(dl)} accumulate={trainer.accumulate_grad_batches} GPUs={trainer.world_size} "
                             f"epoch_bat_dau={trainer.current_epoch} max_epochs={trainer.max_epochs}")
        except Exception as e:
            self._p(trainer, f"[info] khong doc duoc: {e!r}")

    def on_train_epoch_start(self, trainer, pl_module):
        try:
            self.t_ep = time.time()
            self._p(trainer, f"[epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] bat dau")
        except Exception:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            n, tot = batch_idx + 1, trainer.num_training_batches
            if n % self.every == 0 or n == tot:
                el = time.time() - self.t_ep
                ips = n / max(el, 1e-6)
                rem = (tot - n) / ips
                self._p(trainer, f"[epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] {n}/{tot} "
                                 f"({100 * n / tot:.0f}%) {ips:.2f} it/s | da {el / 60:.1f} ph | con ~{rem / 60:.1f} ph")
            # chi kiem tra o moi moc `every` batch de hai GPU quyet dinh cung luc
            if (self.max_s is not None and not self.guard_hit and n % self.every == 0
                    and time.time() - self.t0 > self.max_s):
                self.guard_hit = True
                trainer.should_stop = True
                self._p(trainer, f"[guard] het {self.max_s / 3600:.1f} gio cua lan chay nay -> dung")
        except Exception:
            pass

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            e = trainer.current_epoch + 1
            m = trainer.callback_metrics
            parts = []
            for k in ("auc", "ap", "acc", "loss"):
                v = m.get(f"valid/FFPP/{k}")
                if v is not None:
                    parts.append(f"{k}={float(v):.4f}")
            self._p(trainer, f"[val] epoch {e} valid/FFPP: " + (" ".join(parts) or "chua co metric"))
            self._p(trainer, f"[epoch {e}/{trainer.max_epochs}] XONG sau {(time.time() - self.t_ep) / 60:.1f} ph "
                             f"| tong {(time.time() - self.t0) / 3600:.2f} gio")
        except Exception:
            pass
''')

open(f"{REPO}/train_ours.py", "w").write(textwrap.dedent('''
    import os, sys, gc, logging, warnings, torch
    sys.path.insert(0, os.getcwd())
    import src.utility.cv2_video  # vá VideoReader trước khi import dataset
    from src.utility.builtin import ODTrainer, ODLightningCLI
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level="INFO", format="[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s")
    warnings.filterwarnings("ignore")
    cli = ODLightningCLI(run=False, trainer_class=ODTrainer,
                         save_config_kwargs={"config_filename": "setting.yaml"},
                         auto_configure_optimizers=True, seed_everything_default=1019)
    cli.datamodule.affine_model(cli.model)
    cli.datamodule.affine_trainer(cli.trainer)
    try:
        from src.utility.progress_cb import ProgressLog
        cli.trainer.callbacks.append(ProgressLog(max_hours=float(os.environ.get("MAX_HOURS", "0")) or None))
    except Exception as e:
        print("[progress_cb] khong gan duoc:", repr(e), flush=True)
    ckpt = cli.config.ckpt_path if cli.config.ckpt_mode == "cont" else None
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt)
    gc.collect(); torch.cuda.empty_cache()
'''))

cfg = yaml.safe_load(open(f"{REPO}/configs/base.yaml"))
cfg["lr_scheduler"].pop("verbose", None)
yaml.safe_dump(cfg, open(f"{REPO}/configs/base_k.yaml", "w"), sort_keys=False)

open(f"{REPO}/configs/ours.yaml", "w").write(textwrap.dedent(f'''
trainer:
  devices: 2
  deterministic: warn
  limit_val_batches: 1.0
  accumulate_grad_batches: 8
  default_root_dir: {W}/logs
  logger:
    init_args:
      offline: true
      save_dir: {W}/logs
data:
  class_path: src.dataset.base.ODDeepFakeDataModule
  init_args:
    batch_size: 8
    num_workers: 4
    clip_duration: 3
    num_frames: 10
    train_datamodules:
    - class_path: src.dataset.ours.OursDataModule
      init_args:
        batch_size: 32
        master_csv: {CSV}
        video_root: {ROOT}
        data_dir: {ROOT}
        vid_ext: .avi
        strategy: FORCE_PAIR
        augmentations: [NORMAL, VIDEO, VIDEO_RRC, FRAME]
        max_clips: 3
    val_datamodules:
    - class_path: src.dataset.ours.OursDataModule
      init_args:
        master_csv: {CSV}
        video_root: {ROOT}
        data_dir: {ROOT}
        vid_ext: .avi
        strategy: NORMAL
        augmentations: [NONE]
        max_clips: 1
'''))
log("đã ghi các file")

# ---------- 4. tạo cache thời lượng video TRƯỚC khi chạy DDP + kiểm tra dữ liệu ----------
open(f"{W}/precache.py", "w").write(r'''
import os, sys
sys.path.insert(0, os.getcwd())
import src.utility.cv2_video
from src.dataset.ours import OursFFPP, GROUP_DIRS, FFPPSampleStrategy, FFPPAugmentation
CSV, ROOT = os.environ["CSV"], os.environ["ROOT"]
for split, want in {"train": 5380, "val": 1050, "test": 1040}.items():
    ds = OursFFPP(CSV, ROOT, "/kaggle/working/meta_cache.pkl",
                  df_types=list(GROUP_DIRS), compressions=["c23"], n_px=224,
                  strategy=FFPPSampleStrategy.NORMAL, augmentations=FFPPAugmentation.NONE,
                  force_random_speed=None, data_dir=ROOT, vid_ext=".avi", num_frames=10,
                  clip_duration=3, split=split, transform=lambda x: x, pack=False,
                  ratio=1.0, max_clips=1)
    n = len(ds.video_list)
    print(f"[data] {split}: {n} video (mong doi {want})", flush=True)
    if n != want:
        raise SystemExit(f"So video {split} khong khop: {n} != {want}")
''')
out = sh(f"NO_ALBUMENTATIONS_UPDATE=1 CSV={CSV} ROOT={ROOT} python {W}/precache.py", cwd=REPO)
for l in out.splitlines():
    if l.startswith("[data]"):
        log(l)

# ---------- 5. train (chặn tới khi xong) ----------
env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="offline", PYTHONUNBUFFERED="1",
           MAX_HOURS=str(MAX_HOURS))
cmd = ["python", "train_ours.py",
       "--config", "configs/base_k.yaml", "--config", "configs/clip/L14/ffg.yaml", "--config", "configs/ours.yaml",
       f"--trainer.max_epochs={MAX_EPOCHS}",
       "--ckpt_path", RESUME_CKPT, "--ckpt_mode", "cont"]
shutil.rmtree(f"{W}/logs", ignore_errors=True)       # chỉ xóa logs; KHÔNG đụng tới ckpt_run1/
log("bắt đầu train: " + " ".join(cmd))

SHOW = ("[epoch", "[val", "[info", "[guard", "[progress_cb]", "Traceback", "Error", "Starting with 2", "Trainable params")
p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, errors="replace")
with open(LOG, "a") as lf:
    for line in p.stdout:
        lf.write(line)
        lf.flush()
        s = line.strip()
        if s and any(k in s for k in SHOW) and len(s) < 400:
            print(s, flush=True)
rc = p.wait()
log(f"train kết thúc, exit code = {rc}")

# ---------- 6. gom checkpoint vào /kaggle/working/<RUN_NAME>/ + tóm tắt val AUC ----------
dst = f"{W}/{RUN_NAME}"
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{W}/logs/**/checkpoints/*.ckpt", recursive=True):
    shutil.move(f, os.path.join(dst, os.path.basename(f)))
for s in glob.glob(f"{W}/logs/**/setting.yaml", recursive=True):
    shutil.copy2(s, os.path.join(dst, "setting.yaml"))
    break
for d in (PREV_RUN, RUN_NAME):
    for f in sorted(glob.glob(f"{W}/{d}/*")):
        log(f"  {d}/{os.path.basename(f)}  {os.path.getsize(f) / 2**30:.2f} GB")

rows = re.findall(r"
val
epoch (\d+) valid/FFPP: auc=([0-9.]+)", open(LOG).read())
if rows:
    best = max(rows, key=lambda r: float(r[1]))
    log("val AUC theo epoch: " + ", ".join(f"{e}:{a}" for e, a in rows))
    log(f"AUC val cao nhất của lần chạy này: epoch {best[0]} = {best[1]}")
if rc != 0:
    log("!!! TRAIN THAT BAI: xem dòng Traceback ở trên hoặc trong file " + LOG)
log(f"XONG, tổng {(time.time() - T0) / 3600:.2f} giờ")

