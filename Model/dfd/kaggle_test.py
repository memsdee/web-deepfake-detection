import os, subprocess, textwrap, yaml, json, glob

W, REPO = "/kaggle/working", "/kaggle/working/DFD-FCG"
CKPT = "/kaggle/input/notebooks/nhalac/dfd-train-2/ckpt_run2/epoch=26-step=1377.ckpt"
assert os.path.exists(CKPT), CKPT

def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, executable="/bin/bash")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{(r.stdout + r.stderr)[-2500:]}")
    return r.stdout.strip()

def find(default, tail):
    if os.path.exists(default): return default
    hits = glob.glob(f"/kaggle/input/*/{tail}") + glob.glob(f"/kaggle/input/*/*/{tail}")
    if hits: return hits[0]
    raise SystemExit(f"Không thấy {tail}")

CSV  = find("/kaggle/input/datasets/nhalac/ffpp-master-747/master_747.csv", "ffpp-master-747/master_747.csv")
ROOT = find("/kaggle/input/datasets/nhalac/dfd-preprocessed-b", "dfd-preprocessed-b")

sh(f"rm -rf {REPO} && git clone -q https://github.com/aiiu-lab/DFD-FCG.git {REPO} && git -C {REPO} checkout -q 3c0b840")
sh("pip install -q lightning 'jsonargparse[signatures]' open_clip_torch ftfy wandb 'albumentations>=1.4,<1.5' 2>&1")

# ---- cv2_video.py và ours.py: COPY NGUYÊN VĂN từ cell train ----
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

open(f"{REPO}/test_ours.py", "w").write(textwrap.dedent('''
    import os, sys, json, warnings, torch
    sys.path.insert(0, os.getcwd())
    import src.utility.cv2_video
    from src.utility.builtin import ODTrainer, ODLightningCLI
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore")
    cli = ODLightningCLI(run=False, trainer_class=ODTrainer,
                         save_config_kwargs={"config_filename": "setting_test.yaml", "overwrite": True},
                         auto_configure_optimizers=True, seed_everything_default=1019)
    cli.datamodule.affine_model(cli.model)
    cli.datamodule.affine_trainer(cli.trainer)
    res = cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=os.environ["TEST_CKPT"])
    print("[TEST_RESULT]", json.dumps(res), flush=True)
    json.dump(res, open(os.environ["TEST_OUT"], "w"), indent=2)
'''))

cfg = yaml.safe_load(open(f"{REPO}/configs/base.yaml"))
cfg["lr_scheduler"].pop("verbose", None)
yaml.safe_dump(cfg, open(f"{REPO}/configs/base_k.yaml", "w"), sort_keys=False)

open(f"{REPO}/configs/ours_test.yaml", "w").write(textwrap.dedent(f'''
trainer:
  devices: 1
  deterministic: warn
  default_root_dir: {W}/logs_test
  logger:
    init_args:
      offline: true
      save_dir: {W}/logs_test
data:
  class_path: src.dataset.base.ODDeepFakeDataModule
  init_args:
    batch_size: 8
    num_workers: 4
    clip_duration: 3
    num_frames: 10
    train_datamodules: []
    val_datamodules: []
    test_datamodules:
    - class_path: src.dataset.ours.OursDataModule
      init_args:
        master_csv: {CSV}
        video_root: {ROOT}
        data_dir: {ROOT}
        vid_ext: .avi
        strategy: NORMAL
        augmentations: [NONE]
'''))

env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="offline", PYTHONUNBUFFERED="1",
           TEST_CKPT=CKPT, TEST_OUT=f"{W}/test_result_run2.json")
cmd = ["python", "test_ours.py", "--config", "configs/base_k.yaml",
       "--config", "configs/clip/L14/ffg.yaml", "--config", "configs/ours_test.yaml"]
p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
open(f"{W}/test_run2.log", "w").write(p.stdout + p.stderr)
print((p.stdout + p.stderr)[-3000:])