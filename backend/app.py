"""app.py — Backend FastAPI cho web demo phát hiện deepfake video.

Chạy:  uvicorn app:app --host 0.0.0.0 --port 8000
Một process duy nhất, trạng thái job nằm trong RAM.

Contract với frontend (js/config.js, js/api-client.js, js/result-view.js):
  POST /api/analyze       multipart field "file" -> 202 {"job_id"}
  GET  /api/jobs/{id}     {"status": queued|running|done|error,
                           "queue_position", "result", "error", "error_code"}
  result: {"p_fake", "low", "high", "zone", "processing_time", "n_clips", "clips"}
  error:  chuỗi tiếng Việt (FE hiển thị thẳng), error_code: "no_face"|"internal"
"""
import asyncio
import logging
import os
import shutil
import signal
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import validation
from inference import InferenceEngine, NoFaceError

# ---------------------------------------------------------------- cấu hình
LOW, HIGH = 0.35, 0.55              # ngưỡng tạm, phải hiệu chỉnh trên val split
MAX_QUEUED = 3                      # job ĐANG CHỜ tối đa (không tính đang xử lý)
QUEUE_FULL_RETRY_AFTER = 30         # giây, header Retry-After khi 429
RESULT_TTL_SECONDS = 10 * 60        # giữ kết quả 10 phút sau khi job kết thúc
CLEANUP_INTERVAL = 30               # chu kỳ dọn job hết hạn (giây)
DEVICE_ENV = "DFD_STUB"             # DFD_STUB=1 -> luôn dùng stub trên CPU
DEFAULT_PORT = 8000

APP_VERSION = "1.0"

MSG_NO_FACE = "Không tìm thấy khuôn mặt rõ ràng trong video"
MSG_INTERNAL = "Có lỗi khi xử lý video. Vui lòng thử lại sau."
MSG_FFPROBE_MISSING = "Máy chủ thiếu ffprobe (gói ffmpeg), không kiểm tra được video"

STATUS_QUEUED, STATUS_PROCESSING = "queued", "running"
STATUS_DONE, STATUS_ERROR = "done", "error"

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT.parent if (ROOT.parent / "index.html").is_file() else ROOT

log = logging.getLogger("dfd")


# ------------------------------------------------------------------ hàng chờ
@dataclass
class Job:
    id: str
    path: str                       # file video tạm, xoá ngay sau predict()
    status: str = STATUS_QUEUED
    result: dict | None = None
    error: str | None = None
    error_code: str | None = None
    finished_at: float | None = None
    meta: dict = field(default_factory=dict)


class JobQueue:
    """Hàng chờ FIFO + bảng job. Mọi thay đổi nằm dưới một asyncio.Lock."""

    def __init__(self, max_queued: int):
        self._max_queued = max_queued
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self.processing = 0

    async def submit(self, job: Job) -> int | None:
        """Trả vị trí trong hàng chờ (1-based) hoặc None nếu hàng đầy."""
        async with self._cond:
            if len(self._pending) >= self._max_queued:
                return None
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._cond.notify()
            return len(self._pending)

    async def take(self) -> Job | None:
        async with self._cond:
            while not self._pending:
                await self._cond.wait()
            job = self._jobs.get(self._pending.pop(0))
            if job is None:                       # đã bị dọn trong lúc chờ
                return None
            job.status = STATUS_PROCESSING
            self.processing += 1
            return job

    async def finish(self, job: Job, result: dict | None, error: str | None, code: str | None) -> None:
        async with self._lock:
            job.status = STATUS_DONE if error is None else STATUS_ERROR
            job.result, job.error, job.error_code = result, error, code
            job.finished_at = time.time()
            self.processing = max(0, self.processing - 1)

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    def position(self, job_id: str) -> int | None:
        try:
            return self._pending.index(job_id) + 1
        except ValueError:
            return None

    async def purge_expired(self) -> int:
        now = time.time()
        async with self._cond:
            expired = [jid for jid, j in self._jobs.items()
                       if j.finished_at is not None and now - j.finished_at > RESULT_TTL_SECONDS]
            for jid in expired:
                self._jobs.pop(jid, None)
            return len(expired)

    async def counts(self) -> tuple[int, int]:
        async with self._lock:
            return len(self._pending), self.processing


# ------------------------------------------------------------ suy luận
class EngineFactory:
    """Tạo một InferenceEngine theo device; nếu thiếu torch/GPU -> stub cpu."""

    def __init__(self):
        self.engines: dict[str, InferenceEngine] = {}
        self.devices: list[str] = []
        self.stub = False

    def build(self) -> list[str]:
        devices = self._target_devices()
        for dev in devices:
            self.engines[dev] = InferenceEngine(dev)
        self.devices = devices
        return devices

    def _target_devices(self) -> list[str]:
        forced_stub = os.environ.get(DEVICE_ENV) == "1"
        if forced_stub:
            self.stub = True
            return ["cpu"]
        try:
            import torch
            count = torch.cuda.device_count()
        except Exception:                         # không có torch -> stub
            self.stub = True
            return ["cpu"]
        raw = os.environ.get("DEVICES", "").strip()
        if count <= 0:
            self.stub = True
            return ["cpu"]
        devices = [d.strip() for d in raw.split(",") if d.strip()] if raw else \
            [f"cuda:{i}" for i in range(count)]
        return devices or ["cuda:0"]


factory = EngineFactory()


def zone_of(p: float) -> str:
    if p < LOW:
        return "low"
    if p >= HIGH:
        return "high"
    return "uncertain"


async def run_worker(device: str, queue: JobQueue) -> None:
    engine = factory.engines[device]
    while True:
        job = await queue.take()
        if job is None:
            continue
        result = error = code = None
        try:
            raw = await asyncio.to_thread(engine.predict, job.path)
            p = float(raw["prob_fake"])
            timings = raw.get("timings") or {}
            clips = raw.get("clips") or []
            total = timings.get("total", 0.0)
            result = {
                "p_fake": p,
                "low": LOW,                   # nguồn duy nhất cho ngưỡng phía FE
                "high": HIGH,
                "zone": zone_of(p),
                "processing_time": round(float(total), 3),
                "n_clips": int(raw.get("n_clips", len(clips))),
                "clips": clips,
            }
        except NoFaceError:
            error, code = MSG_NO_FACE, "no_face"
        except Exception:
            log.error("predict failed (job %s)\n%s", job.id, traceback.format_exc())
            error, code = MSG_INTERNAL, "internal"
        finally:
            validation.safe_remove(job.path)      # xoá video ngay, cả khi lỗi
            job.path = ""
        await queue.finish(job, result, error, code)


async def cleanup_loop(queue: JobQueue) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            await queue.purge_expired()
        except Exception:
            log.exception("cleanup failed")


# ------------------------------------------------------------------- app
def clean_tmp_dir() -> None:
    os.makedirs(validation.TMP_DIR, exist_ok=True)
    for entry in os.scandir(validation.TMP_DIR):
        try:
            if entry.is_dir():
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.remove(entry.path)
        except OSError:
            log.warning("cannot remove leftover %s", entry.path)


@dataclass
class State:
    queue: JobQueue | None = None
    workers: list[asyncio.Task] = field(default_factory=list)
    cleanup: asyncio.Task | None = None


state = State()


def cleanup_done(task: asyncio.Task) -> None:
    """Worker chết bất thường -> dừng process để không treo app."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.critical("worker %s crashed", task.get_name(), exc_info=exc)
        os.kill(os.getpid(), signal.SIGINT)


async def lifespan(app: FastAPI):
    clean_tmp_dir()
    devices = factory.build()                 # lỗi ở đây -> app không khởi động
    log.info("inference engine ready: %s (stub=%s)", devices, factory.stub)
    state.queue = JobQueue(MAX_QUEUED)
    state.workers = [asyncio.create_task(run_worker(d, state.queue), name=f"worker-{d}")
                     for d in devices]
    for t in state.workers:
        t.add_done_callback(cleanup_done)
    state.cleanup = asyncio.create_task(cleanup_loop(state.queue), name="cleanup")
    try:
        yield
    finally:
        for t in state.workers + [state.cleanup]:
            t.cancel()
        await asyncio.gather(*state.workers, state.cleanup, return_exceptions=True)


app = FastAPI(title="Deepfake demo", lifespan=lifespan)

# Tài nguyên tĩnh của frontend (css/, js/, video mẫu). App chạy 1 process nên
# mount tĩnh không ảnh hưởng gì tới hàng chờ.
app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")


@app.get("/test_sample.mp4", include_in_schema=False)
async def sample_video():
    # Video mẫu cho nút "Demo Mode" của frontend (nếu có)
    path = WEB_DIR / "test_sample.mp4"
    if not path.is_file():
        raise HTTPException(404, "Không có video mẫu")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/analyze", status_code=202)
async def analyze(request: Request, file: UploadFile = File(...)):
    queue: JobQueue = state.queue
    # Contract FE: field multipart tên "file", thành công trả {"job_id": ...}
    path = await validation.save_upload(file)
    try:
        try:
            await validation.validate_video(path)
        except FileNotFoundError:
            # validation.py gọi trực tiếp "ffprobe"; thiếu binary thì ném FileNotFoundError
            log.error("ffprobe không tồn tại trên máy chủ")
            raise HTTPException(503, MSG_FFPROBE_MISSING)
        job = Job(id=uuid.uuid4().hex, path=path)
        position = await queue.submit(job)
        if position is None:
            # Contract FE: 429 -> "Máy chủ đang bận", dùng header Retry-After
            raise HTTPException(429, "Hệ thống đang bận, vui lòng thử lại sau.",
                                headers={"Retry-After": str(QUEUE_FULL_RETRY_AFTER)})
        job.meta["filename"] = file.filename
        job.meta["client"] = request.client.host if request.client else None
        return {"job_id": job.id, "queue_position": position}
    except BaseException:
        validation.safe_remove(path)          # lỗi validate / hàng đầy -> xoá ngay
        raise


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    queue: JobQueue = state.queue
    job = await queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy hoặc đã hết hạn")
    out = {"job_id": job.id, "status": job.status}
    if job.status == STATUS_QUEUED:
        out["queue_position"] = queue.position(job.id)
    elif job.status == STATUS_DONE:
        out["result"] = job.result
    elif job.status == STATUS_ERROR:
        out["error"] = job.error
        out["error_code"] = job.error_code
    return out


@app.get("/api/health")
async def health():
    queued, processing = await state.queue.counts()
    return {
        "status": "ok",
        "workers": len(factory.devices),
        "devices": factory.devices,
        "stub": factory.stub,
        "queued": queued,
        "processing": processing,
    }


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


# Gọi trực tiếp `python app.py` để tự chọn 1 worker
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", DEFAULT_PORT)),
                workers=1, log_level="info")
