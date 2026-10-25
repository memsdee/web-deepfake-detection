# validation.py
# Lưu và kiểm tra video tải lên. Dùng NGUYÊN VĂN của repo gốc, chỉ đổi TMP_DIR
# thành thư mục con riêng của demo (app.py tạo & dọn thư mục này lúc khởi động).
import asyncio, json, os, subprocess, tempfile, uuid
from fastapi import HTTPException, UploadFile

MAX_BYTES   = 60 * 1024 * 1024   # 60 MB
MAX_SECONDS = 15.0             
MIN_SECONDS = 3.0
ALLOWED_EXT = {".mp4", ".mov", ".webm"}

# Thư mục RAM riêng cho demo; vẫn có fallback sang temp của hệ thống.
_SHARED_TMP = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
TMP_DIR     = os.path.join(_SHARED_TMP, "dfd_demo")
CHUNK       = 1024 * 1024


async def save_upload(file: UploadFile) -> str:
    """Lưu theo chunk, ngắt ngay nếu vượt MAX_BYTES. Trả về đường dẫn file tạm."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, "Định dạng không hỗ trợ. Chỉ nhận mp4, mov, webm.")

    path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")
    size = 0
    try:
        with open(path, "wb") as f:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video vượt quá 60 MB.")
                f.write(chunk)
        if size == 0:
            raise HTTPException(422, "File rỗng.")
    except BaseException:
        safe_remove(path)
        raise
    return path


def _ffprobe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        raise HTTPException(422, "Không đọc được video (quá thời gian).")
    if out.returncode != 0:
        raise HTTPException(422, "File không phải video hợp lệ.")
    return json.loads(out.stdout)


def _duration(info: dict, stream: dict) -> float | None:
    # webm đôi khi không có duration ở format, thử cả stream
    for src in (info.get("format", {}), stream):
        try:
            return float(src["duration"])
        except (KeyError, ValueError, TypeError):
            continue
    return None


async def validate_video(path: str) -> dict:
    """Chạy ffprobe trong thread để không chặn event loop. Ném HTTPException nếu lỗi."""
    info = await asyncio.to_thread(_ffprobe, path)
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise HTTPException(422, "File không có luồng video.")
    dur = _duration(info, vstreams[0])
    if dur is None:
        raise HTTPException(422, "Không xác định được thời lượng video.")
    if dur > MAX_SECONDS:
        raise HTTPException(413, f"Video dài {dur:.1f}s, tối đa {MAX_SECONDS:.0f}s.")
    if dur < MIN_SECONDS:
        raise HTTPException(422, "Video quá ngắn.")
    return {"duration": dur,
            "width": vstreams[0].get("width"),
            "height": vstreams[0].get("height")}


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
