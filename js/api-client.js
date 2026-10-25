/* ==========================================================================
 GỬI VIDEO & POLLING KẾT QUẢ
 ========================================================================== */
import {
  API_BASE, UPLOAD_FIELD, ENDPOINT_SUBMIT, ENDPOINT_JOB,
  STATUS, MAX_MB, MAX_SECONDS, POLL_MS, CLIENT_TIMEOUT_MS, RESULT_TTL_MIN
} from "./config.js";
import { showStatus, failStatus, hideStatus, setProgressFromUpload } from "./status-ui.js";
import { renderResult } from "./result-view.js";
import { hideError } from "./error-banner.js";

let onUploadingChange = () => { };
let xhr = null, pollTimer = null, deadline = 0, jobId = null;

export function setUploadingHook(fn) { onUploadingChange = fn; }

function xhrBodySafe(x) {
  try { return x && x.response; } catch (_) { return null; }
}

function serverMessage(x) {
  const body = xhrBodySafe(x);
  if (body && typeof body === "object" && typeof body.detail === "string") return body.detail; // FastAPI {"detail": "..."}
  return "";
}

function abortUpload() {
  if (xhr) { xhr.abort(); xhr = null; }
}

export function submit(file, errorBox) {
  if (xhr) return;
  hideError(errorBox);
  onUploadingChange(true);
  showStatus("Đang tải video lên máy chủ…", "Vui lòng giữ kết nối ổn định.", true, 15);

  const form = new FormData();
  form.append(UPLOAD_FIELD, file, file.name || "video.mp4");

  xhr = new XMLHttpRequest();
  xhr.open("POST", API_BASE + ENDPOINT_SUBMIT);
  xhr.responseType = "json";
  xhr.timeout = 120000;

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setProgressFromUpload(Math.round((e.loaded / e.total) * 100));
  };

  xhr.addEventListener("loadend", () => {
    if (!xhr) return;
    const req = xhr;
    xhr = null;
    onUploadResponse(req);
  });

  xhr.addEventListener("timeout", () => {
    xhr = null;
    onUploadingChange(false);
    failStatus("Quá hạn thời gian tải lên", "Máy chủ không phản hồi trong 120 giây. Vui lòng thử lại.");
  });

  xhr.send(form);
}

function onUploadResponse(x) {
  if (!x) return;
  const http = x.status;
  const data = xhrBodySafe(x);

  if (http >= 200 && http < 300) {
    if (data && typeof data === "object" && typeof data.job_id === "string") {
      jobId = data.job_id;
      deadline = Date.now() + CLIENT_TIMEOUT_MS;
      showStatus("Đang xếp hàng chờ xử lý…", "Máy chủ tiếp nhận mã nhiệm vụ #" + jobId.slice(0, 8), true, 40);
      pollNow();
      return;
    }
    failStatus("Phản hồi bất thường", "Máy chủ không trả về Job ID hợp lệ.");
  } else if (http === 413) {
    failStatus("Video quá lớn", "Máy chủ từ chối vì tệp vượt " + MAX_MB + " MB hoặc dài hơn " + MAX_SECONDS + " giây.");
  } else if (http === 429) {
    failStatus("Máy chủ đang bận", serverMessage(x) || "Đang có video chờ xử lý. Vui lòng thử lại sau giây lát.");
  } else if (http === 404) {
    failStatus("Không tìm thấy API", "Địa chỉ API không đúng hoặc máy chủ chưa sẵn sàng.");
  } else if (http >= 500) {
    failStatus("Lỗi máy chủ nội bộ", "Máy chủ gặp sự cố khi xử lý video. Vui lòng kiểm tra log backend.");
  } else if (http === 0) {
    failStatus("Mất kết nối máy chủ", "Không thể kết nối đến backend API. Hãy đảm bảo server đã chạy.");
  } else {
    failStatus("Gửi video thất bại", serverMessage(x) || ("Máy chủ trả về mã HTTP " + http));
  }
  onUploadingChange(false);
}

function pollNow() {
  clearTimeout(pollTimer);
  fetch(API_BASE + ENDPOINT_JOB + encodeURIComponent(jobId), { cache: "no-store" })
    .then(async (res) => {
      if (res.status === 404) { onPollStop("expired"); return; }
      if (!res.ok) { onPollStop("server", "Lỗi HTTP " + res.status + " khi truy vấn trạng thái."); return; }
      const data = await res.json().catch(() => null);
      if (!data || typeof data !== "object") { onPollStop("server", "Dữ liệu trả về không hợp lệ."); return; }
      onJobState(data);
    })
    .catch(() => onPollStop("network"));
}

function onJobState(job) {
  if (job.status === STATUS.DONE) {
    stopPolling();
    onUploadingChange(false);
    hideStatus();
    renderResult(job.result);
    return;
  }
  if (job.status === STATUS.ERROR) {
    stopPolling();
    onUploadingChange(false);
    failStatus("Không thể phân tích video", typeof job.error === "string" && job.error ? job.error : "Không tìm thấy khuôn mặt rõ ràng hoặc video không đạt chuẩn.");
    return;
  }
  if (job.status === STATUS.RUNNING) {
    showStatus("Đang phân tích khuôn mặt & kiểm tra giả mạo…", "Trích xuất đặc trưng không gian & tính toán xác suất.", true, 75);
  } else {
    showStatus("Đang trong hàng đợi xử lý…", "Máy chủ đang xử lý video của bạn.", true, 50);
  }
  if (Date.now() > deadline) { onPollStop("timeout"); return; }
  pollTimer = setTimeout(pollNow, POLL_MS);
}

function onPollStop(kind, extra) {
  stopPolling();
  onUploadingChange(false);
  if (kind === "expired") {
    failStatus("Kết quả đã hết hạn", "Dữ liệu chỉ lưu tạm thời " + RESULT_TTL_MIN + " phút trên máy chủ.");
  } else if (kind === "network") {
    failStatus("Mất kết nối", "Không thể kiểm tra trạng thái từ backend API.");
  } else if (kind === "timeout") {
    failStatus("Quá thời gian xử lý", "Đã chờ quá thời gian quy định mà chưa hoàn tất.");
  } else {
    failStatus("Lỗi lấy kết quả", extra || "Có sự cố không mong muốn từ máy chủ.");
  }
}

export function stopPolling() {
  clearTimeout(pollTimer);
  pollTimer = null;
}

export function cancel() {
  const wasRunning = !!xhr || !!pollTimer;
  stopPolling();
  abortUpload();
  onUploadingChange(false);
  jobId = null;
  return wasRunning;
}
