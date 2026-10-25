/* ==========================================================================
 THẺ TRẠNG THÁI: đang chạy / hoàn tất / thất bại + thanh tiến trình
 ========================================================================== */
import { $ } from "./format.js";
import { showError, hideError } from "./error-banner.js";
import { setScanning } from "./video-picker.js";

const readyPlaceholder = $("ready-placeholder");
const statusCard = $("status-card");
const statusTitle = $("status-title"), statusSub = $("status-sub");
const progress = $("progress"), progressBarFill = $("progress-bar-fill"), statusActions = $("status-actions");
const statusError = $("status-error");

/* Thanh chỉ báo quét chạy trên khung video */
function markScanning(on) { setScanning(on); }

export function showStatus(title, sub, showCancel, progressPct) {
  readyPlaceholder.hidden = true;
  statusCard.hidden = false;
  hideError(statusError);
  statusTitle.textContent = title;
  statusSub.textContent = sub || "";
  progress.hidden = false;
  statusActions.hidden = !showCancel;
  markScanning(true);

  if (typeof progressPct === "number") {
    progressBarFill.style.width = Math.min(100, Math.max(5, progressPct)) + "%";
  }
}

export function hideStatus() {
  statusCard.hidden = true;
  progress.hidden = true;
  statusActions.hidden = true;
  hideError(statusError);
  markScanning(false);
}

export function failStatus(title, msg) {
  progress.hidden = true;
  statusActions.hidden = true;
  statusTitle.textContent = "Thất bại";
  statusSub.textContent = "";
  markScanning(false);
  showError(statusError, title, msg);
}

export function setProgressFromUpload(pct) {
  statusSub.textContent = "Đã gửi " + pct + "% dữ liệu tệp tin...";
  if (progressBarFill) progressBarFill.style.width = Math.max(10, Math.min(90, pct * 0.9)) + "%";
}

export function setSub(text) { statusSub.textContent = text; }

/* Hủy yêu cầu: đưa thẻ trạng thái sang dạng "đã hủy" thay vì biến mất */
export function markCanceled() {
  progress.hidden = true;
  statusActions.hidden = true;
  statusTitle.textContent = "Đã hủy yêu cầu";
  statusSub.textContent = "Bạn có thể phân tích lại bất cứ lúc nào.";
  markScanning(false);
}

export { statusCard, statusError, readyPlaceholder };
