/* ==========================================================================
 ĐIỂM VÀO: theme, sự kiện, nối các module
 ========================================================================== */
import { $ } from "./format.js";
import { initTheme } from "./theme.js";
import { showError } from "./error-banner.js";
import {
  handleFile, clearPreview, getCurrent,
  setResetHook, setAnalyzeEnabled, fileInput, errorBox
} from "./video-picker.js";
import { hideStatus, markCanceled } from "./status-ui.js";
import { hideResult, revealPlaceholder } from "./result-view.js";
import { submit, setUploadingHook, cancel } from "./api-client.js";
import { initDemo } from "./demo.js";
import { initHealth } from "./health.js";

const dropFrame = $("drop-frame"), dropIdle = $("drop-idle");

/* ----- Theme & trạng thái máy chủ ----- */
initTheme();
initHealth();

/* Khi chọn video mới: dọn kết quả/trạng thái cũ */
setResetHook(() => {
  hideStatus();
  hideResult();
});

/* Trong lúc upload không cho bấm Phân tích lần nữa */
setUploadingHook((uploading) => {
  setAnalyzeEnabled(!uploading && !!getCurrent());
});

/* ----- Chọn video ----- */
dropIdle.addEventListener("click", () => fileInput.click());
dropIdle.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

["dragenter", "dragover"].forEach((ev) => dropFrame.addEventListener(ev, (e) => {
  e.preventDefault();
  dropFrame.classList.add("dragover");
}));

["dragleave", "dragend"].forEach((ev) => dropFrame.addEventListener(ev, () => dropFrame.classList.remove("dragover")));

dropFrame.addEventListener("drop", (e) => {
  e.preventDefault();
  dropFrame.classList.remove("dragover");
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length > 1) {
    showError(errorBox, "Chỉ chọn một video", "Hệ thống chỉ phân tích 1 video mỗi lần. Vui lòng kéo thả lại 1 tệp tin.");
    return;
  }
  if (files && files[0]) handleFile(files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ----- Nút thao tác ----- */
$("btn-analyze").addEventListener("click", () => {
  const cur = getCurrent();
  if (cur && cur.file) submit(cur.file, errorBox);
});

$("btn-reset").addEventListener("click", () => {
  cancel();
  clearPreview();
  hideStatus();
  hideResult();
  revealPlaceholder();
  fileInput.click();
});

$("btn-cancel").addEventListener("click", () => {
  if (cancel()) markCanceled();
});

/* ----- Demo & dọn khi đóng tab ----- */
initDemo();

window.addEventListener("beforeunload", () => {
  const cur = getCurrent();
  if (cur && cur.url) URL.revokeObjectURL(cur.url);
});
