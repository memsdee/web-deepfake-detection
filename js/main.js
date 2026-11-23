/* ==========================================================================
 ĐIỂM VÀO: theme, sự kiện, nối các module
 ========================================================================== */
import { $ } from "./format.js";
import { el } from "./dom.js";
import { initTheme } from "./theme.js";
import { showError } from "./error-banner.js";
import {
  handleFile, clearPreview, getCurrent,
  setResetHook, setAnalyzeEnabled, fileInput, errorBox
} from "./video-picker.js";
import { hideStatus, markCanceled } from "./status-ui.js";
import { hideResult, revealPlaceholder } from "./result-view.js";
import { submit, submitSelected, setUploadingHook, setFacesDetectedHook, cancel } from "./api-client.js";
import { initDemo } from "./demo.js";
import { initHealth } from "./health.js";

const dropFrame = $("drop-frame"), dropIdle = $("drop-idle");
const videoEl = $("video");
const pickerOverlays = $("face-picker-overlays");
const readyPlaceholder = $("ready-placeholder");
const facePickerCard = $("face-picker-card");
const facePickerGrid = $("face-picker-grid");
const facePickerCount = $("face-picker-count");
const btnToggleAllFaces = $("btn-toggle-all-faces");
const btnAnalyzeSelected = $("btn-analyze-selected");
const btnAnalyzeSelectedLabel = $("btn-analyze-selected-label");
const btnCancelPicker = $("btn-cancel-picker");

let pickerSessionId = null;
let pickerFaces = [];
let pickerSelected = new Set();
let pickerHoveredTid = null;

function renderPickerOverlays() {
  if (!pickerOverlays || !videoEl || !pickerFaces.length || facePickerCard.hidden) {
    if (pickerOverlays) pickerOverlays.hidden = true;
    return;
  }
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const cw = videoEl.clientWidth, ch = videoEl.clientHeight;
  if (!vw || !vh || !cw || !ch) {
    pickerOverlays.hidden = true;
    return;
  }
  const scale = Math.min(cw / vw, ch / vh);
  const rw = vw * scale, rh = vh * scale;
  const ox = (cw - rw) / 2, oy = (ch - rh) / 2;

  pickerOverlays.replaceChildren();
  pickerFaces.forEach((f) => {
    if (!Array.isArray(f.face_box) || f.face_box.length !== 4) return;
    const [nx, ny, nw, nh] = f.face_box;
    const topPx = oy + ny * rh;
    const on = pickerSelected.has(f.track_id);
    const hov = pickerHoveredTid === f.track_id;
    const box = el("button", "face-pick-overlay" + (on ? " selected" : "") + (hov ? " hovered" : ""));
    box.type = "button";
    box.style.left = (ox + nx * rw) + "px";
    box.style.top = topPx + "px";
    box.style.width = (nw * rw) + "px";
    box.style.height = (nh * rh) + "px";
    box.title = (f.label || ("Khuôn mặt #" + (f.track_id + 1))) + (on ? " (đang chọn)" : " (bấm để chọn)");
    const tag = el("span", "face-pick-overlay-tag" + (topPx < 24 ? " inside" : ""), "#" + (f.track_id + 1) + (on ? " ✓" : ""));
    box.append(tag);
    box.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      pickerHoveredTid = null;
      if (pickerSelected.has(f.track_id)) pickerSelected.delete(f.track_id);
      else pickerSelected.add(f.track_id);
      syncPickerUi();
    });
    box.addEventListener("mouseenter", () => { pickerHoveredTid = f.track_id; renderPickerOverlays(); });
    box.addEventListener("mouseleave", () => { pickerHoveredTid = null; renderPickerOverlays(); });
    pickerOverlays.append(box);
  });
  pickerOverlays.hidden = false;
}

function syncPickerUi() {
  if (!facePickerGrid) return;
  const n = pickerSelected.size;
  facePickerGrid.querySelectorAll(".face-pick-item").forEach((btn) => {
    const tid = Number(btn.dataset.tid);
    const on = pickerSelected.has(tid);
    btn.classList.toggle("selected", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
  if (btnToggleAllFaces) {
    btnToggleAllFaces.textContent = n === pickerFaces.length ? "Bỏ chọn tất cả" : "Chọn tất cả";
  }
  if (btnAnalyzeSelected) btnAnalyzeSelected.disabled = n === 0;
  if (btnAnalyzeSelectedLabel) {
    btnAnalyzeSelectedLabel.textContent = n > 0
      ? ("Phân tích (" + n + " khuôn mặt)")
      : "Chọn ít nhất 1 khuôn mặt";
  }
  renderPickerOverlays();
}

function hideFacePicker() {
  if (facePickerCard) facePickerCard.hidden = true;
  if (facePickerGrid) facePickerGrid.replaceChildren();
  if (pickerOverlays) {
    pickerOverlays.hidden = true;
    pickerOverlays.replaceChildren();
  }
  if (videoEl) videoEl.controls = true;
  pickerSessionId = null;
  pickerFaces = [];
  pickerSelected.clear();
  pickerHoveredTid = null;
}

function showFacePicker(sessionId, faces) {
  if (!facePickerCard || !facePickerGrid) return;
  pickerSessionId = sessionId;
  pickerFaces = faces;
  pickerSelected = new Set(faces.map((f) => f.track_id));
  pickerHoveredTid = null;
  facePickerGrid.replaceChildren();
  if (facePickerCount) facePickerCount.textContent = faces.length + " khuôn mặt";

  faces.forEach((f) => {
    const btn = el("button", "face-pick-item selected");
    btn.type = "button";
    btn.dataset.tid = String(f.track_id);
    btn.setAttribute("aria-pressed", "true");
    const img = document.createElement("img");
    img.src = f.thumb_b64 || "";
    img.alt = f.label || ("Khuôn mặt #" + (f.track_id + 1));
    btn.append(
      img,
      el("span", "face-pick-check", "✓"),
      el("span", "face-pick-label", f.label || ("Khuôn mặt #" + (f.track_id + 1)))
    );
    btn.addEventListener("click", () => {
      if (pickerSelected.has(f.track_id)) pickerSelected.delete(f.track_id);
      else pickerSelected.add(f.track_id);
      if (videoEl && typeof f.frame_time === "number" && videoEl.src) {
        videoEl.pause();
        if (Math.abs(videoEl.currentTime - f.frame_time) > 0.15) {
          videoEl.addEventListener("seeked", renderPickerOverlays, { once: true });
          videoEl.currentTime = f.frame_time;
        }
      }
      syncPickerUi();
    });
    btn.addEventListener("mouseenter", () => { pickerHoveredTid = f.track_id; renderPickerOverlays(); });
    btn.addEventListener("mouseleave", () => { pickerHoveredTid = null; renderPickerOverlays(); });
    facePickerGrid.append(btn);
  });

  if (readyPlaceholder) readyPlaceholder.hidden = true;
  facePickerCard.hidden = false;

  if (videoEl && videoEl.src) {
    videoEl.pause();
    // Ẩn thanh kéo thời gian mặc định trên video khi đang chọn khuôn mặt (tránh che mặt trên mobile)
    videoEl.controls = false;
    if (typeof faces[0].frame_time === "number" && Math.abs(videoEl.currentTime - faces[0].frame_time) > 0.15) {
      videoEl.addEventListener("seeked", renderPickerOverlays, { once: true });
      videoEl.currentTime = faces[0].frame_time;
    }
  }
  syncPickerUi();
}

window.addEventListener("resize", renderPickerOverlays);
if (videoEl) {
  videoEl.addEventListener("play", () => {
    if (pickerOverlays) pickerOverlays.hidden = true;
  });
}

/* ----- Theme & trạng thái máy chủ ----- */
initTheme();
initHealth();
setFacesDetectedHook(showFacePicker);

/* Khi chọn video mới: dọn kết quả/trạng thái cũ */
setResetHook(() => {
  hideFacePicker();
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
  if (cur && cur.file) {
    hideFacePicker();
    submit(cur.file, errorBox);
  }
});

if (btnToggleAllFaces) {
  btnToggleAllFaces.addEventListener("click", () => {
    if (pickerSelected.size === pickerFaces.length) pickerSelected.clear();
    else pickerFaces.forEach((f) => pickerSelected.add(f.track_id));
    syncPickerUi();
  });
}

if (btnAnalyzeSelected) {
  btnAnalyzeSelected.addEventListener("click", () => {
    if (!pickerSessionId || pickerSelected.size === 0) return;
    const sid = pickerSessionId;
    const ids = Array.from(pickerSelected).sort((a, b) => a - b);
    hideFacePicker();
    submitSelected(sid, ids);
  });
}

if (btnCancelPicker) {
  btnCancelPicker.addEventListener("click", () => {
    hideFacePicker();
    revealPlaceholder();
  });
}

$("btn-reset").addEventListener("click", () => {
  cancel();
  hideFacePicker();
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
