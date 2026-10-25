/* ==========================================================================
 CHỌN / KIỂM TRA VIDEO TRONG KHUNG
 ========================================================================== */
import { MAX_MB, MAX_SECONDS } from "./config.js";
import { $, fmtMB, fmtSec, isOkType, probeDuration } from "./format.js";
import { showError, hideError } from "./error-banner.js";
import { renderMetaChips } from "./dom.js";

const dropFrame = $("drop-frame"), dropIdle = $("drop-idle"), dropActive = $("drop-active");
const fileInput = $("file-input"), video = $("video"), videoWrapper = $("video-wrapper");
const previewBadge = $("preview-badge"), errorBox = $("error-box");
const metaChips = $("meta-chips-container");
const btnAnalyze = $("btn-analyze");

let current = null; // { file, url, duration, demo }
let resetHook = () => { }; // main.js gán hàm dọn kết quả/trạng thái cũ

export function getCurrent() { return current; }
export function setResetHook(fn) { resetHook = fn; }
export function setAnalyzeEnabled(enabled) { btnAnalyze.disabled = !enabled; }

function showPreview(file, url, duration, name, demo) {
  if (current && current.url && !current.demo) URL.revokeObjectURL(current.url);
  current = { file: file, url: url, duration: duration, demo: !!demo };
  video.src = url;

  renderMetaChips(metaChips, {
    name: name,
    sizeLabel: fmtMB(file.size),
    durationLabel: duration === null ? "—" : fmtSec(duration)
  });

  // Chuyển sang chế độ xem trước ngay trong khung chọn
  dropIdle.hidden = true;
  dropActive.hidden = false;
  dropFrame.classList.add("has-video");
  if (previewBadge) {
    previewBadge.textContent = "Sẵn sàng";
    previewBadge.classList.add("dz-tag-highlight");
  }
  btnAnalyze.disabled = false;
}

export async function handleFile(file) {
  if (!file) return;
  hideError(errorBox);
  resetHook();

  if (!isOkType(file)) {
    showError(errorBox, "Định dạng không hỗ trợ",
      "Chỉ nhận video định dạng MP4, MOV hoặc WEBM. Vui lòng chọn tệp tin khác.");
    return;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(errorBox, "Video vượt kích thước cho phép",
      "Video hiện tại có dung lượng " + fmtMB(file.size) + ", vượt giới hạn cho phép (" + MAX_MB + " MB). Vui lòng chọn video dung lượng nhỏ hơn.");
    return;
  }

  const duration = (await probeDuration(file)) ?? (await mp4Duration(file));
  // duration === null: trình duyệt không đọc được metadata (ví dụ codec mpeg4).
  // Không chặn; server (ffprobe) là nguồn kiểm tra thời lượng/định dạng.
  if (duration > MAX_SECONDS) {
    showError(errorBox, "Thời lượng video quá dài",
      "Video dài " + fmtSec(duration) + ", vượt quy định " + MAX_SECONDS + " giây. Vui lòng chọn video dưới " + MAX_SECONDS + " giây.");
    return;
  }

  showPreview(file, URL.createObjectURL(file), duration, file.name, false);
}

/* Demo Mode: hiện khung xem trước với video mẫu nếu chưa có video */
export function showDemoPreview() {
  if (current) return;
  showPreview({ size: 1.1 * 1024 * 1024 }, "test_sample.mp4", 5, "demo_interview_deepfake.mp4", true);
}

/* Xoá video hiện tại khỏi khung xem trước, trả về trạng thái chờ */
export function clearPreview() {
  if (current && current.url && !current.demo) URL.revokeObjectURL(current.url);
  current = null;
  video.removeAttribute("src");
  video.load();
  fileInput.value = "";
  metaChips.replaceChildren();
  dropActive.hidden = true;
  dropIdle.hidden = false;
  dropFrame.classList.remove("has-video");
  if (previewBadge) {
    previewBadge.textContent = "Chờ video";
    previewBadge.classList.remove("dz-tag-highlight");
  }
  btnAnalyze.disabled = true;
  setScanning(false);
}

export function setScanning(on) {
  if (videoWrapper) videoWrapper.classList.toggle("scanning", !!on);
}

export { fileInput, errorBox, btnAnalyze, dropIdle, dropFrame, video };


/* Đọc thời lượng từ hộp moov/mvhd của mp4/mov, không cần giải mã (dùng khi <video> không đọc được codec). */
async function mp4Duration(file) {
  const tag = (dv, o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
  try {
    let off = 0;
    while (off + 8 <= file.size) {
      const h = new DataView(await file.slice(off, off + 16).arrayBuffer());
      let size = h.getUint32(0), hdr = 8;
      if (size === 1) { size = Number(h.getBigUint64(8)); hdr = 16; }
      else if (size === 0) size = file.size - off;
      if (size < hdr) return null;
      if (tag(h, 4) === "moov") {
        if (size > 32 * 1024 * 1024) return null;
        const m = new DataView(await file.slice(off + hdr, off + size).arrayBuffer());
        let o = 0;
        while (o + 8 <= m.byteLength) {
          let sz = m.getUint32(o);
          if (sz < 8) return null;
          if (tag(m, o + 4) === "mvhd") {
            const v = m.getUint8(o + 8);
            const timescale = v === 1 ? m.getUint32(o + 28) : m.getUint32(o + 20);
            const dur = v === 1 ? Number(m.getBigUint64(o + 32)) : m.getUint32(o + 24);
            const sec = dur / timescale;
            return Number.isFinite(sec) && sec > 0 ? sec : null;
          }
          o += sz;
        }
        return null;
      }
      off += size;
    }
  } catch (e) { return null; }
  return null;
}
