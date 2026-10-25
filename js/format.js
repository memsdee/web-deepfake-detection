/* ==========================================================================
 FORMAT & KIỂM TRA TỆP
 ========================================================================== */
export const $ = (id) => document.getElementById(id);

export function fmtMB(bytes) { return (bytes / (1024 * 1024)).toFixed(1) + " MB"; }
export function fmtSec(s) { return (Math.round(s * 10) / 10).toString().replace(".", ",") + " giây"; }
export function fmtPct(p) { return (p * 100).toFixed(1).replace(".", ",") + "%"; }

export function isOkType(file) {
  const t = (file.type || "").toLowerCase();
  if (t === "video/mp4" || t === "video/quicktime" || t === "video/webm") return true;
  return /\.(mp4|mov|webm)$/i.test(file.name || "");
}

/* Đọc thời lượng bằng thẻ <video> ẩn; trả null nếu không đọc được (8s). */
export function probeDuration(file) {
  const probe = document.createElement("video");
  probe.preload = "metadata";
  probe.muted = true;
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      probe.removeAttribute("src");
      URL.revokeObjectURL(url);
      resolve(value);
    };
    probe.onloadedmetadata = () => {
      const d = probe.duration;
      done(Number.isFinite(d) && d > 0 ? d : null);
    };
    probe.onerror = () => done(null);
    setTimeout(() => done(null), 8000);
    probe.src = url;
  });
}
