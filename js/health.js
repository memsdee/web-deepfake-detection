/* ==========================================================================
 TRẠNG THÁI MÁY CHỦ (poll GET /api/health)
 ========================================================================== */
import { API_BASE, HEALTH_ENDPOINT, HEALTH_POLL_MS } from "./config.js";
import { $ } from "./format.js";

const pill = $("server-status");
const label = $("server-status-label");

const TEXT = {
  checking: "Đang kiểm tra…",
  online: "Máy chủ trực tuyến",
  busy: "Máy chủ đang bận",
  offline: "Mất kết nối"
};

let timer = null;

function setState(state, detail) {
  pill.dataset.state = state;
  label.textContent = TEXT[state] || TEXT.offline;
  pill.title = detail || TEXT[state];
  pill.setAttribute("aria-label", "Trạng thái máy chủ: " + (detailsMap(state) || TEXT[state]));
}

function detailsMap(state) {
  if (state === "online") return "Máy chủ trực tuyến";
  if (state === "busy") return "Máy chủ đang bận, vui lòng đợi";
  if (state === "offline") return "Không tới được máy chủ";
  return "Đang kiểm tra máy chủ";
}

function healthUrl() {
  // Contract backend: GET /api/health -> {status, workers, devices, queued, processing}
  return API_BASE + HEALTH_ENDPOINT;
}

async function checkOnce() {
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 5000);
  try {
    const res = await fetch(healthUrl(), { cache: "no-store", signal: ctrl.signal });
    if (!res.ok) { setState("offline"); return; }
    const data = await res.json();
    if (!data || data.status !== "ok") { setState("offline"); return; }

    const devices = Array.isArray(data.devices) ? data.devices.join(", ") : "?";
    const busy = Number(data.queued || 0) >= 3;   // backend MAX_QUEUED = 3
    // Chi tiết chỉ là thông tin công khai, không lộ gì từ client
    const detail = "Thiết bị: " + devices
      + " · Đang chờ: " + (data.queued || 0)
      + " · Đang xử lý: " + (data.processing || 0)
      + (busy ? " · Hàng chờ đã đầy" : "");
    setState(busy ? "busy" : "online", detail);
  } catch (_) {
    setState("offline");
  } finally {
    clearTimeout(timeout);
  }
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    await checkOnce();
    schedule();
  }, HEALTH_POLL_MS);
}

export function initHealth() {
  if (!pill || !label) return;
  setState("checking");
  checkOnce().then(schedule);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") checkOnce();
  });
}
