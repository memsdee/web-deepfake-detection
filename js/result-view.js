/* ==========================================================================
 HIỂN THỊ KẾT QUẢ: xác suất, gauge 3 vùng, metric, bảng phụ
 ========================================================================== */
import { DEFAULT_LOW, DEFAULT_HIGH } from "./config.js";
import { $, fmtPct } from "./format.js";
import { el } from "./dom.js";
import { failStatus } from "./status-ui.js";

let LOW = DEFAULT_LOW, HIGH = DEFAULT_HIGH;

const readyPlaceholder = $("ready-placeholder");
const resultCard = $("result");
const probValue = $("prob-value"), gauge = $("gauge"), gaugeMarker = $("gauge-marker");
const segLow = $("seg-low"), segMid = $("seg-mid"), segHigh = $("seg-high");
const legendLow = $("legend-low"), legendMid = $("legend-mid"), legendHigh = $("legend-high");
const thresholdNote = $("threshold-note");
const metricsGrid = $("metrics-grid");

const LEVEL_COLOR = { low: "--safe", mid: "--warning", high: "--danger" };

/* Tên 3 vùng trùng nhãn kết luận; vùng thấp chỉ nói "chưa phát hiện", không khẳng định video thật */
export const ZONE_LABEL = {
  low: "Chưa phát hiện dấu hiệu giả mạo",
  mid: "Chưa đủ cơ sở kết luận",
  high: "Có dấu hiệu giả mạo"
};

export function levelOf(p) { return p < LOW ? "low" : (p >= HIGH ? "high" : "mid"); }

export function setThresholds(lo, hi) {
  // Ngưỡng ưu tiên lấy từ response nếu hợp lệ
  if (typeof lo === "number" && typeof hi === "number" && lo >= 0 && hi <= 1 && lo < hi) {
    LOW = lo; HIGH = hi;
  }
  segLow.style.flex = LOW + " 1 0%";
  segMid.style.flex = (HIGH - LOW) + " 1 0%";
  segHigh.style.flex = (1 - HIGH) + " 1 0%";
  legendLow.textContent = "0%";
  legendMid.textContent = "Ngưỡng " + fmtPct(LOW) + " – " + fmtPct(HIGH);
  legendHigh.textContent = "100%";
  gauge.setAttribute("aria-label", "Thang xác suất 0 đến 100%. Chưa đủ cơ sở kết luận từ " + fmtPct(LOW) + " đến " + fmtPct(HIGH) + ".");
  thresholdNote.textContent =
    "Dưới " + fmtPct(LOW) + ": " + ZONE_LABEL.low
    + " · " + fmtPct(LOW) + " – " + fmtPct(HIGH) + ": " + ZONE_LABEL.mid
    + " · Từ " + fmtPct(HIGH) + ": " + ZONE_LABEL.high + ".";
}

/* Chỉ lấy các trường server thực sự trả về, không tự bịa số liệu */
function metricsOf(result) {
  const list = [];
  if (typeof result.processing_time === "number") list.push({ label: "Thời gian xử lý", value: result.processing_time.toFixed(2) + "s" });
  return list;
}

function renderMetrics(list) {
  metricsGrid.replaceChildren();
  for (const m of list) {
    const card = el("div", "metric-card");
    card.append(el("div", "metric-label", m.label), el("div", "metric-number", m.value));
    metricsGrid.append(card);
  }
}

export function renderResult(result) {
  if (!result || typeof result !== "object") {
    failStatus("Kết quả rỗng", "Máy chủ không trả về dữ liệu kết quả.");
    return;
  }
  const p = typeof result.p_fake === "number" && Number.isFinite(result.p_fake) ? result.p_fake : null;
  if (p === null) {
    failStatus("Kết quả không hợp lệ", "Thiếu trường p_fake trong dữ liệu phản hồi.");
    return;
  }

  setThresholds(
    typeof result.low === "number" ? result.low : LOW,
    typeof result.high === "number" ? result.high : HIGH
  );

  const pct = Math.max(0, Math.min(1, p));
  const level = levelOf(pct);

  probValue.textContent = fmtPct(pct);
  probValue.style.color = "var(" + LEVEL_COLOR[level] + ")";

  gaugeMarker.style.left = (pct * 100) + "%";
  gauge.setAttribute("aria-label", "Xác suất có dấu hiệu giả mạo: " + fmtPct(pct));

  const metrics = metricsOf(result);
  renderMetrics(metrics);

  readyPlaceholder.hidden = true;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function hideResult() { resultCard.hidden = true; }

export function revealPlaceholder() { readyPlaceholder.hidden = false; }
