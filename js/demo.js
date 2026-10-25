/* ==========================================================================
 DEMO MODE: xem thử giao diện kết quả với dữ liệu mẫu (không gọi API)
 ========================================================================== */
import { $ } from "./format.js";
import { showDemoPreview } from "./video-picker.js";
import { renderResult } from "./result-view.js";

const btnDemoData = $("btn-demo-data");

const SAMPLE = {
  p_fake: 0.887,
  low: 0.35,
  high: 0.55,
  processing_time: 4.38
};

export function initDemo() {
  if (!btnDemoData) return;
  btnDemoData.addEventListener("click", () => {
    showDemoPreview();
    renderResult(SAMPLE);
  });
}
