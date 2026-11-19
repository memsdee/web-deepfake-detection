/* ==========================================================================
 DEMO MODE: xem thử giao diện kết quả với dữ liệu mẫu (không gọi API)
 ========================================================================== */
import { $ } from "./format.js";
import { showDemoPreview, video } from "./video-picker.js";
import { renderResult } from "./result-view.js";

const btnDemoData = $("btn-demo-data");

const DEMO_CLIPS_META = [
  { t_start: 0.0, t_end: 0.63, frame_time: 0.3, prob_fake: 0.42, face_box: [0.34, 0.14, 0.32, 0.48] },
  { t_start: 0.63, t_end: 1.25, frame_time: 0.9, prob_fake: 0.78, face_box: [0.35, 0.14, 0.32, 0.48] },
  { t_start: 1.25, t_end: 1.88, frame_time: 1.5, prob_fake: 0.94, face_box: [0.34, 0.13, 0.33, 0.49] },
  { t_start: 1.88, t_end: 2.5, frame_time: 2.2, prob_fake: 0.91, face_box: [0.35, 0.14, 0.32, 0.48] },
  { t_start: 2.5, t_end: 3.13, frame_time: 2.8, prob_fake: 0.61, face_box: [0.36, 0.15, 0.31, 0.47] },
  { t_start: 3.13, t_end: 3.75, frame_time: 3.4, prob_fake: 0.89, face_box: [0.35, 0.14, 0.32, 0.48] },
  { t_start: 3.75, t_end: 4.38, frame_time: 4.0, prob_fake: 0.95, face_box: [0.34, 0.14, 0.32, 0.48] },
  { t_start: 4.38, t_end: 5.0, frame_time: 4.7, prob_fake: 0.86, face_box: [0.35, 0.15, 0.31, 0.47] },
];

function makeDemoImages(probFake, faceBox, frameTime, shift = 0) {
  const S = 224;
  const c1 = document.createElement("canvas");
  c1.width = S; c1.height = S;
  const ctx1 = c1.getContext("2d");

  if (video && video.readyState >= 2 && video.videoWidth > 0) {
    const [nx, ny, nw, nh] = faceBox;
    ctx1.drawImage(
      video,
      nx * video.videoWidth, ny * video.videoHeight, nw * video.videoWidth, nh * video.videoHeight,
      0, 0, S, S
    );
  } else {
    ctx1.fillStyle = "#1e293b";
    ctx1.fillRect(0, 0, S, S);
    ctx1.fillStyle = "#334155";
    ctx1.beginPath();
    ctx1.ellipse(S / 2, S / 2, 68, 86, 0, 0, Math.PI * 2);
    ctx1.fill();
    ctx1.fillStyle = "#0f172a";
    ctx1.beginPath();
    ctx1.arc(82, 92, 9, 0, Math.PI * 2);
    ctx1.arc(142, 92, 9, 0, Math.PI * 2);
    ctx1.fill();
    ctx1.strokeStyle = "#0f172a";
    ctx1.lineWidth = 5;
    ctx1.beginPath();
    ctx1.arc(S / 2, 146, 22, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx1.stroke();
  }
  const face_b64 = c1.toDataURL("image/jpeg", 0.85);

  const c2 = document.createElement("canvas");
  c2.width = S; c2.height = S;
  const ctx2 = c2.getContext("2d");
  // Nền xanh đậm giống COLORMAP_JET khi giá trị saliency thấp
  ctx2.fillStyle = "#00008b";
  ctx2.fillRect(0, 0, S, S);

  const spots = [
    { x: 112 + shift, y: 146, r: 50, a: Math.min(1, 0.9 * probFake) },
    { x: 84, y: 92 + shift * 0.5, r: 34, a: Math.min(1, 0.75 * probFake) },
    { x: 140, y: 92 - shift * 0.5, r: 34, a: Math.min(1, 0.7 * probFake) },
  ];
  for (const s of spots) {
    const g = ctx2.createRadialGradient(s.x, s.y, 2, s.x, s.y, s.r);
    g.addColorStop(0, `rgba(239, 68, 68, ${s.a})`);
    g.addColorStop(0.45, `rgba(234, 179, 8, ${s.a * 0.85})`);
    g.addColorStop(0.8, `rgba(6, 182, 212, ${s.a * 0.5})`);
    g.addColorStop(1, "rgba(0, 0, 139, 0)");
    ctx2.fillStyle = g;
    ctx2.fillRect(0, 0, S, S);
  }
  const heatmap_b64 = c2.toDataURL("image/jpeg", 0.85);
  return { face_b64, heatmap_b64 };
}

export function initDemo() {
  if (!btnDemoData) return;
  btnDemoData.addEventListener("click", () => {
    showDemoPreview();
    const renderDemo = () => {
      const clips = DEMO_CLIPS_META.map((m) => {
        const main = makeDemoImages(m.prob_fake, m.face_box, m.frame_time, 0);
        const frames = [0, 1, 2, 3].map((k) => ({
          t: Number((m.t_start + k * 0.12).toFixed(2)),
          ...makeDemoImages(m.prob_fake, m.face_box, m.frame_time, (k - 1.5) * 4),
        }));
        return { ...m, ...main, frames };
      });
      renderResult({
        p_fake: 0.887,
        low: 0.35,
        high: 0.55,
        processing_time: 4.38,
        clips,
      });
    };
    if (video && video.readyState >= 2) {
      renderDemo();
    } else {
      if (video) video.addEventListener("loadeddata", renderDemo, { once: true });
      renderDemo();
    }
  });
}
