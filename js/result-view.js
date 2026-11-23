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
const scoreLabel = $("score-label");
const probValue = $("prob-value"), gauge = $("gauge"), gaugeMarker = $("gauge-marker");
const segLow = $("seg-low"), segMid = $("seg-mid"), segHigh = $("seg-high");
const legendLow = $("legend-low"), legendMid = $("legend-mid"), legendHigh = $("legend-high");
const thresholdNote = $("threshold-note");
const metricsGrid = $("metrics-grid");
const multiFaceSection = $("multi-face-section");
const multiFaceNote = $("multi-face-note");
const faceTabs = $("face-tabs");
const evidenceSection = $("evidence-section");
const timelineTrack = $("timeline-track");
const timelineChips = $("timeline-chips");
const heatmapBlock = $("heatmap-block");
const heatmapBadge = $("heatmap-badge");
const faceImg = $("evidence-face-img");
const heatmapBaseImg = $("evidence-heatmap-base");
const heatmapImg = $("evidence-heatmap-img");
const opacitySlider = $("heatmap-opacity");
const opacityVal = $("heatmap-opacity-val");
const temporalStripWrap = $("temporal-strip-wrap");
const temporalGrid = $("temporal-grid");
const videoEl = $("video");
const faceOverlay = $("face-overlay-box");

let lastResult = null;
let activeFaces = [];
let selectedFaceIdx = -1;
let activeClips = [];
let selectedIdx = -1;

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
function metricsOf(result, clipsOverride) {
  const list = [];
  if (typeof result.processing_time === "number") list.push({ label: "Thời gian xử lý", value: result.processing_time.toFixed(2) + "s" });
  if (Array.isArray(result.faces) && result.faces.length > 1) {
    list.push({ label: "Số khuôn mặt", value: result.faces.length + " người" });
  }
  const clips = Array.isArray(clipsOverride) ? clipsOverride : result.clips;
  if (Array.isArray(clips) && clips.length > 0) {
    const suspicious = clips.filter((c) => c.prob_fake >= HIGH).length;
    list.push({ label: "Đoạn nghi vấn", value: suspicious + " / " + clips.length + " đoạn" });
  }
  return list;
}

function updateScoreDisplay(prob, labelSuffix) {
  const pct = Math.max(0, Math.min(1, prob));
  const level = levelOf(pct);
  if (scoreLabel) {
    scoreLabel.textContent = labelSuffix
      ? ("Xác suất có dấu hiệu giả mạo (" + labelSuffix + ")")
      : "Xác suất có dấu hiệu giả mạo";
  }
  probValue.textContent = fmtPct(pct);
  probValue.style.color = "var(" + LEVEL_COLOR[level] + ")";
  gaugeMarker.style.left = (pct * 100) + "%";
  gauge.setAttribute("aria-label", "Xác suất có dấu hiệu giả mạo: " + fmtPct(pct));
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

  lastResult = result;
  updateScoreDisplay(p, "");

  const metrics = metricsOf(result);
  renderMetrics(metrics);
  renderFaces(result.faces, result.clips);

  readyPlaceholder.hidden = true;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateFaceOverlay(clip) {
  if (!faceOverlay || !videoEl || !Array.isArray(clip.face_box) || clip.face_box.length !== 4) {
    if (faceOverlay) faceOverlay.hidden = true;
    return;
  }
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const cw = videoEl.clientWidth, ch = videoEl.clientHeight;
  if (!vw || !vh || !cw || !ch) {
    faceOverlay.hidden = true;
    return;
  }
  const scale = Math.min(cw / vw, ch / vh);
  const rw = vw * scale, rh = vh * scale;
  const ox = (cw - rw) / 2, oy = (ch - rh) / 2;
  const [nx, ny, nw, nh] = clip.face_box;

  faceOverlay.style.left = (ox + nx * rw) + "px";
  faceOverlay.style.top = (oy + ny * rh) + "px";
  faceOverlay.style.width = (nw * rw) + "px";
  faceOverlay.style.height = (nh * rh) + "px";
  faceOverlay.dataset.level = levelOf(clip.prob_fake);
  faceOverlay.dataset.label = "Nghi vấn " + fmtPct(clip.prob_fake);
  faceOverlay.hidden = false;
}

function selectClip(idx, seekVideo) {
  if (idx < 0 || idx >= activeClips.length) return;
  selectedIdx = idx;
  const clip = activeClips[idx];

  timelineTrack.querySelectorAll(".timeline-seg").forEach((n, i) => n.classList.toggle("active", i === idx));
  timelineChips.querySelectorAll(".clip-chip").forEach((n, i) => n.classList.toggle("active", i === idx));

  if (clip.heatmap_b64 || clip.face_b64) {
    const baseSrc = clip.face_b64 || clip.heatmap_b64;
    faceImg.src = baseSrc;
    if (heatmapBaseImg) heatmapBaseImg.src = baseSrc;
    heatmapImg.src = clip.heatmap_b64 || baseSrc;
    heatmapBadge.textContent = clip.t_start.toFixed(1) + "s – " + clip.t_end.toFixed(1) + "s · " + fmtPct(clip.prob_fake);
    heatmapBlock.hidden = false;
  } else {
    heatmapBlock.hidden = true;
  }

  renderTemporalStrip(clip);

  if (seekVideo && videoEl && videoEl.src) {
    const t = typeof clip.frame_time === "number" ? clip.frame_time : clip.t_start;
    seekToFrame(t, clip);
  } else {
    updateFaceOverlay(clip);
  }
}

function seekToFrame(t, clip) {
  if (!videoEl || !videoEl.src) return;
  const applyOverlay = () => updateFaceOverlay(clip);
  videoEl.pause();
  if (Math.abs(videoEl.currentTime - t) < 0.05) {
    applyOverlay();
  } else {
    videoEl.addEventListener("seeked", applyOverlay, { once: true });
    videoEl.currentTime = t;
  }
}

function renderTemporalStrip(clip) {
  if (!temporalStripWrap || !temporalGrid) return;
  temporalGrid.replaceChildren();
  const frames = Array.isArray(clip.frames) ? clip.frames : [];
  if (frames.length === 0) {
    temporalStripWrap.hidden = true;
    return;
  }
  frames.forEach((f, idx) => {
    const btn = el("button", "temporal-card");
    btn.type = "button";
    btn.title = "Khung #" + (idx + 1) + " (" + f.t.toFixed(2) + "s)";
    const wrap = el("div", "heatmap-img-wrap");
    const base = document.createElement("img");
    base.src = f.face_b64 || f.heatmap_b64;
    base.alt = "Khung hình " + (idx + 1);
    const over = document.createElement("img");
    over.className = "heatmap-overlay-img";
    over.src = f.heatmap_b64 || f.face_b64;
    over.alt = "";
    wrap.append(base, over);
    btn.append(wrap, el("span", "temporal-time", f.t.toFixed(2) + "s"));
    btn.addEventListener("click", () => seekToFrame(f.t, clip));
    temporalGrid.append(btn);
  });
  temporalStripWrap.hidden = false;
}

if (opacitySlider && heatmapBlock) {
  const syncOpacity = () => {
    const v = Number(opacitySlider.value);
    heatmapBlock.style.setProperty("--hm-opacity", String(v / 100));
    if (opacityVal) opacityVal.textContent = v + "%";
  };
  opacitySlider.addEventListener("input", syncOpacity);
  syncOpacity();
}

function renderEvidence(clips) {
  if (!evidenceSection) return;
  activeClips = Array.isArray(clips) ? clips : [];
  selectedIdx = -1;
  timelineTrack.replaceChildren();
  timelineChips.replaceChildren();
  if (faceOverlay) faceOverlay.hidden = true;

  if (activeClips.length === 0) {
    evidenceSection.hidden = true;
    heatmapBlock.hidden = true;
    return;
  }

  let bestIdx = 0;
  activeClips.forEach((clip, i) => {
    if (clip.prob_fake > activeClips[bestIdx].prob_fake) bestIdx = i;
    const lvl = levelOf(clip.prob_fake);

    const seg = el("button", "timeline-seg");
    seg.type = "button";
    seg.dataset.level = lvl;
    seg.title = clip.t_start.toFixed(1) + "s – " + clip.t_end.toFixed(1) + "s: " + fmtPct(clip.prob_fake);
    seg.setAttribute("aria-label", "Đoạn " + (i + 1) + ": " + seg.title);
    seg.addEventListener("click", () => selectClip(i, true));
    timelineTrack.append(seg);

    const chip = el("button", "clip-chip");
    chip.type = "button";
    chip.dataset.level = lvl;
    chip.append(
      el("span", "clip-chip-time", clip.t_start.toFixed(1) + "s – " + clip.t_end.toFixed(1) + "s"),
      el("span", "clip-chip-prob", fmtPct(clip.prob_fake))
    );
    chip.addEventListener("click", () => selectClip(i, true));
    timelineChips.append(chip);
  });

  evidenceSection.hidden = false;
  selectClip(bestIdx, false);
}

function selectFace(idx) {
  if (idx < 0 || idx >= activeFaces.length) return;
  selectedFaceIdx = idx;
  const face = activeFaces[idx];
  if (faceTabs) {
    faceTabs.querySelectorAll(".face-tab-card").forEach((btn, i) => {
      const on = i === idx;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    });
  }
  if (typeof face.prob_fake === "number") {
    const lbl = activeFaces.length > 1 ? (face.label || ("Khuôn mặt #" + (idx + 1))) : "";
    updateScoreDisplay(face.prob_fake, lbl);
  }
  if (lastResult) {
    renderMetrics(metricsOf(lastResult, face.clips));
  }
  renderEvidence(face.clips);
}

function renderFaces(faces, fallbackClips) {
  activeFaces = Array.isArray(faces) ? faces : [];
  selectedFaceIdx = -1;
  if (faceTabs) faceTabs.replaceChildren();

  if (!multiFaceSection || !faceTabs || activeFaces.length <= 1) {
    if (multiFaceSection) multiFaceSection.hidden = true;
    const clips = activeFaces.length === 1 ? activeFaces[0].clips : fallbackClips;
    renderEvidence(clips);
    return;
  }

  if (multiFaceNote) {
    multiFaceNote.textContent =
      "Phát hiện " + activeFaces.length + " khuôn mặt trong video. Chọn từng khuôn mặt bên dưới để xem chi tiết nghi vấn riêng.";
  }

  activeFaces.forEach((face, i) => {
    const btn = el("button", "face-tab-card");
    btn.type = "button";
    btn.role = "tab";
    btn.dataset.level = levelOf(face.prob_fake);
    if (face.thumb_b64) {
      const img = document.createElement("img");
      img.className = "face-tab-thumb";
      img.src = face.thumb_b64;
      img.alt = face.label || ("Khuôn mặt #" + (i + 1));
      btn.append(img);
    }
    const meta = el("span", "face-tab-meta");
    meta.append(
      el("span", "face-tab-name", face.label || ("Khuôn mặt #" + (i + 1))),
      el("span", "face-tab-prob", fmtPct(face.prob_fake))
    );
    btn.append(meta);
    btn.addEventListener("click", () => selectFace(i));
    faceTabs.append(btn);
  });

  multiFaceSection.hidden = false;
  selectFace(0);
}

if (videoEl) {
  videoEl.addEventListener("play", () => {
    if (faceOverlay) faceOverlay.hidden = true;
  });
}

export function hideResult() {
  resultCard.hidden = true;
  if (multiFaceSection) multiFaceSection.hidden = true;
  if (evidenceSection) evidenceSection.hidden = true;
  if (heatmapBlock) heatmapBlock.hidden = true;
  if (faceOverlay) faceOverlay.hidden = true;
  lastResult = null;
  activeFaces = [];
  selectedFaceIdx = -1;
  activeClips = [];
  selectedIdx = -1;
}

export function revealPlaceholder() { readyPlaceholder.hidden = false; }
