/* ==========================================================================
 TẠO DOM BẰNG textContent (tránh XSS với dữ liệu từ server)
 ========================================================================== */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function clear(node) {
  if (node) node.replaceChildren();
}

/* Chip metadata video: tên tệp, dung lượng, thời lượng */
const ICONS = {
  file: '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/>',
  size: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>',
  time: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
};

function metaChip(icon, text, truncate) {
  const chip = el("div", "meta-chip");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.innerHTML = ICONS[icon]; // SVG tĩnh
  const span = el("span", null, text);
  if (truncate) {
    span.style.maxWidth = "220px";
    span.style.overflow = "hidden";
    span.style.textOverflow = "ellipsis";
    span.style.whiteSpace = "nowrap";
    span.title = text;
  }
  chip.append(svg, span);
  return chip;
}

export function renderMetaChips(container, file) {
  container.replaceChildren(
    metaChip("file", file.name || "video", true),
    metaChip("size", file.sizeLabel),
    metaChip("time", file.durationLabel)
  );
}
