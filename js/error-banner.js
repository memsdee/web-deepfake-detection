/* ==========================================================================
 CẢNH BÁO LỖI (alert-banner) — nội dung luôn đặt bằng textContent
 ========================================================================== */
const ICON_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';

export function showError(container, title, msg) {
  container.replaceChildren();
  const box = document.createElement("div");
  box.className = "alert-banner";

  // Icon là SVG tĩnh trong file, title/msg đến từ server nên chỉ dùng textContent
  const icon = document.createElement("span");
  icon.innerHTML = ICON_SVG;

  const body = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = title;
  const p = document.createElement("p");
  p.textContent = msg;
  body.append(strong, p);

  box.append(icon, body);
  container.append(box);
  container.hidden = false;
}

export function hideError(container) {
  container.hidden = true;
  container.replaceChildren();
}
