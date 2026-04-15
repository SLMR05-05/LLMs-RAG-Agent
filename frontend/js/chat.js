/* ============================================================
   NotebookLM – Giao Diện 2: Notebook Detail / Chat View
   main.js
   ============================================================ */

'use strict';

// ─── DOM REFERENCES ───────────────────────────────────────
const workspace       = document.querySelector('.workspace');
const panelLeft       = document.getElementById('panelLeft');
const panelRight      = document.getElementById('panelRight');
const collapseLeft    = document.getElementById('collapseLeft');
const collapseRight   = document.getElementById('collapseRight');
const chatMessages    = document.getElementById('chatMessages');
const chatInput       = document.getElementById('chatInput');
const sendBtn         = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const checkAll        = document.getElementById('checkAll');
const sourceList      = document.getElementById('sourceList');


// ─── STATE ────────────────────────────────────────────────
const state = {
  leftCollapsed:  false,
  rightCollapsed: false,
};


// ─── PANEL COLLAPSE ───────────────────────────────────────

/**
 * Cập nhật class trên workspace dựa theo trạng thái collapsed
 */
function updateWorkspaceClass() {
  workspace.classList.toggle('left-collapsed',  state.leftCollapsed  && !state.rightCollapsed);
  workspace.classList.toggle('right-collapsed', state.rightCollapsed && !state.leftCollapsed);
  workspace.classList.toggle('both-collapsed',  state.leftCollapsed  && state.rightCollapsed);
}

collapseLeft.addEventListener('click', () => {
  state.leftCollapsed = !state.leftCollapsed;
  panelLeft.classList.toggle('collapsed', state.leftCollapsed);
  updateWorkspaceClass();

  // Đổi icon
  collapseLeft.querySelector('i').className = state.leftCollapsed
    ? 'bi bi-layout-sidebar-inset'
    : 'bi bi-layout-sidebar';
});

collapseRight.addEventListener('click', () => {
  state.rightCollapsed = !state.rightCollapsed;
  panelRight.classList.toggle('collapsed', state.rightCollapsed);
  updateWorkspaceClass();

  collapseRight.querySelector('i').className = state.rightCollapsed
    ? 'bi bi-layout-sidebar-inset-reverse'
    : 'bi bi-layout-sidebar-reverse';
});


// ─── CHAT: AUTO-RESIZE TEXTAREA ──────────────────────────

/**
 * Tự động điều chỉnh chiều cao textarea theo nội dung
 * @param {HTMLTextAreaElement} el
 */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  sendBtn.disabled = el.value.trim().length === 0;
}

// Khởi tạo trạng thái button gửi
sendBtn.disabled = true;

chatInput.addEventListener('input', () => {
  autoResize(chatInput);
});


// ─── CHAT: ENTER KEY ─────────────────────────────────────

/**
 * Nhấn Enter để gửi, Shift+Enter để xuống dòng
 * @param {KeyboardEvent} e
 */
function handleEnterKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
}


// ─── CHAT: SEND MESSAGE ───────────────────────────────────

/**
 * Dữ liệu phản hồi AI giả lập
 */
const AI_RESPONSES = [
  `Dựa trên tài liệu PRD, hệ thống sử dụng cơ sở dữ liệu quan hệ để lưu trữ thông tin thí sinh và tổ hợp môn. <strong>Schema chính</strong> bao gồm các bảng: <code>thi_sinh</code>, <code>to_hop</code>, <code>diem_mon</code>, và <code>ket_qua_xet_tuyen</code> <sup class="cite-badge">1</sup>.`,

  `Theo tài liệu, quy trình xét tuyển gồm <strong>4 bước chính</strong>: <br/><br/>
   <ul class="chat-list">
     <li>Thu thập điểm thi từ Bộ GD&ĐT qua API</li>
     <li>Tính điểm ưu tiên cho từng tổ hợp</li>
     <li>Xếp hạng theo từng ngành/trường</li>
     <li>Xác nhận kết quả và thông báo thí sinh <sup class="cite-badge">2</sup></li>
   </ul>`,

  `Hệ thống được thiết kế để xử lý tối đa <strong>4.000 thí sinh</strong> đồng thời với thời gian phản hồi dưới <strong>2 giây</strong> cho mỗi truy vấn. Chiến lược tối ưu Skip Optimization giúp giảm tải đáng kể cho cơ sở dữ liệu <sup class="cite-badge">1</sup>, <sup class="cite-badge">3</sup>.`,

  `Về bảo mật, PRD yêu cầu: <br/><br/>
   <ul class="chat-list">
     <li>Mã hoá dữ liệu cá nhân thí sinh theo chuẩn <strong>AES-256</strong></li>
     <li>Xác thực 2 lớp cho tài khoản quản trị viên</li>
     <li>Nhật ký truy cập (audit log) đầy đủ cho mọi thao tác <sup class="cite-badge">2</sup></li>
   </ul>`,
];

let responseIndex = 0;

/**
 * Tạo bubble tin nhắn người dùng
 * @param {string} text
 * @returns {HTMLElement}
 */
function createUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'chat-msg user-msg';
  div.innerHTML = `<div class="user-msg-bubble">${escapeHtml(text)}</div>`;
  return div;
}

/**
 * Tạo bubble tin nhắn AI
 * @param {string} html
 * @returns {HTMLElement}
 */
function createAIMessage(html) {
  const div = document.createElement('div');
  div.className = 'chat-msg ai-msg';
  div.innerHTML = `<div class="chat-msg-content">${html}</div>`;
  // Thêm hiệu ứng fade-in
  div.style.animation = 'fadeIn 0.3s ease both';
  return div;
}

/**
 * Escape HTML để tránh XSS
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Cuộn chat xuống cuối
 */
function scrollToBottom() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

/**
 * Gửi tin nhắn và nhận phản hồi giả lập
 */
function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  // 1. Thêm tin nhắn người dùng
  chatMessages.appendChild(createUserMessage(text));
  scrollToBottom();

  // 2. Reset input
  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;

  // 3. Hiện typing indicator
  typingIndicator.style.display = 'flex';
  scrollToBottom();

  // 4. Giả lập AI phản hồi sau 1.2–2s
  const delay = 1200 + Math.random() * 800;
  setTimeout(() => {
    typingIndicator.style.display = 'none';
    const aiResponse = AI_RESPONSES[responseIndex % AI_RESPONSES.length];
    responseIndex++;
    chatMessages.appendChild(createAIMessage(aiResponse));
    scrollToBottom();
  }, delay);
}


// ─── SOURCES: TOGGLE ALL ──────────────────────────────────

/**
 * Chọn/bỏ chọn tất cả nguồn
 * @param {HTMLInputElement} masterCheckbox
 */
function toggleAllSources(masterCheckbox) {
  const checks = sourceList.querySelectorAll('.source-check');
  checks.forEach(c => { c.checked = masterCheckbox.checked; });
  updateSourceCount();
}

/**
 * Cập nhật badge "N nguồn" trong input box
 */
function updateSourceCount() {
  const total   = sourceList.querySelectorAll('.source-check').length;
  const checked = sourceList.querySelectorAll('.source-check:checked').length;
  const badge   = document.querySelector('.source-count-badge');
  if (badge) badge.textContent = `${checked} nguồn`;

  // Đồng bộ trạng thái checkbox "chọn tất cả"
  if (checkAll) {
    checkAll.checked       = checked > 0;
    checkAll.indeterminate = checked > 0 && checked < total;
  }
}

// Lắng nghe thay đổi trên từng checkbox nguồn
sourceList.addEventListener('change', e => {
  if (e.target.classList.contains('source-check')) {
    updateSourceCount();
  }
});


// ─── STUDIO ACTIONS ───────────────────────────────────────

/**
 * Xử lý click vào Studio feature
 * @param {string} featureName
 */
function studioAction(featureName) {
  showToast(`✨ Đang tạo: ${featureName}...`);
}


// ─── TOAST NOTIFICATION ───────────────────────────────────

/**
 * Hiện toast thông báo ngắn ở góc dưới
 * @param {string} message
 * @param {number} [duration=2800]
 */
function showToast(message, duration = 2800) {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = 'toast-msg';
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
    if (!container.children.length) container.remove();
  }, duration);
}


// ─── KEYBOARD SHORTCUTS ───────────────────────────────────

document.addEventListener('keydown', e => {
  // Ctrl/Cmd + / : focus vào ô chat
  if ((e.ctrlKey || e.metaKey) && e.key === '/') {
    e.preventDefault();
    chatInput.focus();
  }

  // Ctrl/Cmd + B : toggle left panel
  if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
    e.preventDefault();
    collapseLeft.click();
  }
});


// ─── INIT ─────────────────────────────────────────────────

/**
 * Khởi tạo trang
 */
(function init() {
  updateSourceCount();
  chatInput.focus();
  scrollToBottom();
})();