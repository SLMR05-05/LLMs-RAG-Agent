// ── Navigate to chat page (card click)
// Không navigate nếu người dùng đang click vào button ⋮ hoặc dropdown
function goToChat(e) {
    const ignored = e.target.closest('.card-more-wrap');
    if (ignored) return;
    location.href = 'chat.html';
}

// ── Tab filter
function setTab(el) {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
}

// ── Sort dropdown
function toggleSort() {
    document.getElementById('sortDropdown').classList.toggle('show');
}
function setSort(label) {
    const parts = document.getElementById('sortBtn').childNodes;
    parts[0].textContent = label + ' ';
    document.getElementById('sortDropdown').classList.remove('show');
}

// ── Card more dropdown
function toggleCardMenu(e, id) {
    e.stopPropagation();
    const all = document.querySelectorAll('.dropdown-menu-custom');
    all.forEach(m => { if (m.id !== id) m.classList.remove('show'); });
    document.getElementById(id).classList.toggle('show');
}

// ── Close all dropdowns on outside click
document.addEventListener('click', () => {
    document.querySelectorAll('.dropdown-menu-custom').forEach(m => m.classList.remove('show'));
});

// ── Grid / List view toggle
document.querySelectorAll('.tool-icon-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        const group = this.closest('.toolbar-right');
        // Only toggle active for view buttons (grid/list)
        const viewBtns = group.querySelectorAll('.tool-icon-btn:nth-child(4), .tool-icon-btn:nth-child(5)');
        if ([...viewBtns].includes(this)) {
            viewBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        }
    });
});