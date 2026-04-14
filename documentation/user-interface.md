# Phân Tích Giao Diện UI – NotebookLM Clone

> **Vai trò:** Senior Frontend Developer  
> **Mục tiêu:** Phân tích bố cục, thành phần, màu sắc trước khi bắt tay vào code clone.

---

## GIAO DIỆN 1 – Trang Chủ / Dashboard (Home Screen)

### 1. Tổng quan bố cục

Giao diện theo dạng **Full-width Single Page Layout** với cấu trúc:

```
┌────────────────────────────────────────────────┐
│                  TOP NAVIGATION BAR             │
├────────────────────────────────────────────────┤
│              FILTER / ACTION TOOLBAR            │
├────────────────────────────────────────────────┤
│                                                 │
│       SECTION 1: SỔ GHI CHÚ NỔI BẬT           │
│         (Horizontal Scroll Card Grid)           │
│                                                 │
├────────────────────────────────────────────────┤
│                                                 │
│       SECTION 2: SỔ GHI CHÚ GẦN ĐÂY           │
│         (Card Grid – 4 columns)                 │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### 2. Các thành phần chi tiết

#### 2.1 Top Navigation Bar
- **Vị trí:** Fixed top, full width
- **Bên trái:** Logo NotebookLM (icon sóng wifi dạng notebook + tên chữ đen đậm)
- **Bên phải:**
  - Icon ⚙ Cài đặt (text + icon)
  - Icon ⠿ (grid/apps)
  - Avatar người dùng (hình tròn, màu tím gradient, chữ "An")
- **Màu nền:** Trắng (`#FFFFFF`) hoặc rất nhạt (`#FAFAFA`)
- **Border bottom:** Nhạt hoặc không có

#### 2.2 Filter / Action Toolbar
- **Vị trí:** Dưới navbar, full width, padding ngang lớn
- **Bên trái – Tab Filter (3 tabs dạng pill/chip):**
  - "Tất cả" – active state, nền trắng, border đậm, rounded-full
  - "Sổ ghi chú của tôi" – inactive, text thường
  - "Sổ ghi chú nổi bật" – inactive, text thường
- **Bên phải – Action Group:**
  - 🔍 Icon tìm kiếm (button tròn)
  - ✓ Icon chọn (button tròn)
  - ⊞ Icon grid view (button tròn, active)
  - ≡ Icon list view (button tròn)
  - Dropdown "Gần đây nhất ▾" – outlined pill button
  - Button "+ Tạo mới" – **filled black**, text trắng, rounded-full, font medium

#### 2.3 Section: Sổ Ghi Chú Nổi Bật
- **Tiêu đề section:** Text lớn, font weight 500–600, `font-size: ~20px`, màu đen/dark gray
- **Layout:** Horizontal scroll row, 5 cards hiển thị cùng lúc (có thể scroll thêm)
- **Card thumbnail (featured):**
  - Kích thước: ~240×180px, `border-radius: 12px`
  - **Nền:** Ảnh cover fill 100% (có overlay tối ở dưới để đọc text)
  - **Góc trái trên:** Icon source (logo tờ báo) + tên nguồn, text nhỏ màu trắng
  - **Dưới cùng (text overlay):**
    - Tiêu đề notebook: font trắng, bold, ~14–16px, 2–3 dòng
    - Metadata: ngày tạo + số nguồn + icon globe – font nhỏ, trắng mờ ~70%
  - **Hover state:** Có thể có scale hoặc overlay tối hơn
- **"Xem tất cả ›":** Text link màu xám/đen, góc phải dưới section

#### 2.4 Section: Sổ Ghi Chú Gần Đây
- **Tiêu đề section:** Tương tự section trên
- **Layout:** CSS Grid 4 cột, gap đều nhau
- **Card loại 1 – "Tạo sổ ghi chú mới":**
  - Nền: Xám nhạt (`#F3F4F6` hoặc `#F0F0F0`), border dashed hoặc không có
  - Trung tâm: Icon "+" hình tròn nhạt hơn, text bên dưới
  - `border-radius: 12px`
- **Card loại 2 – Notebook Card:**
  - Nền: Trắng hoặc off-white (`#FAFAFA`), có border nhạt
  - **Góc phải trên:** Icon "⋮" (more options), màu xám nhạt
  - **Trung tâm trên:** Icon emoji/illustration lớn (~48px) – ví dụ: 🎓, ⚖️, 🔒
  - **Bên dưới icon:**
    - Tên notebook: font medium, ~14–16px, dark gray/black
    - Metadata: ngày + số nguồn – font nhỏ, màu xám nhạt `#9CA3AF`
  - `border-radius: 12px`, có shadow nhẹ

---

### 3. Màu Sắc

| Element | Màu |
|---|---|
| Background tổng | `#FFFFFF` / `#FAFAFA` |
| Text tiêu đề chính | `#111827` (gần đen) |
| Text metadata/phụ | `#6B7280` / `#9CA3AF` |
| Tab active border | `#111827` |
| Button "Tạo mới" | bg: `#111827`, text: `#FFFFFF` |
| Card nền (recent) | `#FFFFFF` với border `#E5E7EB` |
| Card nền (create) | `#F3F4F6` |
| Avatar | Gradient tím `#7C3AED` → `#6D28D9` |
| Featured card overlay | `rgba(0,0,0,0.4–0.6)` gradient bottom |

---

### 4. Typography

- **Font chính:** Sans-serif, có thể là Google Sans hoặc tương tự (clean, rounded)
- **Logo:** Semi-bold/bold
- **Tiêu đề section:** `~20px`, `font-weight: 500–600`
- **Tên notebook (card):** `~14–15px`, `font-weight: 500`
- **Metadata:** `~12px`, `font-weight: 400`, màu nhạt

---

### 5. Spacing & Radius

- Padding tổng ngang: `~48–64px`
- Card gap: `~16px`
- Border-radius card: `~12px`
- Border-radius button pill: `~9999px`

---
---

## GIAO DIỆN 2 – Notebook Detail / Chat View

### 1. Tổng quan bố cục

Giao diện **3-column layout** chiều ngang (sidebar trái | main content | sidebar phải):

```
┌──────────────────────────────────────────────────────────────┐
│                    TOP APP BAR (full width)                   │
├──────────────┬──────────────────────────┬────────────────────┤
│              │                          │                    │
│   LEFT       │     CENTER PANEL         │   RIGHT PANEL      │
│   PANEL      │     (Chat / Cuộc         │   (Studio)         │
│   (Nguồn)    │      trò chuyện)         │                    │
│              │                          │                    │
│              │                          │                    │
│              ├──────────────────────────┤                    │
│              │    CHAT INPUT BOX        │                    │
└──────────────┴──────────────────────────┴────────────────────┘
```

**Tỉ lệ cột (ước tính):** ~25% | ~48% | ~27%

---

### 2. Các thành phần chi tiết

#### 2.1 Top App Bar
- **Vị trí:** Fixed top, full width, nền trắng
- **Bên trái:** Icon NotebookLM (nhỏ hơn homepage) + Tiêu đề notebook dài (font medium, ~16px)
- **Bên phải:**
  - Button "+ Tạo sổ ghi chú" – **filled black**, pill shape
  - Button "< Chia sẻ" – outlined, pill shape
  - Icon ⚙ Cài đặt
  - Icon ⠿ (apps)
  - Avatar người dùng
- **Border bottom:** Có, mỏng, màu `#E5E7EB`

#### 2.2 Left Panel – "Nguồn" (Sources)
- **Header:** Text "Nguồn", icon ☐ (collapse panel) – nằm cạnh nhau
- **Button "+ Thêm nguồn":** Outlined, full width, có icon "+", text centered, border dashed hoặc solid nhạt, `border-radius: 8px`
- **Search bar:** Input "Tìm nguồn mới trên web", icon 🔍 trái, nền nhạt
- **Filter pills:** "🌐 Web ▾" và "↗ Nghiên cứu nhanh ▾" – dạng pill có dropdown, icon + text + mũi tên
- **Checkbox "Chọn tất cả nguồn":** Checkbox + text, màu xám
- **Source item:**
  - Icon PDF (màu đỏ) + tên file text
  - Checkbox bên phải (checked = màu xanh/đen)
- **Nền panel:** Trắng hoặc rất nhạt, border-right mỏng

#### 2.3 Center Panel – "Cuộc trò chuyện" (Chat)
- **Header:** Text "Cuộc trò chuyện" + icon ⚙ filter + icon ⋮ more options
- **Chat content area:**
  - Hiển thị nội dung dạng **rich markdown**: bullet points, bold text, sub-bullets
  - Text size: `~14–15px`, line-height rộng (~1.6)
  - Citation badges: Số nhỏ superscript trong vòng tròn nhỏ màu xám nhạt – ví dụ: `¹`, `²`, `³`
  - **Bold inline:** Từ khóa quan trọng in đậm xen lẫn text bình thường
  - Nền content: Trắng, không có bubble/card riêng (flat text)
- **"Tóm lại:" label:** Không có styling đặc biệt, chỉ in đậm
- **Chat Input Box (bottom):**
  - Vị trí: Sticky bottom trong panel
  - Placeholder text: "Bắt đầu nhập..."
  - Bên phải: Badge "1 nguồn" (chip nhỏ, rounded) + Button gửi (icon mũi tên → tròn, filled black)
  - Border: Có outline, `border-radius: 12–16px`
  - **Footer note:** Dòng chữ nhỏ cảnh báo độ chính xác bên dưới input

#### 2.4 Right Panel – "Studio"
- **Header:** Text "Studio" + icon ☐ (collapse)
- **Feature Grid:** 2 cột × N hàng, mỗi item là một card nhỏ:
  - Icon + Text label + "›" arrow
  - Một số item có badge **"BETA"** màu xanh lá nhạt/teal
  - Items: Tổng quan bằng âm..., Bản trình bày, Tổng quan bằng video, Bản đồ tư duy, Báo cáo, Thẻ ghi nhớ, Bài kiểm tra (BETA), Bản đồ hóa thông tin (BETA), Bảng dữ liệu
  - Card hover: Có highlight nhạt
  - `border-radius: 8–10px`, nền nhạt `#F9FAFB` hoặc có border
- **Empty state prompt (bottom):**
  - Icon ✨ (sparkle)
  - Link text màu xanh dương: "Đầu ra của Studio sẽ được lưu ở đây."
  - Sub-text mô tả nhỏ, màu xám
- **Button "+ Thêm ghi chú":** Outlined/ghost, icon + text, full width hoặc centered, `border-radius: 8px`

---

### 3. Màu Sắc

| Element | Màu |
|---|---|
| Background tổng | `#FFFFFF` |
| Border giữa các panel | `#E5E7EB` |
| Text tiêu đề panel | `#111827` |
| Text nội dung chat | `#1F2937` |
| Text metadata/phụ | `#6B7280` |
| Citation badge bg | `#F3F4F6` border `#D1D5DB` |
| Button filled (CTA) | bg `#111827`, text `#FFFFFF` |
| Button outlined | border `#D1D5DB`, text `#374151` |
| Badge "BETA" | bg `#DCFCE7` hoặc `#D1FAE5`, text `#065F46` |
| Studio card hover | `#F3F4F6` |
| PDF icon | `#EF4444` (đỏ) |
| Link text (Studio) | `#2563EB` |
| Checkbox active | `#111827` hoặc xanh đậm |

---

### 4. Typography

- **App bar title:** `~15–16px`, `font-weight: 500`, bị truncate với `...`
- **Panel header:** `~14–15px`, `font-weight: 600`
- **Chat text:** `~14px`, `font-weight: 400`, line-height `1.6–1.7`
- **Bold in chat:** `font-weight: 600–700`
- **Metadata/small:** `~12px`, `font-weight: 400`
- **Studio card label:** `~13px`, `font-weight: 500`

---

### 5. Spacing & Radius

- Panel padding ngang: `~16–20px`
- Studio card gap: `~8px`
- Chat input border-radius: `~12–16px`
- Studio card border-radius: `~8–10px`
- Button pill border-radius: `~9999px`

---

## TỔNG KẾT SO SÁNH 2 GIAO DIỆN

| Tiêu chí | Giao diện 1 (Home) | Giao diện 2 (Detail) |
|---|---|---|
| Layout | Single column, sectioned | 3-column horizontal |
| Focal point | Card grid/gallery | Chat + Sources |
| Navigation | Tabs + filter toolbar | App bar với actions |
| Primary CTA | "Tạo mới" (pill black) | "Tạo sổ ghi chú" (pill black) |
| Content density | Medium (cards airy) | High (dense text + panels) |
| Color tone | Light, minimal | Light, functional |
| Typography role | Display + labels | Reading + interactive |
| Scrolling | Vertical page + horizontal cards | Vertical per-panel |

---

*Phân tích bởi: Senior Frontend Developer role*  
*Sẵn sàng bắt đầu code sau khi xác nhận.*