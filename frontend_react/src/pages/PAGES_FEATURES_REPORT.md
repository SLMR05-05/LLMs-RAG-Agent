# Báo Cáo Tính Năng Của Các Pages - Frontend React LLM-RAG-Agent

## Mục Lục
1. [Tổng Quan Kiến Trúc](#tổng-quan-kiến-trúc)
2. [HomePage](#homepage)
3. [NotebookPage](#notebookpage)
4. [Luồng Dữ Liệu Toàn Cục](#luồng-dữ-liệu-toàn-cục)
5. [API Integration](#api-integration)

---

## Tổng Quan Kiến Trúc

### Cấu Trúc Ứng Dụng
```
App (Entry Point)
├── HomePage: Quản lý danh sách Notebook
└── NotebookPage: Workspace 3 cột (Sources, Chat, Studio)
    ├── LeftSidebar: Quản lý tài liệu nguồn
    ├── ChatArea: Chat với AI
    └── RightStudio: Công cụ hỗ trợ
```

### Lưu Trữ State (Zustand Store)
**Vị trí:** `src/store/useAppStore.ts`

- **Notebooks**: Danh sách sổ ghi chú (Notebook)
- **Active Notebook**: ID của notebook hiện tại
- **Sources**: Danh sách tài liệu nguồn của notebook
- **Chat Messages**: Lịch sử chat của notebook
- **Chat Session Map**: Ánh xạ session_id theo notebook_id
- **Upload Queue**: Hàng đợi upload tài liệu
- **UI State**: Trạng thái bảng điều khiển (collapsed/expanded)

### API Service
**Vị trí:** `src/services/api.ts`

Tất cả các HTTP request đều được thực hiện thông qua `apiService` object, với base URL từ environment variable `VITE_API_URL` (mặc định: `http://localhost:8000`).

---

## HomePage

### 📋 Tính Năng

1. **Hiển Thị Danh Sách Notebook**
   - Lấy tất cả notebook từ backend
   - Hiển thị dạng card grid (responsive: 1 cột mobile, 2 cột tablet, 4 cột desktop)
   - Hiển thị skeleton loading khi đang tải
   - Hiển thị trống khi chưa có notebook nào

2. **Tạo Notebook Mới**
   - Mở modal nhập tên sổ ghi chú
   - Gọi API tạo notebook
   - Tự động chuyển hướng sang notebook vừa tạo
   - Cập nhật danh sách notebook trên UI

3. **Đổi Tên Notebook**
   - Chọn notebook → bấm menu → Đổi tên
   - Mở modal nhập tên mới
   - Gọi API cập nhật
   - Cập nhật lại danh sách trên UI

4. **Xóa Notebook**
   - Chọn notebook → bấm menu → Xóa
   - Hiện confirm modal xác nhận
   - Gọi API xóa
   - Cập nhật danh sách
   - Nếu xóa notebook đang active → quay về HomePage

5. **Mở Notebook**
   - Bấm vào card notebook
   - Cập nhật `activeNotebookId`
   - Chuyển hướng đến route `/notebook/:id`

### 🔄 Cơ Chế Hoạt Động

#### Initialization (useEffect)
```
HomePage mount
  ↓
useEffect dependencies: [fetchNotebooks]
  ↓
Gọi fetchNotebooks() từ store
  ↓
store gọi apiService.getNotebooks()
  ↓
Cập nhật state.notebooks
  ↓
Render danh sách notebook
```

#### Create Notebook Flow
```
User bấm "Tạo mới"
  ↓
Mở CreateNotebookModal
  ↓
User nhập tên → submit
  ↓
handleCreateNotebook() {
  - setIsSubmitting(true)
  - setError(null)
  - notebook = apiService.createNotebook(name)
  - addNotebook(notebook) [update store]
  - setActiveNotebook(notebook.id) [set active]
  - navigate(`/notebook/${notebook.id}`) [chuyển trang]
}
  ↓
Hiển thị notebook mới trong danh sách
```

#### Delete Notebook Flow
```
User bấm xóa
  ↓
Mở DeleteNotebookModal (confirm)
  ↓
handleDeleteNotebook() {
  - setIsSubmitting(true)
  - apiService.deleteNotebook(id)
  - deleteNotebook(id) [update store]
  - if (activeNotebookId === id) navigate('/') [quay về]
}
  ↓
Cập nhật danh sách, xóa khỏi UI
```

### 🌐 API Calls

| Tính Năng | Phương Thức | Endpoint | Payload | Response |
|-----------|------------|----------|---------|----------|
| Lấy danh sách | GET | `/notebooks` | - | `ApiNotebook[]` |
| Tạo mới | POST | `/notebooks` | `{notebook_name: string}` | `ApiNotebook` |
| Đổi tên | PATCH | `/notebooks/:id` | `{notebook_name: string}` | `ApiNotebook` |
| Xóa | DELETE | `/notebooks/:id` | - | void |
| Chi tiết | GET | `/notebooks/:id` | - | `ApiNotebook` |

### 📊 Data Flow

```
HomePage Component
├── State (Zustand):
│   ├── notebooks: Notebook[]
│   ├── activeNotebookId: string | null
│   ├── isFetchingNotebooks: boolean
│   └── error: string | null
│
├── Thao tác:
│   ├── Create → addNotebook() → UI update
│   ├── Rename → renameNotebook() → UI update
│   ├── Delete → deleteNotebook() → UI update
│   └── Select → setActiveNotebook() → navigate
│
└── Render:
    └── Danh sách card NotebookCard
        └── Mỗi card có menu (edit, delete, view)
```

---

## NotebookPage

### 📋 Tính Năng

NotebookPage là workspace chính với **3 cột layout**:

#### **1. LEFT SIDEBAR - Quản Lý Tài Liệu Nguồn (LeftSidebar.tsx)**

**Tính Năng:**
- **Hiển thị danh sách sources**
  - Danh sách các tài liệu/trang web đã upload
  - Checkbox chọn/bỏ chọn nguồn tham gia chat
  - Hiển thị loại nguồn (PDF, DOCX, TXT, hình ảnh, web)
  - Menu context cho từng source

- **Upload tài liệu**
  - Drag-drop file hoặc bấm để chọn
  - Hỗ trợ: PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, GIF, TIFF
  - Hiển thị hàng đợi upload với progress bar
  - Polling job status cho đến khi hoàn tất

- **Thêm nguồn từ web**
  - Input URL để lấy nội dung từ web
  - Gọi API xử lý async, nhận job_id
  - Polling job status cho đến khi xong

- **Quản lý source**
  - Xem chi tiết (view detail modal)
  - Đổi tên tài liệu
  - Xóa tài liệu
  - Chọn/bỏ chọn tất cả

- **Tìm kiếm**
  - Filter sources theo tên

#### **2. CENTER - Chat Area (ChatArea.tsx)**

**Tính Năng:**
- **Hiển thị lịch sử chat**
  - Danh sách tin nhắn (user/assistant)
  - Render markdown với formatting đẹp
  - Hiển thị citations (tham chiếu tài liệu)

- **Gửi câu hỏi**
  - Textarea auto-expand khi gõ
  - Tự động detect ngôn ngữ (tiếng Việt/Anh)
  - Submit bằng Enter hoặc nút Send
  - Chỉ gửi được khi có ít nhất 1 source active

- **Chat Settings**
  - Chiều dài response: short/medium/long
  - Roleplay: cấu hình nhân vật AI
  - Mode: normal/study_guide/critical_thinking

- **Clear Chat History**
  - Xóa toàn bộ lịch sử chat
  - Confirm trước xóa

- **Citations & References**
  - Badge hiển thị số citation [1] [2]...
  - Bấp badge để xem snippet + source name

#### **3. RIGHT SIDEBAR - Studio (RightStudio.tsx)**

**Tính Năng:**
- **4 mode khác nhau:**

  1. **Dàn Bài (Outline)**
     - Hiển thị outline/cấu trúc chủ đề
     - Giúp người dùng hiểu cấu trúc tài liệu

  2. **Câu Hỏi Phổ Biến (FAQ)**
     - Hiển thị frequently asked questions
     - Giúp người dùng nhanh chóng tìm câu trả lời

  3. **Tóm Tắt (Briefing)**
     - Tóm tắt nội dung chính
     - Overview nhanh về tài liệu

  4. **Tùy Chỉnh (Custom)**
     - Tạo content tùy chỉnh dựa trên câu hỏi

### 🔄 Cơ Chế Hoạt Động

#### Page Load & Initialization

```
NotebookPage mounted with route param :id
  ↓
useEffect runs (dependencies: [id, ...])
  ↓
mounted = true
closeSourceDetail() [reset detail view]
resetChatHistory() [reset chat]
setActiveNotebook(id) [set active]
  ↓
Check if notebook exists in store
  ├─ If NO: fetch notebook details from API
  └─ If YES: use existing notebook
  ↓
Promise.all([
  apiService.getNotebookSources(id),
  apiService.getNotebookChatMessages(id)
])
  ↓
Update store:
├─ setSources(sources)
├─ setChatMessages(chat.messages)
├─ setChatSessionForNotebook(id, chat.sessionId)
  ↓
Render MainLayout (3 columns)
```

#### Chat Message Flow

```
User types question in textarea
  ↓
Click Send / Press Enter
  ↓
handleSendMessage() {
  // Kiểm tra điều kiện
  ├─ activeNotebookId must exist
  └─ hasActiveSources() must be true
  
  // Chuẩn bị payload
  ├─ question: string
  ├─ session_id: from store chatSessionByNotebook[id]
  ├─ answer_language: detectQuestionLanguage(question)
  ├─ source_names: selected sources names
  ├─ model_name: "qwen2.5:1.5b"
  ├─ top_k: 4
  └─ chatSettings: from store
  
  // Add user message to UI
  addChatMessage({
    role: 'user',
    content: question,
    timestamp: now()
  })
  
  // Call API
  response = apiService.chatNotebook(id, payload)
  
  // Add assistant message to UI
  addChatMessage({
    role: 'assistant',
    content: response.answer,
    citations: response.citations,
    timestamp: now()
  })
  
  // Update session
  setChatSessionForNotebook(id, response.session_id)
}
```

#### Upload & Index Flow

```
User drag-drop / select files
  ↓
handleFiles(files[])
  ├─ Filter by accepted extensions
  ├─ setError(null)
  └─ addUploadJobs(fileArray)
  ↓
Store adds to uploadQueue with status='pending'
  ↓
UI shows queue items with progress
  ↓
Batch upload processing:
  - Get pending jobs
  - For each job:
    ├─ Call apiService.uploadNotebookFile()
    ├─ Update progress: updateJobProgress()
    └─ When done: updateJobStatus('success'/'error')
  ↓
After successful upload:
  - Re-fetch sources: getNotebookSources()
  - Update sources in store
  ├─ Show success notification
  └─ Remove from queue
```

#### Web Source Addition Flow

```
User enters URL in input
  ↓
Click "Add Web Source"
  ↓
handleAddWebSource() {
  ├─ Validate URL format
  ├─ addWebLink(url) [add to queue]
  ├─ Call apiService.addWebSource(notebookId, url)
  ├─ Receive job_id
  └─ Update queue with job_id
  ↓
Start polling getJob(job_id)
  ├─ Poll every 2-3 seconds
  ├─ When status='completed': success
  ├─ When status='failed': show error
  └─ Stop polling
  ↓
After success:
  - Re-fetch sources
  - Update store
  - Clear input
}
```

#### Source Detail View Flow

```
User clicks on source in list
  ↓
handleViewSourceDetail(source)
  ├─ openSourceDetail(source)
  ├─ Call apiService.getSourceDetail(notebookId, sourceId)
  ├─ Receive full source with chunks data
  └─ setSelectedSourceDetail(fullSource)
  ↓
SourceDetailView renders:
  ├─ Source metadata (title, type, size, pages)
  ├─ Source preview (PDF viewer / markdown)
  ├─ List of chunks (text segments)
  └─ Search within source
```

### 🌐 API Calls

| Tính Năng | Method | Endpoint | Payload | Response |
|-----------|--------|----------|---------|----------|
| Lấy details | GET | `/notebooks/:id` | - | `ApiNotebook` |
| Lấy sources | GET | `/notebooks/:id/sources` | - | `ApiSourcesResponse` |
| Lấy source detail | GET | `/notebooks/:id/sources/:docId` | - | `ApiSourceDetailResponse` |
| Chat | POST | `/notebooks/:id/chat` | `ChatPayload` | `ChatApiResponse` |
| Lấy chat messages | GET | `/notebooks/:id/chat/messages` | - | `ChatMessagesApiResponse` |
| Clear chat | DELETE | `/notebooks/:id/chat` | - | `ChatClearApiResponse` |
| Upload file | POST | `/notebooks/:id/sources/upload` | FormData | `UploadApiResponse` |
| Add web source | POST | `/notebooks/:id/sources/web` | `{url: string}` | `WebSourceApiResponse` |
| Check job status | GET | `/jobs/:jobId` | - | `JobApiResponse` |
| Rename source | PATCH | `/notebooks/:id/sources/:docId` | `{source_name: string}` | `SourceManageApiResponse` |
| Delete source | DELETE | `/notebooks/:id/sources/:docId` | - | `SourceManageApiResponse` |
| Get job status | GET | `/jobs/:jobId` | - | `JobApiResponse` |

### 📊 Data Flow DetailedNotebookPage

```
NotebookPage Component
│
├── Route Params: :id (notebook_id)
│
├── Store Dependencies:
│   ├── activeNotebookId: string | null
│   ├── sources: SourceDocument[]
│   ├── chatMessages: ChatMessage[]
│   ├── chatSessionByNotebook: Record<string, string>
│   ├── uploadQueue: UploadJob[]
│   ├── chatSettings: ChatSettings
│   └── UI state (panels collapsed)
│
├── Child Components:
│   │
│   ├── LeftSidebar (Source Management)
│   │   ├── API Call: getNotebookSources()
│   │   ├── Actions:
│   │   │   ├── uploadFile() → POST /upload
│   │   │   ├── addWebSource() → POST /web + polling
│   │   │   ├── renameSource() → PATCH /sources/:id
│   │   │   ├── deleteSource() → DELETE /sources/:id
│   │   │   └── toggleSelection() [local state]
│   │   └── Polling: getJob() for upload/web jobs
│   │
│   ├── ChatArea (Chat Interface)
│   │   ├── API Call: getNotebookChatMessages()
│   │   ├── Actions:
│   │   │   ├── sendMessage() → POST /chat
│   │   │   ├── clearHistory() → DELETE /chat
│   │   │   └── detectLanguage() [local]
│   │   └── Features:
│   │       ├── Auto-detect language (VI/EN)
│   │       ├── Pass selected sources in payload
│   │       ├── Store session_id by notebook
│   │       └── Render citations
│   │
│   └── RightStudio (Utilities)
│       ├── Modes: Outline, FAQ, Briefing, Custom
│       ├── Display content based on mode
│       └── No API calls (static content)
│
└── State Management Flow:
    ├── Load: getNotebookSources() → setSources()
    ├── Load: getNotebookChatMessages() → setChatMessages()
    ├── Chat: chatNotebook() → addChatMessage() + setChatSession()
    ├── Upload: uploadFile() → updateJobProgress() → setSources()
    └── Source: getSourceDetail() → setSelectedSourceDetail()
```

---

## Luồng Dữ Liệu Toàn Cục

### End-to-End Chat Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INITIATES CHAT                                      │
├─────────────────────────────────────────────────────────────┤
│ User types question in ChatArea textarea                    │
│ Selects active sources from LeftSidebar                     │
│ Chooses chat settings (length, roleplay, mode)             │
│ Presses Enter or clicks Send button                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FRONTEND PROCESSING                                      │
├─────────────────────────────────────────────────────────────┤
│ - Validate activeNotebookId & hasActiveSources()           │
│ - Detect question language using language.ts utility       │
│ - Build ChatPayload:                                        │
│   {                                                         │
│     question: string,                                       │
│     session_id: chatSessionByNotebook[id],                 │
│     answer_language: 'vi' | 'en',                          │
│     source_names: selected sources,                        │
│     model_name: 'qwen2.5:1.5b',                            │
│     top_k: 4,                                              │
│     chatSettings: ChatSettings object                      │
│   }                                                         │
│ - Add user message to UI immediately                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. API REQUEST                                              │
├─────────────────────────────────────────────────────────────┤
│ POST /notebooks/:id/chat                                    │
│ Headers: { 'Content-Type': 'application/json' }            │
│ Body: ChatPayload (JSON)                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. BACKEND PROCESSING                                       │
├─────────────────────────────────────────────────────────────┤
│ - Retrieve session from session_id                         │
│ - Fetch source documents by name                           │
│ - Generate embeddings for question                         │
│ - Vector search (top_k=4) against source chunks            │
│ - Pass to LLM with:                                        │
│   * Retrieved contexts                                     │
│   * Language hint                                          │
│   * Chat settings (roleplay, mode, length)                │
│ - Generate answer                                          │
│ - Extract citations from answer                           │
│ - Return response with session_id                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FRONTEND RESPONSE HANDLING                               │
├─────────────────────────────────────────────────────────────┤
│ Response: ChatApiResponse                                  │
│ {                                                           │
│   notebook_id: string,                                      │
│   session_id: string,                                       │
│   rewritten_question: string,                              │
│   answer: string (with [1], [2]... citation markers),      │
│   citations: [                                             │
│     {                                                       │
│       source_name: string,                                 │
│       page?: number,                                       │
│       snippet: string                                      │
│     }                                                       │
│   ]                                                         │
│ }                                                           │
│                                                             │
│ Actions:                                                    │
│ - Add assistant message to ChatMessage[]                   │
│ - Update session: setChatSessionForNotebook(id, resp.sid) │
│ - Render answer with markdown + citation badges          │
│ - Store citations in chatMessage.citationDetails          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. UI RENDERING                                             │
├─────────────────────────────────────────────────────────────┤
│ ChatArea displays:                                          │
│ - User message (right-aligned, gray bg)                   │
│ - Assistant message (left-aligned, white bg)              │
│ - Markdown formatted text                                  │
│ - Citation badges [1] [2] [3]...                          │
│ - Hover citation to see:                                  │
│   * Source name                                            │
│   * Page number (if applicable)                           │
│   * Quote snippet                                         │
│ - Auto-scroll to latest message                           │
└─────────────────────────────────────────────────────────────┘
```

### Document Upload & Indexing Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. FILE SELECTION                                           │
├─────────────────────────────────────────────────────────────┤
│ User:                                                       │
│ - Drag-drop files OR                                       │
│ - Click and select from file picker                        │
│ - Files filtered by extension                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FRONTEND QUEUE MANAGEMENT                               │
├─────────────────────────────────────────────────────────────┤
│ - Add files to uploadQueue                                 │
│ - Mark status as 'pending'                                 │
│ - Display queue in LeftSidebar                             │
│ - Group files for batch upload                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. BATCH UPLOAD                                             │
├─────────────────────────────────────────────────────────────┤
│ For each file in batch:                                    │
│ POST /notebooks/:id/sources/upload                         │
│ Body: FormData {                                           │
│   files: File[]                                            │
│ }                                                           │
│                                                             │
│ Response: UploadApiResponse                                │
│ {                                                           │
│   uploaded_count: number,                                  │
│   skipped_count: number,                                   │
│   rejected_count: number,                                  │
│   job_id: string,                                          │
│   status: 'processing' | 'completed'                       │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. POLLING & STATUS TRACKING                                │
├─────────────────────────────────────────────────────────────┤
│ - Store job_id from response                               │
│ - Update queue item: status = 'uploading'                  │
│ - Poll GET /jobs/:jobId every 2 seconds                    │
│ - Response: JobApiResponse                                 │
│   {                                                         │
│     status: 'pending'|'running'|'completed'|'failed',     │
│     result?: {...},                                        │
│     error?: string                                         │
│   }                                                         │
│ - Update progress bar in UI                                │
│ - Stop polling when status = completed/failed              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. BACKEND PROCESSING                                       │
├─────────────────────────────────────────────────────────────┤
│ - Parse uploaded file (PDF, DOCX, TXT, IMG)                │
│ - Extract text content                                     │
│ - Split into chunks                                        │
│ - Generate embeddings for each chunk                       │
│ - Store in vector DB + metadata DB                         │
│ - Build search index                                       │
│ - Update job status → 'completed'                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. FRONTEND COMPLETION                                      │
├─────────────────────────────────────────────────────────────┤
│ When job completed:                                         │
│ - Re-fetch sources: getNotebookSources()                   │
│ - Update store: setSources(newSources)                     │
│ - Remove from queue: removeUploadJob(id)                   │
│ - Show success notification                                │
│ - New sources available in chat                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Persistence & Session Management

```
Chat Session Persistence:
├─ First message: backend creates session, returns session_id
├─ Stored in store: chatSessionByNotebook[notebookId] = sessionId
├─ Subsequent messages: include session_id in request
└─ Backend maintains conversation context using session

Source Data Persistence:
├─ Loaded on NotebookPage init
├─ Stored in store: sources[]
├─ Chunks stored separately in SourceDocument.chunks[]
├─ User selections (checkboxes) are local state
└─ Selected sources affect chat queries

Chat History Persistence:
├─ Loaded on NotebookPage init: getNotebookChatMessages()
├─ Stored in store: chatMessages[]
├─ Messages include timestamp for sorting
├─ Can be cleared: DELETE /notebooks/:id/chat
└─ History scoped to notebook_id
```

---

## API Integration

### Authentication
- **Current**: No authentication (development mode)
- **Base URL**: From `VITE_API_URL` env variable (default: `http://localhost:8000`)
- **Content-Type**: `application/json` for JSON, `multipart/form-data` for uploads

### Language Handling

```typescript
// src/utils/language.ts
detectQuestionLanguage(question: string): 'vi' | 'en'

Function behavior:
├─ Detects Vietnamese characters (ă, â, ê, ô, ơ, ư, etc.)
├─ Detects English text patterns
├─ Returns 'vi' for Vietnamese, 'en' for English
└─ Default fallback: 'en'

Usage:
├─ Frontend sends language hint to backend
├─ Backend uses hint to:
│   ├─ Format response in appropriate language
│   ├─ Extract relevant citations
│   └─ Adjust tone/formality
└─ Has its own fallback detection
```

### Error Handling

```javascript
// Request error parsing:
if (!response.ok) {
  try {
    const payload = await response.json();
    // Try detail field
    detailMessage = payload?.detail || 
                    payload?.detail?.message || 
                    JSON.stringify(payload);
  } catch {
    detailMessage = await response.text();
  }
  throw new Error(detailMessage);
}

Error display:
├─ Store.setError(message)
├─ UI shows error notification
├─ User can dismiss error
└─ Allows retry
```

### Request/Response Flow

```
Frontend
  ↓
apiService.request<T>(path, init?)
  ├─ Build full URL: ${API_BASE_URL}${path}
  ├─ Merge headers: { 'Content-Type': 'application/json' }
  ├─ Fetch request
  ├─ Check response.ok
  │   ├─ If error: parse detail message → throw Error
  │   └─ If success: response.json()
  └─ Return typed data: Promise<T>
  ↓
Handler catches error
  ├─ setError(message)
  ├─ setIsSubmitting(false)
  └─ Show UI notification
```

### Supported File Types

| Loại | Extension | Description |
|------|-----------|-------------|
| PDF | `.pdf` | Tài liệu PDF |
| Word | `.docx` | Tài liệu Word |
| Text | `.txt` | Tệp văn bản |
| Image | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.tif` | Hình ảnh (OCR) |

### Payload Examples

#### ChatPayload
```json
{
  "question": "Hệ thống này xử lý bao nhiêu user?",
  "session_id": "sess_123abc",
  "model_name": "qwen2.5:1.5b",
  "top_k": 4,
  "source_names": ["PRD_Document.pdf", "Technical_Spec.docx"],
  "answer_language": "vi",
  "chatSettings": {
    "responseLength": "medium",
    "roleplay": "Expert",
    "mode": "normal"
  }
}
```

#### ChatApiResponse
```json
{
  "notebook_id": "nb_123",
  "session_id": "sess_123abc",
  "rewritten_question": "Hệ thống này có khả năng xử lý bao nhiêu người dùng đồng thời?",
  "answer": "Theo PRD, hệ thống có thể xử lý tối đa 4.000 user đồng thời [1]. Với skip optimization, tải trên DB giảm đáng kể [2].",
  "citations": [
    {
      "source_name": "PRD_Document.pdf",
      "page": 5,
      "snippet": "System supports maximum 4,000 concurrent users"
    },
    {
      "source_name": "Technical_Spec.docx",
      "page": null,
      "snippet": "Skip optimization reduces database load significantly"
    }
  ]
}
```

---

## Kỳ Vọng & Best Practices

### Frontend Guidelines (từ FRONTEND_FLOW.md)

1. **State Management**
   - Keep single source of truth in Zustand store
   - Chat messages scoped by notebook_id
   - Session IDs keyed by notebook_id

2. **API Integration**
   - Always use `apiService` - không direct fetch
   - Add language parameter `answer_language` cho chat
   - New fields → update ChatPayload type first

3. **Chat Behavior**
   - Disable input khi không có sources
   - Detect question language tự động
   - Store session_id per notebook (không mixing)
   - Pass selected source names to backend

4. **Upload Management**
   - Use polling với getJob(jobId)
   - Refresh sources sau upload thành công
   - Không store mixed chat history

5. **TypeScript**
   - Chat message role: `'user' | 'assistant'` chỉ
   - Không thêm mock data vào production
   - Strict type safety

### Performance Considerations

1. **Polling Strategy**
   - Interval: 2-3 seconds cho upload jobs
   - Exponential backoff (optional)
   - Stop khi status = 'completed' | 'failed'

2. **UI Responsiveness**
   - Lazy load source detail on demand
   - Don't fetch all sources chunks by default
   - Virtual scrolling cho long chat histories

3. **Memory Management**
   - Clear source detail when view closes
   - Reset chat when notebook changes
   - Cleanup refs in useEffect return

### Security Reminders

1. **No Mock Data** in production
2. **No Chat Mixing** across notebooks
3. **Strict Typing** - avoid `any` type
4. **Error Handling** - don't expose backend details
5. **Session Management** - keep session_id secure

---

## Tóm Tắt

| Trang | Tính Năng Chính | API Endpoints | Data Flow |
|------|-----------------|---------------|-----------|
| **HomePage** | CRUD Notebook | 5 endpoints | List → Create → Select |
| **NotebookPage** | Chat + Upload | 10+ endpoints | Load → Upload → Chat |
| **LeftSidebar** | Source Management | 6 endpoints | Upload → Poll → Refresh |
| **ChatArea** | AI Chat | 3 endpoints | Send → API → Display |
| **RightStudio** | Utilities | 0 (static) | Display based on mode |

Tất cả thao tác đều được quản lý qua **Zustand Store** và **apiService**, đảm bảo state nhất quán, type-safe, và dễ debug.
