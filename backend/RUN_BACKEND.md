# 🚀 Hướng Dẫn Cài Đặt và Khởi Động Backend

## 📋 Yêu Cầu Hệ Thống
- **Python**: 3.12+
- **OS**: Linux, macOS, Windows (with WSL)

---

## 📦 Bước 1: Cài Đặt Dependencies

### 1.1 Tạo Virtual Environment
```bash
# Từ thư mục gốc project
cd /home/khanh/project/LLM-RAGs-Agent/LLMs-RAG-Agent

# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
# Trên Linux/macOS:
source .venv/bin/activate

# Trên Windows:
.venv\Scripts\activate
```

### 1.2 Cài Đặt Dependencies Backend
```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt tất cả packages từ requirements.txt
pip install -r backend/requirements.txt
```

**Dependencies chính:**
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `langchain` - LLM framework
- `sqlalchemy` + `aiosqlite` - Database ORM
- `pydantic-settings` - Configuration management
- `python-multipart` - File upload support
- `aiofiles` - Async file operations
- `pdfplumber`, `python-docx`, `easyocr` - Document processing

---

## 🔧 Bước 2: Cấu Hình Environment (Tuỳ Chọn)

### 2.1 Tạo File `.env` (nếu cần custom settings)
```bash
# Tạo file .env trong thư mục backend
cd backend
touch .env
```

### 2.2 Nội Dung `.env` (mặc định):
```env
# CORS Configuration
CORS_ALLOW_ORIGINS=*

# File Upload Settings
MAX_UPLOAD_FILES=20
MAX_UPLOAD_SIZE_MB=20

# App Settings
APP_NAME=SmartDoc FastAPI Backend
APP_VERSION=0.2.0
```

---

## ▶️ Bước 3: Khởi Động Backend

### 3.1 Khởi Động với Reload Mode (Development)
```bash
# Từ thư mục backend
cd backend

# Kích hoạt virtual environment (nếu chưa kích hoạt)
source ../.venv/bin/activate  # Linux/macOS
# hoặc
..\.venv\Scripts\activate  # Windows

# Chạy backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Tùy Chọn:**
- `--reload` - Tự động reload khi file thay đổi
- `--host 0.0.0.0` - Lắng nghe trên tất cả network interfaces
- `--port 8000` - Port mặc định (có thể thay đổi)

### 3.2 Khởi Động Production Mode (Không Reload)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🌐 Bước 4: Truy Cập Giao Diện

Sau khi backend chạy thành công, bạn sẽ thấy:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Truy Cập API Documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🐛 Khắc Phục Lỗi Thường Gặp

### Lỗi: `ModuleNotFoundError: No module named 'api'`
**Nguyên nhân:** Chạy uvicorn từ thư mục sai
**Giải pháp:**
```bash
# Đảm bảo bạn đang ở trong thư mục backend
cd backend
uvicorn api.main:app --reload
```

### Lỗi: `[Errno 98] Address already in use`
**Nguyên nhân:** Port 8000 đã bị sử dụng
**Giải pháp:**
```bash
# Tìm và dừng process đang sử dụng port 8000
kill -9 $(lsof -t -i :8000)

# Hoặc sử dụng port khác
uvicorn api.main:app --reload --port 8001
```

### Lỗi: `No module named 'uvicorn'`
**Nguyên nhân:** Dependencies chưa được cài đặt hoặc virtual environment chưa kích hoạt
**Giải pháp:**
```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Cài đặt lại dependencies
pip install -r requirements.txt
```

---

## 📁 Cấu Trúc Project

```
backend/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app entry point
│   └── schemas.py       # Request/Response models
├── core/
│   ├── __init__.py
│   └── config.py        # Configuration settings
├── services/
│   ├── __init__.py
│   ├── job_service.py   # Background jobs
│   ├── notebook_storage.py  # Notebook management
│   └── rag_service.py   # RAG service
├── storage/
│   ├── schema.sql
│   └── notebooks/       # Notebook data
├── requirements.txt
├── RUN_BACKEND.md       # File này
└── README.md
```

---

## 🔗 API Endpoints Chính

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| GET | `/health` | Kiểm tra health status |
| GET | `/docs` | Swagger UI documentation |
| POST | `/notebooks` | Tạo notebook mới |
| POST | `/sources/upload` | Upload tài liệu |
| POST | `/chat` | Gửi tin nhắn chat |
| GET | `/sources` | Lấy danh sách sources |

---

## 💾 Database & Storage

- **Database**: SQLite (aiosqlite)
- **Storage**: `backend/storage/notebooks/` - Lưu trữ vector DB, documents
- **Note**: Thư mục `storage/` đã được thêm vào `.gitignore`

---

## 📝 Ghi Chú

- Backend chạy trên **port 8000** mặc định
- CORS cho phép tất cả origins (`*`)
- Max upload file size: **20MB**
- Max upload files: **20 files**
- Hỗ trợ formats: PDF, DOCX, TXT, PNG, JPG, GIF, TIFF

---

## 🚀 Tiếp Theo

Sau khi backend chạy thành công:
1. Khởi động frontend React: `npm run dev`
2. Truy cập: http://localhost:5173
3. Backend API: http://localhost:8000

---

**Hỏi thêm hoặc gặp vấn đề?** Liên hệ hoặc kiểm tra logs chi tiết trong terminal.
