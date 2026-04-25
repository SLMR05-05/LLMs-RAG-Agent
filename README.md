# LLM-RAG-Agent

LLM-RAG-Agent la mot do an Document Intelligence fullstack, ket hop:
- Backend FastAPI de xu ly notebook, source, chat, job va indexing
- Frontend React de quan ly notebook, upload tai lieu, chat co citation, va xuat ket qua GraphRAG
- RAG vector voi FAISS + Ollama embeddings
- GraphRAG voi FalkorDB de khai thac quan he tri thuc

Du an ho tro xu ly da dinh dang:
- PDF, DOCX, TXT
- Hinh anh co OCR
- Web content tu URL
- Chat song ngu Viet/Anh

## Tong quan kien truc

```text
Frontend React
  -> FastAPI API
	  -> Notebook Storage (SQLite + filesystem)
	  -> RAG Service Layer
		  -> Vector RAG (FAISS + Ollama)
		  -> GraphRAG (FalkorDB)
	  -> Job Service (background tasks)
```

### Diem noi bat

- Chat pipeline tich hop ca RAG va GraphRAG trong cung mot endpoint
- Luu history theo notebook/session de tach nguyen canh
- Ho tro upload file lon va web source qua job bat dong bo
- Co citation va preview source ro rang

## Cau truc thu muc chinh

```text
LLMs-RAG-Agent/
├── backend/
│   ├── api/
│   ├── core/
│   ├── services/
│   ├── storage/
│   ├── README-related reports
│   └── requirements*.txt
├── frontend_react/
│   ├── src/
│   ├── public/
│   └── package.json
├── documentation/
│   ├── project_report.tex
│   └── README.md
└── README.md
```

## Tinh nang chinh

### Backend

- CRUD notebook
- Upload file va add web source
- OCR anh bang EasyOCR
- Parse PDF/DOCX/TXT/URL
- Chunking va embedding local
- FAISS semantic search
- GraphRAG voi FalkorDB
- Async job tracking
- Luu chat history va source metadata

### Frontend

- Dashboard danh sach notebook
- Tao/doi ten/xoa notebook
- Upload file keo-tha
- Them web source
- Source detail view va preview
- Chat co citation
- Hien thi ket qua RAG va GraphRAG rieng

## Tai lieu tham khao

- [Backend architecture report](backend/REPORT_BACKEND_ARCHITECTURE.md)
- [Backend features report](backend/REPORT_BACKEND_FEATURES.md)
- [Detailed backend structure](backend/BACKEND_DETAILED_STRUCTURE.md)
- [Frontend flow report](frontend_react/src/FRONTEND_FLOW.md)
- [Frontend features report](frontend_react/src/pages/PAGES_FEATURES_REPORT.md)

## Yeu cau he thong

- Python 3.8+
- Node.js 18+
- Ollama local server
- Optional: Docker cho FalkorDB/Redis neu ban chay GraphRAG

## Cai dat

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend_react
npm install
```

### 3. Ollama models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

## Chay du an

### Backend

Chay backend theo script/entry point hien co trong repo. Neu ban dung FastAPI, thong thuong se chay bang Uvicorn voi app khai bao trong `backend/api/main.py`.

```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend_react
npm run dev
```

Frontend thuong chay tai:
- `http://localhost:5173`

Backend API thuong chay tai:
- `http://localhost:8000`

## Luong xu ly tong quat

### 1. Upload tai lieu

1. User upload file tu frontend
2. Backend validate notebook, file type, va size
3. Job duoc tao de xu ly bat dong bo
4. Tai lieu duoc parse, clean, chunk
5. Chunk duoc embed bang Ollama
6. Vector duoc luu vao FAISS
7. Metadata va source duoc luu vao SQLite

### 2. Chat hoi dap

1. User gui cau hoi
2. Frontend detect ngon ngu va gui session/source selection
3. Backend embed cau hoi va search FAISS
4. RAG tao answer chinh tu context
5. GraphRAG duoc goi bo sung trong cung luong chat
6. Frontend nhan `answer` va `answer_graph`
7. Hien thi citation va source references

## RAG va GraphRAG

Du an nay khong chay RAG va GraphRAG nhu hai he thong tach roi. Cach dung thuc te la:

- RAG vector la pipeline chinh de lay context va sinh tra loi
- GraphRAG duoc goi bo sung trong cung mot lan chat de khai thac quan he tri thuc
- Frontend se render hai ket qua rieng: `answer` va `answer_graph`

## API tong hop

### Notebook
- `POST /notebooks`
- `GET /notebooks`
- `GET /notebooks/{id}`
- `PATCH /notebooks/{id}`
- `DELETE /notebooks/{id}`

### Source
- `GET /notebooks/{id}/sources`
- `GET /notebooks/{id}/sources/{sid}`
- `GET /notebooks/{id}/sources/{sid}/preview`
- `POST /notebooks/{id}/sources/upload`
- `POST /notebooks/{id}/sources/url`
- `PUT /sources/{sourceId}`
- `DELETE /sources/{sourceId}`

### Chat
- `POST /notebooks/{id}/chat`
- `GET /notebooks/{id}/chat/messages`
- `DELETE /notebooks/{id}/chat`

### Jobs
- `GET /jobs/{jobId}`

### System
- `GET /healthz`

## Cau hinh quan trong

Mo mot file `.env` hoac bien moi truong tuong ung neu du an cua ban co dung config tu backend:

- `OLLAMA_HOST`
- `FALKOR_HOST`
- `FALKOR_PORT`
- `CORS_ALLOW_ORIGINS`
- `MAX_UPLOAD_FILES`
- `MAX_UPLOAD_SIZE_MB`

## Van de thuong gap

- Neu embedding that bai, kiem tra Ollama da chay chua
- Neu GraphRAG loi, kiem tra FalkorDB/Redis va snapshot graph
- Neu frontend goi API loi, kiem tra CORS va base URL
- Neu khong tim duoc context, kiem tra notebook da co source duoc index chua

## Tai lieu chi tiet

Neu ban muon xem khai niem kien truc va flow day du, hay mo cac file sau:

- [backend/REPORT_BACKEND_ARCHITECTURE.md](backend/REPORT_BACKEND_ARCHITECTURE.md)
- [backend/BACKEND_DETAILED_STRUCTURE.md](backend/BACKEND_DETAILED_STRUCTURE.md)
- [backend/REPORT_BACKEND_FEATURES.md](backend/REPORT_BACKEND_FEATURES.md)
- [frontend_react/src/FRONTEND_FLOW.md](frontend_react/src/FRONTEND_FLOW.md)

## Ket luan

LLM-RAG-Agent la mot he thong Document RAG fullstack, co RAG vector la luong chinh va GraphRAG la luong bo sung trong cung pipeline chat. Kien truc nay phu hop cho bai toan tra cuu tai lieu, hoi dap co citation, va khai thac quan he tri thuc trong moi truong local-first.
