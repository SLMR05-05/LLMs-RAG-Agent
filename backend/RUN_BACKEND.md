# Bước 1: Vào thư mục backend
cd /home/khanh/project/LLM-RAGs-Agent/LLMs-RAG-Agent/backend

# Bước 2: Kích hoạt virtual environment
source .venv/bin/activate

# Bước 3: Chạy uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Nếu gặp lỗi  ERROR:    [Errno 98] Address already in use
Gõ lệnh trên terminal: 
lsof -i :8000
kill -9 $(lsof -t -i :8000)
