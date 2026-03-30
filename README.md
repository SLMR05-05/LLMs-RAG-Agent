# SmartDoc AI - Intelligent Document Q&A System

SmartDoc AI la ung dung RAG (Retrieval-Augmented Generation) cho phep:
- Upload file PDF
- Tao embeddings local bang Ollama (`nomic-embed-text`)
- Luu tru va truy xuat bang FAISS
- Tra loi cau hoi bang LLM local qua Ollama (mac dinh: qwen2.5:7b)

## Cau truc thu muc

```text
LLMs-RAG-Agent/
├── app.py
├── README.md
├── requirement.txt
├── requirement_simplified.txt
├── requirements.txt
├── requirements_simplified.txt
├── data/
│   └── gutenberg.pdf
└── documentation/
	├── README.md
	└── project_report.tex
```

## Cai dat moi truong

1. Tao virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

2. Cai dependencies

```bash
pip install -r requirement_simplified.txt
```

3. Cai dat va tai model Ollama

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

## Chay ung dung

```bash
streamlit run app.py
```

Mo trinh duyet tai `http://localhost:8501`.

## Luong xu ly chinh

1. Upload PDF
2. Trich xuat van ban bang PDFPlumber
3. Chia chunk bang RecursiveCharacterTextSplitter
4. Tao embedding local voi `nomic-embed-text`
5. Luu vector vao FAISS va retrieve top-k chunks
6. Tao prompt song ngu (Viet/Anh) va sinh cau tra loi bang Ollama

## Ghi chu

- App duoc toi uu cho local-first, khong can goi API cloud.
- Neu gap loi ket noi LLM, kiem tra Ollama dang chay va model da duoc pull.
