from __future__ import annotations

import hashlib
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import easyocr
import numpy as np
import ollama
from docx import Document as DocxDocument
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from services.notebook_storage import NotebookStorage


class LocalOllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text") -> None:
        host = os.getenv("OLLAMA_HOST")
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model

    def _extract_embeddings(self, response) -> List[List[float]]:
        if hasattr(response, "embeddings"):
            return response.embeddings
        if isinstance(response, dict) and "embeddings" in response:
            return response["embeddings"]
        raise ValueError("Unexpected embedding response format from Ollama")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        return self._extract_embeddings(response)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RAGService:
    SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}

    def __init__(self, storage: Optional[NotebookStorage] = None) -> None:
        self.storage = storage or NotebookStorage()
        self.embedder = LocalOllamaEmbeddings()
        self._ocr_reader: Optional[easyocr.Reader] = None

    def _get_ocr_reader(self) -> easyocr.Reader:
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(["en", "vi"], gpu=False)
        return self._ocr_reader

    def _ollama_client(self) -> ollama.Client:
        host = os.getenv("OLLAMA_HOST")
        return ollama.Client(host=host) if host else ollama.Client()

    def _hash_bytes(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _build_prompt_template(self, is_vi: bool) -> PromptTemplate:
        if is_vi:
            template = (
                "Ban chi duoc tra loi dua tren ngu canh da cung cap. "
                "Neu khong du du lieu, hay noi ro la khong biet.\n\n"
                "Ngu canh:\n{context}\n\n"
                "Cau hoi: {question}\n\n"
                "Tra loi ngan gon va co trich dan theo nguon."
            )
        else:
            template = (
                "Answer strictly based on the provided context. "
                "If context is insufficient, clearly say you do not know.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Give a concise answer with source-grounded statements."
            )
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _is_vietnamese(self, text: str) -> bool:
        chars = "aadeioouuyáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
        lowered = text.lower()
        return any(c in lowered for c in chars)

    def _parse_file_to_documents(self, file_path: Path, source_name: str, file_hash: str) -> Tuple[List[Document], int]:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            docs = PDFPlumberLoader(str(file_path)).load()
            for d in docs:
                page_num = int(d.metadata.get("page", 1))
                d.metadata.update({"source_name": source_name, "page_number": page_num, "file_hash": file_hash})
            return docs, len(docs)

        if suffix == ".docx":
            docx = DocxDocument(str(file_path))
            text = "\n".join([p.text for p in docx.paragraphs if p.text])
            docs = [Document(page_content=text, metadata={"source_name": source_name, "page_number": 1, "file_hash": file_hash})]
            return docs, 1

        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={"source_name": source_name, "page_number": 1, "file_hash": file_hash})]
            return docs, 1

        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}:
            image = Image.open(file_path).convert("RGB")
            image_array = np.array(image)
            lines = self._get_ocr_reader().readtext(image_array, detail=0)
            text = "\n".join(lines).strip()
            docs = [Document(page_content=text, metadata={"source_name": source_name, "page_number": 1, "file_hash": file_hash})]
            return docs, 1

        raise ValueError(f"Unsupported file type: {suffix}")

    def save_upload(self, notebook_id: str, filename: str, content: bytes) -> Optional[dict]:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_FILE_EXTENSIONS:
            return None

        dirs = self.storage.get_notebook_dirs(notebook_id)
        dirs["uploads"].mkdir(parents=True, exist_ok=True)

        file_hash = self._hash_bytes(content)
        disk_name = f"{file_hash}{suffix}"
        upload_path = dirs["uploads"] / disk_name
        upload_path.write_bytes(content)

        document_id = uuid.uuid4().hex
        self.storage.upsert_source_document(
            document_id=document_id,
            notebook_id=notebook_id,
            source_name=filename,
            source_type=suffix.lstrip("."),
            file_hash=file_hash,
            upload_path=str(upload_path),
            page_count=None,
            file_size_bytes=len(content),
        )
        return {
            "document_id": document_id,
            "source_name": filename,
            "source_type": suffix.lstrip("."),
            "file_hash": file_hash,
            "upload_path": str(upload_path),
            "file_size_bytes": len(content),
        }

    def _load_or_create_vector_store(self, notebook_id: str) -> FAISS:
        dirs = self.storage.get_notebook_dirs(notebook_id)
        index_dir = dirs["vector_db"]
        index_file = index_dir / "index.faiss"

        if index_file.exists():
            return FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=self.embedder,
                allow_dangerous_deserialization=True,
            )

        # Create a new FAISS store with a temporary seed document, then clear content.
        seed = Document(page_content="seed", metadata={"source_name": "seed", "page_number": 0})
        vector_store = FAISS.from_documents([seed], self.embedder)
        vector_store.delete(ids=list(vector_store.index_to_docstore_id.values()))
        return vector_store

    def index_notebook(
        self,
        notebook_id: str,
        source_ids: Optional[List[str]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict:
        if not self.storage.notebook_exists(notebook_id):
            raise ValueError("Notebook not found")

        all_sources = self.storage.list_source_documents(notebook_id)
        if source_ids:
            source_id_set = set(source_ids)
            selected_sources = [row for row in all_sources if row["document_id"] in source_id_set]
        else:
            selected_sources = list(all_sources)

        if not selected_sources:
            raise ValueError("No source documents found to index")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vector_store = self._load_or_create_vector_store(notebook_id)

        self.storage.clear_index_data(notebook_id)

        total_chunks = 0
        total_pages = 0
        current_vector_size = len(vector_store.index_to_docstore_id)

        for src in selected_sources:
            source_docs, page_count = self._parse_file_to_documents(
                file_path=Path(src["upload_path"]),
                source_name=src["source_name"],
                file_hash=src["file_hash"],
            )
            total_pages += page_count

            chunks = splitter.split_documents(source_docs)
            if not chunks:
                continue

            chunk_ids: List[str] = []
            for idx, chunk in enumerate(chunks):
                chunk_id = uuid.uuid4().hex
                chunk.metadata.update(
                    {
                        "notebook_id": notebook_id,
                        "document_id": src["document_id"],
                        "chunk_index": idx,
                    }
                )
                page_number = chunk.metadata.get("page_number") or chunk.metadata.get("page") or 1
                page_number = int(page_number)
                chunk_ids.append(chunk_id)

                self.storage.insert_chunk(
                    chunk_id=chunk_id,
                    notebook_id=notebook_id,
                    document_id=src["document_id"],
                    chunk_index=idx,
                    page_number=page_number,
                    text_content=chunk.page_content,
                    metadata_json=json.dumps(chunk.metadata, ensure_ascii=True),
                )

            vector_store.add_documents(documents=chunks, ids=chunk_ids)

            for i, chunk_id in enumerate(chunk_ids):
                vector_id = current_vector_size + i
                page_number = int(chunks[i].metadata.get("page_number") or chunks[i].metadata.get("page") or 1)
                self.storage.upsert_vector_entry(
                    notebook_id=notebook_id,
                    faiss_vector_id=vector_id,
                    chunk_id=chunk_id,
                    source_name=src["source_name"],
                    page_number=page_number,
                )

            current_vector_size += len(chunk_ids)
            total_chunks += len(chunk_ids)

        dirs = self.storage.get_notebook_dirs(notebook_id)
        dirs["vector_db"].mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(dirs["vector_db"]))

        return {
            "notebook_id": notebook_id,
            "indexed_sources": len(selected_sources),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "vector_path": str(dirs["vector_db"]),
        }

    def _rewrite_query(self, notebook_id: str, session_id: str, question: str) -> str:
        history = self.storage.get_recent_chat_messages(notebook_id=notebook_id, session_id=session_id, limit=5)
        if not history:
            return question

        history_lines = [f"{item['role']}: {item['content']}" for item in history]
        prompt = (
            "Rewrite the follow-up question into a standalone query using the recent chat history. "
            "Return only the rewritten query.\n\n"
            f"Chat history:\n{os.linesep.join(history_lines)}\n\n"
            f"Question: {question}\n"
            "Rewritten query:"
        )

        response = self._ollama_client().generate(model="qwen2.5:1.5b", prompt=prompt)
        if hasattr(response, "response"):
            rewritten = response.response
        else:
            rewritten = response.get("response", "")
        rewritten = (rewritten or "").strip()
        return rewritten if rewritten else question

    def _build_chat_context(self, docs: List[Document]) -> str:
        sections = []
        for doc in docs:
            source_name = doc.metadata.get("source_name", "unknown")
            page_number = doc.metadata.get("page_number") or doc.metadata.get("page") or "N/A"
            sections.append(f"[source={source_name} page={page_number}]\n{doc.page_content}")
        return "\n\n".join(sections)

    def chat(
        self,
        notebook_id: str,
        question: str,
        model_name: str,
        top_k: int,
        session_id: Optional[str],
        source_names: Optional[List[str]],
    ) -> dict:
        if not self.storage.notebook_exists(notebook_id):
            raise ValueError("Notebook not found")

        dirs = self.storage.get_notebook_dirs(notebook_id)
        index_file = dirs["vector_db"] / "index.faiss"
        if not index_file.exists():
            raise ValueError("Notebook has no index yet. Call /index first.")

        session_id = self.storage.ensure_chat_session(notebook_id=notebook_id, session_id=session_id)
        rewritten_question = self._rewrite_query(notebook_id=notebook_id, session_id=session_id, question=question)

        vector_store = FAISS.load_local(
            folder_path=str(dirs["vector_db"]),
            embeddings=self.embedder,
            allow_dangerous_deserialization=True,
        )

        metadata_filter = {"notebook_id": notebook_id}
        if source_names:
            metadata_filter["source_name"] = {"$in": source_names}

        docs = vector_store.similarity_search(
            query=rewritten_question,
            k=top_k,
            filter=metadata_filter,
        )

        if not docs:
            answer = "Khong tim thay ngu canh phu hop trong notebook de tra loi cau hoi nay."
            citations = []
        else:
            context = self._build_chat_context(docs)
            prompt_template = self._build_prompt_template(self._is_vietnamese(rewritten_question))
            prompt = prompt_template.format(context=context, question=rewritten_question)

            response = self._ollama_client().generate(
                model=model_name,
                prompt=prompt,
                options={"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
            )
            if hasattr(response, "response"):
                answer = response.response
            else:
                answer = response.get("response", "")
            answer = answer.strip()

            citations = []
            for doc in docs:
                citations.append(
                    {
                        "source_name": doc.metadata.get("source_name", "unknown"),
                        "page": doc.metadata.get("page_number") or doc.metadata.get("page"),
                        "snippet": doc.page_content[:280].replace("\n", " "),
                    }
                )

        self.storage.add_chat_message(notebook_id=notebook_id, session_id=session_id, role="user", content=question)
        self.storage.add_chat_message(notebook_id=notebook_id, session_id=session_id, role="assistant", content=answer)

        return {
            "notebook_id": notebook_id,
            "session_id": session_id,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "citations": citations,
        }
