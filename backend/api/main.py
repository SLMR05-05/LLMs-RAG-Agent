from __future__ import annotations

import re
import importlib
import tempfile
from pathlib import Path as FilePath
from urllib.parse import urljoin, urlparse
import requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Path, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ChatClearResponse,
    ChatRequest,
    ChatMessageItem,
    ChatMessagesResponse,
    ChatResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    JobResponse,
    NotebookCreateRequest,
    NotebookUpdateRequest,
    NotebookResponse,
    SourceManageResponse,
    SourceRenameRequest,
    WebSourceRequest,
    WebSourceResponse,
    SourceItem,
    SourceChunkItem,
    SourceDetailResponse,
    SourceUploadResponse,
    SourcesResponse,
)
from core.config import settings
from services.job_service import job_store
from services.notebook_storage import NotebookStorage, bootstrap_storage
from services.rag_service import RAGService


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}
ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".png": {"image/png", "application/octet-stream"},
    ".jpg": {"image/jpeg", "application/octet-stream"},
    ".jpeg": {"image/jpeg", "application/octet-stream"},
    ".bmp": {"image/bmp", "application/octet-stream"},
    ".gif": {"image/gif", "application/octet-stream"},
    ".tiff": {"image/tiff", "application/octet-stream"},
    ".tif": {"image/tiff", "application/octet-stream"},
}

_trafilatura_module = None
_bs4_module = None


def _load_optional_scrapers() -> tuple[object | None, object | None]:
    global _trafilatura_module, _bs4_module

    if _trafilatura_module is None:
        try:
            _trafilatura_module = importlib.import_module("trafilatura")
        except Exception:
            _trafilatura_module = False

    if _bs4_module is None:
        try:
            _bs4_module = importlib.import_module("bs4")
        except Exception:
            _bs4_module = False

    trafilatura_mod = _trafilatura_module if _trafilatura_module is not False else None
    bs4_mod = _bs4_module if _bs4_module is not False else None
    return trafilatura_mod, bs4_mod


def _validate_upload_payload(filename: str, content_type: str | None, content: bytes) -> str | None:
    suffix = FilePath(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return f"Unsupported file extension: {suffix or 'none'}"

    if len(content) > settings.max_upload_size_bytes:
        return (
            f"File exceeds max upload size of {settings.max_upload_size_mb} MB"
        )

    allowed_types = ALLOWED_MIME_TYPES.get(suffix, {"application/octet-stream"})
    ctype = (content_type or "application/octet-stream").lower()
    if ctype not in allowed_types:
        return f"MIME type '{ctype}' does not match extension '{suffix}'"

    return None


def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        docs, _ = rag_service._parse_file_to_documents(
            file_path=FilePath(temp_path),
            source_name="web_pdf",
            file_hash="web_pdf",
        )
        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
        return text.strip()
    except Exception as exc:
        raise ValueError(f"Khong the xu ly PDF tu URL: {exc}") from exc
    finally:
        if temp_path:
            FilePath(temp_path).unlink(missing_ok=True)


def _extract_web_content(url: str) -> dict[str, object]:
    trafilatura_mod, bs4_mod = _load_optional_scrapers()

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
    except requests.RequestException as exc:
        raise ValueError(f"Không thể kết nối URL: {exc}") from exc

    if response.status_code >= 400:
        raise ValueError(
            f"Không thể lấy nội dung URL (HTTP {response.status_code}). Trang có thể chặn bot hoặc yêu cầu xác thực."
        )

    content_type = (response.headers.get("Content-Type") or "").lower()
    parsed_url = urlparse(url)
    is_pdf_url = parsed_url.path.lower().endswith(".pdf")
    is_pdf_response = "application/pdf" in content_type

    if is_pdf_url or is_pdf_response:
        file_name = FilePath(parsed_url.path).name or "web-source.pdf"
        if not file_name.lower().endswith(".pdf"):
            file_name = f"{file_name}.pdf"
        extracted_text = _extract_pdf_text_from_bytes(response.content)
        if not extracted_text:
            raise ValueError("Khong the trich xuat noi dung PDF tu URL")
        return {
            "mode": "pdf",
            "file_name": file_name,
            "content": response.content,
            "extracted_text": extracted_text,
        }

    html = response.text
    image_links: list[str] = []
    bs4_text = ""

    if bs4_mod is not None:
        soup = bs4_mod.BeautifulSoup(html, "html.parser")
        body_root = soup.body or soup
        for script in body_root(["script", "style", "nav", "footer", "header", "noscript"]):
            script.decompose()

        main_content = body_root.find("main") or body_root.find("article") or body_root.find("section") or body_root
        bs4_text = main_content.get_text("\n", strip=True)
        image_links = [
            urljoin(url, tag.get("src"))
            for tag in main_content.find_all("img")
            if tag.get("src")
        ]

    extracted = ""
    if trafilatura_mod is not None:
        extracted = trafilatura_mod.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
        ) or ""

    if extracted.strip():
        text = extracted.strip()
    elif bs4_text.strip():
        text = bs4_text
    else:
        body_match = re.search(r"<body[^>]*>(.*?)</body>", html, flags=re.IGNORECASE | re.DOTALL)
        raw_body = body_match.group(1) if body_match else html
        raw_body = re.sub(r"<script[^>]*>.*?</script>", " ", raw_body, flags=re.IGNORECASE | re.DOTALL)
        raw_body = re.sub(r"<style[^>]*>.*?</style>", " ", raw_body, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", raw_body)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 350 and len(image_links) >= 3:
        ocr_texts: list[str] = []
        seen_urls: set[str] = set()
        for image_url in image_links:
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            if len(ocr_texts) >= 8:
                break

            try:
                image_response = requests.get(
                    image_url,
                    timeout=12,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                        )
                    },
                )
                if image_response.status_code >= 400:
                    continue

                image_content_type = (image_response.headers.get("Content-Type") or "").lower()
                if "image" not in image_content_type and not image_url.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
                ):
                    continue

                ocr_text = rag_service.extract_text_from_image_bytes(image_response.content)
                if ocr_text:
                    ocr_texts.append(ocr_text)
            except Exception:
                continue

        if ocr_texts:
            text = f"{text}\n\n[OCR from webpage images]\n" + "\n\n".join(ocr_texts)

    if not text:
        raise ValueError(
            "Không trích xuất được nội dung văn bản từ URL. Trang có thể chặn truy cập hoặc chỉ chứa script động."
        )
    return {
        "mode": "text",
        "source_url": url,
        "extracted_text": text,
    }


app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = NotebookStorage()
rag_service = RAGService(storage=storage)


@app.on_event("startup")
def on_startup() -> None:
    bootstrap_storage()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/notebooks", response_model=NotebookResponse)
def create_notebook(payload: NotebookCreateRequest) -> NotebookResponse:
    try:
        notebook = storage.create_notebook_auto(payload.notebook_name)
        row = storage.get_notebook(notebook["notebook_id"])
        if row is None:
            raise ValueError("Notebook created but failed to read it back")
        return NotebookResponse(**dict(row))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/notebooks", response_model=list[NotebookResponse])
def list_notebooks() -> list[NotebookResponse]:
    rows = storage.list_notebooks()
    return [NotebookResponse(**dict(row)) for row in rows]


@app.get("/notebooks/{notebook_id}", response_model=NotebookResponse)
def get_notebook(notebook_id: str = Path(min_length=6)) -> NotebookResponse:
    row = storage.get_notebook(notebook_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return NotebookResponse(**dict(row))


@app.patch("/notebooks/{notebook_id}", response_model=NotebookResponse)
def rename_notebook(
    payload: NotebookUpdateRequest,
    notebook_id: str = Path(min_length=6),
) -> NotebookResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    row = storage.rename_notebook(notebook_id=notebook_id, notebook_name=payload.notebook_name)
    if row is None:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return NotebookResponse(**dict(row))


@app.delete("/notebooks/{notebook_id}")
def delete_notebook(notebook_id: str = Path(min_length=6)) -> dict[str, str]:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    storage.delete_notebook(notebook_id)
    return {"status": "deleted", "notebook_id": notebook_id}


@app.post("/notebooks/{notebook_id}/sources/upload", response_model=SourceUploadResponse)
async def upload_sources(
    background_tasks: BackgroundTasks,
    notebook_id: str = Path(min_length=6),
    files: list[UploadFile] = File(...),
) -> SourceUploadResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "TOO_MANY_FILES",
                "message": f"Maximum {settings.max_upload_files} files per request.",
            },
        )

    accepted_uploads: list[dict[str, object]] = []
    rejected = []

    for file in files:
        if not file.filename:
            rejected.append({"filename": "unknown", "reason": "Missing filename"})
            continue

        content = await file.read()
        validation_error = _validate_upload_payload(file.filename, file.content_type, content)
        if validation_error:
            rejected.append({"filename": file.filename, "reason": validation_error})
            continue

        accepted_uploads.append(
            {
                "filename": file.filename or "unknown",
                "content": content,
            }
        )

    if not accepted_uploads:
        return SourceUploadResponse(
            notebook_id=notebook_id,
            uploaded_count=0,
            skipped_count=0,
            rejected_count=len(rejected),
            rejected=rejected,
            sources=[],
            job_id=None,
            status="error",
        )

    job_id = job_store.create_job(
        notebook_id=notebook_id,
        job_type="upload_sources",
        request_payload={
            "file_names": [item["filename"] for item in accepted_uploads],
            "rejected_count": len(rejected),
        },
    )

    def _run_upload_job() -> None:
        job_store.mark_running(job_id)
        uploaded_items: list[dict[str, object]] = []
        skipped_count_local = 0
        try:
            for item in accepted_uploads:
                result = rag_service.save_upload(
                    notebook_id=notebook_id,
                    filename=str(item["filename"]),
                    content=item["content"],
                )
                if result is None:
                    skipped_count_local += 1
                    continue
                uploaded_items.append(result)

            if uploaded_items:
                rag_service.index_notebook(
                    notebook_id=notebook_id,
                    source_ids=None,
                    chunk_size=1000,
                    chunk_overlap=100,
                )

            job_store.mark_completed(
                job_id,
                result={
                    "uploaded_count": len(uploaded_items),
                    "skipped_count": skipped_count_local,
                    "rejected_count": len(rejected),
                    "sources": uploaded_items,
                },
            )
        except Exception as exc:
            job_store.mark_failed(job_id, error=str(exc), error_code="UPLOAD_JOB_ERROR")

    background_tasks.add_task(_run_upload_job)

    return SourceUploadResponse(
        notebook_id=notebook_id,
        uploaded_count=0,
        skipped_count=0,
        rejected_count=len(rejected),
        rejected=rejected,
        sources=[],
        job_id=job_id,
        status="pending",
    )


@app.post("/notebooks/{notebook_id}/sources/url", response_model=WebSourceResponse)
def add_web_source(
    payload: WebSourceRequest,
    background_tasks: BackgroundTasks,
    notebook_id: str = Path(min_length=6),
) -> WebSourceResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    job_id = job_store.create_job(
        notebook_id=notebook_id,
        job_type="add_web_source",
        request_payload={"url": payload.url},
    )

    def _run_web_source_job() -> None:
        job_store.mark_running(job_id)
        try:
            web_payload = _extract_web_content(payload.url)
            if web_payload.get("mode") == "pdf":
                source = rag_service.save_upload(
                    notebook_id=notebook_id,
                    filename=str(web_payload["file_name"]),
                    content=bytes(web_payload["content"]),
                )
                if source is None:
                    raise ValueError("Khong the luu nguon PDF tu URL")
                extracted_length = len(str(web_payload.get("extracted_text", "")))
            else:
                extracted_text = str(web_payload.get("extracted_text") or "")
                source = rag_service.save_web_source(
                    notebook_id=notebook_id,
                    source_url=payload.url,
                    extracted_text=extracted_text,
                )
                extracted_length = len(extracted_text)

            rag_service.index_notebook(
                notebook_id=notebook_id,
                source_ids=None,
                chunk_size=1000,
                chunk_overlap=100,
            )
            job_store.mark_completed(
                job_id,
                result={
                    "source": source,
                    "extracted_length": extracted_length,
                },
            )
        except Exception as exc:
            job_store.mark_failed(job_id, error=str(exc), error_code="WEB_SOURCE_JOB_ERROR")

    background_tasks.add_task(_run_web_source_job)
    return WebSourceResponse(notebook_id=notebook_id, job_id=job_id, status="pending")


@app.delete("/notebooks/{notebook_id}/chat", response_model=ChatClearResponse)
def clear_chat_history(notebook_id: str = Path(min_length=6)) -> ChatClearResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    storage.delete_chat_history(notebook_id)
    return ChatClearResponse(notebook_id=notebook_id, status="deleted")


@app.get("/notebooks/{notebook_id}/sources", response_model=SourcesResponse)
def list_sources(notebook_id: str = Path(min_length=6)) -> SourcesResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    rows = storage.list_source_documents(notebook_id)
    items = [SourceItem(**dict(row)) for row in rows]
    return SourcesResponse(notebook_id=notebook_id, total=len(items), sources=items)


@app.put("/sources/{source_id}", response_model=SourceManageResponse)
def rename_source(
    payload: SourceRenameRequest,
    source_id: str = Path(min_length=8),
) -> SourceManageResponse:
    source = storage.rename_source_document(document_id=source_id, source_name=payload.source_name)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceManageResponse(
        notebook_id=source["notebook_id"],
        document_id=source["document_id"],
        source_name=source["source_name"],
        source_type=source["source_type"],
        status="updated",
    )


@app.delete("/sources/{source_id}", response_model=SourceManageResponse)
def delete_source(source_id: str = Path(min_length=8)) -> SourceManageResponse:
    try:
        deleted = rag_service.delete_source(document_id=source_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete source failed: {exc}") from exc

    return SourceManageResponse(
        notebook_id=str(deleted["notebook_id"]),
        document_id=str(deleted["document_id"]),
        source_name=str(deleted["source_name"]),
        source_type=str(deleted["source_type"]),
        status="deleted",
    )


@app.get("/notebooks/{notebook_id}/sources/{document_id}", response_model=SourceDetailResponse)
def get_source_detail(
    notebook_id: str = Path(min_length=6),
    document_id: str = Path(min_length=8),
) -> SourceDetailResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    source = storage.get_source_document(notebook_id=notebook_id, document_id=document_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    chunk_rows = storage.list_chunks_for_document(notebook_id=notebook_id, document_id=document_id)
    chunks = [
        SourceChunkItem(
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            page_number=row["page_number"],
            text_content=row["text_content"],
            created_at=row["created_at"],
        )
        for row in chunk_rows
    ]

    parsed_markdown = "\n\n".join(
        [f"### Chunk {chunk.chunk_index + 1} (page {chunk.page_number or '-'})\n\n{chunk.text_content}" for chunk in chunks]
    )

    return SourceDetailResponse(
        notebook_id=notebook_id,
        document_id=source["document_id"],
        source_name=source["source_name"],
        source_type=source["source_type"],
        created_at=source["created_at"],
        page_count=source["page_count"],
        file_size_bytes=source["file_size_bytes"],
        chunks=chunks,
        parsed_markdown=parsed_markdown,
    )


@app.get("/notebooks/{notebook_id}/sources/{document_id}/preview")
def preview_source(
    notebook_id: str = Path(min_length=6),
    document_id: str = Path(min_length=8),
) -> FileResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    source = storage.get_source_document(notebook_id=notebook_id, document_id=document_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source document not found")

    upload_path = FilePath(source["upload_path"])
    if not upload_path.exists() or not upload_path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found on disk")

    return FileResponse(path=str(upload_path), filename=source["source_name"])


@app.post("/notebooks/{notebook_id}/index", response_model=IndexResponse)
def index_notebook(
    payload: IndexRequest,
    background_tasks: BackgroundTasks,
    notebook_id: str = Path(min_length=6),
) -> IndexResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    job_id = job_store.create_job(
        notebook_id=notebook_id,
        job_type="index_notebook",
        request_payload={
            "source_ids": payload.source_ids,
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
        },
    )

    def _run_index() -> None:
        job_store.mark_running(job_id)
        try:
            result = rag_service.index_notebook(
                notebook_id=notebook_id,
                source_ids=payload.source_ids,
                chunk_size=payload.chunk_size,
                chunk_overlap=payload.chunk_overlap,
            )
            job_store.mark_completed(job_id, result=result)
        except Exception as exc:
            job_store.mark_failed(job_id, error=str(exc))

    background_tasks.add_task(_run_index)
    return IndexResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str = Path(min_length=8)) -> JobResponse:
    payload = job_store.get_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**payload)


@app.post("/notebooks/{notebook_id}/chat", response_model=ChatResponse)
def chat_notebook(
    payload: ChatRequest,
    notebook_id: str = Path(min_length=6),
) -> ChatResponse:
    print(f"[API CHAT] Received request - notebook_id={notebook_id}, question={payload.question[:50]}...")
    try:
        print(f"[API CHAT] Calling rag_service.chat()...")
        result = rag_service.chat(
            notebook_id=notebook_id,
            question=payload.question,
            model_name=payload.model_name,
            top_k=payload.top_k,
            session_id=payload.session_id,
            source_names=payload.source_names,
            answer_language=payload.answer_language,
            chat_settings=payload.chat_settings.model_dump() if payload.chat_settings else None,
        )
        print(f"[API CHAT] rag_service.chat() returned successfully")
        print(f"[API CHAT] Creating ChatResponse from result...")
        response = ChatResponse(**result)
        print(f"[API CHAT] ChatResponse created successfully")
        return response
    except ValueError as exc:
        print(f"[API CHAT ERROR] ValueError: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[API CHAT ERROR] Unexpected error: {type(exc).__name__}: {str(exc)}")
        import traceback
        print(f"[API CHAT ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_PIPELINE_ERROR",
                "message": f"Internal server error: {str(exc)}",
            },
        ) from exc


@app.get("/notebooks/{notebook_id}/chat/messages", response_model=ChatMessagesResponse)
def get_chat_messages(notebook_id: str = Path(min_length=6)) -> ChatMessagesResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    latest_session = storage.get_latest_chat_session(notebook_id)
    if latest_session is None:
        return ChatMessagesResponse(notebook_id=notebook_id, session_id=None, messages=[])

    session_id = latest_session["session_id"]
    rows = storage.list_chat_messages(notebook_id=notebook_id, session_id=session_id, limit=200)
    messages = [ChatMessageItem(**dict(row)) for row in rows]
    return ChatMessagesResponse(notebook_id=notebook_id, session_id=session_id, messages=messages)
