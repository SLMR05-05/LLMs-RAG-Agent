from __future__ import annotations

from pathlib import Path as FilePath

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
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
    SourceItem,
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
    skipped_count = 0
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
            skipped_count=skipped_count,
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
        try:
            for item in accepted_uploads:
                result = rag_service.save_upload(
                    notebook_id=notebook_id,
                    filename=str(item["filename"]),
                    content=item["content"],
                )
                if result is None:
                    skipped_count_local = 1
                    continue
                uploaded_items.append(result)

            job_store.mark_completed(
                job_id,
                result={
                    "uploaded_count": len(uploaded_items),
                    "skipped_count": skipped_count,
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
        skipped_count=skipped_count,
        rejected_count=len(rejected),
        rejected=rejected,
        sources=[],
        job_id=job_id,
        status="pending",
    )


@app.get("/notebooks/{notebook_id}/sources", response_model=SourcesResponse)
def list_sources(notebook_id: str = Path(min_length=6)) -> SourcesResponse:
    if not storage.notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")

    rows = storage.list_source_documents(notebook_id)
    items = [SourceItem(**dict(row)) for row in rows]
    return SourcesResponse(notebook_id=notebook_id, total=len(items), sources=items)


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
    try:
        result = rag_service.chat(
            notebook_id=notebook_id,
            question=payload.question,
            model_name=payload.model_name,
            top_k=payload.top_k,
            session_id=payload.session_id,
            source_names=payload.source_names,
            answer_language=payload.answer_language,
        )
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_PIPELINE_ERROR",
                "message": "Internal server error while processing chat request.",
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
