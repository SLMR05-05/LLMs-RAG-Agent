from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NotebookCreateRequest(BaseModel):
    notebook_name: str = Field(min_length=1, max_length=200)


class NotebookUpdateRequest(BaseModel):
    notebook_name: str = Field(min_length=1, max_length=200)


class NotebookResponse(BaseModel):
    notebook_id: str
    notebook_name: str
    folder_path: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SourceUploadResponse(BaseModel):
    notebook_id: str
    uploaded_count: int
    skipped_count: int
    rejected_count: int
    rejected: List[dict]
    sources: List[dict]
    job_id: Optional[str] = None
    status: Optional[str] = None


class SourceItem(BaseModel):
    document_id: str
    source_name: str
    source_type: str
    file_hash: str
    upload_path: str
    page_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    created_at: Optional[str] = None


class SourcesResponse(BaseModel):
    notebook_id: str
    total: int
    sources: List[SourceItem]


class IndexRequest(BaseModel):
    source_ids: Optional[List[str]] = None
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


class IndexResponse(BaseModel):
    job_id: str
    status: str


class Citation(BaseModel):
    source_name: str
    page: Optional[int]
    snippet: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str = Field(min_length=2)
    model_name: str = Field(default="qwen2.5:1.5b")
    top_k: int = Field(default=4, ge=1, le=20)
    source_names: Optional[List[str]] = None


class ChatResponse(BaseModel):
    notebook_id: str
    session_id: str
    rewritten_question: str
    answer: str
    citations: List[Citation]


class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
