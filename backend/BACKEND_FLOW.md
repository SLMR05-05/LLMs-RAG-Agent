# Copilot Backend Flow

## Purpose
- Keep backend edits focused on the real RAG pipeline.
- Prevent Copilot from adding unrelated abstractions.

## Core Backend Modules
- `backend/api/main.py`: FastAPI routes, request validation wiring, background task kickoff.
- `backend/api/schemas.py`: Pydantic request/response models and API contracts.
- `backend/services/rag_service.py`: upload parsing, OCR/text extraction, indexing, retrieval, chat generation.
- `backend/services/notebook_storage.py`: SQLite persistence for notebooks, sources, chunks, vectors, sessions, chat messages, jobs.
- `backend/services/job_service.py`: job lifecycle helpers used by async routes.
- `backend/storage/schema.sql`: canonical DB schema and constraints.

## End-to-End Flow
1. Create notebook
   - Route: `POST /notebooks`
   - Storage creates notebook record + folder structure.
2. Upload sources
   - Route: `POST /notebooks/{id}/sources/upload`
   - Upload job saves files to `storage/notebooks/{id}/docx`
   - Metadata inserted into `source_documents`
   - Auto-index runs after successful upload save
3. Indexing
   - Route: `POST /notebooks/{id}/index` (manual)
   - Service splits docs into chunks, writes `chunks` + `vector_entries`, saves FAISS index.
4. Chat
   - Route: `POST /notebooks/{id}/chat`
   - Session resolved per notebook
   - Retrieval from FAISS with notebook/source filters
   - Prompt enforces answer language (`vi`/`en`)
   - Messages persisted in `chat_messages`
5. Chat history load
   - Route: `GET /notebooks/{id}/chat/messages`
   - Returns latest session and ordered messages for notebook scope.
6. Source preview
   - Route: `GET /notebooks/{id}/sources/{document_id}/preview`
   - Streams original uploaded file for frontend preview.

## Data Ownership Rules
- Notebook scope is mandatory: all source/chat/chunk/vector queries must include `notebook_id`.
- Session scope is notebook-bound: `session_id` from another notebook must not be reused.
- Job rows are source of truth for upload/index async status.

## Language and Answering Rules
- `ChatRequest.answer_language` supports `vi` or `en`.
- If provided, service must honor it.
- If absent, service detects language from question.

## Common Failure Modes
- Upload succeeded but chat has no context:
  - Index missing or stale relative to latest sources.
  - OCR extracted empty text from image source.
- Cross-notebook chat leakage:
  - Missing `notebook_id` in queries or session resolution.

## Copilot Guardrails
- Do not bypass `schemas.py` when adding API fields.
- Do not add new DB writes outside `notebook_storage.py` unless unavoidable.
- Keep route handlers thin; business logic belongs in services.
- Preserve backward compatibility of existing response fields.
- Prefer deterministic error messages for client UX.
