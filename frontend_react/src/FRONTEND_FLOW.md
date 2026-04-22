# Frontend SRC Flow for Copilot

## Goal
- Keep Copilot focused on the actual app flow.
- Avoid unrelated suggestions and duplicated logic.

## Core Entry Points
- `src/main.tsx`: app bootstrap.
- `src/App.tsx`: route container.
- `src/pages/HomePage.tsx`: notebook dashboard, create/rename/delete actions.
- `src/pages/NotebookPage.tsx`: load notebook-specific sources + chat history + session.

## State and Data Contracts
- `src/store/useAppStore.ts`
  - Single source of truth for notebook list, active notebook id, sources, chat messages, upload queue, and chat session map by notebook id.
  - Chat separation rule: `chatSessionByNotebook[notebookId]` must be used when sending chat.
- `src/services/api.ts`
  - All HTTP calls go through this file.
  - `chatNotebook()` payload includes `answer_language` to force VI/EN response.
  - `getNotebookChatMessages()` loads latest session messages per notebook.

## Chat Flow (Notebook Scope)
1. User opens notebook route `notebook/:id`.
2. `NotebookPage` calls in parallel:
   - `getNotebookSources(id)`
   - `getNotebookChatMessages(id)`
3. Store updates:
   - `sources` for current notebook
   - `chatMessages` for current notebook
   - `chatSessionByNotebook[id]`
4. In `ChatArea`:
   - disable input when no sources exist
   - detect question language via `src/utils/language.ts`
   - send `session_id` + `answer_language` + selected source names
5. Backend returns answer + citations + session_id; UI stores session_id by notebook.

## Upload and Indexing Flow
- Upload UI: `src/components/Sidebar/LeftSidebar.tsx`
- Polling: `getJob(jobId)` until completed/failed.
- Sources are refreshed after successful processing.
- Chat can auto-index on backend if index is missing.

## Language Handling Rules
- Detection helper: `src/utils/language.ts`
- Supported values: `vi`, `en`
- Frontend sends language hint per question.
- Backend still has fallback detection if hint is absent.

## Guardrails for Copilot
- Do not add mock chat data into production flow.
- Do not store mixed chat history across notebooks.
- Do not bypass `apiService`; add new endpoint wrappers there first.
- Keep notebook-scoped behavior keyed by `activeNotebookId`.
- Preserve strict TypeScript role types: chat message role is `user | assistant` in UI state.

## Quick Edit Checklist
- New API field?
  - Update backend schema -> route -> service.
  - Update frontend `ChatPayload` and sender.
- New notebook-specific feature?
  - Verify state keying by notebook id.
  - Verify page load rehydrates notebook-specific data.
- Chat UX change?
  - Ensure input disabled behavior and source precondition still hold.
