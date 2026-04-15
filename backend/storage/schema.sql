PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS notebooks (
    notebook_id TEXT PRIMARY KEY,
    notebook_name TEXT NOT NULL,
    folder_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS source_documents (
    document_id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    upload_path TEXT NOT NULL,
    page_count INTEGER,
    file_size_bytes INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE,
    UNIQUE (notebook_id, file_hash)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text_content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES source_documents (document_id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);
-- Stable mapping from FAISS integer vector id to chunk/page metadata.
CREATE TABLE IF NOT EXISTS vector_entries (
    notebook_id TEXT NOT NULL,
    faiss_vector_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    page_number INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (notebook_id, faiss_vector_id),
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    UNIQUE (chunk_id)
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    session_title TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    notebook_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS graph_snapshots (
    graph_id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    graph_path TEXT NOT NULL,
    graph_format TEXT NOT NULL DEFAULT 'networkx',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    notebook_id TEXT,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed')
    ),
    request_json TEXT,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_source_documents_notebook_id ON source_documents (notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_notebook_id ON chunks (notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_vector_entries_notebook_id ON vector_entries (notebook_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_notebook_id ON chat_sessions (notebook_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_notebook_id ON chat_messages (notebook_id);
CREATE INDEX IF NOT EXISTS idx_jobs_notebook_id ON jobs (notebook_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
CREATE TRIGGER IF NOT EXISTS trg_notebooks_updated_at
AFTER
UPDATE ON notebooks FOR EACH ROW BEGIN
UPDATE notebooks
SET updated_at = CURRENT_TIMESTAMP
WHERE notebook_id = NEW.notebook_id;
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at
AFTER
UPDATE ON jobs FOR EACH ROW BEGIN
UPDATE jobs
SET updated_at = CURRENT_TIMESTAMP
WHERE job_id = NEW.job_id;
END;