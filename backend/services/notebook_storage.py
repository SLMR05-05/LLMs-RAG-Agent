from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional


BACKEND_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = BACKEND_ROOT / "storage"
NOTEBOOKS_ROOT = STORAGE_ROOT / "notebooks"
DB_PATH = STORAGE_ROOT / "app_metadata.db"
SCHEMA_PATH = STORAGE_ROOT / "schema.sql"


class NotebookStorage:
    def __init__(self, db_path: Path = DB_PATH, notebooks_root: Path = NOTEBOOKS_ROOT) -> None:
        self.db_path = db_path
        self.notebooks_root = notebooks_root

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @staticmethod
    def _to_json(payload: Optional[dict[str, Any]]) -> Optional[str]:
        if payload is None:
            return None
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _from_json(payload: Optional[str]) -> Optional[dict[str, Any]]:
        if not payload:
            return None
        return json.loads(payload)

    def bootstrap(self) -> None:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        self.notebooks_root.mkdir(parents=True, exist_ok=True)
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema_sql)

    def create_notebook(self, notebook_id: str, notebook_name: str) -> Path:
        notebook_dir = self.notebooks_root / notebook_id
        docx_dir = notebook_dir / "docx"
        vector_dir = notebook_dir / "vector_db"
        graph_dir = notebook_dir / "Graph"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notebooks (notebook_id, notebook_name, folder_path)
                VALUES (?, ?, ?)
                """,
                (notebook_id, notebook_name, str(notebook_dir)),
            )

        for path in (docx_dir, vector_dir, graph_dir):
            path.mkdir(parents=True, exist_ok=True)
        return notebook_dir

    def create_notebook_auto(self, notebook_name: str) -> dict:
        notebook_id = uuid.uuid4().hex
        notebook_dir = self.create_notebook(notebook_id=notebook_id, notebook_name=notebook_name)
        return {
            "notebook_id": notebook_id,
            "notebook_name": notebook_name,
            "folder_path": str(notebook_dir),
        }

    def notebook_exists(self, notebook_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT notebook_id FROM notebooks WHERE notebook_id = ?",
                (notebook_id,),
            ).fetchone()
        return row is not None

    def get_notebook(self, notebook_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT notebook_id, notebook_name, folder_path, created_at, updated_at
                FROM notebooks
                WHERE notebook_id = ?
                """,
                (notebook_id,),
            ).fetchone()

    def list_notebooks(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT notebook_id, notebook_name, folder_path, created_at, updated_at
                FROM notebooks
                ORDER BY created_at DESC
                """
            ).fetchall()

    def get_notebook_dirs(self, notebook_id: str) -> dict:
        notebook_dir = self.notebooks_root / notebook_id
        return {
            "root": notebook_dir,
            "uploads": notebook_dir / "docx",
            "vector_db": notebook_dir / "vector_db",
            "graph": notebook_dir / "Graph",
        }

    def delete_notebook(self, notebook_id: str) -> None:
        notebook_dir = self.notebooks_root / notebook_id

        with self._connect() as conn:
            conn.execute("DELETE FROM notebooks WHERE notebook_id = ?", (notebook_id,))

        if notebook_dir.exists():
            shutil.rmtree(notebook_dir)

    def rename_notebook(self, notebook_id: str, notebook_name: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE notebooks
                SET notebook_name = ?
                WHERE notebook_id = ?
                """,
                (notebook_name, notebook_id),
            )
            return conn.execute(
                """
                SELECT notebook_id, notebook_name, folder_path, created_at, updated_at
                FROM notebooks
                WHERE notebook_id = ?
                """,
                (notebook_id,),
            ).fetchone()

    def upsert_source_document(
        self,
        document_id: str,
        notebook_id: str,
        source_name: str,
        source_type: str,
        file_hash: str,
        upload_path: str,
        page_count: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_documents (
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(notebook_id, file_hash)
                DO UPDATE SET
                    source_name = excluded.source_name,
                    source_type = excluded.source_type,
                    upload_path = excluded.upload_path,
                    page_count = excluded.page_count,
                    file_size_bytes = excluded.file_size_bytes
                """,
                (
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                ),
            )

    def list_source_documents(self, notebook_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                    created_at
                FROM source_documents
                WHERE notebook_id = ?
                ORDER BY created_at DESC
                """,
                (notebook_id,),
            ).fetchall()

    def get_source_document(self, notebook_id: str, document_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                    created_at
                FROM source_documents
                WHERE notebook_id = ? AND document_id = ?
                """,
                (notebook_id, document_id),
            ).fetchone()

    def get_source_document_by_id(self, document_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                    created_at
                FROM source_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

    def rename_source_document(self, document_id: str, source_name: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE source_documents
                SET source_name = ?
                WHERE document_id = ?
                """,
                (source_name, document_id),
            )
            conn.execute(
                """
                UPDATE vector_entries
                SET source_name = ?
                WHERE chunk_id IN (
                    SELECT chunk_id
                    FROM chunks
                    WHERE document_id = ?
                )
                """,
                (source_name, document_id),
            )
            return conn.execute(
                """
                SELECT
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                    created_at
                FROM source_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

    def delete_source_document(self, document_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    document_id,
                    notebook_id,
                    source_name,
                    source_type,
                    file_hash,
                    upload_path,
                    page_count,
                    file_size_bytes,
                    created_at
                FROM source_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is None:
                return None

            conn.execute(
                """
                DELETE FROM source_documents
                WHERE document_id = ?
                """,
                (document_id,),
            )
            return row

    def list_chunks_for_document(self, notebook_id: str, document_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    chunk_id,
                    notebook_id,
                    document_id,
                    chunk_index,
                    page_number,
                    text_content,
                    metadata_json,
                    created_at
                FROM chunks
                WHERE notebook_id = ? AND document_id = ?
                ORDER BY chunk_index ASC
                """,
                (notebook_id, document_id),
            ).fetchall()

    def insert_chunk(
        self,
        chunk_id: str,
        notebook_id: str,
        document_id: str,
        chunk_index: int,
        page_number: Optional[int],
        text_content: str,
        metadata_json: Optional[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    notebook_id,
                    document_id,
                    chunk_index,
                    page_number,
                    text_content,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    notebook_id,
                    document_id,
                    chunk_index,
                    page_number,
                    text_content,
                    metadata_json,
                ),
            )

    def clear_index_data(self, notebook_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vector_entries WHERE notebook_id = ?", (notebook_id,))
            conn.execute("DELETE FROM chunks WHERE notebook_id = ?", (notebook_id,))

    def upsert_vector_entry(
        self,
        notebook_id: str,
        faiss_vector_id: int,
        chunk_id: str,
        source_name: str,
        page_number: Optional[int],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vector_entries (
                    notebook_id,
                    faiss_vector_id,
                    chunk_id,
                    source_name,
                    page_number
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(notebook_id, faiss_vector_id)
                DO UPDATE SET
                    chunk_id = excluded.chunk_id,
                    source_name = excluded.source_name,
                    page_number = excluded.page_number
                """,
                (notebook_id, faiss_vector_id, chunk_id, source_name, page_number),
            )

    def get_vector_entries(self, notebook_id: str, vector_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids = list(vector_ids)
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        query = f"""
            SELECT notebook_id, faiss_vector_id, chunk_id, source_name, page_number
            FROM vector_entries
            WHERE notebook_id = ? AND faiss_vector_id IN ({placeholders})
            ORDER BY faiss_vector_id ASC
        """
        with self._connect() as conn:
            return conn.execute(query, [notebook_id, *ids]).fetchall()

    def create_chat_session(self, notebook_id: str, session_title: Optional[str] = None) -> str:
        session_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, notebook_id, session_title)
                VALUES (?, ?, ?)
                """,
                (session_id, notebook_id, session_title),
            )
        return session_id

    def ensure_chat_session(self, notebook_id: str, session_id: Optional[str]) -> str:
        if not session_id:
            return self.create_chat_session(notebook_id=notebook_id)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id
                FROM chat_sessions
                WHERE session_id = ? AND notebook_id = ?
                """,
                (session_id, notebook_id),
            ).fetchone()
        if row:
            return session_id
        return self.create_chat_session(notebook_id=notebook_id)

    def add_chat_message(self, notebook_id: str, session_id: str, role: str, content: str) -> str:
        message_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (message_id, session_id, notebook_id, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, session_id, notebook_id, role, content),
            )
        return message_id

    def get_recent_chat_messages(
        self,
        notebook_id: str,
        session_id: str,
        limit: int = 5,
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE notebook_id = ? AND session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (notebook_id, session_id, limit),
            ).fetchall()
        return list(reversed(rows))

    def get_latest_chat_session(self, notebook_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT session_id, notebook_id, session_title, created_at
                FROM chat_sessions
                WHERE notebook_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (notebook_id,),
            ).fetchone()

    def list_chat_messages(
        self,
        notebook_id: str,
        session_id: str,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT message_id, role, content, created_at
                FROM chat_messages
                WHERE notebook_id = ? AND session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (notebook_id, session_id, limit),
            ).fetchall()

    def delete_chat_history(self, notebook_id: str) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM chat_messages
                WHERE notebook_id = ?
                """,
                (notebook_id,),
            )
            conn.execute(
                """
                DELETE FROM chat_sessions
                WHERE notebook_id = ?
                """,
                (notebook_id,),
            )
            deleted_count = conn.execute("SELECT changes() AS deleted_count").fetchone()
        return int(deleted_count["deleted_count"] if deleted_count is not None else 0)

    def validate_page_alignment(
        self,
        notebook_id: str,
        faiss_vector_ids: Iterable[int],
    ) -> list[sqlite3.Row]:
        vector_id_list = list(faiss_vector_ids)
        if not vector_id_list:
            return []

        placeholders = ",".join(["?"] * len(vector_id_list))
        sql = f"""
            SELECT
                v.notebook_id,
                v.faiss_vector_id,
                v.page_number AS vector_page_number,
                c.page_number AS chunk_page_number,
                v.source_name,
                c.chunk_id
            FROM vector_entries v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            WHERE v.notebook_id = ?
              AND v.faiss_vector_id IN ({placeholders})
        """

        with self._connect() as conn:
            rows = conn.execute(sql, [notebook_id, *vector_id_list]).fetchall()

        mismatches = [
            row
            for row in rows
            if row["vector_page_number"] != row["chunk_page_number"]
        ]
        return mismatches

    def create_job(
        self,
        job_id: str,
        notebook_id: Optional[str],
        job_type: str,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        request_json = self._to_json(request_payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, notebook_id, job_type, status, request_json)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (job_id, notebook_id, job_type, request_json),
            )

    def mark_job_running(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = COALESCE(started_at, CURRENT_TIMESTAMP)
                WHERE job_id = ?
                """,
                (job_id,),
            )

    def mark_job_completed(
        self,
        job_id: str,
        result_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        result_json = self._to_json(result_payload)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET
                    status = 'completed',
                    result_json = ?,
                    error_code = NULL,
                    error_message = NULL,
                    finished_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (result_json, job_id),
            )

    def mark_job_failed(self, job_id: str, error_code: str, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET
                    status = 'failed',
                    error_code = ?,
                    error_message = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (error_code, error_message, job_id),
            )

    def get_job_row(self, job_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    job_id,
                    status,
                    result_json,
                    error_message
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

    def get_job_payload(self, job_id: str) -> Optional[dict[str, Any]]:
        row = self.get_job_row(job_id)
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "result": self._from_json(row["result_json"]),
            "error": row["error_message"],
        }


def bootstrap_storage() -> None:
    NotebookStorage().bootstrap()


if __name__ == "__main__":
    bootstrap_storage()
