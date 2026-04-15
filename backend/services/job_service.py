from __future__ import annotations

import uuid
from typing import Any, Optional

from services.notebook_storage import NotebookStorage


class JobStore:
    def __init__(self, storage: Optional[NotebookStorage] = None) -> None:
        self._storage = storage or NotebookStorage()

    def create_job(
        self,
        notebook_id: Optional[str] = None,
        job_type: str = "generic",
        request_payload: Optional[dict[str, Any]] = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        self._storage.create_job(
            job_id=job_id,
            notebook_id=notebook_id,
            job_type=job_type,
            request_payload=request_payload,
        )
        return job_id

    def mark_running(self, job_id: str) -> None:
        self._storage.mark_job_running(job_id)

    def mark_completed(self, job_id: str, result: Optional[dict] = None) -> None:
        self._storage.mark_job_completed(job_id=job_id, result_payload=result)

    def mark_failed(self, job_id: str, error: str, error_code: str = "INDEX_JOB_ERROR") -> None:
        self._storage.mark_job_failed(job_id=job_id, error_code=error_code, error_message=error)

    def get_job(self, job_id: str) -> Optional[dict]:
        return self._storage.get_job_payload(job_id)


job_store = JobStore()
