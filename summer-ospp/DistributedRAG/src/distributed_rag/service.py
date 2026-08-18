from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import PurePath
from threading import Lock
from typing import Any, Dict, List, Optional

from .config import AppConfig, load_config
from .distributed.runtime import RayRuntime
from .infrastructure.job_store import JobStore
from .infrastructure.object_storage import ObjectStorage
from .infrastructure.observability import configure_logging, refresh_runtime_metrics
from .ingestion.pipeline import IngestionPipeline
from .retrieval.citations import CitationService
from .retrieval.engine import RetrievalEngine


class DistributedRAGService:
    def __init__(self, config: Optional[AppConfig] = None, profile: Optional[str] = None):
        self.config = config or load_config(profile)
        configure_logging(self.config.observability.log_level)
        self.storage = ObjectStorage(self.config.storage)
        self.jobs = JobStore(self.config.database)
        self.runtime = RayRuntime(self.config)
        self.ingestion = IngestionPipeline(self.config, self.runtime, self.storage, self.jobs)
        self.retrieval = RetrievalEngine(self.config, self.runtime, self.jobs)
        self.citations = CitationService(self.runtime, self.jobs)
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.ray.ingestion_concurrency,
            thread_name_prefix="rag-ingestion",
        )
        self._futures: Dict[str, Future] = {}
        self._future_lock = Lock()

    def ingest(self, file_name: str, content: bytes, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        return self.ingestion.ingest(file_name, content, metadata, trace_id)

    def submit_ingestion(self, file_name: str, content: bytes, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        job, document = self.ingestion.prepare(file_name, content, metadata, trace_id)
        if job["status"] in {"succeeded", "running"}:
            return job
        self._schedule(job, document)
        return self.jobs.get_job(job["job_id"]) or job

    def ask(self, query: str, document_ids: Optional[List[str]] = None, use_hyde: Optional[bool] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        current_trace = trace_id or f"trace_{uuid.uuid4().hex}"
        hits = self.retrieval.retrieve(query, document_ids=document_ids, use_hyde=use_hyde)
        return self.citations.answer(query, hits, current_trace).as_dict()

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get_job(job_id)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job["status"] not in {"pending", "running"}:
            raise ValueError(f"Job {job_id} cannot be cancelled from status {job['status']}")
        self.ingestion.cancel(job_id)
        return self.jobs.get_job(job_id) or {}

    def retry(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        retryable = job["status"] in {"failed", "cancelled"}
        if job["status"] == "running" and self.jobs.is_stale(job_id):
            retryable = True
        if not retryable:
            raise ValueError(f"Job {job_id} is not retryable from status {job['status']}")
        self.jobs.retry_failed_stages(job_id)
        source_uri = self._job_source_uri(job_id)
        content = self.storage.get_bytes(source_uri)
        file_name = PurePath(source_uri).name
        prepared, document = self.ingestion.prepare(file_name, content, trace_id=job["trace_id"])
        self._schedule(prepared, document, force=True)
        return self.jobs.get_job(job_id) or prepared

    def health(self) -> Dict[str, Any]:
        refresh_runtime_metrics()
        values: Dict[str, Any] = {"database": False, "minio": False, "milvus": False, "ray": False}
        try:
            values["database"] = self.jobs.health()
        except Exception:
            pass
        try:
            self.storage.ensure_bucket()
            values["minio"] = True
        except Exception:
            pass
        try:
            values["milvus"] = bool(self.runtime.get(self.runtime.writer.health.remote()))
        except Exception:
            pass
        try:
            import ray

            values["ray"] = ray.is_initialized()
        except Exception:
            pass
        values["healthy"] = all(values.values())
        return values

    def _job_source_uri(self, job_id: str) -> str:
        from .infrastructure.job_store import JobRecord

        with self.jobs.Session() as session:
            record = session.get(JobRecord, job_id)
            if not record:
                raise KeyError(job_id)
            return record.source_uri

    def _schedule(self, job: Dict[str, Any], document: Any, force: bool = False) -> None:
        job_id = job["job_id"]
        with self._future_lock:
            existing = self._futures.get(job_id)
            if existing and not existing.done() and not force:
                return
            future = self.executor.submit(
                self.ingestion.run_prepared,
                job_id,
                job["trace_id"],
                document,
            )
            self._futures[job_id] = future
            future.add_done_callback(lambda completed, current_job_id=job_id: self._discard_future(current_job_id, completed))

    def _discard_future(self, job_id: str, future: Future) -> None:
        with self._future_lock:
            if self._futures.get(job_id) is future:
                self._futures.pop(job_id, None)
