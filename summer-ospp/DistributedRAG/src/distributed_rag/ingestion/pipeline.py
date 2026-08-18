from __future__ import annotations

import json
import logging
import mimetypes
from dataclasses import asdict
from pathlib import PurePath
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ray
import tiktoken

from ..config import AppConfig
from ..distributed.runtime import RayRuntime
from ..distributed.tasks import chunk_elements_task, parse_document_task
from ..domain.exceptions import ErrorKind, PipelineError, classify_error
from ..domain.identifiers import (
    detect_file_type,
    deterministic_object_key,
    make_document_id,
    make_document_version,
    safe_file_name,
    sha256_bytes,
)
from ..domain.models import Document, JobStatus, StageName, StageStatus
from ..infrastructure.job_store import JobStore
from ..infrastructure.object_storage import ObjectStorage
from ..infrastructure.observability import ACTOR_QUEUE, BATCH_SIZE, CHUNKS, DEGRADATIONS, DOCUMENTS, ELEMENTS, PAGES, RETRIES, log_context, stage_timer


LOGGER = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, config: AppConfig, runtime: RayRuntime, storage: ObjectStorage, jobs: JobStore):
        self.config = config
        self.runtime = runtime
        self.storage = storage
        self.jobs = jobs
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self._job_refs: Dict[str, List[Any]] = {}
        self._job_refs_lock = Lock()

    def ingest(self, file_name: str, content: bytes, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        existing, document = self.prepare(file_name, content, metadata, trace_id)
        if existing["status"] == JobStatus.SUCCEEDED.value:
            return existing
        return self.run_prepared(existing["job_id"], existing["trace_id"], document)

    def prepare(self, file_name: str, content: bytes, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> Tuple[Dict[str, Any], Document]:
        metadata = metadata or {}
        checksum = sha256_bytes(content)
        source_identity = str(metadata.get("logical_document_id") or metadata.get("source_url") or safe_file_name(file_name))
        document_id = make_document_id(source_identity)
        document_version = make_document_version(
            checksum,
            self.config.parsing.parser_version,
            self.config.chunking.version,
            self.config.models.embedding_version,
            document_id=document_id,
        )
        raw_key = deterministic_object_key(document_id, document_version, "raw", safe_file_name(file_name))
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        source_uri = self.storage.put_bytes(raw_key, content, content_type, {"sha256": checksum})
        document = Document(
            document_id=document_id,
            document_version=document_version,
            content_checksum=checksum,
            source_uri=source_uri,
            file_name=safe_file_name(file_name),
            file_type=detect_file_type(file_name).value,
            parser_version=self.config.parsing.parser_version,
            metadata=metadata,
        )
        self.jobs.register_document(
            document_id=document_id,
            document_version=document_version,
            checksum=checksum,
            source_uri=source_uri,
            parser_version=self.config.parsing.parser_version,
            chunking_version=self.config.chunking.version,
            embedding_version=self.config.models.embedding_version,
        )
        job = self.jobs.create_or_get_job(document_id, document_version, self.config.pipeline_version, source_uri, trace_id)
        self.jobs.stage_started(job.job_id, StageName.UPLOAD)
        manifest_key = deterministic_object_key(document_id, document_version, "manifest", "document.json")
        self.storage.put_json(manifest_key, document.as_dict())
        self.jobs.stage_finished(job.job_id, StageName.UPLOAD)
        existing = self.jobs.get_job(job.job_id)
        if not existing:
            raise RuntimeError(f"Failed to create ingestion job {job.job_id}")
        return existing, document

    def run_prepared(self, job_id: str, trace_id: str, document: Document) -> Dict[str, Any]:
        for attempt in range(self.config.ray.max_retries + 1):
            try:
                return self._run(job_id, trace_id, document)
            except Exception as exc:
                kind = exc.kind if isinstance(exc, PipelineError) else classify_error(exc)
                if kind != ErrorKind.TRANSIENT or attempt >= self.config.ray.max_retries:
                    raise
                self.jobs.retry_failed_stages(job_id)
                RETRIES.labels(stage=exc.stage if isinstance(exc, PipelineError) else "pipeline").inc()
        raise RuntimeError("Unreachable ingestion retry state")

    def _run(self, job_id: str, trace_id: str, document: Document) -> Dict[str, Any]:
        logger = log_context(LOGGER, trace_id=trace_id, job_id=job_id, document_id=document.document_id)
        outstanding: List[Any] = []
        with self._job_refs_lock:
            self._job_refs[job_id] = outstanding
        chunk_count = 0
        manifest_index = 0
        embedding_buffer: List[Dict[str, Any]] = []
        embedding_tokens = 0
        seen_pages = set()
        active_stage = StageName.PARSE
        if self.jobs.is_cancel_requested(job_id):
            self.jobs.mark_cancelled(job_id)
            with self._job_refs_lock:
                self._job_refs.pop(job_id, None)
            return self.jobs.get_job(job_id) or {}
        self.jobs.start_job(job_id)
        self.jobs.stage_started(job_id, StageName.PARSE)
        self.jobs.stage_started(job_id, StageName.CHUNK)
        self.jobs.stage_started(job_id, StageName.EMBED)
        self.jobs.stage_started(job_id, StageName.INDEX)
        try:
            parser_generator = parse_document_task.options(
                num_cpus=self.config.ray.parser_cpus,
                max_retries=self.config.ray.max_retries,
            ).remote(asdict(document), asdict(self.config.storage), asdict(self.config.parsing))
            outstanding.append(parser_generator)
            for element_batch_ref in parser_generator:
                self.jobs.heartbeat(job_id)
                outstanding.append(element_batch_ref)
                self._check_cancel(job_id, outstanding)
                with stage_timer(StageName.PARSE.value):
                    active_stage = StageName.PARSE
                    raw_elements = self.runtime.get(element_batch_ref)
                outstanding.remove(element_batch_ref)
                elements = self._resolve_special_elements(raw_elements, outstanding)
                if not elements:
                    continue
                for element in elements:
                    ELEMENTS.labels(type=element["element_type"]).inc()
                    page_number = element.get("source_locator", {}).get("page_number")
                    if page_number is not None and page_number not in seen_pages:
                        seen_pages.add(page_number)
                        PAGES.inc()
                element_key = deterministic_object_key(
                    document.document_id, document.document_version, "elements", f"batch-{manifest_index:08d}.jsonl"
                )
                self.storage.put_jsonl(element_key, elements)
                with stage_timer(StageName.CHUNK.value):
                    active_stage = StageName.CHUNK
                    chunk_ref = chunk_elements_task.options(
                        num_cpus=self.config.ray.chunk_cpus,
                        max_retries=self.config.ray.max_retries,
                    ).remote(elements, asdict(self.config.chunking), asdict(self.config.models), chunk_count)
                    outstanding.append(chunk_ref)
                    chunks = self.runtime.get(chunk_ref)
                    outstanding.remove(chunk_ref)
                chunk_count += len(chunks)
                CHUNKS.inc(len(chunks))
                chunk_key = deterministic_object_key(
                    document.document_id, document.document_version, "chunks", f"batch-{manifest_index:08d}.jsonl"
                )
                self.storage.put_jsonl(chunk_key, chunks)
                manifest_index += 1
                for chunk in chunks:
                    token_count = len(self.encoding.encode(chunk["text"]))
                    if embedding_buffer and (
                        len(embedding_buffer) >= self.config.models.embedding_batch_size
                        or embedding_tokens + token_count > self.config.models.embedding_max_tokens
                    ):
                        active_stage = StageName.EMBED
                        self._embed_and_write(embedding_buffer, outstanding)
                        embedding_buffer = []
                        embedding_tokens = 0
                    embedding_buffer.append(chunk)
                    embedding_tokens += token_count
            if embedding_buffer:
                active_stage = StageName.EMBED
                self._embed_and_write(embedding_buffer, outstanding)
            if parser_generator in outstanding:
                outstanding.remove(parser_generator)
            self.jobs.stage_finished(job_id, StageName.PARSE)
            self.jobs.stage_finished(job_id, StageName.CHUNK)
            self.jobs.stage_finished(job_id, StageName.EMBED)
            active_stage = StageName.INDEX
            self.runtime.get(self.runtime.writer.flush.remote())
            self.jobs.stage_finished(job_id, StageName.INDEX)
            self.jobs.stage_started(job_id, StageName.PUBLISH)
            active_stage = StageName.PUBLISH
            self.jobs.finish_job(job_id, chunk_count)
            self.jobs.stage_finished(job_id, StageName.PUBLISH)
            DOCUMENTS.labels(status="succeeded").inc()
            logger.info("document ingestion completed", extra={"stage": StageName.PUBLISH.value})
            with self._job_refs_lock:
                self._job_refs.pop(job_id, None)
            return self.jobs.get_job(job_id) or {}
        except Exception as exc:
            self.runtime.cancel(outstanding)
            kind = exc.kind if isinstance(exc, PipelineError) else classify_error(exc)
            if kind == ErrorKind.CANCELLED or self.jobs.is_cancel_requested(job_id):
                self.jobs.mark_cancelled(job_id)
            else:
                self.jobs.fail_job(job_id, kind.value, str(exc))
            if isinstance(exc, PipelineError) and exc.stage in {stage.value for stage in StageName}:
                active_stage = StageName(exc.stage)
            self.jobs.stage_failed(job_id, active_stage, kind.value, str(exc))
            remaining_status = StageStatus.CANCELLED if kind == ErrorKind.CANCELLED else StageStatus.SKIPPED
            self.jobs.close_open_stages(job_id, active_stage, remaining_status)
            failure_key = deterministic_object_key(document.document_id, document.document_version, "failures", f"{job_id}.json")
            self.storage.put_json(failure_key, {"job_id": job_id, "error_kind": kind.value, "error": str(exc)})
            DOCUMENTS.labels(status="failed").inc()
            logger.exception("document ingestion failed", extra={"stage": active_stage.value})
            with self._job_refs_lock:
                self._job_refs.pop(job_id, None)
            raise

    def cancel(self, job_id: str) -> None:
        self.jobs.request_cancel(job_id)
        with self._job_refs_lock:
            refs = list(self._job_refs.get(job_id, []))
        self.runtime.cancel(refs)

    def _resolve_special_elements(self, elements: List[Dict[str, Any]], outstanding: List[Any]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        pending: Dict[Any, str] = {}
        for element in elements:
            metadata = element.get("metadata", {})
            if metadata.get("requires_ocr"):
                ref = self.runtime.ocr.next().recognize.remote(element, asdict(self.config.storage))
                pending[ref] = "ocr"
            elif metadata.get("requires_asr") and self.config.parsing.enable_asr:
                suffix = PurePath(metadata["asr_uri"]).suffix or ".wav"
                if self.runtime.asr is None:
                    raise RuntimeError("ASR is enabled but no ASR actor pool is configured")
                ref = self.runtime.asr.next().transcribe.remote(element, asdict(self.config.storage), suffix)
                pending[ref] = "asr"
            elif element.get("text", "").strip():
                output.append(element)
            if len(pending) >= self.config.ray.max_in_flight:
                ACTOR_QUEUE.labels(actor_type="ocr_asr").set(len(pending))
                self._drain_one(pending, output, outstanding)
        while pending:
            ACTOR_QUEUE.labels(actor_type="ocr_asr").set(len(pending))
            self._drain_one(pending, output, outstanding)
        ACTOR_QUEUE.labels(actor_type="ocr_asr").set(0)
        return output

    def _drain_one(self, pending: Dict[Any, str], output: List[Dict[str, Any]], outstanding: List[Any]) -> None:
        refs = list(pending)
        outstanding.extend(refs)
        ready, _ = ray.wait(refs, num_returns=1, timeout=self.config.ray.task_timeout_seconds)
        if not ready:
            self.runtime.cancel(refs)
            raise TimeoutError("OCR/ASR actor batch timed out")
        ref = ready[0]
        try:
            output.extend(self.runtime.get(ref))
        except Exception:
            component = pending[ref]
            DEGRADATIONS.labels(component=component).inc()
            if component == "asr" and self.config.parsing.asr_failure_mode != "skip":
                raise
            if component == "ocr" and self.config.parsing.ocr_failure_mode != "skip":
                raise
        finally:
            pending.pop(ref, None)
            for candidate in refs:
                if candidate in outstanding:
                    outstanding.remove(candidate)

    def _embed_and_write(self, chunks: List[Dict[str, Any]], outstanding: List[Any]) -> None:
        self._check_cancel_from_refs(outstanding)
        BATCH_SIZE.labels(stage="embedding").observe(len(chunks))
        with stage_timer(StageName.EMBED.value):
            actor = self.runtime.embedding.next()
            embedding_ref = actor.embed.remote([chunk["text"] for chunk in chunks], False)
            outstanding.append(embedding_ref)
            try:
                vectors = self.runtime.get(embedding_ref)
            except Exception as exc:
                raise PipelineError(str(exc), classify_error(exc), StageName.EMBED.value) from exc
            outstanding.remove(embedding_ref)
        if len(vectors) != len(chunks):
            raise ValueError(f"Embedding result count mismatch: {len(vectors)} != {len(chunks)}")
        records = [{"chunk": chunk, "vector": vector} for chunk, vector in zip(chunks, vectors)]
        for start in range(0, len(records), self.config.milvus.write_batch_size):
            write_batch = records[start:start + self.config.milvus.write_batch_size]
            BATCH_SIZE.labels(stage="milvus_upsert").observe(len(write_batch))
            with stage_timer(StageName.INDEX.value):
                writer_ref = self.runtime.writer.upsert.remote(write_batch)
                outstanding.append(writer_ref)
                try:
                    written = self.runtime.get(writer_ref)
                except Exception as exc:
                    raise PipelineError(str(exc), classify_error(exc), StageName.INDEX.value) from exc
                outstanding.remove(writer_ref)
            if written != len(write_batch):
                raise ValueError(f"Milvus write count mismatch: {written} != {len(write_batch)}")

    def _check_cancel(self, job_id: str, refs: List[Any]) -> None:
        if self.jobs.is_cancel_requested(job_id):
            self.runtime.cancel(refs)
            raise PipelineError("Job cancellation requested", ErrorKind.CANCELLED, "ingestion")

    @staticmethod
    def _check_cancel_from_refs(refs: List[Any]) -> None:
        if len(refs) > 1000:
            raise RuntimeError("Unexpected unbounded Ray reference accumulation")
