from pathlib import Path

from distributed_rag.config import DatabaseConfig
from distributed_rag.domain.exceptions import ErrorKind, classify_error
from distributed_rag.domain.models import StageName
from distributed_rag.infrastructure.job_store import JobStore


def test_job_creation_is_idempotent(tmp_path: Path):
    store = JobStore(DatabaseConfig(url=f"sqlite:///{tmp_path / 'jobs.db'}"))
    first = store.create_or_get_job("doc-1", "dv-1", "pipeline-v1", "minio://bucket/raw")
    second = store.create_or_get_job("doc-1", "dv-1", "pipeline-v1", "minio://bucket/raw")
    assert first.job_id == second.job_id


def test_stage_retry_and_cancel_are_persisted(tmp_path: Path):
    store = JobStore(DatabaseConfig(url=f"sqlite:///{tmp_path / 'jobs.db'}"))
    job = store.create_or_get_job("doc-1", "dv-1", "pipeline-v1", "minio://bucket/raw")
    store.stage_started(job.job_id, StageName.PARSE)
    store.stage_retry(job.job_id, StageName.PARSE)
    store.request_cancel(job.job_id)
    value = store.get_job(job.job_id)
    assert value["status"] == "cancel_requested"
    assert value["stages"][0]["retry_count"] == 1


def test_error_classification_distinguishes_transient_timeout_and_permanent():
    assert classify_error(ConnectionError("temporarily unavailable")) == ErrorKind.TRANSIENT
    assert classify_error(TimeoutError("slow OCR")) == ErrorKind.TIMEOUT
    assert classify_error(ValueError("unsupported file")) == ErrorKind.PERMANENT
