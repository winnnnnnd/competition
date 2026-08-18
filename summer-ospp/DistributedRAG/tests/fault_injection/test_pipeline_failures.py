from dataclasses import replace

import pytest

from distributed_rag.config import AppConfig, RayConfig
from distributed_rag.domain.identifiers import make_chunk_id
from distributed_rag.domain.models import SourceLocator
from distributed_rag.ingestion.pipeline import IngestionPipeline


class RecordingRuntime:
    def __init__(self):
        self.cancelled = []

    def cancel(self, refs):
        self.cancelled.extend(refs)


@pytest.mark.failure_injection
def test_timeout_path_cancels_outstanding_ray_work(monkeypatch):
    runtime = RecordingRuntime()
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.config = AppConfig(ray=RayConfig(task_timeout_seconds=1))
    pipeline.runtime = runtime
    fake_ref = object()
    monkeypatch.setattr("distributed_rag.ingestion.pipeline.ray.wait", lambda *args, **kwargs: ([], [fake_ref]))

    with pytest.raises(TimeoutError):
        pipeline._drain_one({fake_ref: "ocr"}, [], [])
    assert runtime.cancelled == [fake_ref]


@pytest.mark.failure_injection
def test_retry_after_ambiguous_write_uses_the_same_primary_key():
    locator = SourceLocator(page_number=1, text_start=0, text_end=12)
    first_attempt = make_chunk_id("dv-1", locator, "chunk-v1", 0, "same content")
    retry_attempt = make_chunk_id("dv-1", locator, "chunk-v1", 0, "same content")
    simulated_milvus = {}
    simulated_milvus[first_attempt] = {"text": "same content"}
    simulated_milvus[retry_attempt] = {"text": "same content"}
    assert first_attempt == retry_attempt
    assert len(simulated_milvus) == 1
