from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from prometheus_client import Counter, Gauge, Histogram


STAGE_DURATION = Histogram("rag_stage_duration_seconds", "Pipeline stage duration", ["stage", "status"])
DOCUMENTS = Counter("rag_documents_total", "Documents processed", ["status"])
ELEMENTS = Counter("rag_elements_total", "Document elements produced", ["type"])
PAGES = Counter("rag_pages_total", "Distinct document pages processed")
CHUNKS = Counter("rag_chunks_total", "Chunks produced")
RETRIES = Counter("rag_retries_total", "Task retries", ["stage"])
TIMEOUTS = Counter("rag_timeouts_total", "Task timeouts", ["stage"])
DEGRADATIONS = Counter("rag_degradations_total", "Configured fallbacks", ["component"])
ACTOR_QUEUE = Gauge("rag_actor_queue_length", "Actor queue length", ["actor_type"])
BATCH_SIZE = Histogram("rag_batch_size", "Processed batch size", ["stage"])
EXTERNAL_LATENCY = Histogram("rag_external_latency_seconds", "External dependency latency", ["dependency", "operation"])
RETRIEVAL_LATENCY = Histogram("rag_retrieval_duration_seconds", "Retrieval latency")
CITATION_VALIDITY = Counter("rag_citations_total", "Citation validation outcomes", ["status"])
CLUSTER_RESOURCE_UTILIZATION = Gauge("rag_cluster_resource_utilization_ratio", "Ray logical resource utilization", ["resource"])
OBJECT_STORE_BYTES = Gauge("rag_object_store_bytes", "Ray object store memory", ["state"])
DEVICE_UTILIZATION = Gauge("rag_device_utilization_ratio", "Physical accelerator utilization", ["device", "index"])


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        value: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("trace_id", "job_id", "document_id", "stage"):
            candidate = getattr(record, key, None)
            if candidate:
                value[key] = candidate
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return json.dumps(value, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def log_context(logger: logging.Logger, **context: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, context)


def refresh_runtime_metrics() -> None:
    try:
        import ray

        total = ray.cluster_resources()
        available = ray.available_resources()
        for resource in ("CPU", "GPU", "NPU"):
            capacity = float(total.get(resource, 0.0))
            if capacity > 0:
                used = capacity - float(available.get(resource, 0.0))
                CLUSTER_RESOURCE_UTILIZATION.labels(resource=resource).set(max(0.0, min(1.0, used / capacity)))
        object_capacity = float(total.get("object_store_memory", 0.0))
        object_available = float(available.get("object_store_memory", 0.0))
        if object_capacity:
            OBJECT_STORE_BYTES.labels(state="capacity").set(object_capacity)
            OBJECT_STORE_BYTES.labels(state="used").set(max(0.0, object_capacity - object_available))
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            DEVICE_UTILIZATION.labels(device="gpu", index=str(index)).set(float(utilization.gpu) / 100.0)
    except Exception:
        pass


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    started = time.perf_counter()
    status = "succeeded"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        STAGE_DURATION.labels(stage=stage, status=status).observe(time.perf_counter() - started)
