from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from rag_core.service import DistributedRAGService


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))] if ordered else 0.0


def benchmark(document_dir: Path, profile: str, concurrency: int) -> Dict[str, Any]:
    service = DistributedRAGService(profile=profile)
    documents = [path for path in document_dir.rglob("*") if path.is_file()]

    def ingest(path: Path):
        started = time.perf_counter()
        result = service.ingest(path.name, path.read_bytes(), {"logical_document_id": str(path.relative_to(document_dir))})
        return time.perf_counter() - started, result

    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(ingest, path) for path in documents]
        for future in as_completed(futures):
            rows.append(future.result())
    elapsed = time.perf_counter() - started
    latencies = [row[0] for row in rows]
    return {
        "documents": len(rows),
        "elapsed_seconds": elapsed,
        "documents_per_minute": len(rows) * 60.0 / max(elapsed, 1e-9),
        "chunks": sum(row[1].get("chunk_count", 0) for row in rows),
        "latency_seconds": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "job_ids": [row[1]["job_id"] for row in rows],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("document_dir", type=Path)
    parser.add_argument("--profile", choices=["cpu", "accelerated", "npu"], default="accelerated")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("ingestion-benchmark.json"))
    args = parser.parse_args()
    result = benchmark(args.document_dir, args.profile, args.concurrency)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
