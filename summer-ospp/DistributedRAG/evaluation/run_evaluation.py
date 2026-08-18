from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag_core.service import DistributedRAGService


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dcg(relevance: List[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def run(dataset: Path, profile: str) -> Dict[str, Any]:
    service = DistributedRAGService(profile=profile)
    cases = load_cases(dataset)
    rows: List[Dict[str, float]] = []
    latencies: List[float] = []
    for case in cases:
        started = time.perf_counter()
        hits = service.retrieval.retrieve(case["query"], document_ids=case.get("document_ids"))
        response = service.citations.answer(case["query"], hits, f"eval-{len(rows)}")
        latencies.append(time.perf_counter() - started)
        retrieved = [hit.chunk.chunk_id for hit in hits]
        relevant = set(case.get("relevant_chunk_ids", []))
        binary = [int(chunk_id in relevant) for chunk_id in retrieved[:5]]
        recall = len(set(retrieved[:5]).intersection(relevant)) / max(1, len(relevant))
        reciprocal_rank = next((1.0 / (index + 1) for index, value in enumerate(binary) if value), 0.0)
        ideal = sorted(binary, reverse=True)
        ndcg = dcg(binary) / dcg(ideal) if dcg(ideal) else 0.0
        keywords = case.get("answer_keywords", [])
        answer_accuracy = sum(keyword in response.answer for keyword in keywords) / max(1, len(keywords))
        legal_ids = {f"S{index}" for index in range(1, len(hits) + 1)}
        citation_legal = all(citation.source_id in legal_ids for citation in response.citations)
        citation_correct = sum(citation.chunk_id in relevant for citation in response.citations) / max(1, len(response.citations))
        expected_fields = case.get("expected_source_fields", [])
        citation_coverage = sum(
            all(citation.source_locator.as_dict().get(field) is not None for field in expected_fields)
            for citation in response.citations
        ) / max(1, len(response.citations))
        rows.append({
            "recall_at_5": recall,
            "ndcg_at_5": ndcg,
            "mrr": reciprocal_rank,
            "answer_accuracy": answer_accuracy,
            "faithfulness": float(response.evidence_sufficient),
            "citation_legality": float(citation_legal),
            "citation_correctness": citation_correct,
            "citation_coverage": citation_coverage,
        })
    metric_names = rows[0].keys() if rows else []
    return {
        "case_count": len(rows),
        "metrics": {name: statistics.fmean(row[name] for row in rows) for name in metric_names},
        "latency_seconds": {"p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--profile", choices=["cpu", "accelerated", "npu"], default="accelerated")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    args = parser.parse_args()
    result = run(args.dataset, args.profile)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
