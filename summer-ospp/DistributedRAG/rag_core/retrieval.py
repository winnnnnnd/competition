from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import AppConfig
from .models import Chunk, SearchHit
from .observability import RETRIEVAL_LATENCY
from .runtime import RayRuntime
from .state import JobStore
from .vector_store import MilvusVectorStore


HYDE_PROMPT = """根据问题生成一段可能出现在相关资料中的简短文本，仅用于辅助检索。不要声称它是真实答案。\n问题：{query}"""
REWRITE_PROMPT = """将下面的问题改写为更适合资料检索的一句话，保持原意，只输出改写结果。\n问题：{query}"""


class RetrievalEngine:
    def __init__(self, config: AppConfig, runtime: RayRuntime, jobs: JobStore):
        self.config = config
        self.runtime = runtime
        self.jobs = jobs
        self.store = MilvusVectorStore(config.milvus, config.models)

    def retrieve(self, query: str, document_ids: Optional[List[str]] = None, use_hyde: Optional[bool] = None) -> List[SearchHit]:
        started = time.perf_counter()
        try:
            versions = self.jobs.published_versions(document_ids)
            if not versions:
                return []
            original_vector = self._embed_query(query)
            routes: List[Tuple[str, List[Dict[str, Any]]]] = [
                ("query", self.store.search(original_vector, versions, self.config.retrieval.dense_top_n))
            ]
            should_hyde = self.config.retrieval.use_hyde if use_hyde is None else use_hyde
            if should_hyde:
                hypothetical = self.runtime.get(self.runtime.llm.generate.remote(HYDE_PROMPT.format(query=query), 256)).strip()
                if hypothetical:
                    routes.append(("hyde", self.store.search(self._embed_query(hypothetical), versions, self.config.retrieval.dense_top_n)))

            fused = self._rrf(routes)
            provisional = fused[: self.config.retrieval.reranker_top_n]
            provisional_scores = self.runtime.get(
                self.runtime.reranker.next().score.remote(query, [hit.chunk.text for hit in provisional])
            ) if provisional else []
            if self._low_confidence(fused, provisional_scores):
                for _ in range(self.config.retrieval.rewrite_attempts):
                    rewritten = self.runtime.get(self.runtime.llm.generate.remote(REWRITE_PROMPT.format(query=query), 128)).strip()
                    if not rewritten or rewritten == query:
                        break
                    routes.append(("rewrite", self.store.search(self._embed_query(rewritten), versions, self.config.retrieval.dense_top_n)))
                    fused = self._rrf(routes)
                    provisional = fused[: self.config.retrieval.reranker_top_n]
                    provisional_scores = self.runtime.get(
                        self.runtime.reranker.next().score.remote(query, [hit.chunk.text for hit in provisional])
                    ) if provisional else []
                    if not self._low_confidence(fused, provisional_scores):
                        break

            candidates = fused[: self.config.retrieval.reranker_top_n]
            if not candidates:
                return []
            scores = self.runtime.get(
                self.runtime.reranker.next().score.remote(query, [hit.chunk.text for hit in candidates])
            )
            reranked = [replace(hit, score=float(score)) for hit, score in zip(candidates, scores)]
            reranked.sort(key=lambda hit: hit.score, reverse=True)
            if self.config.retrieval.enable_mmr:
                return self._mmr(reranked, original_vector, self.config.retrieval.final_top_k)
            return reranked[: self.config.retrieval.final_top_k]
        finally:
            RETRIEVAL_LATENCY.observe(time.perf_counter() - started)

    def _embed_query(self, text: str) -> List[float]:
        result = self.runtime.get(self.runtime.embedding.next().embed.remote([text], True))
        if not result:
            raise ValueError("Query embedding returned no vectors")
        return result[0]

    def _rrf(self, routes: Sequence[Tuple[str, List[Dict[str, Any]]]]) -> List[SearchHit]:
        scores: Dict[str, float] = {}
        values: Dict[str, Dict[str, Any]] = {}
        names: Dict[str, List[str]] = {}
        for route_name, hits in routes:
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit["chunk"]["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.config.retrieval.rrf_k + rank)
                values.setdefault(chunk_id, hit)
                names.setdefault(chunk_id, []).append(route_name)
        output = [
            SearchHit(
                chunk=Chunk.from_dict(value["chunk"]),
                distance=float(value["distance"]),
                score=scores[chunk_id],
                vector=value.get("vector"),
                retrieval_routes=names[chunk_id],
            )
            for chunk_id, value in values.items()
        ]
        output.sort(key=lambda hit: hit.score, reverse=True)
        return output

    def _low_confidence(self, hits: List[SearchHit], reranker_scores: Sequence[float] = ()) -> bool:
        if len(hits) < 2:
            return True
        sample = hits[: min(5, len(hits))]
        mean_distance = sum(hit.distance for hit in sample) / len(sample)
        ordered = sorted(hit.distance for hit in sample)
        margin = ordered[1] - ordered[0]
        indicators = [
            mean_distance > self.config.retrieval.confidence_topk_mean_max,
            margin < self.config.retrieval.confidence_margin_min,
            not reranker_scores or max(reranker_scores) < self.config.retrieval.reranker_score_min,
        ]
        return sum(indicators) >= 2

    def _mmr(self, hits: List[SearchHit], query_vector: List[float], limit: int) -> List[SearchHit]:
        selected: List[SearchHit] = []
        remaining = list(hits)
        query = np.asarray(query_vector, dtype=np.float32)
        while remaining and len(selected) < limit:
            best: Optional[SearchHit] = None
            best_value = -float("inf")
            for candidate in remaining:
                vector = np.asarray(candidate.vector if candidate.vector is not None else [], dtype=np.float32)
                query_similarity = float(np.dot(query, vector)) if vector.size == query.size else candidate.score
                redundancy = 0.0
                if selected and vector.size:
                    similarities = [
                        float(np.dot(vector, np.asarray(item.vector, dtype=np.float32)))
                        for item in selected if item.vector is not None and len(item.vector) == len(vector)
                    ]
                    redundancy = max(similarities) if similarities else 0.0
                same_source_penalty = 0.05 if any(item.chunk.document_id == candidate.chunk.document_id for item in selected) else 0.0
                value = (
                    self.config.retrieval.mmr_lambda * query_similarity
                    - (1.0 - self.config.retrieval.mmr_lambda) * redundancy
                    - same_source_penalty
                )
                if value > best_value:
                    best_value = value
                    best = candidate
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        return selected
