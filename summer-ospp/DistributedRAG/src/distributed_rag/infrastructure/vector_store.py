from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional

from ..config import MilvusConfig, ModelConfig
from ..domain.models import Chunk
from .observability import EXTERNAL_LATENCY


class MilvusVectorStore:
    def __init__(self, milvus: MilvusConfig | Dict[str, Any], model: ModelConfig | Dict[str, Any]):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        self.config = MilvusConfig(**milvus) if isinstance(milvus, dict) else milvus
        self.model = ModelConfig(**model) if isinstance(model, dict) else model
        self.Collection = Collection
        self.CollectionSchema = CollectionSchema
        self.DataType = DataType
        self.FieldSchema = FieldSchema
        self.utility = utility
        connections.connect(alias="default", host=self.config.host, port=str(self.config.port))
        self.collection = self._ensure_collection()

    def _ensure_collection(self):
        if self.utility.has_collection(self.config.collection):
            collection = self.Collection(self.config.collection)
            vector_field = next(field for field in collection.schema.fields if field.name == "embedding")
            actual_dimension = int(vector_field.params["dim"])
            if actual_dimension != self.model.embedding_dimension:
                raise ValueError(
                    f"Collection dimension {actual_dimension} does not match configured dimension {self.model.embedding_dimension}"
                )
            return collection

        fields = [
            self.FieldSchema("chunk_id", self.DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            self.FieldSchema("document_id", self.DataType.VARCHAR, max_length=80),
            self.FieldSchema("document_version", self.DataType.VARCHAR, max_length=80),
            self.FieldSchema("text", self.DataType.VARCHAR, max_length=65535),
            self.FieldSchema("source_locator", self.DataType.JSON),
            self.FieldSchema("parent_element_ids", self.DataType.JSON),
            self.FieldSchema("chunking_version", self.DataType.VARCHAR, max_length=80),
            self.FieldSchema("embedding_version", self.DataType.VARCHAR, max_length=120),
            self.FieldSchema("chunk_index", self.DataType.INT64),
            self.FieldSchema("metadata", self.DataType.JSON),
            self.FieldSchema("embedding", self.DataType.FLOAT_VECTOR, dim=self.model.embedding_dimension),
        ]
        collection = self.Collection(
            name=self.config.collection,
            schema=self.CollectionSchema(fields, description="Versioned DistributedRAG chunks", enable_dynamic_field=False),
        )
        if self.config.index_type.upper() == "HNSW":
            params = {"M": 16, "efConstruction": 200}
        elif self.config.index_type.upper() == "FLAT":
            params = {}
        else:
            params = {"nlist": self.config.nlist}
        collection.create_index(
            "embedding",
            {"index_type": self.config.index_type, "metric_type": self.config.metric_type, "params": params},
        )
        return collection

    def upsert(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        entities = []
        for record in records:
            chunk = Chunk.from_dict(record["chunk"])
            entities.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
                "text": chunk.text,
                "source_locator": chunk.source_locator.as_dict(),
                "parent_element_ids": chunk.parent_element_ids,
                "chunking_version": chunk.chunking_version,
                "embedding_version": chunk.embedding_version,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "embedding": record["vector"],
            })
        started = time.perf_counter()
        self.collection.upsert(entities)
        EXTERNAL_LATENCY.labels(dependency="milvus", operation="upsert").observe(time.perf_counter() - started)
        return len(entities)

    def flush(self) -> None:
        self.collection.flush()

    def search(self, query_vector: List[float], published_versions: List[str], limit: int, include_vectors: bool = True) -> List[Dict[str, Any]]:
        if not published_versions:
            return []
        self.collection.load()
        escaped = [version.replace('"', '') for version in published_versions]
        version_expr = ",".join(json.dumps(version) for version in escaped)
        expr = f'document_version in [{version_expr}] and embedding_version == {json.dumps(self.model.embedding_version)}'
        params: Dict[str, Any]
        if self.config.index_type.upper() == "HNSW":
            params = {"metric_type": self.config.metric_type, "params": {"ef": max(64, limit)}}
        else:
            params = {"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}}
        output_fields = [
            "chunk_id", "document_id", "document_version", "text", "source_locator",
            "parent_element_ids", "chunking_version", "embedding_version", "chunk_index", "metadata",
        ]
        if include_vectors:
            output_fields.append("embedding")
        started = time.perf_counter()
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=params,
            limit=limit,
            expr=expr,
            output_fields=output_fields,
        )
        EXTERNAL_LATENCY.labels(dependency="milvus", operation="search").observe(time.perf_counter() - started)
        hits: List[Dict[str, Any]] = []
        if not results:
            return hits
        for hit in results[0]:
            entity = hit.entity
            chunk = {
                "chunk_id": entity.get("chunk_id") or hit.id,
                "document_id": entity.get("document_id"),
                "document_version": entity.get("document_version"),
                "text": entity.get("text"),
                "source_locator": entity.get("source_locator"),
                "parent_element_ids": entity.get("parent_element_ids"),
                "chunking_version": entity.get("chunking_version"),
                "embedding_version": entity.get("embedding_version"),
                "chunk_index": entity.get("chunk_index"),
                "metadata": entity.get("metadata") or {},
            }
            hits.append({"chunk": chunk, "distance": float(hit.distance), "vector": entity.get("embedding") if include_vectors else None})
        return hits

    def health(self) -> bool:
        return bool(self.utility.has_collection(self.config.collection))
