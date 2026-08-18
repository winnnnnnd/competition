import os

import pytest

from rag_core.config import MilvusConfig, ModelConfig
from rag_core.vector_store import MilvusVectorStore


@pytest.mark.integration
def test_repeated_upsert_keeps_one_primary_key():
    if os.getenv("RUN_RAG_INTEGRATION_TESTS") != "1":
        pytest.skip("set RUN_RAG_INTEGRATION_TESTS=1")
    model = ModelConfig(embedding_dimension=4, embedding_version="integration-v1")
    store = MilvusVectorStore(MilvusConfig(collection="rag_integration_idempotency", index_type="FLAT"), model)
    chunk = {
        "chunk_id": "ch-stable",
        "document_id": "doc-1",
        "document_version": "dv-1",
        "text": "same chunk",
        "parent_element_ids": ["el-1"],
        "source_locator": {"page_number": 1},
        "chunking_version": "chunk-v1",
        "embedding_version": "integration-v1",
        "chunk_index": 0,
        "metadata": {},
    }
    record = {"chunk": chunk, "vector": [1.0, 0.0, 0.0, 0.0]}
    store.upsert([record])
    store.upsert([record])
    store.flush()
    assert len(store.collection.query('chunk_id == "ch-stable"', output_fields=["chunk_id"])) == 1
