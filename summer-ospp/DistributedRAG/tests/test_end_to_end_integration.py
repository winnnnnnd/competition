import os

import pytest

from rag_core.service import DistributedRAGService


@pytest.mark.integration
def test_minio_to_milvus_to_cited_answer():
    if os.getenv("RUN_RAG_INTEGRATION_TESTS") != "1":
        pytest.skip("set RUN_RAG_INTEGRATION_TESTS=1")
    service = DistributedRAGService(profile=os.getenv("RAG_PROFILE", "cpu"))
    job = service.ingest("integration.txt", "幂等写入使用确定性 chunk_id 和 Milvus upsert。".encode("utf-8"))
    assert job["status"] == "succeeded"
    repeated = service.ingest("integration.txt", "幂等写入使用确定性 chunk_id 和 Milvus upsert。".encode("utf-8"))
    assert repeated["job_id"] == job["job_id"]
    response = service.ask("系统如何避免重复向量？", document_ids=[job["document_id"]], use_hyde=False)
    assert response["citations"]
    assert all(citation["source_id"].startswith("S") for citation in response["citations"])
