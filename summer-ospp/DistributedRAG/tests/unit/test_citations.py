from distributed_rag.domain.models import Chunk, SearchHit, SourceLocator
from distributed_rag.retrieval.citations import CitationMap, CitationService


def _hit(chunk_id: str, version: str = "dv-1") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            document_version=version,
            text="项目使用确定性主键避免重复向量。",
            parent_element_ids=["el-1"],
            source_locator=SourceLocator(page_number=3),
            chunking_version="chunk-v1",
            embedding_version="embed-v1",
            chunk_index=0,
        ),
        distance=0.1,
    )


def test_citation_whitelist_rejects_unknown_source_id():
    service = CitationService(runtime=None)
    mapping = CitationMap.from_hits([_hit("ch-1")])
    citations, invalid = service._validate([{"source_id": "S999", "claim": "伪造引用"}], mapping)
    assert citations == []
    assert invalid is True


def test_citation_resolves_to_server_owned_chunk_and_locator():
    service = CitationService(runtime=None)
    mapping = CitationMap.from_hits([_hit("ch-1")])
    citations, invalid = service._validate([{"source_id": "S1", "claim": "确定性主键"}], mapping)
    assert invalid is False
    assert citations[0].chunk_id == "ch-1"
    assert citations[0].source_locator.page_number == 3
