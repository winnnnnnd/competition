from dataclasses import asdict

import pytest

from rag_core.actors import normalize_vectors
from rag_core.chunking import chunk_element_batch
from rag_core.config import ChunkingConfig, ModelConfig
from rag_core.models import DocumentElement, SourceLocator


def test_mrl_crop_happens_before_l2_normalization():
    vectors = normalize_vectors([[3.0, 4.0, 100.0]], dimension=2)
    assert vectors[0] == pytest.approx([0.6, 0.8])


def test_chunk_retains_parent_and_source_locator():
    element = DocumentElement(
        element_id="el-1",
        document_id="doc-1",
        document_version="dv-1",
        element_type="text",
        text="第一段内容。第二段内容。",
        source_locator=SourceLocator(page_number=7, bbox=[1, 2, 3, 4]),
    )
    chunks = chunk_element_batch(
        [element.as_dict()],
        asdict(ChunkingConfig(chunk_size=4, chunk_overlap=1)),
        asdict(ModelConfig()),
    )
    assert chunks
    assert all(chunk["parent_element_ids"] == ["el-1"] for chunk in chunks)
    assert all(chunk["source_locator"]["page_number"] == 7 for chunk in chunks)
    assert all(chunk["source_locator"]["bbox"] == [1, 2, 3, 4] for chunk in chunks)
