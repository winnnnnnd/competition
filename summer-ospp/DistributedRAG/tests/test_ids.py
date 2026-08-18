from rag_core.ids import make_chunk_id, make_document_id, make_document_version, sha256_bytes
from rag_core.models import SourceLocator


def test_document_version_is_stable_and_processing_version_sensitive():
    checksum = sha256_bytes(b"same document")
    first = make_document_version(checksum, "parser-v1", "chunk-v1", "embed-v1")
    assert first == make_document_version(checksum, "parser-v1", "chunk-v1", "embed-v1")
    assert first != make_document_version(checksum, "parser-v2", "chunk-v1", "embed-v1")


def test_chunk_id_is_deterministic_and_locator_sensitive():
    page_one = SourceLocator(page_number=1, text_start=0, text_end=10)
    page_two = SourceLocator(page_number=2, text_start=0, text_end=10)
    first = make_chunk_id("dv-1", page_one, "chunk-v1", 0, "hello")
    assert first == make_chunk_id("dv-1", page_one, "chunk-v1", 0, "hello")
    assert first != make_chunk_id("dv-1", page_two, "chunk-v1", 0, "hello")


def test_content_change_keeps_logical_document_id_but_changes_version():
    document_id = make_document_id("quarterly-report.pdf")
    first = make_document_version(sha256_bytes(b"v1"), "p1", "c1", "e1", document_id)
    second = make_document_version(sha256_bytes(b"v2"), "p1", "c1", "e1", document_id)
    assert document_id == make_document_id("quarterly-report.pdf")
    assert first != second
