"""Compatibility exports for the original module path."""

from rag_core.actors import EmbeddingActor, LLMActor
from rag_core.ray_tasks import parse_document_task as parse_and_chunk_document

__all__ = ["EmbeddingActor", "LLMActor", "parse_and_chunk_document"]
