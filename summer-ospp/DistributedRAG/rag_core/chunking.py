from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

import tiktoken

from .config import ChunkingConfig, ModelConfig
from .ids import make_chunk_id
from .models import Chunk, DocumentElement, SourceLocator


def chunk_element_batch(element_values: List[Dict[str, Any]], chunking_value: Dict[str, Any], model_value: Dict[str, Any], index_start: int = 0) -> List[Dict[str, Any]]:
    config = ChunkingConfig(**chunking_value)
    model = ModelConfig(**model_value)
    encoding = tiktoken.get_encoding("cl100k_base")
    output: List[Dict[str, Any]] = []
    index = index_start
    for value in element_values:
        element = DocumentElement.from_dict(value)
        if not element.text.strip():
            continue
        token_ids = encoding.encode(element.text)
        start = 0
        local_index = 0
        while start < len(token_ids):
            end = min(start + config.chunk_size, len(token_ids))
            text = encoding.decode(token_ids[start:end]).strip()
            if text:
                text_start = element.text.find(text[: min(40, len(text))])
                if text_start < 0:
                    text_start = 0
                base_start = element.source_locator.text_start or 0
                locator = replace(
                    element.source_locator,
                    text_start=base_start + text_start,
                    text_end=base_start + text_start + len(text),
                )
                chunk = Chunk(
                    chunk_id=make_chunk_id(element.document_version, locator, config.version, local_index, text),
                    document_id=element.document_id,
                    document_version=element.document_version,
                    text=text,
                    parent_element_ids=[element.element_id],
                    source_locator=locator,
                    chunking_version=config.version,
                    embedding_version=model.embedding_version,
                    chunk_index=index,
                    metadata={"element_type": element.element_type, **element.metadata},
                )
                output.append(chunk.as_dict())
                index += 1
                local_index += 1
            if end == len(token_ids):
                break
            start = max(start + 1, end - config.chunk_overlap)
    return output
