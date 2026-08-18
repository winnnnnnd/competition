from __future__ import annotations

from typing import Any, Dict, Iterator, List

import ray

from .chunking import chunk_element_batch
from .parsers import parse_document_batches


@ray.remote
def parse_document_task(document_value: Dict[str, Any], storage_value: Dict[str, Any], parsing_value: Dict[str, Any]) -> Iterator[List[Dict[str, Any]]]:
    yield from parse_document_batches(document_value, storage_value, parsing_value)


@ray.remote
def chunk_elements_task(element_values: List[Dict[str, Any]], chunking_value: Dict[str, Any], model_value: Dict[str, Any], index_start: int) -> List[Dict[str, Any]]:
    return chunk_element_batch(element_values, chunking_value, model_value, index_start)
