from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePath
from typing import Any, Dict

from .models import FileType, SourceLocator


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_document_id(source_identity: str) -> str:
    """Create a stable logical document ID independent of content changes."""
    return f"doc_{stable_hash(source_identity.strip().lower())[:32]}"


def make_document_version(content_checksum: str, parser_version: str, chunking_version: str, embedding_version: str, document_id: str = "") -> str:
    return f"dv_{stable_hash(document_id, content_checksum, parser_version, chunking_version, embedding_version)[:40]}"


def make_element_id(document_version: str, element_type: str, locator: SourceLocator, ordinal: int) -> str:
    return f"el_{stable_hash(document_version, element_type, locator.as_dict(), ordinal)[:40]}"


def make_chunk_id(document_version: str, locator: SourceLocator, chunking_version: str, chunk_index: int, text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return f"ch_{stable_hash(document_version, locator.as_dict(), chunking_version, chunk_index, normalized)[:40]}"


def safe_file_name(name: str) -> str:
    base = PurePath(name).name
    return re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]", "_", base)[:180] or "document.bin"


def detect_file_type(file_name: str) -> FileType:
    suffix = PurePath(file_name).suffix.lower()
    if suffix == ".pdf":
        return FileType.PDF
    if suffix in {".doc", ".docx"}:
        return FileType.WORD
    if suffix in {".ppt", ".pptx"}:
        return FileType.POWERPOINT
    if suffix in {".xls", ".xlsx", ".csv"}:
        return FileType.EXCEL
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return FileType.IMAGE
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        return FileType.AUDIO
    if suffix in {".md", ".markdown"}:
        return FileType.MARKDOWN
    if suffix in {".txt", ".html", ".htm"}:
        return FileType.TEXT
    return FileType.UNKNOWN


def deterministic_object_key(document_id: str, document_version: str, category: str, name: str) -> str:
    return f"documents/{document_id}/{document_version}/{category}/{safe_file_name(name)}"
