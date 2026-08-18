from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileType(str, Enum):
    PDF = "pdf"
    WORD = "word"
    POWERPOINT = "powerpoint"
    EXCEL = "excel"
    IMAGE = "image"
    AUDIO = "audio"
    MARKDOWN = "markdown"
    TEXT = "text"
    WEB = "web"
    UNKNOWN = "unknown"


class ElementType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    OCR_TEXT = "ocr_text"
    ASR_SEGMENT = "asr_segment"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class StageName(str, Enum):
    UPLOAD = "upload"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    PUBLISH = "publish"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SourceLocator:
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    paragraph_number: Optional[int] = None
    heading_path: List[str] = field(default_factory=list)
    audio_start_ms: Optional[int] = None
    audio_end_ms: Optional[int] = None
    bbox: Optional[List[float]] = None
    text_start: Optional[int] = None
    text_end: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Document:
    document_id: str
    document_version: str
    content_checksum: str
    source_uri: str
    file_name: str
    file_type: str
    parser_version: str
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentElement:
    element_id: str
    document_id: str
    document_version: str
    element_type: str
    text: str
    source_locator: SourceLocator
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_locator"] = self.source_locator.as_dict()
        return data

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DocumentElement":
        data = dict(value)
        data["source_locator"] = SourceLocator(**data["source_locator"])
        return cls(**data)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_version: str
    text: str
    parent_element_ids: List[str]
    source_locator: SourceLocator
    chunking_version: str
    embedding_version: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_locator"] = self.source_locator.as_dict()
        return data

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Chunk":
        data = dict(value)
        data["source_locator"] = SourceLocator(**data["source_locator"])
        return cls(**data)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    distance: float
    score: float = 0.0
    vector: Optional[List[float]] = None
    retrieval_routes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    source_id: str
    chunk_id: str
    document_id: str
    document_version: str
    source_locator: SourceLocator
    claim: str = ""
    source_name: Optional[str] = None
    source_url: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_locator"] = self.source_locator.as_dict()
        return data


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    citations: List[Citation]
    trace_id: str
    evidence_sufficient: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [citation.as_dict() for citation in self.citations],
            "trace_id": self.trace_id,
            "evidence_sufficient": self.evidence_sufficient,
            "metadata": self.metadata,
        }
