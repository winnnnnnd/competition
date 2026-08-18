from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import ray

from .config import ModelConfig, ParsingConfig, StorageConfig
from .ids import stable_hash
from .models import DocumentElement, ElementType, SourceLocator
from .storage import ObjectStorage
from .vector_store import MilvusVectorStore


def _device(profile: str) -> str:
    if profile == "npu":
        return "Ascend"
    if profile == "cpu":
        return "CPU"
    return "GPU"


def normalize_vectors(vectors: Sequence[Sequence[float]], dimension: int | None = None) -> List[List[float]]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")
    if dimension:
        array = array[:, :dimension]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, 1e-12)).tolist()


@ray.remote
class EmbeddingActor:
    def __init__(self, model_value: Dict[str, Any], profile: str):
        self.config = ModelConfig(**model_value)
        self.backend = self.config.embedding_backend
        if self.backend == "qwen_mindspore":
            from main_app2.qwen_embedding_model import QwenEmbeddingModel

            try:
                self.model = QwenEmbeddingModel(self.config.embedding_model, device=_device(profile))
            except Exception:
                if not self.config.allow_cpu_fallback or profile == "cpu":
                    raise
                self.model = QwenEmbeddingModel(self.config.embedding_model, device="CPU")
        elif self.backend == "mindnlp":
            from mindnlp.sentence import SentenceTransformer

            self.model = SentenceTransformer(self.config.embedding_model)
        else:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            texts = [f"{self.config.query_instruction}{text}" for text in texts]
        if self.backend == "qwen_mindspore":
            vectors = self.model.encode(texts)
        else:
            vectors = self.model.encode(texts, normalize_embeddings=False)
        return normalize_vectors(vectors, self.config.mrl_dimension)


@ray.remote
class RerankerActor:
    def __init__(self, model_value: Dict[str, Any], profile: str):
        self.config = ModelConfig(**model_value)
        self.backend = self.config.reranker_backend
        self.model = None
        if self.backend == "qwen_mindspore":
            from main_app2.qwen_reranker_model import QwenRerankerModel

            try:
                self.model = QwenRerankerModel(self.config.reranker_model, device=_device(profile))
            except Exception:
                if not self.config.allow_cpu_fallback or profile == "cpu":
                    raise
                self.model = QwenRerankerModel(self.config.reranker_model, device="CPU")

    def score(self, query: str, documents: List[str]) -> List[float]:
        if self.model is not None:
            return self.model.compute_score([(query, document) for document in documents])
        query_terms = set(query.lower().split())
        return [len(query_terms.intersection(document.lower().split())) / max(1, len(query_terms)) for document in documents]


@ray.remote
class LLMActor:
    def __init__(self, model_value: Dict[str, Any], profile: str):
        self.config = ModelConfig(**model_value)
        self.backend = self.config.llm_backend
        if self.backend == "qwen_mindspore":
            from main_app2.qwen_causal_lm import QwenCausalLM

            try:
                self.model = QwenCausalLM(self.config.llm_model, device=_device(profile))
            except Exception:
                if not self.config.allow_cpu_fallback or profile == "cpu":
                    raise
                self.model = QwenCausalLM(self.config.llm_model, device="CPU")
            self.tokenizer = None
        elif self.backend == "mindnlp":
            import mindspore
            from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model, mirror="modelscope")
            self.model = AutoModelForCausalLM.from_pretrained(self.config.llm_model, ms_dtype=mindspore.float32, mirror="modelscope")
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if self.backend == "qwen_mindspore":
            return self.model.generate([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens)
        response, _ = self.model.chat(self.tokenizer, prompt, history=[], max_length=max_new_tokens)
        return response


@ray.remote
class OCRActor:
    def __init__(self, parsing_value: Dict[str, Any]):
        from rapidocr_onnxruntime import RapidOCR

        self.config = ParsingConfig(**parsing_value)
        self.engine = RapidOCR()

    def recognize(self, element_value: Dict[str, Any], storage_value: Dict[str, Any]) -> List[Dict[str, Any]]:
        from PIL import Image

        element = DocumentElement.from_dict(element_value)
        storage = ObjectStorage(StorageConfig(**storage_value))
        image = np.asarray(Image.open(__import__("io").BytesIO(storage.get_bytes(element.metadata["ocr_uri"]))).convert("RGB"))
        result, _ = self.engine(image)
        output: List[Dict[str, Any]] = []
        for index, line in enumerate(result or []):
            bbox, text, confidence = line
            flat_bbox = [float(value) for point in bbox for value in point]
            locator = SourceLocator(**{**element.source_locator.as_dict(), "bbox": flat_bbox})
            value = DocumentElement(
                element_id=f"el_{stable_hash(element.element_id, index, text)[:40]}",
                document_id=element.document_id,
                document_version=element.document_version,
                element_type=ElementType.OCR_TEXT.value,
                text=text,
                source_locator=locator,
                parent_id=element.element_id,
                metadata={
                    **{key: value for key, value in element.metadata.items() if key not in {"requires_ocr", "ocr_uri"}},
                    "ocr_confidence": float(confidence),
                },
            )
            output.append(value.as_dict())
        return output


@ray.remote
class ASRActor:
    def __init__(self, parsing_value: Dict[str, Any]):
        import whisper

        self.config = ParsingConfig(**parsing_value)
        self.model = whisper.load_model(self.config.whisper_model)

    def transcribe(self, element_value: Dict[str, Any], storage_value: Dict[str, Any], suffix: str = ".wav") -> List[Dict[str, Any]]:
        element = DocumentElement.from_dict(element_value)
        storage = ObjectStorage(StorageConfig(**storage_value))
        with tempfile.NamedTemporaryFile(suffix=suffix) as target:
            target.write(storage.get_bytes(element.metadata["asr_uri"]))
            target.flush()
            result = self.model.transcribe(target.name, fp16=False)
        output: List[Dict[str, Any]] = []
        for index, segment in enumerate(result.get("segments", [])):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            locator = SourceLocator(
                audio_start_ms=int(float(segment.get("start", 0)) * 1000),
                audio_end_ms=int(float(segment.get("end", 0)) * 1000),
            )
            output.append(DocumentElement(
                element_id=f"el_{stable_hash(element.element_id, index, text)[:40]}",
                document_id=element.document_id,
                document_version=element.document_version,
                element_type=ElementType.ASR_SEGMENT.value,
                text=text,
                source_locator=locator,
                parent_id=element.element_id,
                metadata={
                    **{key: value for key, value in element.metadata.items() if key not in {"requires_asr", "asr_uri"}},
                    "asr_segment_id": segment.get("id", index),
                },
            ).as_dict())
        return output


@ray.remote
class MilvusWriterActor:
    def __init__(self, milvus_value: Dict[str, Any], model_value: Dict[str, Any]):
        self.store = MilvusVectorStore(milvus_value, model_value)

    def upsert(self, records: List[Dict[str, Any]]) -> int:
        return self.store.upsert(records)

    def flush(self) -> None:
        self.store.flush()

    def health(self) -> bool:
        return self.store.health()
