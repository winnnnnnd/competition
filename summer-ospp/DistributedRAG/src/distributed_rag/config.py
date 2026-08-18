from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StorageConfig:
    endpoint: str = "minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "rag-documents"
    secure: bool = False
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 60


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = "sqlite:///./volumes/rag_jobs.db"
    pool_size: int = 5
    stale_job_seconds: int = 900


@dataclass(frozen=True)
class RayConfig:
    address: str = "ray://127.0.0.1:10001"
    parser_cpus: float = 1.0
    chunk_cpus: float = 0.5
    actor_cpus: float = 1.0
    model_gpus: float = 0.0
    model_npus: float = 0.0
    ocr_actor_count: int = 1
    asr_actor_count: int = 1
    embedding_actor_count: int = 1
    reranker_actor_count: int = 1
    max_in_flight: int = 8
    max_retries: int = 2
    max_restarts: int = 1
    max_task_retries: int = 2
    task_timeout_seconds: int = 600
    ingestion_concurrency: int = 2
    cuda_visible_devices: Optional[str] = None
    ascend_visible_devices: Optional[str] = None

    def model_resources(self) -> Dict[str, float]:
        return {"NPU": self.model_npus} if self.model_npus > 0 else {}


@dataclass(frozen=True)
class ParsingConfig:
    parser_version: str = "parser-v2"
    element_batch_size: int = 16
    enable_ocr_fallback: bool = True
    enable_asr: bool = True
    ocr_language: str = "ch"
    whisper_model: str = "small"
    ocr_failure_mode: str = "fail"
    asr_failure_mode: str = "fail"


@dataclass(frozen=True)
class ChunkingConfig:
    version: str = "chunk-v2"
    chunk_size: int = 600
    chunk_overlap: int = 120
    batch_size: int = 64


@dataclass(frozen=True)
class ModelConfig:
    embedding_backend: str = "qwen_mindspore"
    embedding_model: str = "/app/.mindnlp/model/Qwen3-Embedding"
    embedding_version: str = "qwen3-embedding-v1-l2"
    embedding_dimension: int = 1024
    mrl_dimension: Optional[int] = None
    query_instruction: str = "检索能够回答该问题的相关文档："
    embedding_batch_size: int = 32
    embedding_max_tokens: int = 8192
    reranker_backend: str = "qwen_mindspore"
    reranker_model: str = "/app/.mindnlp/model/Qwen3-Reranker"
    llm_backend: str = "qwen_mindspore"
    llm_model: str = "/app/.mindnlp/model/Qwen2_5-1_5B-Instruct"
    allow_cpu_fallback: bool = True


@dataclass(frozen=True)
class MilvusConfig:
    host: str = "standalone"
    port: int = 19530
    collection: str = "distributed_rag_chunks"
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 1024
    nprobe: int = 16
    write_batch_size: int = 128


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_n: int = 30
    reranker_top_n: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    use_hyde: bool = True
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    confidence_topk_mean_max: float = 0.8
    confidence_margin_min: float = 0.03
    reranker_score_min: float = 0.0
    rewrite_attempts: int = 1


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: str = "INFO"
    metrics_port: int = 9108


@dataclass(frozen=True)
class AppConfig:
    profile: str = "accelerated"
    pipeline_version: str = "distributed-rag-v2"
    storage: StorageConfig = field(default_factory=StorageConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(profile: Optional[str] = None) -> AppConfig:
    selected_profile = profile or _env("RAG_PROFILE", "accelerated")
    is_cpu = selected_profile == "cpu"
    is_npu = selected_profile == "npu"
    default_backend = "mindnlp" if is_cpu else "qwen_mindspore"
    mrl_dimension = int(os.environ["MRL_DIMENSION"]) if os.getenv("MRL_DIMENSION") else None
    default_dimension = 768 if is_cpu else 1024
    return AppConfig(
        profile=selected_profile,
        pipeline_version=_env("PIPELINE_VERSION", "distributed-rag-v2"),
        storage=StorageConfig(
            endpoint=_env("MINIO_HOST", "minio:9000"),
            access_key=_env("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=_env("MINIO_SECRET_KEY", "minioadmin"),
            bucket=_env("MINIO_BUCKET_NAME", "rag-documents"),
            secure=_bool("MINIO_SECURE", False),
            connect_timeout_seconds=_int("MINIO_CONNECT_TIMEOUT", 5),
            read_timeout_seconds=_int("MINIO_READ_TIMEOUT", 60),
        ),
        database=DatabaseConfig(
            url=_env("DATABASE_URL", "sqlite:///./volumes/rag_jobs.db"),
            pool_size=_int("DATABASE_POOL_SIZE", 5),
            stale_job_seconds=_int("STALE_JOB_SECONDS", 900),
        ),
        ray=RayConfig(
            address=_env("RAY_ADDRESS", "ray://127.0.0.1:10001"),
            parser_cpus=_float("RAY_PARSER_CPUS", 1.0),
            chunk_cpus=_float("RAY_CHUNK_CPUS", 0.5),
            actor_cpus=_float("RAY_ACTOR_CPUS", 1.0),
            model_gpus=_float("RAY_MODEL_GPUS", 0.0 if is_cpu or is_npu else 1.0),
            model_npus=_float("RAY_MODEL_NPUS", 1.0 if is_npu else 0.0),
            ocr_actor_count=_int("OCR_ACTOR_COUNT", 1),
            asr_actor_count=_int("ASR_ACTOR_COUNT", 1),
            embedding_actor_count=_int("EMBEDDING_ACTOR_COUNT", 1),
            reranker_actor_count=_int("RERANKER_ACTOR_COUNT", 1),
            max_in_flight=_int("RAY_MAX_IN_FLIGHT", 8),
            max_retries=_int("RAY_MAX_RETRIES", 2),
            max_restarts=_int("RAY_MAX_RESTARTS", 1),
            max_task_retries=_int("RAY_MAX_TASK_RETRIES", 2),
            task_timeout_seconds=_int("TASK_TIMEOUT_SECONDS", 600),
            ingestion_concurrency=_int("INGESTION_CONCURRENCY", 2),
            cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
            ascend_visible_devices=os.getenv("ASCEND_RT_VISIBLE_DEVICES") or os.getenv("NPU_VISIBLE_DEVICES"),
        ),
        parsing=ParsingConfig(
            parser_version=_env("PARSER_VERSION", "parser-v2"),
            element_batch_size=_int("ELEMENT_BATCH_SIZE", 16),
            enable_ocr_fallback=_bool("ENABLE_OCR_FALLBACK", True),
            enable_asr=_bool("ENABLE_ASR", True),
            ocr_language=_env("OCR_LANGUAGE", "ch"),
            whisper_model=_env("WHISPER_MODEL", "small"),
            ocr_failure_mode=_env("OCR_FAILURE_MODE", "fail"),
            asr_failure_mode=_env("ASR_FAILURE_MODE", "fail"),
        ),
        chunking=ChunkingConfig(
            version=_env("CHUNKING_VERSION", "chunk-v2"),
            chunk_size=_int("CHUNK_SIZE", 600),
            chunk_overlap=_int("CHUNK_OVERLAP", 120),
            batch_size=_int("CHUNK_BATCH_SIZE", 64),
        ),
        models=ModelConfig(
            embedding_backend=_env("EMBEDDING_BACKEND", default_backend),
            embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5" if is_cpu else "/app/.mindnlp/model/Qwen3-Embedding"),
            embedding_version=_env("EMBEDDING_VERSION", "bge-base-zh-v1.5-l2" if is_cpu else "qwen3-embedding-v1-l2"),
            embedding_dimension=mrl_dimension or _int("EMBEDDING_DIMENSION", default_dimension),
            mrl_dimension=mrl_dimension,
            query_instruction=_env("QUERY_INSTRUCTION", "检索能够回答该问题的相关文档："),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 32),
            embedding_max_tokens=_int("EMBEDDING_MAX_TOKENS", 8192),
            reranker_backend=_env("RERANKER_BACKEND", "none" if is_cpu else "qwen_mindspore"),
            reranker_model=_env("RERANKER_MODEL", "/app/.mindnlp/model/Qwen3-Reranker"),
            llm_backend=_env("LLM_BACKEND", default_backend),
            llm_model=_env("LLM_MODEL", "openbmb/MiniCPM-2B-dpo-bf16" if is_cpu else "/app/.mindnlp/model/Qwen2_5-1_5B-Instruct"),
            allow_cpu_fallback=_bool("ALLOW_CPU_MODEL_FALLBACK", True),
        ),
        milvus=MilvusConfig(
            host=_env("MILVUS_HOST", "standalone"),
            port=_int("MILVUS_PORT", 19530),
            collection=_env("MILVUS_COLLECTION", "distributed_rag_chunks"),
            index_type=_env("MILVUS_INDEX_TYPE", "IVF_FLAT"),
            metric_type=_env("MILVUS_METRIC_TYPE", "L2"),
            nlist=_int("MILVUS_NLIST", 1024),
            nprobe=_int("MILVUS_NPROBE", 16),
            write_batch_size=_int("MILVUS_WRITE_BATCH_SIZE", 128),
        ),
        retrieval=RetrievalConfig(
            dense_top_n=_int("DENSE_TOP_N", 30),
            reranker_top_n=_int("RERANKER_TOP_N", 20),
            final_top_k=_int("FINAL_TOP_K", 5),
            rrf_k=_int("RRF_K", 60),
            use_hyde=_bool("USE_HYDE", True),
            enable_mmr=_bool("ENABLE_MMR", True),
            mmr_lambda=_float("MMR_LAMBDA", 0.7),
            confidence_topk_mean_max=_float("CONFIDENCE_TOPK_MEAN_MAX", 0.8),
            confidence_margin_min=_float("CONFIDENCE_MARGIN_MIN", 0.03),
            reranker_score_min=_float("RERANKER_SCORE_MIN", 0.0),
            rewrite_attempts=_int("REWRITE_ATTEMPTS", 1),
        ),
        observability=ObservabilityConfig(
            log_level=_env("LOG_LEVEL", "INFO"),
            metrics_port=_int("METRICS_PORT", 9108),
        ),
    )
