from __future__ import annotations

from dataclasses import asdict
from threading import Lock
from typing import Any, List

import ray

from .actors import ASRActor, EmbeddingActor, LLMActor, MilvusWriterActor, OCRActor, RerankerActor
from .config import AppConfig
from .observability import TIMEOUTS


class ActorPool:
    def __init__(self, actors: List[Any]):
        if not actors:
            raise ValueError("Actor pool cannot be empty")
        self.actors = actors
        self.index = 0
        self.lock = Lock()

    def next(self) -> Any:
        with self.lock:
            actor = self.actors[self.index % len(self.actors)]
            self.index += 1
            return actor


class RayRuntime:
    def __init__(self, config: AppConfig):
        self.config = config
        if not ray.is_initialized():
            ray.init(address=config.ray.address, ignore_reinit_error=True, namespace="distributed-rag")
        model_resources = config.ray.model_resources()
        device_environment = {}
        if config.ray.cuda_visible_devices:
            device_environment["CUDA_VISIBLE_DEVICES"] = config.ray.cuda_visible_devices
        if config.ray.ascend_visible_devices:
            device_environment["ASCEND_RT_VISIBLE_DEVICES"] = config.ray.ascend_visible_devices
        model_options = {
            "num_cpus": config.ray.actor_cpus,
            "num_gpus": config.ray.model_gpus,
            "resources": model_resources,
            "max_restarts": config.ray.max_restarts,
            "max_task_retries": config.ray.max_task_retries,
        }
        if device_environment:
            model_options["runtime_env"] = {"env_vars": device_environment}
        self.embedding = ActorPool([
            EmbeddingActor.options(
                name=f"rag-embedding-{config.profile}-{index}", get_if_exists=True, **model_options
            ).remote(asdict(config.models), config.profile)
            for index in range(config.ray.embedding_actor_count)
        ])
        self.reranker = ActorPool([
            RerankerActor.options(
                name=f"rag-reranker-{config.profile}-{index}", get_if_exists=True, **model_options
            ).remote(asdict(config.models), config.profile)
            for index in range(config.ray.reranker_actor_count)
        ])
        self.llm = LLMActor.options(
            name=f"rag-llm-{config.profile}", get_if_exists=True, **model_options
        ).remote(asdict(config.models), config.profile)
        self.ocr = ActorPool([
            OCRActor.options(
                name=f"rag-ocr-{config.profile}-{index}", get_if_exists=True,
                num_cpus=config.ray.actor_cpus, max_restarts=config.ray.max_restarts,
                max_task_retries=config.ray.max_task_retries,
            ).remote(asdict(config.parsing))
            for index in range(config.ray.ocr_actor_count)
        ])
        self.asr = ActorPool([
            ASRActor.options(
                name=f"rag-asr-{config.profile}-{index}", get_if_exists=True, **model_options
            ).remote(asdict(config.parsing))
            for index in range(config.ray.asr_actor_count)
        ]) if config.parsing.enable_asr else None
        self.writer = MilvusWriterActor.options(
            name=f"rag-milvus-writer-{config.profile}-{config.models.embedding_version}", get_if_exists=True,
            num_cpus=config.ray.actor_cpus, max_restarts=config.ray.max_restarts,
            max_task_retries=config.ray.max_task_retries,
        ).remote(asdict(config.milvus), asdict(config.models))

    def get(self, object_ref: Any) -> Any:
        try:
            return ray.get(object_ref, timeout=self.config.ray.task_timeout_seconds)
        except ray.exceptions.GetTimeoutError:
            ray.cancel(object_ref, force=True, recursive=True)
            TIMEOUTS.labels(stage="ray_task").inc()
            raise TimeoutError(f"Ray task exceeded {self.config.ray.task_timeout_seconds} seconds")

    @staticmethod
    def cancel(refs: List[Any]) -> None:
        for ref in refs:
            try:
                ray.cancel(ref, force=True, recursive=True)
            except Exception:
                pass
