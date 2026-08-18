"""MindNLP model adapters used by the CPU runtime profile."""

from __future__ import annotations

from typing import List


class MindNLPEmbeddingBackend:
    def __init__(self, model_name_or_path: str):
        from mindnlp.sentence import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=False)


class MindNLPLLMBackend:
    def __init__(self, model_name_or_path: str):
        import mindspore
        from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, mirror="modelscope")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            ms_dtype=mindspore.float32,
            mirror="modelscope",
        )

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=max_new_tokens,
        )
        return response
