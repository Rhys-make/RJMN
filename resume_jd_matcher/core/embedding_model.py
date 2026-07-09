"""Pretrained semantic embedding model based on sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import config as project_config


EMBEDDING_MODEL_NAME = getattr(project_config, "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
EMBEDDING_MODEL_LOCAL_PATH = getattr(project_config, "EMBEDDING_MODEL_LOCAL_PATH", None)


def _usable_local_model_dir(path_value) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_dir():
        return None
    if (path / "modules.json").exists() or (path / "config.json").exists():
        return path
    return None


class EmbeddingModel:
    """Load and use BGE sentence embeddings with safe failure reporting."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.backend = "sentence-transformers"
        self.model = None
        self.error = ""
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            project_local_dir = _usable_local_model_dir(EMBEDDING_MODEL_LOCAL_PATH)
            if project_local_dir:
                self.model = SentenceTransformer(str(project_local_dir), local_files_only=True)
                self.load_mode = f"project_local:{project_local_dir}"
            else:
                try:
                    self.model = SentenceTransformer(self.model_name, local_files_only=True)
                    self.load_mode = "huggingface_cache"
                except Exception as local_exc:  # noqa: BLE001
                    self.model = SentenceTransformer(self.model_name)
                    self.load_mode = f"online_download; local_cache_miss={local_exc}"
            self.available = True
            self.error = ""
        except Exception as exc:  # noqa: BLE001 - expose model errors to UI without crashing.
            self.model = None
            self.available = False
            self.error = f"预训练语义模型 {self.model_name} 加载失败：{exc}"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not self.available or self.model is None:
            raise RuntimeError(self.error or "预训练语义模型不可用")
        cleaned = [str(text or "").strip() for text in texts]
        embeddings = self.model.encode(cleaned, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=float)

    def compute_cosine_similarity(self, a, b) -> np.ndarray:
        return cosine_similarity_matrix(a, b)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_name": self.model_name,
            "backend": self.backend,
            "load_mode": getattr(self, "load_mode", "unavailable"),
            "mode": "pretrained_embedding" if self.available else "model_unavailable",
            "error": self.error,
        }


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Cache the embedding model so Streamlit reruns do not reload it repeatedly."""
    return EmbeddingModel(EMBEDDING_MODEL_NAME)


def get_embedding_status() -> dict[str, Any]:
    return get_embedding_model().status()


def encode_texts(texts: list[str]) -> np.ndarray:
    return get_embedding_model().encode_texts(texts)


def cosine_similarity_matrix(a_embeddings, b_embeddings) -> np.ndarray:
    """Compute cosine similarity for two embedding matrices."""
    a = np.asarray(a_embeddings, dtype=float)
    b = np.asarray(b_embeddings, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T
