"""
Embeddings generation module supporting local SentenceTransformers
and remote OpenAI embeddings with L2 normalization.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for text embedding generators."""

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings and return a 2D float32 numpy array."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return a 1D float32 numpy array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the generated vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the embedding model."""
        pass


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding generator using HuggingFace sentence-transformers.
    Runs 100% offline without requiring external API credentials.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        logger.info(f"Loading local SentenceTransformer model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{model_name}': {e}. "
                "Ensure sentence-transformers and PyTorch are installed."
            ) from e

        # Determine embedding dimension
        dummy = self.model.encode(["dimension probe"], normalize_embeddings=True)
        self._dim = dummy.shape[1]
        logger.info(f"Model {model_name} initialized. Embedding dimension: {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Always normalize embeddings to enable fast cosine similarity with inner product
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query string cannot be empty")
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return embedding.astype(np.float32)


class OpenAIEmbedder(BaseEmbedder):
    """
    Remote embedding generator using the official OpenAI API.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAIEmbedder requires an API key. Set OPENAI_API_KEY in your environment or .env."
            )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

        # Determine dimension with a probe call
        resp = self.client.embeddings.create(input=["dimension probe"], model=self._model_name)
        self._dim = len(resp.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(input=batch, model=self._model_name)
            for item in resp.data:
                vec = np.array(item.embedding, dtype=np.float32)
                # L2 normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                all_vectors.append(vec)

        return np.vstack(all_vectors)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query string cannot be empty")
        resp = self.client.embeddings.create(input=[query], model=self._model_name)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def get_embedder(
    provider: str = "local",
    model_name: Optional[str] = None,
) -> BaseEmbedder:
    """
    Factory to instantiate the appropriate embedder.
    Defaults to the local, offline-capable SentenceTransformer ('all-MiniLM-L6-v2').
    """
    if provider.lower() == "openai":
        return OpenAIEmbedder(model_name=model_name or "text-embedding-3-small")
    elif provider.lower() in ("local", "sentence_transformers", "huggingface"):
        return LocalSentenceTransformerEmbedder(model_name=model_name or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedder provider: '{provider}'. Choose 'local' or 'openai'.")
