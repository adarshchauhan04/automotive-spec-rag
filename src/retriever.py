"""
FAISS vector retrieval module.
Manages building, saving, loading, and querying dense vector indices
for fast similarity search across service manual chunks.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from src.chunker import DocumentChunk
from src.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with similarity score and ranking."""
    rank: int
    score: float
    chunk_id: str
    text: str
    page_number: int
    section: str
    subsections: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FAISSRetriever:
    """
    Vector retriever using FAISS IndexFlatIP (cosine similarity on L2-normalized embeddings).
    Persists index and metadata to disk for instant subsequent querying.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        index_dir: str | Path = "data",
        index_filename: str = "faiss_index.bin",
        metadata_filename: str = "chunks_metadata.json",
    ) -> None:
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / index_filename
        self.metadata_path = self.index_dir / metadata_filename

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    @property
    def is_indexed(self) -> bool:
        """Check if index is loaded and ready for queries."""
        return self.index is not None and len(self.chunks_metadata) > 0

    def build_index(self, chunks: List[DocumentChunk], batch_size: int = 64) -> None:
        """
        Builds a FAISS index from document chunks and saves to disk.

        Args:
            chunks: List of DocumentChunk objects.
            batch_size: Embedding batch size.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        # Embed the enriched contextual text for better semantic matching
        texts = [c.contextual_text for c in chunks]
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP (dim={dim}, count={len(chunks)})...")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks_metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "contextual_text": c.contextual_text,
                "page_number": c.page_number,
                "section": c.section,
                "subsections": c.subsections,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        self.save()
        logger.info(f"Index successfully built and saved to {self.index_dir}")

    def save(self) -> None:
        """Persists the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.embedder.model_name,
                    "dimension": self.embedder.dimension,
                    "total_chunks": len(self.chunks_metadata),
                    "chunks": self.chunks_metadata,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Saved index and metadata to {self.index_dir}")

    def load(self) -> bool:
        """
        Loads the FAISS index and metadata from disk if available.
        Returns True if loaded successfully, False if index files do not exist.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks_metadata = data.get("chunks", [])

        logger.info(
            f"Loaded index ({self.index.ntotal} vectors) and {len(self.chunks_metadata)} chunk records"
        )
        return True

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieves top-k most relevant chunks for a user query.

        Args:
            query: The user specification question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of RetrievalResult objects sorted by relevance score descending.
        """
        if not self.is_indexed:
            # Attempt to auto-load from disk
            if not self.load():
                raise RuntimeError(
                    f"Vector index not found in {self.index_dir}. "
                    "Run index build command first."
                )

        if not query.strip():
            raise ValueError("Query cannot be empty")

        top_k = min(top_k, len(self.chunks_metadata))
        if top_k <= 0:
            return []

        # Embed query vector
        query_vec = self.embedder.embed_query(query)
        query_mat = np.expand_dims(query_vec, axis=0)

        # Search index
        scores, indices = self.index.search(query_mat, top_k)

        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            meta = self.chunks_metadata[idx]
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(score),
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    page_number=meta["page_number"],
                    section=meta["section"],
                    subsections=meta.get("subsections", []),
                    metadata=meta.get("metadata", {}),
                )
            )

        return results
