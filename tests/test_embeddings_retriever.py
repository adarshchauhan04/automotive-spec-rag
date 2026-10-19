"""
Unit tests for embeddings and FAISSRetriever.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.chunker import DocumentChunk
from src.embeddings import get_embedder
from src.retriever import FAISSRetriever


@pytest.fixture(scope="module")
def local_embedder():
    return get_embedder("local", model_name="all-MiniLM-L6-v2")


def test_embedder_dimension_and_normalization(local_embedder):
    assert local_embedder.dimension == 384
    vec = local_embedder.embed_query("brake caliper bolt torque")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Normalized vector has norm ~= 1.0
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_retriever_index_build_save_load(local_embedder):
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="Brake caliper flow bolt torque is 35 Nm (26 lb-ft).",
            page_number=636,
            section="SECTION 206-03: Front Disc Brake",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            text="Rear axle differential lubricant fill capacity is 2.6 L (5.5 pt).",
            page_number=258,
            section="SECTION 205-02A: Rear Drive Axle",
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            text="Tire inflation pressure is 35 psi for cold front and rear tires.",
            page_number=145,
            section="SECTION 204-04: Wheels and Tires",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        retriever = FAISSRetriever(embedder=local_embedder, index_dir=tmp_dir)
        retriever.build_index(chunks)

        assert (Path(tmp_dir) / "faiss_index.bin").exists()
        assert (Path(tmp_dir) / "chunks_metadata.json").exists()

        # Query the newly built index
        results = retriever.retrieve("torque for brake caliper bolts", top_k=2)
        assert len(results) == 2
        top = results[0]
        assert top.page_number == 636
        assert "Brake caliper" in top.text
        assert top.score > 0.4

        # Test reload in a fresh retriever instance
        loaded_retriever = FAISSRetriever(embedder=local_embedder, index_dir=tmp_dir)
        assert loaded_retriever.load() is True
        assert loaded_retriever.is_indexed is True

        reloaded_results = loaded_retriever.retrieve("torque for brake caliper bolts", top_k=1)
        assert reloaded_results[0].chunk_id == "chunk_1"
        assert reloaded_results[0].page_number == 636
