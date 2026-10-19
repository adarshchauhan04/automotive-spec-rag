"""
Integration tests for SpecificationPipeline and CLI commands.
"""

from pathlib import Path
import tempfile
import pytest

from src.pipeline import SpecificationPipeline


def test_pipeline_build_and_query_flow():
    manual_path = Path("sample-service-manual 1.pdf")
    if not manual_path.exists():
        pytest.skip("sample-service-manual 1.pdf not found in workspace")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = SpecificationPipeline(
            index_dir=tmp_dir,
            embedder_provider="local",
            embedding_model="all-MiniLM-L6-v2",
        )

        # Build index on first 5 pages
        pipeline.build_index(
            pdf_path=manual_path,
            chunk_size=900,
            chunk_overlap=100,
            max_pages=5,
        )

        assert (Path(tmp_dir) / "faiss_index.bin").exists()
        assert (Path(tmp_dir) / "chunks_metadata.json").exists()

        # Retrieve relevant chunks for suspension query
        results = pipeline.retriever.retrieve("Suspension system inspection", top_k=3)
        assert len(results) > 0
        assert results[0].page_number in [1, 2, 3, 4, 5]


def test_pipeline_query_unbuilt_index_raises():
    with tempfile.TemporaryDirectory() as empty_dir:
        pipeline = SpecificationPipeline(
            index_dir=empty_dir,
            embedder_provider="local",
        )
        with pytest.raises(RuntimeError):
            pipeline.retriever.retrieve("test query")
