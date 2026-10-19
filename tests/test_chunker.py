"""
Unit tests for TextChunker.
"""

import pytest
from src.chunker import DocumentChunk, TextChunker
from src.pdf_extractor import ExtractedPage


def test_chunker_validation():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=-10)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=500, chunk_overlap=600)


def test_chunk_empty_page():
    chunker = TextChunker()
    empty_page = ExtractedPage(page_number=1, text="   ")
    chunks = chunker.chunk_page(empty_page)
    assert chunks == []


def test_chunk_small_page_fits_single_chunk():
    chunker = TextChunker(chunk_size=1000)
    page_text = (
        "SECTION 206-03: Front Disc Brake\n"
        "Brake caliper anchor plate bolts: 250 Nm (184 lb-ft)\n"
        "Brake caliper flow bolt: 35 Nm (26 lb-ft)"
    )
    page = ExtractedPage(
        page_number=636,
        text=page_text,
        section="SECTION 206-03: Front Disc Brake",
        subsections=["SPECIFICATIONS"],
    )

    chunks = chunker.chunk_page(page)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_number == 636
    assert chunk.section == "SECTION 206-03: Front Disc Brake"
    assert "35 Nm" in chunk.text
    assert "[Section: SECTION 206-03: Front Disc Brake | Page: 636]" in chunk.contextual_text


def test_chunk_large_page_splits_with_overlap():
    chunker = TextChunker(chunk_size=300, chunk_overlap=80)
    # Generate multi-paragraph text
    paragraphs = [
        f"Paragraph {i}: Detailed automotive maintenance specification instructions describing procedure step {i}."
        for i in range(10)
    ]
    large_text = "\n\n".join(paragraphs)
    page = ExtractedPage(
        page_number=10,
        text=large_text,
        section="SECTION 204-01: Front Suspension",
    )

    chunks = chunker.chunk_page(page)
    assert len(chunks) > 1

    # Verify every chunk has proper metadata
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.page_number == 10
        assert c.section == "SECTION 204-01: Front Suspension"
        assert len(c.text) <= 450  # soft ceiling with atomic line units
