"""
Unit and integration tests for PDFExtractor.
"""

from pathlib import Path
import pytest
from src.pdf_extractor import ExtractedPage, ExtractionSummary, PDFExtractor


def test_is_noise_line():
    extractor = PDFExtractor()

    # Known noise lines should be identified
    assert extractor.is_noise_line("Page 1 sur 2")
    assert extractor.is_noise_line("Page 14 sur 25")
    assert extractor.is_noise_line("2014 F-150 Workshop Manual")
    assert extractor.is_noise_line("2014-03-01")
    assert extractor.is_noise_line("file:///C:/TSO/tsocache/something.HTM")
    assert extractor.is_noise_line("repair4less")
    assert extractor.is_noise_line("   ")  # blank line

    # Valid specification lines should NOT be treated as noise
    assert not extractor.is_noise_line("Brake caliper anchor plate bolts")
    assert not extractor.is_noise_line("35 Nm (26 lb-ft)")
    assert not extractor.is_noise_line("SECTION 206-03: Front Disc Brake")
    assert not extractor.is_noise_line("Motorcraft® High Performance DOT 3 Motor Vehicle Brake Fluid")


def test_missing_pdf_raises():
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_document("non_existent_manual.pdf")


def test_extract_sample_service_manual():
    manual_path = Path("sample-service-manual 1.pdf")
    if not manual_path.exists():
        pytest.skip("sample-service-manual 1.pdf not found in workspace")

    extractor = PDFExtractor()
    # Extract first 5 pages for rapid test
    pages, summary = extractor.extract_document(manual_path, max_pages=5)

    assert isinstance(summary, ExtractionSummary)
    assert summary.total_pages_in_pdf == 852
    assert summary.content_pages > 0
    assert len(pages) == summary.content_pages

    # Check first page metadata
    first_page = pages[0]
    assert isinstance(first_page, ExtractedPage)
    assert first_page.page_number == 1
    assert len(first_page.text) > 0
    assert "Suspension System" in first_page.text
    assert "SECTION" in first_page.section


def test_extract_brake_specification_page():
    manual_path = Path("sample-service-manual 1.pdf")
    if not manual_path.exists():
        pytest.skip("sample-service-manual 1.pdf not found in workspace")

    extractor = PDFExtractor()
    # Page 636 contains front disc brake torque specifications
    pages, _ = extractor.extract_document(manual_path, start_page=636, max_pages=1)
    assert len(pages) == 1
    page_636 = pages[0]

    assert page_636.page_number == 636
    # Verify exact torque specifications and table format are preserved
    assert "Brake caliper flow bolt" in page_636.text
    assert "35" in page_636.text
