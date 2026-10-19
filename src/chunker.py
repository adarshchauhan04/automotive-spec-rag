"""
Text chunking module for automotive service manuals.
Splits extracted document pages into logical, searchable chunks while
preserving table rows, section context, and source page traceability.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional
from src.pdf_extractor import ExtractedPage

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a discrete, searchable chunk of text with metadata."""
    chunk_id: str
    text: str
    page_number: int
    section: str
    subsections: List[str] = field(default_factory=list)
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def contextual_text(self) -> str:
        """Text enriched with section and page context for embedding/retrieval."""
        header = f"[Section: {self.section} | Page: {self.page_number}]"
        return f"{header}\n{self.text}"


class TextChunker:
    """
    Chunks extracted pages into semantically cohesive passages.
    Avoids splitting table rows or specification lines across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        min_chunk_chars: int = 40,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must be between 0 and chunk_size, got {chunk_overlap}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def _split_into_units(self, text: str) -> List[str]:
        """
        Splits text into atomic units (paragraphs, table blocks, or lines)
        to prevent breaking individual specification rows.
        """
        paragraphs = text.split("\n\n")
        units: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If paragraph itself is very long, break by line
            if len(para) > self.chunk_size:
                lines = [line.strip() for line in para.split("\n") if line.strip()]
                units.extend(lines)
            else:
                units.append(para)
        return units

    def chunk_page(self, page: ExtractedPage) -> List[DocumentChunk]:
        """
        Splits a single ExtractedPage into one or more DocumentChunk objects.
        """
        raw_text = page.text.strip()
        if not raw_text:
            return []

        # If page fits within chunk_size, keep it intact as a single chunk
        if len(raw_text) <= self.chunk_size:
            chunk = DocumentChunk(
                chunk_id=f"p{page.page_number}_c1",
                text=raw_text,
                page_number=page.page_number,
                section=page.section,
                subsections=page.subsections,
                char_count=len(raw_text),
                metadata={
                    "page_number": page.page_number,
                    "section": page.section,
                    "subsections": page.subsections,
                    "has_tables": page.has_tables,
                },
            )
            return [chunk]

        # Multi-chunk splitting for longer pages
        units = self._split_into_units(raw_text)
        chunks: List[DocumentChunk] = []
        current_units: List[str] = []
        current_length = 0
        chunk_idx = 1

        for unit in units:
            unit_len = len(unit) + 2  # account for newline separation
            if current_length + unit_len > self.chunk_size and current_units:
                chunk_text = "\n\n".join(current_units).strip()
                if len(chunk_text) >= self.min_chunk_chars:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"p{page.page_number}_c{chunk_idx}",
                            text=chunk_text,
                            page_number=page.page_number,
                            section=page.section,
                            subsections=page.subsections,
                            char_count=len(chunk_text),
                            metadata={
                                "page_number": page.page_number,
                                "section": page.section,
                                "subsections": page.subsections,
                                "has_tables": page.has_tables,
                            },
                        )
                    )
                    chunk_idx += 1

                # Retain overlap from end of current units
                overlap_units: List[str] = []
                overlap_len = 0
                for u in reversed(current_units):
                    if overlap_len + len(u) <= self.chunk_overlap:
                        overlap_units.insert(0, u)
                        overlap_len += len(u) + 2
                    else:
                        break
                current_units = overlap_units
                current_length = overlap_len

            current_units.append(unit)
            current_length += unit_len

        # Flush any remaining units
        if current_units:
            chunk_text = "\n\n".join(current_units).strip()
            if len(chunk_text) >= self.min_chunk_chars:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"p{page.page_number}_c{chunk_idx}",
                        text=chunk_text,
                        page_number=page.page_number,
                        section=page.section,
                        subsections=page.subsections,
                        char_count=len(chunk_text),
                        metadata={
                            "page_number": page.page_number,
                            "section": page.section,
                            "subsections": page.subsections,
                            "has_tables": page.has_tables,
                        },
                    )
                )

        return chunks

    def chunk_document(self, pages: List[ExtractedPage]) -> List[DocumentChunk]:
        """
        Chunks all extracted pages in the document.

        Args:
            pages: List of ExtractedPage objects.

        Returns:
            List of DocumentChunk objects across all pages.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)

        logger.info(
            f"Chunked {len(pages)} pages into {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks
