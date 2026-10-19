"""
PDF text extraction module for automotive service manuals.
Extracts text page-by-page, strips repetitive headers/footers,
preserves structured tables, and tracks active section headings.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pymupdf

logger = logging.getLogger(__name__)

# Common header/footer noise patterns found in workshop service manuals
DEFAULT_FOOTER_PATTERNS = [
    r"^Page\s+\d+\s+sur\s+\d+$",
    r"^\d+\s+sur\s+\d+$",
    r"^Page\s+\d+\s+of\s+\d+$",
    r"^2014\s+F-150\s+Workshop\s+Manual$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^file:///.*$",
    r"^repair4less$",
]

# Pattern matching manual major section titles
# e.g., "SECTION 206-03: Front Disc Brake"
SECTION_PATTERN = re.compile(
    r"(SECTION\s+\d{3}-\d{2}[A-Z]?:\s*[^\n\r]+)",
    re.IGNORECASE,
)

# Common subsection headers found on manual pages
SUBSECTION_KEYWORDS = [
    "SPECIFICATIONS",
    "Torque Specifications",
    "General Specifications",
    "Material",
    "DIAGNOSIS AND TESTING",
    "GENERAL PROCEDURES",
    "DISASSEMBLY AND ASSEMBLY",
    "REMOVAL AND INSTALLATION",
    "Inspection and Verification",
    "Symptom Chart",
]


@dataclass
class ExtractedPage:
    """Represents text extracted from a single PDF page with metadata."""
    page_number: int  # 1-indexed
    text: str
    section: str = ""
    subsections: List[str] = field(default_factory=list)
    has_tables: bool = False
    char_count: int = 0


@dataclass
class ExtractionSummary:
    """Summary metrics of the PDF extraction process."""
    pdf_path: str
    total_pages_in_pdf: int
    content_pages: int
    skipped_empty_pages: int
    total_characters: int
    sections_found: List[str]


class PDFExtractor:
    """
    Extracts text from automotive service manual PDFs.
    Preserves section structure, tabular data, and eliminates header/footer noise.
    """

    def __init__(
        self,
        footer_patterns: Optional[List[str]] = None,
        extract_tables: bool = True,
        min_page_chars: int = 15,
    ) -> None:
        patterns = footer_patterns or DEFAULT_FOOTER_PATTERNS
        self.footer_regex = re.compile("|".join(patterns), re.IGNORECASE)
        self.extract_tables = extract_tables
        self.min_page_chars = min_page_chars

    def is_noise_line(self, line: str) -> bool:
        """Check if a line matches known repetitive header/footer noise."""
        clean = line.strip()
        if not clean:
            return True
        return bool(self.footer_regex.match(clean))

    @staticmethod
    def _rect_intersects(r1: Tuple[float, float, float, float], r2: Tuple[float, float, float, float]) -> bool:
        """Check whether two bounding boxes (x0, y0, x1, y1) intersect."""
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def _extract_page_content(self, page: pymupdf.Page) -> Tuple[str, bool]:
        """
        Extracts content from a page. If tables exist and table extraction is enabled,
        formats tables into Markdown-style tables and interleaves with non-table text.
        """
        tab_bboxes = []
        table_items: List[Tuple[float, str]] = []

        if self.extract_tables:
            try:
                tables = page.find_tables()
                for table in tables.tables:
                    tab_bboxes.append(table.bbox)
                    rows = table.extract()
                    table_lines = []
                    for row in rows:
                        cleaned_row = [
                            str(cell).replace("\n", " ").strip() if cell else ""
                            for cell in row
                        ]
                        if any(cleaned_row):
                            table_lines.append("| " + " | ".join(cleaned_row) + " |")
                    if table_lines:
                        table_items.append((table.bbox[1], "\n".join(table_lines)))
            except Exception as e:
                logger.warning(f"Table extraction failed on page {page.number + 1}: {e}")

        # Extract text blocks that do not overlap with detected tables
        text_blocks: List[Tuple[float, str]] = []
        try:
            raw_blocks = page.get_text("blocks")
            for block in raw_blocks:
                bbox = block[:4]
                text = block[4]
                # If block intersects any extracted table, skip it to avoid duplicate rows
                if any(self._rect_intersects(bbox, tb) for tb in tab_bboxes):
                    continue
                # Filter noise lines from the text block
                filtered_lines = [
                    line.strip()
                    for line in text.splitlines()
                    if not self.is_noise_line(line)
                ]
                if filtered_lines:
                    text_blocks.append((bbox[1], "\n".join(filtered_lines)))
        except Exception as e:
            logger.warning(f"Block extraction failed on page {page.number + 1}: {e}")
            # Fallback to plain text extraction
            plain_lines = [
                line.strip()
                for line in page.get_text().splitlines()
                if not self.is_noise_line(line)
            ]
            return "\n".join(plain_lines), bool(table_items)

        # Merge text blocks and formatted tables ordered by vertical coordinate (y0)
        all_items = [(y, content) for y, content in (text_blocks + table_items)]
        all_items.sort(key=lambda x: x[0])

        combined_text = "\n\n".join(content for _, content in all_items).strip()
        # Clean excessive blank lines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", combined_text)
        return cleaned_text, bool(table_items)

    def extract_document(
        self,
        pdf_path: str | Path,
        start_page: int = 1,
        max_pages: Optional[int] = None,
    ) -> Tuple[List[ExtractedPage], ExtractionSummary]:
        """
        Extracts pages from the PDF, cleaning noise and tracking active sections.

        Args:
            pdf_path: Path to the automotive service manual PDF.
            start_page: 1-indexed start page number (default: 1).
            max_pages: Optional maximum number of pages to process.

        Returns:
            Tuple of (list of ExtractedPage, ExtractionSummary).
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path.resolve()}")

        doc = pymupdf.open(str(path))
        total_pages = len(doc)
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, start_idx + max_pages) if max_pages is not None else total_pages

        extracted_pages: List[ExtractedPage] = []
        current_section = "General Information"
        sections_found: Dict[str, int] = {}
        total_chars = 0
        skipped_pages = 0

        for page_idx in range(start_idx, end_idx):
            page_number = page_idx + 1
            page = doc[page_idx]

            try:
                page_text, has_tables = self._extract_page_content(page)
            except Exception as e:
                logger.error(f"Error reading page {page_number}: {e}")
                raise RuntimeError(f"Failed to extract page {page_number} from {pdf_path}: {e}") from e

            # Look for new section title in this page's content
            section_match = SECTION_PATTERN.search(page_text)
            if section_match:
                current_section = section_match.group(1).strip()
                sections_found[current_section] = sections_found.get(current_section, 0) + 1

            # Detect any subsection keywords present on this page
            subsections = [
                sub for sub in SUBSECTION_KEYWORDS
                if re.search(rf"\b{re.escape(sub)}\b", page_text, re.IGNORECASE)
            ]

            # Filter out pages that contain little or no text (e.g. diagrams or blank pages)
            clean_char_count = len(re.sub(r"\s+", "", page_text))
            if clean_char_count < self.min_page_chars:
                skipped_pages += 1
                continue

            extracted_page = ExtractedPage(
                page_number=page_number,
                text=page_text,
                section=current_section,
                subsections=subsections,
                has_tables=has_tables,
                char_count=len(page_text),
            )
            extracted_pages.append(extracted_page)
            total_chars += len(page_text)

        doc.close()

        summary = ExtractionSummary(
            pdf_path=str(path),
            total_pages_in_pdf=total_pages,
            content_pages=len(extracted_pages),
            skipped_empty_pages=skipped_pages,
            total_characters=total_chars,
            sections_found=list(sections_found.keys()),
        )

        return extracted_pages, summary
