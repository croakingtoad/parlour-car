"""PDF document parser using PyMuPDF (fitz).

Extracts text blocks with positional data and uses font-size heuristics
to reconstruct document structure (headings, paragraphs, footnotes).
Handles both born-digital and OCR'd PDFs with quality detection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from author_library.errors import ParsingError
from author_library.parsing.base import DocumentParser
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

log = structlog.get_logger(__name__)

# Font size thresholds (relative to median body size)
_HEADING_SIZE_RATIO = 1.2  # 20% larger than body text
_FOOTNOTE_SIZE_RATIO = 0.85  # 15% smaller than body text
_FOOTNOTE_Y_THRESHOLD = 0.8  # bottom 20% of page
_OCR_QUALITY_THRESHOLD = 0.3  # ratio of alphanumeric chars to total


class PdfParser(DocumentParser):
    """Parser for PDF documents."""

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    async def parse(self, source: Path | str) -> ParsedDocument:
        import pymupdf

        source = Path(source)
        if not source.exists():
            raise ParsingError(
                f"PDF file not found: {source}",
                context={"path": str(source)},
            )

        log.info("parsing_pdf", path=str(source))

        try:
            doc: Any = pymupdf.open(str(source))  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise ParsingError(
                f"Failed to open PDF: {exc}",
                context={"path": str(source)},
                cause=exc,
            ) from exc

        warnings: list[str] = []
        metadata = self._extract_metadata(doc, warnings)

        root = DocumentNode(
            node_type=NodeType.BOOK, metadata={"title": metadata.title or ""}
        )
        raw_text_parts: list[str] = []

        if doc.page_count == 0:
            warnings.append("PDF has no pages")
            doc.close()
            return ParsedDocument(
                source_path=str(source),
                format="pdf",
                metadata=metadata,
                tree=root,
                raw_text="",
                parse_warnings=warnings,
            )

        # First pass: compute median font size across entire document
        all_font_sizes: list[float] = []
        page_blocks_cache: list[list[dict[str, Any]]] = []

        for page_num in range(doc.page_count):
            page: Any = doc[page_num]
            blocks: list[dict[str, Any]] = page.get_text(
                "dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE
            )["blocks"]
            enriched: list[dict[str, Any]] = []
            for block in blocks:
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size: float = span.get("size", 0.0)
                        if size > 0:
                            all_font_sizes.append(size)
                enriched.append(block)
            page_blocks_cache.append(enriched)

        median_size = self._median(all_font_sizes) if all_font_sizes else 12.0

        # Second pass: build document tree
        current_chapter: DocumentNode | None = None

        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_height: float = page.rect.height
            blocks = (
                page_blocks_cache[page_num]
                if page_num < len(page_blocks_cache)
                else []
            )

            for block in blocks:
                for line in block.get("lines", []):
                    spans: list[dict[str, Any]] = line.get("spans", [])
                    if not spans:
                        continue

                    text_parts: list[str] = []
                    max_size = 0.0
                    is_bold = False
                    avg_y = 0.0

                    for span in spans:
                        text_parts.append(span.get("text", ""))
                        size = span.get("size", 0.0)
                        if size > max_size:
                            max_size = size
                        flags: int = span.get("flags", 0)
                        if flags & 16:  # bold flag
                            is_bold = True
                        origin = span.get("origin")
                        avg_y = (
                            origin[1] if isinstance(origin, list) else 0.0
                        )

                    text = "".join(text_parts).strip()
                    if not text:
                        continue

                    raw_text_parts.append(text)

                    # Classify the text block
                    node = self._classify_block(
                        text=text,
                        font_size=max_size,
                        median_size=median_size,
                        is_bold=is_bold,
                        y_position=avg_y,
                        page_height=page_height,
                        page_num=page_num,
                    )

                    # Determine where to attach
                    if node.node_type == NodeType.HEADING and (
                        max_size >= median_size * 1.4
                        or _is_chapter_heading(text)
                    ):
                        current_chapter = DocumentNode(
                            node_type=NodeType.CHAPTER,
                            metadata={
                                "title": text,
                                "page_number": page_num + 1,
                            },
                        )
                        current_chapter.children.append(node)
                        root.children.append(current_chapter)
                        continue

                    if current_chapter is not None:
                        current_chapter.children.append(node)
                    else:
                        root.children.append(node)

        doc.close()

        raw_text = "\n".join(raw_text_parts)
        metadata.word_count = len(raw_text.split())

        # OCR quality check
        if raw_text:
            alpha_ratio = sum(1 for c in raw_text if c.isalnum()) / len(
                raw_text
            )
            if alpha_ratio < _OCR_QUALITY_THRESHOLD:
                warnings.append(
                    f"Low OCR quality detected: {alpha_ratio:.0%} "
                    "alphanumeric characters"
                )

        return ParsedDocument(
            source_path=str(source),
            format="pdf",
            metadata=metadata,
            tree=root,
            raw_text=raw_text,
            parse_warnings=warnings,
        )

    def _extract_metadata(
        self, doc: Any, warnings: list[str]
    ) -> DocumentMetadata:
        """Extract metadata from PDF info dict."""
        meta: dict[str, str] = doc.metadata
        if not meta:
            warnings.append("No metadata found in PDF")
            return DocumentMetadata()

        # Try to extract ISBN from subject/keywords
        isbn: str | None = None
        for field in ("subject", "keywords"):
            val = meta.get(field, "")
            if val:
                match = re.search(
                    r"(97[89][\d-]{10,}|\d{9}[\dXx])", str(val)
                )
                if match:
                    isbn = match.group(1)
                    break

        return DocumentMetadata(
            title=meta.get("title") or None,
            author=meta.get("author") or None,
            publisher=meta.get("producer") or None,
            publication_date=meta.get("creationDate") or None,
            isbn=isbn,
        )

    def _classify_block(
        self,
        *,
        text: str,
        font_size: float,
        median_size: float,
        is_bold: bool,
        y_position: float,
        page_height: float,
        page_num: int,
    ) -> DocumentNode:
        """Classify a text block into the appropriate node type."""
        # Footnote detection: small font near bottom of page
        if (
            font_size < median_size * _FOOTNOTE_SIZE_RATIO
            and page_height > 0
            and y_position / page_height > _FOOTNOTE_Y_THRESHOLD
        ):
            return DocumentNode(
                node_type=NodeType.FOOTNOTE,
                text=text,
                metadata={"page_number": page_num + 1},
            )

        # Heading detection: larger or bold font
        if font_size > median_size * _HEADING_SIZE_RATIO or (
            is_bold and font_size >= median_size
        ):
            return DocumentNode(
                node_type=NodeType.HEADING,
                text=text,
                metadata={"page_number": page_num + 1},
            )

        # Default: paragraph
        return DocumentNode(
            node_type=NodeType.PARAGRAPH,
            text=text,
            metadata={"page_number": page_num + 1},
        )

    @staticmethod
    def _median(values: list[float]) -> float:
        """Compute median of a list of floats."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]


def _is_chapter_heading(text: str) -> bool:
    """Check if text looks like a chapter heading."""
    return bool(
        re.match(
            r"^(chapter|part|book|prologue|epilogue|introduction"
            r"|conclusion|appendix)"
            r"[\s.:]*(\d+|[ivxlcdm]+)?",
            text.strip(),
            re.IGNORECASE,
        )
    )
