"""Tests for the PDF parser.

Creates minimal PDF files programmatically using pymupdf for testing.
"""

import pytest

from author_library.errors import ParsingError
from author_library.parsing.models import NodeType
from author_library.parsing.pdf_parser import PdfParser


def _create_test_pdf(path: object) -> None:
    """Create a minimal PDF with structured content for testing."""
    from pathlib import Path

    import pymupdf

    doc = pymupdf.open()
    doc.set_metadata({
        "title": "Test PDF Document",
        "author": "Test Author",
        "subject": "Testing",
    })

    # Page 1: Title and paragraphs
    page = doc.new_page()
    # Large heading
    page.insert_text(
        pymupdf.Point(72, 100),
        "Chapter 1: Introduction",
        fontsize=24,
        fontname="helv",
    )
    # Body text
    page.insert_text(
        pymupdf.Point(72, 160),
        "This is the first paragraph of the introduction.",
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        pymupdf.Point(72, 190),
        "This is the second paragraph with more content.",
        fontsize=12,
        fontname="helv",
    )
    # Footnote at bottom
    page.insert_text(
        pymupdf.Point(72, 750),
        "1. This is a footnote reference.",
        fontsize=9,
        fontname="helv",
    )

    # Page 2: Another chapter
    page2 = doc.new_page()
    page2.insert_text(
        pymupdf.Point(72, 100),
        "Chapter 2: Methods",
        fontsize=24,
        fontname="helv",
    )
    page2.insert_text(
        pymupdf.Point(72, 160),
        "The methods section describes the approach taken.",
        fontsize=12,
        fontname="helv",
    )

    doc.save(str(Path(str(path))))
    doc.close()


@pytest.fixture
def parser() -> PdfParser:
    return PdfParser()


@pytest.fixture
def test_pdf(tmp_path: object) -> object:
    from pathlib import Path

    path = Path(str(tmp_path)) / "test.pdf"
    _create_test_pdf(path)
    return path


class TestPdfParser:
    async def test_supported_extensions(self, parser: PdfParser) -> None:
        assert parser.supported_extensions() == [".pdf"]

    async def test_parse_basic(self, parser: PdfParser, test_pdf: object) -> None:
        result = await parser.parse(test_pdf)  # type: ignore[arg-type]
        assert result.format == "pdf"
        assert result.tree.node_type == NodeType.BOOK

    async def test_metadata_extraction(self, parser: PdfParser, test_pdf: object) -> None:
        result = await parser.parse(test_pdf)  # type: ignore[arg-type]
        assert result.metadata.title == "Test PDF Document"
        assert result.metadata.author == "Test Author"

    async def test_text_extraction(self, parser: PdfParser, test_pdf: object) -> None:
        result = await parser.parse(test_pdf)  # type: ignore[arg-type]
        assert "first paragraph" in result.raw_text
        assert "methods section" in result.raw_text
        assert result.metadata.word_count > 0

    async def test_heading_detection(self, parser: PdfParser, test_pdf: object) -> None:
        result = await parser.parse(test_pdf)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        headings = find_type(result.tree, NodeType.HEADING)
        heading_texts = [h.text for h in headings]  # type: ignore[union-attr]
        assert any("Introduction" in t for t in heading_texts)

    async def test_file_not_found(self, parser: PdfParser) -> None:
        with pytest.raises(ParsingError, match="not found"):
            await parser.parse("/nonexistent/doc.pdf")

    async def test_page_number_metadata(self, parser: PdfParser, test_pdf: object) -> None:
        result = await parser.parse(test_pdf)  # type: ignore[arg-type]
        # At least some nodes should have page_number metadata
        found_page = False

        def check_pages(node: object) -> None:
            from author_library.parsing.models import DocumentNode

            nonlocal found_page
            assert isinstance(node, DocumentNode)
            if node.metadata.get("page_number"):
                found_page = True
            for child in node.children:
                check_pages(child)

        check_pages(result.tree)
        assert found_page

    def test_median_calculation(self, parser: PdfParser) -> None:
        assert parser._median([1.0, 2.0, 3.0]) == 2.0
        assert parser._median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert parser._median([5.0]) == 5.0
        assert parser._median([]) == 0.0
