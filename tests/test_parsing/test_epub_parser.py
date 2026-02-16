"""Tests for the EPUB parser.

Creates minimal EPUB files programmatically using ebooklib for testing.
"""

import pytest
from ebooklib import epub

from author_library.errors import ParsingError
from author_library.parsing.epub_parser import EpubParser
from author_library.parsing.models import NodeType


def _create_test_epub(
    path: object, *, title: str = "Test Book", author: str = "Test Author"
) -> None:
    """Create a minimal EPUB file for testing."""
    from pathlib import Path

    book = epub.EpubBook()
    book.set_identifier("test-id-123")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    # Chapter 1
    ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    ch1.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
    <h1>Chapter 1: The Beginning</h1>
    <p>This is the first paragraph of chapter one.</p>
    <p>This is the second paragraph with more content.</p>
    <blockquote>A wise person once said something profound.</blockquote>
</body>
</html>"""
    book.add_item(ch1)

    # Chapter 2
    ch2 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
    ch2.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
    <h1>Chapter 2: The Middle</h1>
    <p>The story continues in chapter two.</p>
    <h2>Section 2.1</h2>
    <p>A subsection within the chapter.</p>
    <ul>
        <li>First item</li>
        <li>Second item</li>
    </ul>
</body>
</html>"""
    book.add_item(ch2)

    # Table of contents
    book.toc = [
        epub.Link("ch1.xhtml", "Chapter 1: The Beginning", "ch1"),
        epub.Link("ch2.xhtml", "Chapter 2: The Middle", "ch2"),
    ]

    # Spine
    book.spine = ["nav", ch1, ch2]

    # Navigation
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(Path(str(path))), book)


@pytest.fixture
def parser() -> EpubParser:
    return EpubParser()


@pytest.fixture
def test_epub(tmp_path: object) -> object:
    from pathlib import Path

    path = Path(str(tmp_path)) / "test.epub"
    _create_test_epub(path)
    return path


class TestEpubParser:
    async def test_supported_extensions(self, parser: EpubParser) -> None:
        assert parser.supported_extensions() == [".epub"]

    async def test_parse_basic(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]
        assert result.format == "epub"
        assert result.source_path == str(test_epub)
        assert result.tree.node_type == NodeType.BOOK

    async def test_metadata_extraction(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]
        assert result.metadata.title == "Test Book"
        assert result.metadata.author == "Test Author"
        assert result.metadata.language == "en"

    async def test_toc_extraction(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]
        assert len(result.metadata.table_of_contents) >= 2

    async def test_chapter_structure(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        # Should have chapters from the two spine items (plus possible nav)
        assert len(chapters) >= 2

    async def test_paragraph_extraction(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        paragraphs = find_type(result.tree, NodeType.PARAGRAPH)
        assert len(paragraphs) >= 3

    async def test_blockquote_detection(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        quotes = find_type(result.tree, NodeType.BLOCK_QUOTE)
        assert len(quotes) >= 1
        assert "wise person" in quotes[0].text  # type: ignore[union-attr]

    async def test_raw_text(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]
        assert len(result.raw_text) > 0
        assert result.metadata.word_count > 0

    async def test_file_not_found(self, parser: EpubParser) -> None:
        with pytest.raises(ParsingError, match="not found"):
            await parser.parse("/nonexistent/book.epub")

    async def test_list_extraction(self, parser: EpubParser, test_epub: object) -> None:
        result = await parser.parse(test_epub)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        lists = find_type(result.tree, NodeType.LIST)
        assert len(lists) >= 1
