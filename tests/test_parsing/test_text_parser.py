"""Tests for the plain text parser."""

import pytest

from author_library.errors import ParsingError
from author_library.parsing.models import NodeType
from author_library.parsing.text_parser import TextParser


@pytest.fixture
def parser() -> TextParser:
    return TextParser()


@pytest.fixture
def simple_text_file(tmp_path: object) -> object:
    from pathlib import Path

    p = Path(str(tmp_path)) / "simple.txt"
    p.write_text(
        "Chapter 1\n\n"
        "This is the first paragraph of chapter one.\n\n"
        "This is the second paragraph.\n\n"
        "Chapter 2\n\n"
        "Another chapter begins here.\n"
    )
    return p


@pytest.fixture
def text_with_all_caps_headings(tmp_path: object) -> object:
    from pathlib import Path

    p = Path(str(tmp_path)) / "caps.txt"
    p.write_text(
        "THE FIRST SECTION\n\n"
        "Some text under the first section.\n\n"
        "THE SECOND SECTION\n\n"
        "More text here.\n"
    )
    return p


@pytest.fixture
def empty_text_file(tmp_path: object) -> object:
    from pathlib import Path

    p = Path(str(tmp_path)) / "empty.txt"
    p.write_text("")
    return p


class TestTextParser:
    async def test_supported_extensions(self, parser: TextParser) -> None:
        assert ".txt" in parser.supported_extensions()
        assert ".text" in parser.supported_extensions()

    async def test_parse_simple(self, parser: TextParser, simple_text_file: object) -> None:
        result = await parser.parse(simple_text_file)  # type: ignore[arg-type]
        assert result.format == "txt"
        assert result.tree.node_type == NodeType.BOOK
        # Should have chapter nodes
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        assert len(chapters) == 2
        assert chapters[0].metadata.get("title") == "Chapter 1"
        assert chapters[1].metadata.get("title") == "Chapter 2"
        # First chapter should have paragraphs
        paras = [c for c in chapters[0].children if c.node_type == NodeType.PARAGRAPH]
        assert len(paras) >= 1
        assert result.metadata.word_count > 0

    async def test_parse_all_caps_headings(
        self, parser: TextParser, text_with_all_caps_headings: object
    ) -> None:
        result = await parser.parse(text_with_all_caps_headings)  # type: ignore[arg-type]
        sections = [c for c in result.tree.children if c.node_type == NodeType.SECTION]
        assert len(sections) == 2

    async def test_parse_empty_file(self, parser: TextParser, empty_text_file: object) -> None:
        result = await parser.parse(empty_text_file)  # type: ignore[arg-type]
        assert len(result.parse_warnings) > 0
        assert "empty" in result.parse_warnings[0].lower()

    async def test_file_not_found(self, parser: TextParser) -> None:
        with pytest.raises(ParsingError, match="not found"):
            await parser.parse("/nonexistent/file.txt")

    async def test_raw_text_populated(self, parser: TextParser, simple_text_file: object) -> None:
        result = await parser.parse(simple_text_file)  # type: ignore[arg-type]
        assert len(result.raw_text) > 0
        assert "first paragraph" in result.raw_text

    async def test_chapter_patterns(self, parser: TextParser, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "chapters.txt"
        p.write_text(
            "Prologue\n\n"
            "The story begins.\n\n"
            "Chapter I\n\n"
            "First chapter text.\n\n"
            "Chapter 2\n\n"
            "Second chapter text.\n\n"
            "Epilogue\n\n"
            "The end.\n"
        )
        result = await parser.parse(p)
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        assert len(chapters) == 4  # Prologue, Ch I, Ch 2, Epilogue
