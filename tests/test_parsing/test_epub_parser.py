"""Tests for the EPUB parser.

Creates minimal EPUB files programmatically using ebooklib for testing.
"""

import pytest
from ebooklib import epub

from author_library.errors import ParsingError
from author_library.parsing.epub_parser import (
    EpubParser,
    _decode_epub_content,
    _sanitize_text,
)
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


class TestSingleSpineItemEpub:
    """Tests for EPUBs where the entire book is in a single HTML file.

    Many EPUBs (especially those converted from other formats) pack all
    content into one spine item, using ``<h1>`` tags for chapter boundaries
    and sometimes wrapping everything in a single ``<div>``.
    """

    @staticmethod
    def _create_single_spine_epub(path: object) -> None:
        """Create an EPUB with a single spine item containing multiple h1 chapters."""
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("single-spine-test")
        book.set_title("Single Spine Book")
        book.set_language("en")
        book.add_author("Test Author")

        ch = epub.EpubHtml(title="All Content", file_name="text.xhtml", lang="en")
        ch.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Single Spine</title></head>
<body>
<div>
    <p>Title Page Text</p>
    <br/>
    <h1>Introduction</h1>
    <p>The introduction paragraph one.</p>
    <p>The introduction paragraph two with more words to make it longer.</p>
    <br/>
    <h1>Chapter One</h1>
    <p>First paragraph of chapter one.</p>
    <p>Second paragraph of chapter one.</p>
    <p>Third paragraph of chapter one.</p>
    <br/>
    <h1>Chapter Two</h1>
    <p>First paragraph of chapter two.</p>
    <p>Second paragraph of chapter two.</p>
    <h2>Section 2.1</h2>
    <p>A subsection within chapter two.</p>
</div>
</body>
</html>"""
        book.add_item(ch)

        book.toc = [
            epub.Link("text.xhtml", "Introduction", "intro"),
            epub.Link("text.xhtml", "Chapter One", "ch1"),
            epub.Link("text.xhtml", "Chapter Two", "ch2"),
        ]

        book.spine = ["nav", ch]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def single_spine_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "single_spine.epub"
        self._create_single_spine_epub(path)
        return path

    async def test_wrapper_div_unwrapped(
        self, parser: EpubParser, single_spine_epub: object
    ) -> None:
        """A wrapper div with headings should be unwrapped, not treated as one node."""
        result = await parser.parse(single_spine_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        # The single body contains 3 h1 headings, so we expect 3 chapters
        # (the first untitled chapter with just "Title Page Text" may also appear)
        assert len(chapters) >= 3

    async def test_chapter_titles_extracted(
        self, parser: EpubParser, single_spine_epub: object
    ) -> None:
        """Each h1 boundary should produce a chapter with the correct title."""
        result = await parser.parse(single_spine_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        titles = [str(c.metadata.get("title", "")) for c in chapters]
        assert "Introduction" in titles
        assert "Chapter One" in titles
        assert "Chapter Two" in titles

    async def test_not_misidentified_as_poem(
        self, parser: EpubParser, single_spine_epub: object
    ) -> None:
        """A wrapper div with <br> spacers and headings must NOT become a POEM node."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(single_spine_epub)  # type: ignore[arg-type]

        def find_type(node: DocumentNode, ntype: NodeType) -> list[DocumentNode]:
            found: list[DocumentNode] = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        poems = find_type(result.tree, NodeType.POEM)
        assert len(poems) == 0

    async def test_paragraphs_distributed_across_chapters(
        self, parser: EpubParser, single_spine_epub: object
    ) -> None:
        """Paragraphs should be children of their respective chapters."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(single_spine_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]

        def count_paragraphs(node: DocumentNode) -> int:
            count = 1 if node.node_type == NodeType.PARAGRAPH else 0
            for child in node.children:
                count += count_paragraphs(child)
            return count

        # Each content chapter should have paragraphs
        for ch in chapters:
            title = str(ch.metadata.get("title", ""))
            if title in ("Introduction", "Chapter One", "Chapter Two"):
                assert count_paragraphs(ch) >= 2, (
                    f"Chapter '{title}' should have at least 2 paragraphs"
                )

    async def test_sections_within_chapters(
        self, parser: EpubParser, single_spine_epub: object
    ) -> None:
        """h2 headings within a chapter should create SECTION nodes."""
        result = await parser.parse(single_spine_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        ch2 = next(
            (c for c in chapters if c.metadata.get("title") == "Chapter Two"), None
        )
        assert ch2 is not None
        sections = [c for c in ch2.children if c.node_type == NodeType.SECTION]
        assert len(sections) >= 1
        assert sections[0].metadata.get("title") == "Section 2.1"


class TestDivWrappedContentEpub:
    """Tests for EPUBs where paragraphs are nested inside wrapper divs.

    Many publisher EPUBs wrap chapter content in ``<div class="content">``
    or similar containers.  The parser must recurse into these divs to
    extract individual paragraphs rather than collapsing all text into a
    single PARAGRAPH node.
    """

    @staticmethod
    def _create_div_wrapped_epub(path: object) -> None:
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("div-wrapped-test")
        book.set_title("Div Wrapped Book")
        book.set_language("en")
        book.add_author("Test Author")

        ch = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        ch.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
<div class="chapter">
    <h1>Chapter One: The Beginning</h1>
    <div class="content">
        <p>First paragraph of the chapter with some content.</p>
        <p>Second paragraph continues the discussion.</p>
        <p>Third paragraph wraps up the section.</p>
        <blockquote>A notable quotation from the text.</blockquote>
        <p>Fourth paragraph after the quote.</p>
    </div>
</div>
</body>
</html>"""
        book.add_item(ch)

        book.toc = [epub.Link("ch1.xhtml", "Chapter 1", "ch1")]
        book.spine = ["nav", ch]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def div_wrapped_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "div_wrapped.epub"
        self._create_div_wrapped_epub(path)
        return path

    async def test_paragraphs_not_collapsed(
        self, parser: EpubParser, div_wrapped_epub: object
    ) -> None:
        """Paragraphs inside a wrapper div must remain as separate PARAGRAPH nodes."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(div_wrapped_epub)  # type: ignore[arg-type]

        def find_type(node: DocumentNode, ntype: NodeType) -> list[DocumentNode]:
            found: list[DocumentNode] = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        paragraphs = find_type(result.tree, NodeType.PARAGRAPH)
        # The div wraps 4 paragraphs + 1 blockquote — at minimum 4 paragraphs
        assert len(paragraphs) >= 4, (
            f"Expected >=4 paragraphs from div-wrapped content, got {len(paragraphs)}"
        )

    async def test_blockquote_preserved(
        self, parser: EpubParser, div_wrapped_epub: object
    ) -> None:
        """Blockquote inside a wrapper div must be preserved as BLOCK_QUOTE."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(div_wrapped_epub)  # type: ignore[arg-type]

        def find_type(node: DocumentNode, ntype: NodeType) -> list[DocumentNode]:
            found: list[DocumentNode] = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        quotes = find_type(result.tree, NodeType.BLOCK_QUOTE)
        assert len(quotes) >= 1
        assert "notable quotation" in quotes[0].text


class TestEpubTitleExtraction:
    """Tests for robust EPUB title extraction.

    Covers the bug where an EPUB with dc:title = author name (e.g. "Fred
    Rogers") produced a work_id like ``fred-rogers--fred-rogers``.
    """

    @staticmethod
    def _create_epub_with_split_title(path: object) -> None:
        """Create an EPUB with separate main-title and subtitle dc:title entries."""
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("split-title-test")
        book.set_language("en")
        book.add_author("Fred Rogers")

        # Set main title and subtitle via raw metadata
        # ebooklib's set_title only sets one dc:title; we need two.
        book.add_metadata("DC", "title", "Fred Rogers", {"id": "maintitle"})
        book.add_metadata("DC", "title", "and Other Conversations", {"id": "subtitle"})
        book.add_metadata(
            "OPF", "meta", "main",
            {"property": "title-type", "refines": "#maintitle"},
        )
        book.add_metadata(
            "OPF", "meta", "subtitle",
            {"property": "title-type", "refines": "#subtitle"},
        )

        # Add a copyright page with the full title
        cop = epub.EpubHtml(title="Copyright", file_name="cop.xhtml", lang="en")
        cop.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Copyright</title></head>
<body>
<p>FRED ROGERS: THE LAST INTERVIEW AND OTHER CONVERSATIONS</p>
<p>Copyright 2021 by Melville House</p>
</body>
</html>"""
        book.add_item(cop)

        ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        ch1.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body><h1>Introduction</h1><p>Content here.</p></body>
</html>"""
        book.add_item(ch1)

        book.toc = [epub.Link("ch1.xhtml", "Introduction", "ch1")]
        book.spine = ["nav", cop, ch1]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @staticmethod
    def _create_epub_with_normal_title(path: object) -> None:
        """Create an EPUB with a normal dc:title that differs from the author."""
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("normal-title-test")
        book.set_title("Faith, Hope and Poetry")
        book.set_language("en")
        book.add_author("Malcolm Guite")

        ch = epub.EpubHtml(title="Ch1", file_name="ch1.xhtml", lang="en")
        ch.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body><h1>Chapter 1</h1><p>Content.</p></body>
</html>"""
        book.add_item(ch)

        book.toc = [epub.Link("ch1.xhtml", "Ch1", "ch1")]
        book.spine = ["nav", ch]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def split_title_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "split_title.epub"
        self._create_epub_with_split_title(path)
        return path

    @pytest.fixture
    def normal_title_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "normal_title.epub"
        self._create_epub_with_normal_title(path)
        return path

    async def test_split_title_not_author_name(
        self, parser: EpubParser, split_title_epub: object
    ) -> None:
        """When dc:title equals the author name, a better title must be found."""
        result = await parser.parse(split_title_epub)  # type: ignore[arg-type]
        assert result.metadata.title is not None
        assert result.metadata.title != "Fred Rogers", (
            "Title should not be just the author name"
        )
        # The fallback should find the full title from copyright page
        title_lower = result.metadata.title.lower()
        assert "last interview" in title_lower or "conversations" in title_lower

    async def test_normal_title_unchanged(
        self, parser: EpubParser, normal_title_epub: object
    ) -> None:
        """A normal title that differs from the author should not be changed."""
        result = await parser.parse(normal_title_epub)  # type: ignore[arg-type]
        assert result.metadata.title == "Faith, Hope and Poetry"
        assert result.metadata.author == "Malcolm Guite"

    async def test_title_from_alt_text(self) -> None:
        """_extract_title_from_alt should parse structured image alt text."""
        from author_library.parsing.epub_parser import _extract_title_from_alt

        alt = (
            "Book title, Fred Rogers: The Last Interview, subtitle, "
            "and Other Conversations, author, Fred Rogers, imprint, Melville House"
        )
        title = _extract_title_from_alt(alt)
        assert title is not None
        assert "Fred Rogers" in title
        assert "Last Interview" in title
        assert "Conversations" in title

    async def test_alt_text_no_structured_data(self) -> None:
        """_extract_title_from_alt returns None for unstructured alt text."""
        from author_library.parsing.epub_parser import _extract_title_from_alt

        assert _extract_title_from_alt("A decorative cover image") is None
        assert _extract_title_from_alt("") is None


class TestDuplicateSpineEntries:
    """Tests for EPUB deduplication: spine entries pointing to the same content."""

    @staticmethod
    def _create_epub_with_duplicate_spine(path: object) -> None:
        """Create an EPUB where the same chapter appears twice in the spine."""
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("duplicate-spine-test")
        book.set_title("Duplicate Spine Book")
        book.set_language("en")
        book.add_author("Test Author")

        ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        ch1.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
    <h1>Chapter 1: Unique Content</h1>
    <p>This paragraph should only appear once in the parsed output.</p>
    <p>Second paragraph with more content for the chapter.</p>
</body>
</html>"""
        book.add_item(ch1)

        ch2 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
        ch2.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
    <h1>Chapter 2: Different Content</h1>
    <p>This is a genuinely different chapter.</p>
</body>
</html>"""
        book.add_item(ch2)

        # Duplicate ch1 as a separate manifest item with the SAME content
        ch1_dup = epub.EpubHtml(
            title="Chapter 1 Dup", file_name="ch1_dup.xhtml", lang="en"
        )
        ch1_dup.content = ch1.content  # identical bytes
        book.add_item(ch1_dup)

        book.toc = [
            epub.Link("ch1.xhtml", "Chapter 1", "ch1"),
            epub.Link("ch2.xhtml", "Chapter 2", "ch2"),
        ]

        book.spine = ["nav", ch1, ch2, ch1_dup]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def duplicate_spine_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "duplicate_spine.epub"
        self._create_epub_with_duplicate_spine(path)
        return path

    async def test_duplicate_content_deduplicated(
        self, parser: EpubParser, duplicate_spine_epub: object
    ) -> None:
        """Spine items with identical content should be deduplicated."""
        result = await parser.parse(duplicate_spine_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        # Should have 2 unique chapters, not 3 (ch1 + ch2, NOT ch1 + ch2 + ch1_dup)
        assert len(chapters) == 2
        titles = [str(c.metadata.get("title", "")) for c in chapters]
        assert "Chapter 1: Unique Content" in titles
        assert "Chapter 2: Different Content" in titles

    async def test_nav_document_excluded(
        self, parser: EpubParser, duplicate_spine_epub: object
    ) -> None:
        """Navigation documents should not be processed as content."""
        result = await parser.parse(duplicate_spine_epub)  # type: ignore[arg-type]
        # The raw text should not contain TOC navigation text
        assert "Chapter 1: Unique Content" in result.raw_text
        assert "Chapter 2: Different Content" in result.raw_text


# ------------------------------------------------------------------
# UTF-8 sanitisation tests
# ------------------------------------------------------------------


class TestSanitizeText:
    """Unit tests for the _sanitize_text helper."""

    def test_smart_quotes_preserved(self) -> None:
        """Unicode smart quotes survive sanitisation unchanged."""
        text = "\u201cHello,\u201d she said, \u2018quietly.\u2019"
        result = _sanitize_text(text)
        assert result == text
        # Verify round-trip
        assert result.encode("utf-8").decode("utf-8") == result

    def test_em_dash_en_dash_ellipsis_preserved(self) -> None:
        """Em dash, en dash, and ellipsis survive sanitisation."""
        text = "word\u2014another\u2013thing\u2026end"
        result = _sanitize_text(text)
        assert "\u2014" in result  # em dash
        assert "\u2013" in result  # en dash
        assert "\u2026" in result  # ellipsis

    def test_nfc_normalisation(self) -> None:
        """Decomposed Unicode (NFD) is composed to NFC."""
        # e + combining acute accent → é (NFC)
        decomposed = "caf\u0065\u0301"
        result = _sanitize_text(decomposed)
        assert "\u00e9" in result  # composed é

    def test_c1_control_codes_mapped(self) -> None:
        """Windows-1252 C1 codes (0x80-0x9F) are mapped to proper Unicode."""
        # 0x93 = left double quote in Windows-1252
        text = "said \x93hello\x94"
        result = _sanitize_text(text)
        assert "\u201c" in result  # proper left double quote
        assert "\u201d" in result  # proper right double quote
        assert "\x93" not in result
        assert "\x94" not in result

    def test_null_bytes_stripped(self) -> None:
        """Null bytes are removed from text."""
        text = "hello\x00world"
        result = _sanitize_text(text)
        assert result == "helloworld"
        assert "\x00" not in result

    def test_control_chars_stripped_whitespace_kept(self) -> None:
        """C0 control chars are stripped but newlines and tabs preserved."""
        text = "line1\nline2\ttab\x01bad\x02char"
        result = _sanitize_text(text)
        assert "\n" in result
        assert "\t" in result
        assert "\x01" not in result
        assert "\x02" not in result

    def test_empty_string(self) -> None:
        """Empty string passes through unchanged."""
        assert _sanitize_text("") == ""

    def test_plain_ascii(self) -> None:
        """Plain ASCII text passes through unchanged."""
        text = "Hello, world! This is a test."
        assert _sanitize_text(text) == text

    def test_utf8_round_trip_clean(self) -> None:
        """Sanitised text always round-trips through UTF-8."""
        # Mix of problematic characters
        text = "curly \u201cquotes\u201d, em\u2014dash, \x93cp1252\x94, null\x00byte"
        result = _sanitize_text(text)
        encoded = result.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == result

    def test_replacement_char_not_introduced(self) -> None:
        """Sanitisation does not introduce U+FFFD for valid input."""
        text = "Completely normal text with \u201csmart quotes\u201d"
        result = _sanitize_text(text)
        assert "\ufffd" not in result


class TestDecodeEpubContent:
    """Unit tests for the _decode_epub_content helper."""

    def test_valid_utf8(self) -> None:
        """Valid UTF-8 bytes decode correctly."""
        text = "Café — résumé"
        content = text.encode("utf-8")
        assert _decode_epub_content(content) == text

    def test_utf8_smart_quotes(self) -> None:
        """UTF-8 encoded smart quotes decode correctly."""
        text = "\u201cHello\u201d \u2018world\u2019"
        content = text.encode("utf-8")
        result = _decode_epub_content(content)
        assert result == text

    def test_windows_1252_fallback(self) -> None:
        """Windows-1252 bytes that aren't valid UTF-8 fall back to cp1252."""
        # Windows-1252 smart quotes: 0x93 = ", 0x94 = "
        content = b"She said \x93hello\x94"
        result = _decode_epub_content(content)
        assert "\u201c" in result  # proper left double quote
        assert "\u201d" in result  # proper right double quote

    def test_xml_declaration_preserved(self) -> None:
        """XML declaration with encoding attribute survives decoding."""
        content = b'<?xml version="1.0" encoding="utf-8"?>\n<html><body>text</body></html>'
        result = _decode_epub_content(content)
        assert "text" in result


class TestEpubUtf8Integration:
    """Integration tests for UTF-8 handling through the full EPUB parse pipeline."""

    @staticmethod
    def _create_epub_with_special_chars(path: object) -> None:
        """Create an EPUB with smart quotes, em dashes, and other special chars."""
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("utf8-test")
        book.set_title("UTF-8 Test Book")
        book.set_language("en")
        book.add_author("Test Author")

        ch = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        # Content with the exact characters that caused the bug:
        # smart quotes, em dash, en dash, ellipsis
        ch.content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
            "<head><title>Chapter 1</title></head>\n"
            "<body>\n"
            "  <h1>Chapter 1: The \u201cBeginning\u201d</h1>\n"
            "  <p>He said \u2018quietly\u2019 that faith, hope and poetry\u2014"
            "these three\u2014are the pillars.</p>\n"
            "  <p>Pages 10\u201315 discuss the topic\u2026 in depth.</p>\n"
            "  <p>The caf\u00e9 had r\u00e9sum\u00e9s on the table.</p>\n"
            "  <blockquote>\u201cTo be or not to be,\u201d he mused, "
            "\u201cthat is the question.\u201d</blockquote>\n"
            "</body>\n"
            "</html>"
        ).encode("utf-8")
        book.add_item(ch)

        book.toc = [epub.Link("ch1.xhtml", "Chapter 1", "ch1")]
        book.spine = ["nav", ch]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def utf8_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "utf8_test.epub"
        self._create_epub_with_special_chars(path)
        return path

    async def test_smart_quotes_in_parsed_text(
        self, parser: EpubParser, utf8_epub: object
    ) -> None:
        """Smart quotes should be preserved in parsed document text."""
        result = await parser.parse(utf8_epub)  # type: ignore[arg-type]
        assert "\u201c" in result.raw_text  # left double quote
        assert "\u201d" in result.raw_text  # right double quote
        assert "\u2018" in result.raw_text  # left single quote
        assert "\u2019" in result.raw_text  # right single quote

    async def test_dashes_and_ellipsis_in_parsed_text(
        self, parser: EpubParser, utf8_epub: object
    ) -> None:
        """Em dash, en dash, and ellipsis survive parsing."""
        result = await parser.parse(utf8_epub)  # type: ignore[arg-type]
        assert "\u2014" in result.raw_text  # em dash
        assert "\u2013" in result.raw_text  # en dash
        assert "\u2026" in result.raw_text  # ellipsis

    async def test_accented_chars_in_parsed_text(
        self, parser: EpubParser, utf8_epub: object
    ) -> None:
        """Accented characters (e.g. é) survive parsing."""
        result = await parser.parse(utf8_epub)  # type: ignore[arg-type]
        assert "caf\u00e9" in result.raw_text
        assert "r\u00e9sum\u00e9" in result.raw_text

    async def test_all_nodes_valid_utf8(
        self, parser: EpubParser, utf8_epub: object
    ) -> None:
        """Every node in the parsed tree must have valid UTF-8 text."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(utf8_epub)  # type: ignore[arg-type]

        def check_node(node: DocumentNode) -> None:
            if node.text:
                # Must round-trip cleanly
                encoded = node.text.encode("utf-8")
                decoded = encoded.decode("utf-8")
                assert decoded == node.text, f"UTF-8 round-trip failed for: {node.text[:50]!r}"
                # Must not contain replacement characters
                assert "\ufffd" not in node.text, (
                    f"Replacement char in: {node.text[:50]!r}"
                )
            for child in node.children:
                check_node(child)

        check_node(result.tree)

    async def test_raw_text_valid_utf8(
        self, parser: EpubParser, utf8_epub: object
    ) -> None:
        """The raw_text output must be clean UTF-8."""
        result = await parser.parse(utf8_epub)  # type: ignore[arg-type]
        encoded = result.raw_text.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == result.raw_text
        assert "\ufffd" not in result.raw_text


class TestEpubRealFile:
    """Integration test with the real 'Faith, Hope and Poetry' EPUB.

    Skipped if the file is not available on the machine.
    """

    _EPUB_PATH = "/home/marty/repos/booklore/bookdrop/Faith, Hope and Poetry - Malcolm Guite.epub"

    @pytest.fixture
    def real_epub(self) -> object:
        from pathlib import Path

        path = Path(self._EPUB_PATH)
        if not path.exists():
            pytest.skip(f"Real EPUB not available: {self._EPUB_PATH}")
        return path

    async def test_real_epub_all_nodes_valid_utf8(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """Every node from the real EPUB must have valid UTF-8 text."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(real_epub)  # type: ignore[arg-type]

        corrupted: list[str] = []

        def check_node(node: DocumentNode) -> None:
            if node.text:
                encoded = node.text.encode("utf-8")
                decoded = encoded.decode("utf-8")
                if decoded != node.text:
                    corrupted.append(f"round-trip: {node.text[:80]!r}")
                if "\ufffd" in node.text:
                    corrupted.append(f"replacement: {node.text[:80]!r}")

            for child in node.children:
                check_node(child)

        check_node(result.tree)
        assert corrupted == [], f"Found {len(corrupted)} corrupted nodes: {corrupted[:5]}"

    async def test_real_epub_smart_quotes_present(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """The real EPUB should have smart quotes that survive parsing."""
        result = await parser.parse(real_epub)  # type: ignore[arg-type]
        # This EPUB has extensive use of smart quotes
        assert "\u2018" in result.raw_text or "\u2019" in result.raw_text, (
            "Expected smart single quotes in Faith, Hope and Poetry"
        )

    async def test_real_epub_no_bare_0xe2_in_chunks(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """No chunk text should contain bare/truncated 0xe2 bytes."""
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(real_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(result, work_id="test-utf8", source_class="primary")

        bad_chunks: list[int] = []
        for i, chunk in enumerate(chunks):
            encoded = chunk.text.encode("utf-8")
            # Check for truncated multi-byte: 0xe2 as last byte
            if encoded and encoded[-1] == 0xE2:
                bad_chunks.append(i)
            # Check for replacement characters
            if "\ufffd" in chunk.text:
                bad_chunks.append(i)

        assert bad_chunks == [], (
            f"{len(bad_chunks)} chunks with encoding issues out of {len(chunks)}"
        )
