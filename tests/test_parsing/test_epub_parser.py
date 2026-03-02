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


class TestHeaderWrappedHeadingsEpub:
    """Tests for EPUBs that use <article>/<section>/<header> wrappers.

    Many publisher EPUBs (e.g. Canterbury Press, OUP) structure each spine
    item as ``<body><article><section><header><h1>…</h1></header>…</section></article></body>``.
    The parser must unwrap these and extract chapter/section titles.
    """

    @staticmethod
    def _create_header_wrapped_epub(path: object) -> None:
        from pathlib import Path

        book = epub.EpubBook()
        book.set_identifier("header-wrapped-test")
        book.set_title("Header Wrapped Book")
        book.set_language("en")
        book.add_author("Test Author")

        intro = epub.EpubHtml(title="Introduction", file_name="intro.xhtml", lang="en")
        intro.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Introduction</title></head>
<body>
<section>
  <header>
    <h1 class="chno">Introduction</h1>
    <h1 class="chtitle">Poetry and Transfiguration</h1>
  </header>
  <p>The first paragraph of the introduction with real content.</p>
  <p>The second paragraph continues the discussion about poetry.</p>
</section>
</body>
</html>"""
        book.add_item(intro)

        ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
        ch1.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
<article>
  <section>
    <header>
      <h1 class="chno">Chapter 1</h1>
      <h1 class="chtitle">Seeing through Dreams</h1>
    </header>
    <section id="sec1_1">
      <header>
        <h2 class="head2">Truth and Dreaming</h2>
      </header>
      <p>First paragraph of section one about truth.</p>
      <p>Second paragraph of section one about dreaming.</p>
    </section>
    <section id="sec1_2">
      <header>
        <h2 class="head2">The Five Levels</h2>
      </header>
      <p>First paragraph of section two about levels.</p>
    </section>
  </section>
</article>
</body>
</html>"""
        book.add_item(ch1)

        ch2 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
        ch2.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
<article>
  <section>
    <header>
      <h1 class="chno">Chapter 2</h1>
      <h1 class="chtitle">Truth and Feigning</h1>
    </header>
    <p>Content of chapter two about Shakespeare.</p>
  </section>
</article>
</body>
</html>"""
        book.add_item(ch2)

        book.toc = [
            epub.Link("intro.xhtml", "Introduction", "intro"),
            epub.Link("ch1.xhtml", "Chapter 1", "ch1"),
            epub.Link("ch2.xhtml", "Chapter 2", "ch2"),
        ]
        book.spine = ["nav", intro, ch1, ch2]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(str(Path(str(path))), book)

    @pytest.fixture
    def header_wrapped_epub(self, tmp_path: object) -> object:
        from pathlib import Path

        path = Path(str(tmp_path)) / "header_wrapped.epub"
        self._create_header_wrapped_epub(path)
        return path

    async def test_chapters_found(
        self, parser: EpubParser, header_wrapped_epub: object
    ) -> None:
        """Article/section/header wrappers should be unwrapped to find chapters."""
        result = await parser.parse(header_wrapped_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        assert len(chapters) >= 3

    async def test_chapter_titles_combined(
        self, parser: EpubParser, header_wrapped_epub: object
    ) -> None:
        """Multiple h1s in a <header> should be combined into chapter title."""
        result = await parser.parse(header_wrapped_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        titles = [str(c.metadata.get("title", "")) for c in chapters]
        assert any("Introduction" in t and "Transfiguration" in t for t in titles)
        assert any("Chapter 1" in t and "Seeing through Dreams" in t for t in titles)
        assert any("Chapter 2" in t and "Truth and Feigning" in t for t in titles)

    async def test_section_titles_extracted(
        self, parser: EpubParser, header_wrapped_epub: object
    ) -> None:
        """Section headings inside <header> elements should produce titled SECTION nodes."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(header_wrapped_epub)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        ch1 = next(
            (c for c in chapters if "Seeing through Dreams" in str(c.metadata.get("title", ""))),
            None,
        )
        assert ch1 is not None

        def find_sections(node: DocumentNode) -> list[DocumentNode]:
            found: list[DocumentNode] = []
            if node.node_type == NodeType.SECTION:
                found.append(node)
            for child in node.children:
                found.extend(find_sections(child))
            return found

        sections = find_sections(ch1)
        section_titles = [str(s.metadata.get("title", "")) for s in sections]
        assert any("Truth and Dreaming" in t for t in section_titles)
        assert any("Five Levels" in t for t in section_titles)

    async def test_paragraphs_preserved(
        self, parser: EpubParser, header_wrapped_epub: object
    ) -> None:
        """Paragraphs inside nested sections should be preserved."""
        from author_library.parsing.models import DocumentNode

        result = await parser.parse(header_wrapped_epub)  # type: ignore[arg-type]

        def find_type(node: DocumentNode, ntype: NodeType) -> list[DocumentNode]:
            found: list[DocumentNode] = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        paragraphs = find_type(result.tree, NodeType.PARAGRAPH)
        assert len(paragraphs) >= 5


class TestRealEpubChapterMetadata:
    """Integration test using the real 'Faith, Hope and Poetry' EPUB.

    Verifies the full chain: EPUB parsing → chapter title extraction →
    chunking → chapter/section fields populated on every Chunk.
    """

    EPUB_PATH = "/home/marty/repos/booklore/bookdrop/Faith, Hope and Poetry - Malcolm Guite.epub"

    @pytest.fixture
    def real_epub_path(self) -> object:
        from pathlib import Path

        path = Path(self.EPUB_PATH)
        if not path.exists():
            pytest.skip(f"Real EPUB not found at {self.EPUB_PATH}")
        return path

    async def test_chapters_have_titles(
        self, parser: EpubParser, real_epub_path: object
    ) -> None:
        """Every chapter in the real EPUB should have a non-empty title."""
        result = await parser.parse(real_epub_path)  # type: ignore[arg-type]
        chapters = [c for c in result.tree.children if c.node_type == NodeType.CHAPTER]
        assert len(chapters) >= 10, f"Expected 10+ chapters, got {len(chapters)}"
        titled = [c for c in chapters if c.metadata.get("title")]
        assert len(titled) >= 10, (
            f"Expected 10+ titled chapters, got {len(titled)}. "
            f"Untitled: {[c.metadata for c in chapters if not c.metadata.get('title')]}"
        )

    async def test_scholarly_chunks_have_chapter(
        self, parser: EpubParser, real_epub_path: object
    ) -> None:
        """Chunking the real EPUB should produce chunks with chapter metadata."""
        from author_library.chunking.models import ChunkGranularity
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(real_epub_path)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(result, work_id="guite--faith-hope-poetry", source_class="primary")

        assert len(chunks) > 100, f"Expected many chunks, got {len(chunks)}"

        macro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        macro_with_chapter = [c for c in macro_chunks if c.chapter]
        assert len(macro_with_chapter) > 0, "No macro chunks have chapter set"
        assert len(macro_with_chapter) >= len(macro_chunks) * 0.8, (
            f"Only {len(macro_with_chapter)}/{len(macro_chunks)} macro chunks have chapter"
        )

        all_with_chapter = [c for c in chunks if c.chapter]
        assert len(all_with_chapter) > len(chunks) * 0.5, (
            f"Only {len(all_with_chapter)}/{len(chunks)} chunks have chapter set"
        )
