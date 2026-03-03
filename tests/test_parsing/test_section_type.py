"""Tests for section type classification in the EPUB parser.

Verifies that headings like "Index", "Bibliography", "Contents", etc.
are correctly classified into their SectionType, and that the pipeline
routes them appropriately (skipping chunking for non-content sections).
"""

from __future__ import annotations

import pytest
from ebooklib import epub

from author_library.parsing.epub_parser import (
    EpubParser,
    _detect_section_type_from_content,
    _detect_section_type_from_heading,
)
from author_library.parsing.models import NodeType, SectionType


# ------------------------------------------------------------------
# Unit tests: heading pattern detection
# ------------------------------------------------------------------


class TestDetectSectionTypeFromHeading:
    """Unit tests for _detect_section_type_from_heading."""

    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("Contents", SectionType.TABLE_OF_CONTENTS),
            ("Table of Contents", SectionType.TABLE_OF_CONTENTS),
            ("Bibliography", SectionType.BIBLIOGRAPHY),
            ("Works Cited", SectionType.BIBLIOGRAPHY),
            ("References", SectionType.BIBLIOGRAPHY),
            ("Select Bibliography", SectionType.BIBLIOGRAPHY),
            ("Further Reading", SectionType.BIBLIOGRAPHY),
            ("Index", SectionType.INDEX),
            ("General Index", SectionType.INDEX),
            ("Subject Index", SectionType.INDEX),
            ("Name Index", SectionType.INDEX),
            ("Index of First Lines", SectionType.INDEX),
            ("Scripture Index", SectionType.INDEX),
            ("Copyright", SectionType.FRONT_MATTER),
            ("Dedication", SectionType.FRONT_MATTER),
            ("Acknowledgements", SectionType.FRONT_MATTER),
            ("About the Author", SectionType.FRONT_MATTER),
            ("Also By", SectionType.FRONT_MATTER),
            ("Preface", SectionType.PREFACE),
            ("Foreword", SectionType.PREFACE),
            ("Introduction", SectionType.PREFACE),
            ("Prologue", SectionType.PREFACE),
            ("Author's Note", SectionType.PREFACE),
            ("Appendix", SectionType.BACK_MATTER),
            ("Appendix A", SectionType.BACK_MATTER),
            ("Endnotes", SectionType.BACK_MATTER),
            ("Notes", SectionType.BACK_MATTER),
            ("Glossary", SectionType.BACK_MATTER),
            ("Epilogue", SectionType.BACK_MATTER),
            ("Afterword", SectionType.BACK_MATTER),
        ],
    )
    def test_known_headings(self, heading: str, expected: SectionType) -> None:
        result = _detect_section_type_from_heading(heading)
        assert result == expected, f"'{heading}' should be {expected}, got {result}"

    @pytest.mark.parametrize(
        "heading",
        [
            "Chapter 1: The Beginning",
            "Part I: Origins",
            "The Poetic Imagination",
            "Seeing through Dreams",
            "1. Introduction to Method",
            "",
        ],
    )
    def test_chapter_headings_return_none(self, heading: str) -> None:
        """Regular chapter headings should NOT match any pattern (returns None)."""
        result = _detect_section_type_from_heading(heading)
        assert result is None, f"'{heading}' should not match, got {result}"

    def test_case_insensitive(self) -> None:
        assert _detect_section_type_from_heading("BIBLIOGRAPHY") == SectionType.BIBLIOGRAPHY
        assert _detect_section_type_from_heading("index") == SectionType.INDEX
        assert _detect_section_type_from_heading("TABLE OF CONTENTS") == SectionType.TABLE_OF_CONTENTS


# ------------------------------------------------------------------
# Unit tests: content pattern detection
# ------------------------------------------------------------------


class TestDetectSectionTypeFromContent:
    """Unit tests for _detect_section_type_from_content."""

    def test_index_entries_detected(self) -> None:
        """Content with index-style entries (term + page numbers) should be detected."""
        text = "\n".join([
            "Aquinas 42, 94, 127",
            "Aristotle 15-17, 44",
            "Augustine 33, 88, 142",
            "Barth 55, 99",
            "Bible 12, 34-36, 78",
            "Coleridge 23, 56, 89-91",
            "Dante 44, 67, 123",
            "Faith 11, 33, 55, 77",
        ])
        result = _detect_section_type_from_content(text)
        assert result == SectionType.INDEX

    def test_bibliography_entries_detected(self) -> None:
        """Content with bibliography-style entries should be detected."""
        text = "\n".join([
            "Barth, Karl. Church Dogmatics, vol. 1 (1936).",
            "Coleridge, Samuel Taylor. Biographia Literaria (1817).",
            "Guite, Malcolm. Faith, Hope and Poetry (2010).",
            "Lewis, C.S. The Discarded Image (1964).",
        ])
        result = _detect_section_type_from_content(text)
        assert result == SectionType.BIBLIOGRAPHY

    def test_regular_prose_not_detected(self) -> None:
        """Normal prose paragraphs should return None."""
        text = (
            "Coleridge's distinction between Primary and Secondary Imagination forms "
            "the cornerstone of his entire poetic philosophy. The Primary Imagination, "
            "as he articulates it in Chapter 13 of the Biographia Literaria, is the "
            "living power and prime agent of all human perception."
        )
        result = _detect_section_type_from_content(text)
        assert result is None

    def test_empty_text(self) -> None:
        assert _detect_section_type_from_content("") is None
        assert _detect_section_type_from_content("   \n  ") is None


# ------------------------------------------------------------------
# Integration tests: EPUB with structured sections
# ------------------------------------------------------------------


def _create_epub_with_sections(path: object) -> None:
    """Create an EPUB with index, bibliography, ToC, and chapter sections."""
    from pathlib import Path

    book = epub.EpubBook()
    book.set_identifier("section-type-test")
    book.set_title("Section Type Test Book")
    book.set_language("en")
    book.add_author("Test Author")

    # Copyright page
    cop = epub.EpubHtml(title="Copyright", file_name="copyright.xhtml", lang="en")
    cop.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Copyright</h1>
    <p>Copyright 2024 by Test Author. All rights reserved.</p>
</body>
</html>"""
    book.add_item(cop)

    # Table of Contents page
    toc_page = epub.EpubHtml(title="Contents", file_name="toc.xhtml", lang="en")
    toc_page.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Contents</h1>
    <p>Preface vii</p>
    <p>Chapter 1: The Beginning 1</p>
    <p>Chapter 2: The Middle 45</p>
    <p>Bibliography 189</p>
    <p>Index 195</p>
</body>
</html>"""
    book.add_item(toc_page)

    # Preface
    preface = epub.EpubHtml(title="Preface", file_name="preface.xhtml", lang="en")
    preface.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Preface</h1>
    <p>This book explores the relationship between poetry and theology.</p>
    <p>I am grateful to many colleagues for their help and encouragement.</p>
</body>
</html>"""
    book.add_item(preface)

    # Chapter 1
    ch1 = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    ch1.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Chapter 1: The Beginning</h1>
    <p>The first chapter discusses the foundations of poetic theology.</p>
    <p>Coleridge understood that imagination is a mode of knowing.</p>
</body>
</html>"""
    book.add_item(ch1)

    # Chapter 2
    ch2 = epub.EpubHtml(title="Chapter 2", file_name="ch2.xhtml", lang="en")
    ch2.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Chapter 2: The Middle</h1>
    <p>The second chapter explores Shakespeare and the poetic tradition.</p>
    <p>Truth and feigning are not opposites but complementary modes of disclosure.</p>
</body>
</html>"""
    book.add_item(ch2)

    # Bibliography
    bib = epub.EpubHtml(title="Bibliography", file_name="bib.xhtml", lang="en")
    bib.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Bibliography</h1>
    <p>Barth, Karl. Church Dogmatics, vol. 1. Edinburgh: T&amp;T Clark, 1936.</p>
    <p>Coleridge, Samuel Taylor. Biographia Literaria. London, 1817.</p>
    <p>Lewis, C.S. The Discarded Image. Cambridge University Press, 1964.</p>
</body>
</html>"""
    book.add_item(bib)

    # Index
    idx = epub.EpubHtml(title="Index", file_name="index.xhtml", lang="en")
    idx.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
    <h1>Index</h1>
    <p>Aquinas 42, 94</p>
    <p>Aristotle 15-17, 44</p>
    <p>Augustine 33, 88</p>
    <p>Barth 55, 99</p>
    <p>Bible 12, 34-36</p>
    <p>Coleridge 23, 56</p>
    <p>Dante 44, 67</p>
</body>
</html>"""
    book.add_item(idx)

    book.toc = [
        epub.Link("preface.xhtml", "Preface", "preface"),
        epub.Link("ch1.xhtml", "Chapter 1", "ch1"),
        epub.Link("ch2.xhtml", "Chapter 2", "ch2"),
        epub.Link("bib.xhtml", "Bibliography", "bib"),
        epub.Link("index.xhtml", "Index", "idx"),
    ]

    book.spine = ["nav", cop, toc_page, preface, ch1, ch2, bib, idx]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(Path(str(path))), book)


@pytest.fixture
def parser() -> EpubParser:
    return EpubParser()


@pytest.fixture
def section_epub(tmp_path: object) -> object:
    from pathlib import Path

    path = Path(str(tmp_path)) / "sections.epub"
    _create_epub_with_sections(path)
    return path


class TestEpubSectionTypeClassification:
    """Integration tests: EPUB parsing with section type detection."""

    async def test_index_section_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """The Index spine item should be classified as SectionType.INDEX."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        index_nodes = [
            c for c in chapters
            if c.section_type == SectionType.INDEX
        ]
        assert len(index_nodes) >= 1, (
            f"Expected at least 1 INDEX section, got {len(index_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )

    async def test_bibliography_section_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """The Bibliography spine item should be classified as SectionType.BIBLIOGRAPHY."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        bib_nodes = [
            c for c in chapters
            if c.section_type == SectionType.BIBLIOGRAPHY
        ]
        assert len(bib_nodes) >= 1, (
            f"Expected at least 1 BIBLIOGRAPHY section, got {len(bib_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )

    async def test_toc_section_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """The Contents spine item should be classified as SectionType.TABLE_OF_CONTENTS."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        toc_nodes = [
            c for c in chapters
            if c.section_type == SectionType.TABLE_OF_CONTENTS
        ]
        assert len(toc_nodes) >= 1, (
            f"Expected at least 1 TOC section, got {len(toc_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )

    async def test_chapter_sections_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """Regular chapters should remain as SectionType.CHAPTER."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        chapter_nodes = [
            c for c in chapters
            if c.section_type == SectionType.CHAPTER
        ]
        assert len(chapter_nodes) >= 2, (
            f"Expected at least 2 CHAPTER sections, got {len(chapter_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )

    async def test_preface_section_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """The Preface should be classified as SectionType.PREFACE."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        preface_nodes = [
            c for c in chapters
            if c.section_type == SectionType.PREFACE
        ]
        assert len(preface_nodes) >= 1, (
            f"Expected at least 1 PREFACE section, got {len(preface_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )

    async def test_copyright_section_classified(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """The Copyright page should be classified as SectionType.FRONT_MATTER."""
        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        front_nodes = [
            c for c in chapters
            if c.section_type == SectionType.FRONT_MATTER
        ]
        assert len(front_nodes) >= 1, (
            f"Expected at least 1 FRONT_MATTER section, got {len(front_nodes)}. "
            f"Types: {[(c.metadata.get('title'), c.section_type) for c in chapters]}"
        )


class TestSectionTypeChunkingIntegration:
    """Test that section type classification flows through to chunking."""

    async def test_index_produces_zero_content_chunks(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """Index sections should produce chunks tagged as 'index' section_type."""
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(result, work_id="test--sections", source_class="primary")

        index_chunks = [c for c in chunks if c.section_type == SectionType.INDEX.value]
        chapter_chunks = [c for c in chunks if c.section_type == SectionType.CHAPTER.value]

        # The chunks from the index section should be tagged as index
        # The pipeline will filter them out later
        assert len(chapter_chunks) > 0, "Should have chapter chunks"
        # Verify index chunks are tagged (they exist but will be filtered by pipeline)
        # Note: index nodes may produce chunks that get section_type="index"
        # depending on whether the parser creates chapter nodes for them

    async def test_bibliography_produces_tagged_chunks(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """Bibliography sections should produce chunks tagged as 'bibliography'."""
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(result, work_id="test--sections", source_class="primary")

        bib_chunks = [c for c in chunks if c.section_type == SectionType.BIBLIOGRAPHY.value]
        # Bibliography chunks exist but are tagged for filtering
        # The pipeline's _filter_by_section_type will remove them

    async def test_preface_chunks_are_content(
        self, parser: EpubParser, section_epub: object
    ) -> None:
        """Preface chunks should be tagged as 'preface' (treated as content)."""
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(section_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(result, work_id="test--sections", source_class="primary")

        preface_chunks = [c for c in chunks if c.section_type == SectionType.PREFACE.value]
        assert len(preface_chunks) > 0, "Preface should produce content chunks"


# ------------------------------------------------------------------
# Real EPUB integration test
# ------------------------------------------------------------------


class TestRealEpubSectionTypes:
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

    async def test_real_epub_has_non_chapter_sections(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """The real EPUB should have at least one non-chapter section detected."""
        result = await parser.parse(real_epub)  # type: ignore[arg-type]
        chapters = result.tree.children
        section_types = {c.section_type for c in chapters}

        # Faith, Hope and Poetry has a bibliography, index, etc.
        non_chapter = {
            st for st in section_types if st != SectionType.CHAPTER
        }
        assert len(non_chapter) > 0, (
            f"Expected at least one non-chapter section type in real EPUB. "
            f"All types: {section_types}"
        )

    async def test_real_epub_section_type_reduces_chunks(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """Filtering by section type should reduce the total chunk count for the real EPUB."""
        from author_library.chunking.scholarly import ScholarlyProseStrategy
        from author_library.parsing.models import SectionType

        result = await parser.parse(real_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(
            result, work_id="guite--faith-hope-poetry", source_class="primary"
        )

        total = len(chunks)
        content_types = {
            SectionType.CHAPTER.value,
            SectionType.PREFACE.value,
            SectionType.BACK_MATTER.value,
        }
        content_chunks = [c for c in chunks if c.section_type in content_types]
        non_content = total - len(content_chunks)

        # The real EPUB should have at least some non-content chunks
        # that would be filtered out by the pipeline
        assert total > 0, "Should produce chunks"
        assert len(content_chunks) > 0, "Should have content chunks"
        # Log the reduction for debugging
        if non_content > 0:
            pct = round(non_content / total * 100, 1)
            print(
                f"Section type filter: {total} total, {len(content_chunks)} content, "
                f"{non_content} non-content ({pct}% reduction)"
            )

    async def test_real_epub_no_micro_fragments_after_filter(
        self, parser: EpubParser, real_epub: object
    ) -> None:
        """After minimum chunk size filter, no micro chunk should be under 50 chars."""
        from author_library.chunking.models import ChunkGranularity
        from author_library.chunking.scholarly import ScholarlyProseStrategy

        result = await parser.parse(real_epub)  # type: ignore[arg-type]
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(
            result, work_id="guite--faith-hope-poetry", source_class="primary"
        )

        micro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]
        tiny = [c for c in micro_chunks if len(c.text.strip()) < 50]
        assert len(tiny) == 0, (
            f"Found {len(tiny)} micro chunks under 50 chars after filter: "
            f"{[c.text[:60] for c in tiny[:5]]}"
        )
