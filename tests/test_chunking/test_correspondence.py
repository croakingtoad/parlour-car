"""Tests for letter, blog, and interview chunking strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.correspondence import (
    BlogStrategy,
    InterviewStrategy,
    LetterStrategy,
)
from author_library.chunking.models import ChunkGranularity

if TYPE_CHECKING:
    from author_library.parsing.models import ParsedDocument


class TestLetterStrategy:
    def setup_method(self) -> None:
        self.strategy = LetterStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "letters" in genres
        assert "correspondence" in genres

    def test_produces_chunks(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        assert len(chunks) > 0

    def test_macro_for_collection(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro) == 1

    def test_meso_per_letter(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # 2 letters in fixture
        assert len(meso) == 2

    def test_recipient_metadata(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        recipients = [c.metadata.get("recipient") for c in meso]
        assert "Rowan Williams" in recipients
        assert "Luci Shaw" in recipients

    def test_date_metadata(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        dates = [c.metadata.get("date") for c in meso]
        assert "2015-03-14" in dates
        assert "2016-07-22" in dates

    def test_micro_chunks_from_paragraphs(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        micro = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]
        # Some paragraphs should become micro chunks
        assert len(micro) >= 1

    def test_genre_metadata(self, letter_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            letter_document,
            work_id="guite--collected-letters",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "correspondence"


class TestBlogStrategy:
    def setup_method(self) -> None:
        self.strategy = BlogStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "blog" in genres
        assert "blog_post" in genres

    def test_short_blog_is_single_meso(self, blog_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            blog_document,
            work_id="guite--herbert-lent",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # Short blog post should be one meso chunk
        assert len(meso) == 1

    def test_url_metadata(self, blog_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            blog_document,
            work_id="guite--herbert-lent",
            source_class="primary",
        )
        meso = next(c for c in chunks if c.granularity == ChunkGranularity.MESO)
        assert meso.metadata.get("url") == "https://blog.malcolmguite.com/rereading-herbert"

    def test_date_metadata(self, blog_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            blog_document,
            work_id="guite--herbert-lent",
            source_class="primary",
        )
        meso = next(c for c in chunks if c.granularity == ChunkGranularity.MESO)
        assert meso.metadata.get("date") == "2019-03-15"

    def test_genre_metadata(self, blog_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            blog_document,
            work_id="guite--herbert-lent",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "blog"


class TestInterviewStrategy:
    def setup_method(self) -> None:
        self.strategy = InterviewStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "interview" in genres
        assert "q_and_a" in genres

    def test_macro_is_full_interview(self, interview_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro) == 1

    def test_meso_are_qa_pairs(self, interview_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # 2 Q&A pairs in fixture
        assert len(meso) == 2

    def test_qa_splitting(self, interview_document: ParsedDocument) -> None:
        """Question and answer should be split in metadata."""
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for m in meso:
            assert "question" in m.metadata
            assert "answer" in m.metadata
            assert m.metadata.get("question_source_class") == "secondary"
            assert m.metadata.get("answer_source_class") == "primary-adjacent"

    def test_qa_text_format(self, interview_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for m in meso:
            assert m.text.startswith("Q: ")
            assert "\n\nA: " in m.text

    def test_interviewer_metadata(self, interview_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        macro = next(c for c in chunks if c.granularity == ChunkGranularity.MACRO)
        assert macro.metadata.get("interviewer") == "Jane Smith"

    def test_genre_metadata(self, interview_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            interview_document,
            work_id="guite--imagination-theology",
            source_class="secondary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "interview"
