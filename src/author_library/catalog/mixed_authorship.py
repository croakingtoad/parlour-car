"""Mixed-authorship handling for The Author Library.

Detects and handles documents that contain material from multiple authorial
voices — edited collections with subject-author chapters, interviews with
Q&A splitting, and foreword/afterword detection. Each detected segment
gets its own classification annotation for downstream processing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from author_library.catalog.models import SourceClass

if TYPE_CHECKING:
    from author_library.parsing.models import DocumentNode, ParsedDocument

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Segment model
# ---------------------------------------------------------------------------


class SegmentType(StrEnum):
    """Types of segments within a mixed-authorship document."""

    CHAPTER = "chapter"
    FOREWORD = "foreword"
    AFTERWORD = "afterword"
    INTRODUCTION = "introduction"
    INTERVIEW_QUESTION = "interview_question"
    INTERVIEW_RESPONSE = "interview_response"
    EDITORIAL_FRAMING = "editorial_framing"
    APPENDIX = "appendix"


class AuthorshipSegment(BaseModel):
    """A segment of a mixed-authorship document with its own classification.

    Each segment identifies the portion of the document, the attributed
    author of that segment, and the appropriate source classification
    for downstream processing.
    """

    segment_type: SegmentType
    title: str
    attributed_author: str
    source_class: SourceClass
    start_node_id: str | None = None
    end_node_id: str | None = None
    text_preview: str = ""
    is_primary_adjacent: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    notes: str = ""


class MixedAuthorshipResult(BaseModel):
    """Result of mixed-authorship analysis for a document."""

    is_mixed: bool = False
    segments: list[AuthorshipSegment] = Field(default_factory=list)
    primary_adjacent_count: int = 0
    requires_extraction: bool = False
    analysis_notes: str = ""


# ---------------------------------------------------------------------------
# Detection constants
# ---------------------------------------------------------------------------

_FOREWORD_INDICATORS = frozenset({
    "foreword",
    "preface",
    "fore-word",
})

_AFTERWORD_INDICATORS = frozenset({
    "afterword",
    "postscript",
    "epilogue",
    "after-word",
})

_INTRODUCTION_INDICATORS = frozenset({
    "introduction",
    "editor's introduction",
    "editor's note",
    "editorial introduction",
})

_INTERVIEW_INDICATORS = frozenset({
    "interview",
    "conversation",
    "dialogue",
    "q&a",
    "questions and answers",
    "in conversation with",
    "a conversation with",
})


# ---------------------------------------------------------------------------
# Mixed-authorship analyzer
# ---------------------------------------------------------------------------


class MixedAuthorshipAnalyzer:
    """Analyzes documents for mixed-authorship content.

    Detects:
    - Edited collections containing chapters by the subject author
    - Interviews with Q&A splitting
    - Foreword/afterword by subject author in another's book (or vice versa)
    """

    def __init__(self, subject_author: str) -> None:
        self._subject_author = subject_author
        self._subject_author_lower = subject_author.lower()

    def analyze(
        self,
        document: ParsedDocument,
        *,
        document_source_class: SourceClass,
    ) -> MixedAuthorshipResult:
        """Analyze a document for mixed-authorship segments.

        Args:
            document: The parsed document to analyze.
            document_source_class: The overall source classification of the document.

        Returns:
            MixedAuthorshipResult with detected segments.
        """
        segments: list[AuthorshipSegment] = []

        # Only analyze secondary or contextual sources for mixed content.
        # Primary sources may have forewords by others, which we detect separately.
        if document_source_class == SourceClass.PRIMARY:
            segments.extend(self._detect_foreign_segments_in_primary(document))
        elif document_source_class == SourceClass.SECONDARY:
            segments.extend(self._detect_primary_segments_in_secondary(document))
            segments.extend(self._detect_interview_segments(document))
        elif document_source_class == SourceClass.CONTEXTUAL:
            # Contextual sources don't typically contain subject-author content
            pass

        primary_adjacent_count = sum(1 for s in segments if s.is_primary_adjacent)

        result = MixedAuthorshipResult(
            is_mixed=len(segments) > 0,
            segments=segments,
            primary_adjacent_count=primary_adjacent_count,
            requires_extraction=primary_adjacent_count > 0,
            analysis_notes=self._build_analysis_notes(segments, document_source_class),
        )

        if result.is_mixed:
            log.info(
                "mixed_authorship_detected",
                title=document.metadata.title,
                segment_count=len(segments),
                primary_adjacent=primary_adjacent_count,
            )

        return result

    # -----------------------------------------------------------------------
    # Detection methods
    # -----------------------------------------------------------------------

    def _detect_primary_segments_in_secondary(
        self,
        document: ParsedDocument,
    ) -> list[AuthorshipSegment]:
        """Detect subject-author chapters in an edited collection or secondary work.

        Per classification-examples.md §2: "If Guite contributed a chapter,
        that specific chapter is primary material."
        """
        segments: list[AuthorshipSegment] = []

        for node in self._walk_chapter_nodes(document.tree):
            title_lower = self._node_title(node).lower()
            text_lower = (node.text or "").lower()

            # Check if chapter metadata attributes authorship to subject
            chapter_author = node.metadata.get("author", "")
            if isinstance(chapter_author, str) and self._is_subject_author_name(chapter_author):
                segments.append(
                    AuthorshipSegment(
                        segment_type=SegmentType.CHAPTER,
                        title=self._node_title(node),
                        attributed_author=self._subject_author,
                        source_class=SourceClass.PRIMARY,
                        start_node_id=node.id,
                        text_preview=self._text_preview(node),
                        is_primary_adjacent=True,
                        confidence=0.9,
                        notes=(
                            "Chapter attributed to subject author in edited collection. "
                            "Extract and process as primary."
                        ),
                    )
                )
                continue

            # Check if chapter title contains "by {subject_author}"
            if self._subject_author_lower in title_lower and "by" in title_lower:
                segments.append(
                    AuthorshipSegment(
                        segment_type=SegmentType.CHAPTER,
                        title=self._node_title(node),
                        attributed_author=self._subject_author,
                        source_class=SourceClass.PRIMARY,
                        start_node_id=node.id,
                        text_preview=self._text_preview(node),
                        is_primary_adjacent=True,
                        confidence=0.8,
                        notes=(
                            "Chapter title suggests subject-author contribution. "
                            "Verify and extract as primary."
                        ),
                    )
                )
                continue

            # Detect foreword/introduction by subject author
            if (
                self._is_foreword_or_intro(title_lower)
                and self._mentions_subject_author(text_lower)
                and self._first_person_voice_detected(node)
            ):
                segments.append(
                    AuthorshipSegment(
                        segment_type=SegmentType.FOREWORD,
                        title=self._node_title(node),
                        attributed_author=self._subject_author,
                        source_class=SourceClass.PRIMARY,
                        start_node_id=node.id,
                        text_preview=self._text_preview(node),
                        is_primary_adjacent=True,
                        confidence=0.7,
                        notes=(
                            "Foreword/introduction with first-person voice "
                            "mentioning subject author — may be authored by "
                            "subject. Verify."
                        ),
                    )
                )

        return segments

    def _detect_foreign_segments_in_primary(
        self,
        document: ParsedDocument,
    ) -> list[AuthorshipSegment]:
        """Detect non-subject-author segments in a primary source.

        Per classification-examples.md §4: "Rowan Williams writes the foreword
        to Faith, Hope and Poetry" → secondary.
        """
        segments: list[AuthorshipSegment] = []

        for node in self._walk_chapter_nodes(document.tree):
            title_lower = self._node_title(node).lower()

            # Check for forewords/introductions by someone else
            if self._is_foreword_or_intro(title_lower):
                chapter_author = node.metadata.get("author", "")
                if (
                    isinstance(chapter_author, str)
                    and chapter_author
                    and not self._is_subject_author_name(chapter_author)
                ):
                    segments.append(
                        AuthorshipSegment(
                            segment_type=SegmentType.FOREWORD,
                            title=self._node_title(node),
                            attributed_author=str(chapter_author),
                            source_class=SourceClass.SECONDARY,
                            start_node_id=node.id,
                            text_preview=self._text_preview(node),
                            is_primary_adjacent=False,
                            confidence=0.85,
                            notes=(
                                "Foreword/introduction in primary work by another author. "
                                "Process as secondary — do not include in voice profile."
                            ),
                        )
                    )

        return segments

    def _detect_interview_segments(
        self,
        document: ParsedDocument,
    ) -> list[AuthorshipSegment]:
        """Detect interview Q&A patterns for response extraction.

        Per classification-examples.md §3: Interviewer questions are secondary,
        subject author responses are primary-adjacent.
        """
        title = document.metadata.title or ""
        title_lower = title.lower()

        # Only process if the document appears to be an interview
        is_interview = any(indicator in title_lower for indicator in _INTERVIEW_INDICATORS)
        if not is_interview:
            # Also check metadata
            genre_hint = str(document.tree.metadata.get("genre", "")).lower()
            is_interview = "interview" in genre_hint

        if not is_interview:
            return []

        # Mark the entire document as having interview structure
        segments = [
            AuthorshipSegment(
                segment_type=SegmentType.INTERVIEW_RESPONSE,
                title=f"Interview responses — {title}",
                attributed_author=self._subject_author,
                source_class=SourceClass.PRIMARY,
                is_primary_adjacent=True,
                confidence=0.75,
                text_preview="",
                notes=(
                    "Interview detected. Subject author's responses are primary-adjacent. "
                    "Interviewer questions and framing are secondary. "
                    "Q&A splitting required during chunking phase."
                ),
            ),
            AuthorshipSegment(
                segment_type=SegmentType.INTERVIEW_QUESTION,
                title=f"Interview questions/framing — {title}",
                attributed_author=document.metadata.author or "Interviewer",
                source_class=SourceClass.SECONDARY,
                is_primary_adjacent=False,
                confidence=0.75,
                text_preview="",
                notes=(
                    "Interviewer questions and editorial framing. "
                    "Secondary source — not subject author's voice."
                ),
            ),
        ]

        return segments

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _walk_chapter_nodes(self, root: DocumentNode) -> list[DocumentNode]:
        """Walk the document tree and yield chapter-level nodes."""
        from author_library.parsing.models import NodeType

        chapters: list[DocumentNode] = []

        def _walk(node: DocumentNode) -> None:
            if node.node_type in (
                NodeType.CHAPTER,
                NodeType.FRONT_MATTER,
                NodeType.SECTION,
            ):
                chapters.append(node)
            for child in node.children:
                _walk(child)

        _walk(root)
        return chapters

    def _is_subject_author_name(self, name: str) -> bool:
        """Check if a name matches the subject author (fuzzy)."""
        name_lower = name.lower().strip()
        subject_lower = self._subject_author_lower

        # Exact match
        if name_lower == subject_lower:
            return True

        # Check if subject author's last name appears in name
        subject_parts = subject_lower.split()
        if subject_parts:
            last_name = subject_parts[-1]
            if last_name in name_lower and len(last_name) > 2:
                return True

        return False

    def _mentions_subject_author(self, text: str) -> bool:
        """Check if text mentions the subject author."""
        return self._subject_author_lower in text

    def _first_person_voice_detected(self, node: DocumentNode) -> bool:
        """Heuristic: detect first-person voice in a node's text."""
        text = (node.text or "").lower()
        if len(text) < 50:
            # Gather text from children
            child_texts = []
            for child in node.children:
                if child.text:
                    child_texts.append(child.text.lower())
            text = " ".join(child_texts)

        first_person_markers = {
            "i have", "i was", "i am", "my own",
            "in my", "i believe", "i wrote",
        }
        return any(marker in text for marker in first_person_markers)

    @staticmethod
    def _is_foreword_or_intro(title_lower: str) -> bool:
        """Check if a title indicates a foreword, preface, or introduction."""
        return any(
            indicator in title_lower
            for indicator in _FOREWORD_INDICATORS | _INTRODUCTION_INDICATORS
        )

    @staticmethod
    def _is_afterword(title_lower: str) -> bool:
        """Check if a title indicates an afterword."""
        return any(indicator in title_lower for indicator in _AFTERWORD_INDICATORS)

    @staticmethod
    def _node_title(node: DocumentNode) -> str:
        """Extract a title from a node, falling back to metadata or node type."""
        title = node.metadata.get("title", "")
        if isinstance(title, str) and title:
            return title
        # Use the first heading child if available
        from author_library.parsing.models import NodeType

        for child in node.children:
            if child.node_type == NodeType.HEADING and child.text:
                return child.text
        return f"[{node.node_type.value}]"

    @staticmethod
    def _text_preview(node: DocumentNode) -> str:
        """Extract a text preview from a node (first 200 chars)."""
        text = node.text or ""
        if not text:
            for child in node.children:
                if child.text:
                    text = child.text
                    break
        return text[:200]

    @staticmethod
    def _build_analysis_notes(
        segments: list[AuthorshipSegment],
        document_source_class: SourceClass,
    ) -> str:
        """Build human-readable analysis notes."""
        if not segments:
            return f"No mixed-authorship detected in {document_source_class.value} source."

        primary_count = sum(1 for s in segments if s.is_primary_adjacent)
        secondary_count = len(segments) - primary_count

        parts = [
            f"Mixed-authorship detected: {len(segments)} segment(s) identified.",
        ]
        if primary_count:
            parts.append(
                f"  {primary_count} segment(s) classified as primary-adjacent "
                f"(subject author content within a {document_source_class.value} work)."
            )
        if secondary_count:
            parts.append(
                f"  {secondary_count} segment(s) classified as secondary "
                f"(other author content)."
            )
        return "\n".join(parts)
