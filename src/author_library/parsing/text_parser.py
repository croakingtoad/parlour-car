"""Plain text document parser with heuristic structure detection.

Uses blank-line-separated paragraph detection and heuristic heading
recognition (ALL CAPS, "Chapter N", Roman numerals) to build a document tree.
"""

from __future__ import annotations

import re
from pathlib import Path

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

# Chapter heading patterns
_CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+(\d+|[ivxlcdm]+)", re.IGNORECASE),
    re.compile(r"^part\s+(\d+|[ivxlcdm]+|one|two|three|four|five)", re.IGNORECASE),
    re.compile(
        r"^(prologue|epilogue|introduction|conclusion|afterword|foreword|preface)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(appendix)\s*[a-z]?$", re.IGNORECASE),
]

# All-caps line that looks like a heading (at least 2 words, max ~80 chars)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\d:.,!?\-\u2014\u2013]{2,80}$")

# Roman numeral standalone line
_ROMAN_RE = re.compile(r"^[IVXLCDM]+\.?$")


class TextParser(DocumentParser):
    """Parser for plain text documents."""

    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]

    async def parse(self, source: Path | str) -> ParsedDocument:
        source = Path(source)
        if not source.exists():
            raise ParsingError(
                f"Text file not found: {source}",
                context={"path": str(source)},
            )

        log.info("parsing_text", path=str(source))

        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = source.read_text(encoding="latin-1")
            except Exception as exc:
                raise ParsingError(
                    f"Failed to read text file: {exc}",
                    context={"path": str(source)},
                    cause=exc,
                ) from exc

        warnings: list[str] = []
        if not content.strip():
            warnings.append("Text file is empty")

        root = DocumentNode(node_type=NodeType.BOOK)
        raw_text_parts: list[str] = []

        # Split into blocks by blank lines
        blocks = re.split(r"\n\s*\n", content)

        current_chapter: DocumentNode | None = None

        for block in blocks:
            text = block.strip()
            if not text:
                continue

            raw_text_parts.append(text)

            # Check if this block is a chapter heading
            if self._is_chapter_heading(text):
                current_chapter = DocumentNode(
                    node_type=NodeType.CHAPTER,
                    metadata={"title": text},
                )
                current_chapter.children.append(
                    DocumentNode(
                        node_type=NodeType.HEADING,
                        text=text,
                    )
                )
                root.children.append(current_chapter)
                continue

            # Check if this block is a section heading (ALL CAPS or short bold-like)
            if self._is_section_heading(text):
                section = DocumentNode(
                    node_type=NodeType.SECTION,
                    metadata={"title": text},
                )
                section.children.append(
                    DocumentNode(
                        node_type=NodeType.HEADING,
                        text=text,
                    )
                )
                target = current_chapter or root
                target.children.append(section)
                continue

            # Regular paragraph
            para = DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
            target = current_chapter or root
            target.children.append(para)

        metadata = DocumentMetadata(
            word_count=len(content.split()),
        )

        # Try to derive title from first heading
        if root.children:
            first = root.children[0]
            if first.node_type in (NodeType.CHAPTER, NodeType.HEADING):
                title = first.metadata.get("title") or first.text
                if isinstance(title, str):
                    metadata.title = title

        return ParsedDocument(
            source_path=str(source),
            format="txt",
            metadata=metadata,
            tree=root,
            raw_text=content,
            parse_warnings=warnings,
        )

    @staticmethod
    def _is_chapter_heading(text: str) -> bool:
        """Check if text matches a chapter heading pattern."""
        # Multi-line blocks are not headings
        if "\n" in text:
            return False
        return any(pattern.match(text.strip()) for pattern in _CHAPTER_PATTERNS)

    @staticmethod
    def _is_section_heading(text: str) -> bool:
        """Check if text looks like a section heading."""
        # Multi-line → not a heading
        if "\n" in text:
            return False
        stripped = text.strip()
        # Short ALL CAPS lines
        if _ALL_CAPS_RE.match(stripped) and len(stripped.split()) >= 2:
            return True
        # Roman numeral lines
        return bool(_ROMAN_RE.match(stripped))
