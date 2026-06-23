"""Poetry collection chunking strategy for mixed prose + poetry books.

Handles books like "To Bless the Space Between Us" where each chapter contains:
  - A prose essay introduction
  - Multiple individual poems/blessings

Key difference from PoetryStrategy: each poem AND each essay intro becomes
its own MACRO chunk (first-class retrievable unit), rather than poems being
meso chunks under a section-level macro.

Granularity mapping:
  - MACRO: Each poem (complete text) and each essay/intro (complete text)
  - MESO: For essays — paragraph groups (~150-500 words)
  - MICRO: For essays — individual paragraphs; for long poems — stanzas

This strategy is selected by genre tags: poetry_collection, blessings,
mixed_poetry_prose.

Usage:
    Build a ParsedDocument tree with this structure:

        BOOK (root)
        ├── CHAPTER: "Introduction" (section_type=preface)
        │   └── PARAGRAPH nodes...
        ├── CHAPTER: "1. Beginnings"
        │   ├── SECTION: "Essay Introduction" (prose intro)
        │   │   └── PARAGRAPH nodes...
        │   ├── POEM: "Matins"
        │   │   └── STANZA nodes...
        │   ├── POEM: "A Morning Offering"
        │   ...
        └── CHAPTER: "To Retrieve the Lost Art of Blessing" (closing essay)
            └── PARAGRAPH nodes...

    Then use PoetryCollectionStrategy (or set genre_tags to include
    "poetry_collection") and the pipeline will produce the correct chunks.
"""

from __future__ import annotations

import re

import structlog

from author_library.chunking._tree_utils import collect_text, find_nodes, word_count
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()

# Poems shorter than this line count skip micro (stanza) chunks.
_STANZA_MICRO_LINE_THRESHOLD = 20

# Target word counts for meso chunks built from prose paragraphs.
_MESO_TARGET_MIN = 150
_MESO_TARGET_MAX = 500


class PoetryCollectionStrategy(ChunkingStrategy):
    """Chunking strategy for poetry collections with prose introductions."""

    def supported_genres(self) -> list[str]:
        return ["poetry_collection", "blessings", "mixed_poetry_prose"]

    def chunk(
        self,
        document: ParsedDocument,
        work_id: str,
        source_class: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        pos: dict[ChunkGranularity, int] = {g: 0 for g in ChunkGranularity}

        root = document.tree

        # Walk top-level children (chapters, standalone poems, essays)
        for child in root.children:
            if child.node_type == NodeType.CHAPTER:
                self._chunk_chapter(child, work_id, source_class, chunks, pos)
            elif child.node_type == NodeType.POEM:
                self._chunk_poem(child, work_id, source_class, chunks, pos, parent_id=None)
            elif child.node_type == NodeType.SECTION:
                # A standalone essay section at root level
                self._chunk_essay(child, work_id, source_class, chunks, pos)

        logger.info(
            "poetry_collection_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
            macro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MACRO),
            meso=sum(1 for c in chunks if c.granularity == ChunkGranularity.MESO),
            micro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MICRO),
        )
        return chunks

    def _chunk_chapter(
        self,
        chapter: DocumentNode,
        work_id: str,
        source_class: str,
        chunks: list[Chunk],
        pos: dict[ChunkGranularity, int],
    ) -> None:
        """Process a chapter that may contain an essay intro and poems."""
        chapter_title = str(chapter.metadata.get("title", "")) or None
        section_type = chapter.section_type.value

        # Separate children into essay sections and poems
        essay_sections: list[DocumentNode] = []
        poem_nodes: list[DocumentNode] = []

        for child in chapter.children:
            if child.node_type == NodeType.POEM:
                poem_nodes.append(child)
            elif child.node_type == NodeType.SECTION:
                # Check if this section contains poems or is prose
                inner_poems = find_nodes(child, NodeType.POEM)
                if inner_poems:
                    # Section contains poems — process them individually
                    essay_sections.append(child)  # section text itself may be essay
                    poem_nodes.extend(inner_poems)
                else:
                    essay_sections.append(child)
            elif child.node_type in (NodeType.PARAGRAPH, NodeType.HEADING, NodeType.BLOCK_QUOTE):
                # Prose paragraphs directly under the chapter — part of essay intro
                pass  # Handled below

        # If the chapter itself has text (not just in children) or has paragraph
        # children directly, treat that as the essay intro
        has_direct_prose = any(
            c.node_type == NodeType.PARAGRAPH for c in chapter.children
        )

        if has_direct_prose and not poem_nodes:
            # Pure essay chapter (like the Introduction or closing essay)
            self._chunk_essay(chapter, work_id, source_class, chunks, pos)
            return

        if has_direct_prose:
            # Chapter has both prose intro and poems
            # Extract just the prose paragraphs as an essay intro
            prose_text = self._collect_prose_text(chapter)
            if prose_text.strip() and word_count(prose_text) > 30:
                intro_title = f"{chapter_title} — Introduction" if chapter_title else "Introduction"
                self._emit_essay_chunks(
                    prose_text,
                    title=intro_title,
                    chapter_name=chapter_title,
                    section_type=section_type,
                    work_id=work_id,
                    source_class=source_class,
                    chunks=chunks,
                    pos=pos,
                )

        # Process essay sections that aren't poems
        for section in essay_sections:
            section_text = collect_text(section)
            inner_poems_in_section = find_nodes(section, NodeType.POEM)
            if inner_poems_in_section:
                # Don't double-count poem text in the essay
                continue
            if section_text.strip() and word_count(section_text) > 30:
                section_title = str(section.metadata.get("title", "")) or chapter_title
                self._emit_essay_chunks(
                    section_text,
                    title=section_title,
                    chapter_name=chapter_title,
                    section_type=section_type,
                    work_id=work_id,
                    source_class=source_class,
                    chunks=chunks,
                    pos=pos,
                )

        # Process each poem as its own macro chunk
        for poem_node in poem_nodes:
            self._chunk_poem(
                poem_node, work_id, source_class, chunks, pos,
                parent_id=None, chapter_name=chapter_title,
            )

    def _chunk_essay(
        self,
        node: DocumentNode,
        work_id: str,
        source_class: str,
        chunks: list[Chunk],
        pos: dict[ChunkGranularity, int],
    ) -> None:
        """Process a pure essay/prose node (e.g., Introduction, closing essay)."""
        text = collect_text(node)
        if not text.strip():
            return
        title = str(node.metadata.get("title", "")) or None
        section_type = node.section_type.value
        self._emit_essay_chunks(
            text,
            title=title,
            chapter_name=title,
            section_type=section_type,
            work_id=work_id,
            source_class=source_class,
            chunks=chunks,
            pos=pos,
        )

    def _emit_essay_chunks(
        self,
        text: str,
        *,
        title: str | None,
        chapter_name: str | None,
        section_type: str,
        work_id: str,
        source_class: str,
        chunks: list[Chunk],
        pos: dict[ChunkGranularity, int],
    ) -> None:
        """Emit macro + meso + micro chunks for a prose essay."""
        # MACRO: the full essay text
        macro = Chunk(
            text=text,
            granularity=ChunkGranularity.MACRO,
            work_id=work_id,
            source_class=source_class,
            chapter=chapter_name,
            section=title,
            section_type=section_type,
            position=pos[ChunkGranularity.MACRO],
            metadata={"genre": "poetry_collection", "content_type": "essay"},
        )
        pos[ChunkGranularity.MACRO] += 1
        chunks.append(macro)

        # Split into paragraphs for meso/micro
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # MESO: group paragraphs into ~150-500 word blocks
        meso_groups = self._group_paragraphs(paragraphs)
        for meso_text in meso_groups:
            meso = Chunk(
                text=meso_text,
                granularity=ChunkGranularity.MESO,
                work_id=work_id,
                source_class=source_class,
                chapter=chapter_name,
                section=title,
                section_type=section_type,
                position=pos[ChunkGranularity.MESO],
                parent_chunk_id=macro.id,
                metadata={"genre": "poetry_collection", "content_type": "essay"},
            )
            pos[ChunkGranularity.MESO] += 1
            chunks.append(meso)

            # MICRO: individual paragraphs within this meso group
            micro_paras = [p.strip() for p in re.split(r"\n\s*\n", meso_text) if p.strip()]
            for para in micro_paras:
                if word_count(para) < 15:
                    continue
                micro = Chunk(
                    text=para,
                    granularity=ChunkGranularity.MICRO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=chapter_name,
                    section=title,
                    section_type=section_type,
                    position=pos[ChunkGranularity.MICRO],
                    parent_chunk_id=meso.id,
                    metadata={"genre": "poetry_collection", "content_type": "essay"},
                )
                pos[ChunkGranularity.MICRO] += 1
                chunks.append(micro)

    def _chunk_poem(
        self,
        poem_node: DocumentNode,
        work_id: str,
        source_class: str,
        chunks: list[Chunk],
        pos: dict[ChunkGranularity, int],
        *,
        parent_id: str | None = None,
        chapter_name: str | None = None,
    ) -> None:
        """Emit a macro chunk for a poem, plus micro stanza chunks if long enough."""
        poem_text = collect_text(poem_node)
        if not poem_text.strip():
            return

        poem_title = str(poem_node.metadata.get("title", "")) or None
        first_line = poem_text.strip().split("\n")[0][:120]
        dedication = str(poem_node.metadata.get("dedication", "")) or None

        meta: dict[str, str | int | bool | list[str]] = {
            "genre": "poetry_collection",
            "content_type": "poem",
            "first_line": first_line,
        }
        if poem_title:
            meta["poem_title"] = poem_title
        if dedication:
            meta["dedication"] = dedication

        # MACRO: the complete poem
        macro = Chunk(
            text=poem_text,
            granularity=ChunkGranularity.MACRO,
            work_id=work_id,
            source_class=source_class,
            chapter=poem_title or chapter_name,
            section=poem_title,
            section_type="chapter",
            position=pos[ChunkGranularity.MACRO],
            parent_chunk_id=parent_id,
            metadata=meta,
        )
        pos[ChunkGranularity.MACRO] += 1
        chunks.append(macro)

        # MICRO: stanza-level, only for poems exceeding the line threshold
        non_blank_lines = [ln for ln in poem_text.split("\n") if ln.strip()]
        if len(non_blank_lines) > _STANZA_MICRO_LINE_THRESHOLD:
            stanzas = _split_stanzas(poem_node, poem_text)
            for idx, stanza_text in enumerate(stanzas):
                if not stanza_text.strip():
                    continue
                stanza_meta: dict[str, str | int | bool | list[str]] = {
                    "genre": "poetry_collection",
                    "content_type": "stanza",
                    "stanza_number": idx + 1,
                }
                if poem_title:
                    stanza_meta["poem_title"] = poem_title
                micro = Chunk(
                    text=stanza_text,
                    granularity=ChunkGranularity.MICRO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=poem_title or chapter_name,
                    section=poem_title,
                    section_type="chapter",
                    position=pos[ChunkGranularity.MICRO],
                    parent_chunk_id=macro.id,
                    metadata=stanza_meta,
                )
                pos[ChunkGranularity.MICRO] += 1
                chunks.append(micro)

    def _collect_prose_text(self, chapter: DocumentNode) -> str:
        """Collect text from direct paragraph children of a chapter (not poems)."""
        parts: list[str] = []
        for child in chapter.children:
            if child.node_type in (NodeType.PARAGRAPH, NodeType.BLOCK_QUOTE):
                text = collect_text(child)
                if text.strip():
                    parts.append(text)
            elif child.node_type == NodeType.HEADING:
                # Include headings in prose context
                text = collect_text(child)
                if text.strip():
                    parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _group_paragraphs(paragraphs: list[str]) -> list[str]:
        """Group paragraphs into meso-sized blocks (~150-500 words)."""
        groups: list[str] = []
        current: list[str] = []
        current_words = 0

        for para in paragraphs:
            para_words = word_count(para)
            if current_words + para_words > _MESO_TARGET_MAX and current:
                groups.append("\n\n".join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words

                if current_words >= _MESO_TARGET_MIN:
                    groups.append("\n\n".join(current))
                    current = []
                    current_words = 0

        if current:
            # Merge short trailing group with previous if possible
            if groups and current_words < _MESO_TARGET_MIN:
                groups[-1] += "\n\n" + "\n\n".join(current)
            else:
                groups.append("\n\n".join(current))

        return groups


def _split_stanzas(poem_node: DocumentNode, poem_text: str) -> list[str]:
    """Extract stanzas from a poem node — prefer STANZA children, fall back to blank-line split."""
    stanza_nodes = find_nodes(poem_node, NodeType.STANZA)
    if stanza_nodes:
        return [collect_text(s) for s in stanza_nodes if collect_text(s).strip()]

    # Fall back to blank-line splitting
    stanzas: list[str] = []
    current: list[str] = []
    for line in poem_text.split("\n"):
        if not line.strip():
            if current:
                stanzas.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        stanzas.append("\n".join(current))
    return stanzas
