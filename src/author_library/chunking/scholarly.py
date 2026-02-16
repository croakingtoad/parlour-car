"""Scholarly prose chunking strategy (monographs, academic papers).

Implements chunking-guide Section 2:
- Macro: chapter-level summaries
- Meso: section/argument boundaries
- Micro: individual paragraphs

Special handling for footnotes, block quotations, and bibliographies.
"""

from __future__ import annotations

import structlog

from author_library.chunking._tree_utils import (
    collect_footnotes,
    collect_text,
    find_children_of_type,
    find_nodes,
    word_count,
)
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()


class ScholarlyProseStrategy(ChunkingStrategy):
    """Chunking strategy for scholarly prose (monographs, academic papers)."""

    def supported_genres(self) -> list[str]:
        return [
            "scholarly_prose",
            "monograph",
            "academic_paper",
            "academic",
            "theology",
            "literary_criticism",
        ]

    def chunk(
        self,
        document: ParsedDocument,
        work_id: str,
        source_class: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        position_counters: dict[ChunkGranularity, int] = {
            g: 0 for g in ChunkGranularity
        }

        chapters = find_nodes(document.tree, NodeType.CHAPTER)
        if not chapters:
            # No chapter structure — treat the whole document as one chapter
            chapters = [document.tree]

        for chapter_node in chapters:
            chapter_title = chapter_node.metadata.get("title", "")
            if not chapter_title and chapter_node.text:
                chapter_title = chapter_node.text.split("\n")[0][:100]
            chapter_title_str = str(chapter_title) if chapter_title else None

            # --- MACRO: chapter-level summary text ---
            chapter_text = collect_text(chapter_node)
            if not chapter_text.strip():
                continue

            macro_chunk = Chunk(
                text=chapter_text,
                granularity=ChunkGranularity.MACRO,
                work_id=work_id,
                source_class=source_class,
                chapter=chapter_title_str,
                position=position_counters[ChunkGranularity.MACRO],
                metadata={"genre": "scholarly_prose"},
            )
            position_counters[ChunkGranularity.MACRO] += 1
            chunks.append(macro_chunk)

            # Collect footnotes at chapter level for attachment to paragraphs
            footnotes = collect_footnotes(chapter_node)

            # --- MESO: section-level chunks ---
            sections = find_children_of_type(chapter_node, NodeType.SECTION)
            if not sections:
                # No explicit sections — build meso from paragraphs
                meso_chunks = self._paragraphs_to_meso_chunks(
                    chapter_node,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=chapter_title_str,
                    parent_id=macro_chunk.id,
                    footnotes=footnotes,
                    position_counters=position_counters,
                )
                chunks.extend(meso_chunks)
            else:
                for section_node in sections:
                    section_title = section_node.metadata.get("title", "")
                    if not section_title and section_node.text:
                        section_title = section_node.text.split("\n")[0][:100]
                    section_title_str = str(section_title) if section_title else None

                    section_text = collect_text(section_node)
                    if not section_text.strip():
                        continue

                    wc = word_count(section_text)

                    if wc > 500:
                        # Section too long for single meso — split at paragraph boundaries
                        meso_chunks = self._paragraphs_to_meso_chunks(
                            section_node,
                            work_id=work_id,
                            source_class=source_class,
                            chapter=chapter_title_str,
                            section=section_title_str,
                            parent_id=macro_chunk.id,
                            footnotes=footnotes,
                            position_counters=position_counters,
                        )
                        chunks.extend(meso_chunks)
                    else:
                        meso_chunk = Chunk(
                            text=section_text,
                            granularity=ChunkGranularity.MESO,
                            work_id=work_id,
                            source_class=source_class,
                            chapter=chapter_title_str,
                            section=section_title_str,
                            position=position_counters[ChunkGranularity.MESO],
                            parent_chunk_id=macro_chunk.id,
                            metadata={"genre": "scholarly_prose"},
                        )
                        position_counters[ChunkGranularity.MESO] += 1
                        chunks.append(meso_chunk)

                        # Micro chunks for paragraphs within this section
                        micro_chunks = self._paragraphs_to_micro_chunks(
                            section_node,
                            work_id=work_id,
                            source_class=source_class,
                            chapter=chapter_title_str,
                            section=section_title_str,
                            parent_id=meso_chunk.id,
                            footnotes=footnotes,
                            position_counters=position_counters,
                        )
                        chunks.extend(micro_chunks)

        logger.info(
            "scholarly_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
            macro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MACRO),
            meso=sum(1 for c in chunks if c.granularity == ChunkGranularity.MESO),
            micro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MICRO),
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _paragraphs_to_meso_chunks(
        self,
        parent_node: DocumentNode,
        *,
        work_id: str,
        source_class: str,
        chapter: str | None,
        section: str | None = None,
        parent_id: str | None = None,
        footnotes: dict[str, str],
        position_counters: dict[ChunkGranularity, int],
    ) -> list[Chunk]:
        """Group paragraphs into meso-sized chunks, emitting micro chunks within each."""
        paragraphs = _content_paragraphs(parent_node)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_paragraphs: list[DocumentNode] = []
        buffer_words = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_paragraphs, buffer_words
            if not buffer:
                return
            meso_text = "\n\n".join(buffer)
            meso_chunk = Chunk(
                text=meso_text,
                granularity=ChunkGranularity.MESO,
                work_id=work_id,
                source_class=source_class,
                chapter=chapter,
                section=section,
                position=position_counters[ChunkGranularity.MESO],
                parent_chunk_id=parent_id,
                metadata={"genre": "scholarly_prose"},
            )
            position_counters[ChunkGranularity.MESO] += 1
            chunks.append(meso_chunk)

            # Micro chunks for paragraphs that formed this meso
            for para in buffer_paragraphs:
                para_text = _paragraph_text_with_footnotes(para, footnotes)
                if not para_text.strip():
                    continue
                meta: dict[str, str | int | bool | list[str]] = {
                    "genre": "scholarly_prose",
                }
                if para.node_type == NodeType.BLOCK_QUOTE:
                    quoted = para.metadata.get("quoted_author", "")
                    if quoted:
                        meta["quoted_author"] = str(quoted)
                micro = Chunk(
                    text=para_text,
                    granularity=ChunkGranularity.MICRO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=chapter,
                    section=section,
                    position=position_counters[ChunkGranularity.MICRO],
                    parent_chunk_id=meso_chunk.id,
                    metadata=meta,
                )
                position_counters[ChunkGranularity.MICRO] += 1
                chunks.append(micro)

            buffer = []
            buffer_paragraphs = []
            buffer_words = 0

        for para_node in paragraphs:
            para_text = _paragraph_text_with_footnotes(para_node, footnotes)
            if not para_text.strip():
                continue
            wc = word_count(para_text)

            # If adding this paragraph would push past 500 words, flush first
            if buffer_words + wc > 500 and buffer:
                flush_buffer()

            buffer.append(para_text)
            buffer_paragraphs.append(para_node)
            buffer_words += wc

        # Merge undersized trailing buffer with previous meso chunk if possible
        if buffer and buffer_words < 150 and chunks:
            # Find the last meso chunk and extend it
            last_meso = next(
                (c for c in reversed(chunks) if c.granularity == ChunkGranularity.MESO),
                None,
            )
            if last_meso is not None:
                last_meso.text += "\n\n" + "\n\n".join(buffer)
                # Still emit micro chunks for these paragraphs
                for para in buffer_paragraphs:
                    para_text = _paragraph_text_with_footnotes(para, footnotes)
                    if not para_text.strip():
                        continue
                    micro = Chunk(
                        text=para_text,
                        granularity=ChunkGranularity.MICRO,
                        work_id=work_id,
                        source_class=source_class,
                        chapter=chapter,
                        section=section,
                        position=position_counters[ChunkGranularity.MICRO],
                        parent_chunk_id=last_meso.id,
                        metadata={"genre": "scholarly_prose"},
                    )
                    position_counters[ChunkGranularity.MICRO] += 1
                    chunks.append(micro)
                buffer = []
                buffer_paragraphs = []
                buffer_words = 0

        flush_buffer()
        return chunks

    def _paragraphs_to_micro_chunks(
        self,
        section_node: DocumentNode,
        *,
        work_id: str,
        source_class: str,
        chapter: str | None,
        section: str | None,
        parent_id: str,
        footnotes: dict[str, str],
        position_counters: dict[ChunkGranularity, int],
    ) -> list[Chunk]:
        """Emit micro chunks for paragraphs within a section."""
        paragraphs = _content_paragraphs(section_node)
        chunks: list[Chunk] = []
        for para_node in paragraphs:
            para_text = _paragraph_text_with_footnotes(para_node, footnotes)
            if not para_text.strip():
                continue
            meta: dict[str, str | int | bool | list[str]] = {"genre": "scholarly_prose"}
            if para_node.node_type == NodeType.BLOCK_QUOTE:
                quoted = para_node.metadata.get("quoted_author", "")
                if quoted:
                    meta["quoted_author"] = str(quoted)
            micro = Chunk(
                text=para_text,
                granularity=ChunkGranularity.MICRO,
                work_id=work_id,
                source_class=source_class,
                chapter=chapter,
                section=section,
                position=position_counters[ChunkGranularity.MICRO],
                parent_chunk_id=parent_id,
                metadata=meta,
            )
            position_counters[ChunkGranularity.MICRO] += 1
            chunks.append(micro)
        return chunks


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _content_paragraphs(node: DocumentNode) -> list[DocumentNode]:
    """Return paragraph-like children suitable for chunking.

    Includes PARAGRAPH and BLOCK_QUOTE nodes.  Skips BIBLIOGRAPHY,
    FOOTNOTE, and ENDNOTE nodes (handled separately).
    """
    skip_types = {
        NodeType.BIBLIOGRAPHY,
        NodeType.BIB_ENTRY,
        NodeType.FOOTNOTE,
        NodeType.ENDNOTE,
    }
    result: list[DocumentNode] = []
    for child in node.children:
        if child.node_type in skip_types:
            continue
        if child.node_type in (NodeType.PARAGRAPH, NodeType.BLOCK_QUOTE):
            result.append(child)
        elif child.node_type in (NodeType.SECTION, NodeType.HEADING):
            # Recurse into sub-sections to gather their paragraphs
            result.extend(_content_paragraphs(child))
        else:
            # For other node types (LIST, TABLE, etc.), treat as text block
            text = collect_text(child)
            if text.strip():
                # Wrap in a synthetic paragraph node for uniform handling
                result.append(child)
    return result


def _paragraph_text_with_footnotes(
    para: DocumentNode,
    footnotes: dict[str, str],
) -> str:
    """Return the text of a paragraph with substantive footnotes appended."""
    text = collect_text(para)
    # Check for footnote references in the paragraph's metadata
    fn_refs = para.metadata.get("footnote_refs", [])
    if isinstance(fn_refs, list):
        for ref in fn_refs:
            fn_text = footnotes.get(str(ref), "")
            if fn_text and word_count(fn_text) > 20:
                # Substantive footnote — append
                text += f"\n[Footnote: {fn_text}]"
    return text
