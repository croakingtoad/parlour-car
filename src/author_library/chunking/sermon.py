"""Sermon and lecture chunking strategy.

Implements chunking-guide Section 4:
- Macro: entire sermon/lecture summary
- Meso: movement-level (sermons typically have 3-5 major movements)
- Micro: individual paragraphs within movements

Special handling for venue, occasion, date, and scripture references.
"""

from __future__ import annotations

import re

import structlog

from author_library.chunking._tree_utils import (
    collect_text,
    find_children_of_type,
    find_nodes,
)
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()

# Pattern for detecting scripture references (e.g. "John 3:16", "Genesis 1:1-3")
_SCRIPTURE_RE = re.compile(
    r"\b(?:Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth"
    r"|1\s*Samuel|2\s*Samuel|1\s*Kings|2\s*Kings|1\s*Chronicles|2\s*Chronicles"
    r"|Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs|Ecclesiastes|Song\s*of\s*Solomon"
    r"|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah"
    r"|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi"
    r"|Matthew|Mark|Luke|John|Acts|Romans|1\s*Corinthians|2\s*Corinthians"
    r"|Galatians|Ephesians|Philippians|Colossians"
    r"|1\s*Thessalonians|2\s*Thessalonians|1\s*Timothy|2\s*Timothy"
    r"|Titus|Philemon|Hebrews|James|1\s*Peter|2\s*Peter"
    r"|1\s*John|2\s*John|3\s*John|Jude|Revelation)"
    r"\s+\d+(?::\d+(?:\s*[-\u2013]\s*\d+)?)?",
    re.IGNORECASE,
)


class SermonStrategy(ChunkingStrategy):
    """Chunking strategy for sermons, lectures, and transcripts."""

    def supported_genres(self) -> list[str]:
        return ["sermon", "lecture", "transcript", "homily", "address"]

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

        # Each chapter (or the root) represents one sermon/lecture
        sermons = find_nodes(document.tree, NodeType.CHAPTER)
        if not sermons:
            sermons = [document.tree]

        for sermon_node in sermons:
            sermon_text = collect_text(sermon_node)
            if not sermon_text.strip():
                continue

            sermon_title = str(sermon_node.metadata.get("title", "")) or None

            # Pull contextual metadata from the document/node
            occasion = str(sermon_node.metadata.get("occasion", ""))
            venue = str(sermon_node.metadata.get("venue", ""))
            date = str(sermon_node.metadata.get("date", ""))
            if not occasion:
                occasion = str(document.metadata.title or "")
            scripture_refs = _extract_scripture_refs(sermon_text)

            macro_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "sermon",
            }
            if occasion:
                macro_meta["occasion"] = occasion
            if venue:
                macro_meta["venue"] = venue
            if date:
                macro_meta["date"] = date
            if scripture_refs:
                macro_meta["scripture_refs"] = scripture_refs

            # --- MACRO: entire sermon summary ---
            macro_chunk = Chunk(
                text=sermon_text,
                granularity=ChunkGranularity.MACRO,
                work_id=work_id,
                source_class=source_class,
                chapter=sermon_title,
                position=position_counters[ChunkGranularity.MACRO],
                metadata=macro_meta,
            )
            position_counters[ChunkGranularity.MACRO] += 1
            chunks.append(macro_chunk)

            # --- MESO: movements of the sermon ---
            movements = _identify_movements(sermon_node)
            for idx, movement_node in enumerate(movements):
                movement_text = collect_text(movement_node)
                if not movement_text.strip():
                    continue

                movement_title = str(movement_node.metadata.get("title", ""))
                movement_scripture = _extract_scripture_refs(movement_text)

                meso_meta: dict[str, str | int | bool | list[str]] = {
                    "genre": "sermon",
                    "movement_number": idx + 1,
                }
                if movement_scripture:
                    meso_meta["scripture_refs"] = movement_scripture
                if occasion:
                    meso_meta["occasion"] = occasion
                if venue:
                    meso_meta["venue"] = venue

                meso_chunk = Chunk(
                    text=movement_text,
                    granularity=ChunkGranularity.MESO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=sermon_title,
                    section=movement_title or f"Movement {idx + 1}",
                    position=position_counters[ChunkGranularity.MESO],
                    parent_chunk_id=macro_chunk.id,
                    metadata=meso_meta,
                )
                position_counters[ChunkGranularity.MESO] += 1
                chunks.append(meso_chunk)

                # --- MICRO: paragraphs within movements ---
                paragraphs = find_nodes(movement_node, NodeType.PARAGRAPH)
                for para in paragraphs:
                    para_text = collect_text(para)
                    if not para_text.strip():
                        continue
                    para_scripture = _extract_scripture_refs(para_text)
                    micro_meta: dict[str, str | int | bool | list[str]] = {
                        "genre": "sermon",
                    }
                    if para_scripture:
                        micro_meta["scripture_refs"] = para_scripture

                    micro_chunk = Chunk(
                        text=para_text,
                        granularity=ChunkGranularity.MICRO,
                        work_id=work_id,
                        source_class=source_class,
                        chapter=sermon_title,
                        section=movement_title or f"Movement {idx + 1}",
                        position=position_counters[ChunkGranularity.MICRO],
                        parent_chunk_id=meso_chunk.id,
                        metadata=micro_meta,
                    )
                    position_counters[ChunkGranularity.MICRO] += 1
                    chunks.append(micro_chunk)

        logger.info(
            "sermon_chunking_complete",
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


def _identify_movements(sermon_node: DocumentNode) -> list[DocumentNode]:
    """Identify major movements in a sermon.

    Movements are represented as SECTION nodes. If no sections exist, fall
    back to grouping paragraphs into roughly equal-sized movements (3-5).
    """
    sections = find_children_of_type(sermon_node, NodeType.SECTION)
    if sections:
        return sections

    # No explicit sections — group paragraphs into 3-5 movements
    paragraphs = find_nodes(sermon_node, NodeType.PARAGRAPH)
    if len(paragraphs) <= 3:
        return paragraphs if paragraphs else [sermon_node]

    # Target ~4 movements
    target_movements = min(5, max(3, len(paragraphs) // 3))
    per_movement = max(1, len(paragraphs) // target_movements)

    movements: list[DocumentNode] = []
    for i in range(0, len(paragraphs), per_movement):
        group = paragraphs[i : i + per_movement]
        if not group:
            continue
        # Create a synthetic section node wrapping these paragraphs
        combined_text = "\n\n".join(collect_text(p) for p in group)
        movement_node = DocumentNode(
            node_type=NodeType.SECTION,
            text=combined_text,
            children=list(group),
            metadata={"synthetic": True},
        )
        movements.append(movement_node)

    return movements


def _extract_scripture_refs(text: str) -> list[str]:
    """Extract scripture references from text."""
    return list(dict.fromkeys(_SCRIPTURE_RE.findall(text)))
