"""Poetry chunking strategy.

Implements chunking-guide Section 3.

Critical rule: NEVER SPLIT A POEM.  Each poem is atomic at the meso level.
Stanza-level micro chunks are emitted ONLY for poems exceeding 40 lines.
"""

from __future__ import annotations

import structlog

from author_library.chunking._tree_utils import collect_text, find_nodes
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()

# Poems shorter than this (in lines) are never split into micro chunks.
_STANZA_MICRO_LINE_THRESHOLD = 40


class PoetryStrategy(ChunkingStrategy):
    """Chunking strategy for poetry collections."""

    def supported_genres(self) -> list[str]:
        return ["poetry", "poems", "verse", "sonnet_sequence"]

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

        # --- MACRO: collection/section-level summaries ---
        # Look for top-level sections or the book root as macro containers.
        macro_nodes = _macro_containers(document.tree)
        poem_nodes = find_nodes(document.tree, NodeType.POEM)

        for macro_node in macro_nodes:
            macro_text = collect_text(macro_node)
            if not macro_text.strip():
                continue
            macro_title = str(macro_node.metadata.get("title", "")) or None
            macro_chunk = Chunk(
                text=macro_text,
                granularity=ChunkGranularity.MACRO,
                work_id=work_id,
                source_class=source_class,
                chapter=macro_title,
                position=position_counters[ChunkGranularity.MACRO],
                metadata={"genre": "poetry"},
            )
            position_counters[ChunkGranularity.MACRO] += 1
            chunks.append(macro_chunk)

        # --- MESO: each poem is one meso chunk (NEVER SPLIT) ---
        if not poem_nodes:
            # Fallback: no explicit POEM nodes — treat chapters/sections as poems
            poem_nodes = find_nodes(document.tree, NodeType.CHAPTER)
            if not poem_nodes:
                poem_nodes = find_nodes(document.tree, NodeType.SECTION)

        # Find the closest macro parent for each poem
        macro_lookup = _build_macro_lookup(document.tree, macro_nodes)

        for poem_node in poem_nodes:
            poem_text = collect_text(poem_node)
            if not poem_text.strip():
                continue

            poem_title = str(poem_node.metadata.get("title", "")) or None
            first_line = poem_text.strip().split("\n")[0][:120]
            parent_macro_id = macro_lookup.get(poem_node.id)

            meso_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "poetry",
                "first_line": first_line,
            }
            if poem_title:
                meso_meta["poem_title"] = poem_title

            # Epigraphs and dedications → metadata, not separate chunks
            epigraph = poem_node.metadata.get("epigraph", "")
            if epigraph:
                meso_meta["epigraph"] = str(epigraph)
            dedication = poem_node.metadata.get("dedication", "")
            if dedication:
                meso_meta["dedication"] = str(dedication)

            meso_chunk = Chunk(
                text=poem_text,
                granularity=ChunkGranularity.MESO,
                work_id=work_id,
                source_class=source_class,
                chapter=poem_title,
                position=position_counters[ChunkGranularity.MESO],
                parent_chunk_id=parent_macro_id,
                metadata=meso_meta,
            )
            position_counters[ChunkGranularity.MESO] += 1
            chunks.append(meso_chunk)

            # --- MICRO: stanza-level, ONLY for poems > 40 lines ---
            line_count = len([ln for ln in poem_text.split("\n") if ln.strip()])
            if line_count > _STANZA_MICRO_LINE_THRESHOLD:
                stanza_chunks = _stanza_micro_chunks(
                    poem_node,
                    poem_title=poem_title,
                    work_id=work_id,
                    source_class=source_class,
                    parent_id=meso_chunk.id,
                    position_counters=position_counters,
                )
                chunks.extend(stanza_chunks)

        logger.info(
            "poetry_chunking_complete",
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


def _macro_containers(root: DocumentNode) -> list[DocumentNode]:
    """Identify macro-level containers (sections or the root itself)."""
    # Top-level sections in poetry collections (e.g. "Part I", "Sonnets")
    sections = [c for c in root.children if c.node_type in (NodeType.SECTION, NodeType.CHAPTER)]
    if sections:
        return sections
    # No sections — the whole book is one macro container
    return [root]


def _build_macro_lookup(
    root: DocumentNode,
    macro_nodes: list[DocumentNode],
) -> dict[str, str | None]:
    """Map each poem node ID to its closest macro chunk's ID (if any)."""
    lookup: dict[str, str | None] = {}
    macro_ids = {n.id for n in macro_nodes}

    def _walk(node: DocumentNode, current_macro_id: str | None) -> None:
        if node.id in macro_ids:
            current_macro_id = node.id
        if node.node_type == NodeType.POEM:
            lookup[node.id] = current_macro_id
        for child in node.children:
            _walk(child, current_macro_id)

    _walk(root, None)
    return lookup


def _stanza_micro_chunks(
    poem_node: DocumentNode,
    *,
    poem_title: str | None,
    work_id: str,
    source_class: str,
    parent_id: str,
    position_counters: dict[ChunkGranularity, int],
) -> list[Chunk]:
    """Emit stanza-level micro chunks for a long poem."""
    stanzas = find_nodes(poem_node, NodeType.STANZA)
    chunks: list[Chunk] = []

    if stanzas:
        for idx, stanza in enumerate(stanzas):
            stanza_text = collect_text(stanza)
            if not stanza_text.strip():
                continue
            meta: dict[str, str | int | bool | list[str]] = {
                "genre": "poetry",
                "stanza_number": idx + 1,
            }
            if poem_title:
                meta["poem_title"] = poem_title
            chunk = Chunk(
                text=stanza_text,
                granularity=ChunkGranularity.MICRO,
                work_id=work_id,
                source_class=source_class,
                chapter=poem_title,
                position=position_counters[ChunkGranularity.MICRO],
                parent_chunk_id=parent_id,
                metadata=meta,
            )
            position_counters[ChunkGranularity.MICRO] += 1
            chunks.append(chunk)
    else:
        # No explicit stanza nodes — split on blank lines
        poem_text = collect_text(poem_node)
        stanza_texts = _split_on_blank_lines(poem_text)
        for idx, stanza_text in enumerate(stanza_texts):
            if not stanza_text.strip():
                continue
            meta = {
                "genre": "poetry",
                "stanza_number": idx + 1,
            }
            if poem_title:
                meta["poem_title"] = poem_title
            chunk = Chunk(
                text=stanza_text,
                granularity=ChunkGranularity.MICRO,
                work_id=work_id,
                source_class=source_class,
                chapter=poem_title,
                position=position_counters[ChunkGranularity.MICRO],
                parent_chunk_id=parent_id,
                metadata=meta,
            )
            position_counters[ChunkGranularity.MICRO] += 1
            chunks.append(chunk)

    return chunks


def _split_on_blank_lines(text: str) -> list[str]:
    """Split text into stanzas at blank-line boundaries."""
    stanzas: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            if current:
                stanzas.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        stanzas.append("\n".join(current))
    return stanzas
