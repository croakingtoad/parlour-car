"""Chunking strategies for letters, blog posts, and interviews.

Implements chunking-guide Sections 5, 6, and 7.

Letters: Individual letter = meso chunk. Collection = macro. Paragraphs = micro.
Blog posts: Typically single meso chunk. Long posts split at heading breaks.
Interviews: Q&A pairs as meso chunks. Interviewer questions tagged secondary,
            author responses tagged primary-adjacent.
"""

from __future__ import annotations

import structlog

from author_library.chunking._tree_utils import (
    collect_text,
    find_children_of_type,
    find_nodes,
    word_count,
)
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()


# =====================================================================
# Letters / Correspondence
# =====================================================================


class LetterStrategy(ChunkingStrategy):
    """Chunking strategy for letters and correspondence (chunking-guide Section 5)."""

    def supported_genres(self) -> list[str]:
        return ["letters", "correspondence", "epistolary"]

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

        # Each chapter represents a letter (or group of letters)
        letters = find_nodes(document.tree, NodeType.CHAPTER)
        if not letters:
            letters = find_nodes(document.tree, NodeType.SECTION)
        if not letters:
            # Whole document is one letter
            letters = [document.tree]

        # --- MACRO: correspondence-period summary ---
        collection_text = collect_text(document.tree)
        if collection_text.strip():
            macro_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "correspondence",
            }
            macro_chunk = Chunk(
                text=collection_text,
                granularity=ChunkGranularity.MACRO,
                work_id=work_id,
                source_class=source_class,
                chapter=str(document.metadata.title or "Correspondence"),
                position=position_counters[ChunkGranularity.MACRO],
                metadata=macro_meta,
            )
            position_counters[ChunkGranularity.MACRO] += 1
            chunks.append(macro_chunk)
        else:
            macro_chunk = None

        # --- MESO: individual letters ---
        for letter_node in letters:
            letter_text = collect_text(letter_node)
            if not letter_text.strip():
                continue

            recipient = str(letter_node.metadata.get("recipient", ""))
            date = str(letter_node.metadata.get("date", ""))
            letter_title = str(letter_node.metadata.get("title", "")) or None

            meso_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "correspondence",
            }
            if recipient:
                meso_meta["recipient"] = recipient
            if date:
                meso_meta["date"] = date

            wc = word_count(letter_text)

            if wc <= 2000:
                # Standard letter — one meso chunk
                meso_chunk = Chunk(
                    text=letter_text,
                    granularity=ChunkGranularity.MESO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=letter_title,
                    position=position_counters[ChunkGranularity.MESO],
                    parent_chunk_id=macro_chunk.id if macro_chunk else None,
                    metadata=meso_meta,
                )
                position_counters[ChunkGranularity.MESO] += 1
                chunks.append(meso_chunk)
            else:
                # Long letter — split into meso chunks at topic transitions
                meso_chunks = _split_long_letter(
                    letter_node,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=letter_title,
                    parent_id=macro_chunk.id if macro_chunk else None,
                    base_meta=meso_meta,
                    position_counters=position_counters,
                )
                chunks.extend(meso_chunks)

            # --- MICRO: paragraphs within the letter ---
            paragraphs = find_nodes(letter_node, NodeType.PARAGRAPH)
            parent_meso_id = chunks[-1].id if chunks else None
            for para in paragraphs:
                para_text = collect_text(para)
                if not para_text.strip():
                    continue
                # Skip salutations / closings if very short
                if word_count(para_text) < 5:
                    continue
                micro = Chunk(
                    text=para_text,
                    granularity=ChunkGranularity.MICRO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=letter_title,
                    position=position_counters[ChunkGranularity.MICRO],
                    parent_chunk_id=parent_meso_id,
                    metadata=dict(meso_meta),
                )
                position_counters[ChunkGranularity.MICRO] += 1
                chunks.append(micro)

        logger.info(
            "letter_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
        )
        return chunks


# =====================================================================
# Blog Posts / Short Essays
# =====================================================================


class BlogStrategy(ChunkingStrategy):
    """Chunking strategy for blog posts and short essays (chunking-guide Section 6)."""

    def supported_genres(self) -> list[str]:
        return ["blog", "blog_post", "essay", "short_essay", "online"]

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

        # Blogs typically have one post per document (or chapter)
        posts = find_nodes(document.tree, NodeType.CHAPTER)
        if not posts:
            posts = [document.tree]

        for post_node in posts:
            post_text = collect_text(post_node)
            if not post_text.strip():
                continue

            post_title = str(post_node.metadata.get("title", "")) or None
            url = str(post_node.metadata.get("url", ""))
            date = str(post_node.metadata.get("date", ""))

            base_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "blog",
            }
            if url:
                base_meta["url"] = url
            if date:
                base_meta["date"] = date

            wc = word_count(post_text)

            if wc <= 2000:
                # Single meso chunk for the whole post
                meso_chunk = Chunk(
                    text=post_text,
                    granularity=ChunkGranularity.MESO,
                    work_id=work_id,
                    source_class=source_class,
                    chapter=post_title,
                    position=position_counters[ChunkGranularity.MESO],
                    metadata=base_meta,
                )
                position_counters[ChunkGranularity.MESO] += 1
                chunks.append(meso_chunk)
            else:
                # Long post — split at sections/headings
                sections = find_children_of_type(post_node, NodeType.SECTION)
                if sections:
                    for sec in sections:
                        sec_text = collect_text(sec)
                        if not sec_text.strip():
                            continue
                        sec_title = str(sec.metadata.get("title", "")) or None
                        meso = Chunk(
                            text=sec_text,
                            granularity=ChunkGranularity.MESO,
                            work_id=work_id,
                            source_class=source_class,
                            chapter=post_title,
                            section=sec_title,
                            position=position_counters[ChunkGranularity.MESO],
                            metadata=base_meta,
                        )
                        position_counters[ChunkGranularity.MESO] += 1
                        chunks.append(meso)
                else:
                    # No sections — create meso from grouped paragraphs
                    _emit_grouped_paragraphs(
                        post_node,
                        chunks=chunks,
                        work_id=work_id,
                        source_class=source_class,
                        chapter=post_title,
                        base_meta=base_meta,
                        position_counters=position_counters,
                    )

                # Micro chunks for independently quotable passages
                paragraphs = find_nodes(post_node, NodeType.PARAGRAPH)
                last_meso_id = None
                for c in reversed(chunks):
                    if c.granularity == ChunkGranularity.MESO:
                        last_meso_id = c.id
                        break
                for para in paragraphs:
                    para_text = collect_text(para)
                    if not para_text.strip() or word_count(para_text) < 30:
                        continue
                    micro = Chunk(
                        text=para_text,
                        granularity=ChunkGranularity.MICRO,
                        work_id=work_id,
                        source_class=source_class,
                        chapter=post_title,
                        position=position_counters[ChunkGranularity.MICRO],
                        parent_chunk_id=last_meso_id,
                        metadata=base_meta,
                    )
                    position_counters[ChunkGranularity.MICRO] += 1
                    chunks.append(micro)

        logger.info(
            "blog_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
        )
        return chunks


# =====================================================================
# Interviews
# =====================================================================


class InterviewStrategy(ChunkingStrategy):
    """Chunking strategy for interviews (chunking-guide Section 7).

    Critical: interviewer questions are tagged as secondary framing;
    author responses are tagged as primary-adjacent.
    """

    def supported_genres(self) -> list[str]:
        return ["interview", "dialogue", "conversation", "q_and_a"]

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

        # Whole interview as macro
        full_text = collect_text(document.tree)
        if not full_text.strip():
            return []

        interview_title = str(document.metadata.title or "Interview")
        interviewer = str(document.tree.metadata.get("interviewer", ""))

        macro_meta: dict[str, str | int | bool | list[str]] = {
            "genre": "interview",
        }
        if interviewer:
            macro_meta["interviewer"] = interviewer

        macro_chunk = Chunk(
            text=full_text,
            granularity=ChunkGranularity.MACRO,
            work_id=work_id,
            source_class=source_class,
            chapter=interview_title,
            position=position_counters[ChunkGranularity.MACRO],
            metadata=macro_meta,
        )
        position_counters[ChunkGranularity.MACRO] += 1
        chunks.append(macro_chunk)

        # --- MESO: Q&A pairs ---
        qa_pairs = _extract_qa_pairs(document.tree)
        for idx, (question, answer) in enumerate(qa_pairs):
            qa_text = f"Q: {question}\n\nA: {answer}"
            meso_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "interview",
                "qa_index": idx + 1,
                "question": question,
                "answer": answer,
                "question_source_class": "secondary",
                "answer_source_class": "primary-adjacent",
            }
            if interviewer:
                meso_meta["interviewer"] = interviewer

            meso_chunk = Chunk(
                text=qa_text,
                granularity=ChunkGranularity.MESO,
                work_id=work_id,
                source_class=source_class,
                chapter=interview_title,
                section=f"Q&A {idx + 1}",
                position=position_counters[ChunkGranularity.MESO],
                parent_chunk_id=macro_chunk.id,
                metadata=meso_meta,
            )
            position_counters[ChunkGranularity.MESO] += 1
            chunks.append(meso_chunk)

            # MICRO: extended monologue responses may be split
            if word_count(answer) > 200:
                answer_paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
                for para_idx, para_text in enumerate(answer_paragraphs):
                    if not para_text.strip() or word_count(para_text) < 30:
                        continue
                    micro = Chunk(
                        text=para_text,
                        granularity=ChunkGranularity.MICRO,
                        work_id=work_id,
                        source_class=source_class,
                        chapter=interview_title,
                        section=f"Q&A {idx + 1}",
                        position=position_counters[ChunkGranularity.MICRO],
                        parent_chunk_id=meso_chunk.id,
                        metadata={
                            "genre": "interview",
                            "answer_source_class": "primary-adjacent",
                            "response_part": para_idx + 1,
                        },
                    )
                    position_counters[ChunkGranularity.MICRO] += 1
                    chunks.append(micro)

        # Fallback: if no Q&A pairs were extracted, chunk by paragraphs
        if not qa_pairs:
            _emit_grouped_paragraphs(
                document.tree,
                chunks=chunks,
                work_id=work_id,
                source_class=source_class,
                chapter=interview_title,
                base_meta=macro_meta,
                position_counters=position_counters,
                parent_id=macro_chunk.id,
            )

        logger.info(
            "interview_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
            qa_pairs=len(qa_pairs),
        )
        return chunks


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _split_long_letter(
    letter_node: DocumentNode,
    *,
    work_id: str,
    source_class: str,
    chapter: str | None,
    parent_id: str | None,
    base_meta: dict[str, str | int | bool | list[str]],
    position_counters: dict[ChunkGranularity, int],
) -> list[Chunk]:
    """Split a long letter (>2000 words) into meso chunks at topic transitions."""
    paragraphs = find_nodes(letter_node, NodeType.PARAGRAPH)
    if not paragraphs:
        # Fallback: one big meso chunk
        text = collect_text(letter_node)
        chunk = Chunk(
            text=text,
            granularity=ChunkGranularity.MESO,
            work_id=work_id,
            source_class=source_class,
            chapter=chapter,
            position=position_counters[ChunkGranularity.MESO],
            parent_chunk_id=parent_id,
            metadata=dict(base_meta),
        )
        position_counters[ChunkGranularity.MESO] += 1
        return [chunk]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal buffer, buffer_words
        if not buffer:
            return
        text = "\n\n".join(buffer)
        chunk = Chunk(
            text=text,
            granularity=ChunkGranularity.MESO,
            work_id=work_id,
            source_class=source_class,
            chapter=chapter,
            position=position_counters[ChunkGranularity.MESO],
            parent_chunk_id=parent_id,
            metadata=dict(base_meta),
        )
        position_counters[ChunkGranularity.MESO] += 1
        chunks.append(chunk)
        buffer = []
        buffer_words = 0

    for para in paragraphs:
        para_text = collect_text(para)
        if not para_text.strip():
            continue
        wc = word_count(para_text)
        if buffer_words + wc > 500 and buffer:
            flush()
        buffer.append(para_text)
        buffer_words += wc

    flush()
    return chunks


def _emit_grouped_paragraphs(
    node: DocumentNode,
    *,
    chunks: list[Chunk],
    work_id: str,
    source_class: str,
    chapter: str | None,
    base_meta: dict[str, str | int | bool | list[str]],
    position_counters: dict[ChunkGranularity, int],
    parent_id: str | None = None,
) -> None:
    """Group paragraphs into meso-sized chunks (~150-500 words)."""
    paragraphs = find_nodes(node, NodeType.PARAGRAPH)
    buffer: list[str] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal buffer, buffer_words
        if not buffer:
            return
        text = "\n\n".join(buffer)
        chunk = Chunk(
            text=text,
            granularity=ChunkGranularity.MESO,
            work_id=work_id,
            source_class=source_class,
            chapter=chapter,
            position=position_counters[ChunkGranularity.MESO],
            parent_chunk_id=parent_id,
            metadata=dict(base_meta),
        )
        position_counters[ChunkGranularity.MESO] += 1
        chunks.append(chunk)
        buffer = []
        buffer_words = 0

    for para in paragraphs:
        para_text = collect_text(para)
        if not para_text.strip():
            continue
        wc = word_count(para_text)
        if buffer_words + wc > 500 and buffer:
            flush()
        buffer.append(para_text)
        buffer_words += wc

    flush()


def _extract_qa_pairs(root: DocumentNode) -> list[tuple[str, str]]:
    """Extract question-answer pairs from an interview document tree.

    Detection strategy:
    1. Look for nodes with metadata ``role=question`` / ``role=answer``
    2. Fall back to alternating paragraph pattern (Q, A, Q, A, ...)
    3. Fall back to text-pattern detection (lines starting with Q: / A:)
    """
    # Strategy 1: metadata-tagged roles
    pairs = _qa_from_metadata(root)
    if pairs:
        return pairs

    # Strategy 2: text-pattern detection (Q: / A: prefixes)
    full_text = collect_text(root)
    pairs = _qa_from_text_patterns(full_text)
    if pairs:
        return pairs

    # Strategy 3: alternating paragraphs (interviewer, author, interviewer, ...)
    pairs = _qa_from_alternating_paragraphs(root)
    return pairs


def _qa_from_metadata(root: DocumentNode) -> list[tuple[str, str]]:
    """Extract Q&A pairs from nodes tagged with role metadata."""
    questions: list[DocumentNode] = []
    answers: list[DocumentNode] = []
    for node in root.children:
        role = str(node.metadata.get("role", ""))
        if role == "question":
            questions.append(node)
        elif role == "answer":
            answers.append(node)

    if not questions or len(questions) != len(answers):
        return []

    return [
        (collect_text(q), collect_text(a))
        for q, a in zip(questions, answers, strict=False)
    ]


def _qa_from_text_patterns(text: str) -> list[tuple[str, str]]:
    """Extract Q&A pairs from text with Q:/A: prefixes."""
    import re

    # Match lines starting with Q: or A: (case-insensitive)
    parts = re.split(r"(?m)^(Q:|A:)\s*", text)
    if len(parts) < 3:
        return []

    pairs: list[tuple[str, str]] = []
    i = 1  # parts[0] is text before first Q:/A:
    while i < len(parts) - 1:
        label = parts[i].strip().upper()
        content = parts[i + 1].strip()
        if label == "Q:" and i + 3 <= len(parts):
            next_label = parts[i + 2].strip().upper() if i + 2 < len(parts) else ""
            next_content = parts[i + 3].strip() if i + 3 < len(parts) else ""
            if next_label == "A:" and next_content:
                pairs.append((content, next_content))
                i += 4
                continue
        i += 2

    return pairs


def _qa_from_alternating_paragraphs(root: DocumentNode) -> list[tuple[str, str]]:
    """Extract Q&A pairs from alternating paragraphs."""
    paragraphs = find_nodes(root, NodeType.PARAGRAPH)
    if len(paragraphs) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(paragraphs):
        q_text = collect_text(paragraphs[i])
        a_text = collect_text(paragraphs[i + 1])
        if q_text.strip() and a_text.strip():
            pairs.append((q_text.strip(), a_text.strip()))
        i += 2

    return pairs
