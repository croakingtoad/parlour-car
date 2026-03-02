"""Contextual annotation engine for chunk embedding enrichment.

Implements chunking-guide Section 9: prepends source-class-aware annotations
to each chunk BEFORE embedding.  Uses the Anthropic API for LLM-generated
fields (topic summary, positioning).

Three templates:
- PRIMARY: author's own work
- SECONDARY: work by others about the author
- CONTEXTUAL: works the author engages with
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from author_library.config import Settings, get_settings
from author_library.errors import IngestionError
from author_library.text_utils import sanitize_text

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk

logger = structlog.get_logger()

# Maximum chunks per LLM batch request
_BATCH_SIZE = 10


@dataclass(frozen=True)
class AnnotationContext:
    """Metadata needed to generate contextual annotations.

    These fields come from the catalog entry / work metadata,
    not from the chunk itself.
    """

    work_title: str
    publication_year: int | str
    author: str  # the author of *this* work (may differ from subject author)
    subject_author: str  # the subject author of the library
    chapter_title: str | None = None
    chapter_number: int | str | None = None
    relationship_type: str | None = None  # e.g. "critical study", "biography"
    perspective_note: str | None = None  # from catalog: critic's perspective
    engagement_note: str | None = None  # from catalog: how subject engages
    engagement_works: str | None = None  # works where subject engages this source


class ChunkAnnotator:
    """Generates and attaches contextual annotations to chunks.

    Annotations are prepended to chunk text before embedding to improve
    retrieval quality.  Source classification markers are critical — they
    prevent voice contamination at retrieval time.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def annotate_chunks(
        self,
        chunks: list[Chunk],
        context: AnnotationContext,
    ) -> list[Chunk]:
        """Annotate a list of chunks with contextual annotations.

        For each chunk, a source-class-appropriate annotation is generated
        and assigned to ``chunk.annotation``.  LLM-generated fields (topic,
        positioning) are populated via the Anthropic API in batches.

        Args:
            chunks: Chunks to annotate (modified in place).
            context: Catalog-level metadata for annotation templates.

        Returns:
            The same list of chunks, now with annotations populated.
        """
        if not chunks:
            return chunks

        # Group chunks by source class for template selection
        primary_chunks = [c for c in chunks if c.source_class in ("primary",)]
        secondary_chunks = [c for c in chunks if c.source_class in ("secondary",)]
        contextual_chunks = [c for c in chunks if c.source_class in ("contextual",)]
        other_chunks = [
            c
            for c in chunks
            if c.source_class not in ("primary", "secondary", "contextual")
        ]

        # Generate LLM-enriched annotations in batches
        api_key = self._settings.api_keys.anthropic_api_key.get_secret_value()
        use_llm = bool(api_key)

        if not use_llm:
            logger.warning("no_anthropic_api_key", msg="Falling back to template-only annotations")

        tasks: list[asyncio.Task[None]] = []
        if primary_chunks:
            tasks.append(
                asyncio.create_task(
                    self._annotate_batch(primary_chunks, context, "primary", use_llm)
                )
            )
        if secondary_chunks:
            tasks.append(
                asyncio.create_task(
                    self._annotate_batch(secondary_chunks, context, "secondary", use_llm)
                )
            )
        if contextual_chunks:
            tasks.append(
                asyncio.create_task(
                    self._annotate_batch(contextual_chunks, context, "contextual", use_llm)
                )
            )
        if other_chunks:
            # Treat tertiary / unknown as secondary template
            tasks.append(
                asyncio.create_task(
                    self._annotate_batch(other_chunks, context, "secondary", use_llm)
                )
            )

        if tasks:
            await asyncio.gather(*tasks)

        annotated_count = sum(1 for c in chunks if c.annotation)
        logger.info(
            "annotation_complete",
            total_chunks=len(chunks),
            annotated=annotated_count,
            used_llm=use_llm,
        )
        return chunks

    async def _annotate_batch(
        self,
        chunks: list[Chunk],
        context: AnnotationContext,
        source_class: str,
        use_llm: bool,
    ) -> None:
        """Annotate a batch of same-source-class chunks.

        LLM annotation batches run concurrently (up to _ANNOTATION_CONCURRENCY
        parallel API calls) for performance.
        """
        if not use_llm:
            for chunk in chunks:
                chunk.annotation = _format_annotation(chunk, context, source_class, {})
            return

        concurrency = int(
            getattr(self._settings.llm, "annotation_concurrency", 0) or 5
        )
        semaphore = asyncio.Semaphore(concurrency)

        # Split into sub-batches
        batches = [
            chunks[i : i + _BATCH_SIZE]
            for i in range(0, len(chunks), _BATCH_SIZE)
        ]

        total_batches = len(batches)
        logger.info(
            "annotation_starting",
            total_chunks=len(chunks),
            total_batches=total_batches,
            concurrency=concurrency,
            source_class=source_class,
        )

        async def _process_one_batch(batch: list[Chunk], batch_idx: int) -> None:
            async with semaphore:
                try:
                    llm_results = await self._llm_batch_annotate(
                        batch, context, source_class
                    )
                    for chunk, llm_data in zip(batch, llm_results, strict=True):
                        # Sanitize LLM output — API responses can contain
                        # smart quotes and other chars that become invalid
                        # byte sequences when stored in PostgreSQL.
                        llm_data = {
                            k: sanitize_text(v) if isinstance(v, str) else v
                            for k, v in llm_data.items()
                        }
                        chunk.annotation = _format_annotation(
                            chunk, context, source_class, llm_data
                        )
                except Exception as exc:
                    logger.error(
                        "llm_annotation_failed",
                        error=str(exc),
                        batch_idx=batch_idx + 1,
                        batch_size=len(batch),
                        source_class=source_class,
                    )
                    # Fall back to template-only for this batch
                    for chunk in batch:
                        chunk.annotation = _format_annotation(
                            chunk, context, source_class, {}
                        )

        await asyncio.gather(
            *(_process_one_batch(b, i) for i, b in enumerate(batches))
        )

    async def _llm_batch_annotate(
        self,
        chunks: list[Chunk],
        context: AnnotationContext,
        source_class: str,
    ) -> list[dict[str, str]]:
        """Call the Anthropic API to generate topic/positioning for a batch of chunks.

        Returns a list of dicts, one per chunk, with keys:
        - ``topic``: brief topic description
        - ``positioning``: how the passage fits in the chapter's argument (primary only)
        - ``preceding_context``: summary of what comes before (primary only)
        - ``following_context``: summary of what follows (primary only)
        """
        import anthropic

        api_key = self._settings.api_keys.anthropic_api_key.get_secret_value()
        if not api_key:
            raise IngestionError("Anthropic API key required for LLM annotation")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = self._settings.llm.ingestion_model

        # Build the batch prompt
        chunk_descriptions = []
        for idx, chunk in enumerate(chunks):
            preview = chunk.text[:500] + ("..." if len(chunk.text) > 500 else "")
            chunk_descriptions.append(
                f"Chunk {idx + 1} ({chunk.granularity.value}, {source_class}):\n{preview}"
            )

        source_class_instruction = _source_class_prompt(source_class)

        prompt = (
            f"You are annotating text chunks from \"{context.work_title}\" "
            f"({context.publication_year}) by {context.author}.\n\n"
            f"The subject author of this library is {context.subject_author}.\n\n"
            f"{source_class_instruction}\n\n"
            f"For each chunk below, provide a JSON array with one object per chunk. "
            f"Each object must have:\n"
            f"- \"topic\": A brief (1-2 sentence) description of the chunk's topic.\n"
        )

        if source_class == "primary":
            prompt += (
                '- "positioning": How this passage fits in the chapter\'s argument (1 sentence).\n'
                '- "preceding_context": Summary of what comes before (1 sentence).\n'
                '- "following_context": Summary of what follows (1 sentence).\n'
            )

        prompt += (
            "\nRespond with ONLY valid JSON (an array of objects). No other text.\n\n"
            "Chunks:\n\n" + "\n\n---\n\n".join(chunk_descriptions)
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            import json

            response_text = response.content[0].text  # type: ignore[union-attr]
            # Strip markdown code fences if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            results: list[dict[str, str]] = json.loads(response_text)

            # Ensure we have the right number of results
            if len(results) != len(chunks):
                logger.warning(
                    "llm_result_count_mismatch",
                    expected=len(chunks),
                    got=len(results),
                )
                # Pad or truncate
                while len(results) < len(chunks):
                    results.append({})
                results = results[: len(chunks)]

            return results

        except Exception as exc:
            raise IngestionError(
                "LLM annotation batch failed",
                context={"model": model, "batch_size": len(chunks)},
                cause=exc,
            ) from exc


# ------------------------------------------------------------------
# Annotation formatting
# ------------------------------------------------------------------


def _format_annotation(
    chunk: Chunk,
    context: AnnotationContext,
    source_class: str,
    llm_data: dict[str, str],
) -> str:
    """Format an annotation string for the given source class."""
    if source_class == "primary":
        return _primary_annotation(chunk, context, llm_data)
    elif source_class == "secondary":
        return _secondary_annotation(chunk, context, llm_data)
    elif source_class == "contextual":
        return _contextual_annotation(chunk, context, llm_data)
    else:
        # Tertiary or unknown — use secondary template
        return _secondary_annotation(chunk, context, llm_data)


def _primary_annotation(
    chunk: Chunk,
    ctx: AnnotationContext,
    llm: dict[str, str],
) -> str:
    """Format a PRIMARY source annotation (chunking-guide Section 9)."""
    topic = llm.get("topic", f"{chunk.granularity.value} chunk")
    positioning = llm.get("positioning", "")
    preceding = llm.get("preceding_context", "")
    following = llm.get("following_context", "")

    lines = [
        f'[PRIMARY] From "{ctx.work_title}" ({ctx.publication_year}) by {ctx.subject_author}.',
    ]
    if ctx.chapter_number and ctx.chapter_title:
        lines.append(f'Chapter {ctx.chapter_number}: "{ctx.chapter_title}".')
    elif chunk.chapter:
        lines.append(f'Chapter: "{chunk.chapter}".')
    lines.append(f"This {chunk.granularity.value} covers: {topic}.")
    if positioning:
        lines.append(f"In the larger argument of this chapter, {positioning}.")
    if preceding:
        lines.append(f"Preceding context: {preceding}.")
    if following:
        lines.append(f"Following context: {following}.")

    return "\n".join(lines)


def _secondary_annotation(
    chunk: Chunk,
    ctx: AnnotationContext,
    llm: dict[str, str],
) -> str:
    """Format a SECONDARY source annotation (chunking-guide Section 9)."""
    topic = llm.get("topic", f"{chunk.granularity.value} chunk")
    perspective = ctx.perspective_note or ""

    lines = [
        f"[SECONDARY: Written by {ctx.author} about {ctx.subject_author}]",
        f'From "{ctx.work_title}" ({ctx.publication_year})',
    ]
    if ctx.relationship_type:
        lines[-1] += f", a {ctx.relationship_type}"
    lines[-1] += "."

    if ctx.chapter_number and ctx.chapter_title:
        lines.append(f'Chapter {ctx.chapter_number}: "{ctx.chapter_title}".')
    elif chunk.chapter:
        lines.append(f'Chapter: "{chunk.chapter}".')
    lines.append(f"This passage discusses: {topic}.")
    if perspective:
        lines.append(f"The critic's perspective: {perspective}.")

    return "\n".join(lines)


def _contextual_annotation(
    chunk: Chunk,
    ctx: AnnotationContext,
    llm: dict[str, str],
) -> str:
    """Format a CONTEXTUAL source annotation (chunking-guide Section 9)."""
    topic = llm.get("topic", f"{chunk.granularity.value} chunk")
    engagement = ctx.engagement_note or ""
    engagement_works = ctx.engagement_works or ""

    lines = [
        f"[CONTEXTUAL: By {ctx.author}, referenced by {ctx.subject_author}]",
        f'From "{ctx.work_title}" ({ctx.publication_year}).',
    ]
    if ctx.chapter_number and ctx.chapter_title:
        lines.append(f'Chapter {ctx.chapter_number}: "{ctx.chapter_title}".')
    elif chunk.chapter:
        lines.append(f'Chapter: "{chunk.chapter}".')
    if engagement:
        lines.append(f"This passage is relevant because: {engagement}.")
    if engagement_works:
        lines.append(f"{ctx.subject_author} engages with this material in: {engagement_works}.")
    lines.append(f"Topic: {topic}.")

    return "\n".join(lines)


_SOURCE_CLASS_PROMPTS: dict[str, str] = {
    "primary": (
        "These chunks are from a PRIMARY source \u2014 the subject author's own words. "
        "Analyze the author's argument, theme, and rhetorical moves."
    ),
    "secondary": (
        "These chunks are from a SECONDARY source \u2014 written by someone else ABOUT "
        "the subject author. Focus on the critic's argument and perspective."
    ),
    "contextual": (
        "These chunks are from a CONTEXTUAL source \u2014 a work the subject author "
        "engages with. Focus on what concepts or arguments the subject author "
        "draws from this work."
    ),
}


def _source_class_prompt(source_class: str) -> str:
    """Return the LLM prompt instruction tailored to the source class."""
    return _SOURCE_CLASS_PROMPTS.get(source_class, "Analyze the content of these chunks.")
