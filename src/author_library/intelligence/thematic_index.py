"""Thematic index generation from author corpus.

Analyzes the subject author's primary corpus to identify recurring themes,
map their appearances across works, and build a pre-computed thematic index
for efficient query-time retrieval.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IntelligenceError

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.config import Settings
    from author_library.storage.manager import StorageManager
    from author_library.storage.repositories import (
        ChunkRepository,
        ThematicRepository,
        WorkRepository,
    )

from pydantic import BaseModel, Field

from author_library.intelligence.lesson_writer import get_lesson_context

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Thematic index data models
# ---------------------------------------------------------------------------


class KeyPassage(BaseModel):
    """A key passage that illustrates a theme."""

    chunk_id: str
    text_excerpt: str
    work_id: str


class ThematicAppearance(BaseModel):
    """How a theme appears in a specific work."""

    work_id: str
    chapters: list[str] = Field(default_factory=list)
    treatment_summary: str


class ThematicEntry(BaseModel):
    """A single entry in the thematic index."""

    theme: str
    author_stance: str
    appearances: list[ThematicAppearance] = Field(default_factory=list)
    related_themes: list[str] = Field(default_factory=list)
    key_passages: list[KeyPassage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

THEME_IDENTIFICATION_SYSTEM = """\
You are a literary scholar analyzing an author's corpus to identify \
the major recurring themes in their work. A "theme" is a substantial \
conceptual thread that appears across multiple works or receives \
significant treatment in a single work.

Analyze the provided text samples and identify the 5-20 most significant \
themes. For each theme, describe the author's stance or position.

Respond with valid JSON matching this schema:
{
  "themes": [
    {
      "theme": "<canonical theme name, e.g. 'Sacramental Imagination'>",
      "author_stance": "<1-2 sentence summary of the author's position>",
      "related_themes": ["<related theme 1>", ...]
    },
    ...
  ]
}

Guidelines:
- Use clear, descriptive theme names (capitalize major words)
- Stance should reflect the author's actual position, not generic description
- Related themes should reference other themes in your list when possible
- Prefer specific themes over generic ones ("The Sacramental Nature of Poetry" \
over "Poetry")
- Only identify themes with clear textual evidence in the samples\
"""

THEME_MAPPING_SYSTEM = """\
You are a literary scholar mapping how a specific theme appears in \
text samples from an author's work.

CRITICAL: Your ENTIRE response must be a single JSON object. \
No prose, no commentary, no markdown fences. Start with { and end with }.

JSON schema:
{"appearances": [{"sample_index": <int>, "present": true/false, \
"treatment_summary": "<ONE sentence, max 50 words>", \
"is_key_passage": true/false}]}

Rules:
- Only include entries where present=true
- treatment_summary must be ONE concise sentence (not a paragraph)
- Mark 2-5 of the most representative passages as key passages
- Do NOT explain your reasoning — output ONLY the JSON object\
"""


# ---------------------------------------------------------------------------
# Chunk batching
# ---------------------------------------------------------------------------

MAX_CHUNKS_PER_BATCH = 75


def _batch_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = MAX_CHUNKS_PER_BATCH,
) -> list[list[dict[str, Any]]]:
    """Split chunks into batches for LLM processing."""
    return [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format chunk list into numbered text samples for prompting."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        work_id = chunk.get("work_id", "unknown")
        chapter = chunk.get("chapter", "")
        text = chunk.get("text", "")
        chunk_id = str(chunk.get("id", chunk.get("chunk_id", f"chunk-{i}")))

        header = f"--- Sample {i} [chunk_id: {chunk_id}, work: {work_id}"
        if chapter:
            header += f", chapter: {chapter}"
        header += "] ---"

        parts.append(header)
        parts.append(text)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Thematic index generator
# ---------------------------------------------------------------------------


class ThematicIndexGenerator:
    """Generates a thematic index from an author's primary corpus.

    Phase 1: Identify themes from a sample of the corpus.
    Phase 2: Map each theme's appearances across all chunks.
    Phase 3: Store results via ThematicRepository.
    """

    def __init__(
        self,
        settings: Settings,
        storage: StorageManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._api_key = settings.api_keys.anthropic_api_key.get_secret_value()
        self._model = settings.llm.ingestion_model

    async def generate(
        self,
        *,
        author_id: str,
        author_name: str,
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
        thematic_repo: ThematicRepository,
    ) -> list[ThematicEntry]:
        """Generate and store the thematic index for an author.

        Args:
            author_id: The author's slug identifier.
            author_name: The author's canonical display name.
            work_repo: Repository for accessing work metadata.
            chunk_repo: Repository for accessing corpus chunks.
            thematic_repo: Repository for storing thematic entries.

        Returns:
            List of ThematicEntry objects generated.

        Raises:
            IntelligenceError: If generation fails.
        """
        if not self._api_key:
            raise IntelligenceError(
                "Anthropic API key is required for thematic index generation",
                context={"author_id": author_id},
            )

        # Gather all primary chunks
        primary_chunks = await self._gather_primary_chunks(
            author_id=author_id,
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        if not primary_chunks:
            raise IntelligenceError(
                "No primary corpus chunks found for thematic index generation",
                context={"author_id": author_id},
            )

        log.info(
            "generating_thematic_index",
            author_id=author_id,
            total_chunks=len(primary_chunks),
            model=self._model,
        )

        # Phase 1: Identify themes from a representative sample
        sample = primary_chunks[:MAX_CHUNKS_PER_BATCH]
        themes = await self._identify_themes(
            author_name=author_name,
            chunks=sample,
        )

        log.info(
            "themes_identified",
            author_id=author_id,
            n_themes=len(themes),
            themes=[t.theme for t in themes],
        )

        # Phase 2: Map theme appearances across the full corpus (parallel)
        mapping_errors: list[str] = []
        semaphore = asyncio.Semaphore(5)

        async def _map_one_theme(theme: ThematicEntry) -> None:
            async with semaphore:
                try:
                    appearances = await self._map_theme_appearances(
                        theme=theme,
                        all_chunks=primary_chunks,
                    )
                    theme.appearances = appearances
                except Exception as exc:
                    log.error(
                        "theme_mapping_failed",
                        theme=theme.theme,
                        error=str(exc),
                    )
                    mapping_errors.append(f"{theme.theme}: {exc}")

        await asyncio.gather(*(_map_one_theme(t) for t in themes))

        if mapping_errors:
            log.warning(
                "thematic_mapping_partial",
                mapped=len(themes) - len(mapping_errors),
                failed=len(mapping_errors),
                errors=mapping_errors,
            )

        # Phase 3: Store in thematic repository
        await self._store_themes(
            author_id=author_id,
            themes=themes,
            thematic_repo=thematic_repo,
        )

        log.info(
            "thematic_index_complete",
            author_id=author_id,
            n_themes=len(themes),
            total_appearances=sum(len(t.appearances) for t in themes),
        )

        return themes

    async def _gather_primary_chunks(
        self,
        *,
        author_id: str,
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
    ) -> list[dict[str, Any]]:
        """Gather all meso chunks from the author's primary works."""
        works = await work_repo.list_by_author(author_id)
        primary_work_ids = [
            w["work_id"] for w in works if w.get("source_class") == "primary"
        ]

        all_chunks: list[dict[str, Any]] = []
        for work_id in primary_work_ids:
            chunks = await chunk_repo.list_by_work(work_id, granularity="meso")
            primary = [c for c in chunks if c.get("source_class") == "primary"]
            all_chunks.extend(primary)

        return all_chunks

    async def _identify_themes(
        self,
        *,
        author_name: str,
        chunks: list[dict[str, Any]],
    ) -> list[ThematicEntry]:
        """Use LLM to identify major themes from corpus samples."""
        chunks_text = _format_chunks_for_prompt(chunks)

        user_prompt = (
            f"Author: {author_name}\n\n"
            f"Text samples from across the author's corpus:\n\n"
            f"{chunks_text}\n\n"
            "Identify the major recurring themes. Respond with JSON only."
        )

        # Inject lesson context for theme identification
        lesson_context = ""
        lesson_ids = []
        if self._storage is not None:
            lesson_context, lesson_ids = await get_lesson_context(
                self._storage, "thematic_index"
            )

        system = THEME_IDENTIFICATION_SYSTEM
        if lesson_context:
            system = f"{system}\n\n{lesson_context}"

        data = await self._call_anthropic(
            system=system,
            user_prompt=user_prompt,
        )

        # Track applied lessons
        if lesson_ids and self._storage is not None:
            for lid in lesson_ids:
                try:
                    await self._storage.lessons.increment_applied(lid)
                except Exception as exc:
                    log.warning(
                        "lesson_increment_applied_failed",
                        lesson_id=str(lid),
                        error=str(exc),
                    )

        raw_themes = data.get("themes", [])
        entries: list[ThematicEntry] = []
        for t in raw_themes:
            entries.append(
                ThematicEntry(
                    theme=t["theme"],
                    author_stance=t.get("author_stance", ""),
                    related_themes=t.get("related_themes", []),
                )
            )

        return entries

    async def _map_theme_appearances(
        self,
        *,
        theme: ThematicEntry,
        all_chunks: list[dict[str, Any]],
    ) -> list[ThematicAppearance]:
        """Map where a specific theme appears across the corpus."""
        batches = _batch_chunks(all_chunks)
        appearances_by_work: dict[str, ThematicAppearance] = {}

        for batch in batches:
            chunks_text = _format_chunks_for_prompt(batch)

            user_prompt = (
                f"Theme to map: {theme.theme}\n"
                f"Author's stance: {theme.author_stance}\n\n"
                f"Text samples:\n\n{chunks_text}\n\n"
                "Map this theme's appearances across these samples. "
                "Respond with JSON only."
            )

            data = await self._call_anthropic(
                system=THEME_MAPPING_SYSTEM,
                user_prompt=user_prompt,
            )

            for app in data.get("appearances", []):
                idx = app.get("sample_index", -1)
                if idx < 0 or idx >= len(batch):
                    continue
                if not app.get("present", False):
                    continue

                chunk = batch[idx]
                work_id = chunk.get("work_id", "unknown")
                chapter = chunk.get("chapter", "")
                treatment = app.get("treatment_summary", "")
                chunk_id = str(chunk.get("id", chunk.get("chunk_id", "")))

                if work_id in appearances_by_work:
                    existing = appearances_by_work[work_id]
                    if chapter and chapter not in existing.chapters:
                        existing.chapters.append(chapter)
                    if treatment:
                        existing.treatment_summary += f" {treatment}"
                else:
                    appearances_by_work[work_id] = ThematicAppearance(
                        work_id=work_id,
                        chapters=[chapter] if chapter else [],
                        treatment_summary=treatment,
                    )

                # Track key passages
                if app.get("is_key_passage", False):
                    text = chunk.get("text", "")
                    theme.key_passages.append(
                        KeyPassage(
                            chunk_id=chunk_id,
                            text_excerpt=text[:200],
                            work_id=work_id,
                        )
                    )

        return list(appearances_by_work.values())

    async def _store_themes(
        self,
        *,
        author_id: str,
        themes: list[ThematicEntry],
        thematic_repo: ThematicRepository,
    ) -> list[UUID]:
        """Store thematic entries and their appearances in the repository."""
        entry_ids: list[UUID] = []

        for theme in themes:
            entry_id = await thematic_repo.create_entry(
                {
                    "author_id": author_id,
                    "theme": theme.theme,
                    "author_stance": theme.author_stance,
                    "related_themes": theme.related_themes,
                    "key_passages": [kp.model_dump() for kp in theme.key_passages],
                }
            )
            entry_ids.append(entry_id)

            for appearance in theme.appearances:
                await thematic_repo.add_appearance(
                    {
                        "entry_id": entry_id,
                        "work_id": appearance.work_id,
                        "chapters": appearance.chapters,
                        "treatment_summary": appearance.treatment_summary,
                    }
                )

        log.info(
            "themes_stored",
            author_id=author_id,
            entries=len(entry_ids),
        )

        return entry_ids

    async def _call_anthropic(
        self, *, system: str, user_prompt: str, _retry: int = 0
    ) -> dict[str, Any]:
        """Call the Anthropic API and parse JSON response.

        Retries once with a stronger constraint if the LLM returns
        prose instead of JSON (common for pervasive themes).
        """
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise IntelligenceError(
                f"Anthropic API error during thematic analysis: {exc}",
                context={"model": self._model, "status_code": getattr(exc, "status_code", None)},
                cause=exc,
            ) from exc

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text
                break

        if not response_text:
            raise IntelligenceError(
                "Empty response from Anthropic API during thematic analysis",
                context={"model": self._model},
            )

        try:
            return self._parse_json_response(response_text)
        except IntelligenceError:
            if _retry >= 1:
                raise
            log.warning(
                "thematic_json_retry",
                response_length=len(response_text),
                retry=_retry + 1,
            )
            # Retry with a stronger JSON instruction appended
            return await self._call_anthropic(
                system=system,
                user_prompt=user_prompt + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. No prose, no explanation, no markdown.",
                _retry=_retry + 1,
            )

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse a JSON response from an LLM, handling common malformations."""
        from author_library.intelligence.json_parser import extract_json

        try:
            return extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise IntelligenceError(
                f"Failed to parse thematic response as JSON: {exc}",
                context={"response_text": response_text[:500]},
                cause=exc,
            ) from exc
