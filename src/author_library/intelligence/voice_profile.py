"""Voice profile extraction from primary source corpus.

Analyzes the subject author's primary works to generate a structured
voice profile capturing register, sentence patterns, vocabulary tendencies,
rhetorical moves, characteristic phrases, and representative passages.

Only chunks from primary sources with voice_profile_eligible=true are
considered, preventing voice contamination from secondary/contextual sources.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IntelligenceError

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager
    from author_library.storage.repositories import ChunkRepository, WorkRepository

from pydantic import BaseModel, Field

from author_library.intelligence.lesson_writer import get_lesson_context

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Voice profile data model
# ---------------------------------------------------------------------------


class VoiceProfile(BaseModel):
    """Structured representation of an author's distinctive voice.

    Extracted from the primary corpus using LLM analysis of representative
    meso-level chunks sampled across works, chapters, and genres.
    """

    model_config = {"arbitrary_types_allowed": True}

    author_id: str
    register: str
    sentence_patterns: list[str]
    vocabulary_tendencies: list[str]
    rhetorical_moves: list[str]
    characteristic_phrases: list[str]
    humor_style: str | None = None
    example_passages: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

VOICE_EXTRACTION_SYSTEM = """\
You are a literary analyst specializing in authorial voice and style. \
You analyze prose samples from a single author's corpus and produce a \
structured voice profile capturing the distinctive elements of their \
writing style.

Your analysis must be grounded exclusively in the provided text samples. \
Do not speculate beyond what the samples demonstrate. If the corpus is \
small, note this in your confidence score (lower confidence for fewer \
samples).

Respond with valid JSON matching this schema exactly:
{
  "register": "<description of overall register, e.g. 'academic but accessible'>",
  "sentence_patterns": ["<pattern 1>", "<pattern 2>", ...],
  "vocabulary_tendencies": ["<tendency 1>", "<tendency 2>", ...],
  "rhetorical_moves": ["<move 1>", "<move 2>", ...],
  "characteristic_phrases": ["<exact phrase from text>", ...],
  "humor_style": "<description or null if no humor evident>",
  "example_passages": ["<passage 1>", "<passage 2>", ...],
  "confidence": <float 0.0-1.0>
}

Guidelines:
- register: Describe the overall tone and level (e.g., "densely academic", \
"conversational with scholarly underpinning", "pastoral and meditative")
- sentence_patterns: Structural habits (e.g., "favors long periodic sentences \
with embedded clauses", "alternates between short declarative and expansive")
- vocabulary_tendencies: Recurring word choices, specialized terminology, \
favorite modifiers (e.g., "frequently uses 'sacramental', 'incarnational'")
- rhetorical_moves: Argumentative strategies (e.g., "builds arguments through \
close reading of poetry", "uses typological parallels between Old and New Testament")
- characteristic_phrases: Direct quotes (3-8 words) that exemplify the voice
- humor_style: Only if clearly present in samples; null otherwise
- example_passages: 3-5 representative passages (50-150 words each) that \
best capture the author's distinctive voice
- confidence: Based on corpus size and diversity of samples \
(0.3-0.5 for <5 samples, 0.5-0.7 for 5-15, 0.7-0.9 for 15-30, 0.9+ for 30+)\
"""


def _build_extraction_prompt(
    *,
    author_name: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Build the user prompt with sampled corpus chunks."""
    parts = [
        f"Author: {author_name}",
        f"Number of text samples: {len(chunks)}",
        "",
        "Text samples from across the author's corpus:",
    ]

    for i, chunk in enumerate(chunks, 1):
        work_id = chunk.get("work_id", "unknown")
        chapter = chunk.get("chapter", "")
        granularity = chunk.get("granularity", "meso")
        text = chunk.get("text", "")

        header = f"--- Sample {i} [work: {work_id}"
        if chapter:
            header += f", chapter: {chapter}"
        header += f", level: {granularity}] ---"

        parts.append(header)
        parts.append(text)
        parts.append("")

    parts.append(
        "Based on these samples, produce a comprehensive voice profile. "
        "Respond with JSON only, no additional text."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunk sampling strategy
# ---------------------------------------------------------------------------

MAX_CHUNKS_PER_EXTRACTION = 30
MIN_CHUNKS_FOR_EXTRACTION = 3


def _sample_diverse_chunks(
    chunks: list[dict[str, Any]],
    max_count: int = MAX_CHUNKS_PER_EXTRACTION,
) -> list[dict[str, Any]]:
    """Select diverse meso chunks across works, chapters, and positions.

    Strategy: group by work_id, then sample proportionally from each work
    to ensure cross-corpus representation. Within each work, prefer
    chunks from different chapters/sections.
    """
    if len(chunks) <= max_count:
        return chunks

    # Group by work_id
    by_work: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        wid = chunk.get("work_id", "unknown")
        by_work.setdefault(wid, []).append(chunk)

    # Proportional allocation per work
    n_works = len(by_work)
    per_work = max(1, max_count // n_works)
    remainder = max_count - (per_work * n_works)

    sampled: list[dict[str, Any]] = []
    for work_chunks in by_work.values():
        take = min(per_work, len(work_chunks))
        # Within a work, sample chunks from different chapters
        chapters_seen: set[str | None] = set()
        diverse: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []

        for c in work_chunks:
            ch = c.get("chapter")
            if ch not in chapters_seen:
                chapters_seen.add(ch)
                diverse.append(c)
            else:
                rest.append(c)

        selected = diverse[:take]
        if len(selected) < take:
            selected.extend(rest[: take - len(selected)])
        sampled.extend(selected)

    # Use remainder slots for additional diversity
    if remainder > 0:
        remaining = [c for c in chunks if c not in sampled]
        if remaining:
            extra = random.sample(remaining, min(remainder, len(remaining)))
            sampled.extend(extra)

    return sampled[:max_count]


# ---------------------------------------------------------------------------
# Voice profile extractor
# ---------------------------------------------------------------------------


class VoiceProfileExtractor:
    """Extracts structured voice profiles from an author's primary corpus.

    Uses the Anthropic API to analyze representative meso-level chunks
    from primary sources (voice_profile_eligible=true only).
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

    async def extract(
        self,
        *,
        author_id: str,
        author_name: str,
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
    ) -> VoiceProfile:
        """Extract a voice profile from the author's primary corpus.

        Args:
            author_id: The author's slug identifier.
            author_name: The author's canonical display name.
            work_repo: Repository for accessing work metadata.
            chunk_repo: Repository for accessing corpus chunks.

        Returns:
            A structured VoiceProfile.

        Raises:
            IntelligenceError: If extraction fails or corpus is insufficient.
        """
        if not self._api_key:
            raise IntelligenceError(
                "Anthropic API key is required for voice profile extraction",
                context={"author_id": author_id},
            )

        # Gather eligible primary works
        eligible_chunks = await self._gather_eligible_chunks(
            author_id=author_id,
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        if len(eligible_chunks) < MIN_CHUNKS_FOR_EXTRACTION:
            raise IntelligenceError(
                f"Insufficient primary corpus for voice extraction: "
                f"found {len(eligible_chunks)} chunks, need at least {MIN_CHUNKS_FOR_EXTRACTION}",
                context={"author_id": author_id, "chunks_found": len(eligible_chunks)},
            )

        # Sample diverse chunks
        sampled = _sample_diverse_chunks(eligible_chunks)

        log.info(
            "extracting_voice_profile",
            author_id=author_id,
            total_eligible=len(eligible_chunks),
            sampled=len(sampled),
            model=self._model,
        )

        user_prompt = _build_extraction_prompt(
            author_name=author_name,
            chunks=sampled,
        )

        # Fetch lesson context for injection
        lesson_context = ""
        lesson_ids = []
        if self._storage is not None:
            lesson_context, lesson_ids = await get_lesson_context(
                self._storage, "voice_profile"
            )

        try:
            profile_data = await self._call_anthropic(
                user_prompt, lesson_context=lesson_context
            )
        except IntelligenceError:
            raise
        except Exception as exc:
            raise IntelligenceError(
                f"Voice profile extraction failed: {exc}",
                context={"author_id": author_id},
                cause=exc,
            ) from exc

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

        profile = VoiceProfile(
            author_id=author_id,
            **profile_data,
        )

        log.info(
            "voice_profile_extracted",
            author_id=author_id,
            confidence=profile.confidence,
            n_characteristic_phrases=len(profile.characteristic_phrases),
        )

        return profile

    async def _gather_eligible_chunks(
        self,
        *,
        author_id: str,
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
    ) -> list[dict[str, Any]]:
        """Gather meso chunks from primary works where voice_profile_eligible=true."""
        works = await work_repo.list_by_author(author_id)

        eligible_work_ids: list[str] = []
        for work in works:
            if work.get("source_class") != "primary":
                continue
            # Check voice_profile_eligible in source_metadata
            meta = work.get("source_metadata", {})
            if isinstance(meta, str):
                import json as _json

                meta = _json.loads(meta)
            # Default to True for primary sources (matching PrimaryCatalogEntry default)
            if meta.get("voice_profile_eligible", True):
                eligible_work_ids.append(work["work_id"])

        log.info(
            "eligible_primary_works",
            author_id=author_id,
            total_works=len(works),
            eligible=len(eligible_work_ids),
        )

        # Fetch meso-level chunks from eligible works
        all_chunks: list[dict[str, Any]] = []
        for work_id in eligible_work_ids:
            chunks = await chunk_repo.list_by_work(work_id, granularity="meso")
            # Only include primary source chunks
            primary_chunks = [c for c in chunks if c.get("source_class") == "primary"]
            all_chunks.extend(primary_chunks)

        return all_chunks

    async def _call_anthropic(
        self, user_prompt: str, *, lesson_context: str = ""
    ) -> dict[str, Any]:
        """Call the Anthropic API and parse the voice profile response."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        system = VOICE_EXTRACTION_SYSTEM
        if lesson_context:
            system = f"{system}\n\n{lesson_context}"

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise IntelligenceError(
                f"Anthropic API error during voice extraction: {exc}",
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
                "Empty response from Anthropic API during voice extraction",
                context={"model": self._model},
            )

        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse the LLM's JSON response into voice profile data."""
        from author_library.intelligence.json_parser import extract_json

        try:
            data: dict[str, Any] = extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise IntelligenceError(
                f"Failed to parse voice profile response as JSON: {exc}",
                context={"response_text": response_text[:500]},
                cause=exc,
            ) from exc

        required_fields = [
            "register",
            "sentence_patterns",
            "vocabulary_tendencies",
            "rhetorical_moves",
            "characteristic_phrases",
            "confidence",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise IntelligenceError(
                f"Voice profile response missing required fields: {missing}",
                context={"fields_present": list(data.keys())},
            )

        return data
