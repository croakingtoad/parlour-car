"""LLM-powered entity extraction from document chunks.

Extracts themes, arguments, concepts, and persons from chunks using
the Anthropic API, then writes corresponding nodes and edges to Neo4j.

Edge rules enforced by source classification:
  - EXPLORES_THEME: all source classes
  - MAKES_ARGUMENT: PRIMARY only
  - ATTRIBUTED_BY_CRITIC: SECONDARY only
  - REFERENCES_PERSON: all source classes
  - CONCEPT_USED_IN: all source classes
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic
import structlog

from author_library.catalog.models import SourceClass
from author_library.errors import IngestionError

from author_library.intelligence.lesson_writer import get_lesson_context

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.config import APIKeySettings, LLMSettings
    from author_library.storage.manager import StorageManager
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Maximum chunks per LLM batch call
_BATCH_SIZE = 10

# Maximum retries for a batch when JSON parsing fails
_MAX_BATCH_RETRIES = 2

_CODE_FENCE_RE = re.compile(r"^```\w*\s*\n(.*?)```\s*$", re.DOTALL)

# Handles truncated responses where closing ``` is missing
_CODE_FENCE_OPEN_RE = re.compile(r"^```\w*\s*\n(.*)$", re.DOTALL)

# Fenced block anywhere in the text (prose preamble), closed fence optional
_CODE_FENCE_ANY_RE = re.compile(r"```\w*\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles both ```json ... ``` and ``` ... ``` variants,
    including truncated responses where the closing fence is
    missing (e.g. due to max_tokens truncation).
    Returns the inner content if fences are found, otherwise
    returns the stripped input.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    # Fallback: opening fence present but closing fence missing (truncated)
    match = _CODE_FENCE_OPEN_RE.match(stripped)
    if match:
        return match.group(1).strip()
    # LLM sometimes writes prose before the payload ("Looking at these
    # chunks... ```json [...]") — find a fenced block anywhere, closed or not
    match = _CODE_FENCE_ANY_RE.search(stripped)
    if match:
        return match.group(1).strip()
    # Prose followed by a bare JSON array
    first_bracket = stripped.find("[")
    if first_bracket > 0:
        return stripped[first_bracket:]
    return stripped


_EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction engine for an author-studies knowledge graph.
Given one or more text chunks from a literary/scholarly work, extract structured entities.

For EACH chunk, return a JSON object with:
- "chunk_id": the chunk's id (string)
- "themes": list of {"name": str, "canonical_name": str} — broad thematic topics (max 5)
- "arguments": list of {"claim": str, "evidence_summary": str} — specific intellectual claims (max 3)
- "concepts": list of {"name": str, "canonical_name": str} — technical/philosophical terms (max 5)
- "persons": list of {"name": str, "canonical_name": str, "role": str} — people mentioned (max 3)

Rules:
- canonical_name: lowercase, hyphenated form for deduplication (e.g. "primary-imagination")
- CRITICAL for themes: If an existing theme in the knowledge graph covers the same concept, \
REUSE its exact canonical_name. Do NOT create a new variant. For example, if \
"imagination-and-theology" already exists, do not create "imagination-and-the-divine" or \
"imagination-as-divine-faculty" — use "imagination-and-theology" instead.
- Themes should be broad (e.g. "poetry-as-truth-bearing", "sacramental-theology")
- Only create a new theme if no existing theme adequately covers the concept
- Arguments should capture specific claims or positions the author takes
- Concepts are technical terms, not general vocabulary
- Persons include authors, thinkers, historical figures referenced — not the work's author
- role for persons: "referenced", "quoted", "discussed", "influenced-by"
- Keep evidence_summary under 50 words
- Respect the per-chunk maximums above to ensure complete JSON output

Return a JSON array of extraction objects, one per chunk.
"""


@dataclass(frozen=True)
class ExtractedEntity:
    """A single entity extracted from a chunk."""

    entity_type: str  # "theme", "argument", "concept", "person"
    name: str
    canonical_name: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkExtraction:
    """All entities extracted from a single chunk."""

    chunk_id: str
    themes: list[ExtractedEntity]
    arguments: list[ExtractedEntity]
    concepts: list[ExtractedEntity]
    persons: list[ExtractedEntity]


@dataclass
class ExtractionResult:
    """Aggregated result of entity extraction across multiple chunks."""

    extractions: list[ChunkExtraction] = field(default_factory=list)
    nodes_created: int = 0
    edges_created: int = 0
    errors: list[str] = field(default_factory=list)


class EntityExtractor:
    """Extracts entities from chunks via LLM and persists to Neo4j."""

    def __init__(
        self,
        neo4j: Neo4jConnection,
        api_keys: APIKeySettings,
        llm_settings: LLMSettings,
        storage: StorageManager | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._llm_settings = llm_settings
        self._storage = storage
        api_key = api_keys.anthropic_api_key.get_secret_value()
        if not api_key:
            raise IngestionError(
                "Anthropic API key required for entity extraction",
                context={"component": "entity_extraction"},
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _fetch_existing_themes(self) -> list[str]:
        """Fetch existing Theme canonical_names from Neo4j.

        Used to provide the LLM with context about what themes already
        exist so it can reuse them instead of generating near-duplicates.
        Returns up to 500 theme names sorted by relationship count.
        """
        try:
            records = await self._neo4j.execute_read(
                """MATCH (t:Theme)
                OPTIONAL MATCH (t)<-[r:EXPLORES_THEME]-()
                WITH t, count(r) AS rel_count
                RETURN t.canonical_name AS name
                ORDER BY rel_count DESC
                LIMIT 500"""
            )
            return [r["name"] for r in records if r.get("name")]
        except Exception:
            log.warning("entity_extraction_fetch_themes_failed", exc_info=True)
            return []

    async def extract_and_persist(
        self,
        chunks: list[Chunk],
        *,
        work_title: str = "",
        author: str = "",
    ) -> ExtractionResult:
        """Extract entities from chunks and write nodes/edges to Neo4j.

        Chunks are processed in batches of _BATCH_SIZE for LLM efficiency.
        Batches run concurrently (up to ``entity_extraction_concurrency``
        parallel API calls).  On JSON parse failure, a batch is split in
        half and retried up to _MAX_BATCH_RETRIES times.
        """
        result = ExtractionResult()
        if not chunks:
            return result

        # Fetch existing themes to help the LLM reuse canonical names
        existing_themes = await self._fetch_existing_themes()

        # Fetch lesson context once for all batches
        lesson_context = ""
        lesson_ids = []
        if self._storage is not None:
            lesson_context, lesson_ids = await get_lesson_context(
                self._storage, "entity_extraction"
            )

        concurrency = self._llm_settings.entity_extraction_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        # Build (batch, batch_chunks) pairs
        batches: list[list[Chunk]] = []
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batches.append(chunks[batch_start : batch_start + _BATCH_SIZE])

        total_batches = len(batches)
        wall_start = time.monotonic()

        log.info(
            "entity_extraction_starting",
            total_chunks=len(chunks),
            total_batches=total_batches,
            concurrency=concurrency,
        )

        async def _process_batch(batch: list[Chunk]) -> list[ChunkExtraction]:
            async with semaphore:
                return await self._extract_batch_with_retry(
                    batch,
                    work_title=work_title,
                    author=author,
                    result=result,
                    existing_themes=existing_themes,
                    lesson_context=lesson_context,
                )

        # Run all batches concurrently (bounded by semaphore)
        batch_results = await asyncio.gather(
            *(_process_batch(b) for b in batches),
            return_exceptions=True,
        )

        wall_elapsed = time.monotonic() - wall_start

        # Persist results sequentially (Neo4j writes are fast, not the bottleneck)
        for batch, batch_extractions in zip(batches, batch_results):
            if isinstance(batch_extractions, BaseException):
                error_msg = f"Batch extraction raised: {batch_extractions}"
                log.error("entity_extraction_batch_exception", error=error_msg)
                result.errors.append(error_msg)
                continue

            for extraction in batch_extractions:
                chunk = next((c for c in batch if c.id == extraction.chunk_id), None)
                if chunk is None:
                    continue
                try:
                    nodes, edges = await self._persist_extraction(extraction, chunk)
                    result.nodes_created += nodes
                    result.edges_created += edges
                except Exception as exc:
                    error_msg = f"Persist failed for chunk {extraction.chunk_id}: {exc}"
                    log.error("entity_persist_failed", chunk_id=extraction.chunk_id, error=str(exc))
                    result.errors.append(error_msg)

        # Track that these lessons were applied
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

        log.info(
            "entity_extraction_complete",
            chunks_processed=len(chunks),
            total_batches=total_batches,
            concurrency=concurrency,
            wall_clock_seconds=round(wall_elapsed, 1),
            nodes_created=result.nodes_created,
            edges_created=result.edges_created,
            errors=len(result.errors),
            lessons_injected=len(lesson_ids),
        )
        return result

    async def _extract_batch_with_retry(
        self,
        batch: list[Chunk],
        *,
        work_title: str,
        author: str,
        result: ExtractionResult,
        existing_themes: list[str] | None = None,
        lesson_context: str = "",
        _retry_depth: int = 0,
    ) -> list[ChunkExtraction]:
        """Extract entities from a batch, retrying with smaller sub-batches on JSON parse failure.

        On json.JSONDecodeError, the batch is split in half and each half is
        retried independently. This handles the case where the LLM produces
        malformed JSON (unterminated strings, etc.) for large batches.

        Args:
            batch: List of chunks to extract from.
            work_title: Title of the work being processed.
            author: Author name.
            result: Accumulator for errors.
            existing_themes: List of existing theme canonical_names for reuse.
            _retry_depth: Current retry depth (max _MAX_BATCH_RETRIES).

        Returns:
            List of ChunkExtraction objects from successful parses.
        """
        try:
            return await self._extract_batch(
                batch,
                work_title=work_title,
                author=author,
                existing_themes=existing_themes,
                lesson_context=lesson_context,
            )
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError) as exc:
            if _retry_depth >= _MAX_BATCH_RETRIES:
                error_msg = (
                    f"Batch JSON parse failed after {_retry_depth} retries "
                    f"({len(batch)} chunks): {exc}"
                )
                log.error(
                    "entity_extraction_json_failed_final",
                    batch_size=len(batch),
                    retry_depth=_retry_depth,
                    error=str(exc),
                )
                result.errors.append(error_msg)
                return []

            if len(batch) <= 1:
                # Single chunk batch still failing -- log and skip
                error_msg = (
                    f"Single-chunk extraction JSON parse failed "
                    f"(chunk {batch[0].id}): {exc}"
                )
                log.error(
                    "entity_extraction_single_chunk_json_failed",
                    chunk_id=batch[0].id,
                    error=str(exc),
                )
                result.errors.append(error_msg)
                return []

            # Split and retry each half
            mid = len(batch) // 2
            log.warning(
                "entity_extraction_json_retry",
                batch_size=len(batch),
                retry_depth=_retry_depth + 1,
                split_sizes=[mid, len(batch) - mid],
                error=str(exc),
            )

            left = await self._extract_batch_with_retry(
                batch[:mid],
                work_title=work_title,
                author=author,
                result=result,
                existing_themes=existing_themes,
                lesson_context=lesson_context,
                _retry_depth=_retry_depth + 1,
            )
            right = await self._extract_batch_with_retry(
                batch[mid:],
                work_title=work_title,
                author=author,
                result=result,
                existing_themes=existing_themes,
                lesson_context=lesson_context,
                _retry_depth=_retry_depth + 1,
            )
            return left + right
        except Exception as exc:
            error_msg = f"Batch extraction failed ({len(batch)} chunks): {exc}"
            log.error("entity_extraction_batch_failed", batch_size=len(batch), error=str(exc))
            result.errors.append(error_msg)
            return []

    async def _extract_batch(
        self,
        chunks: list[Chunk],
        *,
        work_title: str,
        author: str,
        existing_themes: list[str] | None = None,
        lesson_context: str = "",
    ) -> list[ChunkExtraction]:
        """Call Anthropic API to extract entities from a batch of chunks.

        Raises json.JSONDecodeError if the response is not valid JSON
        (handled by _extract_batch_with_retry for splitting and retrying).
        """
        chunks_payload = []
        for chunk in chunks:
            chunks_payload.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source_class": chunk.source_class,
                    "granularity": str(chunk.granularity),
                    "chapter": chunk.chapter or "",
                    "section": chunk.section or "",
                }
            )

        # Build user message with optional existing theme context
        theme_context = ""
        if existing_themes:
            # Format as JSON objects to reinforce expected output structure
            theme_examples = [
                f'{{"name": "{cn.replace("-", " ").title()}", "canonical_name": "{cn}"}}'
                for cn in existing_themes[:100]
            ]
            theme_context = (
                f"\n\nExisting themes in the knowledge graph (reuse these "
                f"exact canonical_names when the concept matches — return them "
                f"as objects, not bare strings):\n[{', '.join(theme_examples)}]\n"
            )

        user_message = (
            f"Work: {work_title}\nAuthor: {author}\n"
            f"{theme_context}\n"
            f"Extract entities from these {len(chunks)} chunks:\n\n"
            f"{json.dumps(chunks_payload, indent=2)}"
        )

        system = _EXTRACTION_SYSTEM_PROMPT
        if lesson_context:
            system = f"{system}\n\n{lesson_context}"

        response = await self._client.messages.create(
            model=self._llm_settings.ingestion_model,
            max_tokens=16384,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text  # type: ignore[union-attr]

        # Detect max_tokens truncation so the salvage path can fire
        if response.stop_reason == "max_tokens":
            log.warning(
                "entity_extraction_response_truncated",
                batch_size=len(chunks),
                response_length=len(response_text),
                max_tokens=16384,
            )

        return self._parse_extraction_response(response_text)

    def _parse_extraction_response(self, response_text: str) -> list[ChunkExtraction]:
        """Parse the LLM JSON response into ChunkExtraction objects.

        Handles common LLM output issues:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - Logs raw response text on parse failure for debugging

        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON
                after stripping code fences.
        """
        text = _strip_code_fences(response_text)

        try:
            raw_list: list[dict[str, Any]] = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to salvage truncated JSON (e.g. max_tokens cut off
            # the response mid-array).  Find the last complete object by
            # locating the final '}' and closing the array.
            last_brace = text.rfind("}")
            if last_brace > 0:
                candidate = text[: last_brace + 1].rstrip().rstrip(",") + "]"
                try:
                    raw_list = json.loads(candidate)
                    log.warning(
                        "entity_extraction_json_truncated_salvaged",
                        original_length=len(response_text),
                        salvaged_objects=len(raw_list),
                    )
                except json.JSONDecodeError:
                    log.error(
                        "entity_extraction_json_parse_failed",
                        response_length=len(response_text),
                        response_preview=response_text[:500],
                    )
                    raise
            else:
                log.error(
                    "entity_extraction_json_parse_failed",
                    response_length=len(response_text),
                    response_preview=response_text[:500],
                )
                raise
        extractions: list[ChunkExtraction] = []

        for raw in raw_list:
            if not isinstance(raw, dict):
                log.warning("entity_extraction_skip_non_dict", type=type(raw).__name__, preview=str(raw)[:100])
                continue
            chunk_id = raw.get("chunk_id", "")
            if not chunk_id:
                continue
            # CRITICAL: LLM sometimes returns list items as bare strings instead of
            # dicts (e.g. themes: ["Belovedness"] instead of [{name: "Belovedness"}]).
            # All four entity lists MUST guard with isinstance(x, dict) before calling
            # .get() — failure to do so causes 'str object has no attribute get' which
            # triggers the batch-retry cascade and wastes API calls.
            themes = []
            for t in raw.get("themes", []):
                if isinstance(t, dict) and (t.get("name") or t.get("canonical_name")):
                    themes.append(ExtractedEntity(
                        entity_type="theme",
                        name=t.get("name", ""),
                        canonical_name=t.get("canonical_name", "")
                        or t.get("name", "").lower().replace(" ", "-")[:80],
                    ))
                elif isinstance(t, str) and t.strip():
                    # LLM returned a bare canonical_name string — recover it
                    cn = t.strip().lower().replace(" ", "-")[:80]
                    themes.append(ExtractedEntity(
                        entity_type="theme",
                        name=cn.replace("-", " ").title(),
                        canonical_name=cn,
                    ))
            arguments = []
            for a in raw.get("arguments", []):
                if isinstance(a, dict) and a.get("claim"):
                    arguments.append(ExtractedEntity(
                        entity_type="argument",
                        name=a.get("claim", ""),
                        canonical_name=a.get("claim", "")[:80].lower().replace(" ", "-"),
                        properties={"evidence_summary": a.get("evidence_summary", "")},
                    ))
                elif isinstance(a, str) and a.strip():
                    # LLM returned the claim as a bare string — recover it
                    claim = a.strip()
                    arguments.append(ExtractedEntity(
                        entity_type="argument",
                        name=claim,
                        canonical_name=claim[:80].lower().replace(" ", "-"),
                        properties={"evidence_summary": ""},
                    ))
            concepts = []
            for c in raw.get("concepts", []):
                if isinstance(c, dict) and (c.get("name") or c.get("canonical_name")):
                    concepts.append(ExtractedEntity(
                        entity_type="concept",
                        name=c.get("name", ""),
                        canonical_name=c.get("canonical_name", "")
                        or c.get("name", "").lower().replace(" ", "-")[:80],
                    ))
                elif isinstance(c, str) and c.strip():
                    cn = c.strip().lower().replace(" ", "-")[:80]
                    concepts.append(ExtractedEntity(
                        entity_type="concept",
                        name=cn.replace("-", " ").title(),
                        canonical_name=cn,
                    ))
            persons = []
            for p in raw.get("persons", []):
                if isinstance(p, dict) and (p.get("name") or p.get("canonical_name")):
                    persons.append(ExtractedEntity(
                        entity_type="person",
                        name=p.get("name", ""),
                        canonical_name=p.get("canonical_name", "")
                        or p.get("name", "").lower().replace(" ", "-")[:80],
                        properties={"role": p.get("role", "referenced")},
                    ))
                elif isinstance(p, str) and p.strip():
                    cn = p.strip().lower().replace(" ", "-")[:80]
                    persons.append(ExtractedEntity(
                        entity_type="person",
                        name=cn.replace("-", " ").title(),
                        canonical_name=cn,
                        properties={"role": "referenced"},
                    ))
            # Log if LLM returned bare strings (recovered as entities, not lost)
            for field_name, items in (
                ("themes", raw.get("themes", [])),
                ("arguments", raw.get("arguments", [])),
                ("concepts", raw.get("concepts", [])),
                ("persons", raw.get("persons", [])),
            ):
                non_dicts = [x for x in items if isinstance(x, str)]
                if non_dicts:
                    log.debug(
                        "entity_extraction_bare_strings_recovered",
                        chunk_id=chunk_id,
                        field=field_name,
                        count=len(non_dicts),
                        preview=str(non_dicts[:3])[:200],
                    )
            extractions.append(
                ChunkExtraction(
                    chunk_id=chunk_id,
                    themes=themes,
                    arguments=arguments,
                    concepts=concepts,
                    persons=persons,
                )
            )

        return extractions

    async def _persist_extraction(
        self,
        extraction: ChunkExtraction,
        chunk: Chunk,
    ) -> tuple[int, int]:
        """Write extracted entities as Neo4j nodes and edges.

        Returns (nodes_created, edges_created).
        """
        nodes = 0
        edges = 0
        source_class = chunk.source_class

        # Ensure chunk node exists
        await self._neo4j.execute_write(
            """MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.work_id = $work_id,
                c.text_preview = $text_preview,
                c.granularity = $granularity,
                c.source_class = $source_class""",
            {
                "chunk_id": chunk.id,
                "work_id": chunk.work_id,
                "text_preview": chunk.text[:200],
                "granularity": str(chunk.granularity),
                "source_class": source_class,
            },
        )

        # Themes — all source classes get EXPLORES_THEME
        for theme in extraction.themes:
            await self._neo4j.execute_write(
                """MERGE (t:Theme {canonical_name: $canonical_name})
                SET t.name = $name""",
                {"canonical_name": theme.canonical_name, "name": theme.name},
            )
            nodes += 1
            await self._neo4j.execute_write(
                """MATCH (c:Chunk {chunk_id: $chunk_id}),
                       (t:Theme {canonical_name: $canonical_name})
                MERGE (c)-[:EXPLORES_THEME]->(t)""",
                {"chunk_id": chunk.id, "canonical_name": theme.canonical_name},
            )
            edges += 1

        # Arguments — only create nodes when an edge will also be created.
        # PRIMARY → MAKES_ARGUMENT, SECONDARY → ATTRIBUTED_BY_CRITIC.
        # CONTEXTUAL/TERTIARY/REFERENCE get no argument edges, so skip node creation too.
        if source_class in (SourceClass.PRIMARY, SourceClass.SECONDARY):
            for argument in extraction.arguments:
                await self._neo4j.execute_write(
                    """MERGE (a:Argument {canonical_name: $canonical_name})
                    SET a.claim = $claim, a.evidence_summary = $evidence_summary""",
                    {
                        "canonical_name": argument.canonical_name,
                        "claim": argument.name,
                        "evidence_summary": argument.properties.get("evidence_summary", ""),
                    },
                )
                nodes += 1

                if source_class == SourceClass.PRIMARY:
                    await self._neo4j.execute_write(
                        """MATCH (c:Chunk {chunk_id: $chunk_id}),
                               (a:Argument {canonical_name: $canonical_name})
                        MERGE (c)-[:MAKES_ARGUMENT]->(a)""",
                        {"chunk_id": chunk.id, "canonical_name": argument.canonical_name},
                    )
                    edges += 1
                else:  # SECONDARY
                    await self._neo4j.execute_write(
                        """MATCH (c:Chunk {chunk_id: $chunk_id}),
                               (a:Argument {canonical_name: $canonical_name})
                        MERGE (c)-[r:ATTRIBUTED_BY_CRITIC]->(a)
                        SET r.work_id = $work_id""",
                        {
                            "chunk_id": chunk.id,
                            "canonical_name": argument.canonical_name,
                            "work_id": chunk.work_id,
                        },
                    )
                    edges += 1

        # Concepts — all source classes
        for concept in extraction.concepts:
            await self._neo4j.execute_write(
                """MERGE (co:Concept {canonical_name: $canonical_name})
                SET co.name = $name""",
                {"canonical_name": concept.canonical_name, "name": concept.name},
            )
            nodes += 1
            await self._neo4j.execute_write(
                """MATCH (c:Chunk {chunk_id: $chunk_id}),
                       (co:Concept {canonical_name: $canonical_name})
                MERGE (c)-[:CONCEPT_USED_IN]->(co)""",
                {"chunk_id": chunk.id, "canonical_name": concept.canonical_name},
            )
            edges += 1

        # Persons — all source classes
        for person in extraction.persons:
            await self._neo4j.execute_write(
                """MERGE (p:Person {canonical_name: $canonical_name})
                SET p.name = $name""",
                {"canonical_name": person.canonical_name, "name": person.name},
            )
            nodes += 1
            await self._neo4j.execute_write(
                """MATCH (c:Chunk {chunk_id: $chunk_id}),
                       (p:Person {canonical_name: $canonical_name})
                MERGE (c)-[r:REFERENCES_PERSON]->(p)
                SET r.role = $role""",
                {
                    "chunk_id": chunk.id,
                    "canonical_name": person.canonical_name,
                    "role": person.properties.get("role", "referenced"),
                },
            )
            edges += 1

        return nodes, edges
