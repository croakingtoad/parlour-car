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

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic
import structlog

from author_library.catalog.models import SourceClass
from author_library.errors import IngestionError

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.config import APIKeySettings, LLMSettings
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Maximum chunks per LLM batch call
_BATCH_SIZE = 10

# Maximum retries for a batch when JSON parsing fails
_MAX_BATCH_RETRIES = 2

_CODE_FENCE_RE = re.compile(r"^```\w*\s*\n(.*?)```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles both ```json ... ``` and ``` ... ``` variants.
    Returns the inner content if fences are found, otherwise
    returns the stripped input.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


_EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction engine for an author-studies knowledge graph.
Given one or more text chunks from a literary/scholarly work, extract structured entities.

For EACH chunk, return a JSON object with:
- "chunk_id": the chunk's id (string)
- "themes": list of {"name": str, "canonical_name": str} — broad thematic topics
- "arguments": list of {"claim": str, "evidence_summary": str} — specific intellectual claims
- "concepts": list of {"name": str, "canonical_name": str} — technical/philosophical terms
- "persons": list of {"name": str, "canonical_name": str, "role": str} — people mentioned

Rules:
- canonical_name: lowercase, hyphenated form for deduplication (e.g. "primary-imagination")
- Themes should be broad (e.g. "poetry-as-truth-bearing", "sacramental-theology")
- Arguments should capture specific claims or positions the author takes
- Concepts are technical terms, not general vocabulary
- Persons include authors, thinkers, historical figures referenced — not the work's author
- role for persons: "referenced", "quoted", "discussed", "influenced-by"

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
    ) -> None:
        self._neo4j = neo4j
        self._llm_settings = llm_settings
        api_key = api_keys.anthropic_api_key.get_secret_value()
        if not api_key:
            raise IngestionError(
                "Anthropic API key required for entity extraction",
                context={"component": "entity_extraction"},
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def extract_and_persist(
        self,
        chunks: list[Chunk],
        *,
        work_title: str = "",
        author: str = "",
    ) -> ExtractionResult:
        """Extract entities from chunks and write nodes/edges to Neo4j.

        Chunks are processed in batches of _BATCH_SIZE for LLM efficiency.
        On JSON parse failure, the batch is split in half and retried up to
        _MAX_BATCH_RETRIES times.
        """
        result = ExtractionResult()

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            extractions = await self._extract_batch_with_retry(
                batch, work_title=work_title, author=author, result=result
            )

            for extraction in extractions:
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

        log.info(
            "entity_extraction_complete",
            chunks_processed=len(chunks),
            nodes_created=result.nodes_created,
            edges_created=result.edges_created,
            errors=len(result.errors),
        )
        return result

    async def _extract_batch_with_retry(
        self,
        batch: list[Chunk],
        *,
        work_title: str,
        author: str,
        result: ExtractionResult,
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
            _retry_depth: Current retry depth (max _MAX_BATCH_RETRIES).

        Returns:
            List of ChunkExtraction objects from successful parses.
        """
        try:
            return await self._extract_batch(batch, work_title=work_title, author=author)
        except json.JSONDecodeError as exc:
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
                _retry_depth=_retry_depth + 1,
            )
            right = await self._extract_batch_with_retry(
                batch[mid:],
                work_title=work_title,
                author=author,
                result=result,
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

        user_message = (
            f"Work: {work_title}\nAuthor: {author}\n\n"
            f"Extract entities from these {len(chunks)} chunks:\n\n"
            f"{json.dumps(chunks_payload, indent=2)}"
        )

        response = await self._client.messages.create(
            model=self._llm_settings.ingestion_model,
            max_tokens=4096,
            system=_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text  # type: ignore[union-attr]
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
            log.error(
                "entity_extraction_json_parse_failed",
                response_length=len(response_text),
                response_preview=response_text[:500],
            )
            raise
        extractions: list[ChunkExtraction] = []

        for raw in raw_list:
            chunk_id = raw["chunk_id"]
            themes = [
                ExtractedEntity(
                    entity_type="theme",
                    name=t["name"],
                    canonical_name=t["canonical_name"],
                )
                for t in raw.get("themes", [])
            ]
            arguments = [
                ExtractedEntity(
                    entity_type="argument",
                    name=a["claim"],
                    canonical_name=a["claim"][:80].lower().replace(" ", "-"),
                    properties={"evidence_summary": a.get("evidence_summary", "")},
                )
                for a in raw.get("arguments", [])
            ]
            concepts = [
                ExtractedEntity(
                    entity_type="concept",
                    name=c["name"],
                    canonical_name=c["canonical_name"],
                )
                for c in raw.get("concepts", [])
            ]
            persons = [
                ExtractedEntity(
                    entity_type="person",
                    name=p["name"],
                    canonical_name=p["canonical_name"],
                    properties={"role": p.get("role", "referenced")},
                )
                for p in raw.get("persons", [])
            ]
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

        # Arguments — source-class-gated edges
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
                # PRIMARY sources MAKE arguments
                await self._neo4j.execute_write(
                    """MATCH (c:Chunk {chunk_id: $chunk_id}),
                           (a:Argument {canonical_name: $canonical_name})
                    MERGE (c)-[:MAKES_ARGUMENT]->(a)""",
                    {"chunk_id": chunk.id, "canonical_name": argument.canonical_name},
                )
                edges += 1
            elif source_class == SourceClass.SECONDARY:
                # SECONDARY sources ATTRIBUTE arguments to the critic
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
            # CONTEXTUAL and TERTIARY: no argument edges

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
