"""Cross-work thematic evolution analysis.

Traces how the author's treatment of each theme develops chronologically
across their corpus, identifying shifts in thinking, explicit self-reflection,
and evolution of key arguments. Creates DEVELOPS_FROM edges in Neo4j
between related arguments in different works.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IntelligenceError

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.intelligence.thematic_index import ThematicEntry
    from author_library.storage.repositories import (
        ChunkRepository,
        GraphRepository,
        WorkRepository,
    )

from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


def _dated_primary_work_years(works: list[dict[str, Any]]) -> dict[str, int]:
    """Map dated primary works to years for chronological analysis."""
    years: dict[str, int] = {}
    for work in works:
        year = work.get("publication_year")
        if work.get("source_class") == "primary" and isinstance(year, int):
            years[work["work_id"]] = year
    return years


# ---------------------------------------------------------------------------
# Evolution data models
# ---------------------------------------------------------------------------


class EvolutionStep(BaseModel):
    """A single step in the evolution of a theme across works."""

    work_id: str
    publication_year: int
    summary: str
    key_chunk_ids: list[str] = Field(default_factory=list)
    self_reflection: bool = False
    self_reflection_note: str | None = None


class ThematicEvolution(BaseModel):
    """Chronological evolution of a theme across an author's corpus."""

    theme: str
    narrative: str
    steps: list[EvolutionStep] = Field(default_factory=list)
    develops_from_edges: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVOLUTION_SYSTEM = """\
You are a literary scholar tracing how an author's thinking on a specific \
theme evolves chronologically across their body of work. You identify \
shifts in position, deepening of understanding, explicit self-reflection \
(where the author writes about their own earlier work), and connections \
between arguments in different works.

Given text samples organized by work (in chronological order), analyze \
the evolution of the specified theme.

Respond with valid JSON matching this schema:
{
  "narrative": "<2-5 paragraph chronological narrative of how the author's thinking evolved>",
  "steps": [
    {
      "work_id": "<work identifier>",
      "publication_year": <int>,
      "summary": "<how this theme is treated in this work>",
      "key_chunk_ids": ["<chunk_id 1>", ...],
      "self_reflection": true/false,
      "self_reflection_note": "<if true, describe the self-reflection>"
    },
    ...
  ],
  "develops_from": [
    {
      "from_chunk_id": "<earlier chunk>",
      "to_chunk_id": "<later chunk that develops from it>",
      "relationship_note": "<how the later passage develops from the earlier>"
    },
    ...
  ]
}

Guidelines:
- Order steps chronologically by publication year
- Self-reflection is GOLD: flag any passage where the author explicitly \
refers to their own earlier work, revises a previous position, or notes \
how their thinking has changed
- develops_from edges connect specific passages (by chunk_id) where a later \
argument clearly builds on, responds to, or revises an earlier one
- The narrative should tell a coherent story of intellectual development
- If no clear evolution exists, say so honestly in the narrative\
"""


# ---------------------------------------------------------------------------
# Thematic evolution analyzer
# ---------------------------------------------------------------------------


class ThematicEvolutionAnalyzer:
    """Analyzes how themes evolve across an author's corpus chronologically.

    For each theme in the thematic index, traces development across works
    and creates DEVELOPS_FROM edges in the knowledge graph.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.api_keys.anthropic_api_key.get_secret_value()
        self._model = settings.llm.ingestion_model

    async def analyze(
        self,
        *,
        author_id: str,
        themes: list[ThematicEntry],
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
        graph_repo: GraphRepository,
    ) -> list[ThematicEvolution]:
        """Analyze thematic evolution and create graph edges.

        Args:
            author_id: The author's slug identifier.
            themes: Thematic entries from the thematic index.
            work_repo: Repository for accessing work metadata.
            chunk_repo: Repository for accessing corpus chunks.
            graph_repo: Neo4j graph repository for creating edges.

        Returns:
            List of ThematicEvolution objects.

        Raises:
            IntelligenceError: If analysis fails.
        """
        if not self._api_key:
            raise IntelligenceError(
                "Anthropic API key is required for thematic evolution analysis",
                context={"author_id": author_id},
            )

        # Build work chronology
        works = await work_repo.list_by_author(author_id)
        work_years = _dated_primary_work_years(works)

        evolutions: list[ThematicEvolution] = []

        for theme in themes:
            if not theme.appearances:
                continue

            log.info(
                "analyzing_evolution",
                author_id=author_id,
                theme=theme.theme,
                n_appearances=len(theme.appearances),
            )

            evolution = await self._analyze_theme_evolution(
                theme=theme,
                work_years=work_years,
                chunk_repo=chunk_repo,
            )

            # Create DEVELOPS_FROM edges in Neo4j
            for edge in evolution.develops_from_edges:
                await graph_repo.create_edge(
                    from_label="Chunk",
                    from_key="chunk_id",
                    from_value=edge["from_chunk_id"],
                    rel_type="DEVELOPS_FROM",
                    to_label="Chunk",
                    to_key="chunk_id",
                    to_value=edge["to_chunk_id"],
                    properties={
                        "theme": theme.theme,
                        "relationship_note": edge.get("relationship_note", ""),
                    },
                )

            evolutions.append(evolution)

        log.info(
            "evolution_analysis_complete",
            author_id=author_id,
            n_themes_analyzed=len(evolutions),
            total_edges=sum(len(e.develops_from_edges) for e in evolutions),
        )

        return evolutions

    async def _analyze_theme_evolution(
        self,
        *,
        theme: ThematicEntry,
        work_years: dict[str, int],
        chunk_repo: ChunkRepository,
    ) -> ThematicEvolution:
        """Analyze evolution of a single theme across works."""
        # Gather chunks for each appearance, sorted chronologically
        work_chunks: list[tuple[str, int, list[dict[str, Any]]]] = []

        for appearance in theme.appearances:
            year = work_years.get(appearance.work_id)
            if year is None:
                continue
            chunks = await chunk_repo.list_by_work(
                appearance.work_id, granularity="meso"
            )
            primary = [c for c in chunks if c.get("source_class") == "primary"]
            if primary:
                work_chunks.append((appearance.work_id, year, primary))

        # Sort by publication year
        work_chunks.sort(key=lambda x: x[1])

        if not work_chunks:
            return ThematicEvolution(
                theme=theme.theme,
                narrative="Insufficient data to trace evolution.",
            )

        # Build prompt with chronologically ordered samples
        parts: list[str] = []
        for work_id, year, chunks in work_chunks:
            parts.append(f"\n=== {work_id} (published {year}) ===")
            # Take up to 5 representative chunks per work
            for chunk in chunks[:5]:
                chunk_id = str(chunk.get("id", chunk.get("chunk_id", "")))
                text = chunk.get("text", "")
                parts.append(f"[chunk_id: {chunk_id}]")
                parts.append(text[:500])
                parts.append("")

        corpus_text = "\n".join(parts)

        user_prompt = (
            f"Theme: {theme.theme}\n"
            f"Author's stance: {theme.author_stance}\n\n"
            f"Chronologically ordered text samples:\n{corpus_text}\n\n"
            "Analyze the evolution of this theme across these works. "
            "Respond with JSON only."
        )

        data = await self._call_anthropic(user_prompt)

        # Build evolution steps
        steps: list[EvolutionStep] = []
        for step_data in data.get("steps", []):
            publication_year = work_years.get(step_data.get("work_id"))
            if publication_year is None:
                continue
            steps.append(
                EvolutionStep(
                    work_id=step_data["work_id"],
                    publication_year=publication_year,
                    summary=step_data.get("summary", ""),
                    key_chunk_ids=step_data.get("key_chunk_ids", []),
                    self_reflection=step_data.get("self_reflection", False),
                    self_reflection_note=step_data.get("self_reflection_note"),
                )
            )

        # Build edge list
        edges: list[dict[str, str]] = []
        for edge_data in data.get("develops_from", []):
            if "from_chunk_id" in edge_data and "to_chunk_id" in edge_data:
                edges.append(
                    {
                        "from_chunk_id": edge_data["from_chunk_id"],
                        "to_chunk_id": edge_data["to_chunk_id"],
                        "relationship_note": edge_data.get("relationship_note", ""),
                    }
                )

        return ThematicEvolution(
            theme=theme.theme,
            narrative=data.get("narrative", ""),
            steps=steps,
            develops_from_edges=edges,
        )

    async def _call_anthropic(self, user_prompt: str) -> dict[str, Any]:
        """Call the Anthropic API and parse JSON response."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=EVOLUTION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise IntelligenceError(
                f"Anthropic API error during evolution analysis: {exc}",
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
                "Empty response from Anthropic API during evolution analysis",
                context={"model": self._model},
            )

        return self._parse_json_response(response_text)

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse a JSON response from an LLM, handling common malformations."""
        from author_library.intelligence.json_parser import extract_json

        try:
            return extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise IntelligenceError(
                f"Failed to parse evolution response as JSON: {exc}",
                context={"response_text": response_text[:500]},
                cause=exc,
            ) from exc
