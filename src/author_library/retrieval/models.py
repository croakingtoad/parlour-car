"""Shared data models for the retrieval engine.

Defines the core types used across vector search, full-text search,
hybrid fusion, graph expansion, orchestration, and context assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


class QuestionType(StrEnum):
    """Question type classification for multi-pass retrieval."""

    FACTUAL = "factual"
    THEMATIC = "thematic"
    GENERATIVE = "generative"
    QUOTE = "quote"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieval result with chunk content and provenance.

    Shared across vector search, full-text search, and fusion layers
    to enable uniform ranking and deduplication.
    """

    chunk_id: UUID
    work_id: str
    text: str
    score: float
    granularity: str
    source_class: str
    source: str  # e.g., "vector", "fulltext", "phrase", "graph"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphExpansionResult:
    """A retrieval result produced by graph traversal expansion.

    Carries relationship metadata from the graph edge that produced it.
    """

    chunk_id: str
    work_id: str
    text_preview: str
    granularity: str
    source_class: str
    relationship_type: str  # ENGAGES_WITH, EXPLORES_THEME, DEVELOPS_FROM
    confidence: str
    evidence: str


@dataclass(slots=True)
class ContextWindow:
    """Assembled context window ready for LLM consumption.

    Manages token budget and prioritization of retrieved content.
    """

    voice_profile_text: str
    system_prompt: str
    passages: list[ContextPassage]
    thematic_summaries: list[str]
    total_tokens_estimate: int
    token_budget: int


@dataclass(frozen=True, slots=True)
class ContextPassage:
    """A single passage included in the context window."""

    text: str
    work_id: str
    source_class: str
    relevance_score: float
    citation_label: str  # e.g., "[Work Title, Chapter 3]"
    source: str  # how it was retrieved
