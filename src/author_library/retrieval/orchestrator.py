"""Multi-pass retrieval orchestration with question classification.

Classifies user questions into types (factual, thematic, generative, quote),
then runs a three-pass retrieval pipeline:
  Pass 1: Hybrid vector + full-text search
  Pass 2: Graph expansion from Pass 1 results
  Pass 3: Supporting micro-chunk evidence
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import RetrievalError
from author_library.retrieval.fusion import reciprocal_rank_fusion
from author_library.retrieval.graph_retrieval import graph_augmented_retrieval
from author_library.retrieval.models import (
    GraphExpansionResult,
    QuestionType,
    RetrievalResult,
)
from author_library.retrieval.text_search import keyword_search, phrase_search
from author_library.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.graph.queries import GraphQueryService
    from author_library.storage.postgres import PostgresPool
    from author_library.storage.repositories import EmbeddingRepository

log = structlog.get_logger(__name__)

_CODE_FENCE_RE = re.compile(r"^```\w*\s*\n(.*?)```\s*$", re.DOTALL)

# Maximum retries for empty/malformed classification responses
_CLASSIFICATION_MAX_RETRIES = 1


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM = """\
You are a question classifier for an author-intelligence retrieval system.
Classify the user's question into exactly one of these types:

- **factual**: Questions about specific facts, dates, publications, events.
  Examples: "When did Lewis publish Mere Christianity?", "What university?"

- **thematic**: Questions about themes, ideas, recurring motifs, positions.
  Examples: "How does Lewis treat joy?", "What is his theology of imagination?"

- **generative**: Questions asking what the author would think/say about something.
  Examples: "What would Lewis say about social media?"

- **quote**: Questions seeking specific passages or quotations.
  Examples: "Find where Lewis says 'I believe in Christianity...'"

Respond with JSON only:
{"question_type": "<factual|thematic|generative|quote>", "reasoning": "<brief>"}\
"""


async def classify_question(
    question: str,
    settings: Settings,
) -> QuestionType:
    """Classify a question using the Anthropic API.

    Falls back to THEMATIC if classification fails (safest default
    for comprehensive retrieval).

    Handles common LLM response issues:
    - Empty responses (retried once, then falls back)
    - Markdown code fences around JSON
    - Malformed JSON

    Args:
        question: The user's natural-language question.
        settings: Application settings with API key and model config.

    Returns:
        Classified QuestionType.
    """
    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    if not api_key:
        log.warning("no_anthropic_key_for_classification", fallback="thematic")
        return QuestionType.THEMATIC

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)

    for attempt in range(_CLASSIFICATION_MAX_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=settings.llm.query_model,
                max_tokens=256,
                system=CLASSIFICATION_SYSTEM,
                messages=[{"role": "user", "content": question}],
            )

            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                    break

            # Handle empty response with retry
            if not response_text.strip():
                if attempt < _CLASSIFICATION_MAX_RETRIES:
                    log.warning(
                        "classification_empty_response_retrying",
                        attempt=attempt + 1,
                    )
                    continue
                log.warning(
                    "classification_empty_response",
                    fallback="thematic",
                )
                return QuestionType.THEMATIC

            # Strip code fences before parsing
            cleaned = _strip_code_fences(response_text)
            data = json.loads(cleaned)
            q_type = data.get("question_type", "thematic")
            classified = QuestionType(q_type)

            log.info(
                "question_classified",
                question_type=classified.value,
                reasoning=data.get("reasoning", ""),
            )
            return classified

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
            log.warning(
                "classification_failed",
                error=str(exc),
                attempt=attempt + 1,
                response_preview=response_text[:200] if response_text else "(empty)",
                fallback="thematic",
            )
            if attempt < _CLASSIFICATION_MAX_RETRIES:
                continue
            return QuestionType.THEMATIC

    # Should not reach here, but fall back safely
    return QuestionType.THEMATIC


# ---------------------------------------------------------------------------
# Multi-pass retrieval
# ---------------------------------------------------------------------------


class RetrievalOrchestrator:
    """Orchestrates multi-pass retrieval based on question type.

    Pass 1: Initial hybrid search (vector + full-text, fused via RRF)
    Pass 2: Graph expansion from Pass 1 results
    Pass 3: Supporting quotes from micro-chunk index
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        embedding_repo: EmbeddingRepository,
        pg_pool: PostgresPool,
        graph_query_service: GraphQueryService,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._embedding_repo = embedding_repo
        self._pg_pool = pg_pool
        self._graph = graph_query_service

    async def retrieve(
        self,
        question: str,
        *,
        author_id: str | None = None,
        source_class_filter: str | None = None,
        limit: int = 20,
        question_type: QuestionType | None = None,
    ) -> OrchestratedResult:
        """Run the full multi-pass retrieval pipeline.

        Args:
            question: The user's natural-language question.
            author_id: Optional author filter (currently unused, reserved).
            source_class_filter: Optional source class filter.
            limit: Maximum primary results to return.
            question_type: Pre-classified question type. If None,
                classification is done via the Anthropic API.

        Returns:
            OrchestratedResult with pass 1-3 results and metadata.
        """
        # Step 0: Classify question
        if question_type is None:
            question_type = await classify_question(question, self._settings)

        log.info(
            "orchestration_starting",
            question_type=question_type.value,
            question_length=len(question),
        )

        # Configure retrieval strategy based on question type
        strategy = _strategy_for_question_type(question_type)

        # Pass 1: Hybrid search
        pass1 = await self._pass1_hybrid_search(
            question,
            question_type=question_type,
            source_class_filter=source_class_filter,
            limit=strategy["pass1_limit"],
            vector_weight=strategy["vector_weight"],
            text_weight=strategy["text_weight"],
        )

        # Pass 2: Graph expansion (graceful degradation if Neo4j is unavailable)
        pass2 = await self._pass2_graph_expansion(
            pass1,
            question_type=question_type,
        )

        # Pass 3: Supporting evidence (micro chunks)
        pass3 = await self._pass3_supporting_evidence(
            question,
            pass1=pass1,
            source_class_filter=source_class_filter,
            limit=strategy["pass3_limit"],
        )

        # Cross-pass consensus: boost chunks appearing in multiple passes
        all_primary = _apply_consensus_boost(pass1, pass2, pass3)

        return OrchestratedResult(
            question=question,
            question_type=question_type,
            primary_results=all_primary[:limit],
            graph_expansions=pass2,
            supporting_evidence=pass3,
        )

    async def _pass1_hybrid_search(
        self,
        question: str,
        *,
        question_type: QuestionType,
        source_class_filter: str | None,
        limit: int,
        vector_weight: float,
        text_weight: float,
    ) -> list[RetrievalResult]:
        """Pass 1: Hybrid vector + full-text search with RRF fusion."""

        # For quote questions, prefer phrase search over keyword search
        if question_type == QuestionType.QUOTE:
            text_results = await phrase_search(
                self._pg_pool,
                question,
                source_class_filter=source_class_filter,
                limit=limit,
            )
        else:
            text_results = await keyword_search(
                self._pg_pool,
                question,
                source_class_filter=source_class_filter,
                limit=limit,
            )

        # Vector search with granularity preference based on question type
        granularity = _preferred_granularity(question_type)
        vec_results = await vector_search(
            question,
            embedding_provider=self._embedding_provider,
            embedding_repo=self._embedding_repo,
            limit=limit,
            source_class_filter=source_class_filter,
            granularity_filter=granularity,
        )

        # Fuse results via RRF
        fused = reciprocal_rank_fusion(
            vec_results,
            text_results,
            weights=[vector_weight, text_weight],
            limit=limit,
        )

        log.info(
            "pass1_complete",
            vector_count=len(vec_results),
            text_count=len(text_results),
            fused_count=len(fused),
        )
        return fused

    async def _pass2_graph_expansion(
        self,
        pass1_results: list[RetrievalResult],
        *,
        question_type: QuestionType,
    ) -> list[GraphExpansionResult]:
        """Pass 2: Graph expansion from Pass 1 results."""
        if not pass1_results:
            return []

        # Extract themes from pass1 metadata for thematic/generative questions
        theme_names: list[str] | None = None
        if question_type in (QuestionType.THEMATIC, QuestionType.GENERATIVE):
            theme_names = _extract_theme_hints(pass1_results)

        try:
            expansions = await graph_augmented_retrieval(
                self._graph,
                pass1_results,
                theme_names=theme_names,
                max_engagement_depth=3,
            )
        except Exception as exc:
            # Graceful degradation: if Neo4j is down or graph expansion fails,
            # proceed with vector-only retrieval
            log.warning(
                "pass2_graph_expansion_failed",
                error=str(exc),
                degraded=True,
            )
            expansions = []

        log.info("pass2_complete", expanded=len(expansions))
        return expansions

    async def _pass3_supporting_evidence(
        self,
        question: str,
        *,
        pass1: list[RetrievalResult],
        source_class_filter: str | None,
        limit: int,
    ) -> list[RetrievalResult]:
        """Pass 3: Retrieve supporting micro-chunk quotes as evidence."""
        if not pass1:
            return []

        # Search micro chunks for supporting quotes
        try:
            micro_results = await vector_search(
                question,
                embedding_provider=self._embedding_provider,
                embedding_repo=self._embedding_repo,
                limit=limit,
                source_class_filter=source_class_filter,
                granularity_filter="micro",
            )
        except RetrievalError:
            log.warning("pass3_micro_search_failed")
            micro_results = []

        # Exclude chunks already in pass1
        pass1_ids = {str(r.chunk_id) for r in pass1}
        evidence = [
            r for r in micro_results if str(r.chunk_id) not in pass1_ids
        ]

        log.info("pass3_complete", evidence_count=len(evidence))
        return evidence


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class OrchestratedResult:
    """Container for the full multi-pass retrieval result."""

    __slots__ = (
        "graph_expansions",
        "primary_results",
        "question",
        "question_type",
        "supporting_evidence",
    )

    def __init__(
        self,
        *,
        question: str,
        question_type: QuestionType,
        primary_results: list[RetrievalResult],
        graph_expansions: list[GraphExpansionResult],
        supporting_evidence: list[RetrievalResult],
    ) -> None:
        self.question = question
        self.question_type = question_type
        self.primary_results = primary_results
        self.graph_expansions = graph_expansions
        self.supporting_evidence = supporting_evidence

    @property
    def total_chunks(self) -> int:
        return (
            len(self.primary_results)
            + len(self.graph_expansions)
            + len(self.supporting_evidence)
        )


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def _strategy_for_question_type(q_type: QuestionType) -> dict[str, Any]:
    """Return retrieval strategy parameters for a question type."""
    strategies: dict[QuestionType, dict[str, Any]] = {
        QuestionType.FACTUAL: {
            "pass1_limit": 15,
            "pass3_limit": 5,
            "vector_weight": 0.6,
            "text_weight": 0.4,
        },
        QuestionType.THEMATIC: {
            "pass1_limit": 20,
            "pass3_limit": 10,
            "vector_weight": 0.7,
            "text_weight": 0.3,
        },
        QuestionType.GENERATIVE: {
            "pass1_limit": 20,
            "pass3_limit": 10,
            "vector_weight": 0.8,
            "text_weight": 0.2,
        },
        QuestionType.QUOTE: {
            "pass1_limit": 15,
            "pass3_limit": 10,
            "vector_weight": 0.3,
            "text_weight": 0.7,
        },
    }
    return strategies[q_type]


def _preferred_granularity(q_type: QuestionType) -> str | None:
    """Return preferred chunk granularity, or None for any."""
    granularity_map: dict[QuestionType, str | None] = {
        QuestionType.FACTUAL: "micro",
        QuestionType.THEMATIC: "macro",
        QuestionType.GENERATIVE: "meso",
        QuestionType.QUOTE: "micro",
    }
    return granularity_map.get(q_type)


def _extract_theme_hints(results: list[RetrievalResult]) -> list[str]:
    """Extract potential theme names from retrieval result metadata."""
    themes: list[str] = []
    for r in results:
        if "theme_names" in r.metadata:
            raw = r.metadata["theme_names"]
            if isinstance(raw, list):
                themes.extend(str(t) for t in raw)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in themes:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _apply_consensus_boost(
    pass1: list[RetrievalResult],
    pass2: list[GraphExpansionResult],
    pass3: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Boost scores for chunks appearing in multiple passes.

    Chunks that appear in both Pass 1 and Pass 3 get a 1.5x boost.
    Chunks that appear in Pass 1 and have graph expansion connections
    get a 1.3x boost.
    """
    pass3_ids = {str(r.chunk_id) for r in pass3}
    graph_chunk_ids = {r.chunk_id for r in pass2}

    boosted: list[RetrievalResult] = []
    for r in pass1:
        cid = str(r.chunk_id)
        boost = 1.0
        if cid in pass3_ids:
            boost *= 1.5
        if cid in graph_chunk_ids:
            boost *= 1.3

        if boost > 1.0:
            boosted.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    work_id=r.work_id,
                    text=r.text,
                    score=r.score * boost,
                    granularity=r.granularity,
                    source_class=r.source_class,
                    source=r.source,
                    metadata={**r.metadata, "consensus_boost": boost},
                )
            )
        else:
            boosted.append(r)

    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted
