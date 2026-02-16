"""Retrieval engine for The Author Library.

Provides multi-pass retrieval combining vector similarity search,
full-text search, hybrid fusion, graph-augmented expansion,
orchestration, and context assembly with voice calibration.

Modules:
    models          — Shared data models (RetrievalResult, QuestionType, etc.)
    vector_search   — pgvector HNSW cosine similarity search
    text_search     — Full-text / BM25 keyword and phrase search
    fusion          — Reciprocal Rank Fusion (RRF)
    graph_retrieval — Graph-augmented context expansion via Neo4j
    orchestrator    — Multi-pass retrieval orchestration
    context_assembly — Context window assembly with voice calibration
"""

from author_library.retrieval.context_assembly import (
    assemble_context,
    build_voice_system_prompt,
)
from author_library.retrieval.fusion import reciprocal_rank_fusion
from author_library.retrieval.graph_retrieval import graph_augmented_retrieval
from author_library.retrieval.models import (
    ContextPassage,
    ContextWindow,
    GraphExpansionResult,
    QuestionType,
    RetrievalResult,
)
from author_library.retrieval.orchestrator import (
    OrchestratedResult,
    RetrievalOrchestrator,
    classify_question,
)
from author_library.retrieval.text_search import keyword_search, phrase_search
from author_library.retrieval.vector_search import vector_search

__all__ = [
    "ContextPassage",
    "ContextWindow",
    "GraphExpansionResult",
    "OrchestratedResult",
    "QuestionType",
    "RetrievalOrchestrator",
    "RetrievalResult",
    "assemble_context",
    "build_voice_system_prompt",
    "classify_question",
    "graph_augmented_retrieval",
    "keyword_search",
    "phrase_search",
    "reciprocal_rank_fusion",
    "vector_search",
]
