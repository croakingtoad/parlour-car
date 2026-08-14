"""Shared fixtures for graph module tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.config import APIKeySettings, DatabaseSettings, LLMSettings
from author_library.storage.neo4j import Neo4jConnection

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_neo4j = pytest.mark.skipif(
    os.environ.get("SKIP_NEO4J", "0") == "1",
    reason="Neo4j not available (SKIP_NEO4J=1)",
)

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def neo4j_conn(
    assert_graph_is_disposable: None,
) -> AsyncGenerator[Neo4jConnection]:
    """Provide a connected Neo4j instance with clean schema.

    IMPORTANT: Only cleans up test-created nodes (work_id starting with
    'test--' or known test work_ids). The previous implementation used
    MATCH (n) DETACH DELETE n which destroyed ALL production graph data.
    """
    _TEST_WORK_IDS = [
        "guite--faith-hope-poetry",
        "coleridge--biographia-literaria",
        "coleridge--statesmans-manual",
        "ward--romantic-theology",
        "w1", "w2",  # Used by theme_dedup integration tests
    ]
    # Also clean up simple chunk IDs used by integration tests
    _TEST_CHUNK_IDS = ["c1", "c2", "c3", "c4", "c5"]

    settings = DatabaseSettings()
    conn = Neo4jConnection(settings)
    await conn.connect()
    await conn.init_schema()

    async def _scoped_cleanup(c: Neo4jConnection) -> None:
        """Remove only test-created nodes, preserving production data."""
        for wid in _TEST_WORK_IDS:
            await c.execute_write(
                "MATCH (ch:Chunk {work_id: $wid}) DETACH DELETE ch", {"wid": wid}
            )
            await c.execute_write(
                "MATCH (w:Work {work_id: $wid}) DETACH DELETE w", {"wid": wid}
            )
        for cid in _TEST_CHUNK_IDS:
            await c.execute_write(
                "MATCH (ch:Chunk {chunk_id: $cid}) DETACH DELETE ch", {"cid": cid}
            )
        await c.execute_write(
            "MATCH (ch:Chunk) WHERE ch.work_id STARTS WITH 'test--' DETACH DELETE ch"
        )
        await c.execute_write(
            "MATCH (w:Work) WHERE w.work_id STARTS WITH 'test--' DETACH DELETE w"
        )
        # Clean test-created entity nodes BY NAME PREFIX, never by orphan
        # heuristic. Production vocabulary nodes can legitimately have zero
        # edges (e.g. restored themes awaiting extraction backfill) — an
        # orphan sweep deletes them (td-aef7c5, 2026-07-02 incident).
        for label in ("Theme", "Person", "Concept", "Argument"):
            await c.execute_write(
                f"MATCH (n:{label}) WHERE n.canonical_name STARTS WITH 'test--' "
                "DETACH DELETE n"
            )
        # Legacy unprefixed entity names from older tests
        _LEGACY_TEST_ENTITY_NAMES = [
            "canon", "dup", "imagination-theology", "imagination-divine",
            "poetry-form", "primary-imagination", "imagination",
        ]
        for label in ("Theme", "Concept"):
            await c.execute_write(
                f"MATCH (n:{label}) WHERE n.canonical_name IN $names "
                "AND NOT (n)--() DELETE n",
                {"names": _LEGACY_TEST_ENTITY_NAMES},
            )

    await _scoped_cleanup(conn)
    yield conn
    await _scoped_cleanup(conn)
    await conn.close()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_keys() -> APIKeySettings:
    """Provide API key settings from environment."""
    return APIKeySettings()


@pytest.fixture
def llm_settings() -> LLMSettings:
    """Provide LLM settings."""
    return LLMSettings()


# ---------------------------------------------------------------------------
# Chunk fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def primary_chunks() -> list[Chunk]:
    """Provide sample primary source chunks for testing."""
    return [
        Chunk(
            id="primary-chunk-001",
            text=(
                "The Primary Imagination I hold to be the living Power and prime Agent "
                "of all human Perception, and as a repetition in the finite mind of the "
                "eternal act of creation in the infinite I AM. The secondary Imagination "
                "I consider as an echo of the former, co-existing with the conscious will."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            chapter="Chapter 3: Coleridge and the Power of Imagination",
            section="The Two Imaginations",
            position=5,
            metadata={"genre": "scholarly-prose"},
        ),
        Chunk(
            id="primary-chunk-002",
            text=(
                "Poetry is the spontaneous overflow of powerful feelings: it takes its "
                "origin from emotion recollected in tranquillity. See Coleridge, "
                "Biographia Literaria, Chapter XIII for the distinction between fancy "
                "and imagination that undergirds this entire discussion."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            chapter="Chapter 3: Coleridge and the Power of Imagination",
            section="Fancy vs Imagination",
            position=6,
            metadata={"genre": "scholarly-prose"},
        ),
        Chunk(
            id="primary-chunk-003",
            text=(
                "The sacramental vision that Coleridge articulates finds its deepest "
                "expression in the symbol, which participates in the reality it renders "
                "intelligible (Coleridge, 1816). This esemplastic power — the ability "
                "to shape disparate elements into living unity — is what distinguishes "
                "the poetic act from mere mechanical aggregation."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            chapter="Chapter 4: Symbol and Sacrament",
            section="The Esemplastic Power",
            position=10,
            metadata={"genre": "scholarly-prose"},
        ),
    ]


@pytest.fixture
def contextual_chunks() -> list[Chunk]:
    """Provide sample contextual source chunks for testing."""
    return [
        Chunk(
            id="ctx-chunk-001",
            text=(
                "The IMAGINATION then I consider either as primary, or secondary. "
                "The primary IMAGINATION I hold to be the living Power and prime Agent "
                "of all human Perception, and as a repetition in the finite mind of the "
                "eternal act of creation in the infinite I AM."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="coleridge--biographia-literaria",
            source_class="contextual",
            chapter="Chapter XIII",
            section="On the Imagination",
            position=1,
            metadata={"genre": "scholarly-prose"},
        ),
        Chunk(
            id="ctx-chunk-002",
            text=(
                "The secondary Imagination I consider as an echo of the former, "
                "co-existing with the conscious will, yet still as identical with "
                "the primary in the kind of its agency, and differing only in degree, "
                "and in the mode of its operation."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="coleridge--biographia-literaria",
            source_class="contextual",
            chapter="Chapter XIII",
            section="On the Imagination",
            position=2,
            metadata={"genre": "scholarly-prose"},
        ),
        Chunk(
            id="ctx-chunk-003",
            text=(
                "A symbol is characterized by the translucence of the Special in "
                "the Individual or of the General in the Especial or of the Universal "
                "in the General. Above all by the translucence of the Eternal through "
                "and in the Temporal. It always partakes of the Reality which it renders "
                "intelligible; and while it enunciates the whole, abides itself as a "
                "living part in that Unity, of which it is the representative."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="coleridge--statesmans-manual",
            source_class="contextual",
            chapter="Appendix B",
            section="On Symbols",
            position=1,
            metadata={"genre": "scholarly-prose"},
        ),
    ]


@pytest.fixture
def secondary_chunks() -> list[Chunk]:
    """Provide sample secondary source chunks for testing."""
    return [
        Chunk(
            id="sec-chunk-001",
            text=(
                "Guite's reading of Coleridge represents a significant departure from "
                "the standard critical tradition. Where most scholars emphasize the "
                "philosophical debt to Schelling, Guite foregrounds the theological "
                "dimensions of the Primary Imagination, arguing that Coleridge's concept "
                "is fundamentally sacramental rather than idealist."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="ward--romantic-theology",
            source_class="secondary",
            chapter="Chapter 7",
            section="Guite's Coleridge",
            position=3,
            metadata={"genre": "scholarly-prose"},
        ),
    ]


@pytest.fixture
def works_metadata() -> dict[str, dict[str, str]]:
    """Provide metadata for works used in tests."""
    return {
        "coleridge--biographia-literaria": {
            "work_id": "coleridge--biographia-literaria",
            "title": "Biographia Literaria",
            "author": "Test Coleridge",
            "source_class": "contextual",
        },
        "coleridge--statesmans-manual": {
            "work_id": "coleridge--statesmans-manual",
            "title": "The Statesman's Manual",
            "author": "Test Coleridge",
            "source_class": "contextual",
        },
        "guite--faith-hope-poetry": {
            "work_id": "guite--faith-hope-poetry",
            "title": "Faith Hope and Poetry",
            "author": "Test Guite",
            "source_class": "primary",
        },
        "ward--romantic-theology": {
            "work_id": "ward--romantic-theology",
            "title": "Romantic Theology",
            "author": "Michael Ward",
            "source_class": "secondary",
        },
    }
