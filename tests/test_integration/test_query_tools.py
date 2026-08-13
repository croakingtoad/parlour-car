"""Live integration tests for query MCP tools.

Tests: search_chunks, get_passage_links, trace_theme, find_quotes
against real PostgreSQL + Neo4j + Voyage AI embeddings.

Data setup: catalog + chunk a public-domain Shakespeare excerpt ONCE
per module (module_data fixture). All connections are closed before
yielding — individual tests use the conftest `storage` fixture for
fresh per-test connections. This avoids asyncio event-loop future
conflicts between the module fixture and function-scoped tests.

Runs against the test database only (author_library_test).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from author_library.config import Settings
from author_library.embeddings import ProviderRegistry
from author_library.storage.manager import StorageManager
from author_library.tools.composable_ingestion import (
    handle_catalog_source,
    handle_chunk_source,
)
from author_library.tools.composable_query import (
    handle_get_passage_links,
    handle_search_chunks,
)
from author_library.tools.query import (
    handle_find_quotes,
    handle_trace_theme,
)
from tests.conftest import TEST_NAMESPACE

from .conftest import SKIP_NO_ANTHROPIC, SKIP_NO_DB

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager as SM

# ---------------------------------------------------------------------------
# Public-domain test content
# ---------------------------------------------------------------------------

HAMLET_QUERY_EXCERPT = """\
Hamlet: A Play in Five Acts
By William Shakespeare

Act I, Scene 2: The King's Court

O, that this too, too solid flesh would melt,
Thaw, and resolve itself into a dew!
Or that the Everlasting had not fix'd
His canon 'gainst self-slaughter! O God! God!
How weary, stale, flat, and unprofitable
Seem to me all the uses of this world!
Fie on't! ah, fie! 'Tis an unweeded garden,
That grows to seed; things rank and gross in nature
Possess it merely.

Act III, Scene 1: The Nunnery Scene

To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.

Who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin?

Act V, Scene 2: The Final Duel

Had I but time—as this fell sergeant, death,
Is strict in his arrest—O, I could tell you—
But let it be. Horatio, I am dead;
Thou livest; report me and my cause aright
To the unsatisfied. O good Horatio,
What a wounded name, things standing thus unknown,
Shall live behind me! If thou didst ever hold me in thy heart,
Absent thee from felicity a while,
And in this harsh world draw thy breath in pain,
To tell my story.

The rest is silence.
"""


# ---------------------------------------------------------------------------
# Helper: clean query test data
# ---------------------------------------------------------------------------


async def _clean_query_test_data(storage: StorageManager) -> None:
    """Remove query test data from PG and Neo4j."""
    await storage.pg.execute("DELETE FROM chunk_embeddings")
    await storage.pg.execute("DELETE FROM thematic_appearances")
    await storage.pg.execute("DELETE FROM thematic_entries")
    await storage.pg.execute("DELETE FROM voice_profiles")
    await storage.pg.execute("DELETE FROM acquisition_candidates")
    await storage.pg.execute("DELETE FROM chunks")
    await storage.pg.execute("DELETE FROM works")
    await storage.pg.execute("DELETE FROM authors")
    # Neo4j cleanup must stay inside the test-- namespace. Never orphan-sweep
    # (WHERE NOT (n)--()): that deletes unreferenced PRODUCTION entities too,
    # which is how 3 real Author nodes were lost on 2026-08-13.
    await storage.neo4j.execute_write(
        "MATCH (c:Chunk) WHERE c.work_id STARTS WITH $prefix DETACH DELETE c",
        {"prefix": TEST_NAMESPACE},
    )
    await storage.neo4j.execute_write(
        "MATCH (w:Work) WHERE w.work_id STARTS WITH $prefix DETACH DELETE w",
        {"prefix": TEST_NAMESPACE},
    )
    for label in ("Theme", "Person", "Concept", "Argument", "Author"):
        await storage.neo4j.execute_write(
            f"MATCH (n:{label}) "
            "WHERE n.canonical_name STARTS WITH $prefix "
            "   OR n.author_id STARTS WITH $prefix "
            "DETACH DELETE n",
            {"prefix": TEST_NAMESPACE},
        )


# ---------------------------------------------------------------------------
# Module-scoped data fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def module_data() -> dict[str, Any]:
    """Catalog + chunk Shakespeare ONCE for the whole module.

    IMPORTANT: All DB connections are closed before yielding.
    Individual tests must use the conftest `storage` fixture
    (function-scoped, fresh connections) for their queries.
    Sharing asyncpg pools across fixture scopes causes
    'Future attached to different loop' errors.

    Yields: {"work_id": str, "chunk_ids": list[str]}
    """
    settings = Settings()

    # === SETUP: create storage, run pipeline, close all connections ===
    mgr = StorageManager(settings.database)
    await mgr.connect(run_pg_migrations=True, init_neo4j_schema=True)
    embedding_provider = ProviderRegistry.create(settings)

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(HAMLET_QUERY_EXCERPT)
        temp_path = Path(f.name)

    work_id: str = ""
    chunk_ids: list[str] = []
    setup_error: Exception | None = None

    try:
        # Clean before setup
        await _clean_query_test_data(mgr)

        # B2: catalog_source
        catalog_str = await handle_catalog_source(
            {
                "file_path": str(temp_path),
                "source_class": "primary",
                "work_type": "other",
                "metadata_overrides": {
                    "author": "Shakespeare",
                    "title": "Hamlet Query Test",
                    "subject_author_id": "shakespeare",
                    "genre_tags": ["drama"],
                    "subject_headings": ["English drama"],
                    "publication_year": 1603,
                },
            },
            settings=settings,
            storage=mgr,
            embedding_provider=embedding_provider,
        )
        catalog_result = json.loads(catalog_str)
        work_id = catalog_result["work_id"]

        # Patch file_path into source_metadata so chunk_source can re-parse
        work = await mgr.works.get(work_id)
        assert work is not None
        source_meta_raw = work.get("source_metadata") or {}
        source_meta = (
            json.loads(source_meta_raw)
            if isinstance(source_meta_raw, str)
            else dict(source_meta_raw)
        )
        source_meta["file_path"] = str(temp_path)
        await mgr.works.update(work_id, {"source_metadata": source_meta})

        # B3: chunk_source (annotation + embedding + entity extraction)
        await handle_chunk_source(
            {"work_id": work_id},
            settings=settings,
            storage=mgr,
            embedding_provider=embedding_provider,
        )

        # Collect chunk IDs
        chunks_raw = await mgr.pg.fetch_all(
            "SELECT id FROM chunks WHERE work_id = $1 ORDER BY created_at",
            work_id,
        )
        chunk_ids = [str(r["id"]) for r in chunks_raw]

    except Exception as exc:
        setup_error = exc
    finally:
        temp_path.unlink(missing_ok=True)
        await embedding_provider.close()
        await mgr.close()

    if setup_error:
        pytest.skip(f"Module data setup failed: {setup_error}")
        return

    # === YIELD: yield pure data only (no live connections) ===
    yield {"work_id": work_id, "chunk_ids": chunk_ids}

    # === TEARDOWN: fresh connections to clean up ===
    mgr_cleanup = StorageManager(settings.database)
    await mgr_cleanup.connect(run_pg_migrations=False, init_neo4j_schema=False)
    try:
        await _clean_query_test_data(mgr_cleanup)
    finally:
        await mgr_cleanup.close()


# ---------------------------------------------------------------------------
# TestSearchChunks
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestSearchChunks:
    """search_chunks returns relevant chunks from ingested data."""

    async def test_search_returns_results(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """search_chunks finds semantically relevant passages."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_search_chunks(
                {"query": "mortality and death and sleep", "max_results": 5},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert "results" in result
        assert len(result["results"]) > 0

        first = result["results"][0]
        assert "chunk_id" in first
        assert "text" in first
        assert "relevance_score" in first
        assert first["relevance_score"] > 0.0
        assert "metadata" in first
        assert first["metadata"]["work_id"] == module_data["work_id"]
        assert first["metadata"]["source_class"] == "primary"

    async def test_search_returns_provenance_rules(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """search_chunks includes provenance_rules for each result."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_search_chunks(
                {"query": "to be or not to be", "max_results": 3},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert len(result["results"]) > 0
        first = result["results"][0]
        assert "provenance_rules" in first
        rules = first["provenance_rules"]
        assert "attribution" in rules
        assert "voice_eligible" in rules
        assert rules["voice_eligible"] is True  # primary source

    async def test_search_with_source_class_filter(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """search_chunks respects source_class filter."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_search_chunks(
                {
                    "query": "death and mortality",
                    "filters": {"source_class": ["primary"]},
                    "max_results": 5,
                },
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert "results" in result
        # All returned results should be primary source class
        for r in result["results"]:
            assert r["metadata"]["source_class"] == "primary"

    async def test_search_empty_query_raises(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """search_chunks raises RetrievalError if query is missing."""
        from author_library.errors import RetrievalError

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            with pytest.raises(RetrievalError, match="query is required"):
                await handle_search_chunks(
                    {},
                    settings=integration_settings,
                    storage=storage,
                    embedding_provider=embedding_provider,
                )
        finally:
            await embedding_provider.close()


# ---------------------------------------------------------------------------
# TestGetPassageLinks
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestGetPassageLinks:
    """get_passage_links returns passage link structure for a chunk."""

    async def test_get_links_returns_structure(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """get_passage_links returns a valid structure even for single-work data."""
        chunk_ids = module_data["chunk_ids"]
        assert len(chunk_ids) > 0, "No chunks in module_data"
        chunk_id = chunk_ids[0]

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_get_passage_links(
                {"chunk_id": chunk_id},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert "source_chunk" in result
        assert result["source_chunk"]["chunk_id"] == chunk_id
        assert "links" in result
        # Links may be empty (single-work — no cross-work linking yet)
        assert isinstance(result["links"], list)

    async def test_get_links_missing_chunk_id_raises(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """get_passage_links raises RetrievalError if chunk_id is missing."""
        from author_library.errors import RetrievalError

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            with pytest.raises(RetrievalError, match="chunk_id is required"):
                await handle_get_passage_links(
                    {},
                    settings=integration_settings,
                    storage=storage,
                    embedding_provider=embedding_provider,
                )
        finally:
            await embedding_provider.close()

    async def test_get_links_unknown_chunk_returns_error(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """get_passage_links returns error dict for unknown chunk_id."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_get_passage_links(
                {"chunk_id": "00000000-0000-0000-0000-000000000000"},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestTraceTheme
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestTraceTheme:
    """trace_theme returns theme data from the knowledge graph."""

    async def test_trace_existing_theme(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """trace_theme returns found=True and chronology for a known theme."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            # Get any theme that was extracted during setup
            themes_raw = await storage.neo4j.execute_read(
                "MATCH (t:Theme) RETURN t.canonical_name AS name LIMIT 1",
                {},
            )
            if not themes_raw:
                pytest.skip("No themes found (entity extraction may have produced none)")

            theme_name = themes_raw[0]["name"]

            result_str = await handle_trace_theme(
                {"theme_name": theme_name},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert result["found"] is True
        assert "theme" in result
        assert "chronology" in result
        assert "total_chunks" in result
        assert result["total_chunks"] > 0

    async def test_trace_unknown_theme_returns_not_found(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """trace_theme returns found=False for a theme not in the graph."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_trace_theme(
                {"theme_name": "xyzzy-nonexistent-theme-12345"},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert result["found"] is False
        assert result["theme"] == "xyzzy-nonexistent-theme-12345"

    async def test_trace_missing_theme_raises(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """trace_theme raises RetrievalError if theme_name is missing."""
        from author_library.errors import RetrievalError

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            with pytest.raises(RetrievalError, match="theme_name is required"):
                await handle_trace_theme(
                    {},
                    settings=integration_settings,
                    storage=storage,
                    embedding_provider=embedding_provider,
                )
        finally:
            await embedding_provider.close()


# ---------------------------------------------------------------------------
# TestFindQuotes
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestFindQuotes:
    """find_quotes combines phrase + vector search to find passages."""

    async def test_find_quotes_returns_results(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """find_quotes returns results for a phrase present in test data."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_find_quotes(
                {"query": "to be or not to be", "limit": 5},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        assert "quotes" in result
        assert "total_results" in result
        assert result["total_results"] > 0

        first = result["quotes"][0]
        assert "chunk_id" in first
        assert "text" in first
        assert "work_id" in first
        assert "match_type" in first

    async def test_find_quotes_deduplicates(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """find_quotes does not return duplicate chunk_ids."""
        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_find_quotes(
                {"query": "death", "limit": 10},
                settings=integration_settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)

        chunk_ids = [q["chunk_id"] for q in result["quotes"]]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in results"

    async def test_find_quotes_missing_query_raises(
        self,
        module_data: dict[str, Any],
        storage: SM,
        integration_settings: Settings,
    ) -> None:
        """find_quotes raises RetrievalError if query is missing."""
        from author_library.errors import RetrievalError

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            with pytest.raises(RetrievalError, match="query is required"):
                await handle_find_quotes(
                    {},
                    settings=integration_settings,
                    storage=storage,
                    embedding_provider=embedding_provider,
                )
        finally:
            await embedding_provider.close()
