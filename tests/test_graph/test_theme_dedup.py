"""Tests for post-extraction theme deduplication.

Tests the theme dedup algorithm at multiple levels:
  - Unit: cosine similarity, greedy clustering logic
  - Integration: Neo4j merge operations (requires running Neo4j)
  - End-to-end: full deduplicate_themes pipeline (requires Neo4j + embeddings)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from author_library.graph.theme_dedup import (
    ThemeDedupResult,
    _cosine_similarity,
    _merge_theme_into_canonical,
    deduplicate_themes,
)

from .conftest import requires_neo4j

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection


# ---------------------------------------------------------------------------
# Unit tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Test the cosine similarity helper function."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """Zero vector should produce 0.0 (no division error)."""
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self) -> None:
        """Similar vectors should have high but not perfect similarity."""
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.0]
        sim = _cosine_similarity(a, b)
        assert sim > 0.99
        assert sim < 1.0

    def test_high_dimensional(self) -> None:
        """Should work with high-dimensional vectors (like real embeddings)."""
        dim = 1024
        a = [1.0] * dim
        b = [1.0] * dim
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unit tests: ThemeDedupResult
# ---------------------------------------------------------------------------


class TestThemeDedupResult:
    """Test the result dataclass."""

    def test_default_values(self) -> None:
        """Default result should have zero counts."""
        result = ThemeDedupResult()
        assert result.original_count == 0
        assert result.canonical_count == 0
        assert result.merged_count == 0
        assert result.errors == []

    def test_accumulation(self) -> None:
        """Result fields should be settable."""
        result = ThemeDedupResult(
            original_count=100,
            canonical_count=25,
            merged_count=75,
            clusters_formed=25,
        )
        assert result.original_count == 100
        assert result.merged_count == 75


# ---------------------------------------------------------------------------
# Unit tests: deduplicate_themes with mocked Neo4j and embeddings
# ---------------------------------------------------------------------------


class TestDeduplicateThemesUnit:
    """Test the dedup algorithm with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero(self) -> None:
        """No themes in graph should return immediately with zero counts."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[])
        embedding = AsyncMock()

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 0
        assert result.merged_count == 0
        assert result.errors == []
        # No embedding calls should have been made
        embedding.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_theme_no_merges(self) -> None:
        """A single theme should produce 1 cluster and 0 merges."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "primary-imagination", "name": "Primary Imagination", "rel_count": 5},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.0, 0.0]],
            model="test",
            provider="test",
            dimensions=3,
        ))

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 1
        assert result.canonical_count == 1
        assert result.merged_count == 0

    @pytest.mark.asyncio
    async def test_distinct_themes_no_merges(self) -> None:
        """Themes with low similarity should not be merged."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "imagination", "name": "Imagination", "rel_count": 10},
            {"canonical_name": "biography", "name": "Biography", "rel_count": 5},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Orthogonal vectors = totally different themes
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            model="test",
            provider="test",
            dimensions=3,
        ))

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 2
        assert result.canonical_count == 2
        assert result.merged_count == 0
        # No merge calls since themes are distinct
        neo4j.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_similar_themes_merged(self) -> None:
        """Themes with high similarity should be merged into the most-connected."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "imagination-and-theology", "name": "Imagination and Theology", "rel_count": 10},
            {"canonical_name": "imagination-and-the-divine", "name": "Imagination and the Divine", "rel_count": 3},
            {"canonical_name": "imagination-as-divine-faculty", "name": "Imagination as Divine Faculty", "rel_count": 1},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # All very similar vectors (cosine sim > 0.85)
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[
                [1.0, 0.1, 0.0],   # imagination-and-theology (canonical)
                [1.0, 0.12, 0.0],  # imagination-and-the-divine (duplicate)
                [1.0, 0.08, 0.0],  # imagination-as-divine-faculty (duplicate)
            ],
            model="test",
            provider="test",
            dimensions=3,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 3
        assert result.canonical_count == 1  # All in one cluster
        assert result.merged_count == 2  # Two duplicates merged
        # execute_write should be called for each merge:
        # 2 merges * 2 calls each (move edges + delete node) = 4
        assert neo4j.execute_write.call_count == 4

    @pytest.mark.asyncio
    async def test_mixed_clusters(self) -> None:
        """Some themes should cluster while others stay separate."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "imagination-theology", "name": "Imagination Theology", "rel_count": 10},
            {"canonical_name": "imagination-divine", "name": "Imagination Divine", "rel_count": 3},
            {"canonical_name": "coleridge-biography", "name": "Coleridge Biography", "rel_count": 8},
            {"canonical_name": "coleridge-life-story", "name": "Coleridge Life Story", "rel_count": 2},
            {"canonical_name": "poetry-form", "name": "Poetry Form", "rel_count": 6},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Cluster 1: imagination (indices 0, 1) - similar
        # Cluster 2: coleridge bio (indices 2, 3) - similar
        # Cluster 3: poetry form (index 4) - distinct
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[
                [1.0, 0.0, 0.0],    # imagination-theology
                [0.99, 0.01, 0.0],  # imagination-divine (similar to 0)
                [0.0, 1.0, 0.0],    # coleridge-biography
                [0.0, 0.99, 0.01],  # coleridge-life-story (similar to 2)
                [0.0, 0.0, 1.0],    # poetry-form (distinct)
            ],
            model="test",
            provider="test",
            dimensions=3,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 5
        assert result.canonical_count == 3  # 3 clusters
        assert result.merged_count == 2  # 2 duplicates merged

    @pytest.mark.asyncio
    async def test_most_connected_becomes_canonical(self) -> None:
        """The theme with the most relationships should be the canonical one."""
        neo4j = AsyncMock()
        # Note: records sorted by rel_count DESC
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "theme-popular", "name": "Theme Popular", "rel_count": 50},
            {"canonical_name": "theme-rare", "name": "Theme Rare", "rel_count": 1},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[
                [1.0, 0.0],
                [0.99, 0.01],  # Very similar
            ],
            model="test",
            provider="test",
            dimensions=2,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding)

        assert result.merged_count == 1

        # Check that the merge was canonical=theme-popular, dup=theme-rare
        # The first write call should move edges from theme-rare to theme-popular
        first_call_args = neo4j.execute_write.call_args_list[0]
        params = first_call_args[0][1]  # Second positional arg is params
        assert params["canon_name"] == "theme-popular"
        assert params["dup_name"] == "theme-rare"

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_error(self) -> None:
        """Embedding failure should return result with error, not raise."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "theme-a", "name": "Theme A", "rel_count": 5},
        ])

        embedding = AsyncMock()
        embedding.embed_batch = AsyncMock(side_effect=Exception("API rate limit"))

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == 1
        assert result.merged_count == 0
        assert len(result.errors) == 1
        assert "API rate limit" in result.errors[0]

    @pytest.mark.asyncio
    async def test_idempotent_no_themes_after_first_run(self) -> None:
        """Running dedup when there are no duplicates should be a no-op."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "a", "name": "A", "rel_count": 5},
            {"canonical_name": "b", "name": "B", "rel_count": 3},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Orthogonal = no duplicates
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            model="test",
            provider="test",
            dimensions=2,
        ))

        result1 = await deduplicate_themes(neo4j, embedding)
        result2 = await deduplicate_themes(neo4j, embedding)

        assert result1.merged_count == 0
        assert result2.merged_count == 0

    @pytest.mark.asyncio
    async def test_custom_similarity_threshold(self) -> None:
        """Custom threshold should control merge sensitivity."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "theme-a", "name": "Theme A", "rel_count": 10},
            {"canonical_name": "theme-b", "name": "Theme B", "rel_count": 5},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Similarity of [1.0, 0.3] and [1.0, 0.0] is about 0.958
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.3], [1.0, 0.0]],
            model="test",
            provider="test",
            dimensions=2,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        # With high threshold (0.97), should NOT merge
        result_strict = await deduplicate_themes(
            neo4j, embedding, similarity_threshold=0.97
        )
        assert result_strict.merged_count == 0

        # With lower threshold (0.90), should merge
        result_loose = await deduplicate_themes(
            neo4j, embedding, similarity_threshold=0.90
        )
        assert result_loose.merged_count == 1

    @pytest.mark.asyncio
    async def test_large_cluster_all_merged_into_one(self) -> None:
        """A large cluster of similar themes should all merge into the canonical."""
        n = 20
        neo4j = AsyncMock()
        theme_records = [
            {
                "canonical_name": f"imagination-variant-{i}",
                "name": f"Imagination Variant {i}",
                "rel_count": n - i,  # First one has most connections
            }
            for i in range(n)
        ]
        neo4j.execute_read = AsyncMock(return_value=theme_records)

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # All nearly identical vectors (slight perturbation)
        base = [1.0] + [0.0] * 9
        vectors = []
        for i in range(n):
            v = list(base)
            v[1] = i * 0.001  # Tiny perturbation
            vectors.append(v)

        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=vectors,
            model="test",
            provider="test",
            dimensions=10,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding)

        assert result.original_count == n
        assert result.canonical_count == 1
        assert result.merged_count == n - 1  # All but canonical merged


# ---------------------------------------------------------------------------
# Unit tests: work-scoped deduplication
# ---------------------------------------------------------------------------


class TestDeduplicateThemesScopedUnit:
    """Test work-scoped dedup with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_scoped_skips_unrelated_clusters(self) -> None:
        """Clusters without any of the work's themes should be skipped."""
        neo4j = AsyncMock()
        # First call: fetch all themes globally
        # Second call: fetch work-scoped themes
        neo4j.execute_read = AsyncMock(side_effect=[
            # Global themes
            [
                {"canonical_name": "imagination-theology", "name": "Imagination Theology", "rel_count": 10},
                {"canonical_name": "imagination-divine", "name": "Imagination Divine", "rel_count": 3},
                {"canonical_name": "biography", "name": "Biography", "rel_count": 8},
                {"canonical_name": "life-story", "name": "Life Story", "rel_count": 2},
            ],
            # Work themes — only biography cluster
            [
                {"canonical_name": "biography"},
                {"canonical_name": "life-story"},
            ],
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Cluster 1: imagination (indices 0,1) — not in work
        # Cluster 2: biography (indices 2,3) — in work
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.99, 0.01],
            ],
            model="test",
            provider="test",
            dimensions=3,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding, work_id="w1")

        # Only biography cluster merged (1 merge), imagination cluster skipped
        assert result.merged_count == 1
        # execute_write: 2 calls for the single merge (move edges + delete)
        assert neo4j.execute_write.call_count == 2

    @pytest.mark.asyncio
    async def test_scoped_no_work_themes_returns_early(self) -> None:
        """If the work has no themes connected, return early with zero counts."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(side_effect=[
            # Global themes exist
            [
                {"canonical_name": "theme-a", "name": "Theme A", "rel_count": 5},
            ],
            # But no themes for this work
            [],
        ])
        embedding = AsyncMock()

        result = await deduplicate_themes(neo4j, embedding, work_id="w1")

        assert result.original_count == 0
        assert result.merged_count == 0
        embedding.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_scoped_merges_cluster_with_work_theme(self) -> None:
        """A cluster containing a work theme should be fully merged."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(side_effect=[
            # Global themes — 3 similar
            [
                {"canonical_name": "grace-theology", "name": "Grace Theology", "rel_count": 10},
                {"canonical_name": "grace-divine", "name": "Grace Divine", "rel_count": 5},
                {"canonical_name": "grace-redemption", "name": "Grace Redemption", "rel_count": 1},
            ],
            # Work has only grace-divine, but the whole cluster should merge
            [
                {"canonical_name": "grace-divine"},
            ],
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[
                [1.0, 0.1, 0.0],
                [1.0, 0.12, 0.0],
                [1.0, 0.08, 0.0],
            ],
            model="test",
            provider="test",
            dimensions=3,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding, work_id="w1")

        assert result.merged_count == 2  # Both duplicates merged

    @pytest.mark.asyncio
    async def test_unscoped_merges_all_clusters(self) -> None:
        """Without work_id, all clusters should be merged (backwards compat)."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"canonical_name": "theme-a", "name": "Theme A", "rel_count": 10},
            {"canonical_name": "theme-a2", "name": "Theme A2", "rel_count": 3},
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.0], [0.99, 0.01]],
            model="test",
            provider="test",
            dimensions=2,
        ))
        neo4j.execute_write = AsyncMock(return_value=[])

        result = await deduplicate_themes(neo4j, embedding)

        assert result.merged_count == 1

    @pytest.mark.asyncio
    async def test_scoped_with_single_theme_no_merge(self) -> None:
        """If work has a single theme that forms its own cluster, no merge needed."""
        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(side_effect=[
            # Global: two distinct themes
            [
                {"canonical_name": "imagination", "name": "Imagination", "rel_count": 10},
                {"canonical_name": "biography", "name": "Biography", "rel_count": 5},
            ],
            # Work has only imagination
            [
                {"canonical_name": "imagination"},
            ],
        ])

        embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult
        # Orthogonal — no clusters to merge
        embedding.embed_batch = AsyncMock(return_value=BatchEmbeddingResult(
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            model="test",
            provider="test",
            dimensions=2,
        ))

        result = await deduplicate_themes(neo4j, embedding, work_id="w1")

        assert result.merged_count == 0
        neo4j.execute_write.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests: merge operations (requires Neo4j)
# ---------------------------------------------------------------------------


@requires_neo4j
class TestMergeThemeIntoCanonicalIntegration:
    """Test the merge operation against real Neo4j."""

    @pytest.mark.asyncio
    async def test_merge_moves_explores_theme_edges(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """Merging should move EXPLORES_THEME edges from duplicate to canonical."""
        # Setup: Create themes and chunks with edges
        await neo4j_conn.execute_write(
            """CREATE (c1:Chunk {chunk_id: 'c1', work_id: 'w1'})
            CREATE (c2:Chunk {chunk_id: 'c2', work_id: 'w1'})
            CREATE (t_canon:Theme {canonical_name: 'imagination-theology', name: 'Imagination Theology'})
            CREATE (t_dup:Theme {canonical_name: 'imagination-divine', name: 'Imagination Divine'})
            CREATE (c1)-[:EXPLORES_THEME]->(t_canon)
            CREATE (c2)-[:EXPLORES_THEME]->(t_dup)"""
        )

        # Merge duplicate into canonical
        await _merge_theme_into_canonical(
            neo4j_conn,
            canonical_name="imagination-theology",
            duplicate_name="imagination-divine",
        )

        # Verify: c2 should now point to canonical theme
        edges = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: 'c2'})-[:EXPLORES_THEME]->(t:Theme)
            RETURN t.canonical_name AS name"""
        )
        assert len(edges) == 1
        assert edges[0]["name"] == "imagination-theology"

        # Verify: duplicate theme node should be deleted
        dups = await neo4j_conn.execute_read(
            "MATCH (t:Theme {canonical_name: 'imagination-divine'}) RETURN t"
        )
        assert len(dups) == 0

    @pytest.mark.asyncio
    async def test_merge_handles_shared_chunk(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """When a chunk already links to both canonical and duplicate, merge should not create duplicate edge."""
        await neo4j_conn.execute_write(
            """CREATE (c1:Chunk {chunk_id: 'c1', work_id: 'w1'})
            CREATE (t_canon:Theme {canonical_name: 'canon', name: 'Canon'})
            CREATE (t_dup:Theme {canonical_name: 'dup', name: 'Dup'})
            CREATE (c1)-[:EXPLORES_THEME]->(t_canon)
            CREATE (c1)-[:EXPLORES_THEME]->(t_dup)"""
        )

        await _merge_theme_into_canonical(
            neo4j_conn,
            canonical_name="canon",
            duplicate_name="dup",
        )

        # c1 should have exactly 1 EXPLORES_THEME edge (not 2)
        edges = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: 'c1'})-[r:EXPLORES_THEME]->(t:Theme)
            RETURN count(r) AS cnt"""
        )
        assert edges[0]["cnt"] == 1

        # Duplicate should be deleted
        dups = await neo4j_conn.execute_read(
            "MATCH (t:Theme {canonical_name: 'dup'}) RETURN t"
        )
        assert len(dups) == 0

    @pytest.mark.asyncio
    async def test_merge_idempotent(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """Merging when the duplicate doesn't exist should be a no-op."""
        await neo4j_conn.execute_write(
            """CREATE (t:Theme {canonical_name: 'canon', name: 'Canon'})"""
        )

        # Merge a non-existent duplicate -- should not raise
        await _merge_theme_into_canonical(
            neo4j_conn,
            canonical_name="canon",
            duplicate_name="nonexistent",
        )

        # Canon should still exist
        themes = await neo4j_conn.execute_read(
            "MATCH (t:Theme) RETURN t.canonical_name AS name"
        )
        assert len(themes) == 1
        assert themes[0]["name"] == "canon"

    @pytest.mark.asyncio
    async def test_merge_multiple_chunks(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """Merging should move all chunk edges, not just the first."""
        await neo4j_conn.execute_write(
            """CREATE (c1:Chunk {chunk_id: 'c1'})
            CREATE (c2:Chunk {chunk_id: 'c2'})
            CREATE (c3:Chunk {chunk_id: 'c3'})
            CREATE (canon:Theme {canonical_name: 'canon', name: 'Canon'})
            CREATE (dup:Theme {canonical_name: 'dup', name: 'Dup'})
            CREATE (c1)-[:EXPLORES_THEME]->(dup)
            CREATE (c2)-[:EXPLORES_THEME]->(dup)
            CREATE (c3)-[:EXPLORES_THEME]->(canon)"""
        )

        await _merge_theme_into_canonical(
            neo4j_conn,
            canonical_name="canon",
            duplicate_name="dup",
        )

        # All 3 chunks should now link to canon
        edges = await neo4j_conn.execute_read(
            """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme {canonical_name: 'canon'})
            RETURN count(c) AS cnt"""
        )
        assert edges[0]["cnt"] == 3

        # Duplicate gone
        dups = await neo4j_conn.execute_read(
            "MATCH (t:Theme {canonical_name: 'dup'}) RETURN t"
        )
        assert len(dups) == 0


# ---------------------------------------------------------------------------
# Integration tests: full deduplicate_themes pipeline (requires Neo4j)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="DESTRUCTIVE (td-aef7c5): deduplicate_themes() runs against the WHOLE "
    "shared Neo4j graph, and the mock embedder maps every theme name without "
    "'imagination'/'poetry' to the same vector — merging ALL production themes "
    "into one node. Verified to have destroyed production Theme nodes on "
    "2026-07-02. Do not re-enable until dedup can run against an isolated graph."
)
@requires_neo4j
class TestDeduplicateThemesIntegration:
    """Full pipeline test with real Neo4j and mocked embeddings."""

    @pytest.mark.asyncio
    async def test_full_dedup_pipeline(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """End-to-end test: create duplicate themes, run dedup, verify merge."""
        # Setup: Create a realistic graph with duplicate themes
        await neo4j_conn.execute_write(
            """CREATE (c1:Chunk {chunk_id: 'c1', work_id: 'w1', source_class: 'primary'})
            CREATE (c2:Chunk {chunk_id: 'c2', work_id: 'w1', source_class: 'primary'})
            CREATE (c3:Chunk {chunk_id: 'c3', work_id: 'w1', source_class: 'primary'})
            CREATE (t1:Theme {canonical_name: 'imagination-theology', name: 'Imagination and Theology'})
            CREATE (t2:Theme {canonical_name: 'imagination-divine', name: 'Imagination and the Divine'})
            CREATE (t3:Theme {canonical_name: 'poetry-form', name: 'Poetry Form'})
            CREATE (c1)-[:EXPLORES_THEME]->(t1)
            CREATE (c2)-[:EXPLORES_THEME]->(t2)
            CREATE (c3)-[:EXPLORES_THEME]->(t3)
            CREATE (c1)-[:EXPLORES_THEME]->(t3)"""
        )

        # Create mock embedding provider that returns
        # similar vectors for imagination themes, different for poetry
        mock_embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult

        async def mock_embed_batch(texts: list[str]) -> BatchEmbeddingResult:
            vectors = []
            for text in texts:
                text_lower = text.lower()
                if "imagination" in text_lower:
                    vectors.append([1.0, 0.1, 0.0])
                elif "poetry" in text_lower:
                    vectors.append([0.0, 0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5, 0.5])
            return BatchEmbeddingResult(
                vectors=vectors,
                model="test",
                provider="test",
                dimensions=3,
            )

        mock_embedding.embed_batch = mock_embed_batch

        # Run dedup
        result = await deduplicate_themes(neo4j_conn, mock_embedding)

        assert result.original_count == 3
        # imagination themes should merge, poetry stays separate
        assert result.canonical_count == 2
        assert result.merged_count == 1
        assert result.errors == []

        # Verify Neo4j state
        themes = await neo4j_conn.execute_read(
            "MATCH (t:Theme) RETURN t.canonical_name AS name ORDER BY name"
        )
        theme_names = [t["name"] for t in themes]
        # imagination-theology should survive (more connections: c1 links to it)
        assert "imagination-theology" in theme_names
        assert "imagination-divine" not in theme_names
        assert "poetry-form" in theme_names
        assert len(theme_names) == 2

        # c2 should now link to imagination-theology
        c2_themes = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: 'c2'})-[:EXPLORES_THEME]->(t:Theme)
            RETURN t.canonical_name AS name"""
        )
        assert len(c2_themes) == 1
        assert c2_themes[0]["name"] == "imagination-theology"

    @pytest.mark.asyncio
    async def test_dedup_idempotent_on_real_graph(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """Running dedup twice on the same graph should produce same result."""
        await neo4j_conn.execute_write(
            """CREATE (c1:Chunk {chunk_id: 'c1'})
            CREATE (t1:Theme {canonical_name: 'a', name: 'Theme A'})
            CREATE (t2:Theme {canonical_name: 'b', name: 'Theme B'})
            CREATE (c1)-[:EXPLORES_THEME]->(t1)
            CREATE (c1)-[:EXPLORES_THEME]->(t2)"""
        )

        mock_embedding = AsyncMock()
        from author_library.embeddings.base import BatchEmbeddingResult

        # Themes are distinct
        async def mock_embed(texts: list[str]) -> BatchEmbeddingResult:
            return BatchEmbeddingResult(
                vectors=[[1.0, 0.0], [0.0, 1.0]][:len(texts)],
                model="test",
                provider="test",
                dimensions=2,
            )

        mock_embedding.embed_batch = mock_embed

        result1 = await deduplicate_themes(neo4j_conn, mock_embedding)
        result2 = await deduplicate_themes(neo4j_conn, mock_embedding)

        assert result1.canonical_count == result2.canonical_count
        assert result1.merged_count == 0
        assert result2.merged_count == 0


# ---------------------------------------------------------------------------
# Tests for existing_themes in entity extraction prompt
# ---------------------------------------------------------------------------


class TestExistingThemesPrompt:
    """Test that the extraction prompt includes existing themes."""

    @pytest.mark.asyncio
    async def test_fetch_existing_themes_returns_names(self) -> None:
        """_fetch_existing_themes should return a list of canonical_names."""
        from unittest.mock import MagicMock

        from pydantic import SecretStr

        from author_library.config import APIKeySettings, LLMSettings
        from author_library.graph.entity_extraction import EntityExtractor

        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(return_value=[
            {"name": "primary-imagination"},
            {"name": "sacramental-theology"},
            {"name": "poetry-as-truth"},
        ])

        api_keys = APIKeySettings(anthropic_api_key=SecretStr("test-key"))
        llm_settings = LLMSettings()
        extractor = EntityExtractor(neo4j, api_keys, llm_settings)

        themes = await extractor._fetch_existing_themes()
        assert themes == ["primary-imagination", "sacramental-theology", "poetry-as-truth"]

    @pytest.mark.asyncio
    async def test_fetch_existing_themes_handles_failure(self) -> None:
        """_fetch_existing_themes should return empty list on failure."""
        from pydantic import SecretStr

        from author_library.config import APIKeySettings, LLMSettings
        from author_library.graph.entity_extraction import EntityExtractor

        neo4j = AsyncMock()
        neo4j.execute_read = AsyncMock(side_effect=Exception("Neo4j down"))

        api_keys = APIKeySettings(anthropic_api_key=SecretStr("test-key"))
        llm_settings = LLMSettings()
        extractor = EntityExtractor(neo4j, api_keys, llm_settings)

        themes = await extractor._fetch_existing_themes()
        assert themes == []

    def test_system_prompt_includes_reuse_instruction(self) -> None:
        """The system prompt should instruct theme canonical_name reuse."""
        from author_library.graph.entity_extraction import _EXTRACTION_SYSTEM_PROMPT

        assert "REUSE" in _EXTRACTION_SYSTEM_PROMPT
        assert "existing theme" in _EXTRACTION_SYSTEM_PROMPT.lower()
