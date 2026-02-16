"""Tests for thematic parallel passage linking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.graph.linking_thematic import (
    ThematicLink,
    ThematicLinkResult,
    ThematicParallelDetector,
    _cosine_similarity,
)

from .conftest import requires_neo4j

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9

    def test_zero_vector(self) -> None:
        """Zero vector should return 0.0 similarity."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_similar_vectors(self) -> None:
        """Similar vectors should have high similarity."""
        sim = _cosine_similarity([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        assert sim > 0.99


class TestThematicLinkModel:
    """Test ThematicLink data model."""

    def test_link_fields(self) -> None:
        """ThematicLink should carry similarity score and shared themes."""
        link = ThematicLink(
            source_chunk_id="p-001",
            target_chunk_id="c-001",
            similarity_score=0.87,
            shared_themes=["primary-imagination", "sacramental-vision"],
        )
        assert link.similarity_score == 0.87
        assert len(link.shared_themes) == 2

    def test_result_accumulates(self) -> None:
        """ThematicLinkResult should accumulate counts."""
        result = ThematicLinkResult()
        result.pairs_evaluated = 10
        result.pairs_above_threshold = 3
        result.pairs_skipped_existing = 2
        result.edges_created = 1
        assert result.pairs_evaluated == 10


class TestThemeOverlapDetection:
    """Test theme-overlapping pair finding (no Neo4j needed)."""

    def test_find_overlapping_pairs(self) -> None:
        """Chunks sharing themes should be identified as candidates."""
        detector = ThematicParallelDetector.__new__(ThematicParallelDetector)

        primary = [
            Chunk(
                id="p1",
                text="text",
                granularity=ChunkGranularity.MESO,
                work_id="w1",
                source_class="primary",
                position=1,
            ),
        ]
        contextual = [
            Chunk(
                id="c1",
                text="text",
                granularity=ChunkGranularity.MESO,
                work_id="w2",
                source_class="contextual",
                position=1,
            ),
        ]

        primary_themes = {"p1": {"imagination", "symbol"}}
        contextual_themes = {"c1": {"imagination", "nature"}}

        pairs = detector._find_theme_overlapping_pairs(
            primary, contextual, primary_themes, contextual_themes
        )
        assert len(pairs) == 1
        assert pairs[0][0] == "p1"
        assert pairs[0][1] == "c1"
        assert "imagination" in pairs[0][2]

    def test_no_overlap_no_pairs(self) -> None:
        """Chunks without shared themes should not be paired."""
        detector = ThematicParallelDetector.__new__(ThematicParallelDetector)

        primary = [
            Chunk(
                id="p1",
                text="text",
                granularity=ChunkGranularity.MESO,
                work_id="w1",
                source_class="primary",
                position=1,
            ),
        ]
        contextual = [
            Chunk(
                id="c1",
                text="text",
                granularity=ChunkGranularity.MESO,
                work_id="w2",
                source_class="contextual",
                position=1,
            ),
        ]

        primary_themes = {"p1": {"imagination"}}
        contextual_themes = {"c1": {"nature"}}

        pairs = detector._find_theme_overlapping_pairs(
            primary, contextual, primary_themes, contextual_themes
        )
        assert len(pairs) == 0


@requires_neo4j
class TestThematicParallelWithNeo4j:
    """Integration tests requiring Neo4j."""

    async def test_no_links_without_shared_themes(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
    ) -> None:
        """No thematic parallels when chunks don't share themes."""
        # Create chunk nodes without theme connections
        for chunk in primary_chunks + contextual_chunks:
            await neo4j_conn.execute_write(
                """MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.work_id = $work_id, c.source_class = $source_class""",
                {
                    "chunk_id": chunk.id,
                    "work_id": chunk.work_id,
                    "source_class": chunk.source_class,
                },
            )

        # Use a simple embedding provider that returns fixed vectors
        provider = _FixedEmbeddingProvider(dim=4)
        detector = ThematicParallelDetector(neo4j_conn, provider)
        result = await detector.detect_and_link(primary_chunks, contextual_chunks)

        # No theme assignments → no candidate pairs → no links
        assert result.pairs_evaluated == 0
        assert result.edges_created == 0

    async def test_thematic_parallel_with_shared_themes(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
    ) -> None:
        """Create thematic parallel edges when chunks share themes and are similar."""
        # Create chunk and theme nodes with EXPLORES_THEME edges
        theme_name = "primary-imagination"
        await neo4j_conn.execute_write(
            "MERGE (t:Theme {canonical_name: $name}) SET t.name = 'Primary Imagination'",
            {"name": theme_name},
        )

        for chunk in [primary_chunks[0], contextual_chunks[0]]:
            await neo4j_conn.execute_write(
                """MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.work_id = $work_id, c.source_class = $source_class,
                    c.text_preview = $text, c.granularity = $gran""",
                {
                    "chunk_id": chunk.id,
                    "work_id": chunk.work_id,
                    "source_class": chunk.source_class,
                    "text": chunk.text[:200],
                    "gran": str(chunk.granularity),
                },
            )
            await neo4j_conn.execute_write(
                """MATCH (c:Chunk {chunk_id: $chunk_id}), (t:Theme {canonical_name: $theme})
                MERGE (c)-[:EXPLORES_THEME]->(t)""",
                {"chunk_id": chunk.id, "theme": theme_name},
            )

        # Use a provider that returns very similar vectors for these texts
        provider = _HighSimilarityProvider(dim=4)
        detector = ThematicParallelDetector(neo4j_conn, provider)
        result = await detector.detect_and_link(
            [primary_chunks[0]],
            [contextual_chunks[0]],
        )

        assert result.pairs_evaluated >= 1
        assert result.pairs_above_threshold >= 1
        assert result.edges_created >= 1

        # Verify edge properties
        edges = await neo4j_conn.execute_read(
            """MATCH ()-[r:THEMATIC_PARALLEL]->()
            RETURN r.confidence AS confidence, r.similarity_score AS score,
                   r.shared_themes AS themes"""
        )
        assert len(edges) >= 1
        assert edges[0]["confidence"] == "low"
        assert edges[0]["score"] >= 0.85

    async def test_skips_pairs_with_existing_engages_with(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
    ) -> None:
        """Pairs with existing ENGAGES_WITH edges should be skipped."""
        theme_name = "imagination-test"
        await neo4j_conn.execute_write(
            "MERGE (t:Theme {canonical_name: $name}) SET t.name = 'Imagination Test'",
            {"name": theme_name},
        )

        p_chunk = primary_chunks[0]
        c_chunk = contextual_chunks[0]

        for chunk in [p_chunk, c_chunk]:
            await neo4j_conn.execute_write(
                """MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.work_id = $work_id, c.source_class = $source_class""",
                {
                    "chunk_id": chunk.id,
                    "work_id": chunk.work_id,
                    "source_class": chunk.source_class,
                },
            )
            await neo4j_conn.execute_write(
                """MATCH (c:Chunk {chunk_id: $chunk_id}), (t:Theme {canonical_name: $theme})
                MERGE (c)-[:EXPLORES_THEME]->(t)""",
                {"chunk_id": chunk.id, "theme": theme_name},
            )

        # Create existing ENGAGES_WITH edge
        await neo4j_conn.execute_write(
            """MATCH (src:Chunk {chunk_id: $src}), (tgt:Chunk {chunk_id: $tgt})
            MERGE (src)-[:ENGAGES_WITH]->(tgt)""",
            {"src": p_chunk.id, "tgt": c_chunk.id},
        )

        provider = _HighSimilarityProvider(dim=4)
        detector = ThematicParallelDetector(neo4j_conn, provider)
        result = await detector.detect_and_link([p_chunk], [c_chunk])

        assert result.pairs_skipped_existing >= 1
        assert result.edges_created == 0


# ---------------------------------------------------------------------------
# Test embedding providers
# ---------------------------------------------------------------------------


class _FixedEmbeddingProvider:
    """Embedding provider that returns fixed vectors for testing."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test-fixed"

    @property
    def dimensions(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> object:
        from author_library.embeddings.base import EmbeddingResult

        return EmbeddingResult(
            vector=[0.5] * self._dim,
            model="test",
            provider="test",
            dimensions=self._dim,
        )

    async def embed_batch(self, texts: list[str]) -> object:
        from author_library.embeddings.base import BatchEmbeddingResult

        return BatchEmbeddingResult(
            vectors=[[0.5] * self._dim for _ in texts],
            model="test",
            provider="test",
            dimensions=self._dim,
        )

    async def embed_query(self, text: str) -> object:
        return await self.embed_text(text)

    async def close(self) -> None:
        pass


class _HighSimilarityProvider:
    """Embedding provider that returns very similar vectors for all texts."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test-high-sim"

    @property
    def dimensions(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> object:
        from author_library.embeddings.base import EmbeddingResult

        # Return nearly identical vectors (small perturbation)
        self._call_count += 1
        base = [1.0] * self._dim
        base[0] += self._call_count * 0.001  # Tiny perturbation
        return EmbeddingResult(
            vector=base,
            model="test",
            provider="test",
            dimensions=self._dim,
        )

    async def embed_batch(self, texts: list[str]) -> object:
        from author_library.embeddings.base import BatchEmbeddingResult

        vectors = []
        for _ in texts:
            self._call_count += 1
            base = [1.0] * self._dim
            base[0] += self._call_count * 0.001
            vectors.append(base)
        return BatchEmbeddingResult(
            vectors=vectors,
            model="test",
            provider="test",
            dimensions=self._dim,
        )

    async def embed_query(self, text: str) -> object:
        return await self.embed_text(text)

    async def close(self) -> None:
        pass
