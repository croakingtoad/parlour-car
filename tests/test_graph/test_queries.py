"""Tests for graph query helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.graph.queries import (
    ChunkResult,
    EngagementChain,
    GraphQueryService,
    ThemeSubgraph,
)

from .conftest import requires_neo4j

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection


class TestQueryResultModels:
    """Test query result data models."""

    def test_chunk_result(self) -> None:
        """ChunkResult carries all required fields."""
        chunk = ChunkResult(
            chunk_id="c1",
            work_id="w1",
            text_preview="Preview text...",
            granularity="meso",
            source_class="primary",
        )
        assert chunk.chunk_id == "c1"
        assert chunk.source_class == "primary"

    def test_theme_subgraph(self) -> None:
        """ThemeSubgraph holds chunks and works."""
        subgraph = ThemeSubgraph(
            theme_name="Primary Imagination",
            canonical_name="test--primary-imagination",
            chunks=[
                ChunkResult("c1", "w1", "text", "meso", "primary"),
            ],
            works=[{"work_id": "w1", "title": "Test Work"}],
        )
        assert len(subgraph.chunks) == 1
        assert len(subgraph.works) == 1

    def test_engagement_chain(self) -> None:
        """EngagementChain holds source and linked chunks."""
        chain = EngagementChain(
            source_chunk=ChunkResult("c1", "w1", "text", "meso", "primary"),
            links=[],
        )
        assert chain.source_chunk.chunk_id == "c1"
        assert len(chain.links) == 0


@requires_neo4j
class TestGraphQueryServiceWithNeo4j:
    """Integration tests requiring Neo4j."""

    async def _setup_graph(self, neo4j: Neo4jConnection) -> None:
        """Create a test graph with known structure."""
        # Works
        for work in [
            ("guite--faith-hope-poetry", "Faith Hope and Poetry", "Test Guite", "primary"),
            ("coleridge--biographia", "Biographia Literaria", "Coleridge", "contextual"),
        ]:
            await neo4j.execute_write(
                """MERGE (w:Work {work_id: $wid})
                SET w.title = $title, w.author = $author, w.source_class = $sc""",
                {"wid": work[0], "title": work[1], "author": work[2], "sc": work[3]},
            )

        # Chunks
        chunks = [
            ("chunk-p1", "guite--faith-hope-poetry", "primary", "Imagination and faith...", "meso"),
            ("chunk-p2", "guite--faith-hope-poetry", "primary", "Symbol as sacrament...", "meso"),
            ("chunk-c1", "coleridge--biographia", "contextual", "Primary Imagination...", "meso"),
        ]
        for cid, wid, sc, text, gran in chunks:
            await neo4j.execute_write(
                """MERGE (c:Chunk {chunk_id: $cid})
                SET c.work_id = $wid, c.source_class = $sc,
                    c.text_preview = $text, c.granularity = $gran""",
                {"cid": cid, "wid": wid, "sc": sc, "text": text, "gran": gran},
            )

        # Themes
        themes = [
            ("test--primary-imagination", "Primary Imagination"),
            ("sacramental-vision", "Sacramental Vision"),
        ]
        for cn, name in themes:
            await neo4j.execute_write(
                "MERGE (t:Theme {canonical_name: $cn}) SET t.name = $name",
                {"cn": cn, "name": name},
            )

        # Persons
        await neo4j.execute_write(
            "MERGE (p:Person {canonical_name: $cn}) SET p.name = $name",
            {"cn": "test--stc", "name": "Test Coleridge"},
        )

        # Arguments
        await neo4j.execute_write(
            """MERGE (a:Argument {canonical_name: $cn})
            SET a.claim = $claim, a.evidence_summary = $ev""",
            {
                "cn": "imagination-is-sacramental",
                "claim": "Imagination is fundamentally sacramental",
                "ev": "Coleridge's definition echoes sacramental theology",
            },
        )

        # Edges: EXPLORES_THEME
        await neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: 'chunk-p1'}),
                   (t:Theme {canonical_name: 'test--primary-imagination'})
            MERGE (c)-[:EXPLORES_THEME]->(t)"""
        )
        await neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: 'chunk-p2'}),
                   (t:Theme {canonical_name: 'sacramental-vision'})
            MERGE (c)-[:EXPLORES_THEME]->(t)"""
        )
        await neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: 'chunk-c1'}),
                   (t:Theme {canonical_name: 'test--primary-imagination'})
            MERGE (c)-[:EXPLORES_THEME]->(t)"""
        )

        # Edges: MAKES_ARGUMENT (PRIMARY only)
        await neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: 'chunk-p1'}),
                   (a:Argument {canonical_name: 'imagination-is-sacramental'})
            MERGE (c)-[:MAKES_ARGUMENT]->(a)"""
        )

        # Edges: REFERENCES_PERSON
        await neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: 'chunk-p1'}),
                   (p:Person {canonical_name: 'test--stc'})
            MERGE (c)-[r:REFERENCES_PERSON]->(p) SET r.role = 'discussed'"""
        )

        # Edges: ENGAGES_WITH (explicit citation)
        await neo4j.execute_write(
            """MATCH (src:Chunk {chunk_id: 'chunk-p1'}), (tgt:Chunk {chunk_id: 'chunk-c1'})
            MERGE (src)-[r:ENGAGES_WITH]->(tgt)
            SET r.link_type = 'explicit_citation', r.confidence = 'high',
                r.detection_method = 'footnote_reference',
                r.evidence = 'See Coleridge, Ch. XIII'"""
        )

    async def test_get_theme_subgraph(self, neo4j_conn: Neo4jConnection) -> None:
        """Query theme subgraph returns chunks and works."""
        await self._setup_graph(neo4j_conn)

        service = GraphQueryService(neo4j_conn)
        subgraph = await service.get_theme_subgraph("test--primary-imagination")

        assert subgraph is not None
        assert subgraph.theme_name == "Primary Imagination"
        assert subgraph.canonical_name == "test--primary-imagination"
        assert len(subgraph.chunks) == 2  # chunk-p1 and chunk-c1
        chunk_ids = {c.chunk_id for c in subgraph.chunks}
        assert "chunk-p1" in chunk_ids
        assert "chunk-c1" in chunk_ids

    async def test_get_theme_subgraph_not_found(self, neo4j_conn: Neo4jConnection) -> None:
        """Non-existent theme returns None."""
        service = GraphQueryService(neo4j_conn)
        result = await service.get_theme_subgraph("nonexistent-theme")
        assert result is None

    async def test_get_engagement_chain(self, neo4j_conn: Neo4jConnection) -> None:
        """Follow ENGAGES_WITH chain from a primary chunk."""
        await self._setup_graph(neo4j_conn)

        service = GraphQueryService(neo4j_conn)
        chain = await service.get_engagement_chain("chunk-p1")

        assert chain is not None
        assert chain.source_chunk.chunk_id == "chunk-p1"
        assert len(chain.links) >= 1
        assert chain.links[0].target_chunk.chunk_id == "chunk-c1"
        assert chain.links[0].link_type == "explicit_citation"
        assert chain.links[0].confidence == "high"

    async def test_get_engagement_chain_not_found(self, neo4j_conn: Neo4jConnection) -> None:
        """Non-existent chunk returns None."""
        service = GraphQueryService(neo4j_conn)
        result = await service.get_engagement_chain("nonexistent-chunk")
        assert result is None

    async def test_get_argument_evolution(self, neo4j_conn: Neo4jConnection) -> None:
        """Get arguments about a theme."""
        await self._setup_graph(neo4j_conn)

        service = GraphQueryService(neo4j_conn)
        evolution = await service.get_argument_evolution("test--primary-imagination")

        assert evolution.theme_name == "test--primary-imagination"
        assert len(evolution.arguments) >= 1
        arg_names = {a.canonical_name for a in evolution.arguments}
        assert "imagination-is-sacramental" in arg_names

    async def test_get_author_network(self, neo4j_conn: Neo4jConnection) -> None:
        """Get an author's network of persons and themes."""
        await self._setup_graph(neo4j_conn)

        service = GraphQueryService(neo4j_conn)
        network = await service.get_author_network("Guite")

        assert network.author_id == "Guite"
        assert len(network.works) >= 1
        assert len(network.persons_referenced) >= 1

        # Coleridge should be referenced
        person_names = {p["canonical_name"] for p in network.persons_referenced}
        assert "test--stc" in person_names

    async def test_get_cross_work_links(self, neo4j_conn: Neo4jConnection) -> None:
        """Get all passage links from/to a work."""
        await self._setup_graph(neo4j_conn)

        service = GraphQueryService(neo4j_conn)
        links = await service.get_cross_work_links("guite--faith-hope-poetry")

        # Should have outgoing ENGAGES_WITH to Coleridge work
        assert len(links.outgoing) >= 1
        assert links.outgoing[0].rel_type == "ENGAGES_WITH"
        assert links.outgoing[0].target_work_id == "coleridge--biographia"

    async def test_get_cross_work_links_empty(self, neo4j_conn: Neo4jConnection) -> None:
        """Work with no links returns empty lists."""
        service = GraphQueryService(neo4j_conn)
        links = await service.get_cross_work_links("nonexistent-work")
        assert len(links.outgoing) == 0
        assert len(links.incoming) == 0
