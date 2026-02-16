"""Graph query helpers for knowledge graph retrieval.

Provides Cypher query templates for common graph traversal patterns,
returning structured results rather than raw Neo4j records.
Optimized using indexes already created in the storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ChunkResult:
    """A chunk returned from a graph query."""

    chunk_id: str
    work_id: str
    text_preview: str
    granularity: str
    source_class: str


@dataclass(frozen=True)
class ThemeSubgraph:
    """All chunks exploring a theme, with their source works."""

    theme_name: str
    canonical_name: str
    chunks: list[ChunkResult]
    works: list[dict[str, Any]]  # {work_id, title, author, source_class}


@dataclass(frozen=True)
class EngagementChain:
    """A chain of ENGAGES_WITH relationships from a source chunk."""

    source_chunk: ChunkResult
    links: list[EngagementChainLink]


@dataclass(frozen=True)
class EngagementChainLink:
    """A single link in an engagement chain."""

    target_chunk: ChunkResult
    link_type: str  # explicit_citation, implicit_engagement
    confidence: str
    evidence: str
    detection_method: str


@dataclass(frozen=True)
class ArgumentEvolution:
    """Evolution of arguments about a theme via DEVELOPS_FROM chains."""

    theme_name: str
    arguments: list[ArgumentNode]
    development_links: list[tuple[str, str]]  # (from_canonical, to_canonical)


@dataclass(frozen=True)
class ArgumentNode:
    """A single argument in an evolution chain."""

    canonical_name: str
    claim: str
    source_chunks: list[ChunkResult]


@dataclass(frozen=True)
class AuthorNetwork:
    """Network of persons referenced and themes explored by an author."""

    author_id: str
    persons_referenced: list[dict[str, Any]]  # {name, canonical_name, reference_count}
    themes_explored: list[dict[str, Any]]  # {name, canonical_name, chunk_count}
    works: list[dict[str, Any]]  # {work_id, title, source_class}


@dataclass(frozen=True)
class CrossWorkLinks:
    """All passage links from/to a specific work."""

    work_id: str
    outgoing: list[PassageLink]
    incoming: list[PassageLink]


@dataclass(frozen=True)
class PassageLink:
    """A passage link between two chunks."""

    source_chunk_id: str
    target_chunk_id: str
    source_work_id: str
    target_work_id: str
    link_type: str
    confidence: str
    rel_type: str  # ENGAGES_WITH or THEMATIC_PARALLEL


class GraphQueryService:
    """Provides structured graph queries over the knowledge graph."""

    def __init__(self, neo4j: Neo4jConnection) -> None:
        self._neo4j = neo4j

    async def get_theme_subgraph(self, theme_name: str) -> ThemeSubgraph | None:
        """Get all chunks exploring a theme and their associated works.

        Uses canonical_name for theme lookup (leveraging Theme uniqueness constraint).
        """
        canonical = theme_name.lower().replace(" ", "-")

        # Get theme node
        theme_records = await self._neo4j.execute_read(
            """MATCH (t:Theme {canonical_name: $canonical_name})
            RETURN t.name AS name, t.canonical_name AS canonical_name""",
            {"canonical_name": canonical},
        )
        if not theme_records:
            return None

        theme_record = theme_records[0]

        # Get all chunks exploring this theme
        chunk_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme {canonical_name: $canonical_name})
            RETURN c.chunk_id AS chunk_id, c.work_id AS work_id,
                   c.text_preview AS text_preview, c.granularity AS granularity,
                   c.source_class AS source_class""",
            {"canonical_name": canonical},
        )
        chunks = [
            ChunkResult(
                chunk_id=r["chunk_id"],
                work_id=r["work_id"],
                text_preview=r["text_preview"] or "",
                granularity=r["granularity"] or "",
                source_class=r["source_class"] or "",
            )
            for r in chunk_records
        ]

        # Get distinct works
        work_ids = {c.work_id for c in chunks}
        works: list[dict[str, Any]] = []
        for work_id in work_ids:
            work_records = await self._neo4j.execute_read(
                """MATCH (w:Work {work_id: $work_id})
                RETURN w.work_id AS work_id, w.title AS title,
                       w.author AS author, w.source_class AS source_class""",
                {"work_id": work_id},
            )
            works.extend(work_records)

        return ThemeSubgraph(
            theme_name=theme_record["name"],
            canonical_name=theme_record["canonical_name"],
            chunks=chunks,
            works=works,
        )

    async def get_engagement_chain(
        self, chunk_id: str, *, max_depth: int = 5
    ) -> EngagementChain | None:
        """Follow ENGAGES_WITH from a chunk to contextual sources.

        Traverses up to max_depth hops to find the full engagement chain.
        """
        # Get source chunk
        source_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN c.chunk_id AS chunk_id, c.work_id AS work_id,
                   c.text_preview AS text_preview, c.granularity AS granularity,
                   c.source_class AS source_class""",
            {"chunk_id": chunk_id},
        )
        if not source_records:
            return None

        src = source_records[0]
        source_chunk = ChunkResult(
            chunk_id=src["chunk_id"],
            work_id=src["work_id"],
            text_preview=src["text_preview"] or "",
            granularity=src["granularity"] or "",
            source_class=src["source_class"] or "",
        )

        # Follow engagement chain with variable-length path
        link_records = await self._neo4j.execute_read(
            "MATCH (c:Chunk {chunk_id: $chunk_id})-[r:ENGAGES_WITH*1.."
            + str(max_depth)
            + """]->(target:Chunk)
            WITH target, r[0] AS first_rel
            RETURN target.chunk_id AS chunk_id, target.work_id AS work_id,
                   target.text_preview AS text_preview, target.granularity AS granularity,
                   target.source_class AS source_class,
                   first_rel.link_type AS link_type,
                   first_rel.confidence AS confidence,
                   first_rel.evidence AS evidence,
                   first_rel.detection_method AS detection_method""",
            {"chunk_id": chunk_id},
        )

        links: list[EngagementChainLink] = []
        for r in link_records:
            target = ChunkResult(
                chunk_id=r["chunk_id"],
                work_id=r["work_id"],
                text_preview=r["text_preview"] or "",
                granularity=r["granularity"] or "",
                source_class=r["source_class"] or "",
            )
            links.append(
                EngagementChainLink(
                    target_chunk=target,
                    link_type=r.get("link_type", ""),
                    confidence=r.get("confidence", ""),
                    evidence=r.get("evidence", ""),
                    detection_method=r.get("detection_method", ""),
                )
            )

        return EngagementChain(source_chunk=source_chunk, links=links)

    async def get_argument_evolution(self, theme_name: str) -> ArgumentEvolution:
        """Get DEVELOPS_FROM chains for arguments about a theme.

        Finds all arguments connected to chunks that explore the given theme,
        then follows DEVELOPS_FROM relationships.
        """
        canonical = theme_name.lower().replace(" ", "-")

        # Get arguments connected to theme-exploring chunks
        arg_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme {canonical_name: $canonical_name}),
                   (c)-[:MAKES_ARGUMENT]->(a:Argument)
            RETURN DISTINCT a.canonical_name AS canonical_name, a.claim AS claim,
                   collect(DISTINCT c.chunk_id) AS chunk_ids,
                   collect(DISTINCT c.work_id) AS work_ids,
                   collect(DISTINCT c.text_preview) AS previews,
                   collect(DISTINCT c.granularity) AS granularities,
                   collect(DISTINCT c.source_class) AS source_classes""",
            {"canonical_name": canonical},
        )

        arguments: list[ArgumentNode] = []
        for r in arg_records:
            chunks = [
                ChunkResult(
                    chunk_id=cid,
                    work_id=wid,
                    text_preview=preview or "",
                    granularity=gran or "",
                    source_class=sc or "",
                )
                for cid, wid, preview, gran, sc in zip(
                    r["chunk_ids"],
                    r["work_ids"],
                    r["previews"],
                    r["granularities"],
                    r["source_classes"],
                    strict=True,
                )
            ]
            arguments.append(
                ArgumentNode(
                    canonical_name=r["canonical_name"],
                    claim=r["claim"] or "",
                    source_chunks=chunks,
                )
            )

        # Get DEVELOPS_FROM links between arguments
        dev_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme {canonical_name: $canonical_name}),
                   (c)-[:MAKES_ARGUMENT]->(a1:Argument),
                   (a1)-[:DEVELOPS_FROM]->(a2:Argument)
            RETURN DISTINCT a1.canonical_name AS from_arg, a2.canonical_name AS to_arg""",
            {"canonical_name": canonical},
        )
        development_links = [(r["from_arg"], r["to_arg"]) for r in dev_records]

        return ArgumentEvolution(
            theme_name=theme_name,
            arguments=arguments,
            development_links=development_links,
        )

    async def get_author_network(self, author_id: str) -> AuthorNetwork:
        """Get the network of persons referenced and themes explored by an author.

        Uses work_id patterns to find works, then aggregates person references
        and theme explorations across all chunks.
        """
        # Get works by this author (author field on Work nodes)
        work_records = await self._neo4j.execute_read(
            """MATCH (w:Work)
            WHERE w.author CONTAINS $author_id
            RETURN w.work_id AS work_id, w.title AS title, w.source_class AS source_class""",
            {"author_id": author_id},
        )
        works = [dict(r) for r in work_records]
        work_ids = [w["work_id"] for w in works]

        if not work_ids:
            return AuthorNetwork(
                author_id=author_id,
                persons_referenced=[],
                themes_explored=[],
                works=[],
            )

        # Get persons referenced across all chunks of this author's works
        person_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk)-[r:REFERENCES_PERSON]->(p:Person)
            WHERE c.work_id IN $work_ids
            RETURN p.name AS name, p.canonical_name AS canonical_name,
                   count(DISTINCT c) AS reference_count
            ORDER BY reference_count DESC""",
            {"work_ids": work_ids},
        )
        persons = [dict(r) for r in person_records]

        # Get themes explored
        theme_records = await self._neo4j.execute_read(
            """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
            WHERE c.work_id IN $work_ids
            RETURN t.name AS name, t.canonical_name AS canonical_name,
                   count(DISTINCT c) AS chunk_count
            ORDER BY chunk_count DESC""",
            {"work_ids": work_ids},
        )
        themes = [dict(r) for r in theme_records]

        return AuthorNetwork(
            author_id=author_id,
            persons_referenced=persons,
            themes_explored=themes,
            works=works,
        )

    async def get_cross_work_links(self, work_id: str) -> CrossWorkLinks:
        """Get all passage links (ENGAGES_WITH + THEMATIC_PARALLEL) from/to a work."""
        # Outgoing links
        out_records = await self._neo4j.execute_read(
            """MATCH (src:Chunk {work_id: $work_id})-[r:ENGAGES_WITH|THEMATIC_PARALLEL]->(tgt:Chunk)
            WHERE tgt.work_id <> $work_id
            RETURN src.chunk_id AS source_chunk_id, tgt.chunk_id AS target_chunk_id,
                   src.work_id AS source_work_id, tgt.work_id AS target_work_id,
                   r.link_type AS link_type, r.confidence AS confidence,
                   type(r) AS rel_type""",
            {"work_id": work_id},
        )
        outgoing = [
            PassageLink(
                source_chunk_id=r["source_chunk_id"],
                target_chunk_id=r["target_chunk_id"],
                source_work_id=r["source_work_id"],
                target_work_id=r["target_work_id"],
                link_type=r.get("link_type", ""),
                confidence=r.get("confidence", ""),
                rel_type=r["rel_type"],
            )
            for r in out_records
        ]

        # Incoming links
        in_records = await self._neo4j.execute_read(
            """MATCH (src:Chunk)-[r:ENGAGES_WITH|THEMATIC_PARALLEL]->(tgt:Chunk {work_id: $work_id})
            WHERE src.work_id <> $work_id
            RETURN src.chunk_id AS source_chunk_id, tgt.chunk_id AS target_chunk_id,
                   src.work_id AS source_work_id, tgt.work_id AS target_work_id,
                   r.link_type AS link_type, r.confidence AS confidence,
                   type(r) AS rel_type""",
            {"work_id": work_id},
        )
        incoming = [
            PassageLink(
                source_chunk_id=r["source_chunk_id"],
                target_chunk_id=r["target_chunk_id"],
                source_work_id=r["source_work_id"],
                target_work_id=r["target_work_id"],
                link_type=r.get("link_type", ""),
                confidence=r.get("confidence", ""),
                rel_type=r["rel_type"],
            )
            for r in in_records
        ]

        return CrossWorkLinks(work_id=work_id, outgoing=outgoing, incoming=incoming)
