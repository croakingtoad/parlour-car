"""Post-extraction theme deduplication using embedding similarity.

After entity extraction generates Theme nodes from independent LLM calls
on separate chunks, many near-duplicate themes accumulate (e.g.
"imagination-and-theology", "imagination-and-the-divine",
"imagination-as-divine-faculty" -- all the same concept).

This module clusters Theme nodes by cosine similarity of their names,
picks a canonical representative for each cluster (the one with the most
relationships), and merges duplicates into it:
  1. Move all EXPLORES_THEME edges from chunks to the canonical theme
  2. Delete the duplicate node

The dedup is idempotent: running it twice produces the same result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Cosine similarity threshold above which two themes are considered duplicates.
# 0.85 is empirically good for short concept phrases (theme canonical_names are
# typically 2-5 words separated by hyphens).
_DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Maximum themes to process in a single dedup run. Safety limit to avoid
# unbounded work on corrupted graphs.
_MAX_THEMES = 10_000

# Batch size for embedding API calls
_EMBED_BATCH_SIZE = 128


@dataclass
class ThemeDedupResult:
    """Result of a theme deduplication pass."""

    original_count: int = 0
    canonical_count: int = 0
    merged_count: int = 0
    clusters_formed: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot: float = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def deduplicate_themes(
    neo4j: Neo4jConnection,
    embedding_provider: EmbeddingProvider,
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    work_id: str | None = None,
) -> ThemeDedupResult:
    """Deduplicate Theme nodes in Neo4j using embedding similarity.

    Algorithm:
    1. Fetch all Theme nodes (canonical_name, name, relationship count).
    2. Embed all theme names via the embedding provider.
    3. Greedily cluster themes by cosine similarity:
       - Sort themes by relationship count (descending) so the most-connected
         theme is picked as canonical first.
       - For each unassigned theme, start a new cluster.
       - Pull in all unassigned themes above the similarity threshold.
    4. For each cluster with >1 member:
       - The first member (most-connected) is the canonical theme.
       - For each duplicate, move all EXPLORES_THEME edges to canonical,
         then delete the duplicate.

    This is idempotent: if there are no duplicates, nothing changes.

    When ``work_id`` is provided, only clusters containing at least one theme
    connected to that work's chunks are processed (merged). All themes are
    still fetched globally for accurate clustering, but the merge phase is
    scoped to the current work's themes.

    Args:
        neo4j: Active Neo4j connection.
        embedding_provider: Provider for generating theme name embeddings.
        similarity_threshold: Cosine similarity above which themes merge.
        work_id: Optional work_id to scope dedup to themes relevant to this work.

    Returns:
        ThemeDedupResult with counts of what was merged.
    """
    result = ThemeDedupResult()
    wall_start = time.monotonic()

    # Step 1: Fetch all Theme nodes with their relationship counts
    theme_records = await neo4j.execute_read(
        """MATCH (t:Theme)
        OPTIONAL MATCH (t)<-[r:EXPLORES_THEME]-()
        WITH t, count(r) AS rel_count
        RETURN t.canonical_name AS canonical_name,
               t.name AS name,
               rel_count
        ORDER BY rel_count DESC""",
    )

    if not theme_records:
        log.info("theme_dedup_no_themes", message="No Theme nodes found")
        result.elapsed_seconds = round(time.monotonic() - wall_start, 2)
        return result

    # If work_id is provided, fetch themes connected to this work's chunks
    work_theme_names: set[str] | None = None
    if work_id is not None:
        work_theme_records = await neo4j.execute_read(
            """MATCH (c:Chunk {work_id: $work_id})-[:EXPLORES_THEME]->(t:Theme)
            RETURN DISTINCT t.canonical_name AS canonical_name""",
            {"work_id": work_id},
        )
        work_theme_names = {r["canonical_name"] for r in work_theme_records}
        if not work_theme_names:
            log.info(
                "theme_dedup_no_work_themes",
                work_id=work_id,
                message="No themes connected to this work's chunks",
            )
            result.elapsed_seconds = round(time.monotonic() - wall_start, 2)
            return result

    result.original_count = len(theme_records)

    if result.original_count > _MAX_THEMES:
        log.warning(
            "theme_dedup_too_many_themes",
            count=result.original_count,
            limit=_MAX_THEMES,
            message="Truncating to safety limit",
        )
        theme_records = theme_records[:_MAX_THEMES]

    log.info(
        "theme_dedup_starting",
        theme_count=len(theme_records),
        similarity_threshold=similarity_threshold,
    )

    # Build lookup structures
    # Use the human-readable name for embedding (better semantic signal),
    # fall back to canonical_name if name is missing
    theme_texts: list[str] = []
    canonical_names: list[str] = []
    rel_counts: list[int] = []

    for rec in theme_records:
        cn = rec["canonical_name"]
        name = rec["name"] or cn
        rc = rec["rel_count"] or 0
        canonical_names.append(cn)
        # Embed the human name for better semantic matching. Replace
        # hyphens in canonical names with spaces for better tokenization.
        embed_text = name if name != cn else cn.replace("-", " ")
        theme_texts.append(embed_text)
        rel_counts.append(rc)

    # Step 2: Embed all theme names
    embeddings: list[list[float]] = []
    for batch_start in range(0, len(theme_texts), _EMBED_BATCH_SIZE):
        batch = theme_texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        try:
            batch_result = await embedding_provider.embed_batch(batch)
            embeddings.extend(batch_result.vectors)
        except Exception as exc:
            error_msg = f"Embedding batch failed at offset {batch_start}: {exc}"
            log.error("theme_dedup_embedding_failed", error=error_msg)
            result.errors.append(error_msg)
            result.elapsed_seconds = round(time.monotonic() - wall_start, 2)
            return result

    if len(embeddings) != len(theme_texts):
        error_msg = (
            f"Embedding count mismatch: got {len(embeddings)} "
            f"for {len(theme_texts)} themes"
        )
        result.errors.append(error_msg)
        result.elapsed_seconds = round(time.monotonic() - wall_start, 2)
        return result

    # Step 3: Greedy clustering
    # theme_records are already sorted by rel_count DESC (most-connected first)
    assigned: set[int] = set()
    clusters: list[list[int]] = []  # Each cluster is a list of indices

    for i in range(len(canonical_names)):
        if i in assigned:
            continue

        # Start a new cluster with this theme as the representative
        cluster = [i]
        assigned.add(i)

        # Find all unassigned themes similar to this one
        for j in range(i + 1, len(canonical_names)):
            if j in assigned:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    result.clusters_formed = len(clusters)
    result.canonical_count = len(clusters)

    # Step 4: Merge duplicates within each cluster
    merge_count = 0
    for cluster in clusters:
        if len(cluster) <= 1:
            continue

        # When scoped to a work, skip clusters that don't contain any
        # of the work's themes.
        if work_theme_names is not None:
            cluster_names = {canonical_names[idx] for idx in cluster}
            if not cluster_names & work_theme_names:
                continue

        # First element is the canonical (most-connected, due to sort order)
        canonical_idx = cluster[0]
        canonical_cn = canonical_names[canonical_idx]

        for dup_idx in cluster[1:]:
            dup_cn = canonical_names[dup_idx]

            try:
                await _merge_theme_into_canonical(
                    neo4j,
                    canonical_name=canonical_cn,
                    duplicate_name=dup_cn,
                )
                merge_count += 1
            except Exception as exc:
                error_msg = (
                    f"Failed to merge theme '{dup_cn}' into "
                    f"'{canonical_cn}': {exc}"
                )
                log.error(
                    "theme_dedup_merge_failed",
                    canonical=canonical_cn,
                    duplicate=dup_cn,
                    error=str(exc),
                )
                result.errors.append(error_msg)

    result.merged_count = merge_count
    result.elapsed_seconds = round(time.monotonic() - wall_start, 2)

    log.info(
        "theme_dedup_complete",
        original=result.original_count,
        canonical=result.canonical_count,
        merged=result.merged_count,
        clusters=result.clusters_formed,
        errors=len(result.errors),
        elapsed_seconds=result.elapsed_seconds,
    )

    return result


async def _merge_theme_into_canonical(
    neo4j: Neo4jConnection,
    *,
    canonical_name: str,
    duplicate_name: str,
) -> None:
    """Merge a duplicate Theme node into the canonical one.

    Moves all EXPLORES_THEME relationships from the duplicate to the
    canonical theme, then deletes the duplicate node.

    In the Author Library graph, EXPLORES_THEME is the only relationship
    type that connects to Theme nodes (Chunk)-[:EXPLORES_THEME]->(Theme).

    Idempotent: if the duplicate doesn't exist, nothing happens. If a
    chunk already has EXPLORES_THEME to the canonical, the MERGE is a no-op.
    """
    # Move EXPLORES_THEME relationships from duplicate to canonical.
    # MERGE handles the case where the chunk already links to canonical
    # (avoiding duplicate edges).
    await neo4j.execute_write(
        """MATCH (dup:Theme {canonical_name: $dup_name})<-[r:EXPLORES_THEME]-(c:Chunk)
        MATCH (canon:Theme {canonical_name: $canon_name})
        MERGE (c)-[:EXPLORES_THEME]->(canon)
        DELETE r""",
        {"dup_name": duplicate_name, "canon_name": canonical_name},
    )

    # Delete the duplicate node. DETACH DELETE removes any remaining edges
    # (safety net in case any unexpected relationship types exist).
    await neo4j.execute_write(
        """MATCH (dup:Theme {canonical_name: $dup_name})
        DETACH DELETE dup""",
        {"dup_name": duplicate_name},
    )

    log.debug(
        "theme_dedup_merged",
        canonical=canonical_name,
        duplicate=duplicate_name,
    )
