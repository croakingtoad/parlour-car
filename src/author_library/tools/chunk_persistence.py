"""Shared helpers for reconciling persisted chunk identities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.chunking.models import Chunk


def canonicalize_stored_chunks(
    chunks: list[Chunk],
    provisional_to_pg_id: dict[str, UUID],
) -> tuple[list[Chunk], dict[str, UUID]]:
    """Adopt PostgreSQL IDs and discard chunks that were not persisted.

    The provisional mapping remains unchanged for parent-FK resolution and
    diagnostics. The returned lookup is keyed by the canonical string IDs used
    on the mutated chunk models, so embedding storage can resolve them safely.
    """
    stored_chunks: list[Chunk] = []
    canonical_id_map: dict[str, UUID] = {}

    for chunk in chunks:
        provisional_id = chunk.id
        pg_id = provisional_to_pg_id.get(provisional_id)
        if pg_id is None:
            continue

        parent_pg_id = (
            provisional_to_pg_id.get(chunk.parent_chunk_id)
            if chunk.parent_chunk_id is not None
            else None
        )
        chunk.id = str(pg_id)
        chunk.parent_chunk_id = str(parent_pg_id) if parent_pg_id is not None else None

        stored_chunks.append(chunk)
        canonical_id_map[chunk.id] = pg_id

    return stored_chunks, canonical_id_map
