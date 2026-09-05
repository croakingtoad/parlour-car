"""Tests for shared chunk persistence reconciliation."""

from __future__ import annotations

from uuid import UUID

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.tools.chunk_persistence import canonicalize_stored_chunks


def _chunk(
    chunk_id: str,
    *,
    position: int,
    parent_chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"Chunk {position}",
        granularity=ChunkGranularity.MACRO,
        work_id="author--work",
        source_class="primary",
        position=position,
        parent_chunk_id=parent_chunk_id,
    )


def test_canonicalize_stored_chunks_rekeys_ids_and_excludes_failed_inserts() -> None:
    parent_pg_id = UUID("11111111-1111-4111-8111-111111111111")
    child_pg_id = UUID("22222222-2222-4222-8222-222222222222")
    parent = _chunk("provisional-parent", position=0)
    child = _chunk(
        "provisional-child",
        position=1,
        parent_chunk_id=parent.id,
    )
    failed = _chunk("provisional-failed", position=2)
    provisional_to_pg_id = {
        parent.id: parent_pg_id,
        child.id: child_pg_id,
    }

    stored_chunks, canonical_id_map = canonicalize_stored_chunks(
        [parent, child, failed],
        provisional_to_pg_id,
    )

    assert stored_chunks == [parent, child]
    assert [chunk.id for chunk in stored_chunks] == [
        str(parent_pg_id),
        str(child_pg_id),
    ]
    assert child.parent_chunk_id == str(parent_pg_id)
    assert canonical_id_map == {
        str(parent_pg_id): parent_pg_id,
        str(child_pg_id): child_pg_id,
    }
    assert provisional_to_pg_id == {
        "provisional-parent": parent_pg_id,
        "provisional-child": child_pg_id,
    }
