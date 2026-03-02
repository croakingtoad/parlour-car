"""Repository pattern for storage operations.

Provides abstract interfaces and concrete implementations for
PostgreSQL and Neo4j data access in The Author Library.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import StorageError
from author_library.text_utils import sanitize_text

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.storage.neo4j import Neo4jConnection
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class WorkRepository(ABC):
    """Interface for work/catalog entry operations."""

    @abstractmethod
    async def create(self, work: dict[str, Any]) -> str:
        """Insert a work and return its work_id."""

    @abstractmethod
    async def get(self, work_id: str) -> dict[str, Any] | None:
        """Retrieve a work by work_id."""

    @abstractmethod
    async def list_by_author(self, author: str) -> list[dict[str, Any]]:
        """List works for a given author slug or display name."""

    @abstractmethod
    async def update(self, work_id: str, fields: dict[str, Any]) -> bool:
        """Update work fields. Returns True if a row was updated."""

    @abstractmethod
    async def delete(self, work_id: str) -> bool:
        """Delete a work. Returns True if a row was deleted."""


class ChunkRepository(ABC):
    """Interface for chunk operations across granularities."""

    @abstractmethod
    async def create(self, chunk: dict[str, Any]) -> UUID:
        """Insert a chunk and return its id."""

    @abstractmethod
    async def get(self, chunk_id: UUID) -> dict[str, Any] | None:
        """Retrieve a chunk by id."""

    @abstractmethod
    async def list_by_work(
        self, work_id: str, *, granularity: str | None = None
    ) -> list[dict[str, Any]]:
        """List chunks for a work, optionally filtered by granularity."""

    @abstractmethod
    async def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by id."""

    @abstractmethod
    async def delete_by_work(self, work_id: str) -> int:
        """Delete all chunks for a work. Returns count deleted."""

    @abstractmethod
    async def get_max_pass_number(self, work_id: str) -> int:
        """Get the maximum pass_number for a work's chunks."""


class EmbeddingRepository(ABC):
    """Interface for chunk embedding storage/retrieval."""

    @abstractmethod
    async def store(
        self,
        chunk_id: UUID,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
    ) -> UUID:
        """Store an embedding vector for a chunk."""

    @abstractmethod
    async def get_by_chunk(self, chunk_id: UUID) -> list[dict[str, Any]]:
        """Retrieve all embeddings for a chunk."""

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        provider: str,
        model: str,
        limit: int = 20,
        source_class_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the query embedding using cosine distance."""


class ThematicRepository(ABC):
    """Interface for thematic index CRUD."""

    @abstractmethod
    async def create_entry(self, entry: dict[str, Any]) -> UUID:
        """Create a thematic entry."""

    @abstractmethod
    async def get_entry(self, entry_id: UUID) -> dict[str, Any] | None:
        """Get a thematic entry by id."""

    @abstractmethod
    async def list_entries(self, author_id: str) -> list[dict[str, Any]]:
        """List thematic entries for an author."""

    @abstractmethod
    async def add_appearance(self, appearance: dict[str, Any]) -> UUID:
        """Add a thematic appearance (theme x work junction)."""

    @abstractmethod
    async def delete_entry(self, entry_id: UUID) -> bool:
        """Delete a thematic entry and its appearances."""


class VoiceProfileRepository(ABC):
    """Interface for voice profile storage."""

    @abstractmethod
    async def store(self, author_id: str, profile: dict[str, Any], version: int) -> UUID:
        """Store a voice profile version."""

    @abstractmethod
    async def get_current(self, author_id: str) -> dict[str, Any] | None:
        """Get the current (latest active) voice profile for an author."""

    @abstractmethod
    async def list_versions(self, author_id: str) -> list[dict[str, Any]]:
        """List all voice profile versions for an author."""


class SessionRepository(ABC):
    """Interface for session tracking operations."""

    @abstractmethod
    async def create(self, session: dict[str, Any]) -> UUID:
        """Create a new session and return its id."""

    @abstractmethod
    async def get(self, session_id: UUID) -> dict[str, Any] | None:
        """Get a session by id."""

    @abstractmethod
    async def get_active(self, user_id: str) -> dict[str, Any] | None:
        """Get the currently active (open) session for a user."""

    @abstractmethod
    async def end_session(self, session_id: UUID) -> bool:
        """End a session by setting date_end and calculating duration."""

    @abstractmethod
    async def add_capture(self, session_id: UUID, chunk_id: UUID, capture_order: int) -> UUID:
        """Add a capture (chunk) to a session."""

    @abstractmethod
    async def add_source(self, session_id: UUID, work_id: str) -> None:
        """Associate a work with a session."""

    @abstractmethod
    async def list_captures(self, session_id: UUID) -> list[dict[str, Any]]:
        """List all captures in a session."""

    @abstractmethod
    async def list_sessions(
        self, user_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List recent sessions for a user."""


class TranscriptCacheRepository(ABC):
    """Interface for transcript cache operations."""

    @abstractmethod
    async def get_cached(self, source_url: str) -> str | None:
        """Return cached transcript text if not expired, or None."""

    @abstractmethod
    async def cache(self, source_url: str, transcript_text: str, ttl_seconds: int = 86400) -> None:
        """Store or replace a transcript in the cache."""

    @abstractmethod
    async def invalidate(self, source_url: str) -> bool:
        """Remove a cached transcript. Returns True if a row was deleted."""

    @abstractmethod
    async def invalidate_expired(self) -> int:
        """Remove all expired cache entries. Returns count deleted."""


class GraphRepository(ABC):
    """Interface for Neo4j graph operations."""

    @abstractmethod
    async def upsert_work_node(self, work: dict[str, Any]) -> None:
        """Create or update a Work node in the graph."""

    @abstractmethod
    async def upsert_chunk_node(self, chunk: dict[str, Any]) -> None:
        """Create or update a Chunk node in the graph."""

    @abstractmethod
    async def create_edge(
        self,
        from_label: str,
        from_key: str,
        from_value: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two nodes."""

    @abstractmethod
    async def get_related_chunks(
        self, chunk_id: str, rel_type: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get chunks related to a given chunk via a specific relationship type."""

    @abstractmethod
    async def get_themes_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        """Get all themes explored by a chunk."""

    @abstractmethod
    async def create_user_reflects_on_edge(
        self,
        *,
        reflection_chunk_id: str,
        target_id: str,
        target_type: str,
        target_label: str,
        target_key: str,
        date_created: str | None = None,
    ) -> None:
        """Create a USER_REFLECTS_ON edge from a personal chunk to a target."""

    @abstractmethod
    async def get_reflections_for_target(
        self, target_id: str, target_key: str, target_label: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get all personal reflection chunks for a target via USER_REFLECTS_ON."""


# ---------------------------------------------------------------------------
# Concrete PostgreSQL implementations
# ---------------------------------------------------------------------------


class PgWorkRepository(WorkRepository):
    """PostgreSQL-backed work repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, work: dict[str, Any]) -> str:
        work_id: str = work["work_id"]
        await self._pool.execute(
            """INSERT INTO works (
                work_id, title, author, source_class, source_class_note,
                publication_year, original_publication_year, edition, publisher, isbn,
                format_ingested, language, word_count, genre_tags, subject_headings,
                ocr_quality, notes, source_metadata,
                url, duration, speakers, date_published, date_consumed,
                transcript_cached, media
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15,
                $16, $17, $18,
                $19, $20, $21, $22, $23,
                $24, $25
            )""",
            work_id,
            work["title"],
            work["author"],
            work["source_class"],
            work["source_class_note"],
            work["publication_year"],
            work.get("original_publication_year"),
            work.get("edition"),
            work["publisher"],
            work.get("isbn"),
            work["format_ingested"],
            work.get("language", "en"),
            work["word_count"],
            work["genre_tags"],
            work["subject_headings"],
            work.get("ocr_quality"),
            work.get("notes"),
            json.dumps(work.get("source_metadata", {})),
            work.get("url"),
            work.get("duration"),
            work.get("speakers"),
            work.get("date_published"),
            work.get("date_consumed"),
            work.get("transcript_cached", False),
            work.get("media"),
        )
        return work_id

    async def get(self, work_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetch_one("SELECT * FROM works WHERE work_id = $1", work_id)
        return dict(row) if row else None

    async def list_by_author(self, author: str) -> list[dict[str, Any]]:
        # Query by author slug in source_metadata JSONB fields first, then
        # fall back to direct author column match.  The source_metadata stores
        # the author slug differently per source class:
        #   primary   → subject_author_id
        #   secondary → about_author_id
        #   contextual→ referenced_by
        rows = await self._pool.fetch_all(
            """SELECT * FROM works
            WHERE source_metadata->>'subject_author_id' = $1
               OR source_metadata->>'about_author_id' = $1
               OR source_metadata->>'referenced_by' = $1
               OR author ILIKE $1
            ORDER BY publication_year""",
            author,
        )
        return [dict(r) for r in rows]

    async def update(self, work_id: str, fields: dict[str, Any]) -> bool:
        if not fields:
            return False
        set_clauses: list[str] = []
        params: list[Any] = []
        idx = 1
        for key, value in fields.items():
            set_clauses.append(f"{key} = ${idx}")
            params.append(json.dumps(value) if key == "source_metadata" else value)
            idx += 1
        params.append(work_id)
        result = await self._pool.execute(
            f"UPDATE works SET {', '.join(set_clauses)}, updated_at = NOW() WHERE work_id = ${idx}",
            *params,
        )
        return result.endswith("1")

    async def delete(self, work_id: str) -> bool:
        result = await self._pool.execute("DELETE FROM works WHERE work_id = $1", work_id)
        return result.endswith("1")


class PgChunkRepository(ChunkRepository):
    """PostgreSQL-backed chunk repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, chunk: dict[str, Any]) -> UUID:
        # Safety net: sanitize text fields before INSERT to prevent
        # invalid UTF-8 byte sequences from reaching PostgreSQL.
        text = sanitize_text(chunk["text"]) if chunk.get("text") else chunk.get("text", "")
        annotation = sanitize_text(chunk["annotation"]) if chunk.get("annotation") else chunk.get("annotation")
        row = await self._pool.fetch_one(
            """INSERT INTO chunks (
                work_id, text, annotation, granularity, source_class,
                chapter, section, position, parent_chunk_id, metadata,
                raw_content, raw_content_window, pass_number
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id""",
            chunk["work_id"],
            text,
            annotation,
            chunk["granularity"],
            chunk["source_class"],
            chunk.get("chapter"),
            chunk.get("section"),
            chunk["position"],
            chunk.get("parent_chunk_id"),
            json.dumps(chunk.get("metadata", {})),
            chunk.get("raw_content"),
            chunk.get("raw_content_window"),
            chunk.get("pass_number", 1),
        )
        if row is None:
            raise StorageError("Failed to insert chunk — no id returned")
        return row["id"]  # type: ignore[no-any-return]

    async def get_max_pass_number(self, work_id: str) -> int:
        """Get the maximum pass_number for a work's chunks.

        Returns 0 if no chunks exist for the work.
        """
        result = await self._pool.fetch_val(
            "SELECT COALESCE(MAX(pass_number), 0) FROM chunks WHERE work_id = $1",
            work_id,
        )
        return int(result)

    async def get(self, chunk_id: UUID) -> dict[str, Any] | None:
        row = await self._pool.fetch_one("SELECT * FROM chunks WHERE id = $1", chunk_id)
        return dict(row) if row else None

    async def list_by_work(
        self, work_id: str, *, granularity: str | None = None
    ) -> list[dict[str, Any]]:
        if granularity:
            rows = await self._pool.fetch_all(
                "SELECT * FROM chunks WHERE work_id = $1 AND granularity = $2 ORDER BY position",
                work_id,
                granularity,
            )
        else:
            rows = await self._pool.fetch_all(
                "SELECT * FROM chunks WHERE work_id = $1 ORDER BY position",
                work_id,
            )
        return [dict(r) for r in rows]

    async def delete(self, chunk_id: UUID) -> bool:
        result = await self._pool.execute("DELETE FROM chunks WHERE id = $1", chunk_id)
        return result.endswith("1")

    async def delete_by_work(self, work_id: str) -> int:
        result = await self._pool.execute("DELETE FROM chunks WHERE work_id = $1", work_id)
        # asyncpg returns "DELETE N" where N is the count
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


class PgEmbeddingRepository(EmbeddingRepository):
    """PostgreSQL + pgvector embedding repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def store(
        self,
        chunk_id: UUID,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
    ) -> UUID:
        # asyncpg doesn't natively serialize vector; pass as string
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        row = await self._pool.fetch_one(
            """INSERT INTO chunk_embeddings (chunk_id, embedding, provider, model, dimensions)
            VALUES ($1, $2::vector, $3, $4, $5)
            ON CONFLICT (chunk_id, provider, model) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    dimensions = EXCLUDED.dimensions,
                    created_at = NOW()
            RETURNING id""",
            chunk_id,
            vec_str,
            provider,
            model,
            dimensions,
        )
        if row is None:
            raise StorageError("Failed to store embedding — no id returned")
        return row["id"]  # type: ignore[no-any-return]

    async def get_by_chunk(self, chunk_id: UUID) -> list[dict[str, Any]]:
        rows = await self._pool.fetch_all(
            """SELECT id, chunk_id, provider, model, dimensions, created_at
            FROM chunk_embeddings WHERE chunk_id = $1""",
            chunk_id,
        )
        return [dict(r) for r in rows]

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        provider: str,
        model: str,
        limit: int = 20,
        source_class_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        conditions = [
            "ce.provider = $2",
            "ce.model = $3",
        ]
        params: list[Any] = [vec_str, provider, model]
        idx = 4

        if source_class_filter is not None:
            conditions.append(f"c.source_class = ${idx}")
            params.append(source_class_filter)
            idx += 1

        params.append(limit)
        where = " AND ".join(conditions)

        sql = f"""
            SELECT
                ce.chunk_id,
                c.work_id,
                c.text,
                c.granularity,
                c.source_class,
                c.pass_number,
                c.metadata->>'speaker' AS speaker,
                (ce.embedding <=> $1::vector) AS distance
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE {where}
            ORDER BY ce.embedding <=> $1::vector
            LIMIT ${idx}
        """

        rows = await self._pool.fetch_all(sql, *params)
        return [dict(r) for r in rows]


class PgThematicRepository(ThematicRepository):
    """PostgreSQL-backed thematic index repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create_entry(self, entry: dict[str, Any]) -> UUID:
        row = await self._pool.fetch_one(
            """INSERT INTO thematic_entries
            (author_id, theme, author_stance, related_themes, key_passages)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id""",
            entry["author_id"],
            entry["theme"],
            entry.get("author_stance"),
            entry.get("related_themes"),
            json.dumps(entry.get("key_passages", [])),
        )
        if row is None:
            raise StorageError("Failed to create thematic entry")
        return row["id"]  # type: ignore[no-any-return]

    async def get_entry(self, entry_id: UUID) -> dict[str, Any] | None:
        row = await self._pool.fetch_one(
            "SELECT * FROM thematic_entries WHERE id = $1", entry_id
        )
        return dict(row) if row else None

    async def list_entries(self, author_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch_all(
            "SELECT * FROM thematic_entries WHERE author_id = $1 ORDER BY theme",
            author_id,
        )
        return [dict(r) for r in rows]

    async def add_appearance(self, appearance: dict[str, Any]) -> UUID:
        row = await self._pool.fetch_one(
            """INSERT INTO thematic_appearances (entry_id, work_id, chapters, treatment_summary)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (entry_id, work_id) DO UPDATE
                SET chapters = EXCLUDED.chapters,
                    treatment_summary = EXCLUDED.treatment_summary
            RETURNING id""",
            appearance["entry_id"],
            appearance["work_id"],
            appearance.get("chapters"),
            appearance.get("treatment_summary"),
        )
        if row is None:
            raise StorageError("Failed to add thematic appearance")
        return row["id"]  # type: ignore[no-any-return]

    async def delete_entry(self, entry_id: UUID) -> bool:
        result = await self._pool.execute(
            "DELETE FROM thematic_entries WHERE id = $1", entry_id
        )
        return result.endswith("1")


class PgVoiceProfileRepository(VoiceProfileRepository):
    """PostgreSQL-backed voice profile repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def store(self, author_id: str, profile: dict[str, Any], version: int) -> UUID:
        # Mark previous versions as non-current
        await self._pool.execute(
            "UPDATE voice_profiles SET is_current = FALSE WHERE author_id = $1",
            author_id,
        )
        row = await self._pool.fetch_one(
            """INSERT INTO voice_profiles (author_id, version, profile, is_current)
            VALUES ($1, $2, $3, TRUE)
            ON CONFLICT (author_id, version) DO UPDATE
                SET profile = EXCLUDED.profile,
                    is_current = TRUE,
                    created_at = NOW()
            RETURNING id""",
            author_id,
            version,
            json.dumps(profile),
        )
        if row is None:
            raise StorageError("Failed to store voice profile")
        return row["id"]  # type: ignore[no-any-return]

    async def get_current(self, author_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetch_one(
            "SELECT * FROM voice_profiles WHERE author_id = $1 AND is_current = TRUE",
            author_id,
        )
        return dict(row) if row else None

    async def list_versions(self, author_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch_all(
            "SELECT * FROM voice_profiles WHERE author_id = $1 ORDER BY version",
            author_id,
        )
        return [dict(r) for r in rows]


class PgSessionRepository(SessionRepository):
    """PostgreSQL-backed session repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, session: dict[str, Any]) -> UUID:
        row = await self._pool.fetch_one(
            """INSERT INTO sessions (title, user_id)
            VALUES ($1, $2)
            RETURNING id""",
            session.get("title"),
            session.get("user_id", "marty"),
        )
        if row is None:
            raise StorageError("Failed to create session — no id returned")
        return row["id"]  # type: ignore[no-any-return]

    async def get(self, session_id: UUID) -> dict[str, Any] | None:
        row = await self._pool.fetch_one(
            "SELECT * FROM sessions WHERE id = $1", session_id
        )
        return dict(row) if row else None

    async def get_active(self, user_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetch_one(
            """SELECT * FROM sessions
            WHERE user_id = $1 AND date_end IS NULL
            ORDER BY date_start DESC LIMIT 1""",
            user_id,
        )
        return dict(row) if row else None

    async def end_session(self, session_id: UUID) -> bool:
        result = await self._pool.execute(
            """UPDATE sessions
            SET date_end = NOW(),
                duration_minutes = EXTRACT(EPOCH FROM (NOW() - date_start))::INT / 60,
                updated_at = NOW()
            WHERE id = $1 AND date_end IS NULL""",
            session_id,
        )
        return result.endswith("1")

    async def add_capture(self, session_id: UUID, chunk_id: UUID, capture_order: int) -> UUID:
        row = await self._pool.fetch_one(
            """INSERT INTO session_captures (session_id, chunk_id, capture_order)
            VALUES ($1, $2, $3)
            ON CONFLICT (session_id, chunk_id) DO UPDATE
                SET capture_order = EXCLUDED.capture_order
            RETURNING id""",
            session_id,
            chunk_id,
            capture_order,
        )
        if row is None:
            raise StorageError("Failed to add capture to session")
        return row["id"]  # type: ignore[no-any-return]

    async def add_source(self, session_id: UUID, work_id: str) -> None:
        await self._pool.execute(
            """INSERT INTO session_sources (session_id, work_id)
            VALUES ($1, $2)
            ON CONFLICT (session_id, work_id) DO NOTHING""",
            session_id,
            work_id,
        )

    async def list_captures(self, session_id: UUID) -> list[dict[str, Any]]:
        rows = await self._pool.fetch_all(
            """SELECT sc.*, c.text, c.work_id, c.granularity, c.source_class
            FROM session_captures sc
            JOIN chunks c ON c.id = sc.chunk_id
            WHERE sc.session_id = $1
            ORDER BY sc.capture_order""",
            session_id,
        )
        return [dict(r) for r in rows]

    async def list_sessions(
        self, user_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self._pool.fetch_all(
            """SELECT * FROM sessions
            WHERE user_id = $1
            ORDER BY date_start DESC
            LIMIT $2""",
            user_id,
            limit,
        )
        return [dict(r) for r in rows]


class PgTranscriptCacheRepository(TranscriptCacheRepository):
    """PostgreSQL-backed transcript cache repository."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def get_cached(self, source_url: str) -> str | None:
        row = await self._pool.fetch_one(
            """SELECT transcript_text FROM transcript_cache
            WHERE source_url = $1
              AND cached_at + make_interval(secs => ttl_seconds) > NOW()""",
            source_url,
        )
        return row["transcript_text"] if row else None

    async def cache(self, source_url: str, transcript_text: str, ttl_seconds: int = 86400) -> None:
        await self._pool.execute(
            """INSERT INTO transcript_cache (source_url, transcript_text, ttl_seconds, cached_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (source_url) DO UPDATE
                SET transcript_text = EXCLUDED.transcript_text,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    cached_at = NOW()""",
            source_url,
            transcript_text,
            ttl_seconds,
        )

    async def invalidate(self, source_url: str) -> bool:
        result = await self._pool.execute(
            "DELETE FROM transcript_cache WHERE source_url = $1",
            source_url,
        )
        return result.endswith("1")

    async def invalidate_expired(self) -> int:
        result = await self._pool.execute(
            "DELETE FROM transcript_cache WHERE cached_at + make_interval(secs => ttl_seconds) <= NOW()"
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


# ---------------------------------------------------------------------------
# Concrete Neo4j implementation
# ---------------------------------------------------------------------------


class Neo4jGraphRepository(GraphRepository):
    """Neo4j-backed graph repository."""

    def __init__(self, neo4j_conn: Neo4jConnection) -> None:
        self._neo4j = neo4j_conn

    async def upsert_work_node(self, work: dict[str, Any]) -> None:
        await self._neo4j.execute_write(
            """MERGE (w:Work {work_id: $work_id})
            SET w.title = $title,
                w.author = $author,
                w.source_class = $source_class,
                w.publication_year = $publication_year""",
            {
                "work_id": work["work_id"],
                "title": work["title"],
                "author": work["author"],
                "source_class": work["source_class"],
                "publication_year": work["publication_year"],
            },
        )

    async def upsert_chunk_node(self, chunk: dict[str, Any]) -> None:
        # Build SET clause dynamically to include user_id for personal chunks
        params = {
            "chunk_id": chunk["chunk_id"],
            "work_id": chunk["work_id"],
            "text_preview": chunk.get("text_preview", chunk.get("text", "")[:200]),
            "granularity": chunk["granularity"],
            "source_class": chunk["source_class"],
        }
        set_clause = """SET c.work_id = $work_id,
                c.text_preview = $text_preview,
                c.granularity = $granularity,
                c.source_class = $source_class"""

        if "user_id" in chunk:
            set_clause += ",\n                c.user_id = $user_id"
            params["user_id"] = chunk["user_id"]

        # Upsert chunk node AND create PART_OF edge to its Work node.
        # The Work node is matched (not merged) because it must already exist
        # — upsert_work_node is called earlier in the ingestion pipeline.
        await self._neo4j.execute_write(
            f"""MERGE (c:Chunk {{chunk_id: $chunk_id}})
            {set_clause}
            WITH c
            MATCH (w:Work {{work_id: $work_id}})
            MERGE (c)-[:PART_OF]->(w)""",
            params,
        )

    async def create_edge(
        self,
        from_label: str,
        from_key: str,
        from_value: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        props_clause = ""
        params: dict[str, Any] = {
            "from_value": from_value,
            "to_value": to_value,
        }
        if properties:
            props_clause = " SET r += $props"
            params["props"] = properties

        # Build the query with literal labels/types (safe — not user input)
        query = (
            f"MATCH (a:{from_label} {{{from_key}: $from_value}}), "
            f"(b:{to_label} {{{to_key}: $to_value}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)"
            f"{props_clause}"
        )
        await self._neo4j.execute_write(query, params)

    async def get_related_chunks(
        self, chunk_id: str, rel_type: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        results = await self._neo4j.execute_read(
            f"""MATCH (c:Chunk {{chunk_id: $chunk_id}})-[r:{rel_type}]-(related:Chunk)
            RETURN related.chunk_id AS chunk_id,
                   related.work_id AS work_id,
                   related.text_preview AS text_preview,
                   related.granularity AS granularity,
                   related.source_class AS source_class,
                   properties(r) AS rel_props
            LIMIT $limit""",
            {"chunk_id": chunk_id, "limit": limit},
        )
        return results

    async def get_themes_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        results = await self._neo4j.execute_read(
            """MATCH (c:Chunk {chunk_id: $chunk_id})-[:EXPLORES_THEME]->(t:Theme)
            RETURN t.name AS name, t.canonical_name AS canonical_name""",
            {"chunk_id": chunk_id},
        )
        return results

    async def create_user_reflects_on_edge(
        self,
        *,
        reflection_chunk_id: str,
        target_id: str,
        target_type: str,
        target_label: str,
        target_key: str,
        date_created: str | None = None,
    ) -> None:
        """Create a USER_REFLECTS_ON edge from a personal chunk to a target.

        Personal chunks (source_class='personal') connect to captures or themes
        via USER_REFLECTS_ON edges. This relationship represents the user's
        reflection on the target content.

        Args:
            reflection_chunk_id: The chunk_id of the personal reflection chunk.
            target_id: The identifier of the target node.
            target_type: The type of target ('capture' or 'theme').
            target_label: The Neo4j label of the target node (e.g., 'Chunk', 'Theme').
            target_key: The property key on the target node (e.g., 'chunk_id', 'canonical_name').
            date_created: ISO date string for when the reflection was created.
        """
        props: dict[str, Any] = {"target_type": target_type}
        if date_created:
            props["date_created"] = date_created

        await self.create_edge(
            from_label="Chunk",
            from_key="chunk_id",
            from_value=reflection_chunk_id,
            rel_type="USER_REFLECTS_ON",
            to_label=target_label,
            to_key=target_key,
            to_value=target_id,
            properties=props,
        )

    async def get_reflections_for_target(
        self, target_id: str, target_key: str, target_label: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get all personal reflection chunks connected to a target via USER_REFLECTS_ON.

        Args:
            target_id: The identifier of the target node.
            target_key: The property key on the target node.
            target_label: The Neo4j label of the target node.
            limit: Maximum number of results.

        Returns:
            List of reflection chunk data with edge properties.
        """
        results = await self._neo4j.execute_read(
            f"""MATCH (c:Chunk)-[r:USER_REFLECTS_ON]->(t:{target_label} {{{target_key}: $target_id}})
            RETURN c.chunk_id AS chunk_id, c.work_id AS work_id,
                   c.text_preview AS text_preview, c.granularity AS granularity,
                   c.source_class AS source_class, c.user_id AS user_id,
                   r.target_type AS target_type, r.date_created AS date_created
            LIMIT $limit""",
            {"target_id": target_id, "limit": limit},
        )
        return results
