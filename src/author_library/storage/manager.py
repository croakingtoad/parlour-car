"""Composite manager owning all storage connections and providing repositories.

StorageManager is the single entry point for the application to interact
with both PostgreSQL and Neo4j, handling lifecycle (connect, migrate, close)
and exposing typed repository accessors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from author_library.storage.migrations.runner import run_migrations
from author_library.storage.neo4j import Neo4jConnection
from author_library.storage.postgres import PostgresPool
from author_library.storage.lessons import LessonRepository
from author_library.storage.repositories import (
    Neo4jGraphRepository,
    PgChunkRepository,
    PgEmbeddingRepository,
    PgSessionRepository,
    PgThematicRepository,
    PgTranscriptCacheRepository,
    PgVoiceProfileRepository,
    PgWorkRepository,
)

if TYPE_CHECKING:
    from author_library.config import DatabaseSettings

log = structlog.get_logger(__name__)


class StorageManager:
    """Owns PostgreSQL pool and Neo4j driver; provides repository instances."""

    def __init__(
        self,
        settings: DatabaseSettings,
        *,
        pg_min_pool: int = 2,
        pg_max_pool: int = 10,
    ) -> None:
        self._pg = PostgresPool(settings, min_size=pg_min_pool, max_size=pg_max_pool)
        self._neo4j = Neo4jConnection(settings)

    # -- Lifecycle -----------------------------------------------------------

    async def connect(
        self, *, run_pg_migrations: bool = True, init_neo4j_schema: bool = True
    ) -> None:
        """Connect to both databases, optionally running migrations and schema init."""
        await self._pg.connect()
        if run_pg_migrations:
            await run_migrations(self._pg)
        await self._neo4j.connect()
        if init_neo4j_schema:
            await self._neo4j.init_schema()
        log.info("storage_manager_ready")

    async def close(self) -> None:
        """Gracefully close all connections."""
        await self._neo4j.close()
        await self._pg.close()
        log.info("storage_manager_closed")

    async def health_check(self) -> dict[str, bool]:
        """Run health checks on both backends."""
        pg_ok = await self._pg.health_check()
        neo4j_ok = await self._neo4j.health_check()
        return {"postgres": pg_ok, "neo4j": neo4j_ok}

    # -- Raw connections (for search, migration runner, etc.) -----------------

    @property
    def pg(self) -> PostgresPool:
        """Access the PostgreSQL pool directly."""
        return self._pg

    @property
    def neo4j(self) -> Neo4jConnection:
        """Access the Neo4j connection directly."""
        return self._neo4j

    # -- Repository accessors ------------------------------------------------

    @property
    def works(self) -> PgWorkRepository:
        """Work/catalog entry repository."""
        return PgWorkRepository(self._pg)

    @property
    def chunks(self) -> PgChunkRepository:
        """Chunk repository."""
        return PgChunkRepository(self._pg)

    @property
    def embeddings(self) -> PgEmbeddingRepository:
        """Embedding repository."""
        return PgEmbeddingRepository(self._pg)

    @property
    def thematic(self) -> PgThematicRepository:
        """Thematic index repository."""
        return PgThematicRepository(self._pg)

    @property
    def voice_profiles(self) -> PgVoiceProfileRepository:
        """Voice profile repository."""
        return PgVoiceProfileRepository(self._pg)

    @property
    def sessions(self) -> PgSessionRepository:
        """Session repository."""
        return PgSessionRepository(self._pg)

    @property
    def transcript_cache(self) -> PgTranscriptCacheRepository:
        """Transcript cache repository."""
        return PgTranscriptCacheRepository(self._pg)

    @property
    def lessons(self) -> LessonRepository:
        """Ingestion lessons repository."""
        return LessonRepository(self._pg)

    @property
    def graph(self) -> Neo4jGraphRepository:
        """Neo4j graph repository."""
        return Neo4jGraphRepository(self._neo4j)
