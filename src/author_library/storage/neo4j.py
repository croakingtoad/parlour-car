"""Neo4j graph database connection management.

Provides async driver management, health checks, schema initialization
with constraints and indexes per the collection-librarian chunking-guide §10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import neo4j
import structlog

from author_library.errors import StorageError

if TYPE_CHECKING:
    from author_library.config import DatabaseSettings

log = structlog.get_logger(__name__)

# Schema constraints — ensure uniqueness on key node properties
_CONSTRAINTS = [
    "CREATE CONSTRAINT work_id_unique IF NOT EXISTS FOR (w:Work) REQUIRE w.work_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT theme_name_unique IF NOT EXISTS "
    "FOR (t:Theme) REQUIRE t.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT person_name_unique IF NOT EXISTS "
    "FOR (p:Person) REQUIRE p.canonical_name IS UNIQUE",
]

# Schema indexes for query performance
_INDEXES = [
    "CREATE INDEX chunk_source_class IF NOT EXISTS FOR (c:Chunk) ON (c.source_class)",
    "CREATE INDEX chunk_work_id IF NOT EXISTS FOR (c:Chunk) ON (c.work_id)",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text_preview]",
    # USER_REFLECTS_ON edge indexes for personal source reflections
    # These enable efficient traversal from personal chunks to their reflection targets
    "CREATE INDEX chunk_user_id IF NOT EXISTS FOR (c:Chunk) ON (c.user_id)",
]


class Neo4jConnection:
    """Async Neo4j driver wrapper with schema initialization."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._uri = settings.neo4j_url
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password.get_secret_value()
        self._driver: neo4j.AsyncDriver | None = None

    @property
    def driver(self) -> neo4j.AsyncDriver:
        """Return the driver, raising if not initialized."""
        if self._driver is None:
            raise StorageError("Neo4j driver not initialized — call connect() first")
        return self._driver

    async def connect(self) -> None:
        """Create the async driver and verify connectivity."""
        if self._driver is not None:
            return
        try:
            self._driver = neo4j.AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            await self._driver.verify_connectivity()
            log.info("neo4j_connected", uri=self._uri)
        except Exception as exc:
            self._driver = None
            raise StorageError(
                "Failed to connect to Neo4j",
                context={"uri": self._uri},
                cause=exc,
            ) from exc

    async def close(self) -> None:
        """Gracefully close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            log.info("neo4j_closed")

    async def health_check(self) -> bool:
        """Verify connectivity by running a simple query."""
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 AS n")
                record = await result.single()
                return record is not None and record["n"] == 1
        except Exception:
            log.warning("neo4j_health_check_failed", exc_info=True)
            return False

    async def init_schema(self) -> None:
        """Apply constraints and indexes to the graph database."""
        async with self.driver.session() as session:
            for stmt in _CONSTRAINTS:
                await session.run(stmt)
            for stmt in _INDEXES:
                await session.run(stmt)
            log.info(
                "neo4j_schema_initialized",
                constraints=len(_CONSTRAINTS),
                indexes=len(_INDEXES),
            )

    async def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a write transaction and return result records as dicts."""
        async with self.driver.session() as session:

            async def _work(tx: neo4j.AsyncManagedTransaction) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters or {})
                records: list[dict[str, Any]] = await result.data()
                return records

            return await session.execute_write(_work)

    async def execute_read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a read transaction and return result records as dicts."""
        async with self.driver.session() as session:

            async def _work(tx: neo4j.AsyncManagedTransaction) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters or {})
                records: list[dict[str, Any]] = await result.data()
                return records

            return await session.execute_read(_work)
