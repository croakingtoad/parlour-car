"""Storage layer for The Author Library.

Provides PostgreSQL (pgvector) and Neo4j connection management,
schema migrations, full-text search, and repository abstractions.
"""

from __future__ import annotations

from author_library.storage.manager import StorageManager
from author_library.storage.neo4j import Neo4jConnection
from author_library.storage.postgres import PostgresPool
from author_library.storage.session_manager import SessionManager

__all__ = [
    "Neo4jConnection",
    "PostgresPool",
    "SessionManager",
    "StorageManager",
]
