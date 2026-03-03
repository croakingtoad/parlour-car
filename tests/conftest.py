"""Root test configuration — ensures tests NEVER touch the production database.

Sets DB_POSTGRES_URL to a dedicated test database before any
DatabaseSettings instances are constructed. Every sub-conftest that
uses DatabaseSettings() will automatically receive the test DB URL.

Override defaults with environment variables:
    TEST_POSTGRES_URL  (default: author_library_test on localhost)
    TEST_NEO4J_URL     (default: bolt://localhost:7687)
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CRITICAL: Override database URLs BEFORE any pydantic-settings model loads.
# DatabaseSettings uses env_prefix="DB_", so setting DB_POSTGRES_URL here
# ensures every DatabaseSettings() construction points at the test database.
# ---------------------------------------------------------------------------

_TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://author_library:author_library@localhost:5432/author_library_test",
)
_TEST_NEO4J_URL = os.environ.get(
    "TEST_NEO4J_URL",
    "bolt://localhost:7687",
)

os.environ["DB_POSTGRES_URL"] = _TEST_POSTGRES_URL
os.environ["DB_NEO4J_URL"] = _TEST_NEO4J_URL
