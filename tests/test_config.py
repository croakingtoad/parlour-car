"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.config import (
    APIKeySettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    ServerSettings,
    Settings,
    get_settings,
)

if TYPE_CHECKING:
    import pytest


class TestDatabaseSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "DB_POSTGRES_URL",
            "DB_NEO4J_URL",
            "DB_NEO4J_USER",
            "DB_NEO4J_PASSWORD",
        ):
            monkeypatch.delenv(name, raising=False)

        s = DatabaseSettings()
        assert "5432" in s.postgres_url
        assert "7687" in s.neo4j_url
        assert s.neo4j_user == "neo4j"
        # SecretStr: value is hidden
        assert "neo4j_dev" not in repr(s.neo4j_password)
        assert s.neo4j_password.get_secret_value() == "neo4j_dev"


class TestAPIKeySettings:
    def test_defaults(self) -> None:
        s = APIKeySettings()
        assert s.anthropic_api_key.get_secret_value() == ""
        assert s.voyage_api_key is None
        assert s.openai_api_key is None


class TestEmbeddingSettings:
    def test_defaults(self) -> None:
        s = EmbeddingSettings()
        assert s.provider == "voyage"
        assert s.model == "voyage-3-large"
        assert s.dimensions == 1024


class TestLLMSettings:
    def test_defaults(self) -> None:
        s = LLMSettings()
        assert "claude" in s.ingestion_model
        assert "claude" in s.query_model


class TestServerSettings:
    def test_defaults(self) -> None:
        s = ServerSettings()
        assert s.transport == "stdio"
        assert s.port == 8080
        assert s.log_level == "INFO"
        assert s.log_format == "console"


class TestSettings:
    def test_aggregates_all_sections(self) -> None:
        s = Settings()
        assert isinstance(s.database, DatabaseSettings)
        assert isinstance(s.api_keys, APIKeySettings)
        assert isinstance(s.embedding, EmbeddingSettings)
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.server, ServerSettings)

    def test_get_settings_returns_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)
