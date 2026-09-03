"""Tests for configuration management."""

from __future__ import annotations

import pytest  # noqa: TC002  (pytest.MonkeyPatch used as a runtime fixture type)

from author_library.config import (
    APIKeySettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    ServerSettings,
    Settings,
    get_settings,
)


class TestDatabaseSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # tests/conftest.py sets DB_POSTGRES_URL and DB_NEO4J_URL so the suite
        # never touches production, which means reading them here would assert
        # the harness's values rather than the application's defaults. Clear
        # them so this tests what it claims. The app default remains the
        # production graph (7687); only the test harness points elsewhere.
        monkeypatch.delenv("DB_POSTGRES_URL", raising=False)
        monkeypatch.delenv("DB_NEO4J_URL", raising=False)
        s = DatabaseSettings()
        assert "5432" in s.postgres_url
        assert "7687" in s.neo4j_url
        assert s.neo4j_user == "neo4j"
        # SecretStr: value is hidden
        assert "neo4j_dev" not in repr(s.neo4j_password)
        assert s.neo4j_password.get_secret_value() == "neo4j_dev"


class TestAPIKeySettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same reasoning as TestDatabaseSettings.test_defaults: anyone who has
        # sourced .env (i.e. anyone running the suite normally) has these set,
        # so reading the ambient environment tested the shell, not the defaults.
        for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
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
