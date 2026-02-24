"""Configuration management for The Author Library.

Uses pydantic-settings to load from environment variables and .env files.
Secrets are wrapped in SecretStr and never appear in logs or repr output.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL and Neo4j connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    postgres_url: str = "postgresql://author_library:author_library@localhost:5432/author_library"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4j_dev")


class APIKeySettings(BaseSettings):
    """API keys for external services."""

    model_config = SettingsConfigDict(env_prefix="")

    anthropic_api_key: SecretStr = SecretStr("")
    voyage_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    parlour_api_key: SecretStr | None = None


class EmbeddingSettings(BaseSettings):
    """Embedding provider configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: str = "voyage"
    model: str = "voyage-3-large"
    dimensions: int = 1024


class LLMSettings(BaseSettings):
    """LLM model configuration for ingestion and query."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    ingestion_model: str = "claude-sonnet-4-6"
    query_model: str = "claude-sonnet-4-6"


class SessionSettings(BaseSettings):
    """Session tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="SESSION_")

    timeout_minutes: int = 60  # Inactivity timeout before auto-ending session
    theme_change_gap_minutes: int = 30  # Gap threshold for theme-change auto-end


class RedisSettings(BaseSettings):
    """Redis connection settings for task queue and caching."""

    model_config = SettingsConfigDict(env_prefix="")

    redis_url: str = "redis://localhost:6379"


class ServerSettings(BaseSettings):
    """MCP server transport and runtime settings."""

    model_config = SettingsConfigDict(env_prefix="SERVER_")

    transport: str = "stdio"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    log_format: str = "console"


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database: DatabaseSettings = DatabaseSettings()
    api_keys: APIKeySettings = APIKeySettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    server: ServerSettings = ServerSettings()
    session: SessionSettings = SessionSettings()
    redis: RedisSettings = RedisSettings()


def get_settings() -> Settings:
    """Create and return application settings.

    Each call constructs a fresh Settings instance, reading from the
    environment and .env file. Callers should cache the result if needed.
    """
    return Settings()
