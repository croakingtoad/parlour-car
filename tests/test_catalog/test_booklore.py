"""Tests for the Booklore metadata resolver."""

from __future__ import annotations

import pytest

from author_library.catalog.booklore import _parse_db_url, resolve_metadata


class TestParseDbUrl:
    def test_full_url_with_scheme(self) -> None:
        result = _parse_db_url("mysql+aiomysql://user:pass@host:3307/mydb")
        assert result == {
            "host": "host",
            "port": 3307,
            "user": "user",
            "password": "pass",
            "db": "mydb",
        }

    def test_mysql_scheme(self) -> None:
        result = _parse_db_url("mysql://user:pass@host:3306/db")
        assert result["user"] == "user"
        assert result["password"] == "pass"

    def test_no_password(self) -> None:
        result = _parse_db_url("mysql+aiomysql://root@localhost:3306/booklore")
        assert result["user"] == "root"
        assert "password" not in result

    def test_defaults(self) -> None:
        result = _parse_db_url("mysql+aiomysql://user@/booklore")
        assert result["host"] == "localhost"
        assert result["port"] == 3306

    def test_url_encoded_password(self) -> None:
        result = _parse_db_url("mysql://user:p%40ss%21@host:3306/db")
        assert result["password"] == "p@ss!"


class TestResolveMetadata:
    @pytest.mark.asyncio
    async def test_missing_aiomysql_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When aiomysql is not installed, resolver returns empty dict."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "aiomysql":
                raise ImportError("no aiomysql")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = await resolve_metadata("/some/file.epub", db_url="mysql://x@y/z")
        assert result == {}

    @pytest.mark.asyncio
    async def test_connection_failure_returns_empty(self) -> None:
        """When DB is unreachable, resolver returns empty dict gracefully."""
        result = await resolve_metadata(
            "/some/file.epub",
            db_url="mysql+aiomysql://nobody:wrong@192.0.2.1:9999/fake",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_empty(self) -> None:
        """File not in Booklore catalog returns empty dict."""
        result = await resolve_metadata("totally-nonexistent-file-abc123.pdf")
        assert result == {}
