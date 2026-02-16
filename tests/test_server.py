"""Tests for the MCP server."""

from __future__ import annotations

from author_library.config import Settings
from author_library.server import create_server


class TestCreateServer:
    async def test_creates_server_instance(self) -> None:
        settings = Settings()
        server = create_server(settings)
        assert server is not None
        assert server.name == "author-library"

    async def test_list_tools_handler_registered(self) -> None:
        settings = Settings()
        server = create_server(settings)
        assert server.request_handlers is not None
        assert server.name == "author-library"
