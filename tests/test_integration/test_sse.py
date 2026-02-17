"""Tests for SSE transport configuration.

Tests verify the SSE transport setup without starting a full server.
"""

from __future__ import annotations

from author_library.config import ServerSettings, Settings
from author_library.server import TOOLS, create_server


class TestSSETransportConfiguration:
    """Verify SSE transport configuration and server setup."""

    def test_server_settings_default_transport(self) -> None:
        settings = ServerSettings()
        assert settings.transport == "stdio"

    def test_server_settings_sse_transport(self) -> None:
        settings = ServerSettings(transport="sse", host="0.0.0.0", port=9090)
        assert settings.transport == "sse"
        assert settings.host == "0.0.0.0"
        assert settings.port == 9090

    def test_sse_transport_import(self) -> None:
        """Verify MCP SDK SSE transport is importable."""
        from mcp.server.sse import SseServerTransport

        transport = SseServerTransport("/messages/")
        assert transport is not None

    def test_sse_transport_endpoint_validation(self) -> None:
        """SSE transport should reject full URLs as endpoints."""
        import pytest
        from mcp.server.sse import SseServerTransport

        with pytest.raises(ValueError, match="not a relative path"):
            SseServerTransport("http://localhost/messages/")

    def test_server_has_all_tools_with_sse(self) -> None:
        """Server should register all tools regardless of transport."""
        settings = Settings()
        settings.server.transport = "sse"
        server = create_server(settings)
        assert server is not None
        assert len(TOOLS) == 11

    def test_starlette_available(self) -> None:
        """Verify Starlette is available for SSE transport."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        assert Starlette is not None
        assert Route is not None

    def test_uvicorn_available(self) -> None:
        """Verify uvicorn is available for SSE transport."""
        import uvicorn

        assert uvicorn is not None
