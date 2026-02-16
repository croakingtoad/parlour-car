"""MCP server for The Author Library.

Provides the core server lifecycle — initialization, tool registration,
and stdio transport handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server

from author_library.logging import get_logger, new_correlation_id, setup_logging

if TYPE_CHECKING:
    from mcp.types import Tool

    from author_library.config import Settings

log = get_logger(__name__)


def create_server(settings: Settings) -> Server:
    """Create and configure the MCP server with tool handlers.

    Args:
        settings: Application settings.

    Returns:
        A configured MCP Server instance.
    """
    server = Server("author-library")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """Return the list of available tools.

        Currently returns an empty list — tools will be registered
        as the catalog, retrieval, and intelligence layers are built.
        """
        return []

    return server


async def run_server(settings: Settings) -> None:
    """Start the MCP server with stdio transport.

    Handles full lifecycle: logging setup, server creation,
    startup announcement, and graceful shutdown.

    Args:
        settings: Application settings.
    """
    setup_logging(
        level=settings.server.log_level,
        log_format=settings.server.log_format,
    )

    cid = new_correlation_id()
    log.info(
        "author_library.starting",
        version="0.1.0",
        transport=settings.server.transport,
        correlation_id=cid,
    )

    server = create_server(settings)

    async with stdio_server() as (read_stream, write_stream):
        log.info("author_library.server_ready", transport="stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    log.info("author_library.shutdown_complete")
