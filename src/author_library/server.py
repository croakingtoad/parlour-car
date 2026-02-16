"""MCP server for The Author Library.

Provides the core server lifecycle — initialization, tool registration,
call routing, and stdio transport handling. Manages StorageManager and
EmbeddingProvider lifecycle across the server's session.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from author_library.embeddings import ProviderRegistry
from author_library.logging import get_logger, new_correlation_id, setup_logging
from author_library.storage import StorageManager
from author_library.tools.ingest import handle_ingest_book, handle_ingest_corpus
from author_library.tools.meta import (
    handle_author_bio,
    handle_library_stats,
    handle_list_authors,
    handle_list_works,
)
from author_library.tools.query import (
    handle_ask_author,
    handle_compare_ideas,
    handle_find_quotes,
    handle_trace_theme,
)

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="ingest_book",
        description=(
            "Ingest a single work into the author library. Parses the document, "
            "classifies its source type, chunks it by genre, generates embeddings, "
            "extracts entities, and creates passage links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file (epub, pdf, txt, html, docx).",
                },
                "subject_author_id": {
                    "type": "string",
                    "description": "Slug identifier for the subject author (e.g. 'cs-lewis').",
                },
                "metadata_hints": {
                    "type": "object",
                    "description": (
                        "Optional overrides for classification and catalog fields. "
                        "Keys may include: source_class, genre_tags, work_type, "
                        "publication_year, publisher, etc."
                    ),
                },
            },
            "required": ["file_path", "subject_author_id"],
        },
    ),
    Tool(
        name="ingest_corpus",
        description=(
            "Bulk-ingest multiple works from a directory or file list. After all "
            "works are processed, runs cross-work analysis including thematic index "
            "generation, voice profile extraction, and thematic evolution analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory containing documents to ingest.",
                },
                "file_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit list of file paths to ingest.",
                },
                "subject_author_id": {
                    "type": "string",
                    "description": "Slug identifier for the subject author.",
                },
                "metadata_hints": {
                    "type": "object",
                    "description": "Shared metadata hints applied to all works.",
                },
            },
            "required": ["subject_author_id"],
        },
    ),
    Tool(
        name="ask_author",
        description=(
            "Ask a question and receive a voice-calibrated response drawing from "
            "the author's corpus. Uses multi-pass retrieval (vector + full-text + "
            "graph expansion) and generates responses in the author's distinctive voice."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask about/to the author.",
                },
                "author_id": {
                    "type": "string",
                    "description": "Slug identifier for the subject author.",
                },
                "works_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of work IDs to limit retrieval.",
                },
                "response_style": {
                    "type": "string",
                    "enum": ["conversational", "academic", "devotional", "lecture"],
                    "description": "Tone and style for the generated response.",
                },
            },
            "required": ["question", "author_id"],
        },
    ),
    Tool(
        name="trace_theme",
        description=(
            "Trace a theme chronologically across an author's works. Shows how "
            "the author's treatment of the theme develops over time, including "
            "contextual engagement passages from sources the author interacts with."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "theme_name": {
                    "type": "string",
                    "description": "Name of the theme to trace (e.g. 'Sacramental Imagination').",
                },
                "author_id": {
                    "type": "string",
                    "description": "Optional author slug to limit the trace.",
                },
            },
            "required": ["theme_name"],
        },
    ),
    Tool(
        name="find_quotes",
        description=(
            "Search for specific passages or quotations using combined phrase "
            "matching and semantic vector search. Returns exact quotes with "
            "work/chapter citations and cross-resource links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (exact phrase or semantic description).",
                },
                "author_id": {
                    "type": "string",
                    "description": "Optional author slug to limit results.",
                },
                "source_class_filter": {
                    "type": "string",
                    "enum": ["primary", "secondary", "contextual", "tertiary"],
                    "description": "Optional filter by source class.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10).",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="compare_ideas",
        description=(
            "Compare how multiple authors treat a topic. Requires at least two "
            "authors loaded in the library. Returns side-by-side thematic treatment "
            "summaries with sample passages."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or theme to compare across authors.",
                },
                "author_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "List of author slug identifiers to compare.",
                },
            },
            "required": ["topic", "author_ids"],
        },
    ),
    Tool(
        name="list_authors",
        description=(
            "List all authors in the library with work counts, source class "
            "breakdowns, and ingestion statistics."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="author_bio",
        description=(
            "Get a biographical and stylistic summary for an author, drawn from "
            "their voice profile, thematic index, and corpus statistics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "Slug identifier for the author.",
                },
            },
            "required": ["author_id"],
        },
    ),
    Tool(
        name="list_works",
        description=(
            "List the works catalog for an author, optionally filtered by "
            "source class. Includes metadata, genre tags, and source-class-specific fields."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "Slug identifier for the author.",
                },
                "source_class": {
                    "type": "string",
                    "enum": ["primary", "secondary", "contextual", "tertiary"],
                    "description": "Optional filter by source class.",
                },
            },
            "required": ["author_id"],
        },
    ),
    Tool(
        name="library_stats",
        description=(
            "Get collection-wide statistics: works ingested, chunks by granularity, "
            "graph node/edge counts, embedding coverage, and source class breakdown."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(settings: Settings) -> Server:
    """Create and configure the MCP server with all tool handlers.

    Args:
        settings: Application settings.

    Returns:
        A configured MCP Server instance.
    """
    server = Server("author-library")

    # State holders initialized lazily in run_server
    _state: dict[str, Any] = {}

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """Return all registered tools with their MCP schemas."""
        return TOOLS

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        """Route a tool call to the appropriate handler."""
        args = arguments or {}
        storage_mgr: StorageManager = _state["storage"]
        embed_provider: EmbeddingProvider = _state["embedding_provider"]

        try:
            if name == "ingest_book":
                result = await handle_ingest_book(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "ingest_corpus":
                result = await handle_ingest_corpus(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "ask_author":
                result = await handle_ask_author(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "trace_theme":
                result = await handle_trace_theme(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "find_quotes":
                result = await handle_find_quotes(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "compare_ideas":
                result = await handle_compare_ideas(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "list_authors":
                result = await handle_list_authors(args, storage=storage_mgr)
            elif name == "author_bio":
                result = await handle_author_bio(
                    args, settings=settings, storage=storage_mgr
                )
            elif name == "list_works":
                result = await handle_list_works(args, storage=storage_mgr)
            elif name == "library_stats":
                result = await handle_library_stats(args, storage=storage_mgr)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:
            log.error("tool_call_failed", tool=name, error=str(exc))
            result = json.dumps({
                "error": str(exc),
                "error_type": type(exc).__name__,
            })

        return [TextContent(type="text", text=result)]

    # Expose state dict for run_server to populate
    server._tool_state = _state  # type: ignore[attr-defined]

    return server


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def run_server(settings: Settings) -> None:
    """Start the MCP server with stdio transport.

    Handles full lifecycle: logging setup, storage/embedding initialization,
    server creation, and graceful shutdown of all connections.

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

    # Initialize storage connections
    storage = StorageManager(settings.database)
    await storage.connect()

    # Initialize embedding provider
    embedding_provider = ProviderRegistry.create(settings)

    # Create server and inject dependencies
    server = create_server(settings)
    server._tool_state["storage"] = storage  # type: ignore[attr-defined]
    server._tool_state["embedding_provider"] = embedding_provider  # type: ignore[attr-defined]

    try:
        async with stdio_server() as (read_stream, write_stream):
            log.info("author_library.server_ready", transport="stdio")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        # Graceful shutdown
        await embedding_provider.close()
        await storage.close()
        log.info("author_library.shutdown_complete")
