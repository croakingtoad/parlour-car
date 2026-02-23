"""MCP server for The Author Library.

Provides the core server lifecycle — initialization, tool registration,
call routing, and stdio transport handling. Manages StorageManager and
EmbeddingProvider lifecycle across the server's session.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from author_library.cache import CacheManager
from author_library.embeddings import CachedEmbeddingProvider, ProviderRegistry
from author_library.logging import get_logger, new_correlation_id, setup_logging
from author_library.queue import TaskQueue
from author_library.storage import StorageManager
from author_library.tools.ingest import handle_ingest_book, handle_ingest_corpus
from author_library.tools.meta import (
    handle_author_bio,
    handle_health_check,
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
    Tool(
        name="health_check",
        description=(
            "Check connectivity and health of all backends: PostgreSQL, Neo4j, "
            "and the embedding provider. Returns per-backend status."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="job_status",
        description=(
            "Check the status of a background job. Returns the current state "
            "(queued, in_progress, complete, failed, not_found) and the result "
            "if the job has completed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job ID returned by an async ingestion operation.",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="ingest_book_async",
        description=(
            "Queue a book ingestion for background processing. Returns a job ID "
            "immediately that can be polled via job_status. Use this for "
            "non-blocking ingestion when you don't need immediate results."
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
                    "description": "Slug identifier for the subject author.",
                },
                "metadata_hints": {
                    "type": "object",
                    "description": "Optional overrides for classification and catalog fields.",
                },
            },
            "required": ["file_path", "subject_author_id"],
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
        cache_mgr: CacheManager | None = _state.get("cache_manager")

        try:
            if name == "ingest_book":
                result = await handle_ingest_book(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
                )
            elif name == "ingest_corpus":
                result = await handle_ingest_corpus(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
                )
            elif name == "ask_author":
                result = await handle_ask_author(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
                )
            elif name == "trace_theme":
                result = await handle_trace_theme(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
                )
            elif name == "find_quotes":
                result = await handle_find_quotes(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
                )
            elif name == "compare_ideas":
                result = await handle_compare_ideas(
                    args,
                    settings=settings,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                    cache_manager=cache_mgr,
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
            elif name == "health_check":
                result = await handle_health_check(
                    args,
                    storage=storage_mgr,
                    embedding_provider=embed_provider,
                )
            elif name == "job_status":
                result = await _handle_job_status(args, state=_state)
            elif name == "ingest_book_async":
                result = await _handle_ingest_book_async(args, state=_state)
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
# Async job handlers
# ---------------------------------------------------------------------------


async def _handle_job_status(args: dict[str, Any], *, state: dict[str, Any]) -> str:
    """Handle the job_status MCP tool call."""
    job_id = args.get("job_id")
    if not job_id:
        return json.dumps({"error": "job_id is required"})

    task_queue = state.get("task_queue")
    if task_queue is None or not task_queue.available:
        return json.dumps({
            "error": "Task queue is not available. Background jobs require Redis.",
            "job_id": job_id,
        })

    from author_library.jobs import get_job_info

    info = await get_job_info(task_queue._pool, job_id)
    return json.dumps(info.to_dict(), indent=2)


async def _handle_ingest_book_async(args: dict[str, Any], *, state: dict[str, Any]) -> str:
    """Handle the ingest_book_async MCP tool call."""
    file_path = args.get("file_path")
    if not file_path:
        return json.dumps({"error": "file_path is required"})

    subject_author_id = args.get("subject_author_id")
    if not subject_author_id:
        return json.dumps({"error": "subject_author_id is required"})

    task_queue = state.get("task_queue")
    if task_queue is None or not task_queue.available:
        return json.dumps({
            "error": (
                "Task queue is not available. Use ingest_book for synchronous "
                "ingestion, or ensure Redis is running."
            ),
        })

    metadata_hints = args.get("metadata_hints") or {}

    job_id = await task_queue.enqueue_ingest_book(
        file_path=file_path,
        subject_author_id=subject_author_id,
        metadata_hints=metadata_hints,
    )

    if job_id is None:
        return json.dumps({"error": "Failed to enqueue job"})

    return json.dumps({
        "job_id": job_id,
        "status": "queued",
        "message": f"Ingestion queued for {file_path}. Poll with job_status tool.",
    })


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def run_server(settings: Settings) -> None:
    """Start the MCP server with the configured transport (stdio or SSE).

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

    # Initialize caching layer
    cache_manager = CacheManager()

    # Initialize embedding provider with cache wrapper
    raw_provider = ProviderRegistry.create(settings)
    embedding_provider = CachedEmbeddingProvider(raw_provider, cache_manager)

    # Initialize task queue (optional — gracefully degrades if Redis unavailable)
    task_queue = TaskQueue()
    await task_queue.connect()

    # Create server and inject dependencies
    server = create_server(settings)
    server._tool_state["storage"] = storage  # type: ignore[attr-defined]
    server._tool_state["embedding_provider"] = embedding_provider  # type: ignore[attr-defined]
    server._tool_state["cache_manager"] = cache_manager  # type: ignore[attr-defined]
    server._tool_state["task_queue"] = task_queue  # type: ignore[attr-defined]

    try:
        if settings.server.transport == "sse":
            await _run_sse(server, settings)
        else:
            await _run_stdio(server)
    finally:
        # Graceful shutdown
        await task_queue.close()
        await embedding_provider.close()
        await storage.close()
        log.info("author_library.shutdown_complete")


async def _run_stdio(server: Server) -> None:
    """Run the server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        log.info("author_library.server_ready", transport="stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def _run_sse(server: Server, settings: Settings) -> None:
    """Run the server with SSE transport via Starlette + uvicorn.

    Creates a Starlette ASGI app with:
      - GET /sse — SSE stream for client connections
      - POST /messages — message endpoint for client commands
      - POST /api/v1/captures — Chrome extension capture endpoint
      - GET /api/v1/captures/status/{job_id} — capture job status
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route

    from author_library.captures.endpoint import handle_capture, handle_capture_status

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Any) -> None:
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages/", endpoint=sse_transport.handle_post_message, methods=["POST"]),
            Route("/api/v1/captures", endpoint=handle_capture, methods=["POST"]),
            Route(
                "/api/v1/captures/status/{job_id:str}",
                endpoint=handle_capture_status,
                methods=["GET"],
            ),
        ],
    )

    # Inject shared state for capture endpoint
    api_key_secret = settings.api_keys.parlour_api_key
    api_key = api_key_secret.get_secret_value() if api_key_secret else ""
    app.state._state = {  # type: ignore[union-attr]
        "api_key": api_key,
        "task_queue": server._tool_state.get("task_queue"),  # type: ignore[attr-defined]
        "storage": server._tool_state.get("storage"),  # type: ignore[attr-defined]
        "settings": settings,
    }

    host = settings.server.host
    port = settings.server.port
    log.info("author_library.server_ready", transport="sse", host=host, port=port)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()
