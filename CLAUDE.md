## MANDATORY: Use td for Task Management

Run td usage --new-session at conversation start (or after /clear). This tells you what to work on next.

Sessions are automatic (based on terminal/agent context). Optional:
- td session "name" to label the current session
- td session --new to force a new session in the same context

Use td usage -q after first read.

## Project Overview

**The Author Library** (codename: Parlour Car) is an MCP server for conversational author intelligence via enriched RAG. It ingests an author's corpus, builds a knowledge graph, extracts voice profiles, and enables voice-calibrated Q&A — all accessible via 11 MCP tools.

## Architecture

```
MCP Client (Claude Desktop, etc.)
  ↕ stdio or SSE transport
Author Library MCP Server (Python 3.13+)
  ├── Tools Layer (11 MCP tools)
  ├── Retrieval Engine (multi-pass: vector + FTS + graph)
  ├── Intelligence (voice profiles, thematic index, evolution)
  ├── Knowledge Graph (entity extraction, 3-tier passage linking)
  ├── Chunking (6 genre-aware strategies)
  ├── Catalog (source classification gate)
  ├── Parsing (epub, pdf, txt, html, docx)
  ├── Embeddings (Voyage AI, OpenAI, Ollama)
  └── Storage (PostgreSQL + pgvector, Neo4j)
```

## Quick Start

```bash
# Start infrastructure
make dev                           # Docker: PG 16+pgvector, Neo4j 5

# Run the MCP server (stdio transport)
uv run python -m author_library

# Or with SSE transport for remote access
SERVER_TRANSPORT=sse SERVER_PORT=8080 uv run python -m author_library
```

## Development

```bash
make test        # Run all tests
make lint        # Ruff linting
make typecheck   # mypy strict mode
make format      # Auto-format
make dev-down    # Stop Docker services
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_POSTGRES_URL` | `postgresql://author_library:author_library@localhost:5432/author_library` | PostgreSQL connection |
| `DB_NEO4J_URL` | `bolt://localhost:7687` | Neo4j bolt URL |
| `DB_NEO4J_USER` | `neo4j` | Neo4j username |
| `DB_NEO4J_PASSWORD` | `neo4j_dev` | Neo4j password |
| `ANTHROPIC_API_KEY` | (required) | For LLM classification and query |
| `VOYAGE_API_KEY` | (optional) | Voyage AI embeddings |
| `OPENAI_API_KEY` | (optional) | OpenAI embeddings |
| `EMBEDDING_PROVIDER` | `voyage` | Embedding provider: voyage, openai, ollama |
| `EMBEDDING_MODEL` | `voyage-3-large` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensions |
| `SERVER_TRANSPORT` | `stdio` | Transport: stdio or sse |
| `SERVER_HOST` | `0.0.0.0` | SSE bind host |
| `SERVER_PORT` | `8080` | SSE bind port |
| `SERVER_LOG_LEVEL` | `INFO` | Logging level |
| `SERVER_LOG_FORMAT` | `console` | Log format: console or json |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL for task queue (arq) |

## MCP Tools (11)

**Ingestion**: `ingest_book`, `ingest_corpus`
**Query**: `ask_author`, `trace_theme`, `find_quotes`, `compare_ideas`
**Meta**: `list_authors`, `author_bio`, `list_works`, `library_stats`, `health_check`

## Key Architectural Decisions

- **Source classification gates ALL processing** — primary/secondary/contextual/tertiary
- **Voice contamination prevention** is the #1 concern — only primary sources feed voice profiles
- **Cross-resource passage linking** uses 3 confidence tiers: explicit citation, implicit engagement, thematic parallel
- **Genre-aware chunking** with 6 strategies; poems are atomic at meso level
- **Graceful degradation** — if Neo4j is down, retrieval continues with vector-only search
- **PostgreSQL + pgvector** for production-first vector storage (no SQLite phase)
- **Neo4j** for knowledge graph (required by cross-resource passage linking)

## MCP Client Configuration

### Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "author-library": {
      "command": "uv",
      "args": ["run", "python", "-m", "author_library"],
      "cwd": "/path/to/author-insight",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "VOYAGE_API_KEY": "pa-..."
      }
    }
  }
}
```

### Remote (SSE)

```json
{
  "mcpServers": {
    "author-library": {
      "url": "http://your-server:8080/sse"
    }
  }
}
```
