# The Author Library

An MCP server for conversational author intelligence via enriched RAG. Ingest an author's corpus, build a knowledge graph with voice profiles and thematic indexes, then have voice-calibrated conversations grounded in the actual texts.

## Quick Start

```bash
# Prerequisites: Docker, Python 3.13+, uv

# Start databases
make dev

# Set required API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export VOYAGE_API_KEY="pa-..."   # or use EMBEDDING_PROVIDER=ollama for local

# Run the MCP server
uv run python -m author_library
```

## MCP Tools

### Ingestion

| Tool | Description |
|------|-------------|
| `ingest_book` | Ingest a single work: parse, classify, chunk, embed, extract entities, create passage links |
| `ingest_corpus` | Bulk-ingest a directory of works with cross-work analysis (thematic index, voice profile, evolution) |

### Query

| Tool | Description |
|------|-------------|
| `ask_author` | Voice-calibrated Q&A using multi-pass retrieval (vector + FTS + graph expansion) |
| `trace_theme` | Chronological theme tracing across works with argument evolution |
| `find_quotes` | Combined phrase matching and semantic vector search with citations |
| `compare_ideas` | Cross-author thematic comparison with sample passages |

### Meta

| Tool | Description |
|------|-------------|
| `list_authors` | All authors with work counts and source class breakdowns |
| `author_bio` | Biographical summary from voice profile and corpus stats |
| `list_works` | Works catalog with metadata and source-class-specific fields |
| `library_stats` | Collection statistics: works, chunks, graph, embeddings, coverage |
| `health_check` | Backend connectivity test: PostgreSQL, Neo4j, embedding provider |

## Architecture

```
MCP Client (Claude Desktop, etc.)
  ↕ stdio or SSE transport
Author Library MCP Server
  ├── Tools (11 MCP tool handlers)
  ├── Retrieval (multi-pass: vector + FTS + graph, RRF fusion)
  ├── Intelligence (voice profiles, thematic index, evolution analysis)
  ├── Knowledge Graph (entity extraction, 3-tier passage linking)
  ├── Chunking (6 genre-aware strategies: scholarly, poetry, sermons, letters, blogs, interviews)
  ├── Catalog (source classification gate: primary/secondary/contextual/tertiary)
  ├── Parsing (epub, pdf, txt, html, docx)
  ├── Embeddings (pluggable: Voyage AI, OpenAI, Ollama)
  ├── Cache (in-memory LRU with TTL, invalidation on ingestion)
  └── Storage (PostgreSQL 16 + pgvector, Neo4j 5)
```

### Source Classification

Every ingested work is classified into one of four source classes that gate all downstream processing:

- **Primary**: The subject author's own works — full enrichment, voice profile extraction
- **Secondary**: Critical/scholarly works about the author — embeddings + attributed graph edges, NO voice profile
- **Contextual**: Works the author engages with — embeddings + cross-resource passage links
- **Tertiary**: Reference works — metadata only, no content processing

Voice contamination prevention (never mixing secondary material into the author's voice) is the #1 architectural concern.

### Retrieval Pipeline

Multi-pass retrieval with question classification:

1. **Pass 1**: Hybrid vector + full-text search, fused via Reciprocal Rank Fusion
2. **Pass 2**: Graph expansion — engagement chains, theme subgraphs, argument development
3. **Pass 3**: Supporting micro-chunk evidence
4. **Consensus boost**: Chunks appearing in multiple passes get score boosts

Graceful degradation: if Neo4j is unavailable, retrieval continues with vector-only search.

## Infrastructure

```bash
# Docker Compose provides:
# - PostgreSQL 16 with pgvector on port 5432
# - Neo4j 5 Community on ports 7687 (bolt) and 7474 (browser)

make dev         # Start
make dev-down    # Stop
```

## Configuration

All configuration via environment variables (with `.env` file support):

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_POSTGRES_URL` | `postgresql://author_library:author_library@localhost:5432/author_library` | PostgreSQL connection string |
| `DB_NEO4J_URL` | `bolt://localhost:7687` | Neo4j bolt URL |
| `DB_NEO4J_USER` | `neo4j` | Neo4j username |
| `DB_NEO4J_PASSWORD` | `neo4j_dev` | Neo4j password |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key for LLM operations |
| `VOYAGE_API_KEY` | *(optional)* | Voyage AI embeddings |
| `OPENAI_API_KEY` | *(optional)* | OpenAI embeddings |
| `EMBEDDING_PROVIDER` | `voyage` | Provider: `voyage`, `openai`, `ollama` |
| `EMBEDDING_MODEL` | `voyage-3-large` | Model name |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensions |
| `LLM_INGESTION_MODEL` | `claude-sonnet-4-5-20250929` | Model for ingestion classification |
| `LLM_QUERY_MODEL` | `claude-sonnet-4-5-20250929` | Model for query responses |
| `SERVER_TRANSPORT` | `stdio` | Transport: `stdio` or `sse` |
| `SERVER_HOST` | `0.0.0.0` | SSE transport bind host |
| `SERVER_PORT` | `8080` | SSE transport bind port |
| `SERVER_LOG_LEVEL` | `INFO` | Log level |
| `SERVER_LOG_FORMAT` | `console` | Format: `console` or `json` |

## MCP Client Setup

### Claude Desktop (stdio — recommended)

Add to your Claude Desktop MCP config:

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

### Remote Access (SSE)

Start the server with SSE transport:

```bash
SERVER_TRANSPORT=sse SERVER_PORT=8080 uv run python -m author_library
```

Connect via SSE URL:

```json
{
  "mcpServers": {
    "author-library": {
      "url": "http://your-server:8080/sse"
    }
  }
}
```

## Development

```bash
make test        # Run all tests (562+ tests)
make lint        # Ruff linting
make typecheck   # mypy strict mode
make format      # Auto-format code
make test-cov    # Tests with coverage report
```

## Testing

- **Unit tests** (`tests/test_*`): Run without infrastructure, test individual modules
- **Integration tests** (`tests/test_integration/`): Require Docker databases (`make dev`)
  - `test_e2e.py`: Full pipeline tests (ingest → query)
  - `test_source_isolation.py`: Voice contamination prevention
  - `test_cache.py`: Cache hit/miss/eviction/invalidation
  - `test_sse.py`: SSE transport configuration
- LLM-dependent tests gated behind `ANTHROPIC_API_KEY`

## License

Proprietary.
