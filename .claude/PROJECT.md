# Project: Parlour Car (author-insight)

**Codename**: Parlour Car
**Full name**: The Author Library

## Overview
An MCP server for conversational author intelligence — an enriched RAG system that lets users chat with an author through their complete body of work. Goes beyond traditional RAG with multi-granularity genre-aware chunking, contextual embeddings, author voice profiles, Neo4j knowledge graphs with cross-resource passage linking, thematic indexes, and source classification with voice contamination prevention.

## Status
- **Phase**: Planning → Design
- **Created**: 2026-02-15
- **Last Updated**: 2026-02-15
- **Overall Progress**: 0/57 stories complete

## Metrics
| Metric | Count |
|--------|-------|
| Epics | 13 |
| Stories | 57 |
| Estimated Hours | ~480 |
| Estimated Days | ~60 (with parallelism: ~30) |

## Tech Stack
- **Runtime**: Python 3.12+
- **MCP Framework**: Python MCP SDK (`mcp`)
- **LLM (Ingestion)**: Claude Sonnet 4.5 via Anthropic API
- **LLM (Query)**: Claude Sonnet 4.5 / Opus 4.5 (configurable)
- **Embeddings**: Pluggable — Voyage AI, OpenAI, Ollama behind abstract interface
- **Vector Store**: PostgreSQL + pgvector
- **Knowledge Graph**: Neo4j (cross-resource passage linking, multi-hop retrieval)
- **Full-Text Search**: PostgreSQL tsvector + GIN indexes
- **Document Parsing**: `unstructured` library (Python-native)
- **Queue (ingestion)**: Celery + Redis (or arq for lighter weight)
- **Config**: Pydantic Settings + .env
- **Package Manager**: uv

## Architecture Decision Records

### ADR-001: Neo4j over PostgreSQL JSONB for Knowledge Graph
The collection-librarian skill's cross-resource passage linking specification requires multi-hop graph traversal (e.g., primary chunk → ENGAGES_WITH → contextual chunk → THEMATIC_PARALLEL → another chunk). Cypher queries for this are natural; recursive CTEs in PG would be unmaintainable. Neo4j is required.

### ADR-002: Source Classification as Pipeline Gate
Every document must be classified (primary/secondary/contextual/tertiary) BEFORE entering ANY enrichment pipeline. Misclassification contaminates voice profiles and knowledge graphs. The catalog-agent handles this with the collection-librarian's decision tree.

### ADR-003: Pluggable Embedding Provider
Abstract interface with implementations for Voyage AI (voyage-3-large), OpenAI (text-embedding-3-large), and Ollama (nomic-embed-text). Provider selected via config. All chunks stored with provider metadata for re-embedding if needed.

### ADR-004: Python Runtime
Chosen over TypeScript for native access to `unstructured` library, better NLP ecosystem, and strong MCP SDK support. Trade-off: TypeScript has slightly more MCP ecosystem examples.

## Epic Overview

| ID | Epic | Agent Team | Phase | Priority | Status |
|----|------|-----------|-------|----------|--------|
| E001 | Project Foundation | foundation-agent | 1 | P0 | planned |
| E002 | Storage Layer (PG + Neo4j) | storage-agent | 2 | P0 | planned |
| E003 | Catalog & Classification | catalog-agent | 3 | P0 | planned |
| E004 | Document Parser | parser-agent | 2 | P0 | planned |
| E005 | Embedding Abstraction | embedding-agent | 2 | P0 | planned |
| E006 | Genre-Aware Chunking | chunking-agent | 3 | P0 | planned |
| E007 | Author Intelligence Extraction | intelligence-agent | 4 | P0 | planned |
| E008 | Knowledge Graph Engine | graph-agent | 4 | P0 | planned |
| E009 | Retrieval Engine | retrieval-agent | 5 | P0 | planned |
| E010 | MCP Ingestion Tools | tools-agent | 6 | P0 | planned |
| E011 | MCP Query Tools | tools-agent | 6 | P0 | planned |
| E012 | MCP Meta Tools | tools-agent | 6 | P1 | planned |
| E013 | Integration & Production | integration-agent | 7 | P0 | planned |

## Phase Dependency Graph

```
Phase 1: [E001 Foundation]
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
Phase 2: [E002]    [E004]    [E005]
         Storage   Parser   Embeddings
    ┌─────┘│              │
    ▼      ▼              │
Phase 3: [E003]    [E006]◄┘
         Catalog   Chunking
    ┌─────┘│    └────┐
    ▼      ▼         ▼
Phase 4: [E007]    [E008]
         Author    Knowledge
         Intel.    Graph
            └───┬───┘
                ▼
Phase 5:     [E009]
            Retrieval
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
Phase 6: [E010]    [E011]    [E012]
         Ingest    Query     Meta
         Tools     Tools     Tools
            └───────┼───────┘
                    ▼
Phase 7:         [E013]
              Integration
```

## Agent Team Assignments

| Agent | Epics | Worktree Branch | Can Parallelize With |
|-------|-------|-----------------|---------------------|
| foundation-agent | E001 | feature/foundation | (none — runs first) |
| storage-agent | E002 | feature/storage | parser-agent, embedding-agent |
| parser-agent | E004 | feature/parser | storage-agent, embedding-agent |
| embedding-agent | E005 | feature/embeddings | storage-agent, parser-agent |
| catalog-agent | E003 | feature/catalog | chunking-agent |
| chunking-agent | E006 | feature/chunking | catalog-agent |
| intelligence-agent | E007 | feature/intelligence | graph-agent |
| graph-agent | E008 | feature/graph | intelligence-agent |
| retrieval-agent | E009 | feature/retrieval | (none — sequential) |
| tools-agent | E010-E012 | feature/mcp-tools | (none — sequential) |
| integration-agent | E013 | feature/integration | (none — runs last) |

## Key Domain Concepts (from collection-librarian)

### Source Classification Hierarchy
- **Primary**: Works BY the subject author → full enrichment pipeline
- **Secondary**: Works ABOUT the subject author → embeddings + attributed knowledge graph edges only
- **Contextual**: Works the subject author ENGAGES WITH → embeddings + cross-resource link targets
- **Tertiary**: Reference works → metadata only, no content ingestion

### Voice Contamination Prevention
Secondary sources NEVER enter voice profile extraction. Storage isolation (separate namespace/partition), knowledge graph edge typing (ATTRIBUTED_BY_CRITIC vs MAKES_ARGUMENT), and retrieval-time labeling enforce this.

### Cross-Resource Passage Linking (3 tiers)
- **Explicit citation** (high confidence): Detected via footnotes, block quotes, inline citations
- **Implicit engagement** (medium confidence): Detected via terminology fingerprinting
- **Thematic parallel** (low confidence): Detected via semantic similarity > 0.85 threshold

## Key Files
- PRD: `author-library-architecture.md`
- Collection Librarian Skill: `.claude/skills/collection-librarian/`
- Tasks: managed via `td` CLI (`.todos/`)
- Plan: `~/.claude/plans/author-insight-plan.md`

## Session Log
| Date | Session | Completed | Notes |
|------|---------|-----------|-------|
| 2026-02-15 | prd-analyzer | PROJECT.md, td issues, plan | Initial project planning |
