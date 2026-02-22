# Parlour Car — Orchestration Plan

**Codename**: Parlour Car
**Full name**: The Author Library
**Repo**: author-insight

---

## Execution Strategy

This plan is designed for an orchestrator using agent teams decomposed by architectural layer. Each agent works in its own git worktree branch. Phases gate parallelism — agents within a phase run concurrently; phases run sequentially.

**Total agents**: 11 (not all active simultaneously)
**Max concurrent agents**: 3 (Phases 2, 3, 4)
**Recommended backend**: tmux (for visibility) or in-process (for simplicity)

---

## Phase 1: Foundation (Sequential — 1 agent)

### Agent: foundation-agent
**Branch**: `feature/foundation`
**Duration**: ~1 day

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Python project scaffolding | 4h | pyproject.toml with uv, src/ layout, dependencies (mcp, anthropic, asyncpg, neo4j, unstructured, pydantic-settings) |
| (auto) | MCP server skeleton | 4h | Python MCP SDK setup with stdio transport, basic server lifecycle, health endpoint |
| (auto) | Configuration management | 3h | Pydantic Settings model: DB URLs, API keys, embedding provider selection, LLM model config. .env support |
| (auto) | Logging & error framework | 2h | structlog setup, consistent error types, request correlation IDs |
| (auto) | Dev environment setup | 3h | Docker Compose for PostgreSQL+pgvector and Neo4j, Makefile/justfile, pre-commit hooks |

**Exit criteria**: `uv run python -m author_library` starts MCP server, connects to PG and Neo4j, responds to list_tools.

---

## Phase 2: Infrastructure (Parallel — 3 agents)

### Agent: storage-agent
**Branch**: `feature/storage`
**Duration**: ~3 days
**Depends on**: E001

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | PostgreSQL schema & migrations | 6h | Tables: authors, works, chunks, chunk_embeddings, thematic_entries, voice_profiles. pgvector extension, HNSW indexes. Migration framework (alembic or simple SQL scripts) |
| (auto) | PostgreSQL connection management | 4h | asyncpg connection pool, health checks, transaction management, query helpers |
| (auto) | Full-text search setup | 4h | tsvector columns on chunks table, GIN indexes, ts_rank search function, exact phrase matching |
| (auto) | Neo4j schema & connection | 6h | Node types (Work, Chunk, Theme, Concept, Person, Argument), edge types per collection-librarian spec (ENGAGES_WITH, THEMATIC_PARALLEL, MAKES_ARGUMENT, ATTRIBUTED_BY_CRITIC, etc.), indexes per chunking-guide Section 10. Neo4j Python driver with async support |
| (auto) | Storage abstraction layer | 4h | Repository pattern interfaces for PG and Neo4j. Clean separation so retrieval engine doesn't touch raw SQL/Cypher directly |

### Agent: parser-agent
**Branch**: `feature/parser`
**Duration**: ~3 days
**Depends on**: E001

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | EPUB parser | 6h | Extract structured document tree (chapters, sections, paragraphs, footnotes) using unstructured or ebooklib. Output JSON AST per architecture spec |
| (auto) | PDF parser | 6h | Handle both born-digital and OCR'd PDFs. OCR quality detection. Structural extraction (headings, paragraphs). Use unstructured library |
| (auto) | DOCX & plain text parsers | 4h | DOCX via python-docx or unstructured. Plain text with heuristic heading detection |
| (auto) | HTML parser | 3h | HTML via BeautifulSoup or unstructured. Handle web articles, blog posts |
| (auto) | Document tree data model | 5h | Pydantic models for the structured document tree (Book → Chapter → Section → Paragraph → Footnote). Metadata extraction (title, author, pub date, TOC). Format-agnostic output all parsers conform to |

### Agent: embedding-agent
**Branch**: `feature/embeddings`
**Duration**: ~2 days
**Depends on**: E001

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Embedding provider interface | 4h | Abstract base class: embed_text(text) → vector, embed_batch(texts) → vectors. Dimension property. Provider metadata for stored embeddings |
| (auto) | Voyage AI provider | 3h | voyage-3-large implementation via Voyage API. Batch support, rate limiting, retry logic |
| (auto) | OpenAI provider | 3h | text-embedding-3-large implementation via OpenAI API. Batch support |
| (auto) | Ollama/local provider | 3h | nomic-embed-text via Ollama HTTP API. Local inference, no API key needed |
| (auto) | Provider configuration & registry | 3h | Config-driven provider selection. Provider registry pattern. Dimension validation against stored embeddings |

---

## Phase 3: Core Pipeline (Parallel — 2 agents)

### Agent: catalog-agent
**Branch**: `feature/catalog`
**Duration**: ~3 days
**Depends on**: E002 (storage)

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Catalog metadata schema | 6h | Implement full catalog-schema.md as Pydantic models. All 4 source classes with their specific fields. Validation rules (work_id format, source_class_note min length, genre_tags controlled vocab) |
| (auto) | Source classification engine | 8h | Decision tree from collection-librarian SKILL.md §2. Signals: authorship attribution, title analysis, publication context, content sampling, bibliographic cross-ref. LLM-assisted classification with confidence scoring. Default-to-secondary safety rule |
| (auto) | Classification pipeline integration | 4h | Every document entering the system passes through classification BEFORE any enrichment. Gate that routes primary/secondary/contextual/tertiary to appropriate downstream pipelines |
| (auto) | Mixed-authorship handling | 4h | Detect edited collections with subject-author chapters. Extract primary chapters from secondary containers. Interview Q&A splitting (interviewer=secondary, responses=primary-adjacent). Per classification-examples.md |

### Agent: chunking-agent
**Branch**: `feature/chunking`
**Duration**: ~3 days
**Depends on**: E004 (parser)

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Chunking framework & models | 4h | Chunk data model (id, text, granularity, work, chapter, position, source_class, metadata). Three granularity tiers: macro (500-1500 words), meso (150-500), micro (30-200). Chunk relationships (parent-child across granularities) |
| (auto) | Scholarly prose chunking | 6h | Per chunking-guide §2. Macro: chapter summaries. Meso: section/argument boundaries. Micro: paragraphs. Special handling: footnotes attached to parent, block quotations tagged with quoted author, bibliography extracted as metadata |
| (auto) | Poetry & sermon chunking | 5h | Poetry (§3): poems atomic at meso, stanza-level micro only for 40+ lines. Never split a poem. Sermons/lectures (§4): movement-level meso, occasion/venue in annotations |
| (auto) | Letters, blogs, interviews chunking | 5h | Letters (§5): individual letter = meso, recipient tagged. Blogs (§6): typically single meso. Interviews (§7): Q&A pairs as meso, interviewer questions tagged secondary, author responses primary-adjacent |
| (auto) | Contextual annotation engine | 6h | LLM-generated annotations prepended to each chunk before embedding. Three templates from chunking-guide §9: PRIMARY, SECONDARY, CONTEXTUAL. Include: work title, year, chapter, topic, positioning, preceding/following context. Source classification markers |

---

## Phase 4: Intelligence (Parallel — 2 agents)

### Agent: intelligence-agent
**Branch**: `feature/intelligence`
**Duration**: ~4 days
**Depends on**: E003 (catalog), E005 (embeddings), E006 (chunking)

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Voice profile extraction | 8h | LLM processes primary corpus to generate structured voice profile: register, sentence patterns, vocabulary tendencies, rhetorical moves, characteristic phrases, humor style, example passages. JSON schema per architecture spec §3a. Only primary sources with voice_profile_eligible=true |
| (auto) | Thematic index generation | 8h | Pre-computed theme→appearances map across corpus. Per architecture spec §3c: theme, author_stance, appearances by work with chapter list and treatment summary, related_themes, key_passages. Chronological ordering for evolution tracking |
| (auto) | Terminology normalization | 6h | Controlled vocabulary mapping variant terms to canonical concepts. Authors use different words for same concept across works (or same word with shifting meaning). LLM-assisted with human review queue for ambiguous cases |
| (auto) | Cross-work thematic evolution | 6h | Chronological analysis of how author's treatment of each theme develops across their corpus. Create DEVELOPS_FROM edges between related arguments in different works. Flag explicit self-reflection (author writing about their own earlier work = gold for evolution tracking) |
| (auto) | Voice profile CRUD & updates | 4h | Store, retrieve, update voice profiles. Re-extraction when new primary works are ingested. Profile versioning |

### Agent: graph-agent
**Branch**: `feature/graph`
**Duration**: ~4 days
**Depends on**: E002 (storage), E003 (catalog)

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Entity extraction pipeline | 8h | LLM extracts entities from chunks: Themes, Arguments, Concepts, Persons. Create Neo4j nodes. Edge types: EXPLORES_THEME, MAKES_ARGUMENT (primary only), ATTRIBUTED_BY_CRITIC (secondary only), REFERENCES_PERSON, CONCEPT_USED_IN |
| (auto) | Cross-resource passage linking — explicit citations | 8h | Per chunking-guide §10. Parse footnotes, endnotes, inline citations from primary sources. Match cited works against contextual sources in collection. Create ENGAGES_WITH edges with link_type: "explicit_citation", confidence: "high" |
| (auto) | Cross-resource passage linking — implicit engagement | 8h | Terminology fingerprinting. Match controlled vocabulary terms back to origin in contextual sources. Create ENGAGES_WITH edges with link_type: "implicit_engagement", confidence: "medium". Handle ambiguous matches (common terms used by many authors) |
| (auto) | Cross-resource passage linking — thematic parallels | 6h | Semantic similarity between primary and contextual chunks sharing canonical themes. Threshold 0.85. Create THEMATIC_PARALLEL edges with confidence: "low". Filter out pairs that already have ENGAGES_WITH edges. Exploratory framing only |
| (auto) | Graph query helpers | 4h | Cypher query templates for common patterns: multi-hop traversal, theme-scoped subgraphs, author intellectual network, cross-work argument chains. Optimized with Neo4j indexes per chunking-guide §10 |

---

## Phase 5: Retrieval (Sequential — 1 agent)

### Agent: retrieval-agent
**Branch**: `feature/retrieval`
**Duration**: ~4 days
**Depends on**: E002, E005, E007, E008

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | Vector similarity search | 4h | pgvector HNSW search across chunk embeddings. Filter by source_class, author, work. Top-k retrieval with score thresholds |
| (auto) | Full-text / BM25 search | 4h | PostgreSQL tsvector search for exact phrases and keyword matching. Quote lookup. Rank fusion ready |
| (auto) | Hybrid retrieval (vector + FTS fusion) | 4h | Reciprocal Rank Fusion or similar score merging. Weight tuning between semantic and keyword results |
| (auto) | Graph-augmented retrieval | 6h | Follow Neo4j edges from retrieved chunks. Expand context: related themes, ENGAGES_WITH targets (contextual source passages), DEVELOPS_FROM chains. Source-class-aware expansion (never present secondary as primary) |
| (auto) | Multi-pass retrieval orchestration | 8h | Per architecture spec ask_author flow. Pass 1: initial vector + graph. Pass 2: expand via graph edges (related themes, arguments). Pass 3: pull supporting quotes from micro chunks. Question type classification drives retrieval strategy |
| (auto) | Context assembly & voice calibration | 6h | Combine retrieved chunks + graph context + summaries into LLM context window. Always include voice profile. Thematic summaries for relevant themes. Voice calibration system prompt. Citation metadata attachment |

---

## Phase 6: MCP Tools (Sequential — 1 agent, 3 epics)

### Agent: tools-agent
**Branch**: `feature/mcp-tools`
**Duration**: ~5 days
**Depends on**: E003, E004, E006, E007, E008, E009

| td ID | Story | Epic | Est. | Description |
|-------|-------|------|------|-------------|
| (auto) | ingest_book tool | E010 | 8h | Full pipeline: parse → classify → chunk → annotate → embed → index → extract entities → create passage links. Progress reporting. Error handling with partial rollback |
| (auto) | ingest_corpus tool | E010 | 8h | Bulk multi-work ingestion. Cross-work analysis after all works processed (thematic evolution, voice profile generation). Ingestion queue with progress tracking |
| (auto) | Ingestion pipeline orchestrator | E010 | 6h | Coordinates all pipeline stages. Routes by source class (primary=full enrichment, secondary=embeddings+attributed graph, contextual=embeddings+link targets, tertiary=metadata only). Idempotent re-ingestion |
| (auto) | ask_author tool | E011 | 8h | Primary conversational tool. Question classification → multi-pass retrieval → context assembly → voice-calibrated generation → citation attachment. Response styles: conversational, academic, devotional, lecture. Works filter option |
| (auto) | trace_theme tool | E011 | 6h | Trace theme across author's works chronologically. Pull thematic index + relevant chunks + ENGAGES_WITH contextual passages showing source evolution alongside author's thinking |
| (auto) | find_quotes tool | E011 | 4h | Full-text + vector search for specific passages. Return exact quotes with work/chapter/page citations. Include cross-resource links if quotes contain explicit citations |
| (auto) | compare_ideas tool | E011 | 6h | Compare how 2+ authors treat same topic. Requires multiple author libraries loaded. Cross-author thematic index comparison |
| (auto) | list_authors / author_bio / list_works | E012 | 4h | Meta tools for library navigation. Author listing with stats, biographical summary from voice profile, works catalog with ingestion status |
| (auto) | library_stats tool | E012 | 3h | Collection statistics: works ingested, chunks by granularity, graph node/edge counts, embedding coverage, source class breakdown |

---

## Phase 7: Integration & Production (Sequential — 1 agent)

### Agent: integration-agent
**Branch**: `feature/integration`
**Duration**: ~4 days
**Depends on**: E010, E011, E012

| td ID | Story | Est. | Description |
|-------|-------|------|-------------|
| (auto) | End-to-end integration tests | 8h | Full pipeline test: ingest a real book (public domain, e.g., Dickens) → verify chunks, embeddings, graph, thematic index → ask_author query → verify grounded response with citations. Secondary source test: verify voice contamination prevention |
| (auto) | SSE transport for remote access | 6h | Add SSE transport alongside stdio. Configurable via env var. Enable remote MCP clients |
| (auto) | Performance & caching | 6h | Query result caching (Redis or in-memory LRU). Embedding batch optimization. Neo4j query plan optimization. Connection pool tuning |
| (auto) | Error handling, resilience & monitoring | 6h | Graceful degradation (Neo4j down → vector-only retrieval). Retry policies for LLM/embedding API calls. Health endpoints. Structured logging with metrics |
| (auto) | Deployment guide & CLAUDE.md | 4h | Docker Compose for full stack. Environment variable documentation. CLAUDE.md with project conventions. MCP client configuration examples |

---

## Dependency Map (td issue IDs)

The orchestrator MUST enforce this dependency graph. Issues within a phase can run in parallel; phases are sequential gates.

### Phase 1: Foundation (no dependencies — start here)
```
td-0b8de4  [E001] Python project scaffolding         ← START HERE
td-ee7ebd  [E001] MCP server skeleton                ← depends on td-0b8de4
td-515027  [E001] Configuration management            ← depends on td-0b8de4
td-49b53c  [E001] Logging & error framework           ← depends on td-0b8de4
td-b690f9  [E001] Dev environment setup               ← depends on td-0b8de4
```
**Gate**: MCP server starts, connects to PG+Neo4j via Docker Compose

### Phase 2: Infrastructure (depends on ALL of Phase 1)
```
STORAGE-AGENT:
td-fc27eb  [E002] PostgreSQL connection management    ← depends on td-b690f9
td-413485  [E002] PostgreSQL schema & migrations      ← depends on td-fc27eb
td-743ae2  [E002] Full-text search setup              ← depends on td-413485
td-2a6e2f  [E002] Neo4j schema & connection           ← depends on td-b690f9
td-605b26  [E002] Storage abstraction layer           ← depends on td-413485, td-2a6e2f

PARSER-AGENT (parallel with storage):
td-6815c2  [E004] Document tree data model            ← depends on td-0b8de4
td-0de79c  [E004] EPUB parser                         ← depends on td-6815c2
td-4eba13  [E004] PDF parser                          ← depends on td-6815c2
td-34ee2a  [E004] DOCX & plain text parsers           ← depends on td-6815c2
td-d130d5  [E004] HTML parser                         ← depends on td-6815c2

EMBEDDING-AGENT (parallel with storage and parser):
td-920d0a  [E005] Embedding provider interface        ← depends on td-0b8de4
td-4f826c  [E005] Voyage AI provider                  ← depends on td-920d0a
td-99457a  [E005] OpenAI provider                     ← depends on td-920d0a
td-a09a74  [E005] Ollama/local provider               ← depends on td-920d0a
td-6de747  [E005] Provider config & registry          ← depends on td-920d0a
```
**Gate**: PG tables exist, Neo4j schema created, EPUB parser extracts chapters, embeddings work

### Phase 3: Core Pipeline (depends on Phase 2)
```
CATALOG-AGENT:
td-984a4d  [E003] Catalog metadata schema             ← depends on td-413485 (PG schema)
td-a7b2e3  [E003] Source classification engine        ← depends on td-984a4d
td-6d479f  [E003] Classification pipeline integration ← depends on td-a7b2e3
td-40c122  [E003] Mixed-authorship handling           ← depends on td-a7b2e3

CHUNKING-AGENT (parallel with catalog):
td-ed5a4d  [E006] Chunking framework & models         ← depends on td-6815c2 (doc model)
td-44a7f2  [E006] Scholarly prose chunking            ← depends on td-ed5a4d
td-5d48a4  [E006] Poetry & sermon chunking            ← depends on td-ed5a4d
td-944a00  [E006] Letters, blogs, interviews chunking ← depends on td-ed5a4d (P1)
td-8fabc8  [E006] Contextual annotation engine        ← depends on td-ed5a4d
```
**Gate**: Classification pipeline gates documents by source class, chunks generated with annotations

### Phase 4: Intelligence (depends on Phase 3)
```
INTELLIGENCE-AGENT:
td-f53821  [E007] Voice profile extraction            ← depends on td-6d479f, td-8fabc8
td-e2455e  [E007] Thematic index generation           ← depends on td-6d479f, td-8fabc8
td-62199a  [E007] Terminology normalization           ← depends on td-e2455e
td-ef6f18  [E007] Cross-work thematic evolution       ← depends on td-e2455e, td-62199a
td-486a56  [E007] Voice profile CRUD & versioning     ← depends on td-f53821

GRAPH-AGENT (parallel with intelligence):
td-333c82  [E008] Entity extraction pipeline          ← depends on td-2a6e2f, td-6d479f
td-0aec66  [E008] Passage linking — explicit citations ← depends on td-333c82
td-b52dda  [E008] Passage linking — implicit engagement ← depends on td-333c82, td-62199a
td-53ab4f  [E008] Passage linking — thematic parallels ← depends on td-333c82, td-6de747
td-cb8e29  [E008] Graph query helpers                 ← depends on td-333c82
```
**Gate**: Voice profile JSON generated, knowledge graph populated, passage links created

### Phase 5: Retrieval (depends on Phase 4)
```
RETRIEVAL-AGENT:
td-fd4631  [E009] Vector similarity search            ← depends on td-605b26, td-6de747
td-e81ccf  [E009] Full-text / BM25 search            ← depends on td-743ae2
td-3baebf  [E009] Hybrid retrieval fusion             ← depends on td-fd4631, td-e81ccf
td-e10c30  [E009] Graph-augmented retrieval           ← depends on td-cb8e29
td-cc92f7  [E009] Multi-pass retrieval orchestration  ← depends on td-3baebf, td-e10c30
td-fcdc78  [E009] Context assembly & voice calibration ← depends on td-cc92f7, td-f53821
```
**Gate**: Multi-pass retrieval returns ranked, cited results from all stores

### Phase 6: MCP Tools (depends on Phase 5 + various)
```
TOOLS-AGENT:
td-cb28ac  [E010] Ingestion pipeline orchestrator     ← depends on td-6d479f, td-8fabc8, td-333c82
td-f276a7  [E010] ingest_book MCP tool                ← depends on td-cb28ac
td-185691  [E010] ingest_corpus MCP tool              ← depends on td-f276a7, td-f53821, td-e2455e
td-2f7baa  [E011] ask_author MCP tool                 ← depends on td-fcdc78
td-b4ef16  [E011] trace_theme MCP tool                ← depends on td-fcdc78, td-e2455e
td-932c31  [E011] find_quotes MCP tool                ← depends on td-fcdc78
td-5df45b  [E011] compare_ideas MCP tool              ← depends on td-2f7baa
td-71867a  [E012] list_authors / author_bio / list_works ← depends on td-605b26
td-1bea0d  [E012] library_stats MCP tool              ← depends on td-605b26
```
**Gate**: All MCP tools respond correctly via stdio

### Phase 7: Integration (depends on Phase 6)
```
INTEGRATION-AGENT:
td-40edd3  [E013] End-to-end integration tests        ← depends on td-f276a7, td-2f7baa
td-070d66  [E013] SSE transport for remote access     ← depends on td-ee7ebd
td-06044b  [E013] Performance & caching               ← depends on td-2f7baa
td-0b6e27  [E013] Error handling & resilience         ← depends on td-2f7baa
td-0a973a  [E013] Deployment guide & CLAUDE.md        ← depends on td-40edd3
```
**Gate**: E2E test passes, SSE transport works, deployment documented

---

## Orchestrator Checklist

### Before Spawning Agents
- [ ] Git repo initialized with main branch
- [ ] Docker Compose running (PG + Neo4j)
- [ ] .env file with API keys (Anthropic, embedding provider)
- [ ] Foundation (E001) completed and merged to main

### Per-Phase Gate Checks
- [ ] **Phase 1→2**: MCP server starts, connects to PG+Neo4j
- [ ] **Phase 2→3**: Tables exist, Neo4j schema created, parsers extract from EPUB
- [ ] **Phase 3→4**: Classification pipeline gates documents, chunks generated with annotations
- [ ] **Phase 4→5**: Voice profile JSON generated, knowledge graph populated, passage links created
- [ ] **Phase 5→6**: Multi-pass retrieval returns ranked, cited results
- [ ] **Phase 6→7**: All MCP tools respond correctly via stdio
- [ ] **Phase 7→Done**: E2E test passes, SSE transport works

### Per-Agent Verification (from ORCHESTRATOR_CHECKLIST.md adapted)
- [ ] Agent worked in correct worktree branch
- [ ] Clean build passes (uv run pytest, uv run mypy)
- [ ] No mock data or placeholder implementations
- [ ] Real API calls where specified
- [ ] Proper error handling (no bare except, no swallowed errors)
- [ ] Code volume appropriate for story complexity
- [ ] Merge preview shows no conflicts

---

## Critical Constraints (from collection-librarian)

1. **Voice contamination is the #1 risk**. Secondary sources must NEVER enter voice profile extraction. This is enforced at the classification gate (E003), storage isolation (E002), and retrieval-time labeling (E009).

2. **Source classification must happen BEFORE any enrichment**. The catalog pipeline (E003) is a hard dependency for all downstream processing.

3. **Never split a poem**. Poetry is atomic at the meso chunk level. The chunking agent must respect genre-specific boundaries per chunking-guide.md.

4. **Cross-resource passage links have three confidence tiers**. Explicit citations (high), implicit engagement (medium), thematic parallels (low). Each tier has different detection methods and different retrieval-time presentation rules.

5. **Default to SECONDARY when classification is uncertain**. Better to temporarily exclude legitimate primary material than to contaminate the voice profile.

6. **The catalog schema is specified**. Use the exact field definitions from catalog-schema.md. Don't simplify or skip fields.
