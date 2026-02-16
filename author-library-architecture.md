# The Author Library
## An MCP Server for Conversational Author Intelligence

*LOCOMOTIVE Agency — Architecture Specification*

---

## Origin

> I would really like to be able to have a database — sort of like a RAG for a language model — but based on books. So if I wanted to upload every book by a particular author, or their papers or something like that, and be able to chat with that author through their writings. That's what I'm looking for. Does something like that exist already? It seems like traditional RAG is too limited in the way that it indexes content. What would you recommend?

This document is the answer to that question. What follows is a full architecture specification for a system that goes well beyond traditional RAG to create a genuine "conversational author intelligence" — an MCP server that lets you load an author's complete body of work and have a grounded, voice-accurate conversation with them through their writings.

---

## What Already Exists

Before building something new, it's worth acknowledging what's already out there:

**Google NotebookLM** is the closest off-the-shelf tool. You can upload up to 50 sources (PDFs, Google Docs, web URLs, etc.), each up to 500,000 words, and it creates a grounded conversational interface with inline citations. It's free, easy, and gives you the "chat with an author's body of work" experience without any engineering. The trade-off: you don't control the retrieval or indexing — it's a black box.

**Dedicated "chat with books" apps** — sites like chat-with-books.com and various open-source projects let you upload documents and chat with them. These tend to be simple RAG wrappers and hit exactly the limitations described above.

**Custom RAG pipelines** — The LangChain/LlamaIndex ecosystem has extensive tooling for building your own. These give you full control but require significant engineering and still suffer from the fundamental chunking problem unless you add the semantic enrichment layer described in this architecture.

The Author Library is what you build when NotebookLM isn't enough and you want the retrieval to actually understand an author's voice, thematic arcs, and intellectual evolution across their body of work.

---

## The Problem

Traditional RAG treats books like databases — it chops text into small chunks, embeds them as vectors, and retrieves the "closest" snippets to a query. This works fine for factual lookup ("What year did the author say X?") but fails catastrophically at the things that actually matter when you want to *converse with an author through their work*:

- **Voice and tone** — A 256-token chunk strips away the cadence and style that makes an author recognizable
- **Thematic arcs** — Ideas that develop across chapters or across books are invisible to chunk-level retrieval
- **Conceptual relationships** — How an author's thinking on Topic A connects to their thinking on Topic B
- **Argumentative structure** — The *reasoning* behind a claim, not just the claim itself
- **Evolution of thought** — How an author's position shifts between early and late works

The Author Library solves this by building a rich semantic layer *on top of* traditional vector retrieval, using the LLM itself during ingestion to create what amounts to an author's intellectual fingerprint.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP CLIENT LAYER                        │
│         (Claude Code, Claude.ai, any MCP client)            │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP Protocol (stdio or SSE)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  THE AUTHOR LIBRARY SERVER                   │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  Ingest Tools  │  │  Query Tools  │  │  Meta Tools    │  │
│  │               │  │               │  │                │  │
│  │ • ingest_book │  │ • ask_author  │  │ • list_authors │  │
│  │ • ingest_paper│  │ • trace_theme │  │ • author_bio   │  │
│  │ • ingest_corpus│ │ • find_quotes │  │ • list_works   │  │
│  │               │  │ • compare_ideas│ │ • library_stats│  │
│  │               │  │ • voice_query │  │                │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘  │
│          │                  │                   │           │
│  ┌───────▼──────────────────▼───────────────────▼────────┐  │
│  │                  ORCHESTRATION LAYER                    │  │
│  │                                                        │  │
│  │  • Multi-pass retrieval (retrieve → reason → retrieve) │  │
│  │  • Context assembly (combines chunks + graph + summary) │  │
│  │  • Voice calibration (prompt with author's style data)  │  │
│  └────────────┬───────────────────────┬──────────────────┘  │
│               │                       │                     │
│  ┌────────────▼────────┐  ┌───────────▼──────────────────┐  │
│  │   RETRIEVAL ENGINE  │  │    KNOWLEDGE GRAPH ENGINE    │  │
│  │                     │  │                              │  │
│  │  Vector Store       │  │  Entities: Authors, Works,   │  │
│  │  (Chunks + Context) │  │    Themes, Arguments,        │  │
│  │                     │  │    Concepts, People           │  │
│  │  Full-Text Index    │  │                              │  │
│  │  (BM25 / hybrid)   │  │  Relations: develops,         │  │
│  │                     │  │    contradicts, references,   │  │
│  │  Summary Index      │  │    evolves_from, responds_to  │  │
│  │  (Chapter + Work)   │  │                              │  │
│  └────────────┬────────┘  └──────────┬───────────────────┘  │
│               │                      │                      │
│  ┌────────────▼──────────────────────▼───────────────────┐  │
│  │                    STORAGE LAYER                       │  │
│  │                                                       │  │
│  │  PostgreSQL + pgvector  │  OR  │  SQLite + vec ext    │  │
│  │  (production)           │      │  (local dev)         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## The Ingestion Pipeline: Where the Magic Happens

The key insight: **use the LLM during ingestion to pre-compute the semantic understanding that traditional RAG tries to reconstruct at query time.** This is more expensive up front but makes every query dramatically better.

### Phase 1: Parse & Structure

Extract text from the source material (EPUB, PDF, plain text) and preserve the document's own structure — chapters, sections, headings, footnotes.

**Input formats:** EPUB (preferred), PDF, DOCX, plain text, HTML
**Output:** A structured document tree with metadata

```
Book
├── Metadata (title, author, publication date, edition)
├── Front Matter (dedication, preface, introduction)
├── Chapter 1: "Title"
│   ├── Section 1.1
│   │   ├── Paragraph[]
│   │   └── Footnotes[]
│   └── Section 1.2
├── Chapter 2: "Title"
│   └── ...
└── Back Matter (appendix, bibliography, index)
```

**Tech:** Use `unstructured` (Python) or `epub.js` / `pdf-parse` (Node). The parser should emit a JSON AST that downstream phases consume.

### Phase 2: Multi-Granularity Chunking

Instead of one chunk size, create **three parallel chunk sets** that serve different retrieval needs:

| Granularity | Size | Purpose | Example |
|-------------|------|---------|---------|
| **Micro** | 100–200 tokens | Precise quote retrieval, specific claims | A single paragraph or key passage |
| **Meso** | 500–1000 tokens | Argument-level retrieval, reasoning chains | A complete section or argument |
| **Macro** | Chapter summaries | Thematic retrieval, structural navigation | LLM-generated chapter digest |

Each chunk carries **contextual metadata** generated by the LLM:

```json
{
  "chunk_id": "guite-theology-ch3-p12",
  "text": "The actual passage text...",
  "granularity": "meso",
  "work": "Faith, Hope and Poetry",
  "author": "malcolm-guite",
  "chapter": 3,
  "chapter_title": "Coleridge and the Re-enchantment of the World",
  "position": 0.34,
  "contextual_summary": "Guite argues that Coleridge's concept of the 'primary imagination' is not merely aesthetic but fundamentally theological — an act of participation in God's creative activity.",
  "themes": ["imagination", "theology", "coleridge", "romanticism", "enchantment"],
  "key_claims": [
    "The primary imagination is a repetition in the finite mind of the eternal act of creation",
    "Coleridge's epistemology is inherently sacramental"
  ],
  "references_to": ["samuel-taylor-coleridge", "barfield-poetic-diction"],
  "tone": "scholarly-devotional",
  "embedding": [0.023, -0.117, ...]
}
```

The `contextual_summary` field is critical — this is what Anthropic calls "[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)." Before embedding each chunk, prepend a short description of where it sits in the larger work. Anthropic's research showed this reduces retrieval failures by 49% (67% when combined with reranking), and prompt caching makes it cost-effective at roughly $1.02 per million document tokens. This is the single highest-ROI improvement over vanilla RAG, and surprisingly few implementations actually use it.

### Phase 3: Author Intelligence Extraction

This is what makes the Author Library different from generic RAG. For each author in the library, the LLM processes their entire corpus and generates:

#### 3a. Author Voice Profile

A structured description of the author's writing characteristics, used to calibrate response generation:

```json
{
  "author": "malcolm-guite",
  "voice_profile": {
    "register": "Scholarly but warm; bridges academic theology and lived devotional experience",
    "sentence_patterns": "Favors long, subordinate-clause-rich sentences that build toward a culminating insight. Uses semicolons generously. Frequently embeds poetry quotations mid-paragraph as evidence.",
    "vocabulary_tendencies": "Draws from both academic theology (kenosis, sacramental, eschatological) and Romantic literary criticism (fancy/imagination distinction, negative capability). Avoids jargon without definition.",
    "rhetorical_moves": "Often begins with a paradox or a question, works through literary and theological sources in dialogue, arrives at a synthesis that reframes the opening question. Heavy use of the 'both/and' rather than 'either/or'.",
    "characteristic_phrases": ["transfigure", "re-enchantment", "the imagination as a truth-bearing faculty", "baptize the imagination"],
    "humor_style": "Gentle, self-deprecating, often in asides. Occasionally playful with etymologies.",
    "example_passages": ["...", "...", "..."]
  }
}
```

#### 3b. Knowledge Graph

A graph database of the author's intellectual landscape:

```
Nodes:
  - Author (malcolm-guite)
  - Work (faith-hope-poetry, word-in-the-wilderness, ...)
  - Theme (imagination, enchantment, incarnation, poetic-truth, ...)
  - Argument (imagination-as-theological, poetry-as-prophecy, ...)
  - Person (coleridge, barfield, lewis, tolkien, donne, herbert, ...)
  - Concept (primary-imagination, negative-capability, kenosis, ...)

Edges:
  - AUTHORED (guite → faith-hope-poetry)
  - EXPLORES_THEME (faith-hope-poetry → imagination)
  - MAKES_ARGUMENT (guite → imagination-as-theological)
  - ARGUMENT_IN_WORK (imagination-as-theological → faith-hope-poetry, ch3)
  - REFERENCES_PERSON (guite → coleridge, relationship: "primary intellectual ancestor")
  - DEVELOPS_FROM (later-argument → earlier-argument)
  - RESPONDS_TO (guite-argument → opposing-argument)
  - CONCEPT_USED_IN (primary-imagination → [work1-ch3, work2-ch7, ...])
```

#### 3c. Thematic Index

A pre-computed map of themes to their appearances across the corpus, with summaries of how the author treats each theme in each work:

```json
{
  "theme": "imagination",
  "author_stance": "Central to Guite's entire project. He treats imagination not as escapist fancy but as a truth-bearing faculty — the primary means by which we perceive and participate in reality.",
  "appearances": [
    {
      "work": "Faith, Hope and Poetry",
      "chapters": [1, 3, 5, 7],
      "treatment": "Develops the argument historically from Coleridge through Barfield to Lewis, building a case for imagination as theological epistemology."
    },
    {
      "work": "Mariner",
      "chapters": [2, 8, 14],
      "treatment": "Reads The Rime of the Ancient Mariner as a sustained meditation on how imagination heals the split between subject and object."
    }
  ],
  "related_themes": ["enchantment", "poetic-truth", "sacramental-vision"],
  "key_passages": ["chunk-id-1", "chunk-id-2", "chunk-id-3"]
}
```

### Phase 4: Embedding & Indexing

With all the enriched data from Phases 2–3, build the retrieval indices:

1. **Vector index** — Embed all chunks (micro, meso, macro) using a high-quality embedding model (e.g., `voyage-3-large` or `text-embedding-3-large`). The contextual summaries are prepended to each chunk before embedding.

2. **Full-text index** — BM25 or similar keyword index for exact phrase matching and quote lookup.

3. **Graph index** — The knowledge graph goes into a graph store (Neo4j for production, or a simple adjacency list in SQLite for local).

4. **Summary index** — Chapter and work-level summaries stored separately for high-level thematic queries.

---

## MCP Tool Definitions

### Ingestion Tools

#### `ingest_book`
```typescript
{
  name: "ingest_book",
  description: "Ingest a book into the Author Library. Processes the text through the full enrichment pipeline: parsing, multi-granularity chunking, contextual annotation, voice profiling, and knowledge graph extraction.",
  inputSchema: {
    type: "object",
    properties: {
      file_path: { type: "string", description: "Path to EPUB, PDF, or text file" },
      author_id: { type: "string", description: "Author identifier (e.g., 'malcolm-guite')" },
      title: { type: "string" },
      publication_year: { type: "number" },
      genre_tags: { type: "array", items: { type: "string" } }
    },
    required: ["file_path", "author_id", "title"]
  }
}
```

#### `ingest_corpus`
```typescript
{
  name: "ingest_corpus",
  description: "Bulk ingest multiple works by the same author. Runs the full pipeline and generates cross-work analysis (thematic evolution, intellectual development over time).",
  inputSchema: {
    type: "object",
    properties: {
      author_id: { type: "string" },
      author_name: { type: "string" },
      works: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file_path: { type: "string" },
            title: { type: "string" },
            publication_year: { type: "number" }
          }
        }
      }
    },
    required: ["author_id", "author_name", "works"]
  }
}
```

### Query Tools

#### `ask_author`
The primary conversational tool. Retrieves relevant context and generates a response *in the author's voice*, grounded in their actual writings.

```typescript
{
  name: "ask_author",
  description: "Ask a question as if conversing with the author. The response is grounded in the author's actual writings and calibrated to their voice and style. Returns both the conversational response and source citations.",
  inputSchema: {
    type: "object",
    properties: {
      author_id: { type: "string" },
      question: { type: "string" },
      works_filter: {
        type: "array",
        items: { type: "string" },
        description: "Optional: limit to specific works"
      },
      response_style: {
        type: "string",
        enum: ["conversational", "academic", "devotional", "lecture"],
        description: "How formal/informal the response should be"
      }
    },
    required: ["author_id", "question"]
  }
}
```

**Orchestration flow for `ask_author`:**

```
1. CLASSIFY the question type:
   - Factual lookup → micro-chunk retrieval
   - Thematic exploration → graph traversal + macro summaries
   - "What would X say about Y?" → voice profile + thematic index + meso chunks
   - Quote request → full-text search + micro chunks

2. RETRIEVE (multi-pass):
   Pass 1: Initial retrieval from vector store + graph
   Pass 2: Expand context using graph edges (related themes, arguments)
   Pass 3: Pull supporting quotes from micro-chunk index

3. ASSEMBLE context window:
   - Author voice profile (always included)
   - Relevant thematic summaries
   - Best meso-level passages (the meat)
   - Supporting micro-level quotes
   - Graph context (related concepts, referenced thinkers)

4. GENERATE response with system prompt:
   "You are responding as {author_name}, drawing only on their
    published writings. Your voice should match: {voice_profile}.
    Ground every claim in the provided passages. When the author
    hasn't directly addressed a topic, say so — but you may
    extrapolate from their known positions if clearly labeled
    as extrapolation."

5. CITE: Attach source references (work, chapter, page) to each claim
```

#### `trace_theme`
```typescript
{
  name: "trace_theme",
  description: "Trace how an author treats a specific theme across their works, showing how their thinking develops or shifts over time.",
  inputSchema: {
    type: "object",
    properties: {
      author_id: { type: "string" },
      theme: { type: "string" },
      chronological: { type: "boolean", default: true }
    },
    required: ["author_id", "theme"]
  }
}
```

#### `compare_ideas`
```typescript
{
  name: "compare_ideas",
  description: "Compare how two or more authors in the library treat the same topic or concept.",
  inputSchema: {
    type: "object",
    properties: {
      author_ids: { type: "array", items: { type: "string" }, minItems: 2 },
      topic: { type: "string" }
    },
    required: ["author_ids", "topic"]
  }
}
```

#### `find_quotes`
```typescript
{
  name: "find_quotes",
  description: "Find specific passages where an author discusses a topic. Returns exact quotes with full citation information.",
  inputSchema: {
    type: "object",
    properties: {
      author_id: { type: "string" },
      topic: { type: "string" },
      max_results: { type: "number", default: 5 }
    },
    required: ["author_id", "topic"]
  }
}
```

---

## Technology Stack Recommendations

### For LOCOMOTIVE (Production / Client Deployments)

| Component | Recommendation | Why |
|-----------|---------------|-----|
| **Runtime** | Node.js / TypeScript | Consistent with your MCP server ecosystem |
| **MCP Framework** | `@modelcontextprotocol/sdk` | Standard MCP TypeScript SDK |
| **LLM (Ingestion)** | Claude Sonnet 4.5 via API | Best balance of intelligence and cost for bulk processing |
| **LLM (Query)** | Claude Sonnet 4.5 or Opus 4.5 | Depending on response quality needs |
| **Embeddings** | Voyage AI `voyage-3-large` or OpenAI `text-embedding-3-large` | Top-tier embedding quality for literary text |
| **Vector Store** | PostgreSQL + pgvector (on DigitalOcean managed DB) | You already use DO; keeps it in your ecosystem |
| **Graph Store** | PostgreSQL with JSONB + recursive CTEs | Avoids adding Neo4j; PG handles moderate graph queries well |
| **Full-Text Search** | PostgreSQL `tsvector` | Already there, good enough for exact phrase matching |
| **File Parsing** | `unstructured` (Python sidecar) or `epub.js` + `pdf-parse` | Battle-tested document parsing |
| **Queue (for ingestion)** | BullMQ + Redis | Ingestion is slow; queue and process async |

### For Local / Personal Use

| Component | Recommendation |
|-----------|---------------|
| **Storage** | SQLite + `sqlite-vec` extension |
| **Graph** | In-memory adjacency list, persisted to SQLite |
| **Embeddings** | Local model via Ollama (e.g., `nomic-embed-text`) or API |
| **LLM** | Claude API (or local via Ollama for privacy) |

---

## Deployment Model

```
┌─────────────────────────────────────┐
│          MCP Client (Claude Code)    │
│                                     │
│  "Ask Malcolm Guite about the       │
│   role of imagination in prayer"    │
└──────────────┬──────────────────────┘
               │ stdio / SSE
               ▼
┌─────────────────────────────────────┐
│     Author Library MCP Server        │
│     (Node.js on DO Droplet or local) │
│                                     │
│     Exposes: ask_author, ingest_book │
│              trace_theme, etc.       │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────────┐
│ PostgreSQL  │  │  Claude API     │
│ (pgvector + │  │  (for ingestion │
│  full-text) │  │   + generation) │
└─────────────┘  └─────────────────┘
```

The server can run:
- **Locally** via `stdio` transport (Claude Code connects directly)
- **Remotely** via SSE transport on a DigitalOcean droplet (like your existing MCP router on Fly.io)
- **Multi-tenant** with author libraries scoped per user/team

---

## Ingestion Cost Estimates

For a typical author corpus (let's say Malcolm Guite's published books — roughly 5 books, ~400,000 words total):

| Phase | API Calls | Estimated Cost |
|-------|-----------|----------------|
| Contextual annotation (per chunk) | ~2,000 Sonnet calls | ~$6–8 |
| Voice profile extraction | 5–10 Sonnet calls | ~$0.50 |
| Knowledge graph extraction | ~50 Sonnet calls | ~$3–5 |
| Thematic index generation | ~20 Sonnet calls | ~$2–3 |
| Embeddings (all chunks) | ~2,000 embedding calls | ~$0.50 |
| **Total per author corpus** | | **~$12–17** |

This is a one-time cost per author. Queries are cheap (one Sonnet/Opus call per question, typically $0.01–0.05).

---

## What Makes This Different

| Feature | Traditional RAG | Author Library |
|---------|----------------|----------------|
| Chunking | Fixed-size, single granularity | Multi-granularity (micro/meso/macro) |
| Embeddings | Raw text chunks | Contextually annotated chunks |
| Retrieval | Single vector similarity pass | Multi-pass: vector + graph + full-text |
| Author voice | Not captured | Explicit voice profile calibrates generation |
| Cross-work themes | Not tracked | Knowledge graph connects ideas across works |
| Intellectual evolution | Invisible | Chronological thematic tracing |
| Citation quality | Chunk IDs | Work, chapter, section, page references |

---

## Suggested Build Phases

### Phase 1: MVP (2–3 weeks)
- Single-author ingestion (EPUB → chunks → embeddings)
- Basic `ask_author` with vector retrieval + contextual summaries
- SQLite storage for fast iteration
- stdio MCP transport for Claude Code

### Phase 2: Intelligence Layer (2–3 weeks)
- Voice profile extraction
- Knowledge graph generation
- Multi-pass retrieval orchestration
- `trace_theme` and `find_quotes` tools

### Phase 3: Production Hardening (2–3 weeks)
- PostgreSQL + pgvector migration
- SSE transport for remote access
- Multi-author support + `compare_ideas`
- Ingestion queue with progress tracking
- Web UI for library management (optional)

### Phase 4: Product Features (ongoing)
- Audio overview generation (NotebookLM-style)
- Collaborative annotations
- API access for third-party integrations
- Public author libraries (think: a literary commons)

---

## Naming Suggestion

Keeping with LOCOMOTIVE's railroad theming:

**"The Reading Room"** — Every great train station has one. A quiet space where you sit with an author's complete works and converse.

Or if you want it more technical: **"Dispatch Library"** — where knowledge gets routed to the right destination.
