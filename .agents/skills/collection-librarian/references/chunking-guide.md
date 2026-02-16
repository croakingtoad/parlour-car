# Chunking Guide by Genre

Detailed patterns for multi-granularity chunking across different text types in the Author Library.

## Table of Contents

1. General Principles
2. Scholarly Prose (Monographs, Academic Papers)
3. Poetry
4. Sermons, Lectures, and Transcripts
5. Letters and Correspondence
6. Blog Posts and Short Essays
7. Interviews
8. Edited Collections with Mixed Authorship
9. Contextual Annotation Templates
10. Cross-Resource Passage Linking

## 1. General Principles

Every chunk at every granularity level must:
- Preserve complete semantic units (never split mid-thought)
- Carry its source classification as metadata (`primary`, `secondary`, `contextual`)
- Receive a contextual annotation before embedding
- Include structural position metadata (work → part → chapter → section → paragraph)

The three granularity levels:
- **Macro** (~500–1500 words): Chapter or major-section summaries. Used for high-level thematic queries.
- **Meso** (~150–500 words): Argument or subsection level. The primary retrieval unit for most queries.
- **Micro** (~30–200 words): Paragraph or passage level. Used for exact quote retrieval and fine-grained semantic matching.

## 2. Scholarly Prose (Monographs, Academic Papers)

**Macro chunks:** One per chapter. Generate a summary that captures the chapter's thesis, main arguments, key evidence cited, and conclusion. Include how this chapter connects to the preceding and following chapters.

**Meso chunks:** Split at natural section boundaries — heading breaks, major rhetorical transitions ("Having established X, we now turn to Y"), or shifts in the author's argumentative focus. Each meso chunk should contain one coherent argument or line of reasoning.

**Micro chunks:** Individual paragraphs. Keep paragraphs intact. If a paragraph exceeds 300 words, it may be split at the most natural sentence boundary, but only if both resulting chunks are semantically self-contained.

**Special handling:**
- Footnotes/endnotes: Attach to their referencing paragraph as metadata, not as separate chunks. If a footnote contains a substantive argument (common in academic writing), create a separate micro chunk linked to its parent.
- Block quotations: Keep within the paragraph that introduces them. Tag the quoted author in metadata so the knowledge graph can distinguish the subject author's words from cited sources.
- Bibliography/references: Do not chunk. Extract as structured metadata for the knowledge graph.

### Example: Chunking a Chapter of *Faith, Hope and Poetry*

```
MACRO: Chapter 3 — "John Donne"
  Summary: Guite reads Donne's Holy Sonnets as exemplifying how poetry
  can hold paradox and ambiguity in tension — a capacity theology needs
  but systematic prose cannot achieve. Connects to Coleridge's theory
  of symbol vs. allegory from Ch. 1.

  MESO: Section 3.1 — Donne's biographical context and conversion
    Guite situates Donne's shift from secular to sacred poetry...

    MICRO: Paragraph 3.1.1 — Opening biographical sketch
    MICRO: Paragraph 3.1.2 — The conversion's effect on Donne's poetics

  MESO: Section 3.2 — Close reading of "Batter my heart"
    Guite performs a detailed reading demonstrating five modes...

    MICRO: Paragraph 3.2.1 — The sonnet's opening paradox
    MICRO: Paragraph 3.2.2 — Guite's analysis of violent imagery
    MICRO: Paragraph 3.2.3 — Connection to kenotic theology

  MESO: Section 3.3 — Donne and the theology of paradox
    Guite argues Donne's poetry embodies what systematic theology...

    MICRO: Paragraph 3.3.1 — The both/and vs. either/or distinction
    MICRO: Paragraph 3.3.2 — Implications for contemporary theology
```

## 3. Poetry

**Critical principle:** Never split a poem. A poem is an atomic semantic unit.

**Macro chunks:** Collection-level or sequence-level summaries. For a sonnet sequence, summarize the arc of the full sequence (seasons, liturgical year, etc.).

**Meso chunks:** Individual poems. Each poem is one meso chunk regardless of length, unless the poem exceeds ~60 lines, in which case section breaks (cantos, parts, numbered stanzas in long-form poems) may be used.

**Micro chunks:** Stanza-level, only for poems exceeding ~40 lines. Each micro chunk must carry the full poem title and its position within the poem as metadata.

**Special handling:**
- Epigraphs: Attach as metadata to the poem, not as a separate chunk.
- Dedications: Attach as metadata.
- For poetry embedded within prose works (e.g., Guite quoting Herbert within a critical chapter), the poem chunk should be cross-referenced to both the original collection (if available) and the prose passage that discusses it.

## 4. Sermons, Lectures, and Transcripts

**Macro chunks:** Full talk summary (~500 words). Capture the occasion, the text/theme, the main movements, and the conclusion.

**Meso chunks:** Major movements of the talk. Most sermons have 3–5 movements (introduction, development of theme, illustration/story, application, conclusion). Split at these natural transitions, typically identifiable by topic shifts, pauses indicated in transcription, or explicit transitional language.

**Micro chunks:** Individual points, illustrations, or extended quotations within a movement.

**Special handling:**
- Audience context matters. A sermon preached at Girton College Chapel carries different contextual weight than a lecture delivered at a conference. Include venue and occasion in the contextual annotation.
- Transcription artifacts (um, uh, repeated starts) should be cleaned before chunking but noted in metadata if the transcript is unedited.

## 5. Letters and Correspondence

**Macro chunks:** Correspondence-period summaries grouped by recipient and time period. Example: "Letters to [recipient], 2010–2015: Primary themes include..."

**Meso chunks:** Individual letters are the atomic unit. One letter = one meso chunk unless the letter exceeds ~2000 words.

**Micro chunks:** For longer letters, split at clear topic transitions. Salutations and closings should remain attached to the nearest substantive paragraph, not isolated as separate chunks.

**Special handling:**
- Recipient identity matters for the knowledge graph. Always tag the recipient.
- Letters often contain the most informal and revealing expressions of the author's thinking. Flag passages where the author states positions more directly than in published work.

## 6. Blog Posts and Short Essays

**Typically single meso chunks.** Most blog posts (500–1500 words) function as self-contained arguments and should not be split.

**For posts exceeding ~2000 words:** Split into meso chunks at heading breaks or major topic transitions.

**Micro chunks:** Rarely needed. Only create micro chunks if the post contains discrete, independently quotable passages.

**Special handling:**
- Blog posts often have a more informal register than published work. Note this in the contextual annotation — it's valuable for voice profile enrichment as it captures the author's range.
- Comments by the author on their own posts are primary source material and should be captured as linked micro chunks.

## 7. Interviews

Interviews require special handling because they contain both primary and secondary material.

**Classification:** The interview as a whole is classified based on who published/conducted it. But the author's direct responses can be extracted as primary-adjacent material.

**Chunking approach:**
- Each question-and-answer pair is one meso chunk
- The interviewer's questions are tagged as `secondary` framing
- The author's responses are tagged as `primary-adjacent` (not full `primary` because the interview context shapes the response)
- Extended monologue responses may be split into multiple micro chunks

## 8. Edited Collections with Mixed Authorship

When the subject author contributes a chapter to a collection edited by someone else:

- The collection-level metadata is `secondary` (it's someone else's editorial project)
- The subject author's contributed chapter is extracted and classified as `primary`
- Other chapters remain `secondary`
- The introduction/editorial framing may contain useful contextual claims about the subject author — classify as `secondary` with `contains_primary_quotes: true`

## 9. Contextual Annotation Templates

Use these templates when writing the contextual annotation prepended to each chunk before embedding.

### Primary Source Template

```
[PRIMARY] From "{work_title}" ({publication_year}) by {subject_author}.
Chapter {chapter_number}: "{chapter_title}".
This {chunk_granularity} covers: {brief_topic_description}.
In the larger argument of this chapter, {positioning_note}.
Preceding context: {what_came_before}.
Following context: {what_comes_after}.
```

### Secondary Source Template

```
[SECONDARY: Written by {external_author} about {subject_author}]
From "{work_title}" ({publication_year}), a {relationship_type}.
Chapter {chapter_number}: "{chapter_title}".
This passage discusses: {brief_topic_description}.
The critic's perspective: {perspective_note}.
```

### Contextual Source Template

```
[CONTEXTUAL: By {original_author}, referenced by {subject_author}]
From "{work_title}" ({publication_year}).
Chapter {chapter_number}: "{chapter_title}".
This passage is relevant because: {engagement_note}.
{subject_author} engages with this material in: {engagement_works}.
```

## 10. Cross-Resource Passage Linking

Beyond thematic linking at the concept level, the system creates **passage-level links** between specific chunks across different works in the collection. This is the mechanism that allows retrieval to follow an author's intellectual conversation with their sources — not just "Guite discusses imagination" but "this specific paragraph in Guite is directly engaging with this specific passage in Coleridge."

### Why Passage Links Exist

Thematic linking (Core Competency 4 in SKILL.md) maps that Guite's argument about Primary Imagination in *Faith, Hope and Poetry* Ch. 1 connects to the same theme in *Mariner* Ch. 2. The contextual source classification tags Coleridge's *Biographia Literaria* as a work Guite engages with, with metadata about *which* of Guite's works reference it.

But neither mechanism creates a **pointer from a specific chunk in Guite's book to a specific chunk in Coleridge's book.** That is the gap passage linking fills.

### Three Types of Passage Links

#### Explicit Citation (High Confidence)

The subject author directly quotes, names, or footnotes a specific passage in another work.

**Detection methods:**
- Quotation marks followed by attribution ("as Coleridge writes...")
- Footnote/endnote references pointing to specific works and page numbers
- Parenthetical citations (Coleridge, *Biographia*, Ch. 13)
- Phrases like "in Chapter 13 of the *Biographia*" or "Coleridge's famous distinction"

**Neo4j edge:**
```cypher
CREATE (guite_chunk:Chunk {id: "guite--faith-hope-poetry--ch1--meso-3"})
  -[:ENGAGES_WITH {
    link_type: "explicit_citation",
    engagement_type: "interprets",
    subject_author: "malcolm-guite",
    direction: "guite-reading-coleridge",
    confidence: "high",
    detection_method: "footnote_reference",
    source_location: "Faith, Hope and Poetry, Ch. 1, p. 34",
    target_location: "Biographia Literaria, Ch. 13, pp. 295-296",
    annotation: "Guite reframes Coleridge's Primary Imagination as a theological faculty, arguing it is not merely a philosophical concept but the means by which humans participate in divine creativity"
  }]->
  (coleridge_chunk:Chunk {id: "coleridge--biographia-literaria--ch13--meso-2"})
```

#### Implicit Engagement (Medium Confidence)

The subject author uses terminology, concepts, or argumentative structures originating in another work without directly citing a specific passage.

**Detection methods:**
- Terminology fingerprinting: Match terms from the controlled vocabulary back to their origin in contextual sources. When Guite writes "Primary Imagination" or "esemplastic power," those are Coleridgean terms even without a footnote.
- Argumentative structure matching: When the subject author's reasoning follows the same logical shape as a contextual source's argument, even with different examples.
- Conceptual dependency: When a claim by the subject author is only intelligible if the reader already knows the contextual source's framework.

**Neo4j edge:**
```cypher
CREATE (guite_chunk:Chunk {id: "guite--word-in-wilderness--lent3--meso-1"})
  -[:ENGAGES_WITH {
    link_type: "implicit_engagement",
    engagement_type: "extends",
    subject_author: "malcolm-guite",
    direction: "guite-extending-coleridge",
    confidence: "medium",
    detection_method: "terminology_fingerprint",
    triggering_terms: ["Primary Imagination", "esemplastic"],
    annotation: "Guite applies Coleridge's vocabulary of imagination to the practice of Lenten devotional reading, extending the concept beyond literary criticism into spiritual discipline"
  }]->
  (coleridge_chunk:Chunk {id: "coleridge--biographia-literaria--ch13--meso-2"})
```

#### Thematic Parallel (Low Confidence)

Both the subject author and a contextual source discuss the same topic, but the subject author isn't directly engaging with that specific passage. The connection is inferred by the system, not authored by the writer.

**Detection methods:**
- High semantic similarity between chunks from different works on the same theme
- Both chunks tagged with the same canonical theme from the controlled vocabulary
- No explicit or implicit citation signals present

**Neo4j edge:**
```cypher
CREATE (guite_chunk:Chunk {id: "guite--faith-hope-poetry--ch8--meso-4"})
  -[:THEMATIC_PARALLEL {
    link_type: "thematic_parallel",
    shared_theme: "poetry-as-truth-bearing",
    confidence: "low",
    detection_method: "semantic_similarity",
    similarity_score: 0.87,
    annotation: "Both passages discuss poetry's epistemological status, but Guite may not be directly engaging with this specific Coleridge passage here"
  }]->
  (coleridge_chunk:Chunk {id: "coleridge--biographia-literaria--ch14--meso-5"})
```

**Important:** Thematic parallels should never be presented to users as "the author responding to this source." The framing should be exploratory: "You might also be interested in this related passage from [contextual source]."

### Passage Link Detection Pipeline

Run this process after both the primary source and the contextual source have been chunked and annotated:

```
1. Explicit citation scan (primary sources only)
   → Parse footnotes, endnotes, and inline citations
   → Match cited works against contextual sources in the collection
   → For each match, identify the most specific target chunk
   → Create ENGAGES_WITH edge with confidence: "high"

2. Terminology fingerprinting (primary sources only)
   → For each primary chunk, extract terms from the controlled vocabulary
   → For each term, look up its origin in contextual sources
   → If the origin passage exists in the collection, create ENGAGES_WITH edge
     with confidence: "medium"
   → If multiple contextual chunks could be the origin, link to the most
     specific one and note alternatives in edge metadata

3. Thematic parallel detection (all source pairs)
   → Compare semantic embeddings between primary chunks and contextual chunks
     that share at least one canonical theme tag
   → For pairs exceeding similarity threshold (suggest 0.85), create
     THEMATIC_PARALLEL edge with confidence: "low"
   → Filter out pairs that already have an ENGAGES_WITH edge (don't duplicate)

4. Human review queue
   → Flag medium-confidence links for review when the terminology match
     is ambiguous (e.g., common terms used by many authors)
   → Flag low-confidence links that have unusually high similarity scores
     (may be implicit engagements that the fingerprinting missed)
```

### Retrieval Behavior for Passage Links

When the retrieval engine returns chunks that have cross-resource passage links, the orchestration layer follows these rules:

**For `ask_author` queries:**
1. Primary chunks are returned as the author's voice (standard behavior)
2. If a returned primary chunk has `ENGAGES_WITH` edges, the linked contextual chunks are included as supplementary context
3. The response labels contextual chunks clearly: "In this passage, [subject author] is engaging with [contextual author]'s argument in [work, chapter]. [Contextual author] wrote: [brief excerpt]. [Subject author]'s interpretation is: [primary chunk content]."

**For `trace_theme` queries:**
1. Follow the theme across primary chunks chronologically (standard behavior)
2. At each stop, check for `ENGAGES_WITH` edges to contextual sources
3. Include the linked contextual passages to show how the author's engagement with their sources evolves alongside their own thinking

**For `find_quotes` queries:**
1. Return primary chunks matching the query (standard behavior)
2. If a returned chunk contains an explicit citation of a contextual source, include the cited passage and note the citation relationship

**For general exploration:**
1. `THEMATIC_PARALLEL` edges may be surfaced as "Related passages" or "You might also find this relevant"
2. Never present thematic parallels as the author's direct engagement — use exploratory framing

### Neo4j Indexes for Passage Links

Create these indexes to support efficient cross-resource traversal:

```cypher
// Index on chunk IDs for fast edge lookups
CREATE INDEX chunk_id_index FOR (c:Chunk) ON (c.id);

// Index on engagement type for filtered traversal
CREATE INDEX engagement_type_index FOR ()-[r:ENGAGES_WITH]-() ON (r.engagement_type);

// Index on link type for confidence-filtered queries
CREATE INDEX link_type_index FOR ()-[r:ENGAGES_WITH]-() ON (r.link_type);

// Index on shared theme for thematic parallel lookups
CREATE INDEX theme_parallel_index FOR ()-[r:THEMATIC_PARALLEL]-() ON (r.shared_theme);

// Composite index for source-author-scoped traversal
CREATE INDEX chunk_source_class FOR (c:Chunk) ON (c.source_class, c.subject_author);
```

### Example: Full Cross-Resource Link Set for One Passage

Consider a meso chunk from *Faith, Hope and Poetry* Ch. 1 where Guite introduces his theory of imagination:

```cypher
// The primary chunk
CREATE (fhp_ch1_m3:Chunk {
  id: "guite--faith-hope-poetry--ch1--meso-3",
  source_class: "primary",
  subject_author: "malcolm-guite",
  work: "Faith, Hope and Poetry",
  chapter: 1,
  granularity: "meso",
  topic: "Introduction of Primary Imagination as theological faculty"
})

// Explicit citation: Guite quotes Biographia Literaria Ch. 13
CREATE (fhp_ch1_m3)-[:ENGAGES_WITH {
  link_type: "explicit_citation",
  engagement_type: "interprets",
  confidence: "high",
  detection_method: "block_quotation_with_attribution",
  annotation: "Guite quotes Coleridge's definition of Primary Imagination and reframes it as participation in divine creativity"
}]->(bl_ch13_m2:Chunk {id: "coleridge--biographia-literaria--ch13--meso-2"})

// Implicit engagement: Guite uses "esemplastic" without footnote
CREATE (fhp_ch1_m3)-[:ENGAGES_WITH {
  link_type: "implicit_engagement",
  engagement_type: "extends",
  confidence: "medium",
  detection_method: "terminology_fingerprint",
  triggering_terms: ["esemplastic"],
  annotation: "Guite uses Coleridge's coined term without citation, suggesting assumed reader familiarity"
}]->(bl_ch10_m4:Chunk {id: "coleridge--biographia-literaria--ch10--meso-4"})

// Cross-work self-link: same argument recurs in Mariner
CREATE (fhp_ch1_m3)-[:DEVELOPS_INTO {
  direction: "chronological",
  annotation: "Guite revisits this argument seven years later with deeper engagement"
}]->(mar_ch2_m1:Chunk {id: "guite--mariner--ch2--meso-1"})

// Thematic parallel: similar topic in Barfield (weaker link)
CREATE (fhp_ch1_m3)-[:THEMATIC_PARALLEL {
  shared_theme: "imagination-as-participation",
  confidence: "low",
  annotation: "Both discuss imagination as participatory rather than merely representational, but Guite may be engaging with Barfield through Coleridge rather than directly here"
}]->(barfield_chunk:Chunk {id: "barfield--poetic-diction--ch5--meso-2"})
```

This gives the retrieval engine a rich graph to traverse: the primary passage, the Coleridge source it directly engages, a related Coleridge passage it implicitly references, its own later development in another work by the same author, and a weaker thematic connection to a third author in the intellectual network.
