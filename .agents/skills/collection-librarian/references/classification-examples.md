# Source Classification: Worked Examples

Ambiguous cases and edge cases for source classification in the Author Library.

## Table of Contents

1. The Core Distinction: Author Writing About Others vs. Others Writing About Author
2. Edited Collections and Anthologies
3. Interviews and Conversations
4. Forewords, Introductions, and Blurbs
5. Co-Authored Works
6. Translations
7. The Author Writing About Their Own Earlier Work
8. Festschriften and Tribute Volumes
9. Edge Cases: When Classification Is Genuinely Ambiguous

## 1. The Core Distinction

This is the most common and most consequential classification decision. Two books may sit side by side on a shelf, both involving the same two names, but they flow in opposite directions.

### Example: Malcolm Guite and Samuel Taylor Coleridge

**Work A:** *Mariner: A Voyage with Samuel Taylor Coleridge* by Malcolm Guite (2017)
- **Classification: PRIMARY**
- **Reasoning:** Guite is the author. The book is Guite's sustained interpretation of Coleridge's *Rime of the Ancient Mariner*. Every sentence reflects Guite's voice, theological framework, and critical method. The fact that the *subject matter* is another author's poem does not change the authorship. This is Guite's intellectual work.
- **Pipeline:** Full enrichment. Voice profile, knowledge graph (Guite `MAKES_ARGUMENT` about Coleridge), thematic index, cross-resource passage links to Coleridge's texts.

**Work B:** *The Cambridge Companion to Coleridge* (various editors, with a chapter by Guite)
- **Classification: SECONDARY (collection-level) / PRIMARY (Guite's chapter only)**
- **Reasoning:** The collection is an editorial project about Coleridge, not by Guite. But if Guite contributed a chapter, that specific chapter is primary material — it's Guite's voice and argument. Extract it.
- **Pipeline:** Collection-level metadata as secondary. Guite's chapter extracted and processed as primary. Other chapters as secondary if they discuss Guite, otherwise not ingested.

**Work C:** *A biography of Malcolm Guite* by a scholar
- **Classification: SECONDARY**
- **Reasoning:** Another person writing about Guite. The biographer's voice, interpretive framework, and editorial choices shape the text. Even direct quotes from Guite embedded in the biography are filtered through the biographer's selection and framing.
- **Pipeline:** Embeddings + knowledge graph with `ATTRIBUTED_BY_CRITIC` edges. Never voice profile.

**Work D:** *Biographia Literaria* by Samuel Taylor Coleridge
- **Classification: CONTEXTUAL**
- **Reasoning:** Written by Coleridge, not by or about Guite. But Guite's theory of imagination is built on Coleridge's work in this text. Including it illuminates why Guite makes the arguments he makes, and enables cross-resource passage links from Guite's chunks to the specific Coleridge passages he engages with.
- **Pipeline:** Embeddings only, tagged for cross-reference retrieval. Serves as a link target for `ENGAGES_WITH` edges from primary source chunks.

## 2. Edited Collections and Anthologies

### Subject Author as Editor

**Scenario:** Guite edits *The Word in the Wilderness: A Poem a Day for Lent and Easter* — a curated collection of poems by other authors with Guite's commentary.

- **Classification: PRIMARY (composite)**
- **Reasoning:** Guite's editorial voice — his selections, his introductions, his commentary — is primary. The poems he selected are by other authors and are contextual. But the act of curation and the commentary revealing *why* he chose each poem are expressions of his intellectual identity.
- **Chunking:** Guite's commentary passages = primary chunks. Selected poems = contextual chunks linked to Guite's commentary via `ENGAGES_WITH` edges. The editorial structure itself (which poems, in what order, grouped how) is captured in macro-level metadata as primary.

### Subject Author as Contributor

**Scenario:** Guite writes Chapter 7 in an edited collection about Inklings theology.

- **Classification: MIXED**
- **Chapter 7: PRIMARY** — Extract and process independently.
- **Editorial introduction mentioning Guite: SECONDARY** — Process with provenance markers.
- **Other chapters: NOT INGESTED** unless they discuss Guite specifically.

## 3. Interviews and Conversations

Interviews are inherently hybrid documents.

### Published Interview in a Magazine

**Scenario:** A literary magazine publishes "A Conversation with Malcolm Guite."

- **Classification: SECONDARY (container) with PRIMARY-ADJACENT (responses)**
- **The interviewer's questions and framing:** Secondary. They shape the conversation but are not Guite's voice.
- **Guite's responses:** Primary-adjacent. These are Guite's own words, but spoken in response to specific prompts, possibly edited for publication. Mark as `voice_profile_eligible: true` but with reduced confidence weight.
- **The magazine's introduction/headnote:** Secondary.

### Podcast or Video Transcript

**Scenario:** A transcript of Guite speaking on a theology podcast.

- **Classification:** Same as above, but note that transcripts may contain more informal language, false starts, and digressions. These are actually *more* valuable for voice profile enrichment because they capture the author's natural speech patterns.

### The Author Interviewing Someone Else

**Scenario:** Guite interviews Rowan Williams for a publication.

- **Classification: PRIMARY (Guite's questions and framing) / CONTEXTUAL (Williams' responses)**
- **Reasoning:** Guite's questions reveal his intellectual interests and how he frames issues. Williams' responses illuminate Guite's intellectual network but are not Guite's voice.

## 4. Forewords, Introductions, and Blurbs

### Subject Author Writes Foreword for Another Book

**Scenario:** Guite writes the foreword to a friend's poetry collection.

- **Classification: PRIMARY**
- **Reasoning:** It's Guite's voice, his evaluation, his framing. Often these short pieces contain some of the most direct statements of the author's critical principles.
- **Chunking:** Typically a single meso chunk. Cross-reference to the book it introduces.

### Someone Writes Foreword for Subject Author's Book

**Scenario:** Rowan Williams writes the foreword to *Faith, Hope and Poetry*.

- **Classification: SECONDARY**
- **Reasoning:** Williams' framing of Guite's work. Valuable for understanding how peers received the work, but not Guite's voice.
- **Extraction:** Link to the primary work's catalog record as supplementary material.

### Blurbs and Endorsements

- **By the subject author:** Primary (micro chunk). Tag as `genre: endorsement`.
- **About the subject author's work:** Secondary metadata only. Do not create content chunks from blurbs — they're too compressed to be useful for retrieval.

## 5. Co-Authored Works

**Scenario:** Guite co-authors a book with another scholar.

- **Classification: PRIMARY with caveats**
- **Set `voice_profile_eligible: false`** — Co-authored prose blends voices. Including it in voice extraction would muddy the profile.
- **Knowledge graph:** Process normally. The intellectual content is valid even if voice is blended.
- **Note in metadata:** Identify co-author and, if discernible, which sections were primarily written by which author.

## 6. Translations

### Subject Author Translated into Another Language

- **Classification: PRIMARY** if the author oversaw the translation
- **Classification: PRIMARY with caveat** if translated by others — the ideas are the author's but the exact language is the translator's. Set `voice_profile_eligible: false`.

### Subject Author as Translator

- **Classification: PRIMARY (the translation itself) + CONTEXTUAL (the original)**
- **Reasoning:** Translation choices reveal the author's interpretive lens. How Guite translates a medieval hymn tells us about Guite's theology.

## 7. The Author Writing About Their Own Earlier Work

**Scenario:** In a later blog post, Guite reflects on how his thinking has changed since writing *Faith, Hope and Poetry*.

- **Classification: PRIMARY**
- **Special value:** This is gold for the thematic evolution tracker. The author is explicitly mapping their own intellectual development. Tag for high priority in the cross-work thematic linking process. Create `DEVELOPS_FROM` edges in Neo4j connecting the later reflection to the earlier work's relevant chunks.

## 8. Festschriften and Tribute Volumes

**Scenario:** A volume of essays published in the subject author's honor.

- **Classification: SECONDARY (collection-level)**
- **Any contribution by the subject author:** PRIMARY
- **Essays about the subject author's work:** SECONDARY with standard provenance
- **Essays on related topics that don't discuss the author:** Generally not ingested unless they illuminate the author's intellectual community

## 9. Edge Cases: When Classification Is Genuinely Ambiguous

**Default to SECONDARY when uncertain.** The reasoning: it is far more damaging to contaminate a voice profile with foreign prose than to temporarily exclude legitimate primary material from voice extraction. Secondary-classified material is still searchable and available for knowledge graph enrichment — it just doesn't touch the voice profile.

### Ambiguous Case: Published Diary or Journal

If the diary was published posthumously by an editor, the editor's footnotes and introduction are secondary, while the diary entries themselves are primary. But was the diary edited for publication? If so, the editor's hand may have reshaped the author's voice. Note the level of editorial intervention in metadata and flag for review.

### Ambiguous Case: Social Media

The subject author's tweets, Facebook posts, or similar are primary — but they're extremely compressed, context-dependent, and sometimes reactive rather than reflective. Ingest as primary but with reduced weight for voice profile extraction. Group by time period rather than treating individually.

### Ambiguous Case: Ghostwritten or Heavily Edited Work

If credible evidence suggests the author had significant editorial assistance, classify as primary but set `voice_profile_eligible: false` and note the concern in `source_class_note`.

### Ambiguous Case: The Subject Author's Book About Another Author, vs. A Third Party's Book About Both

This is the subtlest case and the one most likely to trip up an automated classifier. Consider:

| Work | Author | About | Classification |
|------|--------|-------|---------------|
| *Mariner* | Guite | Coleridge's poem | **Primary** — Guite's voice and argument |
| *Imagination and the Arts in C.S. Lewis* | (Scholar A) | Lewis, with chapter on Guite's reading of Lewis | **Secondary** — Another scholar's framing, even though it discusses Guite |
| *The Inklings and Imagination* | (Scholar B) | Lewis, Tolkien, Barfield, with Guite cited as critic | **Secondary** — Guite appears as a cited source, not as subject or author |
| *Poetry and the Christian Imagination* | Guite + (Co-editor) | Multiple poets | **Primary (Guite's contributions) / Mixed (co-edited framing)** |

The key question is always: **whose voice, whose argument, whose editorial judgment shapes this text?** If the answer is the subject author, it's primary. If someone else, it's secondary — regardless of how prominently the subject author is discussed within it.
