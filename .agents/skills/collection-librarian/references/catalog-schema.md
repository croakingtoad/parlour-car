# Catalog Schema Reference

Complete metadata field definitions for the Author Library catalog system.

## Table of Contents

1. Core Fields (all source classes)
2. Primary Source Fields
3. Secondary Source Fields
4. Contextual Source Fields
5. Tertiary Source Fields
6. Validation Rules
7. Examples

## 1. Core Fields (All Source Classes)

These fields are required for every work entering the library, regardless of source classification.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `work_id` | string | Yes | Unique identifier. Format: `{author-slug}--{title-slug}`. Example: `malcolm-guite--faith-hope-and-poetry` |
| `title` | string | Yes | Full title including subtitle, separated by colon. Example: `Faith, Hope and Poetry: Theology and the Poetic Imagination` |
| `author` | string | Yes | Author as listed on work. Use subject author's canonical name form for primary sources. |
| `source_class` | enum | Yes | One of: `primary`, `secondary`, `tertiary`, `contextual` |
| `source_class_note` | string | Yes | 1-2 sentence justification for the classification. Required even when classification seems obvious — this creates an audit trail. |
| `publication_year` | integer | Yes | Year of first publication of this edition |
| `original_publication_year` | integer | No | Year of first-ever publication, if different from `publication_year` |
| `edition` | string | No | Edition descriptor. Example: `2nd revised edition`, `paperback reissue` |
| `publisher` | string | Yes | Publisher name |
| `isbn` | string | No | ISBN-13 preferred. ISBN-10 acceptable. |
| `format_ingested` | enum | Yes | One of: `epub`, `pdf`, `txt`, `html`, `docx` |
| `language` | string | Yes | ISO 639-1 code. Example: `en` |
| `word_count` | integer | Yes | Approximate word count of full text |
| `genre_tags` | array[string] | Yes | Genre/form descriptors. Use lowercase. Examples: `monograph`, `poetry-collection`, `academic-paper`, `sermon`, `blog-post`, `biography`, `critical-study` |
| `subject_headings` | array[string] | Yes | LCSH-style subject headings adapted for the collection. Examples: `Imagination--Religious aspects--Christianity`, `English poetry--History and criticism`, `Coleridge, Samuel Taylor, 1772-1834--Criticism and interpretation` |
| `ocr_quality` | enum | No | If sourced from OCR: `high` (>99% accuracy), `medium` (95-99%), `low` (<95%), `not-applicable` (born-digital) |
| `ingestion_date` | date | Yes | ISO 8601 date when work was ingested |
| `notes` | string | No | Free-text notes from the librarian about this work |

## 2. Primary Source Additional Fields

These fields supplement the core fields for works authored by the subject.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `subject_author_id` | string | Yes | The subject author's canonical identifier. Example: `malcolm-guite` |
| `work_type` | enum | Yes | One of: `monograph`, `essay-collection`, `poetry-collection`, `academic-paper`, `lecture-transcript`, `blog-post`, `letter`, `sermon`, `foreword`, `interview-responses`, `other` |
| `chronological_position` | integer | No | The work's position in the author's chronological bibliography. Useful for tracing intellectual development. |
| `voice_profile_eligible` | boolean | Yes | Default `true`. Set to `false` only for co-authored works or works where the subject author's voice is significantly mediated (e.g., a heavily edited interview). |
| `dedication` | string | No | Dedicatee(s), if present. Can reveal intellectual relationships. |
| `table_of_contents` | array[string] | No | Chapter/section titles as they appear in the work |

## 3. Secondary Source Additional Fields

These fields supplement the core fields for works about the subject author.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `about_author_id` | string | Yes | The subject author this work is about. Example: `malcolm-guite` |
| `external_author` | string | Yes | The actual author of this secondary work |
| `external_author_affiliation` | string | No | Institutional affiliation of the external author, if known |
| `relationship` | enum | Yes | One of: `biography`, `critical-study`, `review`, `interview`, `companion`, `dissertation`, `edited-collection`, `obituary`, `profile`, `other` |
| `perspective_note` | string | Yes | Brief characterization of the critic's stance, methodology, or school of thought. Example: `Post-structuralist reading focusing on Guite's use of metaphor` or `Sympathetic biographical account by a former student` |
| `contains_primary_quotes` | boolean | Yes | Does this work contain substantial direct quotation from the subject author? If `true`, those quoted passages may be extractable as primary-adjacent material. |
| `quote_extraction_note` | string | No | If `contains_primary_quotes` is `true`, describe the extent and location of quoted material |

## 4. Contextual Source Additional Fields

These fields supplement the core fields for works the subject author engages with.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `referenced_by` | string | Yes | The subject author who engages with this work. Example: `malcolm-guite` |
| `engagement_type` | enum | Yes | One of: `influences`, `responds-to`, `critiques`, `extends`, `interprets`, `frequently-cites` |
| `engagement_note` | string | Yes | How the subject author uses or relates to this work. Example: `Guite builds his entire theory of Primary Imagination on Coleridge's distinction in chapters 13-14 of this work` |
| `engagement_works` | array[string] | No | Which of the subject author's works reference this contextual source. Example: `["faith-hope-and-poetry", "mariner"]` |
| `engagement_frequency` | enum | No | One of: `foundational` (pervasive across corpus), `major` (discussed in multiple works), `minor` (cited occasionally), `single-reference` |

## 5. Tertiary Source Additional Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_type` | enum | Yes | One of: `bibliography`, `encyclopedia-entry`, `catalog-record`, `database`, `index` |
| `coverage_note` | string | No | What aspect of the collection this reference covers |

## 6. Validation Rules

### work_id Format
- Lowercase alphanumeric and hyphens only
- Author slug and title slug separated by double hyphen (`--`)
- Maximum 128 characters
- Must be unique within the library

### source_class_note
- Minimum 10 characters
- Must reference at least one concrete signal (authorship, title, publisher, content)

### subject_headings
- Minimum 1 heading per work
- Use established LCSH patterns where possible
- Subdivisions separated by `--`

### genre_tags
- Minimum 1 tag per work
- Use the controlled vocabulary defined above; add new terms only when existing terms are inadequate

## 7. Examples

### Primary Source Example

```json
{
  "work_id": "malcolm-guite--faith-hope-and-poetry",
  "title": "Faith, Hope and Poetry: Theology and the Poetic Imagination",
  "author": "Malcolm Guite",
  "source_class": "primary",
  "source_class_note": "Listed as sole author on title page. Appears in Guite's bibliography on his personal website. First-person scholarly voice throughout.",
  "publication_year": 2012,
  "original_publication_year": 2010,
  "edition": "Paperback reissue",
  "publisher": "Ashgate",
  "isbn": "9781409449874",
  "format_ingested": "epub",
  "language": "en",
  "word_count": 85000,
  "genre_tags": ["monograph", "literary-criticism", "theology"],
  "subject_headings": [
    "Imagination--Religious aspects--Christianity",
    "English poetry--History and criticism",
    "Theology and literature"
  ],
  "ocr_quality": "not-applicable",
  "ingestion_date": "2026-02-16",
  "subject_author_id": "malcolm-guite",
  "work_type": "monograph",
  "chronological_position": 2,
  "voice_profile_eligible": true,
  "table_of_contents": [
    "Introduction: Five Ways of Reading Poetry",
    "The Dream of the Rood",
    "Shakespeare",
    "Sir John Davies",
    "John Donne",
    "George Herbert",
    "Henry Vaughan",
    "John Milton",
    "Samuel Taylor Coleridge",
    "Thomas Hardy",
    "Philip Larkin",
    "Geoffrey Hill",
    "Seamus Heaney"
  ]
}
```

### Secondary Source Example

```json
{
  "work_id": "holly-ordway--tolkien-modern-middle-ages",
  "title": "Tolkien's Modern Middle Ages",
  "author": "Holly Ordway (ed.)",
  "source_class": "secondary",
  "source_class_note": "Edited collection about Tolkien by external scholars. Guite contributed a chapter, which should be extracted as primary-adjacent. The editorial framing and other chapters are secondary.",
  "publication_year": 2023,
  "publisher": "Academic Press",
  "format_ingested": "pdf",
  "language": "en",
  "word_count": 120000,
  "genre_tags": ["edited-collection", "literary-criticism"],
  "subject_headings": [
    "Tolkien, J.R.R.--Criticism and interpretation",
    "Fantasy literature--History and criticism"
  ],
  "ingestion_date": "2026-02-16",
  "about_author_id": "malcolm-guite",
  "external_author": "Holly Ordway",
  "relationship": "edited-collection",
  "perspective_note": "Sympathetic scholarly collection; multiple contributors. Guite's own chapter should be tagged primary.",
  "contains_primary_quotes": true,
  "quote_extraction_note": "Chapter 7 is authored by Guite himself — extract and classify as primary"
}
```

### Contextual Source Example

```json
{
  "work_id": "st-coleridge--biographia-literaria",
  "title": "Biographia Literaria",
  "author": "Samuel Taylor Coleridge",
  "source_class": "contextual",
  "source_class_note": "Foundational source for Guite's theory of imagination. Coleridge's Primary/Secondary Imagination distinction (Ch. 13-14) is the backbone of Guite's argument in Faith, Hope and Poetry and Mariner.",
  "publication_year": 1817,
  "publisher": "Various",
  "format_ingested": "epub",
  "language": "en",
  "word_count": 95000,
  "genre_tags": ["autobiography", "literary-criticism", "philosophy"],
  "subject_headings": [
    "Coleridge, Samuel Taylor, 1772-1834",
    "Imagination (Philosophy)",
    "Poetry--Philosophy"
  ],
  "ingestion_date": "2026-02-16",
  "referenced_by": "malcolm-guite",
  "engagement_type": "foundational",
  "engagement_note": "Guite builds his entire theory of Primary Imagination on Coleridge's distinction in chapters 13-14. Referenced in virtually every Guite work.",
  "engagement_works": ["faith-hope-and-poetry", "mariner", "word-in-the-wilderness"],
  "engagement_frequency": "foundational"
}
```
