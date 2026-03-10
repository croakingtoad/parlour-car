"""Format-agnostic document tree data model.

All parsers produce a ParsedDocument containing a recursive DocumentNode tree,
extracted metadata, and optional raw text. This is the canonical intermediate
representation between raw document formats and the chunking / indexing pipeline.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class SectionType(StrEnum):
    """Classification of document sections for pipeline routing.

    Controls how each section is processed during ingestion:
    - CHAPTER/BACK_MATTER: full pipeline (chunk, annotate, embed, extract, voice)
    - PREFACE: chunk, annotate, embed, entity extract — but NOT voice profiling
    - BIBLIOGRAPHY: skip chunking; parse cited works for acquisition candidates
    - INDEX: skip chunking; parse terms for vocabulary proposals
    - TOC: skip all processing; use for structural metadata only
    - FRONT_MATTER: minimal processing; extract catalog metadata as single nano chunk
    """

    FRONT_MATTER = "front_matter"
    TABLE_OF_CONTENTS = "toc"
    PREFACE = "preface"
    CHAPTER = "chapter"
    BIBLIOGRAPHY = "bibliography"
    INDEX = "index"
    BACK_MATTER = "back_matter"


class NodeType(StrEnum):
    """Semantic types for document tree nodes."""

    BOOK = "book"
    FRONT_MATTER = "front_matter"
    CHAPTER = "chapter"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    BLOCK_QUOTE = "block_quote"
    POEM = "poem"
    STANZA = "stanza"
    LINE = "line"
    FOOTNOTE = "footnote"
    ENDNOTE = "endnote"
    BIBLIOGRAPHY = "bibliography"
    BIB_ENTRY = "bib_entry"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"


class DocumentNode(BaseModel):
    """A node in the recursive document tree."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_type: NodeType
    section_type: SectionType = SectionType.CHAPTER
    text: str = ""
    children: list[DocumentNode] = Field(default_factory=list)
    metadata: dict[str, str | int | bool | list[str]] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Extracted metadata about the document."""

    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    publisher: str | None = None
    isbn: str | None = None
    language: str = "en"
    table_of_contents: list[str] = Field(default_factory=list)
    word_count: int = 0


class ParsedDocument(BaseModel):
    """The complete output of parsing a document."""

    source_path: str
    format: str  # epub, pdf, txt, html, docx
    metadata: DocumentMetadata
    tree: DocumentNode  # root node (typically NodeType.BOOK)
    raw_text: str = ""  # full plain text concatenation for word count / fallback
    parse_warnings: list[str] = Field(default_factory=list)
