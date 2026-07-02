#!/usr/bin/env python3
"""Ingest a poetry collection epub with mixed prose + poetry chapters.

Parses the epub, identifies chapter essay intros vs individual poems,
builds a custom ParsedDocument tree, and feeds it through the ingestion
pipeline using PoetryCollectionStrategy.

Each poem becomes its own macro chunk (first-class retrievable unit).
Each essay intro becomes its own macro chunk with meso/micro subdivision.

Usage:
    cd /home/marty/repos/parlour/parlour-car

    # Dry run — parse epub, show tree structure, no ingestion
    uv run python scripts/ingest_poetry_collection.py \\
        /path/to/collection.epub \\
        --author-id john-odonohue \\
        --dry-run

    # Full ingestion
    uv run python scripts/ingest_poetry_collection.py \\
        /path/to/collection.epub \\
        --author-id john-odonohue \\
        --title "To Bless the Space Between Us" \\
        --author "John O'Donohue" \\
        --source-class primary

Reusable for any poetry collection where chapters contain:
  - An optional prose introduction/essay
  - Individual poems (shorter pieces with verse formatting)

The script identifies poems vs prose by:
  1. NCX/TOC structure (each poem has its own TOC entry)
  2. Content length heuristic (poems typically < 3000 chars)
  3. Line-break density (poetry has more line breaks per word)
"""

from __future__ import annotations

import argparse
import asyncio
import html.parser
import re
import sys
import zipfile
from pathlib import Path

# Ensure the project src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
    SectionType,
)


# ---------------------------------------------------------------------------
# Epub parsing helpers
# ---------------------------------------------------------------------------

class _TextExtractor(html.parser.HTMLParser):
    """Extract plain text from HTML, preserving line breaks."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._in_body = False
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self._in_body = True
        if tag in ("style", "script"):
            self._skip = True
        if tag == "br":
            self.parts.append("\n")
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "body":
            self._in_body = False
        if tag in ("style", "script"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if self._in_body and not self._skip:
            self.parts.append(data)


def _html_to_text(html_content: str) -> str:
    """Convert HTML to plain text preserving paragraph breaks."""
    ext = _TextExtractor()
    ext.feed(html_content)
    text = "".join(ext.parts).strip()
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clean_title(raw: str) -> str:
    """Clean epub title formatting (remove excess spaces from small-caps styling).

    Many epub generators produce small-caps headings where the first letter of
    each word is separated: "F OR  L IGHT" → "For Light", "M ATINS" → "Matins".

    Only applies transformations if the title looks like it has small-caps
    formatting (all-uppercase text with single-letter separation). Already
    properly-cased titles like "A Morning Offering" pass through unchanged.
    """
    raw = raw.strip()
    if not raw:
        return raw

    # Detect if this title has small-caps formatting:
    # - Contains double-spaces (word separator in small-caps) OR
    # - Matches pattern of all-uppercase single-letter-separated words
    has_double_spaces = "  " in raw
    # Check if mostly uppercase (small-caps artifact)
    alpha_chars = [c for c in raw if c.isalpha()]
    uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)

    if not has_double_spaces and uppercase_ratio < 0.7:
        # Title is already properly formatted — no cleanup needed
        return raw

    # Step 1: Merge "X REST" (single uppercase letter + space + ALL-UPPERCASE continuation)
    # "M ORNING" → "MORNING", "B LESSING" → "BLESSING", "F OR" → "FOR"
    # Only match when the continuation is ALL UPPERCASE (not mixed case like "Morning")
    cleaned = re.sub(r"\b([A-Z]) ([A-Z][A-Z]+)\b", r"\1\2", raw)

    # Step 2: Merge two adjacent single uppercase letters before a word boundary
    # "I N" → "IN", "T O" → "TO", "A T" → "AT"
    cleaned = re.sub(
        r"\b([A-Z]) ([A-Z])\b(?=\s{2}|\s+[A-Z]|\s*$)",
        r"\1\2",
        cleaned,
    )

    # Step 3: Collapse double+ spaces to single space
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # Title case
    return cleaned.title()


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

_CHAPTER_PATTERN = re.compile(r"^(\d+)\s+(.+)$")

_FRONT_MATTER_TITLES = {
    "title page", "dedication", "contents", "copyright",
    "also by", "about the author", "acknowledgments",
}

_BACK_MATTER_TITLES = {
    "acknowledgments", "about the author", "also by",
    "copyright", "notes",
}

# Titles that indicate a standalone essay (not a poem)
_ESSAY_TITLES = {
    "introduction", "preface", "foreword", "afterword",
    "epilogue", "prologue",
}


def _classify_toc_entry(title: str, content: str = "") -> str:
    """Classify a TOC entry as 'chapter_intro', 'poem', 'essay', or 'skip'.

    Uses title patterns first, then falls back to content heuristics:
    long prose content (> 3000 chars) without poetry formatting is classified
    as an essay rather than a poem.
    """
    lower = title.lower().strip()

    # Check front/back matter (skip)
    if lower in _FRONT_MATTER_TITLES:
        return "skip"
    for bm in _BACK_MATTER_TITLES:
        if lower.startswith(bm):
            return "skip"

    # Chapter intros: "1 Beginnings", "2 Desires"
    if _CHAPTER_PATTERN.match(title.strip()):
        return "chapter_intro"

    # Known essay titles
    if lower in _ESSAY_TITLES:
        return "essay"

    # Content-based heuristic: long prose = essay, not poem
    if content and len(content) > 3000:
        words = len(content.split())
        lines = [ln for ln in content.split("\n") if ln.strip()]
        if lines:
            avg_line_len = sum(len(ln.strip()) for ln in lines) / len(lines)
            # Prose has longer average line length than poetry
            if avg_line_len > 60 or words > 1500:
                return "essay"

    return "poem"  # default: individual blessing/poem


def _is_poetry_content(text: str) -> bool:
    """Heuristic: does this text look like poetry (vs prose)?"""
    if not text.strip():
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    words = len(text.split())
    if words == 0:
        return False
    # Poetry has more line breaks per word
    lines_per_100_words = len(lines) / words * 100
    # Average line length in poetry is shorter
    avg_line_len = sum(len(ln.strip()) for ln in lines) / max(len(lines), 1)
    return lines_per_100_words > 8 or avg_line_len < 50


# ---------------------------------------------------------------------------
# ParsedDocument builder
# ---------------------------------------------------------------------------

def parse_poetry_collection_epub(
    epub_path: str | Path,
    *,
    title: str | None = None,
    author: str | None = None,
) -> ParsedDocument:
    """Parse a poetry collection epub into a structured ParsedDocument.

    Returns a tree where:
      - Each chapter = CHAPTER node (with essay intro as direct paragraphs)
      - Each poem = POEM node with STANZA children
      - Introduction/closing essays = CHAPTER nodes with PARAGRAPH children
    """
    epub_path = Path(epub_path)

    with zipfile.ZipFile(epub_path) as zf:
        # Read TOC from NCX
        ncx_files = [n for n in zf.namelist() if n.endswith(".ncx")]
        toc_entries: list[dict[str, str]] = []
        if ncx_files:
            ncx_content = zf.read(ncx_files[0]).decode("utf-8", errors="replace")
            toc_entries = _parse_ncx(ncx_content)

        # Read all HTML files in spine order
        html_files = sorted(
            [n for n in zf.namelist() if n.endswith((".html", ".xhtml", ".htm"))],
        )

        # Build a map of src → html content
        file_contents: dict[str, str] = {}
        for f in html_files:
            raw = zf.read(f).decode("utf-8", errors="replace")
            file_contents[f] = _html_to_text(raw)

        # Try to extract title/author from OPF if not provided
        if not title or not author:
            opf_files = [n for n in zf.namelist() if n.endswith(".opf")]
            if opf_files:
                opf = zf.read(opf_files[0]).decode("utf-8", errors="replace")
                if not title:
                    m = re.search(r"<dc:title>(.*?)</dc:title>", opf)
                    if m:
                        title = m.group(1).strip()
                if not author:
                    m = re.search(r"<dc:creator[^>]*>(.*?)</dc:creator>", opf)
                    if m:
                        author = m.group(1).strip()

    title = title or epub_path.stem
    author = author or "Unknown"

    # Build the document tree
    root = DocumentNode(
        node_type=NodeType.BOOK,
        section_type=SectionType.CHAPTER,
        metadata={"title": title, "author": author},
    )

    # Group TOC entries by chapter
    current_chapter: DocumentNode | None = None
    raw_text_parts: list[str] = []
    total_words = 0
    poem_count = 0
    essay_count = 0

    # Map TOC entries to their content
    toc_with_content: list[dict] = []
    for entry in toc_entries:
        src = entry.get("src", "")
        # Find matching file
        matching = [f for f in file_contents if src in f or f.endswith(src.split("/")[-1])]
        if matching:
            content = file_contents[matching[0]]
        else:
            content = ""
        toc_with_content.append({**entry, "content": content})

    for entry in toc_with_content:
        entry_title = entry["title"]
        content = entry["content"]
        cleaned_title = _clean_title(entry_title)
        entry_type = _classify_toc_entry(cleaned_title, content)

        if not content.strip() or entry_type == "skip":
            continue

        total_words += len(content.split())
        raw_text_parts.append(content)

        if entry_type == "chapter_intro":
            # Start a new chapter
            chapter_match = _CHAPTER_PATTERN.match(cleaned_title.strip())
            chapter_name = chapter_match.group(2).strip() if chapter_match else cleaned_title

            current_chapter = DocumentNode(
                node_type=NodeType.CHAPTER,
                section_type=SectionType.CHAPTER,
                metadata={"title": chapter_name},
            )
            root.children.append(current_chapter)

            # The chapter intro content is prose — add as paragraphs
            # Store text ONLY in paragraph children (not node.text) to avoid
            # duplication when collect_text() walks the tree.
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            for para in paragraphs:
                current_chapter.children.append(DocumentNode(
                    node_type=NodeType.PARAGRAPH,
                    text=para,
                ))
            essay_count += 1

        elif entry_type == "essay":
            # Standalone essay (Introduction, closing essay)
            section_type = SectionType.PREFACE if "introduction" in cleaned_title.lower() else SectionType.CHAPTER

            # Check if this is a very long essay (like the closing essay)
            is_closing = len(content) > 10000

            essay_node = DocumentNode(
                node_type=NodeType.CHAPTER,
                section_type=section_type if not is_closing else SectionType.BACK_MATTER,
                metadata={"title": cleaned_title},
            )

            # Store text ONLY in paragraph children to avoid duplication
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            for para in paragraphs:
                essay_node.children.append(DocumentNode(
                    node_type=NodeType.PARAGRAPH,
                    text=para,
                ))

            root.children.append(essay_node)
            essay_count += 1

        elif entry_type == "poem":
            # Individual poem/blessing
            poem_node = DocumentNode(
                node_type=NodeType.POEM,
                section_type=SectionType.CHAPTER,
                metadata={"title": cleaned_title},
                # DO NOT set node.text — store text only in STANZA children
                # so collect_text() doesn't double-count.
            )

            # Split into stanzas on blank lines
            stanzas = [s.strip() for s in re.split(r"\n\s*\n", content) if s.strip()]

            # First stanza might be a dedication (short, often "For [name]")
            if stanzas and len(stanzas[0].split()) <= 8 and stanzas[0].startswith("For "):
                poem_node.metadata["dedication"] = stanzas[0]
                stanzas = stanzas[1:]

            for stanza_text in stanzas:
                poem_node.children.append(DocumentNode(
                    node_type=NodeType.STANZA,
                    text=stanza_text,
                ))

            # Attach to current chapter or root
            if current_chapter is not None:
                current_chapter.children.append(poem_node)
            else:
                root.children.append(poem_node)
            poem_count += 1

    metadata = DocumentMetadata(
        title=title,
        author=author,
        word_count=total_words,
    )

    doc = ParsedDocument(
        source_path=str(epub_path),
        format="epub",
        metadata=metadata,
        tree=root,
        raw_text="\n\n".join(raw_text_parts),
    )

    print(f"Parsed: {title} by {author}")
    print(f"  {poem_count} poems, {essay_count} essays/intros")
    print(f"  {total_words:,} words total")
    print(f"  {len(root.children)} top-level nodes")

    return doc


def _parse_ncx(ncx_content: str) -> list[dict[str, str]]:
    """Parse NCX file to extract TOC entries with titles and src references."""
    entries: list[dict[str, str]] = []
    # Extract navPoints
    nav_points = re.findall(
        r"<navPoint[^>]*>.*?<text>(.*?)</text>.*?<content\s+src=[\"']([^\"']+)[\"']",
        ncx_content,
        re.DOTALL,
    )
    for title, src in nav_points:
        entries.append({"title": title.strip(), "src": src.strip()})
    return entries


# ---------------------------------------------------------------------------
# Tree display (for --dry-run)
# ---------------------------------------------------------------------------

def print_tree(node: DocumentNode, indent: int = 0) -> None:
    """Print the document tree structure."""
    prefix = "  " * indent
    title = node.metadata.get("title", "")
    text_preview = (node.text or "")[:80].replace("\n", " ")
    wc = len((node.text or "").split())

    type_label = node.node_type.value.upper()
    section_label = f" [{node.section_type.value}]" if node.section_type != SectionType.CHAPTER else ""

    if title:
        print(f"{prefix}{type_label}{section_label}: {title} ({wc} words)")
    elif text_preview:
        print(f"{prefix}{type_label}{section_label}: {text_preview}... ({wc} words)")
    else:
        print(f"{prefix}{type_label}{section_label} (empty)")

    for child in node.children:
        # Don't recurse into stanza/paragraph details for dry-run
        if child.node_type in (NodeType.STANZA, NodeType.PARAGRAPH):
            continue
        print_tree(child, indent + 1)


# ---------------------------------------------------------------------------
# Ingestion runner
# ---------------------------------------------------------------------------

async def run_ingestion(
    document: ParsedDocument,
    *,
    subject_author_id: str,
    source_class: str = "primary",
) -> dict:
    """Run the full ingestion pipeline on a pre-built ParsedDocument."""
    from author_library.config import Settings
    from author_library.embeddings import ProviderRegistry
    from author_library.storage.manager import StorageManager
    from author_library.tools.ingestion_pipeline import IngestionPipeline

    settings = Settings()
    storage = StorageManager(settings.database)
    await storage.connect()

    try:
        embedding_provider = ProviderRegistry.create(settings)
        pipeline = IngestionPipeline(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

        metadata_hints = {
            "source_class": source_class,
            "genre_tags": ["poetry_collection"],
        }

        result = await pipeline.ingest_document(
            document,
            subject_author_id=subject_author_id,
            metadata_hints=metadata_hints,
        )

        print(f"\nIngestion complete: {result.work_id}")
        print(f"  Source class: {result.source_class}")
        print(f"  Route: {result.processing_route}")
        print(f"  Chunks: {result.total_chunks}")
        print(f"  By granularity: {result.chunks_by_granularity}")
        print(f"  Embeddings: {result.embeddings_stored}")
        print(f"  Entities: {result.entity_count}")
        print(f"  Edges: {result.edge_count}")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for e in result.errors[:5]:
                print(f"    - {e}")

        return {
            "work_id": result.work_id,
            "total_chunks": result.total_chunks,
            "embeddings_stored": result.embeddings_stored,
            "entity_count": result.entity_count,
            "edge_count": result.edge_count,
            "errors": result.errors,
        }
    finally:
        await storage.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a poetry collection epub (mixed prose + poetry).",
    )
    parser.add_argument("epub_path", help="Path to the epub file")
    parser.add_argument("--author-id", required=True, help="Subject author slug (e.g., john-odonohue)")
    parser.add_argument("--title", help="Override title from epub metadata")
    parser.add_argument("--author", help="Override author from epub metadata")
    parser.add_argument("--source-class", default="primary", choices=["primary", "secondary", "contextual"])
    parser.add_argument("--dry-run", action="store_true", help="Parse only, show tree structure, no ingestion")

    args = parser.parse_args()

    epub_path = Path(args.epub_path)
    if not epub_path.exists():
        print(f"Error: {epub_path} does not exist", file=sys.stderr)
        sys.exit(1)

    document = parse_poetry_collection_epub(
        epub_path,
        title=args.title,
        author=args.author,
    )

    if args.dry_run:
        print("\n--- Document Tree ---")
        print_tree(document.tree)
        print(f"\nTotal top-level nodes: {len(document.tree.children)}")
        # Count poems and essays
        from author_library.chunking._tree_utils import find_nodes
        poems = find_nodes(document.tree, NodeType.POEM)
        chapters = find_nodes(document.tree, NodeType.CHAPTER)
        print(f"POEM nodes: {len(poems)}")
        print(f"CHAPTER nodes: {len(chapters)}")
        print("\n--- Chunking Preview ---")
        from author_library.chunking.poetry_collection import PoetryCollectionStrategy
        strategy = PoetryCollectionStrategy()
        chunks = strategy.chunk(document, "dry-run", args.source_class)
        print(f"Total chunks: {len(chunks)}")
        by_gran = {}
        for c in chunks:
            g = str(c.granularity)
            by_gran[g] = by_gran.get(g, 0) + 1
        print(f"By granularity: {by_gran}")
        # Show macro chunks
        print("\nMACRO chunks:")
        for c in chunks:
            if str(c.granularity) == "macro":
                content_type = c.metadata.get("content_type", "?")
                wc = len(c.text.split())
                print(f"  [{content_type}] {c.chapter or c.section or '(untitled)'} — {wc} words")
        return

    # Full ingestion
    result = asyncio.run(
        run_ingestion(
            document,
            subject_author_id=args.author_id,
            source_class=args.source_class,
        )
    )

    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
