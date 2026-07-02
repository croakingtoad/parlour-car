#!/usr/bin/env python3
"""End-to-end epistolary ingestion: PDF → letter extraction → ParsedDocument → pipeline.

Extracts individual letters from a scanned epistolary PDF, builds a
ParsedDocument tree with each letter as a CHAPTER node (with recipient,
date, and title metadata), and feeds it through the Parlour Car ingestion
pipeline using the LetterStrategy chunking strategy.

Usage:
    cd parlour-car
    uv run python scripts/ingest_epistolary.py <pdf_path> \
        --author-id coleridge-samuel-taylor \
        --title "Collected Letters of Samuel Taylor Coleridge, Volume I" \
        --author "Samuel Taylor Coleridge" \
        --editor "Earl Leslie Griggs" \
        --year 1956 \
        [--first-page 46] [--last-page 711] [--max-letter 372] \
        [--dry-run]

Requires:
    - PostgreSQL + Neo4j running (make dev)
    - .env with API keys (ANTHROPIC_API_KEY, VOYAGE_API_KEY)
    - pymupdf installed (uv add pymupdf)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import structlog

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
    SectionType,
)

log = structlog.get_logger(__name__)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split letter body text into paragraphs on blank lines."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def build_parsed_document(
    letters: list[dict],
    *,
    title: str,
    author: str,
    editor: str = "",
    publication_year: int | None = None,
    source_path: str = "",
) -> ParsedDocument:
    """Build a ParsedDocument tree from extracted letter dicts.

    Each letter becomes a CHAPTER node with:
      - metadata: recipient, date, title ("Letter N. To [Recipient]")
      - children: PARAGRAPH nodes for each paragraph in the letter body

    Args:
        letters: List of letter dicts from extract_letters_from_pdf().
        title: Collection title.
        author: Author name.
        editor: Editor name.
        publication_year: Year of publication.
        source_path: Original PDF path (for provenance).

    Returns:
        A ParsedDocument ready for the ingestion pipeline.
    """
    root = DocumentNode(
        node_type=NodeType.BOOK,
        section_type=SectionType.CHAPTER,
        metadata={"title": title, "author": author},
    )

    raw_text_parts: list[str] = []
    total_words = 0

    for letter in sorted(letters, key=lambda x: x["num"]):
        letter_title = f"Letter {letter['num']}. To {letter['recipient']}"

        # Build metadata for LetterStrategy
        letter_meta: dict[str, str | int | bool | list[str]] = {
            "title": letter_title,
            "recipient": letter["recipient"],
            "letter_number": letter["num"],
        }
        if letter.get("date"):
            letter_meta["date"] = letter["date"]
        if letter.get("address"):
            letter_meta["address"] = letter["address"]
        if letter.get("manuscript_info"):
            letter_meta["manuscript_info"] = letter["manuscript_info"]

        # Create CHAPTER node for this letter
        chapter_node = DocumentNode(
            node_type=NodeType.CHAPTER,
            section_type=SectionType.CHAPTER,
            metadata=letter_meta,
        )

        # Add heading
        chapter_node.children.append(
            DocumentNode(
                node_type=NodeType.HEADING,
                text=letter_title,
            )
        )

        # Split body into paragraphs
        body = letter.get("body", "")
        paragraphs = _split_into_paragraphs(body)
        for para_text in paragraphs:
            chapter_node.children.append(
                DocumentNode(
                    node_type=NodeType.PARAGRAPH,
                    text=para_text,
                )
            )

        # Also set the text on the chapter node itself (for collect_text)
        chapter_node.text = body

        root.children.append(chapter_node)
        raw_text_parts.append(body)
        total_words += letter.get("word_count", len(body.split()))

    raw_text = "\n\n".join(raw_text_parts)

    metadata = DocumentMetadata(
        title=title,
        author=author,
        publisher=f"ed. {editor}" if editor else None,
        publication_date=str(publication_year) if publication_year else None,
        word_count=total_words,
    )

    return ParsedDocument(
        source_path=source_path,
        format="pdf",
        metadata=metadata,
        tree=root,
        raw_text=raw_text,
        parse_warnings=[],
    )


async def run_ingestion(
    document: ParsedDocument,
    *,
    subject_author_id: str,
    source_class: str = "primary",
    genre_tags: list[str] | None = None,
) -> dict:
    """Run the ingestion pipeline on a pre-built ParsedDocument.

    Args:
        document: ParsedDocument with letter tree.
        subject_author_id: Author slug (e.g. "coleridge-samuel-taylor").
        source_class: Source classification (default: "primary").
        genre_tags: Genre tags for chunking strategy selection.

    Returns:
        IngestionResult as dict.
    """
    from author_library.config import Settings
    from author_library.embeddings import ProviderRegistry
    from author_library.storage.manager import StorageManager
    from author_library.tools.ingest import run_post_ingestion_hooks
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
            "genre_tags": genre_tags or ["correspondence", "letters"],
        }

        result = await pipeline.ingest_document(
            document,
            subject_author_id=subject_author_id,
            metadata_hints=metadata_hints,
        )

        # Run post-pipeline hooks (cross-work analysis, backup, QG2, report)
        # No cache_manager or task_queue available in script context — that's fine
        response = await run_post_ingestion_hooks(
            result=result,
            subject_author_id=subject_author_id,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

        return response
    finally:
        await storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract letters from a scanned epistolary PDF and ingest into Parlour Car"
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the scanned PDF")
    parser.add_argument(
        "--author-id",
        required=True,
        help="Subject author slug (e.g. coleridge-samuel-taylor)",
    )
    parser.add_argument(
        "--title",
        default="Collected Letters of Samuel Taylor Coleridge, Volume I (1785-1800)",
        help="Collection title",
    )
    parser.add_argument(
        "--author",
        default="Samuel Taylor Coleridge",
        help="Author name",
    )
    parser.add_argument(
        "--editor",
        default="Earl Leslie Griggs",
        help="Editor name",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=1956,
        help="Publication year",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        default=46,
        help="0-indexed page where letters begin (default: 46)",
    )
    parser.add_argument(
        "--last-page",
        type=int,
        default=711,
        help="0-indexed page where letters end (default: 711)",
    )
    parser.add_argument(
        "--max-letter",
        type=int,
        default=372,
        help="Maximum expected letter number (default: 372)",
    )
    parser.add_argument(
        "--source-class",
        default="primary",
        help="Source classification (default: primary)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and build document tree but don't ingest",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Extract letters from PDF
    # Import from sibling script in scripts/ directory
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from preprocess_epistolary import extract_letters_from_pdf

    print(f"Extracting letters from {args.pdf_path.name}...")
    letters = extract_letters_from_pdf(
        args.pdf_path,
        first_letter_page=args.first_page,
        last_letter_page=args.last_page,
        max_letter_number=args.max_letter,
    )
    print(f"  Extracted {len(letters)} letters")
    print(f"  With dates: {sum(1 for l in letters if l['date'])}")
    total_words = sum(l['word_count'] for l in letters)
    print(f"  Total words: {total_words:,}")

    # Step 2: Build ParsedDocument
    document = build_parsed_document(
        letters,
        title=args.title,
        author=args.author,
        editor=args.editor,
        publication_year=args.year,
        source_path=str(args.pdf_path),
    )
    print(f"\nParsedDocument built:")
    print(f"  Title: {document.metadata.title}")
    print(f"  Chapters (letters): {len(document.tree.children)}")
    print(f"  Word count: {document.metadata.word_count:,}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping ingestion pipeline")
        # Show first 5 letters for verification
        print("\nFirst 5 letters:")
        for child in document.tree.children[:5]:
            title = child.metadata.get("title", "?")
            recipient = child.metadata.get("recipient", "?")
            date = child.metadata.get("date", "N/A")
            paras = sum(1 for c in child.children if c.node_type == NodeType.PARAGRAPH)
            print(f"  {title} | date={date} | {paras} paragraphs")

        missing = set(range(1, args.max_letter + 1)) - {
            child.metadata.get("letter_number") for child in document.tree.children
        }
        if missing:
            print(f"\nMissing letters ({len(missing)}): {sorted(missing)}")
        return

    # Step 3: Ingest through pipeline
    print(f"\nStarting ingestion pipeline...")
    print(f"  Author ID: {args.author_id}")
    print(f"  Source class: {args.source_class}")
    print(f"  Genre tags: correspondence, letters")

    result = asyncio.run(
        run_ingestion(
            document,
            subject_author_id=args.author_id,
            source_class=args.source_class,
            genre_tags=["correspondence", "letters"],
        )
    )

    print(f"\nIngestion complete!")
    print(f"  Work ID: {result['work_id']}")
    print(f"  Source class: {result['source_class']}")
    print(f"  Processing route: {result['processing_route']}")
    print(f"  Chunks: {result.get('post_ingestion_stats', {}).get('total_chunks', 0)}")
    print(f"  Embeddings: {result.get('post_ingestion_stats', {}).get('embeddings_stored', 0)}")
    print(f"  Entities: {result.get('post_ingestion_stats', {}).get('entity_count', 0)}")
    print(f"  Edges: {result.get('post_ingestion_stats', {}).get('edge_count', 0)}")
    if result.get("errors"):
        print(f"\n  Errors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
