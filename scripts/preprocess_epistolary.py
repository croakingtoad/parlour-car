#!/usr/bin/env python3
"""Pre-processor for scanned epistolary (letter collection) PDFs.

Extracts individual letters from OCR'd PDF scans of letter collections,
detects letter boundaries, corrects common OCR digit errors, and
outputs structured text files that the Parlour Car ingestion pipeline
can process using the LetterStrategy chunking strategy.

Usage:
    uv run python scripts/preprocess_epistolary.py <pdf_path> [--output <dir>] [--dry-run]

The output is a single text file with letter boundaries marked as:
    === LETTER N ===
    RECIPIENT: <name>
    DATE: <date>
    ADDRESS: <address info>
    MS: <manuscript info>
    ---
    <letter body>

This can then be ingested via:
    uv run python -m author_library ingest <output_file> --genre correspondence
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# --- OCR correction rules ---

# Common OCR digit substitutions in scanned books
# The 1956 Oxford Clarendon Press typeface causes:
#   - Leading "3" misread as "8" (most common)
#   - Leading digit "1" prepended to 2-digit numbers (e.g., 30 -> 130)
#   - Leading dash or artifact before number


def _correct_letter_number(
    raw_num: int, max_expected: int, already_found: set[int]
) -> int:
    """Apply OCR correction rules to a letter number.

    Only applies corrections when the raw number is out of range AND
    the corrected number is not already found (to avoid clobbering
    real letters like 130 by "correcting" them to 30).

    Args:
        raw_num: The number as read by OCR.
        max_expected: Maximum expected letter number in the collection.
        already_found: Set of letter numbers already confirmed.

    Returns:
        Corrected letter number.
    """
    # If the number is in valid range, keep it as-is
    if 1 <= raw_num <= max_expected:
        return raw_num

    s = str(raw_num)

    # Rule 1: 8xx -> 3xx (leading "3" misread as "8")
    # Only applies to numbers > max_expected (e.g., 802 -> 302)
    if len(s) == 3 and s[0] == "8":
        corrected = int("3" + s[1:])
        if corrected <= max_expected and corrected not in already_found:
            return corrected

    # Rule 2: 1xxx -> xxx (leading "1" prepended, e.g., 1269 -> 269)
    if len(s) == 4 and s[0] == "1":
        corrected = int(s[1:])
        if 1 <= corrected <= max_expected and corrected not in already_found:
            return corrected

    return raw_num


# --- Letter boundary detection ---

# Primary pattern: "N. To [Recipient]"
_LETTER_START = re.compile(
    r"^[-–—]?\s*(\d{1,4})\.\s+To\s+(.+?)$", re.MULTILINE
)

# Date patterns found in letter headers
_DATE_PATTERN = re.compile(
    r"(?:"
    # "Month Day, Year" or "Month Day [Year]"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(?:\[?\d{4}\]?)?"
    r"|"
    # "[Month-range Year]" like "[February-March 1791]"
    r"\[(?:(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"[^]]*\d{4})\]"
    r"|"
    # "Day Month Year" like "28 November 1791"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)"
    r"\s+\d{4}"
    r"|"
    # Standalone year in brackets after month+day like "October 16th [1791]"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\s+\[\d{4}\]"
    r"|"
    # "Early/Mid/Late Month [Year]"
    r"(?:Early|Mid|Late)\s+(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)"
    r"(?:\s+\[?\d{4}\]?)?"
    r")",
    re.IGNORECASE,
)

# Address line pattern
_ADDRESS_PATTERN = re.compile(
    r"^Address:\s*(.+?)$", re.MULTILINE
)

# Manuscript source pattern
_MS_PATTERN = re.compile(
    r"^(?:MS\.|Pub\.|Transcript)\s*(.+?)$", re.MULTILINE
)

# Page header patterns to strip (running headers from printed pages)
_PAGE_HEADER = re.compile(
    r"^(?:\d+\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\s*$"
    r"|"
    r"^\[\d{1,4}\s*$"
    r"|"
    r"^\d{1,4}\]\s*$"
    r"|"
    r"^To\s+\w.{5,40}\s*$",
    re.MULTILINE,
)


def extract_letters_from_pdf(
    pdf_path: Path,
    *,
    first_letter_page: int = 47,
    last_letter_page: int = 711,
    max_letter_number: int = 372,
) -> list[dict]:
    """Extract individual letters from a scanned epistolary PDF.

    Args:
        pdf_path: Path to the PDF file.
        first_letter_page: 0-indexed page where letters begin.
        last_letter_page: 0-indexed page where letters end (exclusive).
        max_letter_number: Expected maximum letter number in the collection.

    Returns:
        List of letter dicts with keys: num, recipient, date, address,
        manuscript_info, body, raw_num, page_start.
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    total_pages = doc.page_count
    log.info(
        "extracting_letters",
        pdf=str(pdf_path),
        pages=total_pages,
        letter_range=f"{first_letter_page}-{last_letter_page}",
    )

    # Extract full text from letter pages
    full_text = ""
    for i in range(first_letter_page, min(last_letter_page, total_pages)):
        full_text += doc[i].get_text() + "\n"
    doc.close()

    # Find all letter boundaries
    raw_matches = list(_LETTER_START.finditer(full_text))
    log.info("raw_letter_matches", count=len(raw_matches))

    # Two-pass OCR correction:
    # Pass 1: collect all in-range numbers (definitely correct)
    # Pass 2: apply OCR corrections only to out-of-range numbers
    confirmed_nums: set[int] = set()
    for m in raw_matches:
        raw_num = int(m.group(1))
        if 1 <= raw_num <= max_letter_number:
            confirmed_nums.add(raw_num)

    # Build letter records with OCR correction
    letters = []
    seen_nums = set()

    for m in raw_matches:
        raw_num = int(m.group(1))
        corrected_num = _correct_letter_number(
            raw_num, max_letter_number, confirmed_nums
        )
        recipient = m.group(2).strip()

        # Deduplicate: if we already have this corrected number, skip
        if corrected_num in seen_nums:
            continue
        seen_nums.add(corrected_num)

        letters.append(
            {
                "raw_num": raw_num,
                "num": corrected_num,
                "recipient": recipient,
                "pos": m.start(),
                "end_pos": None,
            }
        )

    # Sort by position in text
    letters.sort(key=lambda x: x["pos"])

    # Set end positions
    for i in range(len(letters)):
        if i + 1 < len(letters):
            letters[i]["end_pos"] = letters[i + 1]["pos"]
        else:
            letters[i]["end_pos"] = len(full_text)

    # Extract metadata and body for each letter
    for letter in letters:
        header_and_body = full_text[letter["pos"] : letter["end_pos"]]

        # Extract date (search first 800 chars of letter)
        header = header_and_body[:800]
        date_match = _DATE_PATTERN.search(header)
        letter["date"] = date_match.group().strip() if date_match else None

        # Extract address
        addr_match = _ADDRESS_PATTERN.search(header)
        letter["address"] = addr_match.group(1).strip() if addr_match else None

        # Extract manuscript info
        ms_match = _MS_PATTERN.search(header)
        letter["manuscript_info"] = ms_match.group().strip() if ms_match else None

        # Body: everything after the header section
        # The header typically ends after the date line or "Dear [Name]"
        # For simplicity, include the full text (the chunking strategy
        # and annotation layer handle the rest)
        letter["body"] = header_and_body.strip()

        # Word count
        letter["word_count"] = len(letter["body"].split())

    log.info(
        "letters_extracted",
        total=len(letters),
        with_dates=sum(1 for l in letters if l["date"]),
        coverage=f"{len(letters)}/{max_letter_number}",
    )

    return letters


def write_structured_output(
    letters: list[dict],
    output_path: Path,
    *,
    title: str = "Collected Letters",
    author: str = "Samuel Taylor Coleridge",
    editor: str = "",
    publication_year: int | None = None,
) -> None:
    """Write extracted letters as a structured text file for ingestion.

    The output format uses clear delimiters that the parser can detect
    as chapter boundaries for the LetterStrategy.
    """
    lines = []
    lines.append(f"# {title}")
    lines.append(f"# Author: {author}")
    if editor:
        lines.append(f"# Editor: {editor}")
    if publication_year:
        lines.append(f"# Published: {publication_year}")
    lines.append(f"# Letters: {len(letters)}")
    lines.append("")

    for letter in sorted(letters, key=lambda x: x["num"]):
        lines.append(f"=== LETTER {letter['num']} ===")
        lines.append(f"RECIPIENT: {letter['recipient']}")
        if letter.get("date"):
            lines.append(f"DATE: {letter['date']}")
        if letter.get("address"):
            lines.append(f"ADDRESS: {letter['address']}")
        if letter.get("manuscript_info"):
            lines.append(f"MS: {letter['manuscript_info']}")
        lines.append("---")
        lines.append(letter["body"])
        lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("output_written", path=str(output_path), letters=len(letters))


def write_manifest(letters: list[dict], output_path: Path) -> None:
    """Write a JSON manifest of extracted letters for debugging/verification."""
    manifest = []
    for l in sorted(letters, key=lambda x: x["num"]):
        manifest.append(
            {
                "letter_number": l["num"],
                "raw_ocr_number": l["raw_num"],
                "recipient": l["recipient"],
                "date": l["date"],
                "word_count": l["word_count"],
                "has_address": l["address"] is not None,
                "has_manuscript_info": l["manuscript_info"] is not None,
            }
        )
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("manifest_written", path=str(output_path), entries=len(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process scanned epistolary PDF for Parlour Car ingestion"
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the scanned PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as PDF)",
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
        "--dry-run",
        action="store_true",
        help="Print stats without writing output",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    letters = extract_letters_from_pdf(
        args.pdf_path,
        first_letter_page=args.first_page,
        last_letter_page=args.last_page,
        max_letter_number=args.max_letter,
    )

    if args.dry_run:
        print(f"\nExtracted {len(letters)} letters")
        print(f"With dates: {sum(1 for l in letters if l['date'])}")
        total_words = sum(l["word_count"] for l in letters)
        print(f"Total words: {total_words:,}")
        print(f"\nFirst 10:")
        for l in sorted(letters, key=lambda x: x["num"])[:10]:
            print(
                f"  #{l['num']:3d} To {l['recipient']:<30s} "
                f"date={l['date'] or 'N/A':<25s} words={l['word_count']}"
            )
        missing = set(range(1, args.max_letter + 1)) - {l["num"] for l in letters}
        if missing:
            print(f"\nMissing letters ({len(missing)}): {sorted(missing)}")
        return

    output_dir = args.output or args.pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.pdf_path.stem.split("--")[0].strip().replace(" ", "-").lower()
    txt_path = output_dir / f"{stem}-processed.txt"
    manifest_path = output_dir / f"{stem}-manifest.json"

    write_structured_output(
        letters,
        txt_path,
        title=args.title,
        author=args.author,
        editor=args.editor,
        publication_year=args.year,
    )
    write_manifest(letters, manifest_path)

    print(f"\nOutput: {txt_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Letters: {len(letters)}")
    print(f"\nTo ingest, run:")
    print(f"  uv run python -m author_library ingest '{txt_path}' --genre correspondence")


if __name__ == "__main__":
    main()
