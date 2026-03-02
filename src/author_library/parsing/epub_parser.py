"""EPUB document parser using ebooklib + BeautifulSoup.

Extracts OPF metadata, NCX/navigation table of contents, and builds a
DocumentNode tree from the HTML spine items.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings as _warnings_mod
from pathlib import Path

import structlog
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from ebooklib import ITEM_DOCUMENT, epub

from author_library.errors import ParsingError
from author_library.parsing.base import DocumentParser
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

log = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Text sanitisation helpers — defend against encoding corruption
# ------------------------------------------------------------------

# Characters in the Windows-1252 0x80–0x9F range that have Unicode equivalents.
# These bytes are technically invalid in UTF-8 (they're C1 control codes in
# ISO-8859-1) but Windows-1252 maps them to printable characters.  Publisher
# EPUBs sometimes embed these bytes despite declaring UTF-8 encoding.
_CP1252_TO_UNICODE: dict[int, str] = {
    0x80: "\u20AC",  # €
    0x82: "\u201A",  # ‚
    0x83: "\u0192",  # ƒ
    0x84: "\u201E",  # „
    0x85: "\u2026",  # …
    0x86: "\u2020",  # †
    0x87: "\u2021",  # ‡
    0x88: "\u02C6",  # ˆ
    0x89: "\u2030",  # ‰
    0x8A: "\u0160",  # Š
    0x8B: "\u2039",  # ‹
    0x8C: "\u0152",  # Œ
    0x8E: "\u017D",  # Ž
    0x91: "\u2018",  # '
    0x92: "\u2019",  # '
    0x93: "\u201C",  # "
    0x94: "\u201D",  # "
    0x95: "\u2022",  # •
    0x96: "\u2013",  # –
    0x97: "\u2014",  # —
    0x98: "\u02DC",  # ˜
    0x99: "\u2122",  # ™
    0x9A: "\u0161",  # š
    0x9B: "\u203A",  # ›
    0x9C: "\u0153",  # œ
    0x9E: "\u017E",  # ž
    0x9F: "\u0178",  # Ÿ
}


def _decode_epub_content(content: bytes) -> str:
    """Decode EPUB HTML content bytes, handling common encoding problems.

    Strategy:
    1. Try strict UTF-8 (the EPUB spec requires UTF-8 or UTF-16).
    2. If that fails, try Windows-1252 — many publisher EPUBs declare UTF-8
       but actually contain Windows-1252 smart quotes and dashes.
    3. Last resort: UTF-8 with replacement characters.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("epub_content_not_utf8_trying_cp1252")
        try:
            return content.decode("windows-1252")
        except UnicodeDecodeError:
            log.warning("epub_content_decode_fallback_replace")
            return content.decode("utf-8", errors="replace")


def _sanitize_text(text: str) -> str:
    """Normalise and clean extracted text for safe UTF-8 storage.

    Applies:
    - Unicode NFC normalisation (compose decomposed sequences)
    - Windows-1252 C1 control-code fixup (0x80–0x9F → proper Unicode)
    - Strip null bytes and remaining C0/C1 control characters (keep \\n, \\r, \\t)
    - UTF-8 round-trip verification
    """
    if not text:
        return text

    # 1. NFC normalisation — ensures composed forms (é not e+combining accent)
    text = unicodedata.normalize("NFC", text)

    # 2. Fix C1 control codes that are actually Windows-1252 characters.
    #    These slip through when content is labelled ISO-8859-1 but actually
    #    uses the Windows-1252 superset, or when lxml preserves raw C1 bytes.
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _CP1252_TO_UNICODE:
            chars.append(_CP1252_TO_UNICODE[cp])
        elif cp < 0x20 and ch not in ("\n", "\r", "\t"):
            continue  # strip C0 control chars except whitespace
        elif cp == 0x7F:
            continue  # strip DEL
        elif 0x80 <= cp <= 0x9F:
            continue  # strip unmapped C1 controls
        else:
            chars.append(ch)
    text = "".join(chars)

    # 3. Verify clean UTF-8 round-trip (belt-and-suspenders)
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    return text


# Patterns for detecting structural elements
_CHAPTER_HEADING_RE = re.compile(
    r"^(chapter|part|book|prologue|epilogue|introduction|conclusion|appendix)"
    r"[\s.:]*(\d+|[ivxlcdm]+)?",
    re.IGNORECASE,
)
_POEM_LINE_BREAK_THRESHOLD = 3  # minimum consecutive <br> to consider verse


def _extract_title_from_alt(alt_text: str) -> str | None:
    """Parse a structured image alt-text string for the book title.

    Publisher title-page images often encode metadata in alt text like::

        Book title, Fred Rogers: The Last Interview, subtitle, and Other
        Conversations, author, Fred Rogers, imprint, Melville House

    Returns the combined "main title: subtitle" or ``None``.
    """
    # Split on comma-separated label/value pairs
    parts = [p.strip() for p in alt_text.split(",")]

    title_value: str | None = None
    subtitle_value: str | None = None
    capture_title = False
    capture_subtitle = False
    title_parts: list[str] = []
    subtitle_parts: list[str] = []

    for part in parts:
        lower = part.lower()
        if lower in ("book title", "title"):
            capture_title = True
            capture_subtitle = False
            continue
        if lower == "subtitle":
            capture_subtitle = True
            capture_title = False
            continue
        if lower in ("author", "imprint", "publisher", "series"):
            capture_title = False
            capture_subtitle = False
            continue
        if capture_title:
            title_parts.append(part)
        elif capture_subtitle:
            subtitle_parts.append(part)

    if title_parts:
        title_value = ", ".join(title_parts).strip()
    if subtitle_parts:
        subtitle_value = ", ".join(subtitle_parts).strip()

    if title_value and subtitle_value:
        # If subtitle starts with a conjunction, join with space not colon
        if subtitle_value.lower().startswith(("and ", "or ", "& ")):
            return f"{title_value} {subtitle_value}"
        return f"{title_value}: {subtitle_value}"
    return title_value


class EpubParser(DocumentParser):
    """Parser for EPUB documents."""

    def supported_extensions(self) -> list[str]:
        return [".epub"]

    async def parse(self, source: Path | str) -> ParsedDocument:
        source = Path(source)
        if not source.exists():
            raise ParsingError(
                f"EPUB file not found: {source}",
                context={"path": str(source)},
            )

        log.info("parsing_epub", path=str(source))

        try:
            book = epub.read_epub(str(source), options={"ignore_ncx": False})
        except Exception as exc:
            raise ParsingError(
                f"Failed to read EPUB: {exc}",
                context={"path": str(source)},
                cause=exc,
            ) from exc

        warnings: list[str] = []
        metadata = self._extract_metadata(book, warnings)
        toc_titles = self._extract_toc(book)
        metadata.table_of_contents = toc_titles

        root = DocumentNode(node_type=NodeType.BOOK, metadata={"title": metadata.title or ""})
        raw_text_parts: list[str] = []

        spine_items = self._spine_items(book)
        if not spine_items:
            warnings.append("No document items found in EPUB spine")

        seen_content_hashes: set[str] = set()
        for item in spine_items:
            content = item.get_content()
            if not content:
                continue

            # Deduplicate: some EPUBs have duplicate spine entries or
            # manifest items that reference the same content.
            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash in seen_content_hashes:
                log.debug(
                    "epub_skipping_duplicate_spine_item",
                    item=item.get_name(),
                )
                continue
            seen_content_hashes.add(content_hash)

            # Decode bytes → str with encoding fallback before BS4 parsing.
            # This ensures we control the encoding rather than relying on
            # lxml's auto-detection, which can silently drop characters.
            decoded_content = _decode_epub_content(content)

            try:
                with _warnings_mod.catch_warnings():
                    _warnings_mod.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    soup = BeautifulSoup(decoded_content, "lxml")
            except Exception as exc:
                warnings.append(f"Failed to parse HTML for {item.get_name()}: {exc}")
                continue

            body = soup.find("body")
            if body is None:
                body = soup

            result_node = self._parse_body(body, raw_text_parts, warnings)
            if result_node is not None:
                if (
                    result_node.node_type == NodeType.BOOK
                    and result_node.metadata.get("synthetic")
                ):
                    # _parse_body found multiple chapters in one body;
                    # graft them directly under root.
                    root.children.extend(result_node.children)
                else:
                    root.children.append(result_node)

        raw_text = "\n".join(raw_text_parts)
        metadata.word_count = len(raw_text.split())

        return ParsedDocument(
            source_path=str(source),
            format="epub",
            metadata=metadata,
            tree=root,
            raw_text=raw_text,
            parse_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(
        self, book: epub.EpubBook, warnings: list[str]
    ) -> DocumentMetadata:
        title_entries = book.get_metadata("DC", "title")
        author = book.get_metadata("DC", "creator")
        publisher = book.get_metadata("DC", "publisher")
        date = book.get_metadata("DC", "date")
        language = book.get_metadata("DC", "language")
        identifiers = book.get_metadata("DC", "identifier")

        isbn: str | None = None
        for ident in identifiers:
            val = ident[0] if isinstance(ident, tuple) else str(ident)
            val_str = str(val).replace("-", "")
            if re.match(r"^(97[89])?\d{9}[\dXx]$", val_str):
                isbn = str(val)
                break

        author_name = author[0][0] if author else None
        extracted_title = self._resolve_title(title_entries, author_name, book)

        if extracted_title and author_name:
            # Guard: if the title (or its main component before any subtitle
            # separator) matches the author name, the EPUB metadata is almost
            # certainly wrong — fall back to content-based extraction.
            title_main = extracted_title.split(":")[0].strip()
            title_lower = title_main.lower()
            author_lower = author_name.lower().strip()
            if title_lower == author_lower or title_lower in author_lower:
                fallback = self._title_from_content(book)
                if fallback:
                    log.info(
                        "epub_title_fallback",
                        original=extracted_title,
                        fallback=fallback,
                    )
                    extracted_title = fallback
                else:
                    warnings.append(
                        f"Extracted title '{extracted_title}' matches author name; "
                        "no alternative found"
                    )

        return DocumentMetadata(
            title=extracted_title,
            author=author_name,
            publisher=publisher[0][0] if publisher else None,
            publication_date=date[0][0] if date else None,
            isbn=isbn,
            language=language[0][0] if language else "en",
        )

    @staticmethod
    def _resolve_title(
        title_entries: list[tuple[str, dict[str, str]]],
        author_name: str | None,
        book: epub.EpubBook,
    ) -> str | None:
        """Combine dc:title entries (main + subtitle) into a single title string.

        EPUB 3 allows multiple ``dc:title`` elements distinguished by
        ``title-type`` refines (main, subtitle, etc.).  Many publisher
        EPUBs split the title this way — e.g. "Fred Rogers" (main) +
        "and Other Conversations" (subtitle).
        """
        if not title_entries:
            return None

        if len(title_entries) == 1:
            return title_entries[0][0]

        # Multiple title entries — check OPF refines for title-type hints
        opf_ns = "http://www.idpf.org/2007/opf"
        meta_entries = book.metadata.get(opf_ns, {}).get("meta", [])

        title_types: dict[str, str] = {}  # id → title-type
        for value, attrs in meta_entries:
            prop = attrs.get("property", "")
            refines = attrs.get("refines", "")
            if prop == "title-type" and refines.startswith("#") and value:
                title_types[refines[1:]] = str(value)

        main_parts: list[str] = []
        subtitle_parts: list[str] = []

        for entry_val, entry_attrs in title_entries:
            entry_id = entry_attrs.get("id", "")
            ttype = title_types.get(entry_id, "")
            if ttype == "subtitle":
                subtitle_parts.append(entry_val)
            else:
                main_parts.append(entry_val)

        combined = ": ".join(main_parts) if main_parts else ""
        if subtitle_parts:
            sub = " ".join(subtitle_parts)
            # Avoid doubling a colon if the main title already ends with one
            if combined and not combined.rstrip().endswith(":"):
                combined += ": " + sub
            elif combined:
                combined += " " + sub
            else:
                combined = sub

        return combined or title_entries[0][0]

    def _title_from_content(self, book: epub.EpubBook) -> str | None:
        """Try to extract the book title from spine content.

        Looks at the first few spine items for image alt text (title-page
        images) or the copyright page for a full-title line, or the first
        ``<h1>`` heading in the content.
        """
        spine_items = self._spine_items(book)

        for item in spine_items[:5]:
            content = item.get_content()
            if not content:
                continue
            try:
                decoded_content = _decode_epub_content(content)
                with _warnings_mod.catch_warnings():
                    _warnings_mod.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    soup = BeautifulSoup(decoded_content, "lxml")
            except Exception:
                continue

            # Strategy 1: title-page image alt text (common in publisher EPUBs)
            for img in soup.find_all("img"):
                alt = str(img.get("alt", ""))
                if "book title" in alt.lower() or "title" in alt.lower():
                    # Parse structured alt text like "Book title, Fred Rogers:
                    # The Last Interview, subtitle, and Other Conversations, ..."
                    title = _extract_title_from_alt(alt)
                    if title:
                        return title

            # Strategy 2: copyright page often has the full title
            body = soup.find("body")
            if body:
                first_p = body.find("p")
                if first_p:
                    first_text = first_p.get_text(strip=True)
                    # Copyright pages often start with the full title
                    if ":" in first_text and len(first_text) < 200:
                        # Trim at "Copyright" if present
                        if "Copyright" in first_text:
                            first_text = first_text[: first_text.index("Copyright")].strip()
                        if first_text:
                            return first_text

        return None

    def _extract_toc(self, book: epub.EpubBook) -> list[str]:
        toc = book.toc
        titles: list[str] = []
        self._walk_toc(toc, titles)
        return titles

    def _walk_toc(self, items: list[object] | tuple[object, ...], titles: list[str]) -> None:
        for item in items:
            if isinstance(item, tuple):
                # (section, [children])
                section, children = item
                if hasattr(section, "title"):
                    titles.append(str(section.title))
                if isinstance(children, (list, tuple)):
                    self._walk_toc(children, titles)
            elif hasattr(item, "title"):
                titles.append(str(item.title))

    @staticmethod
    def _spine_items(book: epub.EpubBook) -> list[epub.EpubItem]:
        """Return document items in spine reading order, excluding nav docs.

        ``book.get_items_of_type(ITEM_DOCUMENT)`` returns *all* manifest
        items of type ITEM_DOCUMENT regardless of whether they are in the
        spine.  That set often includes the EPUB navigation document
        (nav.xhtml) which duplicates the table of contents and should not
        be chunked as prose.

        This method resolves the spine to actual item objects and skips
        navigation items (both EPUB 3 ``EpubNav`` and EPUB 2 NCX).  If
        the spine is empty or cannot be resolved, falls back to manifest
        items (excluding navigation) for robustness.
        """
        items: list[epub.EpubItem] = []
        seen_ids: set[str] = set()

        for spine_id, _ in book.spine:
            if spine_id in seen_ids:
                continue
            seen_ids.add(spine_id)
            item = book.get_item_with_id(spine_id)
            if item is None:
                continue
            # Skip navigation documents: EpubNav (EPUB3) or non-chapter items
            if isinstance(item, epub.EpubNav):
                continue
            items.append(item)

        if items:
            return items

        # Fallback: spine unresolvable — use manifest items minus nav
        log.warning("epub_spine_unresolvable_using_manifest")
        return [
            item
            for item in book.get_items_of_type(ITEM_DOCUMENT)
            if not isinstance(item, epub.EpubNav)
        ]

    # ------------------------------------------------------------------
    # HTML body → DocumentNode tree
    # ------------------------------------------------------------------

    def _parse_body(
        self,
        body: Tag,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> DocumentNode | None:
        """Parse an HTML body element into one or more chapter DocumentNodes.

        Returns a single chapter node if the body contains one chapter, or a
        synthetic BOOK node wrapping multiple chapters when the body contains
        multiple ``<h1>`` chapter boundaries (common in single-file EPUBs).
        """
        # Unwrap pure wrapper divs: if body's only meaningful child is a
        # single <div> that itself contains headings, recurse into it.
        effective_body = self._unwrap_container(body)

        chapters: list[DocumentNode] = []
        chapter = DocumentNode(node_type=NodeType.CHAPTER)
        current_section: DocumentNode | None = None

        for element in effective_body.children:
            if not isinstance(element, Tag):
                text = _sanitize_text(str(element).strip())
                if text:
                    raw_text_parts.append(text)
                    target = current_section if current_section else chapter
                    target.children.append(
                        DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
                    )
                continue

            node = self._element_to_node(element, raw_text_parts, warnings)
            if node is None:
                continue

            # If it's a heading, determine whether to create a chapter title or section
            if node.node_type == NodeType.HEADING:
                heading_level = node.metadata.get("level", 0)
                if heading_level in (1, 2) and not chapter.metadata.get("title"):
                    chapter.metadata["title"] = node.text
                    chapter.children.append(node)
                elif heading_level == 1 and chapter.metadata.get("title"):
                    # New h1 while we already have a titled chapter —
                    # finalize the current chapter and start a new one.
                    if chapter.children:
                        chapters.append(chapter)
                    chapter = DocumentNode(
                        node_type=NodeType.CHAPTER,
                        metadata={"title": node.text},
                    )
                    chapter.children.append(node)
                    current_section = None
                elif heading_level in (2, 3, 4, 5, 6):
                    current_section = DocumentNode(
                        node_type=NodeType.SECTION,
                        metadata={"title": node.text},
                    )
                    current_section.children.append(node)
                    chapter.children.append(current_section)
                else:
                    target = current_section if current_section else chapter
                    target.children.append(node)
            else:
                target = current_section if current_section else chapter
                target.children.append(node)

        # Finalize the last chapter
        if chapter.children:
            chapters.append(chapter)

        if not chapters:
            return None
        if len(chapters) == 1:
            return chapters[0]

        # Multiple chapters found in a single body — return a wrapper node
        # so the caller can attach all of them to the root.
        wrapper = DocumentNode(node_type=NodeType.BOOK, metadata={"synthetic": True})
        wrapper.children = chapters
        return wrapper

    # ------------------------------------------------------------------
    # Container unwrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_container(body: Tag) -> Tag:
        """Unwrap thin structural wrappers around the actual chapter content.

        Many EPUBs wrap content in single-child containers like
        ``<body><div>…</div></body>`` or
        ``<body><article><section>…</section></article></body>``.
        Recursively unwrap up to 3 levels so that ``_parse_body`` can iterate
        directly over headings and paragraphs.
        """
        _WRAPPER_TAGS = {"div", "section", "article"}
        effective = body
        for _ in range(3):
            direct_tags = [c for c in effective.children if isinstance(c, Tag)]
            if len(direct_tags) != 1:
                break
            wrapper = direct_tags[0]
            if wrapper.name not in _WRAPPER_TAGS:
                break
            # Only unwrap if the wrapper contains heading tags — a strong
            # signal that it is structural content, not a styled container.
            if wrapper.find(["h1", "h2", "h3"]):
                effective = wrapper
            else:
                break
        return effective

    def _element_to_node(
        self,
        element: Tag,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> DocumentNode | None:
        """Convert a single HTML element to a DocumentNode."""
        tag = element.name

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _sanitize_text(element.get_text(strip=True))
            if text:
                raw_text_parts.append(text)
                return DocumentNode(
                    node_type=NodeType.HEADING,
                    text=text,
                    metadata={"level": int(tag[1])},
                )
            return None

        if tag == "p":
            text = _sanitize_text(element.get_text(strip=True))
            if not text:
                return None
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.PARAGRAPH, text=text)

        if tag == "blockquote":
            text = _sanitize_text(element.get_text(strip=True))
            if text:
                raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.BLOCK_QUOTE, text=text)

        if tag in ("ul", "ol"):
            list_node = DocumentNode(node_type=NodeType.LIST)
            for li in element.find_all("li", recursive=False):
                li_text = _sanitize_text(li.get_text(strip=True))
                if li_text:
                    raw_text_parts.append(li_text)
                    list_node.children.append(
                        DocumentNode(node_type=NodeType.LIST_ITEM, text=li_text)
                    )
            return list_node if list_node.children else None

        if tag == "table":
            text = _sanitize_text(element.get_text(separator=" | ", strip=True))
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.TABLE, text=text)

        if tag == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            return DocumentNode(
                node_type=NodeType.IMAGE,
                metadata={"alt": str(alt), "src": str(src)},
            )

        if tag == "header":
            # HTML5 <header> elements often wrap headings in EPUB documents.
            # Extract all heading children and combine them into a single
            # HEADING node (e.g. "Chapter 1" + "Seeing through Dreams" →
            # "Chapter 1: Seeing through Dreams").
            headings = element.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            if headings:
                best_level = min(int(h.name[1]) for h in headings)
                texts: list[str] = []
                for h in headings:
                    text = h.get_text(strip=True)
                    if text:
                        texts.append(text)
                        raw_text_parts.append(text)
                combined = ": ".join(texts)
                if combined:
                    return DocumentNode(
                        node_type=NodeType.HEADING,
                        text=combined,
                        metadata={"level": best_level},
                    )
            # No headings found — extract as paragraph fallback
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
            return None

        if tag in ("aside", "div"):
            # Check for footnote / endnote patterns
            role = str(element.get("role") or element.get("epub:type") or "")
            class_attr = element.get("class")
            if isinstance(class_attr, list):
                classes = " ".join(str(c) for c in class_attr)
            else:
                classes = str(class_attr or "")
            if "footnote" in str(role).lower() or "footnote" in classes.lower():
                text = _sanitize_text(element.get_text(strip=True))
                if text:
                    raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.FOOTNOTE, text=text if text else "")
            if "endnote" in str(role).lower() or "endnote" in classes.lower():
                text = _sanitize_text(element.get_text(strip=True))
                if text:
                    raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.ENDNOTE, text=text if text else "")

        # Check for poetry: a div/pre containing many <br> or line-structured content.
        # Only treat as poem if the element does NOT contain heading tags — a div
        # with headings is structural content, not a poem, even if it has <br> spacers.
        if tag in ("pre", "div"):
            has_headings = bool(element.find(["h1", "h2", "h3", "h4", "h5", "h6"]))
            if not has_headings:
                br_count = len(element.find_all("br"))
                if br_count >= _POEM_LINE_BREAK_THRESHOLD:
                    return self._parse_poem(element, raw_text_parts)

        # Structural container: a div/aside/section/article that wraps
        # paragraphs, headings, or other block-level content.  Recurse
        # into children so individual paragraphs are preserved instead of
        # being collapsed into a single PARAGRAPH node.
        _STRUCTURAL_TAGS = {
            "p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "header",
        }
        if tag in ("div", "aside", "section", "article"):
            has_structural = element.find(list(_STRUCTURAL_TAGS), recursive=False)
            if has_structural:
                container = DocumentNode(node_type=NodeType.SECTION)
                for child in element.children:
                    if isinstance(child, Tag):
                        child_node = self._element_to_node(child, raw_text_parts, warnings)
                        if child_node is not None:
                            container.children.append(child_node)
                            # Extract section title from first heading child
                            if (
                                child_node.node_type == NodeType.HEADING
                                and "title" not in container.metadata
                            ):
                                container.metadata["title"] = child_node.text
                    else:
                        text = _sanitize_text(str(child).strip())
                        if text:
                            raw_text_parts.append(text)
                            container.children.append(
                                DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
                            )
                return container if container.children else None

        # Fallback: extract text from any other element
        text = _sanitize_text(element.get_text(strip=True))
        if text:
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.PARAGRAPH, text=text)

        return None

    def _parse_poem(self, element: Tag, raw_text_parts: list[str]) -> DocumentNode:
        """Parse a poem element into a POEM node with STANZA/LINE children."""
        poem = DocumentNode(node_type=NodeType.POEM)
        # Split content by double line breaks for stanzas
        html_str = str(element)
        # Split on double <br> for stanza boundaries
        stanza_parts = re.split(r"<br\s*/?\s*>\s*<br\s*/?\s*>", html_str)

        for part in stanza_parts:
            stanza_soup = BeautifulSoup(part, "lxml")
            text = stanza_soup.get_text()
            lines = [_sanitize_text(line.strip()) for line in text.split("\n") if line.strip()]
            if not lines:
                continue
            stanza = DocumentNode(node_type=NodeType.STANZA)
            for line_text in lines:
                raw_text_parts.append(line_text)
                stanza.children.append(
                    DocumentNode(node_type=NodeType.LINE, text=line_text)
                )
            poem.children.append(stanza)

        return poem
