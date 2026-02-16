"""EPUB document parser using ebooklib + BeautifulSoup.

Extracts OPF metadata, NCX/navigation table of contents, and builds a
DocumentNode tree from the HTML spine items.
"""

from __future__ import annotations

import re
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

# Patterns for detecting structural elements
_CHAPTER_HEADING_RE = re.compile(
    r"^(chapter|part|book|prologue|epilogue|introduction|conclusion|appendix)"
    r"[\s.:]*(\d+|[ivxlcdm]+)?",
    re.IGNORECASE,
)
_POEM_LINE_BREAK_THRESHOLD = 3  # minimum consecutive <br> to consider verse


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

        spine_items = list(book.get_items_of_type(ITEM_DOCUMENT))
        if not spine_items:
            warnings.append("No document items found in EPUB spine")

        for item in spine_items:
            content = item.get_content()
            if not content:
                continue

            try:
                with _warnings_mod.catch_warnings():
                    _warnings_mod.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    soup = BeautifulSoup(content, "lxml")
            except Exception as exc:
                warnings.append(f"Failed to parse HTML for {item.get_name()}: {exc}")
                continue

            body = soup.find("body")
            if body is None:
                body = soup

            chapter_node = self._parse_body(body, raw_text_parts, warnings)
            if chapter_node is not None:
                root.children.append(chapter_node)

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
        title = book.get_metadata("DC", "title")
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

        return DocumentMetadata(
            title=title[0][0] if title else None,
            author=author[0][0] if author else None,
            publisher=publisher[0][0] if publisher else None,
            publication_date=date[0][0] if date else None,
            isbn=isbn,
            language=language[0][0] if language else "en",
        )

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

    # ------------------------------------------------------------------
    # HTML body → DocumentNode tree
    # ------------------------------------------------------------------

    def _parse_body(
        self,
        body: Tag,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> DocumentNode | None:
        """Parse an HTML body element into a chapter DocumentNode."""
        chapter = DocumentNode(node_type=NodeType.CHAPTER)
        current_section: DocumentNode | None = None

        for element in body.children:
            if not isinstance(element, Tag):
                text = str(element).strip()
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
                level = element.name  # h1, h2, etc.
                if level in ("h1", "h2") and not chapter.metadata.get("title"):
                    chapter.metadata["title"] = node.text
                    chapter.children.append(node)
                elif level in ("h2", "h3", "h4", "h5", "h6"):
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

        # Skip empty chapters
        if not chapter.children:
            return None

        return chapter

    def _element_to_node(
        self,
        element: Tag,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> DocumentNode | None:
        """Convert a single HTML element to a DocumentNode."""
        tag = element.name

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
                return DocumentNode(
                    node_type=NodeType.HEADING,
                    text=text,
                    metadata={"level": int(tag[1])},
                )
            return None

        if tag == "p":
            text = element.get_text(strip=True)
            if not text:
                return None
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.PARAGRAPH, text=text)

        if tag == "blockquote":
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.BLOCK_QUOTE, text=text)

        if tag in ("ul", "ol"):
            list_node = DocumentNode(node_type=NodeType.LIST)
            for li in element.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text:
                    raw_text_parts.append(li_text)
                    list_node.children.append(
                        DocumentNode(node_type=NodeType.LIST_ITEM, text=li_text)
                    )
            return list_node if list_node.children else None

        if tag == "table":
            text = element.get_text(separator=" | ", strip=True)
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.TABLE, text=text)

        if tag == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            return DocumentNode(
                node_type=NodeType.IMAGE,
                metadata={"alt": str(alt), "src": str(src)},
            )

        if tag in ("aside", "div"):
            # Check for footnote / endnote patterns
            role = str(element.get("role") or element.get("epub:type") or "")
            class_attr = element.get("class")
            if isinstance(class_attr, list):
                classes = " ".join(str(c) for c in class_attr)
            else:
                classes = str(class_attr or "")
            if "footnote" in str(role).lower() or "footnote" in classes.lower():
                text = element.get_text(strip=True)
                if text:
                    raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.FOOTNOTE, text=text if text else "")
            if "endnote" in str(role).lower() or "endnote" in classes.lower():
                text = element.get_text(strip=True)
                if text:
                    raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.ENDNOTE, text=text if text else "")

        # Check for poetry: a div/pre containing many <br> or line-structured content
        if tag in ("pre", "div"):
            br_count = len(element.find_all("br"))
            if br_count >= _POEM_LINE_BREAK_THRESHOLD:
                return self._parse_poem(element, raw_text_parts)

        # Fallback: extract text from any other element
        text = element.get_text(strip=True)
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
            lines = [line.strip() for line in text.split("\n") if line.strip()]
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
