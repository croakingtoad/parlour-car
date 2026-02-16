"""HTML document parser using BeautifulSoup.

Handles web articles, blog posts, and generic HTML documents.
Strips navigation, headers, footers, and ads to extract content.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from bs4 import BeautifulSoup, Tag

from author_library.errors import ParsingError
from author_library.parsing.base import DocumentParser
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

log = structlog.get_logger(__name__)

# HTML elements to strip (non-content)
_STRIP_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript", "form", "iframe"}
_STRIP_ROLES = {"navigation", "banner", "contentinfo", "complementary"}
_STRIP_CLASSES = {
    "nav", "navbar", "sidebar", "footer", "header",
    "ad", "ads", "advertisement", "menu", "cookie",
}


class HtmlParser(DocumentParser):
    """Parser for HTML documents."""

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm", ".xhtml"]

    async def parse(self, source: Path | str) -> ParsedDocument:
        source = Path(source)
        if not source.exists():
            raise ParsingError(
                f"HTML file not found: {source}",
                context={"path": str(source)},
            )

        log.info("parsing_html", path=str(source))

        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = source.read_text(encoding="latin-1")
            except Exception as exc:
                raise ParsingError(
                    f"Failed to read HTML file: {exc}",
                    context={"path": str(source)},
                    cause=exc,
                ) from exc

        warnings: list[str] = []

        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse HTML: {exc}",
                context={"path": str(source)},
                cause=exc,
            ) from exc

        metadata = self._extract_metadata(soup)

        # Strip non-content elements
        self._strip_non_content(soup)

        # Find the main content area
        main_content = self._find_main_content(soup)

        root = DocumentNode(node_type=NodeType.BOOK, metadata={"title": metadata.title or ""})
        raw_text_parts: list[str] = []

        current_section: DocumentNode | None = None

        for element in main_content.children:
            if not isinstance(element, Tag):
                text = str(element).strip()
                if text:
                    raw_text_parts.append(text)
                    target = current_section or root
                    target.children.append(
                        DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
                    )
                continue

            node, new_section = self._element_to_node(element, raw_text_parts, warnings)
            if node is None:
                continue

            if new_section:
                current_section = DocumentNode(
                    node_type=NodeType.SECTION,
                    metadata={"title": node.text},
                )
                current_section.children.append(node)
                root.children.append(current_section)
            else:
                target = current_section or root
                target.children.append(node)

        raw_text = "\n".join(raw_text_parts)
        metadata.word_count = len(raw_text.split())

        return ParsedDocument(
            source_path=str(source),
            format="html",
            metadata=metadata,
            tree=root,
            raw_text=raw_text,
            parse_warnings=warnings,
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> DocumentMetadata:
        """Extract metadata from HTML head elements."""
        title: str | None = None
        author: str | None = None
        date: str | None = None

        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Check meta tags
        for meta in soup.find_all("meta"):
            name_val = meta.get("name") or meta.get("property") or ""
            name = str(name_val).lower()
            content_val = str(meta.get("content") or "")
            if name in ("author", "article:author", "dc.creator"):
                author = content_val
            elif name in (
                "date", "article:published_time", "dc.date", "pubdate"
            ):
                date = content_val

        # Check h1 for title if no title tag
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        return DocumentMetadata(
            title=title,
            author=author,
            publication_date=date,
        )

    def _strip_non_content(self, soup: BeautifulSoup) -> None:
        """Remove non-content elements from the soup."""
        # Remove by tag name
        for tag_name in _STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove by ARIA role
        for role in _STRIP_ROLES:
            for tag in soup.find_all(attrs={"role": role}):
                tag.decompose()

        # Remove by class name patterns
        for tag in soup.find_all(True):
            classes = tag.get("class")
            if isinstance(classes, list):
                class_str = " ".join(str(c) for c in classes).lower()
            elif classes:
                class_str = str(classes).lower()
            else:
                class_str = ""
            if any(cls in class_str for cls in _STRIP_CLASSES):
                tag.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Tag:
        """Find the main content area of the HTML document."""
        # Try <main> tag
        main = soup.find("main")
        if main and isinstance(main, Tag):
            return main

        # Try <article> tag
        article = soup.find("article")
        if article and isinstance(article, Tag):
            return article

        # Try role="main"
        main_role = soup.find(attrs={"role": "main"})
        if main_role and isinstance(main_role, Tag):
            return main_role

        # Try common content class names
        content_classes = (
            "content", "post-content", "entry-content",
            "article-content", "main-content",
        )
        for class_name in content_classes:
            div = soup.find(class_=class_name)
            if div and isinstance(div, Tag):
                return div

        # Fallback: body
        body = soup.find("body")
        if body and isinstance(body, Tag):
            return body

        # Last resort: the soup itself
        return soup  # BeautifulSoup is Tag-like; mypy permits this

    def _element_to_node(
        self,
        element: Tag,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> tuple[DocumentNode | None, bool]:
        """Convert an HTML element to a DocumentNode.

        Returns:
            Tuple of (node, is_new_section). is_new_section is True for h2+ headings.
        """
        tag = element.name

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
                level = int(tag[1])
                node = DocumentNode(
                    node_type=NodeType.HEADING,
                    text=text,
                    metadata={"level": level},
                )
                # h2+ creates new sections
                return node, level >= 2
            return None, False

        if tag == "p":
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.PARAGRAPH, text=text), False
            return None, False

        if tag == "blockquote":
            text = element.get_text(strip=True)
            if text:
                raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.BLOCK_QUOTE, text=text), False
            return None, False

        if tag in ("ul", "ol"):
            list_node = DocumentNode(node_type=NodeType.LIST)
            for li in element.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text:
                    raw_text_parts.append(li_text)
                    list_node.children.append(
                        DocumentNode(node_type=NodeType.LIST_ITEM, text=li_text)
                    )
            if list_node.children:
                return list_node, False
            return None, False

        if tag == "table":
            text = element.get_text(separator=" | ", strip=True)
            if text:
                raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.TABLE, text=text), False
            return None, False

        if tag == "figure":
            img = element.find("img")
            if img and isinstance(img, Tag):
                alt = img.get("alt", "")
                src = img.get("src", "")
                return DocumentNode(
                    node_type=NodeType.IMAGE,
                    metadata={"alt": str(alt), "src": str(src)},
                ), False
            return None, False

        if tag == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            return DocumentNode(
                node_type=NodeType.IMAGE,
                metadata={"alt": str(alt), "src": str(src)},
            ), False

        # For divs and other containers, recursively process children
        if tag in ("div", "section", "article", "span"):
            children_nodes: list[DocumentNode] = []
            for child in element.children:
                if isinstance(child, Tag):
                    child_node, _ = self._element_to_node(child, raw_text_parts, warnings)
                    if child_node:
                        children_nodes.append(child_node)
                else:
                    text = str(child).strip()
                    if text:
                        raw_text_parts.append(text)
                        children_nodes.append(
                            DocumentNode(node_type=NodeType.PARAGRAPH, text=text)
                        )
            if len(children_nodes) == 1:
                return children_nodes[0], False
            if children_nodes:
                wrapper = DocumentNode(
                    node_type=NodeType.SECTION,
                    children=children_nodes,
                )
                return wrapper, False
            return None, False

        # Fallback: try to extract text
        text = element.get_text(strip=True)
        if text:
            raw_text_parts.append(text)
            return DocumentNode(node_type=NodeType.PARAGRAPH, text=text), False

        return None, False
