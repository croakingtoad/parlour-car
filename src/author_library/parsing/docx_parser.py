"""DOCX document parser using python-docx.

Extracts headings (Heading styles → chapters/sections), paragraphs,
tables, images, and footnotes/endnotes from .docx files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from author_library.errors import ParsingError
from author_library.parsing.base import DocumentParser
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

log = structlog.get_logger(__name__)


class DocxParser(DocumentParser):
    """Parser for DOCX documents."""

    def supported_extensions(self) -> list[str]:
        return [".docx"]

    async def parse(self, source: Path | str) -> ParsedDocument:
        source = Path(source)
        if not source.exists():
            raise ParsingError(
                f"DOCX file not found: {source}",
                context={"path": str(source)},
            )

        log.info("parsing_docx", path=str(source))

        try:
            doc: Any = DocxDocument(str(source))
        except PackageNotFoundError as exc:
            raise ParsingError(
                f"Invalid DOCX file: {exc}",
                context={"path": str(source)},
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ParsingError(
                f"Failed to read DOCX: {exc}",
                context={"path": str(source)},
                cause=exc,
            ) from exc

        warnings: list[str] = []
        metadata = self._extract_metadata(doc)

        root = DocumentNode(
            node_type=NodeType.BOOK, metadata={"title": metadata.title or ""}
        )
        raw_text_parts: list[str] = []

        current_chapter: DocumentNode | None = None
        current_section: DocumentNode | None = None

        for element in doc.element.body:
            tag: str = element.tag

            # Table element
            if tag.endswith("}tbl"):
                table_node = self._parse_table(element, doc, raw_text_parts)
                if table_node:
                    target = current_section or current_chapter or root
                    target.children.append(table_node)
                continue

            # Paragraph element
            if not tag.endswith("}p"):
                continue

            para = Paragraph(element, doc)
            style_name = para.style.name if para.style else ""
            text = para.text.strip()

            if not text and not style_name.startswith("Heading"):
                continue

            # Heading 1 → Chapter
            if style_name == "Heading 1":
                raw_text_parts.append(text)
                current_chapter = DocumentNode(
                    node_type=NodeType.CHAPTER,
                    metadata={"title": text},
                )
                heading_node = DocumentNode(
                    node_type=NodeType.HEADING,
                    text=text,
                    metadata={"level": 1},
                )
                current_chapter.children.append(heading_node)
                root.children.append(current_chapter)
                current_section = None
                continue

            # Heading 2+ → Section
            if style_name.startswith("Heading"):
                level = self._heading_level(style_name)
                raw_text_parts.append(text)
                current_section = DocumentNode(
                    node_type=NodeType.SECTION,
                    metadata={"title": text},
                )
                heading_node = DocumentNode(
                    node_type=NodeType.HEADING,
                    text=text,
                    metadata={"level": level},
                )
                current_section.children.append(heading_node)
                target = current_chapter or root
                target.children.append(current_section)
                continue

            # Regular paragraph
            if text:
                raw_text_parts.append(text)

                # Check for footnote references in the paragraph XML
                footnotes = self._extract_footnote_refs(element)

                para_node = DocumentNode(
                    node_type=NodeType.PARAGRAPH, text=text
                )
                target = current_section or current_chapter or root
                target.children.append(para_node)

                for fn_text in footnotes:
                    raw_text_parts.append(fn_text)
                    target.children.append(
                        DocumentNode(
                            node_type=NodeType.FOOTNOTE, text=fn_text
                        )
                    )

        # Extract footnotes from footnotes part
        footnote_nodes = self._extract_footnotes(
            doc, raw_text_parts, warnings
        )
        if footnote_nodes:
            for fn_node in footnote_nodes:
                root.children.append(fn_node)

        raw_text = "\n".join(raw_text_parts)
        metadata.word_count = len(raw_text.split())

        return ParsedDocument(
            source_path=str(source),
            format="docx",
            metadata=metadata,
            tree=root,
            raw_text=raw_text,
            parse_warnings=warnings,
        )

    def _extract_metadata(self, doc: Any) -> DocumentMetadata:
        """Extract metadata from DOCX core properties."""
        props = doc.core_properties
        return DocumentMetadata(
            title=props.title or None,
            author=props.author or None,
            publication_date=str(props.created) if props.created else None,
        )

    @staticmethod
    def _heading_level(style_name: str) -> int:
        """Extract heading level from style name like 'Heading 2'."""
        parts = style_name.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 2  # default for unrecognized heading styles

    def _parse_table(
        self,
        element: Any,
        doc: Any,
        raw_text_parts: list[str],
    ) -> DocumentNode | None:
        """Parse a table element into a TABLE node."""
        try:
            table = Table(element, doc)
            rows_text: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows_text.append(" | ".join(cells))
            text = "\n".join(rows_text)
            if text.strip():
                raw_text_parts.append(text)
                return DocumentNode(node_type=NodeType.TABLE, text=text)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_footnote_refs(element: Any) -> list[str]:
        """Extract footnote reference text from paragraph XML."""
        refs: list[str] = []
        for fn_ref in element.findall(f".//{qn('w:footnoteReference')}"):
            fn_id = fn_ref.get(qn("w:id"))
            if fn_id:
                refs.append(f"[footnote:{fn_id}]")
        return refs

    def _extract_footnotes(
        self,
        doc: Any,
        raw_text_parts: list[str],
        warnings: list[str],
    ) -> list[DocumentNode]:
        """Extract footnotes from the footnotes part of the document."""
        nodes: list[DocumentNode] = []
        try:
            footnotes_part = doc.part.package.part_related_by(
                "/word/footnotes.xml"
            )
            if footnotes_part is None:
                return nodes
        except Exception:
            # No footnotes part — that's fine
            return nodes

        try:
            from lxml import etree  # type: ignore[import-untyped]

            tree = etree.fromstring(footnotes_part.blob)
            ns = {
                "w": "http://schemas.openxmlformats.org/"
                "wordprocessingml/2006/main"
            }
            for fn in tree.findall(".//w:footnote", ns):
                fn_id = fn.get(f"{{{ns['w']}}}id")
                # Skip separator footnotes (ids 0 and -1)
                if fn_id in ("0", "-1"):
                    continue
                texts = []
                for p in fn.findall(".//w:t", ns):
                    if p.text:
                        texts.append(p.text)
                fn_text = " ".join(texts).strip()
                if fn_text:
                    raw_text_parts.append(fn_text)
                    nodes.append(
                        DocumentNode(
                            node_type=NodeType.FOOTNOTE,
                            text=fn_text,
                            metadata={"footnote_id": fn_id or ""},
                        )
                    )
        except Exception as exc:
            warnings.append(f"Failed to extract footnotes: {exc}")

        return nodes
