"""Document parsing subsystem — parser registry and public API.

Usage::

    from author_library.parsing import get_parser

    parser = get_parser("book.epub")
    result = await parser.parse("book.epub")
"""

from __future__ import annotations

from pathlib import Path

from author_library.errors import ParsingError
from author_library.parsing.base import DocumentParser
from author_library.parsing.docx_parser import DocxParser
from author_library.parsing.epub_parser import EpubParser
from author_library.parsing.html_parser import HtmlParser
from author_library.parsing.pdf_parser import PdfParser
from author_library.parsing.text_parser import TextParser

_PARSERS: list[DocumentParser] = [
    EpubParser(),
    PdfParser(),
    DocxParser(),
    TextParser(),
    HtmlParser(),
]

_EXTENSION_MAP: dict[str, DocumentParser] = {}
for _parser in _PARSERS:
    for _ext in _parser.supported_extensions():
        _EXTENSION_MAP[_ext.lower()] = _parser


def get_parser(file_path: str | Path) -> DocumentParser:
    """Return the appropriate parser for the given file.

    Args:
        file_path: Path to the document file (used to determine extension).

    Returns:
        A DocumentParser instance suitable for the file type.

    Raises:
        ParsingError: If no parser is registered for the file extension.
    """
    ext = Path(file_path).suffix.lower()
    parser = _EXTENSION_MAP.get(ext)
    if parser is None:
        raise ParsingError(
            f"Unsupported file format: {ext}",
            context={"path": str(file_path), "extension": ext},
        )
    return parser


async def parse_document(
    file_path: str | Path,
    metadata_hints: dict | None = None,  # noqa: ARG001 — reserved for future use
) -> "DocumentParser":
    """Parse a document and return the ParsedDocument result."""
    from author_library.parsing.base import ParsedDocument as _PD  # noqa: F401
    parser = get_parser(file_path)
    return await parser.parse(file_path)


__all__ = [
    "DocumentParser",
    "DocxParser",
    "EpubParser",
    "HtmlParser",
    "PdfParser",
    "TextParser",
    "get_parser",
    "parse_document",
]
