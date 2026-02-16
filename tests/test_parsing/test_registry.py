"""Tests for the parser registry."""

import pytest

from author_library.errors import ParsingError
from author_library.parsing import get_parser
from author_library.parsing.docx_parser import DocxParser
from author_library.parsing.epub_parser import EpubParser
from author_library.parsing.html_parser import HtmlParser
from author_library.parsing.pdf_parser import PdfParser
from author_library.parsing.text_parser import TextParser


class TestGetParser:
    def test_epub_extension(self) -> None:
        parser = get_parser("book.epub")
        assert isinstance(parser, EpubParser)

    def test_pdf_extension(self) -> None:
        parser = get_parser("document.pdf")
        assert isinstance(parser, PdfParser)

    def test_docx_extension(self) -> None:
        parser = get_parser("paper.docx")
        assert isinstance(parser, DocxParser)

    def test_txt_extension(self) -> None:
        parser = get_parser("notes.txt")
        assert isinstance(parser, TextParser)

    def test_text_extension(self) -> None:
        parser = get_parser("notes.text")
        assert isinstance(parser, TextParser)

    def test_html_extension(self) -> None:
        parser = get_parser("page.html")
        assert isinstance(parser, HtmlParser)

    def test_htm_extension(self) -> None:
        parser = get_parser("page.htm")
        assert isinstance(parser, HtmlParser)

    def test_xhtml_extension(self) -> None:
        parser = get_parser("page.xhtml")
        assert isinstance(parser, HtmlParser)

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(ParsingError, match="Unsupported file format"):
            get_parser("image.png")

    def test_case_insensitive(self) -> None:
        parser = get_parser("BOOK.EPUB")
        assert isinstance(parser, EpubParser)

    def test_path_object(self) -> None:
        from pathlib import Path

        parser = get_parser(Path("/some/dir/file.pdf"))
        assert isinstance(parser, PdfParser)
