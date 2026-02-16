"""Tests for the HTML parser."""

import pytest

from author_library.errors import ParsingError
from author_library.parsing.html_parser import HtmlParser
from author_library.parsing.models import NodeType


@pytest.fixture
def parser() -> HtmlParser:
    return HtmlParser()


@pytest.fixture
def blog_post_html(tmp_path: object) -> object:
    from pathlib import Path

    p = Path(str(tmp_path)) / "post.html"
    p.write_text("""<!DOCTYPE html>
<html>
<head>
    <title>My Blog Post</title>
    <meta name="author" content="Jane Author">
    <meta name="date" content="2024-06-15">
</head>
<body>
    <nav><a href="/">Home</a></nav>
    <article>
        <h1>My Blog Post</h1>
        <p>This is the introduction paragraph.</p>
        <h2>First Section</h2>
        <p>Content of the first section.</p>
        <blockquote>A notable quote from someone important.</blockquote>
        <h2>Second Section</h2>
        <p>Content of the second section.</p>
        <ul>
            <li>Item one</li>
            <li>Item two</li>
            <li>Item three</li>
        </ul>
    </article>
    <footer>Copyright 2024</footer>
</body>
</html>""")
    return p


@pytest.fixture
def minimal_html(tmp_path: object) -> object:
    from pathlib import Path

    p = Path(str(tmp_path)) / "minimal.html"
    p.write_text("<html><body><p>Just a paragraph.</p></body></html>")
    return p


class TestHtmlParser:
    async def test_supported_extensions(self, parser: HtmlParser) -> None:
        exts = parser.supported_extensions()
        assert ".html" in exts
        assert ".htm" in exts
        assert ".xhtml" in exts

    async def test_parse_blog_post(self, parser: HtmlParser, blog_post_html: object) -> None:
        result = await parser.parse(blog_post_html)  # type: ignore[arg-type]
        assert result.format == "html"
        assert result.metadata.title == "My Blog Post"
        assert result.metadata.author == "Jane Author"
        assert result.metadata.publication_date == "2024-06-15"

        # Should have content nodes
        assert len(result.tree.children) > 0
        assert result.metadata.word_count > 0

        # Check that nav and footer were stripped
        assert "Home" not in result.raw_text
        assert "Copyright" not in result.raw_text

    async def test_extracts_headings(self, parser: HtmlParser, blog_post_html: object) -> None:
        result = await parser.parse(blog_post_html)  # type: ignore[arg-type]

        def find_headings(node: object) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            headings = []
            if node.node_type == NodeType.HEADING:
                headings.append(node)
            for child in node.children:
                headings.extend(find_headings(child))
            return headings

        headings = find_headings(result.tree)
        heading_texts = [h.text for h in headings]
        assert "First Section" in heading_texts
        assert "Second Section" in heading_texts

    async def test_extracts_blockquote(self, parser: HtmlParser, blog_post_html: object) -> None:
        result = await parser.parse(blog_post_html)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        quotes = find_type(result.tree, NodeType.BLOCK_QUOTE)
        assert len(quotes) >= 1
        assert "notable quote" in quotes[0].text  # type: ignore[union-attr]

    async def test_extracts_list(self, parser: HtmlParser, blog_post_html: object) -> None:
        result = await parser.parse(blog_post_html)  # type: ignore[arg-type]

        def find_type(node: object, ntype: NodeType) -> list[object]:
            from author_library.parsing.models import DocumentNode

            assert isinstance(node, DocumentNode)
            found = []
            if node.node_type == ntype:
                found.append(node)
            for child in node.children:
                found.extend(find_type(child, ntype))
            return found

        lists = find_type(result.tree, NodeType.LIST)
        assert len(lists) >= 1
        items = find_type(result.tree, NodeType.LIST_ITEM)
        assert len(items) == 3

    async def test_parse_minimal(self, parser: HtmlParser, minimal_html: object) -> None:
        result = await parser.parse(minimal_html)  # type: ignore[arg-type]
        assert result.tree.node_type == NodeType.BOOK
        assert "Just a paragraph" in result.raw_text

    async def test_file_not_found(self, parser: HtmlParser) -> None:
        with pytest.raises(ParsingError, match="not found"):
            await parser.parse("/nonexistent/file.html")
