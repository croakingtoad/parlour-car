"""Tests for the document tree data model."""

from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)


class TestNodeType:
    def test_all_node_types_are_strings(self) -> None:
        for nt in NodeType:
            assert isinstance(nt.value, str)

    def test_key_node_types_exist(self) -> None:
        assert NodeType.BOOK == "book"
        assert NodeType.CHAPTER == "chapter"
        assert NodeType.PARAGRAPH == "paragraph"
        assert NodeType.POEM == "poem"
        assert NodeType.STANZA == "stanza"
        assert NodeType.FOOTNOTE == "footnote"
        assert NodeType.BLOCK_QUOTE == "block_quote"


class TestDocumentNode:
    def test_default_id_generated(self) -> None:
        node = DocumentNode(node_type=NodeType.PARAGRAPH)
        assert len(node.id) == 12

    def test_unique_ids(self) -> None:
        a = DocumentNode(node_type=NodeType.PARAGRAPH)
        b = DocumentNode(node_type=NodeType.PARAGRAPH)
        assert a.id != b.id

    def test_text_defaults_empty(self) -> None:
        node = DocumentNode(node_type=NodeType.CHAPTER)
        assert node.text == ""

    def test_children_defaults_empty(self) -> None:
        node = DocumentNode(node_type=NodeType.CHAPTER)
        assert node.children == []

    def test_recursive_tree(self) -> None:
        leaf = DocumentNode(node_type=NodeType.PARAGRAPH, text="Hello world")
        chapter = DocumentNode(
            node_type=NodeType.CHAPTER,
            children=[leaf],
            metadata={"title": "Ch 1"},
        )
        root = DocumentNode(
            node_type=NodeType.BOOK,
            children=[chapter],
        )
        assert len(root.children) == 1
        assert root.children[0].node_type == NodeType.CHAPTER
        assert root.children[0].children[0].text == "Hello world"

    def test_metadata_flexible_types(self) -> None:
        node = DocumentNode(
            node_type=NodeType.HEADING,
            text="Test",
            metadata={
                "level": 2,
                "bold": True,
                "tags": ["intro", "summary"],
                "title": "Test Heading",
            },
        )
        assert node.metadata["level"] == 2
        assert node.metadata["bold"] is True
        assert node.metadata["tags"] == ["intro", "summary"]

    def test_serialization_roundtrip(self) -> None:
        original = DocumentNode(
            node_type=NodeType.BOOK,
            children=[
                DocumentNode(
                    node_type=NodeType.CHAPTER,
                    children=[
                        DocumentNode(node_type=NodeType.PARAGRAPH, text="Para 1"),
                    ],
                    metadata={"title": "Chapter 1"},
                ),
            ],
        )
        json_str = original.model_dump_json()
        restored = DocumentNode.model_validate_json(json_str)
        assert restored.node_type == NodeType.BOOK
        assert len(restored.children) == 1
        assert restored.children[0].children[0].text == "Para 1"


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        meta = DocumentMetadata()
        assert meta.title is None
        assert meta.language == "en"
        assert meta.word_count == 0
        assert meta.table_of_contents == []

    def test_full_metadata(self) -> None:
        meta = DocumentMetadata(
            title="The Great Book",
            author="Author Name",
            publication_date="2024-01-01",
            publisher="Publisher Inc",
            isbn="978-0-123456-78-9",
            language="en",
            table_of_contents=["Chapter 1", "Chapter 2"],
            word_count=50000,
        )
        assert meta.title == "The Great Book"
        assert meta.isbn == "978-0-123456-78-9"


class TestParsedDocument:
    def test_minimal_parsed_document(self) -> None:
        doc = ParsedDocument(
            source_path="/tmp/test.txt",
            format="txt",
            metadata=DocumentMetadata(),
            tree=DocumentNode(node_type=NodeType.BOOK),
        )
        assert doc.raw_text == ""
        assert doc.parse_warnings == []

    def test_with_warnings(self) -> None:
        doc = ParsedDocument(
            source_path="/tmp/test.pdf",
            format="pdf",
            metadata=DocumentMetadata(word_count=100),
            tree=DocumentNode(node_type=NodeType.BOOK),
            raw_text="some text",
            parse_warnings=["Low OCR quality"],
        )
        assert len(doc.parse_warnings) == 1
        assert doc.parse_warnings[0] == "Low OCR quality"
