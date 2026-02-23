"""Tests for the MCP server."""

from __future__ import annotations

from author_library.config import Settings
from author_library.server import TOOLS, create_server


class TestCreateServer:
    async def test_creates_server_instance(self) -> None:
        settings = Settings()
        server = create_server(settings)
        assert server is not None
        assert server.name == "author-library"

    async def test_list_tools_handler_registered(self) -> None:
        settings = Settings()
        server = create_server(settings)
        assert server.request_handlers is not None
        assert server.name == "author-library"


class TestToolDefinitions:
    """Verify the TOOLS list has correct structure and coverage."""

    EXPECTED_TOOLS: frozenset[str] = frozenset({
        # Original tools (13)
        "ingest_book",
        "ingest_corpus",
        "ask_author",
        "trace_theme",
        "find_quotes",
        "compare_ideas",
        "list_authors",
        "author_bio",
        "list_works",
        "library_stats",
        "health_check",
        "job_status",
        "ingest_book_async",
        # Epic B: Composable Ingestion Tools (5)
        "classify_source",
        "catalog_source",
        "chunk_source",
        "detect_passage_links",
        "flag_acquisition",
        # Epic C: Query Tools + Vocabulary (3)
        "search_chunks",
        "get_passage_links",
        "manage_vocabulary",
        # Epic N: Surfacing & Synthesis (2)
        "surface_related",
        "synthesize_my_thinking",
    })

    def test_all_tools_registered(self) -> None:
        tool_names = {t.name for t in TOOLS}
        assert tool_names == self.EXPECTED_TOOLS

    def test_tool_count(self) -> None:
        assert len(TOOLS) == 23

    def test_all_tools_have_descriptions(self) -> None:
        for tool in TOOLS:
            assert tool.description, f"{tool.name} missing description"
            assert len(tool.description) > 20, f"{tool.name} description too short"

    def test_all_tools_have_input_schemas(self) -> None:
        for tool in TOOLS:
            assert tool.inputSchema is not None, f"{tool.name} missing inputSchema"
            assert tool.inputSchema["type"] == "object"

    def test_ingest_book_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ingest_book")
        assert set(tool.inputSchema["required"]) == {
            "file_path",
            "subject_author_id",
        }

    def test_ingest_book_has_auto_confirm(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ingest_book")
        auto = tool.inputSchema["properties"]["auto_confirm"]
        assert auto["type"] == "boolean"
        assert auto["default"] is True

    def test_ask_author_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ask_author")
        assert set(tool.inputSchema["required"]) == {"question", "author_id"}

    def test_ask_author_response_style_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ask_author")
        style_prop = tool.inputSchema["properties"]["response_style"]
        assert set(style_prop["enum"]) == {
            "conversational",
            "academic",
            "devotional",
            "lecture",
        }

    def test_compare_ideas_requires_topic_and_authors(self) -> None:
        tool = next(t for t in TOOLS if t.name == "compare_ideas")
        assert set(tool.inputSchema["required"]) == {"topic", "author_ids"}

    def test_find_quotes_source_class_filter_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "find_quotes")
        scf = tool.inputSchema["properties"]["source_class_filter"]
        assert set(scf["enum"]) == {
            "primary",
            "secondary",
            "contextual",
            "tertiary",
        }

    def test_list_works_source_class_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "list_works")
        sc = tool.inputSchema["properties"]["source_class"]
        assert set(sc["enum"]) == {
            "primary",
            "secondary",
            "contextual",
            "tertiary",
        }

    def test_library_stats_has_no_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "library_stats")
        assert "required" not in tool.inputSchema

    def test_list_authors_has_no_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "list_authors")
        assert "required" not in tool.inputSchema

    def test_trace_theme_requires_theme_name(self) -> None:
        tool = next(t for t in TOOLS if t.name == "trace_theme")
        assert tool.inputSchema["required"] == ["theme_name"]

    def test_ingest_corpus_requires_subject_author_id(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ingest_corpus")
        assert tool.inputSchema["required"] == ["subject_author_id"]

    # ----- Epic B: Composable Ingestion Tools -----

    def test_classify_source_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "classify_source")
        assert set(tool.inputSchema["required"]) == {"file_path", "subject_author"}

    def test_catalog_source_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "catalog_source")
        assert set(tool.inputSchema["required"]) == {"file_path", "source_class"}

    def test_catalog_source_source_class_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "catalog_source")
        sc = tool.inputSchema["properties"]["source_class"]
        assert set(sc["enum"]) == {
            "primary", "secondary", "contextual", "tertiary", "personal",
        }

    def test_chunk_source_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "chunk_source")
        assert tool.inputSchema["required"] == ["work_id"]

    def test_detect_passage_links_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "detect_passage_links")
        assert tool.inputSchema["required"] == ["work_id"]

    def test_detect_passage_links_scan_types_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "detect_passage_links")
        scan_types = tool.inputSchema["properties"]["scan_types"]
        assert set(scan_types["items"]["enum"]) == {
            "explicit_citation", "implicit_engagement", "thematic_parallel",
        }

    def test_flag_acquisition_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "flag_acquisition")
        assert tool.inputSchema["required"] == ["citations"]

    def test_flag_acquisition_citation_priority_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "flag_acquisition")
        items = tool.inputSchema["properties"]["citations"]["items"]
        assert set(items["properties"]["priority"]["enum"]) == {
            "high", "medium", "low",
        }

    # ----- Epic C: Query Tools + Vocabulary -----

    def test_search_chunks_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "search_chunks")
        assert tool.inputSchema["required"] == ["query"]

    def test_search_chunks_filters_schema(self) -> None:
        tool = next(t for t in TOOLS if t.name == "search_chunks")
        filters = tool.inputSchema["properties"]["filters"]
        assert "source_class" in filters["properties"]
        assert "work_ids" in filters["properties"]
        assert "granularity" in filters["properties"]

    def test_get_passage_links_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "get_passage_links")
        assert tool.inputSchema["required"] == ["chunk_id"]

    def test_get_passage_links_depth_constraints(self) -> None:
        tool = next(t for t in TOOLS if t.name == "get_passage_links")
        depth = tool.inputSchema["properties"]["depth"]
        assert depth["minimum"] == 1
        assert depth["maximum"] == 3

    def test_manage_vocabulary_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t.name == "manage_vocabulary")
        assert tool.inputSchema["required"] == ["action"]

    def test_manage_vocabulary_action_enum(self) -> None:
        tool = next(t for t in TOOLS if t.name == "manage_vocabulary")
        action = tool.inputSchema["properties"]["action"]
        assert set(action["enum"]) == {
            "list", "propose", "promote", "merge", "deprecate",
        }
