"""Tests for Epic D MCP tool registrations (D3).

Tests cover:
  - job_status and ingest_book_async tools are in the TOOLS list
  - Tool schemas have required fields
  - Total tool count includes all tools (21 total: 11 original + 2 Epic D + 5 Epic B + 3 Epic C)
"""

from __future__ import annotations

from author_library.server import TOOLS


class TestToolRegistrations:
    def test_total_tool_count(self) -> None:
        """21 tools: 11 original + 2 Epic D + 5 Epic B + 3 Epic C."""
        assert len(TOOLS) == 21

    def test_job_status_tool_registered(self) -> None:
        names = [t.name for t in TOOLS]
        assert "job_status" in names

    def test_ingest_book_async_tool_registered(self) -> None:
        names = [t.name for t in TOOLS]
        assert "ingest_book_async" in names

    def test_job_status_requires_job_id(self) -> None:
        tool = next(t for t in TOOLS if t.name == "job_status")
        assert "job_id" in tool.inputSchema.get("required", [])
        assert "job_id" in tool.inputSchema.get("properties", {})

    def test_ingest_book_async_requires_file_path_and_author(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ingest_book_async")
        required = tool.inputSchema.get("required", [])
        assert "file_path" in required
        assert "subject_author_id" in required

    def test_ingest_book_async_has_metadata_hints(self) -> None:
        tool = next(t for t in TOOLS if t.name == "ingest_book_async")
        props = tool.inputSchema.get("properties", {})
        assert "metadata_hints" in props

    def test_all_tools_have_description(self) -> None:
        for tool in TOOLS:
            assert tool.description, f"{tool.name} has no description"

    def test_all_tools_have_input_schema(self) -> None:
        for tool in TOOLS:
            assert tool.inputSchema is not None, f"{tool.name} has no inputSchema"
            assert "type" in tool.inputSchema
