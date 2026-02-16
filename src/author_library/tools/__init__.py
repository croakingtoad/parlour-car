"""MCP tool handlers for The Author Library.

Organizes tools into three epics:
  - ingestion: ingest_book, ingest_corpus (E010)
  - query: ask_author, trace_theme, find_quotes, compare_ideas (E011)
  - meta: list_authors, author_bio, list_works, library_stats (E012)

The ingestion_pipeline module provides the shared orchestration logic
used by the ingestion tools.
"""
