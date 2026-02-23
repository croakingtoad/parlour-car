"""MCP tool handlers for The Author Library.

Organizes tools into five groups:
  - ingestion: ingest_book, ingest_corpus (E010)
  - query: ask_author, trace_theme, find_quotes, compare_ideas (E011)
  - meta: list_authors, author_bio, list_works, library_stats (E012)
  - composable_ingestion: classify_source, catalog_source, chunk_source,
      detect_passage_links, flag_acquisition (Epic B)
  - composable_query: search_chunks, get_passage_links, manage_vocabulary (Epic C)

The ingestion_pipeline module provides the shared orchestration logic
used by the ingestion tools.
"""
