"""Tests for ingest tool handlers — input validation and error paths."""

from __future__ import annotations

import pytest

from author_library.errors import IngestionError
from author_library.tools.ingest import handle_ingest_book, handle_ingest_corpus


class TestHandleIngestBookValidation:
    """Validate required argument checks for ingest_book."""

    async def test_missing_file_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_ingest_book(
                {"subject_author_id": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_file_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_ingest_book(
                {"file_path": "", "subject_author_id": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_missing_subject_author_id_raises(self) -> None:
        with pytest.raises(IngestionError, match="subject_author_id is required"):
            await handle_ingest_book(
                {"file_path": "/tmp/test.epub"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            await handle_ingest_book(
                {
                    "file_path": "/tmp/nonexistent_8675309.epub",
                    "subject_author_id": "cs-lewis",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleIngestCorpusValidation:
    """Validate required argument checks for ingest_corpus."""

    async def test_missing_subject_author_id_raises(self) -> None:
        with pytest.raises(IngestionError, match="subject_author_id is required"):
            await handle_ingest_corpus(
                {"directory": "/tmp"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_no_directory_or_file_list_raises(self) -> None:
        with pytest.raises(
            IngestionError, match="Either directory or file_list is required"
        ):
            await handle_ingest_corpus(
                {"subject_author_id": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_nonexistent_directory_raises(self) -> None:
        with pytest.raises(IngestionError, match="Directory not found"):
            await handle_ingest_corpus(
                {
                    "subject_author_id": "cs-lewis",
                    "directory": "/tmp/nonexistent_dir_8675309",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_file_list_raises(self) -> None:
        """An empty file list with no existing files raises."""
        with pytest.raises(IngestionError, match="No valid files found"):
            await handle_ingest_corpus(
                {
                    "subject_author_id": "cs-lewis",
                    "file_list": ["/nonexistent_1.txt", "/nonexistent_2.txt"],
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )
