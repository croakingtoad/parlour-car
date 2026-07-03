"""Tests for ingest tool handlers — input validation and error paths."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import asyncio
import pytest

from author_library.errors import IngestionError
from author_library.tools.ingest import handle_ingest_book, handle_ingest_corpus
from author_library.tools.ingestion_pipeline import IngestionResult


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


class TestHandleIngestBookAutoConfirm:
    """Validate auto_confirm parameter for B6."""

    async def test_auto_confirm_defaults_to_true(self) -> None:
        """With auto_confirm absent, nonexistent file raises normally (pipeline path)."""
        with pytest.raises(IngestionError, match="File not found"):
            await handle_ingest_book(
                {
                    "file_path": "/tmp/nonexistent_auto_confirm.epub",
                    "subject_author_id": "cs-lewis",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_auto_confirm_false_still_validates_file(self) -> None:
        """Even with auto_confirm=false, missing file is caught before classification."""
        with pytest.raises(IngestionError, match="File not found"):
            await handle_ingest_book(
                {
                    "file_path": "/tmp/nonexistent_auto_confirm.epub",
                    "subject_author_id": "cs-lewis",
                    "auto_confirm": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_auto_confirm_false_still_requires_file_path(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_ingest_book(
                {"subject_author_id": "cs-lewis", "auto_confirm": False},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_auto_confirm_false_still_requires_author(self) -> None:
        with pytest.raises(IngestionError, match="subject_author_id is required"):
            await handle_ingest_book(
                {"file_path": "/tmp/test.epub", "auto_confirm": False},
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


class TestIngestBookCrossWorkAnalysis:
    """Verify that handle_ingest_book calls _run_cross_work_analysis for primary sources."""

    def _make_primary_result(self) -> IngestionResult:
        return IngestionResult(
            work_id="guite--faith-hope-and-poetry",
            source_class="primary",
            processing_route="full_enrichment",
            chunks_by_granularity={"meso": 10},
            embeddings_stored=10,
            entity_count=5,
            edge_count=3,
            errors=[],
        )

    def _make_secondary_result(self) -> IngestionResult:
        return IngestionResult(
            work_id="smith--on-guites-poetry",
            source_class="secondary",
            processing_route="embeddings_and_graph",
            chunks_by_granularity={"meso": 8},
            embeddings_stored=8,
            entity_count=2,
            edge_count=1,
            errors=[],
        )

    @patch("author_library.tools.ingest._run_cross_work_analysis", new_callable=AsyncMock)
    @patch("author_library.tools.ingest.IngestionPipeline")
    async def test_primary_source_triggers_cross_work_analysis(
        self, mock_pipeline_cls: AsyncMock, mock_cross_work: AsyncMock, tmp_path
    ) -> None:
        """After ingesting a primary source, _run_cross_work_analysis must be called."""
        # Set up pipeline mock
        mock_pipeline = AsyncMock()
        mock_pipeline.ingest.return_value = self._make_primary_result()
        mock_pipeline_cls.return_value = mock_pipeline

        mock_cross_work.return_value = {
            "voice_profile": {"confidence": 0.85},
            "thematic_index": {"themes_identified": 3},
        }

        # Create a dummy file
        test_file = tmp_path / "test.epub"
        test_file.write_text("dummy content")

        result_json = await handle_ingest_book(
            {
                "file_path": str(test_file),
                "subject_author_id": "malcolm-guite",
            },
            settings=AsyncMock(),
            storage=AsyncMock(),
            embedding_provider=AsyncMock(),
        )

        # Verify _run_cross_work_analysis was called
        mock_cross_work.assert_called_once()
        call_kwargs = mock_cross_work.call_args[1]
        assert call_kwargs["subject_author_id"] == "malcolm-guite"

        # Verify response includes cross_work_analysis
        result = json.loads(result_json)
        assert "cross_work_analysis" in result
        assert result["cross_work_analysis"]["voice_profile"]["confidence"] == 0.85

    @patch("author_library.tools.ingest._run_cross_work_analysis", new_callable=AsyncMock)
    @patch("author_library.tools.ingest.IngestionPipeline")
    async def test_secondary_source_does_not_trigger_cross_work_analysis(
        self, mock_pipeline_cls: AsyncMock, mock_cross_work: AsyncMock, tmp_path
    ) -> None:
        """Secondary sources must NOT trigger voice profile / cross-work analysis."""
        mock_pipeline = AsyncMock()
        mock_pipeline.ingest.return_value = self._make_secondary_result()
        mock_pipeline_cls.return_value = mock_pipeline

        test_file = tmp_path / "test.epub"
        test_file.write_text("dummy content")

        result_json = await handle_ingest_book(
            {
                "file_path": str(test_file),
                "subject_author_id": "malcolm-guite",
            },
            settings=AsyncMock(),
            storage=AsyncMock(),
            embedding_provider=AsyncMock(),
        )

        # _run_cross_work_analysis must NOT be called for secondary sources
        mock_cross_work.assert_not_called()

        # Response should NOT contain cross_work_analysis
        result = json.loads(result_json)
        assert "cross_work_analysis" not in result


class TestPostIngestBackupGuard:
    """The production backup hook must never fire from test runs (td-aef7c5)."""

    async def test_backup_skipped_when_test_database_in_use(
        self, monkeypatch
    ) -> None:
        from author_library.tools import ingest as ingest_mod

        monkeypatch.setenv(
            "DB_POSTGRES_URL",
            "postgresql://author_library:x@localhost:5432/author_library_test",
        )

        def _explode(*args, **kwargs):
            raise AssertionError("backup subprocess must not be spawned from tests")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _explode)
        await ingest_mod._run_post_ingest_backup("test--some-work")
