"""Tests for composable ingestion tool handlers — input validation and error paths.

Tests follow the same pattern as test_ingest.py: validate required arguments,
error handling, and edge cases without needing live database connections.
"""

from __future__ import annotations

import pytest

from author_library.errors import IngestionError
from author_library.tools.composable_ingestion import (
    _infer_work_type,
    handle_catalog_source,
    handle_chunk_source,
    handle_classify_source,
    handle_detect_passage_links,
    handle_flag_acquisition,
)


class TestHandleClassifySourceValidation:
    """Validate required argument checks for classify_source."""

    async def test_missing_file_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_classify_source(
                {"subject_author": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_file_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_classify_source(
                {"file_path": "", "subject_author": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_missing_subject_author_raises(self) -> None:
        with pytest.raises(IngestionError, match="subject_author is required"):
            await handle_classify_source(
                {"file_path": "/tmp/test.epub"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            await handle_classify_source(
                {
                    "file_path": "/tmp/nonexistent_epic_b_8675309.epub",
                    "subject_author": "cs-lewis",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleCatalogSourceValidation:
    """Validate required argument checks for catalog_source."""

    async def test_missing_file_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="file_path is required"):
            await handle_catalog_source(
                {"source_class": "primary"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_missing_source_class_raises(self) -> None:
        with pytest.raises(IngestionError, match="source_class is required"):
            await handle_catalog_source(
                {"file_path": "/tmp/test.epub"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_invalid_source_class_raises(self) -> None:
        with pytest.raises(IngestionError, match="Invalid source_class"):
            await handle_catalog_source(
                {
                    "file_path": "/tmp/test.epub",
                    "source_class": "invalid_class",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            await handle_catalog_source(
                {
                    "file_path": "/tmp/nonexistent_epic_b_8675309.epub",
                    "source_class": "primary",
                },
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_valid_source_class_values(self) -> None:
        """All six source classes should be accepted (modulo file existence)."""
        for sc in (
            "primary",
            "secondary",
            "contextual",
            "tertiary",
            "personal",
            "reference",
        ):
            with pytest.raises(IngestionError, match="File not found"):
                await handle_catalog_source(
                    {
                        "file_path": "/tmp/nonexistent_epic_b_8675309.epub",
                        "source_class": sc,
                    },
                    settings=None,  # type: ignore[arg-type]
                    storage=None,  # type: ignore[arg-type]
                    embedding_provider=None,  # type: ignore[arg-type]
                )


class TestHandleChunkSourceValidation:
    """Validate required argument checks for chunk_source."""

    async def test_missing_work_id_raises(self) -> None:
        with pytest.raises(IngestionError, match="work_id is required"):
            await handle_chunk_source(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_work_id_raises(self) -> None:
        with pytest.raises(IngestionError, match="work_id is required"):
            await handle_chunk_source(
                {"work_id": ""},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleDetectPassageLinksValidation:
    """Validate required argument checks for detect_passage_links."""

    async def test_missing_work_id_raises(self) -> None:
        with pytest.raises(IngestionError, match="work_id is required"):
            await handle_detect_passage_links(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleFlagAcquisitionValidation:
    """Validate required argument checks for flag_acquisition."""

    async def test_missing_citations_raises(self) -> None:
        with pytest.raises(IngestionError, match="citations list is required"):
            await handle_flag_acquisition(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_citations_raises(self) -> None:
        with pytest.raises(IngestionError, match="citations list is required"):
            await handle_flag_acquisition(
                {"citations": []},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_non_list_citations_raises(self) -> None:
        with pytest.raises(IngestionError, match="citations list is required"):
            await handle_flag_acquisition(
                {"citations": "not a list"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestInferWorkType:
    """Test the _infer_work_type helper."""

    def test_poetry_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="This is a poetry collection.",
            signals_detected=["poetry collection detected"],
        )
        assert _infer_work_type(result) == "poetry-collection"

    def test_sermon_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="This is a published sermon.",
            signals_detected=["sermon format"],
        )
        assert _infer_work_type(result) == "sermon"

    def test_lecture_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="Lecture transcript from university series.",
            signals_detected=["lecture format"],
        )
        assert _infer_work_type(result) == "lecture-transcript"

    def test_default_to_monograph(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="A book by the subject author.",
            signals_detected=["author name match"],
        )
        assert _infer_work_type(result) == "monograph"

    def test_academic_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="An academic paper published in a journal.",
            signals_detected=["academic paper"],
        )
        assert _infer_work_type(result) == "academic-paper"

    def test_essay_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="A collection of essays.",
            signals_detected=["essay collection"],
        )
        assert _infer_work_type(result) == "essay-collection"

    def test_interview_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="Interview responses from the author.",
            signals_detected=["interview format"],
        )
        assert _infer_work_type(result) == "interview-responses"

    def test_letter_detection(self) -> None:
        from author_library.catalog.models import ClassificationResult, SourceClass

        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.9,
            reasoning="A collection of personal letters.",
            signals_detected=["letter correspondence"],
        )
        assert _infer_work_type(result) == "letter"
