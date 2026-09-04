"""Tests for composable ingestion tool handlers — input validation and error paths.

Tests follow the same pattern as test_ingest.py: validate required arguments,
error handling, and edge cases without needing live database connections.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.catalog.models import ClassificationResult, SourceClass
from author_library.config import Settings
from author_library.errors import IngestionError
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)
from author_library.tools.composable_ingestion import (
    _infer_work_type,
    handle_catalog_source,
    handle_chunk_source,
    handle_classify_source,
    handle_detect_passage_links,
    handle_flag_acquisition,
)


class FakeWorkRepository:
    def __init__(self) -> None:
        self.works: dict[str, dict[str, Any]] = {}

    async def create(self, work: dict[str, Any]) -> str:
        self.works[work["work_id"]] = work
        return work["work_id"]

    async def get(self, work_id: str) -> dict[str, Any] | None:
        return self.works.get(work_id)

    async def update(self, work_id: str, fields: dict[str, Any]) -> bool:
        self.works[work_id].update(fields)
        return True


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

    @patch("author_library.tools.composable_ingestion.get_parser")
    @patch("author_library.catalog.pipeline.SourceClassifier.classify")
    async def test_subject_headings_round_trip_through_catalog_source(
        self,
        mock_classify: AsyncMock,
        mock_get_parser: MagicMock,
        tmp_path,
    ) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text about poetic imagination.")
        document = ParsedDocument(
            source_path=str(test_file),
            format="txt",
            metadata=DocumentMetadata(
                title="Faith, Hope and Poetry",
                author="Malcolm Guite",
                publisher="Ashgate",
                publication_date="2012",
                word_count=4,
            ),
            tree=DocumentNode(node_type=NodeType.BOOK, text="Text about poetic imagination."),
            raw_text="Text about poetic imagination.",
        )
        parser = AsyncMock()
        parser.parse.return_value = document
        mock_get_parser.return_value = parser
        mock_classify.return_value = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.95,
            reasoning="Authorship matches the configured subject author.",
            signals_detected=["authorship_match"],
        )
        works = FakeWorkRepository()
        storage = MagicMock()
        storage.works = works
        storage.pg = None
        storage.graph = AsyncMock()

        result_json = await handle_catalog_source(
            {
                "file_path": str(test_file),
                "source_class": "primary",
                "work_type": "monograph",
                "metadata_overrides": {
                    "subject_author_id": "malcolm-guite",
                    "subject_headings": ["Christian Poetry"],
                },
            },
            settings=Settings(),
            storage=storage,
            embedding_provider=None,  # type: ignore[arg-type]
        )

        result = json.loads(result_json)
        work_id = result["work_id"]
        assert result["catalog_record"]["subject_headings"] == ["Christian Poetry"]
        assert works.works[work_id]["subject_headings"] == ["Christian Poetry"]


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
