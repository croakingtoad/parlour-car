"""Tests for P1: Re-engagement detector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from author_library.delta.reengagement import (
    ReengagementDetector,
    ReengagementInfo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.works = MagicMock()
    storage.chunks = MagicMock()
    return storage


@pytest.fixture()
def detector(mock_storage):
    return ReengagementDetector(storage=mock_storage)


# ---------------------------------------------------------------------------
# ReengagementInfo tests
# ---------------------------------------------------------------------------


class TestReengagementInfo:
    def test_first_engagement(self):
        info = ReengagementInfo(
            work_id="guite--faith-hope-poetry",
            current_pass=1,
            previous_pass=0,
            is_reengagement=False,
            previous_capture_count=0,
        )
        assert not info.is_reengagement
        assert info.current_pass == 1

    def test_reengagement(self):
        info = ReengagementInfo(
            work_id="guite--faith-hope-poetry",
            current_pass=2,
            previous_pass=1,
            is_reengagement=True,
            previous_capture_count=15,
            metadata={
                "work_title": "Faith, Hope and Poetry",
                "first_engagement_date": "2025-06-15T10:00:00",
                "last_engagement_date": "2026-01-20T14:00:00",
            },
        )
        assert info.is_reengagement
        assert info.current_pass == 2
        assert info.work_title == "Faith, Hope and Poetry"
        assert info.first_engagement_date == "2025-06-15T10:00:00"

    def test_third_pass(self):
        info = ReengagementInfo(
            work_id="guite--faith-hope-poetry",
            current_pass=3,
            previous_pass=2,
            is_reengagement=True,
            previous_capture_count=30,
        )
        assert info.current_pass == 3

    def test_metadata_defaults(self):
        info = ReengagementInfo(
            work_id="w-1",
            current_pass=1,
            previous_pass=0,
            is_reengagement=False,
            previous_capture_count=0,
        )
        assert info.work_title == ""
        assert info.first_engagement_date == ""
        assert info.last_engagement_date == ""


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestDetector:
    @pytest.mark.asyncio()
    async def test_first_engagement(self, detector, mock_storage):
        """No existing captures — first engagement."""
        mock_storage.chunks.get_max_pass_number = AsyncMock(return_value=0)
        mock_storage.pg.fetch_val = AsyncMock(return_value=0)
        mock_storage.works.get = AsyncMock(return_value={
            "title": "Faith, Hope and Poetry",
            "author": "Malcolm Guite",
            "media": "book",
        })
        mock_storage.pg.fetch_one = AsyncMock(return_value=None)

        result = await detector.detect("guite--faith-hope-poetry")

        assert not result.is_reengagement
        assert result.current_pass == 1
        assert result.previous_capture_count == 0

    @pytest.mark.asyncio()
    async def test_reengagement_detected(self, detector, mock_storage):
        """Existing captures with pass_number=1 — re-engagement."""
        mock_storage.chunks.get_max_pass_number = AsyncMock(return_value=1)
        mock_storage.pg.fetch_val = AsyncMock(return_value=15)
        mock_storage.works.get = AsyncMock(return_value={
            "title": "Faith, Hope and Poetry",
            "author": "Malcolm Guite",
            "media": "book",
        })
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "first_date": "2025-06-15T10:00:00",
            "last_date": "2025-06-15T12:00:00",
        })

        result = await detector.detect("guite--faith-hope-poetry")

        assert result.is_reengagement
        assert result.current_pass == 2
        assert result.previous_pass == 1
        assert result.previous_capture_count == 15
        assert result.work_title == "Faith, Hope and Poetry"
        assert result.first_engagement_date == "2025-06-15T10:00:00"

    @pytest.mark.asyncio()
    async def test_third_pass(self, detector, mock_storage):
        """Third engagement increments to pass 3."""
        mock_storage.chunks.get_max_pass_number = AsyncMock(return_value=2)
        mock_storage.pg.fetch_val = AsyncMock(return_value=30)
        mock_storage.works.get = AsyncMock(return_value={
            "title": "Imagination Lecture",
            "author": "Malcolm Guite",
            "media": "video",
        })
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "first_date": "2025-01-01",
            "last_date": "2026-01-01",
        })

        result = await detector.detect("guite--imagination-lecture")

        assert result.is_reengagement
        assert result.current_pass == 3
        assert result.previous_pass == 2

    @pytest.mark.asyncio()
    async def test_no_work_info(self, detector, mock_storage):
        """Work not found in catalog — still works."""
        mock_storage.chunks.get_max_pass_number = AsyncMock(return_value=0)
        mock_storage.pg.fetch_val = AsyncMock(return_value=0)
        mock_storage.works.get = AsyncMock(return_value=None)
        mock_storage.pg.fetch_one = AsyncMock(return_value=None)

        result = await detector.detect("unknown-work")

        assert not result.is_reengagement
        assert result.work_title == ""


class TestPassHistory:
    @pytest.mark.asyncio()
    async def test_empty_history(self, detector, mock_storage):
        """No captures — empty history."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        history = await detector.get_pass_history("guite--faith-hope-poetry")

        assert history == []

    @pytest.mark.asyncio()
    async def test_single_pass(self, detector, mock_storage):
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "pass_number": 1,
                "capture_count": 15,
                "first_capture": "2025-06-15T10:00:00",
                "last_capture": "2025-06-15T12:00:00",
            },
        ])

        history = await detector.get_pass_history("guite--faith-hope-poetry")

        assert len(history) == 1
        assert history[0]["pass_number"] == 1
        assert history[0]["capture_count"] == 15

    @pytest.mark.asyncio()
    async def test_multiple_passes(self, detector, mock_storage):
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "pass_number": 1,
                "capture_count": 15,
                "first_capture": "2025-06-15",
                "last_capture": "2025-06-15",
            },
            {
                "pass_number": 2,
                "capture_count": 8,
                "first_capture": "2026-01-20",
                "last_capture": "2026-01-20",
            },
        ])

        history = await detector.get_pass_history("guite--faith-hope-poetry")

        assert len(history) == 2
        assert history[0]["pass_number"] == 1
        assert history[1]["pass_number"] == 2


class TestGetCapturesByPass:
    @pytest.mark.asyncio()
    async def test_get_captures(self, detector, mock_storage):
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "chunk_id": "chunk-1",
                "work_id": "w-1",
                "text": "First capture",
                "granularity": "meso",
                "source_class": "primary",
                "chapter": "Ch 3",
                "section": None,
                "position": 0,
                "date_created": "2025-06-15",
                "metadata": {},
            },
        ])

        captures = await detector.get_captures_by_pass("w-1", 1)

        assert len(captures) == 1
        assert captures[0]["chunk_id"] == "chunk-1"
        assert captures[0]["granularity"] == "meso"

    @pytest.mark.asyncio()
    async def test_respects_limit(self, detector, mock_storage):
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        await detector.get_captures_by_pass("w-1", 1, limit=5)

        call_args = mock_storage.pg.fetch_all.call_args
        # Verify limit param was passed
        assert call_args[0][-1] == 5
