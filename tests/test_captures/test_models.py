"""Tests for capture data models (K1)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from author_library.captures.models import (
    CaptureMode,
    CapturePayload,
    CaptureResult,
    SourceOverview,
)


class TestCaptureMode:
    def test_enum_values(self) -> None:
        assert CaptureMode.QUICK.value == "quick"
        assert CaptureMode.DEEP.value == "deep"
        assert CaptureMode.VISUAL_QUICK.value == "visual_quick"
        assert CaptureMode.VISUAL_DEEP.value == "visual_deep"


class TestCapturePayload:
    def test_valid_quick_payload(self) -> None:
        payload = CapturePayload(
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            source_title="Test Video",
            timestamp_seconds=42.5,
            mode=CaptureMode.QUICK,
            extension_version="1.0.0",
            captured_at=datetime.now(tz=timezone.utc),
        )
        assert payload.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert payload.mode == CaptureMode.QUICK
        assert not payload.is_visual()
        assert not payload.is_deep()

    def test_valid_deep_payload(self) -> None:
        payload = CapturePayload(
            source_url="https://youtu.be/abc123def45",
            source_title="Deep Video",
            timestamp_seconds=120.0,
            mode=CaptureMode.DEEP,
            annotation="Important point about theology",
            extension_version="1.0.0",
            captured_at=datetime.now(tz=timezone.utc),
        )
        assert payload.is_deep()
        assert not payload.is_visual()

    def test_valid_visual_quick_payload(self) -> None:
        payload = CapturePayload(
            source_url="https://www.youtube.com/watch?v=test12345ab",
            source_title="Visual Test",
            timestamp_seconds=60.0,
            mode=CaptureMode.VISUAL_QUICK,
            screenshot_base64="iVBORw0KGgo=",
            extension_version="1.0.0",
            captured_at=datetime.now(tz=timezone.utc),
        )
        assert payload.is_visual()
        assert not payload.is_deep()

    def test_valid_visual_deep_payload(self) -> None:
        payload = CapturePayload(
            source_url="https://www.youtube.com/watch?v=test12345ab",
            source_title="Visual Deep",
            timestamp_seconds=180.0,
            mode=CaptureMode.VISUAL_DEEP,
            screenshot_base64="iVBORw0KGgo=",
            speaker_override="C.S. Lewis",
            extension_version="1.0.0",
            captured_at=datetime.now(tz=timezone.utc),
        )
        assert payload.is_visual()
        assert payload.is_deep()

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(Exception):
            CapturePayload(
                source_url="https://youtube.com/watch?v=abc",
                # missing source_title
                timestamp_seconds=10.0,
                mode=CaptureMode.QUICK,
                extension_version="1.0.0",
                captured_at=datetime.now(tz=timezone.utc),
            )

    def test_serialization_roundtrip(self) -> None:
        payload = CapturePayload(
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            source_title="Test Video",
            timestamp_seconds=42.5,
            mode=CaptureMode.QUICK,
            extension_version="1.0.0",
            captured_at=datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc),
        )
        data = payload.model_dump(mode="json")
        restored = CapturePayload(**data)
        assert restored.source_url == payload.source_url
        assert restored.mode == payload.mode
        assert restored.timestamp_seconds == payload.timestamp_seconds


class TestCaptureResult:
    def test_result_to_dict(self) -> None:
        result = CaptureResult(
            capture_id="abc123",
            source_url="https://youtube.com/watch?v=test",
            work_id="video--test12345ab",
            chunk_id="uuid-hex",
            mode="quick",
        )
        d = result.to_dict()
        assert d["capture_id"] == "abc123"
        assert d["work_id"] == "video--test12345ab"
        assert d["chunk_id"] == "uuid-hex"
        assert d["errors"] == []

    def test_result_with_errors(self) -> None:
        result = CaptureResult(
            capture_id="def456",
            source_url="https://youtube.com/watch?v=test",
            work_id="video--test",
            mode="deep",
            errors=["No transcript available"],
        )
        assert len(result.errors) == 1


class TestSourceOverview:
    def test_overview_to_dict(self) -> None:
        overview = SourceOverview(
            source_url="https://youtube.com/watch?v=test",
            title="Great Lecture",
            speakers=["Dr. Smith", "Prof. Jones"],
            content_type="lecture",
            topic_summary="A deep discussion about philosophy.",
            structural_arc="Introduction, three main arguments, conclusion.",
        )
        d = overview.to_dict()
        assert d["title"] == "Great Lecture"
        assert len(d["speakers"]) == 2
        assert d["content_type"] == "lecture"
