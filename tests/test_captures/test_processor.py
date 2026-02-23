"""Tests for the capture processor orchestration (K1-K6 integration)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from author_library.captures.models import CaptureMode, CapturePayload, CaptureResult


class TestCaptureResultSerialization:
    def test_to_dict_full(self) -> None:
        result = CaptureResult(
            capture_id="abc123",
            source_url="https://youtube.com/watch?v=test12345ab",
            work_id="video--test12345ab",
            chunk_id="deadbeef",
            mode="quick",
            errors=["warning: something minor"],
        )
        d = result.to_dict()
        assert d["capture_id"] == "abc123"
        assert d["work_id"] == "video--test12345ab"
        assert d["chunk_id"] == "deadbeef"
        assert d["mode"] == "quick"
        assert len(d["errors"]) == 1

    def test_to_dict_no_chunk(self) -> None:
        result = CaptureResult(
            capture_id="xyz789",
            source_url="https://youtube.com/watch?v=test",
            work_id="video--test",
            mode="deep",
        )
        d = result.to_dict()
        assert d["chunk_id"] is None
        assert d["errors"] == []


class TestCapturePayloadModes:
    def _make_payload(self, mode: str, **kwargs: Any) -> CapturePayload:
        base = {
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "source_title": "Test",
            "timestamp_seconds": 42.0,
            "mode": mode,
            "extension_version": "1.0.0",
            "captured_at": datetime.now(tz=timezone.utc),
        }
        base.update(kwargs)
        return CapturePayload(**base)

    def test_quick_mode_properties(self) -> None:
        p = self._make_payload("quick")
        assert not p.is_visual()
        assert not p.is_deep()

    def test_deep_mode_properties(self) -> None:
        p = self._make_payload("deep")
        assert not p.is_visual()
        assert p.is_deep()

    def test_visual_quick_mode_properties(self) -> None:
        p = self._make_payload("visual_quick")
        assert p.is_visual()
        assert not p.is_deep()

    def test_visual_deep_mode_properties(self) -> None:
        p = self._make_payload("visual_deep")
        assert p.is_visual()
        assert p.is_deep()


class TestWorkIdGeneration:
    """Test that work_id is correctly derived from source URLs."""

    def test_youtube_url_produces_video_work_id(self) -> None:
        from author_library.captures.transcript import extract_video_id

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        work_id = f"video--{video_id}"
        assert work_id == "video--dQw4w9WgXcQ"

    def test_non_youtube_url_produces_hash_work_id(self) -> None:
        from author_library.captures.transcript import extract_video_id

        url = "https://example.com/lecture/123"
        video_id = extract_video_id(url)
        assert video_id is None
        # Processor would use hash-based ID
        work_id = f"media--{hash(url) & 0xFFFFFFFF:08x}"
        assert work_id.startswith("media--")


class TestTaskProcessCapture:
    """Test the arq task wrapper can be imported and called."""

    def test_task_function_exists(self) -> None:
        from author_library.tasks import task_process_capture

        assert callable(task_process_capture)

    def test_task_function_name(self) -> None:
        from author_library.tasks import task_process_capture

        assert task_process_capture.__name__ == "task_process_capture"
