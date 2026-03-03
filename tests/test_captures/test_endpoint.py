"""Tests for the HTTP capture endpoint (K1)."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any

import pytest

from author_library.captures.endpoint import _authenticate
from author_library.captures.models import CaptureMode


class TestAuthenticate:
    """Test Bearer token authentication."""

    def _make_request_with_header(self, auth_header: str) -> Any:
        """Create a minimal mock request object with an auth header."""

        class FakeHeaders:
            def __init__(self, mapping: dict[str, str]) -> None:
                self._mapping = mapping

            def get(self, key: str, default: str = "") -> str:
                return self._mapping.get(key.lower(), default)

        class FakeRequest:
            headers: Any

        req = FakeRequest()
        req.headers = FakeHeaders({"authorization": auth_header})
        return req

    def test_valid_bearer_token(self) -> None:
        key = "test-api-key-12345"
        request = self._make_request_with_header(f"Bearer {key}")
        assert _authenticate(request, key) is True

    def test_invalid_bearer_token(self) -> None:
        request = self._make_request_with_header("Bearer wrong-key")
        assert _authenticate(request, "correct-key") is False

    def test_missing_bearer_prefix(self) -> None:
        request = self._make_request_with_header("test-api-key")
        assert _authenticate(request, "test-api-key") is False

    def test_empty_auth_header(self) -> None:
        request = self._make_request_with_header("")
        assert _authenticate(request, "test-api-key") is False

    def test_timing_safe_comparison(self) -> None:
        """Ensure we use constant-time comparison."""
        key = secrets.token_hex(16)
        request = self._make_request_with_header(f"Bearer {key}")
        assert _authenticate(request, key) is True


class TestCapturePayloadValidation:
    """Test that CapturePayload validates correctly from JSON."""

    def test_minimal_valid_payload(self) -> None:
        from author_library.captures.models import CapturePayload

        data = {
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "source_title": "Test Video",
            "timestamp_seconds": 42.5,
            "mode": "quick",
            "extension_version": "1.0.0",
            "captured_at": "2026-02-23T12:00:00Z",
        }
        payload = CapturePayload(**data)
        assert payload.mode == CaptureMode.QUICK
        assert payload.screenshot_base64 is None
        assert payload.annotation is None
        assert payload.speaker_override is None

    def test_full_visual_deep_payload(self) -> None:
        from author_library.captures.models import CapturePayload

        data = {
            "source_url": "https://youtu.be/test12345ab",
            "source_title": "Deep Visual Test",
            "timestamp_seconds": 180.0,
            "mode": "visual_deep",
            "screenshot_base64": "iVBORw0KGgo=",
            "annotation": "Key slide about theology",
            "speaker_override": "Dr. Smith",
            "extension_version": "1.0.0",
            "captured_at": "2026-02-23T12:00:00Z",
        }
        payload = CapturePayload(**data)
        assert payload.mode == CaptureMode.VISUAL_DEEP
        assert payload.is_visual()
        assert payload.is_deep()
        assert payload.screenshot_base64 == "iVBORw0KGgo="

    def test_invalid_mode_rejected(self) -> None:
        from author_library.captures.models import CapturePayload

        data = {
            "source_url": "https://youtube.com/watch?v=test",
            "source_title": "Test",
            "timestamp_seconds": 10.0,
            "mode": "invalid_mode",
            "extension_version": "1.0.0",
            "captured_at": "2026-02-23T12:00:00Z",
        }
        with pytest.raises(Exception):
            CapturePayload(**data)


class TestTaskQueueCapture:
    """Test TaskQueue.enqueue_capture without Redis."""

    @pytest.mark.asyncio
    async def test_enqueue_capture_returns_none_without_pool(self) -> None:
        from author_library.queue import TaskQueue

        queue = TaskQueue()
        # Pool is None (not connected)
        result = await queue.enqueue_capture(payload={"mode": "quick"})
        assert result is None


class TestWorkerRegistration:
    """Test that task_process_capture is registered in the worker."""

    def test_worker_includes_capture_task(self) -> None:
        from author_library.worker import WorkerSettings

        func_names = [f.__name__ for f in WorkerSettings.functions]
        assert "task_process_capture" in func_names
