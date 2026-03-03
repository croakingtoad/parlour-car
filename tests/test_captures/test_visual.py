"""Tests for visual capture processing (K6)."""

from __future__ import annotations

import base64

import pytest

from author_library.captures.visual import (
    _detect_extension,
    _detect_media_type,
    save_screenshot,
)


class TestDetectMediaType:
    def test_png_magic_bytes(self) -> None:
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        b64 = base64.b64encode(png_header).decode()
        assert _detect_media_type(b64) == "image/png"

    def test_jpeg_magic_bytes(self) -> None:
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 28
        b64 = base64.b64encode(jpeg_header).decode()
        assert _detect_media_type(b64) == "image/jpeg"

    def test_data_uri_prefix(self) -> None:
        assert _detect_media_type("data:image/webp;base64,AAAA") == "image/webp"

    def test_unknown_defaults_to_png(self) -> None:
        assert _detect_media_type("AAAA") == "image/png"


class TestDetectExtension:
    def test_png(self) -> None:
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        b64 = base64.b64encode(png_header).decode()
        assert _detect_extension(b64) == "png"

    def test_jpeg(self) -> None:
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 28
        b64 = base64.b64encode(jpeg_header).decode()
        assert _detect_extension(b64) == "jpg"


class TestSaveScreenshot:
    def test_save_creates_file(self, tmp_path: str) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            data = b"fake image data for testing"
            b64 = base64.b64encode(data).decode()
            path = save_screenshot(
                b64,
                capture_id="test-capture-001",
                screenshots_dir=tmpdir,
            )
            assert path is not None
            assert Path(path).exists()
            assert Path(path).read_bytes() == data

    def test_save_invalid_base64_returns_none(self) -> None:
        result = save_screenshot(
            "not-valid-base64!!!",
            capture_id="test-bad",
            screenshots_dir="/tmp/test-screenshots",
        )
        assert result is None
