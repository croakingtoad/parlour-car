"""Tests for transcript retrieval and caching (K2)."""

from __future__ import annotations

import pytest

from author_library.captures.transcript import (
    _format_seconds,
    _parse_line_timestamp,
    _parse_xml_transcript,
    extract_transcript_window,
    extract_video_id,
)


class TestExtractVideoId:
    def test_standard_youtube_url(self) -> None:
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_youtube_url(self) -> None:
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_youtube_url(self) -> None:
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_non_youtube_url_returns_none(self) -> None:
        assert extract_video_id("https://example.com/video") is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_video_id("") is None

    def test_malformed_url_returns_none(self) -> None:
        assert extract_video_id("not-a-url") is None


class TestFormatSeconds:
    def test_seconds_only(self) -> None:
        assert _format_seconds(45) == "0:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_seconds(125) == "2:05"

    def test_hours(self) -> None:
        assert _format_seconds(3661) == "1:01:01"

    def test_zero(self) -> None:
        assert _format_seconds(0) == "0:00"


class TestParseLineTimestamp:
    def test_mm_ss_format(self) -> None:
        assert _parse_line_timestamp("[1:23] Hello world") == 83.0

    def test_hh_mm_ss_format(self) -> None:
        assert _parse_line_timestamp("[1:02:03] Text here") == 3723.0

    def test_no_timestamp(self) -> None:
        assert _parse_line_timestamp("Just plain text") is None

    def test_zero_timestamp(self) -> None:
        assert _parse_line_timestamp("[0:00] Opening") == 0.0


class TestExtractTranscriptWindow:
    SAMPLE_TRANSCRIPT = (
        "[0:00] Welcome to the lecture.\n"
        "[0:10] Today we discuss philosophy.\n"
        "[0:20] Let's begin with Plato.\n"
        "[0:30] Plato believed in forms.\n"
        "[0:40] The allegory of the cave.\n"
        "[0:50] Shadows on the wall.\n"
        "[1:00] Now let's move to Aristotle.\n"
        "[1:10] Aristotle was Plato's student.\n"
        "[1:20] He disagreed on forms.\n"
        "[1:30] Aristotle preferred empiricism.\n"
    )

    def test_window_around_30s(self) -> None:
        result = extract_transcript_window(
            self.SAMPLE_TRANSCRIPT,
            30.0,
            window_seconds=15.0,
        )
        assert "[0:20]" in result
        assert "[0:30]" in result
        assert "[0:40]" in result
        # Lines outside window should not be included
        assert "[0:00]" not in result
        assert "[1:00]" not in result

    def test_window_at_start(self) -> None:
        result = extract_transcript_window(
            self.SAMPLE_TRANSCRIPT,
            5.0,
            window_seconds=10.0,
        )
        assert "[0:00]" in result
        assert "[0:10]" in result
        # Should not include lines past 15s
        assert "[0:20]" not in result

    def test_window_at_end(self) -> None:
        result = extract_transcript_window(
            self.SAMPLE_TRANSCRIPT,
            90.0,
            window_seconds=15.0,
        )
        assert "[1:20]" in result
        assert "[1:30]" in result

    def test_wide_window_gets_everything(self) -> None:
        result = extract_transcript_window(
            self.SAMPLE_TRANSCRIPT,
            45.0,
            window_seconds=60.0,
        )
        assert "[0:00]" in result
        assert "[1:30]" in result

    def test_empty_transcript(self) -> None:
        result = extract_transcript_window("", 30.0, window_seconds=15.0)
        assert result == ""

    def test_no_timestamps_in_text(self) -> None:
        result = extract_transcript_window(
            "Just plain text without timestamps",
            30.0,
            window_seconds=15.0,
        )
        assert result == ""

    def test_deep_window_30s(self) -> None:
        """Deep captures use ±30s window."""
        result = extract_transcript_window(
            self.SAMPLE_TRANSCRIPT,
            30.0,
            window_seconds=30.0,
        )
        assert "[0:00]" in result
        assert "[0:30]" in result
        assert "[0:50]" in result
        assert "[1:00]" in result


class TestParseXmlTranscript:
    def test_basic_xml_parsing(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8" ?>
        <transcript>
            <text start="0" dur="5">Hello world</text>
            <text start="5" dur="3">How are you</text>
        </transcript>"""
        result = _parse_xml_transcript(xml)
        assert result is not None
        assert "Hello world" in result
        assert "How are you" in result

    def test_srv3_p_elements(self) -> None:
        xml = """<timedtext>
            <body>
                <p t="0" d="5000">First segment</p>
                <p t="5000" d="3000">Second segment</p>
            </body>
        </timedtext>"""
        result = _parse_xml_transcript(xml)
        assert result is not None
        assert "First segment" in result
        assert "[0:00]" in result
        assert "[0:05]" in result

    def test_empty_xml(self) -> None:
        result = _parse_xml_transcript("<transcript></transcript>")
        assert result is None

    def test_invalid_xml(self) -> None:
        result = _parse_xml_transcript("not xml at all")
        assert result is None

    def test_html_entities_cleaned(self) -> None:
        xml = """<transcript>
            <text start="0" dur="5">He said &amp; she said &quot;hi&quot;</text>
        </transcript>"""
        result = _parse_xml_transcript(xml)
        assert result is not None
        assert '&' in result
        assert '"' in result
