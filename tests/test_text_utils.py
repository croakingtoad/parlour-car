"""Tests for text_utils sanitization — UTF-8 safety for PostgreSQL storage."""

from __future__ import annotations

from author_library.text_utils import sanitize_text


class TestSanitizeText:
    """Verify sanitize_text handles all known encoding corruption patterns."""

    def test_passthrough_clean_ascii(self) -> None:
        assert sanitize_text("Hello, world!") == "Hello, world!"

    def test_passthrough_valid_unicode(self) -> None:
        text = "Caf\u00e9 na\u00efvet\u00e9"
        assert sanitize_text(text) == text

    def test_smart_quotes_preserved(self) -> None:
        """Smart quotes (\u201c \u201d \u2018 \u2019) are valid Unicode and should pass through."""
        text = "\u201cHello,\u201d she said, \u2018it\u2019s fine.\u2019"
        result = sanitize_text(text)
        assert "\u201c" in result
        assert "\u201d" in result
        assert "\u2018" in result
        assert "\u2019" in result

    def test_em_dash_preserved(self) -> None:
        text = "word\u2014another word"
        assert sanitize_text(text) == text

    def test_en_dash_preserved(self) -> None:
        text = "pages 1\u20135"
        assert sanitize_text(text) == text

    def test_ellipsis_preserved(self) -> None:
        text = "and so\u2026"
        assert sanitize_text(text) == text

    def test_cp1252_smart_quotes_fixed(self) -> None:
        """C1 control codes 0x93/0x94 (CP-1252 smart quotes) get mapped to Unicode."""
        text = "\x93Hello\x94"  # CP-1252 left/right double quotes
        result = sanitize_text(text)
        assert result == "\u201cHello\u201d"

    def test_cp1252_single_quotes_fixed(self) -> None:
        text = "\x91it\x92s"
        result = sanitize_text(text)
        assert result == "\u2018it\u2019s"

    def test_cp1252_em_dash_fixed(self) -> None:
        text = "word\x97another"
        result = sanitize_text(text)
        assert result == "word\u2014another"

    def test_cp1252_en_dash_fixed(self) -> None:
        text = "1\x962"
        result = sanitize_text(text)
        assert result == "1\u20132"

    def test_cp1252_ellipsis_fixed(self) -> None:
        text = "and so\x85"
        result = sanitize_text(text)
        assert result == "and so\u2026"

    def test_null_bytes_stripped(self) -> None:
        text = "hello\x00world"
        assert sanitize_text(text) == "helloworld"

    def test_control_chars_stripped(self) -> None:
        text = "hello\x01\x02\x03world"
        assert sanitize_text(text) == "helloworld"

    def test_whitespace_preserved(self) -> None:
        """Newlines, carriage returns, and tabs should survive."""
        text = "line1\nline2\tindented\rcarriage"
        assert sanitize_text(text) == text

    def test_del_char_stripped(self) -> None:
        text = "hello\x7fworld"
        assert sanitize_text(text) == "helloworld"

    def test_unmapped_c1_controls_stripped(self) -> None:
        """C1 codes without CP-1252 mappings (e.g. 0x81, 0x8D) should be stripped."""
        text = "hello\x81\x8d\x8fworld"
        assert sanitize_text(text) == "helloworld"

    def test_nfc_normalization(self) -> None:
        """Decomposed sequences should be composed (e + combining accent → é)."""
        decomposed = "e\u0301"  # e + combining acute accent
        result = sanitize_text(decomposed)
        assert result == "\u00e9"  # composed é

    def test_empty_string(self) -> None:
        assert sanitize_text("") == ""

    def test_none_passthrough(self) -> None:
        """Empty/falsy values should pass through unchanged."""
        # sanitize_text("") returns "" — the function checks `if not text`
        assert sanitize_text("") == ""

    def test_idempotent(self) -> None:
        """Calling sanitize_text twice should produce the same result."""
        text = "\x93Hello\x94 world\x97test\x85"
        once = sanitize_text(text)
        twice = sanitize_text(once)
        assert once == twice

    def test_mixed_corruption_patterns(self) -> None:
        """Real-world pattern: CP-1252 quotes mixed with valid Unicode."""
        text = "\x93The imagination,\x94 Coleridge wrote, \x91is the living Power.\x92"
        result = sanitize_text(text)
        assert "\u201c" in result  # left double quote
        assert "\u201d" in result  # right double quote
        assert "\u2018" in result  # left single quote
        assert "\u2019" in result  # right single quote
        assert "Coleridge" in result
        assert "imagination" in result

    def test_utf8_round_trip_safety(self) -> None:
        """Result should encode/decode cleanly as UTF-8."""
        text = "\x93Smart quotes\x94 and \x97em dashes\x97"
        result = sanitize_text(text)
        # This should not raise
        encoded = result.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == result

    def test_annotation_like_text(self) -> None:
        """Simulate LLM annotation output with smart quotes that cause PostgreSQL failures."""
        annotation = (
            '[PRIMARY] From "Faith, Hope and Poetry" (2010) by Malcolm Guite.\n'
            "This meso covers: Coleridge\u2019s distinction between Primary and "
            "Secondary Imagination\u2014the \u201cliving Power\u201d of perception."
        )
        result = sanitize_text(annotation)
        # All valid Unicode should be preserved
        assert "\u2019" in result
        assert "\u2014" in result
        assert "\u201c" in result
        assert "\u201d" in result
        # Content integrity
        assert "Coleridge" in result
        assert "PRIMARY" in result
