"""Tests for the robust LLM JSON extraction utility."""

from __future__ import annotations

import json

import pytest

from author_library.intelligence.json_parser import extract_json


class TestExtractJson:
    """Tests for extract_json()."""

    def test_pure_json(self) -> None:
        result = extract_json('{"themes": []}')
        assert result == {"themes": []}

    def test_code_fenced_json(self) -> None:
        text = '```json\n{"themes": ["poetry"]}\n```'
        result = extract_json(text)
        assert result == {"themes": ["poetry"]}

    def test_code_fenced_no_language(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self) -> None:
        text = 'Here is the analysis:\n\n{"themes": ["faith"]}\n\nI hope this helps.'
        result = extract_json(text)
        assert result == {"themes": ["faith"]}

    def test_trailing_comma_before_brace(self) -> None:
        text = '{"themes": ["poetry", "imagination",]}'
        result = extract_json(text)
        assert result == {"themes": ["poetry", "imagination"]}

    def test_trailing_comma_before_bracket(self) -> None:
        text = '{"items": [1, 2, 3,]}'
        result = extract_json(text)
        assert result == {"items": [1, 2, 3]}

    def test_nested_objects(self) -> None:
        text = '{"a": {"b": {"c": 1}}}'
        result = extract_json(text)
        assert result == {"a": {"b": {"c": 1}}}

    def test_braces_in_strings(self) -> None:
        text = '{"text": "the author says {this} and {that}"}'
        result = extract_json(text)
        assert result == {"text": "the author says {this} and {that}"}

    def test_escaped_quotes_in_strings(self) -> None:
        text = '{"text": "He said \\"hello\\" to them"}'
        result = extract_json(text)
        assert result == {"text": 'He said "hello" to them'}

    def test_multiline_code_fence_with_prose(self) -> None:
        text = (
            "Based on my analysis, here are the themes:\n\n"
            "```json\n"
            '{\n  "themes": [\n    {"theme": "Imagination", "stance": "Central"}\n  ]\n}\n'
            "```\n\n"
            "These themes recur throughout the corpus."
        )
        result = extract_json(text)
        assert result["themes"][0]["theme"] == "Imagination"

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            extract_json("This is just plain text with no JSON at all.")

    def test_raises_on_empty(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            extract_json("")

    def test_whitespace_around_json(self) -> None:
        result = extract_json("  \n  {\"key\": 1}  \n  ")
        assert result == {"key": 1}

    def test_real_world_thematic_response(self) -> None:
        """Simulate the kind of response that caused the original failure."""
        text = (
            '{"themes": ['
            '{"theme": "Sacramental Imagination", '
            '"author_stance": "Poetry reveals the sacred in the ordinary, '
            "functioning as what Coleridge called 'the repetition in the finite mind "
            "of the eternal act of creation'\", "
            '"related_themes": ["Transfiguration", "Incarnation"]}'
            "]}"
        )
        result = extract_json(text)
        assert len(result["themes"]) == 1
        assert result["themes"][0]["theme"] == "Sacramental Imagination"
