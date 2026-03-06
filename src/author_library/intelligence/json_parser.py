"""Robust JSON extraction from LLM responses.

LLMs frequently wrap JSON in markdown code fences, add prose before/after
the JSON block, or produce slightly malformed JSON (trailing commas, etc.).
This module provides a single extraction function used by all intelligence
modules to avoid duplicating fragile parsing logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Handles common LLM response patterns:
    1. Pure JSON (ideal)
    2. JSON wrapped in ```json ... ``` code fences
    3. JSON embedded in prose (extracts outermost { ... })
    4. Trailing commas before closing braces/brackets

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    stripped = text.strip()

    # Attempt 1: Direct parse (ideal case)
    try:
        return json.loads(stripped)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown code fences
    if "```" in stripped:
        # Extract content between first ``` and last ```
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

    # Attempt 3: Extract outermost { ... } block
    brace_start = stripped.find("{")
    if brace_start >= 0:
        # Find matching closing brace by counting depth
        depth = 0
        in_string = False
        escape_next = False
        brace_end = -1

        for i in range(brace_start, len(stripped)):
            ch = stripped[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break

        if brace_end > brace_start:
            candidate = stripped[brace_start:brace_end]
            try:
                return json.loads(candidate)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                # Attempt 4: Fix trailing commas and retry
                fixed = _fix_trailing_commas(candidate)
                try:
                    return json.loads(fixed)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

    # All attempts failed — raise with context
    raise json.JSONDecodeError(
        f"Could not extract JSON from LLM response ({len(stripped)} chars)",
        stripped[:200],
        0,
    )


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r",\s*([}\]])", r"\1", text)
