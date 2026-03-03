"""Visual capture processing.

For visual_quick and visual_deep capture modes:
1. Analyzes the screenshot with Claude's vision capability
2. Generates a visual description and context synthesis
3. Combines visual analysis with transcript analysis from Quick/Deep
4. Stores the screenshot reference with the chunk metadata
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.config import Settings

log = structlog.get_logger(__name__)


async def analyze_screenshot(
    screenshot_base64: str,
    *,
    transcript_window: str,
    source_title: str,
    annotation: str | None,
    settings: Settings,
) -> dict[str, Any]:
    """Analyze a screenshot using Claude's vision capability.

    Generates a visual description and synthesizes it with the
    spoken content from the transcript window.

    Args:
        screenshot_base64: Base64-encoded screenshot image.
        transcript_window: Transcript text around the capture timestamp.
        source_title: Title of the source.
        annotation: Optional user annotation.
        settings: Application settings.

    Returns:
        dict with visual_description, context_synthesis, and visual_elements.
    """
    import anthropic

    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    system_prompt = (
        "You are analyzing a screenshot captured during a video alongside "
        "the spoken transcript at that moment. Provide analysis that connects "
        "what is shown visually with what is being said. Respond with JSON:\n"
        '- "visual_description": string — detailed description of what is shown\n'
        '- "visual_elements": array of strings — key visual elements identified '
        "(e.g., slides, diagrams, text on screen, speakers, locations)\n"
        '- "context_synthesis": string — how the visual content relates to and '
        "enriches the spoken content\n"
        '- "notable_text": string | null — any significant text visible in the image\n\n'
        "Respond ONLY with the JSON object."
    )

    # Build user message with image and transcript
    user_content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _detect_media_type(screenshot_base64),
                "data": screenshot_base64,
            },
        },
        {
            "type": "text",
            "text": (
                f"Source: {source_title}\n"
                + (f"User note: {annotation}\n" if annotation else "")
                + f"\nSpoken content at this moment:\n{transcript_window}"
            ),
        },
    ]

    try:
        response = await client.messages.create(
            model=settings.llm.ingestion_model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text if response.content else ""
        if not response_text:
            return {
                "visual_description": "",
                "visual_elements": [],
                "context_synthesis": "",
                "notable_text": None,
            }

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        return json.loads(cleaned)

    except Exception as exc:
        log.error("visual_analysis_failed", error=str(exc))
        return {
            "visual_description": "",
            "visual_elements": [],
            "context_synthesis": "",
            "notable_text": None,
        }


def save_screenshot(
    screenshot_base64: str,
    *,
    capture_id: str,
    screenshots_dir: str = "data/screenshots",
) -> str | None:
    """Save a base64-encoded screenshot to the filesystem.

    Args:
        screenshot_base64: Base64-encoded image data.
        capture_id: Unique capture ID for the filename.
        screenshots_dir: Directory to save screenshots.

    Returns:
        File path of the saved screenshot, or None on failure.
    """
    try:
        data = base64.b64decode(screenshot_base64)
    except Exception as exc:
        log.error("screenshot_decode_failed", error=str(exc))
        return None

    ext = _detect_extension(screenshot_base64)
    dir_path = Path(screenshots_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"{capture_id}.{ext}"
    file_path.write_bytes(data)

    log.info("screenshot_saved", path=str(file_path), size=len(data))
    return str(file_path)


def _detect_media_type(base64_data: str) -> str:
    """Detect image media type from base64 data prefix or magic bytes."""
    # Check for data URI prefix
    if base64_data.startswith("data:"):
        parts = base64_data.split(";", 1)
        return parts[0].replace("data:", "")

    # Try to detect from decoded bytes
    try:
        raw = base64.b64decode(base64_data[:32])
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if raw[:2] == b"\xff\xd8":
            return "image/jpeg"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass

    # Default to PNG
    return "image/png"


def _detect_extension(base64_data: str) -> str:
    """Detect file extension from base64 image data."""
    media_type = _detect_media_type(base64_data)
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }
    return ext_map.get(media_type, "png")
