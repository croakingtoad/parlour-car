"""Data models for capture events from the Chrome extension.

Defines the capture payload schema, processing modes, and result types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CaptureMode(str, Enum):
    """Capture mode from the Chrome extension."""

    QUICK = "quick"
    DEEP = "deep"
    VISUAL_QUICK = "visual_quick"
    VISUAL_DEEP = "visual_deep"


class CapturePayload(BaseModel):
    """Validated capture event from the Chrome extension."""

    source_url: str = Field(description="URL of the video/audio source")
    source_title: str = Field(description="Title of the source content")
    timestamp_seconds: float = Field(description="Capture timestamp in seconds")
    mode: CaptureMode = Field(description="Capture mode")
    screenshot_base64: str | None = Field(
        default=None,
        description="Base64-encoded screenshot (visual modes only)",
    )
    annotation: str | None = Field(
        default=None,
        description="User annotation text",
    )
    speaker_override: str | None = Field(
        default=None,
        description="Manual speaker attribution override",
    )
    transcript: str | None = Field(
        default=None,
        description="Full transcript extracted by the extension (preferred over server-side fetch)",
    )
    extension_version: str = Field(description="Chrome extension version")
    captured_at: datetime = Field(description="ISO 8601 capture timestamp")

    def is_visual(self) -> bool:
        """Whether this capture includes visual data."""
        return self.mode in (CaptureMode.VISUAL_QUICK, CaptureMode.VISUAL_DEEP)

    def is_deep(self) -> bool:
        """Whether this is a deep capture mode."""
        return self.mode in (CaptureMode.DEEP, CaptureMode.VISUAL_DEEP)


class CaptureResult:
    """Result of processing a single capture event."""

    __slots__ = (
        "capture_id",
        "chunk_id",
        "errors",
        "mode",
        "source_url",
        "work_id",
    )

    def __init__(
        self,
        *,
        capture_id: str,
        source_url: str,
        work_id: str,
        chunk_id: str | None = None,
        mode: str,
        errors: list[str] | None = None,
    ) -> None:
        self.capture_id = capture_id
        self.source_url = source_url
        self.work_id = work_id
        self.chunk_id = chunk_id
        self.mode = mode
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "source_url": self.source_url,
            "work_id": self.work_id,
            "chunk_id": self.chunk_id,
            "mode": self.mode,
            "errors": self.errors,
        }


class SourceOverview:
    """Overview generated for a source on first capture."""

    __slots__ = (
        "content_type",
        "source_url",
        "speakers",
        "structural_arc",
        "title",
        "topic_summary",
    )

    def __init__(
        self,
        *,
        source_url: str,
        title: str,
        speakers: list[str],
        content_type: str,
        topic_summary: str,
        structural_arc: str,
    ) -> None:
        self.source_url = source_url
        self.title = title
        self.speakers = speakers
        self.content_type = content_type
        self.topic_summary = topic_summary
        self.structural_arc = structural_arc

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "title": self.title,
            "speakers": self.speakers,
            "content_type": self.content_type,
            "topic_summary": self.topic_summary,
            "structural_arc": self.structural_arc,
        }
