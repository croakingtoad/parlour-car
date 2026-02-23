"""M3: Surfacing response format.

Structures surfacing results into a response format suitable for the
Parlour Sidebar and MCP tool output. Results are grouped by confidence
level with each item carrying:
  - Note title and source attribution
  - Brief excerpt
  - Confidence label
  - Connection type
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from author_library.surfacing.confidence import (
    ConfidenceLevel,
    ScoredConnection,
    classify_batch,
)

from author_library.surfacing.related_content import RelatedItem


@dataclass(frozen=True, slots=True)
class FormattedSurfacingItem:
    """A single surfacing result formatted for presentation."""

    chunk_id: str
    work_id: str
    title: str
    source: str
    excerpt: str
    confidence_level: str
    confidence_label: str
    connection_type: str
    source_class: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SurfacingResponse:
    """Complete surfacing response grouped by confidence level."""

    context_chunk_id: str
    context_work_id: str
    high_confidence: list[FormattedSurfacingItem]
    medium_confidence: list[FormattedSurfacingItem]
    low_confidence: list[FormattedSurfacingItem]
    strategies_used: list[str]
    total_results: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for JSON output."""
        return {
            "context": {
                "chunk_id": self.context_chunk_id,
                "work_id": self.context_work_id,
            },
            "results": {
                "high": [_item_to_dict(i) for i in self.high_confidence],
                "medium": [_item_to_dict(i) for i in self.medium_confidence],
                "low": [_item_to_dict(i) for i in self.low_confidence],
            },
            "total_results": self.total_results,
            "strategies_used": self.strategies_used,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def format_surfacing_results(
    items: list[RelatedItem],
    *,
    context_chunk_id: str = "",
    context_work_id: str = "",
    strategies_used: list[str] | None = None,
    max_per_level: int | None = None,
) -> SurfacingResponse:
    """Format related items into a structured surfacing response.

    Classifies confidence, groups by level, and formats each item
    with presentation metadata.

    Args:
        items: Related items to format.
        context_chunk_id: The chunk that triggered the surfacing.
        context_work_id: The work containing the context chunk.
        strategies_used: Which search strategies produced these results.
        max_per_level: Optional cap on results per confidence level.

    Returns:
        SurfacingResponse grouped by confidence level.
    """
    scored = classify_batch(items)

    high: list[FormattedSurfacingItem] = []
    medium: list[FormattedSurfacingItem] = []
    low: list[FormattedSurfacingItem] = []

    for sc in scored:
        formatted = _format_item(sc)

        if sc.confidence_level == ConfidenceLevel.HIGH:
            if max_per_level is None or len(high) < max_per_level:
                high.append(formatted)
        elif sc.confidence_level == ConfidenceLevel.MEDIUM:
            if max_per_level is None or len(medium) < max_per_level:
                medium.append(formatted)
        else:
            if max_per_level is None or len(low) < max_per_level:
                low.append(formatted)

    return SurfacingResponse(
        context_chunk_id=context_chunk_id,
        context_work_id=context_work_id,
        high_confidence=high,
        medium_confidence=medium,
        low_confidence=low,
        strategies_used=strategies_used or [],
        total_results=len(high) + len(medium) + len(low),
    )


def _format_item(sc: ScoredConnection) -> FormattedSurfacingItem:
    """Format a scored connection into a presentation item."""
    item = sc.item

    # Build title from metadata or work_id
    title = item.metadata.get("work_title", "")
    if not title:
        title = item.work_id.replace("--", " — ").replace("-", " ").title()

    # Build source attribution
    source = _build_source_attribution(item)

    # Build excerpt — truncate long text
    excerpt = item.text[:300].strip()
    if len(item.text) > 300:
        excerpt += "..."

    return FormattedSurfacingItem(
        chunk_id=item.chunk_id,
        work_id=item.work_id,
        title=title,
        source=source,
        excerpt=excerpt,
        confidence_level=sc.confidence_level.value,
        confidence_label=sc.label,
        connection_type=item.connection_type.value,
        source_class=item.source_class,
        metadata={
            "relevance_score": round(sc.raw_score, 4),
            "link_type": item.metadata.get("link_type", ""),
            "theme": item.metadata.get("theme", ""),
            "evidence": item.metadata.get("evidence", ""),
            "date_created": item.metadata.get("date_created", ""),
        },
    )


def _build_source_attribution(item: RelatedItem) -> str:
    """Build a human-readable source attribution string."""
    if item.source_class == "personal":
        date = item.metadata.get("date_created", "")
        if date:
            return f"Your reflection from {date}"
        return "Your personal reflection"

    author = item.metadata.get("author", "")
    title = item.metadata.get("work_title", "")

    if author and title:
        return f"{author}, {title}"
    if title:
        return title
    if author:
        return author
    return item.work_id


def _item_to_dict(item: FormattedSurfacingItem) -> dict[str, Any]:
    """Serialize a formatted item to dict."""
    return {
        "chunk_id": item.chunk_id,
        "work_id": item.work_id,
        "title": item.title,
        "source": item.source,
        "excerpt": item.excerpt,
        "confidence_level": item.confidence_level,
        "confidence_label": item.confidence_label,
        "connection_type": item.connection_type,
        "source_class": item.source_class,
        "metadata": item.metadata,
    }
