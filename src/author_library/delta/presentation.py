"""P3: Delta presentation — format delta analysis for notes and MCP output.

Structures delta analysis into human-readable summaries suitable for:
  - Session notes (brief delta summary with highlights)
  - Source notes (detailed "Re-reading Delta" section)
  - MCP tool output (structured JSON)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from author_library.delta.analyzer import (
    AttentionShift,
    DeltaAnalysis,
    ThemeDelta,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeltaPresentation:
    """Formatted delta presentation for different output contexts."""

    work_id: str
    work_title: str
    summary: str
    session_note_section: str
    source_note_section: str
    highlights: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/MCP output."""
        return {
            "work_id": self.work_id,
            "work_title": self.work_title,
            "summary": self.summary,
            "session_note_section": self.session_note_section,
            "source_note_section": self.source_note_section,
            "highlights": self.highlights,
        }


def format_delta(analysis: DeltaAnalysis) -> DeltaPresentation:
    """Format a delta analysis into presentation-ready output.

    Args:
        analysis: The delta analysis to format.

    Returns:
        DeltaPresentation with formatted sections for different contexts.
    """
    summary = _build_summary(analysis)
    session_section = _build_session_note_section(analysis)
    source_section = _build_source_note_section(analysis)
    highlights = _build_highlights(analysis)

    return DeltaPresentation(
        work_id=analysis.work_id,
        work_title=analysis.work_title,
        summary=summary,
        session_note_section=session_section,
        source_note_section=source_section,
        highlights=highlights,
    )


def _build_summary(analysis: DeltaAnalysis) -> str:
    """Build a brief one-paragraph summary of the delta."""
    parts: list[str] = []

    parts.append(
        f"Comparing pass {analysis.pass_a.pass_number} "
        f"({analysis.pass_a.capture_count} captures) with "
        f"pass {analysis.pass_b.pass_number} "
        f"({analysis.pass_b.capture_count} captures) "
        f"of {analysis.work_title}."
    )

    td = analysis.theme_delta
    if td.new_themes:
        parts.append(
            f"New themes noticed: {', '.join(td.new_themes)}."
        )
    if td.dropped_themes:
        parts.append(
            f"Themes no longer in focus: {', '.join(td.dropped_themes)}."
        )
    if td.persistent_themes:
        parts.append(
            f"Persistent attention: {', '.join(td.persistent_themes)}."
        )

    return " ".join(parts)


def _build_session_note_section(analysis: DeltaAnalysis) -> str:
    """Build the delta section for a session note.

    Session notes get a brief, scannable summary — not the full analysis.
    """
    lines: list[str] = []
    lines.append(f"## Re-reading Delta: {analysis.work_title}")
    lines.append("")
    lines.append(
        f"Pass {analysis.pass_b.pass_number} vs "
        f"pass {analysis.pass_a.pass_number} | "
        f"{analysis.pass_b.capture_count} captures "
        f"(was {analysis.pass_a.capture_count})"
    )
    lines.append("")

    # Highlights
    shifts = analysis.attention_shifts
    if shifts:
        lines.append("**Key shifts:**")
        for shift in shifts[:5]:  # Cap at 5 for session note brevity
            lines.append(f"- {shift.description}")
        lines.append("")

    # Theme changes
    td = analysis.theme_delta
    if td.new_themes:
        lines.append(
            f"**New attention:** {', '.join(td.new_themes)}"
        )
    if td.dropped_themes:
        lines.append(
            f"**Left behind:** {', '.join(td.dropped_themes)}"
        )

    return "\n".join(lines)


def _build_source_note_section(analysis: DeltaAnalysis) -> str:
    """Build the "Re-reading Delta" section for a source note.

    Source notes get the full, detailed analysis.
    """
    lines: list[str] = []
    lines.append("## Re-reading Delta")
    lines.append("")

    # Pass comparison header
    lines.append(
        f"### Pass {analysis.pass_a.pass_number} → "
        f"Pass {analysis.pass_b.pass_number}"
    )
    lines.append("")

    # Quantitative comparison
    lines.append("| Metric | "
                 f"Pass {analysis.pass_a.pass_number} | "
                 f"Pass {analysis.pass_b.pass_number} | Change |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| Captures | {analysis.pass_a.capture_count} | "
        f"{analysis.pass_b.capture_count} | "
        f"{analysis.capture_count_change:+d} |"
    )

    # Granularity distribution
    all_grans = set(analysis.pass_a.granularity_distribution) | set(
        analysis.pass_b.granularity_distribution
    )
    for gran in sorted(all_grans):
        a_count = analysis.pass_a.granularity_distribution.get(gran, 0)
        b_count = analysis.pass_b.granularity_distribution.get(gran, 0)
        change = b_count - a_count
        lines.append(
            f"| {gran.title()} | {a_count} | {b_count} | {change:+d} |"
        )
    lines.append("")

    # Theme delta
    td = analysis.theme_delta
    if td.new_themes or td.dropped_themes or td.persistent_themes:
        lines.append("### Theme Changes")
        lines.append("")
        if td.new_themes:
            lines.append(
                f"**New themes noticed:** {', '.join(td.new_themes)}"
            )
        if td.dropped_themes:
            lines.append(
                f"**No longer in focus:** {', '.join(td.dropped_themes)}"
            )
        if td.persistent_themes:
            lines.append(
                f"**Persistent attention:** {', '.join(td.persistent_themes)}"
            )
        lines.append("")

    # Attention shifts
    if analysis.attention_shifts:
        lines.append("### Attention Shifts")
        lines.append("")
        for shift in analysis.attention_shifts:
            lines.append(f"- {shift.description}")
        lines.append("")

    # Chapters
    a_chapters = set(analysis.pass_a.chapters_covered)
    b_chapters = set(analysis.pass_b.chapters_covered)
    new_chapters = b_chapters - a_chapters
    if new_chapters:
        lines.append("### New Chapters Explored")
        lines.append("")
        for ch in sorted(new_chapters):
            lines.append(f"- {ch}")
        lines.append("")

    return "\n".join(lines)


def _build_highlights(analysis: DeltaAnalysis) -> list[str]:
    """Build a list of key highlights from the delta analysis.

    These are brief, human-readable strings suitable for surfacing
    in a sidebar or MCP response.
    """
    highlights: list[str] = []

    td = analysis.theme_delta

    # Headline: capture count change
    diff = analysis.capture_count_change
    if diff > 0:
        highlights.append(
            f"{diff} more captures on pass {analysis.pass_b.pass_number}"
        )
    elif diff < 0:
        highlights.append(
            f"{abs(diff)} fewer captures on pass {analysis.pass_b.pass_number}"
        )

    # New themes
    for theme in td.new_themes[:3]:
        highlights.append(
            f"New attention to '{theme}'"
        )

    # Dropped themes
    for theme in td.dropped_themes[:3]:
        highlights.append(
            f"'{theme}' no longer in focus"
        )

    # Granularity shift
    gs = analysis.granularity_shift
    if gs.get("micro", 0) > 2:
        highlights.append("Shifted toward finer-grained captures")
    if gs.get("macro", 0) > 2:
        highlights.append("Shifted toward broader captures")

    return highlights
