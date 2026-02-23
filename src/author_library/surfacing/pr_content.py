"""N2: PR content generator for surfaced connections.

Generates human-readable PR body explaining WHY connections exist.
Not just "A links to B" but "This passage in [source] engages with
[other source] because...". Uses confidence labels from Epic M2.

Depends on: N1 (ConnectionScanner), H5 (PR manager in parlour-notes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from author_library.surfacing.connection_scanner import ScanResult, StagedConnection

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


@dataclass
class PRContent:
    """Generated content for a connection PR."""

    title: str
    body: str
    affected_notes: list[str]
    pr_type: str
    labels: list[str]


def generate_pr_content(
    scan_result: ScanResult,
    *,
    work_title: str = "",
    work_author: str = "",
) -> PRContent:
    """Generate PR title and body from a connection scan result.

    Creates a human-readable summary explaining:
    - What was ingested (trigger)
    - What connections were found
    - WHY each connection exists (not just that it exists)

    Args:
        scan_result: Result from ConnectionScanner.
        work_title: Title of the newly ingested work.
        work_author: Author of the newly ingested work.

    Returns:
        PRContent with title, body, affected notes, and metadata.
    """
    total = scan_result.total_found
    high_count = len(scan_result.by_confidence.get("high", []))
    medium_count = len(scan_result.by_confidence.get("medium", []))
    low_count = len(scan_result.by_confidence.get("low", []))
    target_works = len(scan_result.by_target_work)

    # Build title
    source_label = work_title or scan_result.work_id
    if work_author:
        source_label = f"{work_author}: {source_label}"

    title = f"New connections found after ingesting {source_label}"
    if total == 1:
        title = f"New connection found after ingesting {source_label}"

    # Build body
    sections: list[str] = []

    # Summary section
    summary_lines = [
        f"## New Connections from {source_label}",
        "",
        f"After ingesting this work, **{total} new connection{'s' if total != 1 else ''}** "
        f"{'were' if total != 1 else 'was'} found across **{target_works} existing "
        f"work{'s' if target_works != 1 else ''}**.",
        "",
    ]
    if high_count or medium_count or low_count:
        summary_lines.append("| Confidence | Count |")
        summary_lines.append("|:-----------|------:|")
        if high_count:
            summary_lines.append(f"| Strong (directly engages) | {high_count} |")
        if medium_count:
            summary_lines.append(f"| Likely (appears to connect) | {medium_count} |")
        if low_count:
            summary_lines.append(f"| Possible (might be relevant) | {low_count} |")
        summary_lines.append("")

    sections.append("\n".join(summary_lines))

    # High confidence connections (always shown in detail)
    high_connections = scan_result.by_confidence.get("high", [])
    if high_connections:
        sections.append(_format_connection_group(
            "Strong Connections",
            high_connections,
        ))

    # Medium confidence connections
    medium_connections = scan_result.by_confidence.get("medium", [])
    if medium_connections:
        sections.append(_format_connection_group(
            "Likely Connections",
            medium_connections,
        ))

    # Low confidence connections (summarized)
    low_connections = scan_result.by_confidence.get("low", [])
    if low_connections:
        sections.append(_format_connection_summary(
            "Possible Connections",
            low_connections,
        ))

    body = "\n---\n\n".join(sections)

    # Collect affected notes (target work IDs)
    affected_notes = list(scan_result.by_target_work.keys())

    # Labels
    labels = ["parlour/new-connections"]
    if high_count:
        labels.append("parlour/high-confidence")

    return PRContent(
        title=title,
        body=body,
        affected_notes=affected_notes,
        pr_type="new_connection",
        labels=labels,
    )


def _format_connection_group(
    heading: str,
    connections: list[StagedConnection],
) -> str:
    """Format a group of connections with full detail."""
    lines = [f"### {heading}", ""]

    for i, conn in enumerate(connections, 1):
        lines.append(f"**{i}. {conn.confidence_label}** `{conn.target_work_id}`")
        lines.append("")

        if conn.explanation:
            lines.append(f"> {conn.explanation}")
            lines.append("")

        if conn.source_excerpt:
            excerpt = conn.source_excerpt[:200].strip()
            lines.append(f"**From new work:** \"{excerpt}...\"")
            lines.append("")

        if conn.target_excerpt:
            excerpt = conn.target_excerpt[:200].strip()
            lines.append(f"**Connects to:** \"{excerpt}...\"")
            lines.append("")

        lines.append(f"*Type: {conn.connection_type}*")
        lines.append("")

    return "\n".join(lines)


def _format_connection_summary(
    heading: str,
    connections: list[StagedConnection],
) -> str:
    """Format a group of connections as a compact summary."""
    lines = [f"### {heading}", ""]

    # Group by target work
    by_work: dict[str, list[StagedConnection]] = {}
    for conn in connections:
        by_work.setdefault(conn.target_work_id, []).append(conn)

    for work_id, work_conns in by_work.items():
        types = {c.connection_type for c in work_conns}
        type_str = ", ".join(sorted(types))
        lines.append(f"- **{work_id}**: {len(work_conns)} connection{'s' if len(work_conns) != 1 else ''} ({type_str})")

    lines.append("")
    return "\n".join(lines)
