"""P2: Delta analyzer — compare captures across engagement passes.

Given a work that the user has re-engaged with, compares what was captured
on each pass to identify:
  - New themes noticed on later passes
  - Themes dropped from attention on later passes
  - Passages captured on both passes (persistent attention)
  - Shifts in granularity (e.g., first pass captured macro, second pass micro)
  - Quantitative capture comparison (count, distribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.delta.reengagement import ReengagementDetector, ReengagementInfo

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PassCaptureSummary:
    """Summary of captures for a single pass."""

    pass_number: int
    capture_count: int
    themes: list[str]
    granularity_distribution: dict[str, int]
    chapters_covered: list[str]
    date_range: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ThemeDelta:
    """Theme-level changes between passes."""

    new_themes: list[str]
    dropped_themes: list[str]
    persistent_themes: list[str]


@dataclass(frozen=True, slots=True)
class AttentionShift:
    """A specific shift in attention between passes."""

    description: str
    category: str  # "theme_new", "theme_dropped", "granularity_shift", "focus_shift"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeltaAnalysis:
    """Complete delta analysis between engagement passes."""

    work_id: str
    work_title: str
    pass_a: PassCaptureSummary
    pass_b: PassCaptureSummary
    theme_delta: ThemeDelta
    attention_shifts: list[AttentionShift]
    capture_count_change: int
    granularity_shift: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/MCP output."""
        return {
            "work_id": self.work_id,
            "work_title": self.work_title,
            "passes_compared": [self.pass_a.pass_number, self.pass_b.pass_number],
            "pass_a": {
                "pass_number": self.pass_a.pass_number,
                "capture_count": self.pass_a.capture_count,
                "themes": self.pass_a.themes,
                "granularity": self.pass_a.granularity_distribution,
                "chapters": self.pass_a.chapters_covered,
            },
            "pass_b": {
                "pass_number": self.pass_b.pass_number,
                "capture_count": self.pass_b.capture_count,
                "themes": self.pass_b.themes,
                "granularity": self.pass_b.granularity_distribution,
                "chapters": self.pass_b.chapters_covered,
            },
            "theme_delta": {
                "new_themes": self.theme_delta.new_themes,
                "dropped_themes": self.theme_delta.dropped_themes,
                "persistent_themes": self.theme_delta.persistent_themes,
            },
            "attention_shifts": [
                {
                    "description": shift.description,
                    "category": shift.category,
                    "evidence": shift.evidence,
                }
                for shift in self.attention_shifts
            ],
            "capture_count_change": self.capture_count_change,
            "granularity_shift": self.granularity_shift,
        }


class DeltaAnalyzer:
    """Compares captures across engagement passes to identify intellectual shifts.

    Uses ReengagementDetector's pass history to load captures per pass,
    then analyzes theme differences, granularity changes, and attention
    shifts between passes.
    """

    def __init__(self, *, storage: StorageManager) -> None:
        self._storage = storage
        self._detector = ReengagementDetector(storage=storage)

    async def analyze(
        self,
        work_id: str,
        *,
        pass_a: int = 1,
        pass_b: int | None = None,
    ) -> DeltaAnalysis | None:
        """Analyze the delta between two engagement passes.

        Args:
            work_id: The work to analyze.
            pass_a: First pass number (default: 1).
            pass_b: Second pass number (default: latest).

        Returns:
            DeltaAnalysis or None if the work doesn't have enough passes.
        """
        # Get pass history to validate
        history = await self._detector.get_pass_history(work_id)
        if len(history) < 2:
            return None

        # Determine pass_b
        if pass_b is None:
            pass_b = max(h["pass_number"] for h in history)

        if pass_a == pass_b:
            return None

        # Get work metadata
        work_info = await self._storage.works.get(work_id)
        work_title = work_info.get("title", work_id) if work_info else work_id

        # Load captures for each pass
        captures_a = await self._detector.get_captures_by_pass(work_id, pass_a)
        captures_b = await self._detector.get_captures_by_pass(work_id, pass_b)

        if not captures_a and not captures_b:
            return None

        # Build pass summaries
        summary_a = await self._build_pass_summary(pass_a, captures_a)
        summary_b = await self._build_pass_summary(pass_b, captures_b)

        # Compute theme delta
        theme_delta = self._compute_theme_delta(summary_a.themes, summary_b.themes)

        # Identify attention shifts
        attention_shifts = self._identify_attention_shifts(
            summary_a, summary_b, theme_delta,
        )

        # Granularity shift: change in distribution between passes
        granularity_shift = _compute_granularity_shift(
            summary_a.granularity_distribution,
            summary_b.granularity_distribution,
        )

        return DeltaAnalysis(
            work_id=work_id,
            work_title=work_title,
            pass_a=summary_a,
            pass_b=summary_b,
            theme_delta=theme_delta,
            attention_shifts=attention_shifts,
            capture_count_change=summary_b.capture_count - summary_a.capture_count,
            granularity_shift=granularity_shift,
        )

    async def analyze_latest(self, work_id: str) -> DeltaAnalysis | None:
        """Analyze delta between the two most recent passes.

        Convenience method for the common case of comparing the latest
        re-engagement with the previous one.

        Args:
            work_id: The work to analyze.

        Returns:
            DeltaAnalysis or None if not enough passes.
        """
        history = await self._detector.get_pass_history(work_id)
        if len(history) < 2:
            return None

        sorted_passes = sorted(h["pass_number"] for h in history)
        return await self.analyze(
            work_id,
            pass_a=sorted_passes[-2],
            pass_b=sorted_passes[-1],
        )

    async def _build_pass_summary(
        self,
        pass_number: int,
        captures: list[dict[str, Any]],
    ) -> PassCaptureSummary:
        """Build a summary of captures for a single pass."""
        themes: list[str] = []
        granularity_counts: dict[str, int] = {}
        chapters: list[str] = []
        dates: list[str] = []

        for cap in captures:
            # Count granularity
            gran = cap.get("granularity", "unknown")
            granularity_counts[gran] = granularity_counts.get(gran, 0) + 1

            # Collect chapters
            chapter = cap.get("chapter")
            if chapter and chapter not in chapters:
                chapters.append(chapter)

            # Collect dates
            date = cap.get("date_created", "")
            if date:
                dates.append(date)

        # Get themes from graph for these chunks
        chunk_ids = [cap["chunk_id"] for cap in captures if cap.get("chunk_id")]
        themes = await self._get_themes_for_chunks(chunk_ids)

        date_range = None
        if dates:
            date_range = (min(dates), max(dates))

        return PassCaptureSummary(
            pass_number=pass_number,
            capture_count=len(captures),
            themes=themes,
            granularity_distribution=granularity_counts,
            chapters_covered=chapters,
            date_range=date_range,
        )

    async def _get_themes_for_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Get distinct themes from graph for a set of chunks."""
        if not chunk_ids:
            return []

        try:
            records = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE c.chunk_id IN $chunk_ids
                RETURN DISTINCT t.canonical_name AS theme
                ORDER BY theme""",
                {"chunk_ids": chunk_ids},
            )
            return [r["theme"] for r in records if r.get("theme")]
        except Exception:
            log.debug("get_themes_for_chunks_failed", count=len(chunk_ids))
            return []

    @staticmethod
    def _compute_theme_delta(
        themes_a: list[str],
        themes_b: list[str],
    ) -> ThemeDelta:
        """Compute theme-level changes between two passes."""
        set_a = set(themes_a)
        set_b = set(themes_b)

        return ThemeDelta(
            new_themes=sorted(set_b - set_a),
            dropped_themes=sorted(set_a - set_b),
            persistent_themes=sorted(set_a & set_b),
        )

    @staticmethod
    def _identify_attention_shifts(
        pass_a: PassCaptureSummary,
        pass_b: PassCaptureSummary,
        theme_delta: ThemeDelta,
    ) -> list[AttentionShift]:
        """Identify specific attention shifts between passes."""
        shifts: list[AttentionShift] = []

        # New themes
        for theme in theme_delta.new_themes:
            shifts.append(AttentionShift(
                description=f"New attention to '{theme}' on pass {pass_b.pass_number}",
                category="theme_new",
                evidence={"theme": theme, "pass": pass_b.pass_number},
            ))

        # Dropped themes
        for theme in theme_delta.dropped_themes:
            shifts.append(AttentionShift(
                description=(
                    f"Theme '{theme}' noticed on pass {pass_a.pass_number} "
                    f"but not on pass {pass_b.pass_number}"
                ),
                category="theme_dropped",
                evidence={"theme": theme, "pass": pass_a.pass_number},
            ))

        # Granularity shifts
        gran_a = pass_a.granularity_distribution
        gran_b = pass_b.granularity_distribution

        # Check for shift toward finer granularity
        micro_a = gran_a.get("micro", 0)
        micro_b = gran_b.get("micro", 0)
        macro_a = gran_a.get("macro", 0)
        macro_b = gran_b.get("macro", 0)

        if micro_b > micro_a and micro_a > 0:
            shifts.append(AttentionShift(
                description=(
                    f"More fine-grained captures on pass {pass_b.pass_number} "
                    f"(micro: {micro_a} -> {micro_b})"
                ),
                category="granularity_shift",
                evidence={"micro_a": micro_a, "micro_b": micro_b},
            ))

        if macro_b > macro_a and macro_a > 0:
            shifts.append(AttentionShift(
                description=(
                    f"More broad captures on pass {pass_b.pass_number} "
                    f"(macro: {macro_a} -> {macro_b})"
                ),
                category="granularity_shift",
                evidence={"macro_a": macro_a, "macro_b": macro_b},
            ))

        # Focus shift: different chapters covered
        chapters_a = set(pass_a.chapters_covered)
        chapters_b = set(pass_b.chapters_covered)
        new_chapters = chapters_b - chapters_a
        if new_chapters:
            shifts.append(AttentionShift(
                description=(
                    f"New chapters explored on pass {pass_b.pass_number}: "
                    f"{', '.join(sorted(new_chapters))}"
                ),
                category="focus_shift",
                evidence={"new_chapters": sorted(new_chapters)},
            ))

        # Capture volume change
        count_diff = pass_b.capture_count - pass_a.capture_count
        if abs(count_diff) > 2:
            direction = "more" if count_diff > 0 else "fewer"
            shifts.append(AttentionShift(
                description=(
                    f"{abs(count_diff)} {direction} captures on pass "
                    f"{pass_b.pass_number} vs pass {pass_a.pass_number}"
                ),
                category="focus_shift",
                evidence={
                    "count_a": pass_a.capture_count,
                    "count_b": pass_b.capture_count,
                },
            ))

        return shifts


def _compute_granularity_shift(
    dist_a: dict[str, int],
    dist_b: dict[str, int],
) -> dict[str, int]:
    """Compute the change in granularity distribution between passes."""
    all_keys = set(dist_a) | set(dist_b)
    return {k: dist_b.get(k, 0) - dist_a.get(k, 0) for k in sorted(all_keys)}
