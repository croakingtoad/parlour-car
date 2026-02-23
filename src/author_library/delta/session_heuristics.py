"""P4: Session auto-detection heuristics.

Extends the existing SessionManager with richer auto-detection logic:
  - Same source, <60min gap → same session
  - Different source, <30min gap → same session (source switching)
  - >60min gap (configurable) → new session
  - Different themes + >30min gap → new session
  - Manual start/stop → always wins

Also adds source-switching detection to identify when the user
interleaves multiple sources within a single study session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.config import SessionSettings
    from author_library.storage.repositories import SessionRepository

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionDecision:
    """The heuristic decision about whether a capture belongs to the current session."""

    action: str  # "continue", "new_session", "end_and_start"
    reason: str
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Context for a capture event used in session decisions."""

    work_id: str
    themes: list[str]
    timestamp: datetime
    source_class: str = "primary"


class SessionHeuristics:
    """Enhanced session auto-detection heuristics.

    Implements the PRD §6.1 heuristic table with source-switching awareness.
    Works alongside the existing SessionManager, providing decision logic
    that the manager can use to determine session boundaries.
    """

    def __init__(
        self,
        *,
        timeout_minutes: int = 60,
        theme_gap_minutes: int = 30,
        source_switch_gap_minutes: int = 30,
    ) -> None:
        self._timeout_minutes = timeout_minutes
        self._theme_gap_minutes = theme_gap_minutes
        self._source_switch_gap_minutes = source_switch_gap_minutes

    def evaluate(
        self,
        current: CaptureContext,
        previous: CaptureContext | None,
    ) -> SessionDecision:
        """Evaluate whether a new capture belongs to the current session.

        Implements the heuristic table from PRD §6.1:
        1. Same source, <60min → continue session
        2. Different source, <30min → continue session (source switching)
        3. >60min gap → new session
        4. Different themes + >30min gap → new session

        Args:
            current: The current capture context.
            previous: The previous capture context (None if first capture).

        Returns:
            SessionDecision with action and reasoning.
        """
        if previous is None:
            return SessionDecision(
                action="new_session",
                reason="First capture — starting new session.",
                confidence=1.0,
            )

        gap_minutes = _compute_gap_minutes(previous.timestamp, current.timestamp)

        # Rule 3: >60min gap → always new session
        if gap_minutes > self._timeout_minutes:
            return SessionDecision(
                action="new_session",
                reason=f"Inactivity gap of {gap_minutes:.0f} minutes exceeds timeout ({self._timeout_minutes}min).",
                confidence=1.0,
                metadata={"gap_minutes": gap_minutes},
            )

        same_source = current.work_id == previous.work_id

        # Rule 1: Same source, within timeout → continue
        if same_source and gap_minutes <= self._timeout_minutes:
            return SessionDecision(
                action="continue",
                reason=f"Same source, {gap_minutes:.0f}min gap — continuing session.",
                confidence=0.95,
                metadata={"gap_minutes": gap_minutes, "same_source": True},
            )

        # Rule 4: Different themes + >30min gap → new session
        themes_changed = _themes_significantly_changed(
            current.themes, previous.themes,
        )
        if themes_changed and gap_minutes > self._theme_gap_minutes:
            return SessionDecision(
                action="new_session",
                reason=(
                    f"Theme change with {gap_minutes:.0f}min gap "
                    f"(threshold: {self._theme_gap_minutes}min)."
                ),
                confidence=0.85,
                metadata={
                    "gap_minutes": gap_minutes,
                    "themes_changed": True,
                    "current_themes": current.themes,
                    "previous_themes": previous.themes,
                },
            )

        # Rule 2: Different source, <30min → continue (source switching)
        if not same_source and gap_minutes <= self._source_switch_gap_minutes:
            return SessionDecision(
                action="continue",
                reason=(
                    f"Source switch within {gap_minutes:.0f}min — "
                    f"continuing session (user switching between sources)."
                ),
                confidence=0.8,
                metadata={
                    "gap_minutes": gap_minutes,
                    "source_switch": True,
                    "from_work": previous.work_id,
                    "to_work": current.work_id,
                },
            )

        # Fallback: different source, 30-60min gap, no theme change
        # This is ambiguous — lean toward continuing but with lower confidence
        if not same_source and gap_minutes <= self._timeout_minutes:
            return SessionDecision(
                action="continue",
                reason=(
                    f"Different source, {gap_minutes:.0f}min gap — "
                    f"possibly continuing session."
                ),
                confidence=0.6,
                metadata={"gap_minutes": gap_minutes, "ambiguous": True},
            )

        # Should not reach here, but default to new session
        return SessionDecision(
            action="new_session",
            reason="Unable to determine — defaulting to new session.",
            confidence=0.5,
        )

    def should_end_session(
        self,
        last_activity: datetime,
        *,
        current_time: datetime | None = None,
    ) -> bool:
        """Check if a session should be auto-ended based on inactivity.

        Args:
            last_activity: Timestamp of last activity.
            current_time: Current time (defaults to now).

        Returns:
            True if the session should be ended.
        """
        now = current_time or datetime.now(timezone.utc)
        gap = _compute_gap_minutes(last_activity, now)
        return gap > self._timeout_minutes


def _compute_gap_minutes(
    earlier: datetime,
    later: datetime,
) -> float:
    """Compute the gap in minutes between two timestamps."""
    if earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=timezone.utc)
    if later.tzinfo is None:
        later = later.replace(tzinfo=timezone.utc)
    return (later - earlier).total_seconds() / 60


def _themes_significantly_changed(
    current: list[str],
    previous: list[str],
) -> bool:
    """Determine if themes have significantly changed.

    Uses the same 50% overlap threshold as the existing SessionManager.
    """
    if not current or not previous:
        return False

    current_set = {t.lower() for t in current}
    previous_set = {t.lower() for t in previous}

    overlap = len(current_set & previous_set)
    total = max(len(current_set), len(previous_set))

    return overlap / total < 0.5 if total > 0 else False
