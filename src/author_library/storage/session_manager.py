"""Session management with auto-start and auto-end heuristics.

A session represents a contiguous period of engagement with source material.
Sessions are automatically started on the first capture event and ended
when inactivity exceeds the configured timeout (default 60 minutes,
configurable via SESSION_TIMEOUT_MINUTES) or when a theme change is
detected with a gap > 30 minutes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.config import SessionSettings
    from author_library.storage.repositories import SessionRepository

log = structlog.get_logger(__name__)


class SessionManager:
    """Manages engagement sessions with auto-start/stop heuristics.

    Auto-start: first capture event starts a session.
    Auto-end: 60min inactivity (configurable) or theme change + >30min gap.
    """

    def __init__(
        self,
        *,
        session_repo: SessionRepository,
        settings: SessionSettings,
    ) -> None:
        self._repo = session_repo
        self._timeout_minutes = settings.timeout_minutes
        self._theme_gap_minutes = settings.theme_change_gap_minutes

    async def ensure_active_session(
        self,
        user_id: str,
        *,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Get or create an active session for the user.

        If an active session exists and hasn't timed out, returns it.
        If the session has timed out, ends it and starts a new one.
        If no active session exists, starts a new one.

        Args:
            user_id: The user identifier.
            title: Optional title for a new session.

        Returns:
            The active session dict.
        """
        active = await self._repo.get_active(user_id)

        if active is not None:
            # Check timeout
            if self._is_timed_out(active):
                session_id: UUID = active["id"]
                await self._repo.end_session(session_id)
                log.info(
                    "session_auto_ended_timeout",
                    session_id=str(session_id),
                    timeout_minutes=self._timeout_minutes,
                )
                # Start a new session
                return await self._start_session(user_id, title=title)
            return active

        # No active session — start one
        return await self._start_session(user_id, title=title)

    async def record_capture(
        self,
        *,
        user_id: str,
        chunk_id: UUID,
        work_id: str,
        capture_order: int,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Record a capture event, auto-starting a session if needed.

        Args:
            user_id: The user identifier.
            chunk_id: The chunk being captured.
            work_id: The source work.
            capture_order: Order of this capture in the session.
            title: Optional session title (used if starting a new session).

        Returns:
            The active session dict.
        """
        session = await self.ensure_active_session(user_id, title=title)
        session_id = session["id"]

        await self._repo.add_capture(session_id, chunk_id, capture_order)
        await self._repo.add_source(session_id, work_id)

        log.info(
            "session_capture_recorded",
            session_id=str(session_id),
            chunk_id=str(chunk_id),
            work_id=work_id,
        )
        return session

    async def check_auto_end(
        self,
        user_id: str,
        *,
        current_themes: list[str] | None = None,
        previous_themes: list[str] | None = None,
    ) -> bool:
        """Check if the active session should be auto-ended.

        Conditions for auto-end:
        1. Inactivity > timeout_minutes
        2. Theme change + gap > theme_change_gap_minutes

        Args:
            user_id: The user identifier.
            current_themes: Themes of current capture (optional).
            previous_themes: Themes of previous capture (optional).

        Returns:
            True if the session was auto-ended, False otherwise.
        """
        active = await self._repo.get_active(user_id)
        if active is None:
            return False

        session_id: UUID = active["id"]

        # Check inactivity timeout
        if self._is_timed_out(active):
            await self._repo.end_session(session_id)
            log.info(
                "session_auto_ended_timeout",
                session_id=str(session_id),
                timeout_minutes=self._timeout_minutes,
            )
            return True

        # Check theme change + gap
        if (
            current_themes is not None
            and previous_themes is not None
            and self._themes_changed(current_themes, previous_themes)
        ):
            gap = self._minutes_since_last_activity(active)
            if gap is not None and gap > self._theme_gap_minutes:
                await self._repo.end_session(session_id)
                log.info(
                    "session_auto_ended_theme_change",
                    session_id=str(session_id),
                    gap_minutes=gap,
                    threshold=self._theme_gap_minutes,
                )
                return True

        return False

    async def end_session(self, user_id: str) -> bool:
        """Manually end the active session for a user.

        Returns True if a session was ended, False if no active session.
        """
        active = await self._repo.get_active(user_id)
        if active is None:
            return False

        await self._repo.end_session(active["id"])
        log.info("session_manually_ended", session_id=str(active["id"]))
        return True

    async def _start_session(
        self,
        user_id: str,
        *,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Start a new session."""
        session_id = await self._repo.create({
            "user_id": user_id,
            "title": title,
        })
        session = await self._repo.get(session_id)
        if session is None:
            # Should not happen but handle gracefully
            return {"id": session_id, "user_id": user_id, "title": title}

        log.info(
            "session_started",
            session_id=str(session_id),
            user_id=user_id,
            title=title,
        )
        return session

    def _is_timed_out(self, session: dict[str, Any]) -> bool:
        """Check if a session has exceeded the inactivity timeout."""
        updated_at = session.get("updated_at") or session.get("date_start")
        if updated_at is None:
            return False

        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed_minutes = (now - updated_at).total_seconds() / 60
        return elapsed_minutes > self._timeout_minutes

    def _minutes_since_last_activity(self, session: dict[str, Any]) -> float | None:
        """Calculate minutes since last activity in the session."""
        updated_at = session.get("updated_at")
        if updated_at is None:
            return None

        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return (now - updated_at).total_seconds() / 60

    @staticmethod
    def _themes_changed(
        current_themes: list[str],
        previous_themes: list[str],
    ) -> bool:
        """Determine if themes have significantly changed between captures."""
        if not current_themes or not previous_themes:
            return False

        current_set = set(t.lower() for t in current_themes)
        previous_set = set(t.lower() for t in previous_themes)

        # Consider it a theme change if there's less than 50% overlap
        if not current_set or not previous_set:
            return False

        overlap = len(current_set & previous_set)
        total = max(len(current_set), len(previous_set))
        return overlap / total < 0.5
