"""Tests for session tracking (A8).

Covers:
- SessionManager auto-start on first capture
- SessionManager auto-end on inactivity timeout
- SessionManager auto-end on theme change + gap
- SessionManager._themes_changed logic
- SessionSettings configuration
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from author_library.config import SessionSettings
from author_library.storage.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session(
    *,
    session_id: UUID | None = None,
    user_id: str = "marty",
    updated_at: datetime | None = None,
    date_start: datetime | None = None,
    date_end: datetime | None = None,
) -> dict[str, Any]:
    """Build a session dict as returned by the repository."""
    now = datetime.now(timezone.utc)
    return {
        "id": session_id or uuid4(),
        "user_id": user_id,
        "title": None,
        "date_start": date_start or now,
        "date_end": date_end,
        "updated_at": updated_at or now,
        "duration_minutes": None,
    }


@pytest.fixture
def session_settings() -> SessionSettings:
    return SessionSettings(timeout_minutes=60, theme_change_gap_minutes=30)


@pytest.fixture
def session_repo() -> AsyncMock:
    """A mock SessionRepository for unit testing SessionManager logic."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=uuid4())
    repo.get = AsyncMock(return_value=None)
    repo.get_active = AsyncMock(return_value=None)
    repo.end_session = AsyncMock(return_value=True)
    repo.add_capture = AsyncMock(return_value=uuid4())
    repo.add_source = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def session_manager(
    session_repo: AsyncMock, session_settings: SessionSettings
) -> SessionManager:
    return SessionManager(session_repo=session_repo, settings=session_settings)


# ---------------------------------------------------------------------------
# Auto-start on first capture
# ---------------------------------------------------------------------------


class TestAutoStart:
    async def test_starts_session_when_none_active(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """First capture should auto-start a new session."""
        new_session_id = uuid4()
        new_session = _make_session(session_id=new_session_id)

        session_repo.get_active.return_value = None
        session_repo.create.return_value = new_session_id
        session_repo.get.return_value = new_session

        result = await session_manager.ensure_active_session("marty")
        assert result["id"] == new_session_id
        session_repo.create.assert_awaited_once()

    async def test_returns_existing_active_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """If active session exists and hasn't timed out, return it."""
        existing = _make_session()
        session_repo.get_active.return_value = existing

        result = await session_manager.ensure_active_session("marty")
        assert result["id"] == existing["id"]
        session_repo.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Auto-end on inactivity timeout (60min default)
# ---------------------------------------------------------------------------


class TestInactivityTimeout:
    async def test_timed_out_session_ends_and_creates_new(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """Session older than 60min should be ended, new one started."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=90)
        old_session = _make_session(updated_at=old_time)
        new_session_id = uuid4()
        new_session = _make_session(session_id=new_session_id)

        session_repo.get_active.return_value = old_session
        session_repo.create.return_value = new_session_id
        session_repo.get.return_value = new_session

        result = await session_manager.ensure_active_session("marty")
        session_repo.end_session.assert_awaited_once_with(old_session["id"])
        assert result["id"] == new_session_id

    async def test_session_at_59_minutes_not_timed_out(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """Session at 59min should NOT be timed out."""
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=59)
        session = _make_session(updated_at=recent_time)
        session_repo.get_active.return_value = session

        result = await session_manager.ensure_active_session("marty")
        assert result["id"] == session["id"]
        session_repo.end_session.assert_not_awaited()

    async def test_check_auto_end_timeout(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """check_auto_end should end session exceeding timeout."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=120)
        session = _make_session(updated_at=old_time)
        session_repo.get_active.return_value = session

        result = await session_manager.check_auto_end("marty")
        assert result is True
        session_repo.end_session.assert_awaited_once()

    async def test_check_auto_end_no_timeout(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """check_auto_end should not end a fresh session."""
        recent = _make_session()
        session_repo.get_active.return_value = recent

        result = await session_manager.check_auto_end("marty")
        assert result is False
        session_repo.end_session.assert_not_awaited()


# ---------------------------------------------------------------------------
# Auto-end on theme change + gap
# ---------------------------------------------------------------------------


class TestThemeChangeAutoEnd:
    async def test_theme_change_with_large_gap_ends_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """Theme change + >30min gap should end session."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        session = _make_session(updated_at=old_time)
        session_repo.get_active.return_value = session

        result = await session_manager.check_auto_end(
            "marty",
            current_themes=["sacrament", "liturgy"],
            previous_themes=["imagination", "coleridge"],
        )
        assert result is True
        session_repo.end_session.assert_awaited_once()

    async def test_theme_change_with_small_gap_keeps_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """Theme change but <30min gap should NOT end session."""
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        session = _make_session(updated_at=recent_time)
        session_repo.get_active.return_value = session

        result = await session_manager.check_auto_end(
            "marty",
            current_themes=["sacrament", "liturgy"],
            previous_themes=["imagination", "coleridge"],
        )
        assert result is False

    async def test_same_themes_no_end(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        """Same themes should not trigger auto-end even with gap."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        session = _make_session(updated_at=old_time)
        session_repo.get_active.return_value = session

        result = await session_manager.check_auto_end(
            "marty",
            current_themes=["imagination", "coleridge"],
            previous_themes=["imagination", "coleridge"],
        )
        assert result is False

    async def test_no_active_session_returns_false(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        session_repo.get_active.return_value = None
        result = await session_manager.check_auto_end("marty")
        assert result is False


# ---------------------------------------------------------------------------
# themes_changed logic
# ---------------------------------------------------------------------------


class TestThemesChanged:
    def test_completely_different_themes(self) -> None:
        assert SessionManager._themes_changed(
            ["sacrament", "liturgy"], ["imagination", "coleridge"]
        ) is True

    def test_identical_themes(self) -> None:
        assert SessionManager._themes_changed(
            ["imagination", "coleridge"], ["imagination", "coleridge"]
        ) is False

    def test_partial_overlap_above_threshold(self) -> None:
        # 2 out of 3 overlap → 67% → not changed
        assert SessionManager._themes_changed(
            ["imagination", "coleridge", "herbert"],
            ["imagination", "coleridge", "wordsworth"],
        ) is False

    def test_partial_overlap_below_threshold(self) -> None:
        # 1 out of 4 overlap → 25% → changed
        assert SessionManager._themes_changed(
            ["sacrament", "liturgy", "eucharist", "imagination"],
            ["coleridge", "wordsworth", "romanticism", "imagination"],
        ) is True

    def test_empty_current_themes(self) -> None:
        assert SessionManager._themes_changed([], ["imagination"]) is False

    def test_empty_previous_themes(self) -> None:
        assert SessionManager._themes_changed(["imagination"], []) is False

    def test_case_insensitive(self) -> None:
        assert SessionManager._themes_changed(
            ["IMAGINATION", "Coleridge"], ["imagination", "coleridge"]
        ) is False


# ---------------------------------------------------------------------------
# Manual session end
# ---------------------------------------------------------------------------


class TestManualEnd:
    async def test_end_active_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        session = _make_session()
        session_repo.get_active.return_value = session

        result = await session_manager.end_session("marty")
        assert result is True
        session_repo.end_session.assert_awaited_once()

    async def test_end_no_active_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        session_repo.get_active.return_value = None
        result = await session_manager.end_session("marty")
        assert result is False


# ---------------------------------------------------------------------------
# Record capture
# ---------------------------------------------------------------------------


class TestRecordCapture:
    async def test_record_capture_auto_starts_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        new_session_id = uuid4()
        new_session = _make_session(session_id=new_session_id)
        chunk_id = uuid4()

        session_repo.get_active.return_value = None
        session_repo.create.return_value = new_session_id
        session_repo.get.return_value = new_session

        result = await session_manager.record_capture(
            user_id="marty",
            chunk_id=chunk_id,
            work_id="guite--faith-hope-poetry",
            capture_order=1,
        )
        assert result["id"] == new_session_id
        session_repo.add_capture.assert_awaited_once()
        session_repo.add_source.assert_awaited_once()

    async def test_record_capture_uses_existing_session(
        self, session_manager: SessionManager, session_repo: AsyncMock
    ) -> None:
        existing = _make_session()
        chunk_id = uuid4()
        session_repo.get_active.return_value = existing

        result = await session_manager.record_capture(
            user_id="marty",
            chunk_id=chunk_id,
            work_id="guite--faith-hope-poetry",
            capture_order=1,
        )
        assert result["id"] == existing["id"]
        session_repo.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# SessionSettings configuration
# ---------------------------------------------------------------------------


class TestSessionSettings:
    def test_default_timeout(self) -> None:
        settings = SessionSettings()
        assert settings.timeout_minutes == 60

    def test_default_theme_gap(self) -> None:
        settings = SessionSettings()
        assert settings.theme_change_gap_minutes == 30

    def test_custom_timeout(self) -> None:
        settings = SessionSettings(timeout_minutes=120)
        assert settings.timeout_minutes == 120

    def test_configurable_timeout_used(self) -> None:
        """Custom timeout should be respected by SessionManager."""
        repo = AsyncMock()
        settings = SessionSettings(timeout_minutes=30)
        mgr = SessionManager(session_repo=repo, settings=settings)
        assert mgr._timeout_minutes == 30
