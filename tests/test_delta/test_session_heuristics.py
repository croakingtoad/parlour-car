"""Tests for P4: Session auto-detection heuristics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from author_library.delta.session_heuristics import (
    CaptureContext,
    SessionDecision,
    SessionHeuristics,
    _compute_gap_minutes,
    _themes_significantly_changed,
)

# Session ID constants used across manual-override tests
_SID_A = "session-aaa-111"
_SID_B = "session-bbb-222"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSessionDecision:
    def test_creation(self):
        d = SessionDecision(
            action="continue",
            reason="Same source within timeout.",
            confidence=0.95,
        )
        assert d.action == "continue"
        assert d.confidence == 0.95

    def test_with_metadata(self):
        d = SessionDecision(
            action="new_session",
            reason="Inactivity timeout.",
            confidence=1.0,
            metadata={"gap_minutes": 120},
        )
        assert d.metadata["gap_minutes"] == 120

    def test_default_metadata(self):
        d = SessionDecision(action="continue", reason="R", confidence=0.5)
        assert d.metadata == {}


class TestCaptureContext:
    def test_creation(self):
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ctx = CaptureContext(
            work_id="guite--faith-hope-poetry",
            themes=["imagination", "coleridge"],
            timestamp=ts,
        )
        assert ctx.work_id == "guite--faith-hope-poetry"
        assert len(ctx.themes) == 2
        assert ctx.source_class == "primary"

    def test_custom_source_class(self):
        ts = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        ctx = CaptureContext(
            work_id="w-1", themes=[], timestamp=ts,
            source_class="secondary",
        )
        assert ctx.source_class == "secondary"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestComputeGapMinutes:
    def test_basic_gap(self):
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert _compute_gap_minutes(t1, t2) == 30.0

    def test_zero_gap(self):
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        assert _compute_gap_minutes(t1, t1) == 0.0

    def test_naive_datetimes_treated_as_utc(self):
        t1 = datetime(2026, 1, 15, 10, 0)
        t2 = datetime(2026, 1, 15, 11, 0)
        assert _compute_gap_minutes(t1, t2) == 60.0

    def test_mixed_naive_and_aware(self):
        t1 = datetime(2026, 1, 15, 10, 0)  # naive
        t2 = datetime(2026, 1, 15, 10, 45, tzinfo=timezone.utc)
        assert _compute_gap_minutes(t1, t2) == 45.0

    def test_fractional_minutes(self):
        t1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(seconds=90)  # 90 seconds = 1.5 minutes
        result = _compute_gap_minutes(t1, t2)
        assert result == 1.5


class TestThemesSignificantlyChanged:
    def test_no_overlap_significant(self):
        assert _themes_significantly_changed(
            ["liturgy", "prayer"], ["coleridge", "romanticism"],
        ) is True

    def test_full_overlap_not_significant(self):
        assert _themes_significantly_changed(
            ["imagination", "coleridge"], ["imagination", "coleridge"],
        ) is False

    def test_50_percent_overlap_not_significant(self):
        # 1 overlap out of 2 = 50%, threshold is < 50%
        assert _themes_significantly_changed(
            ["imagination", "liturgy"], ["imagination", "coleridge"],
        ) is False

    def test_below_50_percent_significant(self):
        # 1 overlap out of 3 = 33%, which is < 50%
        assert _themes_significantly_changed(
            ["imagination", "liturgy", "prayer"],
            ["imagination", "coleridge", "romanticism"],
        ) is True

    def test_empty_current_not_significant(self):
        assert _themes_significantly_changed([], ["imagination"]) is False

    def test_empty_previous_not_significant(self):
        assert _themes_significantly_changed(["imagination"], []) is False

    def test_both_empty_not_significant(self):
        assert _themes_significantly_changed([], []) is False

    def test_case_insensitive(self):
        assert _themes_significantly_changed(
            ["Imagination"], ["imagination"],
        ) is False


# ---------------------------------------------------------------------------
# SessionHeuristics.evaluate() tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.fixture()
    def heuristics(self):
        return SessionHeuristics(
            timeout_minutes=60,
            theme_gap_minutes=30,
            source_switch_gap_minutes=30,
        )

    def _ctx(
        self,
        *,
        work_id: str = "w-1",
        themes: list[str] | None = None,
        minutes_offset: int = 0,
    ) -> CaptureContext:
        base = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        return CaptureContext(
            work_id=work_id,
            themes=themes or ["imagination"],
            timestamp=base + timedelta(minutes=minutes_offset),
        )

    def test_first_capture_new_session(self, heuristics):
        decision = heuristics.evaluate(self._ctx(), None)
        assert decision.action == "new_session"
        assert decision.confidence == 1.0
        assert "First" in decision.reason

    def test_same_source_within_timeout_continue(self, heuristics):
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=30)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"
        assert decision.confidence == 0.95
        assert decision.metadata.get("same_source") is True

    def test_same_source_at_timeout_continue(self, heuristics):
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=60)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"

    def test_gap_exceeds_timeout_new_session(self, heuristics):
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=61)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "new_session"
        assert decision.confidence == 1.0
        assert "61" in decision.reason

    def test_large_gap_new_session(self, heuristics):
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=180)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "new_session"

    def test_different_source_within_switch_gap_continue(self, heuristics):
        prev = self._ctx(work_id="w-1", minutes_offset=0)
        curr = self._ctx(work_id="w-2", minutes_offset=20)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"
        assert decision.confidence == 0.8
        assert decision.metadata.get("source_switch") is True

    def test_different_source_at_switch_gap_continue(self, heuristics):
        prev = self._ctx(work_id="w-1", minutes_offset=0)
        curr = self._ctx(work_id="w-2", minutes_offset=30)
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"
        assert decision.confidence == 0.8

    def test_theme_change_beyond_theme_gap_new_session(self, heuristics):
        prev = self._ctx(
            work_id="w-1",
            themes=["coleridge", "romanticism"],
            minutes_offset=0,
        )
        curr = self._ctx(
            work_id="w-2",
            themes=["liturgy", "prayer"],
            minutes_offset=35,
        )
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "new_session"
        assert decision.confidence == 0.85
        assert decision.metadata.get("themes_changed") is True

    def test_theme_change_within_theme_gap_continue(self, heuristics):
        """Theme change but within 30min — source-switching takes precedence."""
        prev = self._ctx(
            work_id="w-1",
            themes=["coleridge", "romanticism"],
            minutes_offset=0,
        )
        curr = self._ctx(
            work_id="w-2",
            themes=["liturgy", "prayer"],
            minutes_offset=25,
        )
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"

    def test_ambiguous_different_source_30_to_60_continue(self, heuristics):
        """Different source, 30-60min gap, no theme change → ambiguous continue."""
        prev = self._ctx(
            work_id="w-1",
            themes=["imagination"],
            minutes_offset=0,
        )
        curr = self._ctx(
            work_id="w-2",
            themes=["imagination"],
            minutes_offset=45,
        )
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"
        assert decision.confidence == 0.6
        assert decision.metadata.get("ambiguous") is True

    def test_custom_timeout(self):
        h = SessionHeuristics(timeout_minutes=30)
        prev = CaptureContext(
            work_id="w-1", themes=["t"],
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
        curr = CaptureContext(
            work_id="w-1", themes=["t"],
            timestamp=datetime(2026, 1, 15, 10, 31, tzinfo=timezone.utc),
        )
        decision = h.evaluate(curr, prev)
        assert decision.action == "new_session"


# ---------------------------------------------------------------------------
# should_end_session tests
# ---------------------------------------------------------------------------


class TestShouldEndSession:
    @pytest.fixture()
    def heuristics(self):
        return SessionHeuristics(timeout_minutes=60)

    def test_within_timeout_no_end(self, heuristics):
        last = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 10, 59, tzinfo=timezone.utc)
        assert heuristics.should_end_session(last, current_time=now) is False

    def test_at_timeout_no_end(self, heuristics):
        last = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        assert heuristics.should_end_session(last, current_time=now) is False

    def test_exceeds_timeout_end(self, heuristics):
        last = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 1, tzinfo=timezone.utc)
        assert heuristics.should_end_session(last, current_time=now) is True

    def test_defaults_to_now(self, heuristics):
        # A timestamp from far in the past should always end
        long_ago = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert heuristics.should_end_session(long_ago) is True

    def test_custom_timeout(self):
        h = SessionHeuristics(timeout_minutes=15)
        last = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 10, 16, tzinfo=timezone.utc)
        assert h.should_end_session(last, current_time=now) is True


# ---------------------------------------------------------------------------
# Manual override API tests
# ---------------------------------------------------------------------------


class TestManualOverrideAPI:
    """Test the manual_start / manual_stop / clear_override / query helpers."""

    @pytest.fixture()
    def heuristics(self):
        return SessionHeuristics(timeout_minutes=60)

    def test_manual_start_sets_override(self, heuristics):
        heuristics.manual_start(_SID_A)
        assert heuristics.has_override(_SID_A) is True
        assert heuristics.get_override(_SID_A) == "started"

    def test_manual_stop_sets_override(self, heuristics):
        heuristics.manual_stop(_SID_A)
        assert heuristics.has_override(_SID_A) is True
        assert heuristics.get_override(_SID_A) == "stopped"

    def test_manual_stop_overwrites_start(self, heuristics):
        heuristics.manual_start(_SID_A)
        heuristics.manual_stop(_SID_A)
        assert heuristics.get_override(_SID_A) == "stopped"

    def test_manual_start_overwrites_stop(self, heuristics):
        heuristics.manual_stop(_SID_A)
        heuristics.manual_start(_SID_A)
        assert heuristics.get_override(_SID_A) == "started"

    def test_clear_override_returns_true_when_present(self, heuristics):
        heuristics.manual_start(_SID_A)
        assert heuristics.clear_override(_SID_A) is True
        assert heuristics.has_override(_SID_A) is False

    def test_clear_override_returns_false_when_absent(self, heuristics):
        assert heuristics.clear_override(_SID_A) is False

    def test_has_override_false_when_empty(self, heuristics):
        assert heuristics.has_override(_SID_A) is False

    def test_get_override_none_when_absent(self, heuristics):
        assert heuristics.get_override(_SID_A) is None

    def test_independent_sessions(self, heuristics):
        heuristics.manual_start(_SID_A)
        heuristics.manual_stop(_SID_B)
        assert heuristics.get_override(_SID_A) == "started"
        assert heuristics.get_override(_SID_B) == "stopped"


# ---------------------------------------------------------------------------
# Manual override ← evaluate() integration
# ---------------------------------------------------------------------------


class TestEvaluateManualOverride:
    """Manual overrides must beat every heuristic rule in evaluate()."""

    @pytest.fixture()
    def heuristics(self):
        return SessionHeuristics(
            timeout_minutes=60,
            theme_gap_minutes=30,
            source_switch_gap_minutes=30,
        )

    def _ctx(
        self,
        *,
        work_id: str = "w-1",
        themes: list[str] | None = None,
        minutes_offset: int = 0,
    ) -> CaptureContext:
        base = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        return CaptureContext(
            work_id=work_id,
            themes=themes or ["imagination"],
            timestamp=base + timedelta(minutes=minutes_offset),
        )

    # --- manual_start overrides ---

    def test_manual_start_beats_timeout(self, heuristics):
        """Even with a 3-hour gap, manual_start keeps the session alive."""
        heuristics.manual_start(_SID_A)
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=180)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "continue"
        assert decision.confidence == 1.0
        assert decision.metadata["manual_override"] == "started"

    def test_manual_start_beats_theme_change(self, heuristics):
        """Manual start overrides theme-change + gap heuristic."""
        heuristics.manual_start(_SID_A)
        prev = self._ctx(
            work_id="w-1", themes=["coleridge", "romanticism"], minutes_offset=0,
        )
        curr = self._ctx(
            work_id="w-2", themes=["liturgy", "prayer"], minutes_offset=45,
        )
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "continue"
        assert decision.metadata["manual_override"] == "started"

    def test_manual_start_beats_first_capture(self, heuristics):
        """Even with no previous context, manual start says continue."""
        heuristics.manual_start(_SID_A)
        decision = heuristics.evaluate(self._ctx(), None, session_id=_SID_A)
        assert decision.action == "continue"
        assert decision.confidence == 1.0

    # --- manual_stop overrides ---

    def test_manual_stop_beats_same_source_within_timeout(self, heuristics):
        """Same source, 5 min gap would normally continue — stop overrides."""
        heuristics.manual_stop(_SID_A)
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=5)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "new_session"
        assert decision.confidence == 1.0
        assert decision.metadata["manual_override"] == "stopped"

    def test_manual_stop_beats_source_switch(self, heuristics):
        """Source switch within gap would normally continue — stop overrides."""
        heuristics.manual_stop(_SID_A)
        prev = self._ctx(work_id="w-1", minutes_offset=0)
        curr = self._ctx(work_id="w-2", minutes_offset=10)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "new_session"
        assert decision.metadata["manual_override"] == "stopped"

    # --- no override → falls through to heuristics ---

    def test_no_override_falls_through(self, heuristics):
        """Without an override, evaluate behaves normally."""
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=30)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "continue"
        assert "manual_override" not in decision.metadata

    def test_no_session_id_ignores_overrides(self, heuristics):
        """When session_id is not passed, overrides are not consulted."""
        heuristics.manual_stop(_SID_A)
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=5)
        # session_id omitted — should continue normally
        decision = heuristics.evaluate(curr, prev)
        assert decision.action == "continue"
        assert "manual_override" not in decision.metadata

    def test_cleared_override_falls_through(self, heuristics):
        """After clearing, evaluate returns to heuristic behavior."""
        heuristics.manual_stop(_SID_A)
        heuristics.clear_override(_SID_A)
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=5)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_A)
        assert decision.action == "continue"
        assert "manual_override" not in decision.metadata

    def test_override_scoped_to_session(self, heuristics):
        """Override on session A does not affect session B."""
        heuristics.manual_stop(_SID_A)
        prev = self._ctx(minutes_offset=0)
        curr = self._ctx(minutes_offset=5)
        decision = heuristics.evaluate(curr, prev, session_id=_SID_B)
        assert decision.action == "continue"
        assert "manual_override" not in decision.metadata


# ---------------------------------------------------------------------------
# Manual override ← should_end_session() integration
# ---------------------------------------------------------------------------


class TestShouldEndSessionManualOverride:
    """Manual overrides must beat the inactivity heuristic."""

    @pytest.fixture()
    def heuristics(self):
        return SessionHeuristics(timeout_minutes=60)

    def test_manual_start_prevents_auto_end(self, heuristics):
        """manual_start keeps the session alive even after 2 hours of inactivity."""
        heuristics.manual_start(_SID_A)
        long_ago = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        assert heuristics.should_end_session(
            long_ago, current_time=now, session_id=_SID_A,
        ) is False

    def test_manual_stop_forces_end_despite_recent_activity(self, heuristics):
        """manual_stop ends the session even with activity 1 minute ago."""
        heuristics.manual_stop(_SID_A)
        recent = datetime(2026, 1, 15, 10, 59, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        assert heuristics.should_end_session(
            recent, current_time=now, session_id=_SID_A,
        ) is True

    def test_no_override_uses_heuristic(self, heuristics):
        """Without override, should_end_session uses the timeout heuristic."""
        last = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert heuristics.should_end_session(
            last, current_time=now, session_id=_SID_A,
        ) is False

    def test_no_session_id_ignores_override(self, heuristics):
        """Without session_id, manual_stop has no effect."""
        heuristics.manual_stop(_SID_A)
        recent = datetime(2026, 1, 15, 10, 59, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        # No session_id — uses heuristic (1 min gap < 60 min timeout)
        assert heuristics.should_end_session(recent, current_time=now) is False

    def test_cleared_override_reverts_to_heuristic(self, heuristics):
        """After clearing, should_end_session returns to heuristic behavior."""
        heuristics.manual_start(_SID_A)
        heuristics.clear_override(_SID_A)
        long_ago = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        assert heuristics.should_end_session(
            long_ago, current_time=now, session_id=_SID_A,
        ) is True

    def test_override_scoped_to_session(self, heuristics):
        """Override on session A does not affect session B."""
        heuristics.manual_start(_SID_A)
        long_ago = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        # Session B has no override — heuristic says end
        assert heuristics.should_end_session(
            long_ago, current_time=now, session_id=_SID_B,
        ) is True
