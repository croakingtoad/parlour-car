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
