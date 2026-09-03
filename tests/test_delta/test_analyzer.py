"""Tests for P2: Delta analyzer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from author_library.delta.analyzer import (
    AttentionShift,
    DeltaAnalysis,
    DeltaAnalyzer,
    PassCaptureSummary,
    ThemeDelta,
    _compute_granularity_shift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.works = MagicMock()
    storage.chunks = MagicMock()
    storage.neo4j = MagicMock()
    return storage


@pytest.fixture()
def analyzer(mock_storage):
    return DeltaAnalyzer(storage=mock_storage)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestPassCaptureSummary:
    def test_creation(self):
        summary = PassCaptureSummary(
            pass_number=1,
            capture_count=10,
            themes=["imagination", "prayer"],
            granularity_distribution={"meso": 7, "micro": 3},
            chapters_covered=["Ch 1", "Ch 3"],
            date_range=("2025-06-15", "2025-06-15"),
        )
        assert summary.pass_number == 1
        assert summary.capture_count == 10
        assert len(summary.themes) == 2


class TestThemeDelta:
    def test_creation(self):
        delta = ThemeDelta(
            new_themes=["liturgy"],
            dropped_themes=["coleridge"],
            persistent_themes=["imagination"],
        )
        assert delta.new_themes == ["liturgy"]
        assert delta.dropped_themes == ["coleridge"]
        assert delta.persistent_themes == ["imagination"]


class TestAttentionShift:
    def test_creation(self):
        shift = AttentionShift(
            description="New attention to 'liturgy' on pass 2",
            category="theme_new",
            evidence={"theme": "liturgy", "pass": 2},
        )
        assert shift.category == "theme_new"


class TestDeltaAnalysis:
    def test_to_dict(self):
        analysis = DeltaAnalysis(
            work_id="guite--faith-hope-poetry",
            work_title="Faith, Hope and Poetry",
            pass_a=PassCaptureSummary(
                pass_number=1, capture_count=10,
                themes=["imagination", "coleridge"],
                granularity_distribution={"meso": 7, "micro": 3},
                chapters_covered=["Ch 1", "Ch 3"],
            ),
            pass_b=PassCaptureSummary(
                pass_number=2, capture_count=8,
                themes=["imagination", "liturgy"],
                granularity_distribution={"meso": 3, "micro": 5},
                chapters_covered=["Ch 3", "Ch 5"],
            ),
            theme_delta=ThemeDelta(
                new_themes=["liturgy"],
                dropped_themes=["coleridge"],
                persistent_themes=["imagination"],
            ),
            attention_shifts=[
                AttentionShift(
                    description="New attention to 'liturgy'",
                    category="theme_new",
                ),
            ],
            capture_count_change=-2,
            granularity_shift={"meso": -4, "micro": 2},
        )
        d = analysis.to_dict()

        assert d["work_id"] == "guite--faith-hope-poetry"
        assert d["passes_compared"] == [1, 2]
        assert d["theme_delta"]["new_themes"] == ["liturgy"]
        assert len(d["attention_shifts"]) == 1
        assert d["capture_count_change"] == -2


# ---------------------------------------------------------------------------
# _compute_granularity_shift tests
# ---------------------------------------------------------------------------


class TestComputeGranularityShift:
    def test_basic_shift(self):
        dist_a = {"meso": 7, "micro": 3}
        dist_b = {"meso": 3, "micro": 5}
        shift = _compute_granularity_shift(dist_a, dist_b)
        assert shift["meso"] == -4
        assert shift["micro"] == 2

    def test_new_granularity(self):
        dist_a = {"meso": 5}
        dist_b = {"meso": 3, "macro": 2}
        shift = _compute_granularity_shift(dist_a, dist_b)
        assert shift["macro"] == 2
        assert shift["meso"] == -2

    def test_empty(self):
        assert _compute_granularity_shift({}, {}) == {}


# ---------------------------------------------------------------------------
# DeltaAnalyzer._compute_theme_delta tests
# ---------------------------------------------------------------------------


class TestComputeThemeDelta:
    def test_basic_delta(self):
        delta = DeltaAnalyzer._compute_theme_delta(
            ["imagination", "coleridge", "prayer"],
            ["imagination", "liturgy", "prayer"],
        )
        assert delta.new_themes == ["liturgy"]
        assert delta.dropped_themes == ["coleridge"]
        assert sorted(delta.persistent_themes) == ["imagination", "prayer"]

    def test_all_new(self):
        delta = DeltaAnalyzer._compute_theme_delta([], ["imagination", "liturgy"])
        assert delta.new_themes == ["imagination", "liturgy"]
        assert delta.dropped_themes == []
        assert delta.persistent_themes == []

    def test_all_dropped(self):
        delta = DeltaAnalyzer._compute_theme_delta(["imagination", "prayer"], [])
        assert delta.new_themes == []
        assert delta.dropped_themes == ["imagination", "prayer"]

    def test_no_change(self):
        delta = DeltaAnalyzer._compute_theme_delta(
            ["imagination", "prayer"],
            ["imagination", "prayer"],
        )
        assert delta.new_themes == []
        assert delta.dropped_themes == []
        assert sorted(delta.persistent_themes) == ["imagination", "prayer"]


# ---------------------------------------------------------------------------
# DeltaAnalyzer._identify_attention_shifts tests
# ---------------------------------------------------------------------------


class TestIdentifyAttentionShifts:
    def test_new_theme_shift(self):
        pass_a = PassCaptureSummary(
            pass_number=1, capture_count=10, themes=[],
            granularity_distribution={"meso": 10}, chapters_covered=[],
        )
        pass_b = PassCaptureSummary(
            pass_number=2, capture_count=10, themes=[],
            granularity_distribution={"meso": 10}, chapters_covered=[],
        )
        theme_delta = ThemeDelta(
            new_themes=["liturgy"], dropped_themes=[], persistent_themes=[],
        )
        shifts = DeltaAnalyzer._identify_attention_shifts(pass_a, pass_b, theme_delta)
        theme_shifts = [s for s in shifts if s.category == "theme_new"]
        assert len(theme_shifts) == 1
        assert "liturgy" in theme_shifts[0].description

    def test_granularity_shift(self):
        pass_a = PassCaptureSummary(
            pass_number=1, capture_count=10, themes=[],
            granularity_distribution={"micro": 3, "meso": 7}, chapters_covered=[],
        )
        pass_b = PassCaptureSummary(
            pass_number=2, capture_count=10, themes=[],
            granularity_distribution={"micro": 8, "meso": 2}, chapters_covered=[],
        )
        theme_delta = ThemeDelta(new_themes=[], dropped_themes=[], persistent_themes=[])
        shifts = DeltaAnalyzer._identify_attention_shifts(pass_a, pass_b, theme_delta)
        gran_shifts = [s for s in shifts if s.category == "granularity_shift"]
        assert len(gran_shifts) >= 1

    def test_focus_shift_new_chapters(self):
        pass_a = PassCaptureSummary(
            pass_number=1, capture_count=10, themes=[],
            granularity_distribution={"meso": 10}, chapters_covered=["Ch 1", "Ch 3"],
        )
        pass_b = PassCaptureSummary(
            pass_number=2, capture_count=10, themes=[],
            granularity_distribution={"meso": 10}, chapters_covered=["Ch 3", "Ch 5"],
        )
        theme_delta = ThemeDelta(new_themes=[], dropped_themes=[], persistent_themes=[])
        shifts = DeltaAnalyzer._identify_attention_shifts(pass_a, pass_b, theme_delta)
        focus_shifts = [s for s in shifts if s.category == "focus_shift"]
        assert any("Ch 5" in s.description for s in focus_shifts)

    def test_capture_volume_change(self):
        pass_a = PassCaptureSummary(
            pass_number=1, capture_count=5, themes=[],
            granularity_distribution={}, chapters_covered=[],
        )
        pass_b = PassCaptureSummary(
            pass_number=2, capture_count=15, themes=[],
            granularity_distribution={}, chapters_covered=[],
        )
        theme_delta = ThemeDelta(new_themes=[], dropped_themes=[], persistent_themes=[])
        shifts = DeltaAnalyzer._identify_attention_shifts(pass_a, pass_b, theme_delta)
        volume_shifts = [s for s in shifts if "more captures" in s.description]
        assert len(volume_shifts) == 1


# ---------------------------------------------------------------------------
# DeltaAnalyzer.analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.mark.asyncio()
    async def test_not_enough_passes(self, analyzer, mock_storage):
        """Returns None when work has fewer than 2 passes."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {"pass_number": 1, "capture_count": 10,
             "first_capture": "2025-06-15", "last_capture": "2025-06-15"},
        ])
        result = await analyzer.analyze("guite--faith-hope-poetry")
        assert result is None

    @pytest.mark.asyncio()
    async def test_same_pass(self, analyzer, mock_storage):
        """Returns None when pass_a == pass_b."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {"pass_number": 1, "capture_count": 10,
             "first_capture": "2025-06-15", "last_capture": "2025-06-15"},
            {"pass_number": 2, "capture_count": 8,
             "first_capture": "2026-01-20", "last_capture": "2026-01-20"},
        ])
        result = await analyzer.analyze("guite--faith-hope-poetry", pass_a=1, pass_b=1)
        assert result is None

    @pytest.mark.asyncio()
    async def test_basic_analysis(self, analyzer, mock_storage):
        """Full analysis with two passes."""
        # Pass history
        mock_storage.pg.fetch_all = AsyncMock(side_effect=[
            # get_pass_history
            [
                {"pass_number": 1, "capture_count": 10,
                 "first_capture": "2025-06-15", "last_capture": "2025-06-15"},
                {"pass_number": 2, "capture_count": 8,
                 "first_capture": "2026-01-20", "last_capture": "2026-01-20"},
            ],
            # get_captures_by_pass (pass 1)
            [
                {"chunk_id": "c-1", "work_id": "w-1", "text": "Capture 1",
                 "granularity": "meso", "source_class": "primary",
                 "chapter": "Ch 1", "section": None, "position": 0,
                 "date_created": "2025-06-15", "metadata": {}},
                {"chunk_id": "c-2", "work_id": "w-1", "text": "Capture 2",
                 "granularity": "meso", "source_class": "primary",
                 "chapter": "Ch 3", "section": None, "position": 1,
                 "date_created": "2025-06-15", "metadata": {}},
            ],
            # get_captures_by_pass (pass 2)
            [
                {"chunk_id": "c-3", "work_id": "w-1", "text": "Capture 3",
                 "granularity": "micro", "source_class": "primary",
                 "chapter": "Ch 3", "section": None, "position": 0,
                 "date_created": "2026-01-20", "metadata": {}},
            ],
        ])

        # Work info
        mock_storage.works.get = AsyncMock(return_value={
            "title": "Faith, Hope and Poetry", "author": "Test Guite",
        })

        # Graph themes
        mock_storage.neo4j.execute_read = AsyncMock(side_effect=[
            # Themes for pass 1 chunks
            [{"theme": "imagination"}, {"theme": "coleridge"}],
            # Themes for pass 2 chunks
            [{"theme": "imagination"}, {"theme": "liturgy"}],
        ])

        result = await analyzer.analyze("guite--faith-hope-poetry")

        assert result is not None
        assert result.work_title == "Faith, Hope and Poetry"
        assert result.pass_a.pass_number == 1
        assert result.pass_b.pass_number == 2
        assert result.pass_a.capture_count == 2
        assert result.pass_b.capture_count == 1
        assert result.capture_count_change == -1

    @pytest.mark.asyncio()
    async def test_analyze_latest(self, analyzer, mock_storage):
        """analyze_latest picks the two most recent passes."""
        mock_storage.pg.fetch_all = AsyncMock(side_effect=[
            # get_pass_history (called by analyze_latest)
            [
                {"pass_number": 1, "capture_count": 10,
                 "first_capture": "2025-01-01", "last_capture": "2025-01-01"},
                {"pass_number": 2, "capture_count": 8,
                 "first_capture": "2025-06-01", "last_capture": "2025-06-01"},
                {"pass_number": 3, "capture_count": 5,
                 "first_capture": "2026-01-01", "last_capture": "2026-01-01"},
            ],
            # get_pass_history (called again inside analyze)
            [
                {"pass_number": 1, "capture_count": 10,
                 "first_capture": "2025-01-01", "last_capture": "2025-01-01"},
                {"pass_number": 2, "capture_count": 8,
                 "first_capture": "2025-06-01", "last_capture": "2025-06-01"},
                {"pass_number": 3, "capture_count": 5,
                 "first_capture": "2026-01-01", "last_capture": "2026-01-01"},
            ],
            # get_captures_by_pass (pass 2)
            [{"chunk_id": "c-1", "work_id": "w-1", "text": "Pass 2 capture",
              "granularity": "meso", "source_class": "primary",
              "chapter": "Ch 1", "section": None, "position": 0,
              "date_created": "2025-06-01", "metadata": {}}],
            # get_captures_by_pass (pass 3)
            [{"chunk_id": "c-2", "work_id": "w-1", "text": "Pass 3 capture",
              "granularity": "micro", "source_class": "primary",
              "chapter": "Ch 1", "section": None, "position": 0,
              "date_created": "2026-01-01", "metadata": {}}],
        ])
        mock_storage.works.get = AsyncMock(return_value={"title": "T", "author": "A"})
        mock_storage.neo4j.execute_read = AsyncMock(side_effect=[[], []])

        result = await analyzer.analyze_latest("w-1")

        assert result is not None
        assert result.pass_a.pass_number == 2
        assert result.pass_b.pass_number == 3

    @pytest.mark.asyncio()
    async def test_analyze_no_captures(self, analyzer, mock_storage):
        """Returns None when both passes have no captures."""
        mock_storage.pg.fetch_all = AsyncMock(side_effect=[
            [
                {"pass_number": 1, "capture_count": 0,
                 "first_capture": None, "last_capture": None},
                {"pass_number": 2, "capture_count": 0,
                 "first_capture": None, "last_capture": None},
            ],
            [],  # pass 1 captures
            [],  # pass 2 captures
        ])
        mock_storage.works.get = AsyncMock(return_value={"title": "T"})
        mock_storage.neo4j.execute_read = AsyncMock(side_effect=[[], []])

        result = await analyzer.analyze("w-1")
        assert result is None

    @pytest.mark.asyncio()
    async def test_not_enough_passes_for_latest(self, analyzer, mock_storage):
        """analyze_latest returns None with only one pass."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {"pass_number": 1, "capture_count": 5,
             "first_capture": "2025-01-01", "last_capture": "2025-01-01"},
        ])
        result = await analyzer.analyze_latest("w-1")
        assert result is None
