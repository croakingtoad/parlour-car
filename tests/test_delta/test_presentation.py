"""Tests for P3: Delta presentation."""

from __future__ import annotations

import pytest

from author_library.delta.analyzer import (
    AttentionShift,
    DeltaAnalysis,
    PassCaptureSummary,
    ThemeDelta,
)
from author_library.delta.presentation import (
    DeltaPresentation,
    _build_highlights,
    _build_session_note_section,
    _build_source_note_section,
    _build_summary,
    format_delta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_analysis(
    *,
    new_themes: list[str] | None = None,
    dropped_themes: list[str] | None = None,
    persistent_themes: list[str] | None = None,
    capture_change: int = 0,
    attention_shifts: list[AttentionShift] | None = None,
) -> DeltaAnalysis:
    return DeltaAnalysis(
        work_id="guite--faith-hope-poetry",
        work_title="Faith, Hope and Poetry",
        pass_a=PassCaptureSummary(
            pass_number=1, capture_count=10,
            themes=["imagination", "coleridge"],
            granularity_distribution={"meso": 7, "micro": 3},
            chapters_covered=["Ch 1", "Ch 3"],
        ),
        pass_b=PassCaptureSummary(
            pass_number=2, capture_count=10 + capture_change,
            themes=["imagination", "liturgy"],
            granularity_distribution={"meso": 3, "micro": 7},
            chapters_covered=["Ch 3", "Ch 5"],
        ),
        theme_delta=ThemeDelta(
            new_themes=new_themes or ["liturgy"],
            dropped_themes=dropped_themes or ["coleridge"],
            persistent_themes=persistent_themes or ["imagination"],
        ),
        attention_shifts=attention_shifts or [
            AttentionShift(
                description="New attention to 'liturgy' on pass 2",
                category="theme_new",
            ),
            AttentionShift(
                description="More fine-grained captures on pass 2",
                category="granularity_shift",
            ),
        ],
        capture_count_change=capture_change,
        granularity_shift={"meso": -4, "micro": 4},
    )


# ---------------------------------------------------------------------------
# DeltaPresentation tests
# ---------------------------------------------------------------------------


class TestDeltaPresentation:
    def test_to_dict(self):
        pres = DeltaPresentation(
            work_id="w-1",
            work_title="Title",
            summary="Summary text",
            session_note_section="## Session",
            source_note_section="## Source",
            highlights=["Highlight 1"],
        )
        d = pres.to_dict()
        assert d["work_id"] == "w-1"
        assert d["summary"] == "Summary text"
        assert len(d["highlights"]) == 1


# ---------------------------------------------------------------------------
# _build_summary tests
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_basic_summary(self):
        analysis = _make_analysis()
        summary = _build_summary(analysis)
        assert "pass 1" in summary
        assert "pass 2" in summary
        assert "Faith, Hope and Poetry" in summary
        assert "10 captures" in summary

    def test_new_themes_in_summary(self):
        analysis = _make_analysis(new_themes=["liturgy", "prayer"])
        summary = _build_summary(analysis)
        assert "liturgy" in summary
        assert "prayer" in summary

    def test_dropped_themes_in_summary(self):
        analysis = _make_analysis(dropped_themes=["coleridge"])
        summary = _build_summary(analysis)
        assert "coleridge" in summary

    def test_persistent_themes_in_summary(self):
        analysis = _make_analysis(persistent_themes=["imagination"])
        summary = _build_summary(analysis)
        assert "imagination" in summary


# ---------------------------------------------------------------------------
# _build_session_note_section tests
# ---------------------------------------------------------------------------


class TestBuildSessionNoteSection:
    def test_contains_header(self):
        analysis = _make_analysis()
        section = _build_session_note_section(analysis)
        assert "## Re-reading Delta:" in section
        assert "Faith, Hope and Poetry" in section

    def test_contains_shifts(self):
        analysis = _make_analysis()
        section = _build_session_note_section(analysis)
        assert "Key shifts" in section
        assert "liturgy" in section

    def test_caps_at_five_shifts(self):
        shifts = [
            AttentionShift(description=f"Shift {i}", category="theme_new")
            for i in range(10)
        ]
        analysis = _make_analysis(attention_shifts=shifts)
        section = _build_session_note_section(analysis)
        # Count bullet points — should be at most 5
        bullet_count = section.count("- Shift")
        assert bullet_count <= 5


# ---------------------------------------------------------------------------
# _build_source_note_section tests
# ---------------------------------------------------------------------------


class TestBuildSourceNoteSection:
    def test_contains_header(self):
        analysis = _make_analysis()
        section = _build_source_note_section(analysis)
        assert "## Re-reading Delta" in section

    def test_contains_table(self):
        analysis = _make_analysis()
        section = _build_source_note_section(analysis)
        assert "| Captures |" in section
        assert "| Meso |" in section
        assert "| Micro |" in section

    def test_contains_theme_changes(self):
        analysis = _make_analysis(new_themes=["liturgy"])
        section = _build_source_note_section(analysis)
        assert "liturgy" in section
        assert "New themes noticed" in section

    def test_contains_attention_shifts(self):
        analysis = _make_analysis()
        section = _build_source_note_section(analysis)
        assert "Attention Shifts" in section

    def test_contains_new_chapters(self):
        analysis = _make_analysis()
        section = _build_source_note_section(analysis)
        assert "Ch 5" in section


# ---------------------------------------------------------------------------
# _build_highlights tests
# ---------------------------------------------------------------------------


class TestBuildHighlights:
    def test_capture_increase(self):
        analysis = _make_analysis(capture_change=5)
        highlights = _build_highlights(analysis)
        assert any("5 more captures" in h for h in highlights)

    def test_capture_decrease(self):
        analysis = _make_analysis(capture_change=-3)
        highlights = _build_highlights(analysis)
        assert any("3 fewer captures" in h for h in highlights)

    def test_new_themes(self):
        analysis = _make_analysis(new_themes=["liturgy", "prayer"])
        highlights = _build_highlights(analysis)
        assert any("liturgy" in h for h in highlights)
        assert any("prayer" in h for h in highlights)

    def test_dropped_themes(self):
        analysis = _make_analysis(dropped_themes=["coleridge"])
        highlights = _build_highlights(analysis)
        assert any("coleridge" in h for h in highlights)

    def test_granularity_shift(self):
        analysis = _make_analysis()
        analysis = DeltaAnalysis(
            work_id="w-1", work_title="T",
            pass_a=PassCaptureSummary(
                pass_number=1, capture_count=10, themes=[],
                granularity_distribution={}, chapters_covered=[],
            ),
            pass_b=PassCaptureSummary(
                pass_number=2, capture_count=10, themes=[],
                granularity_distribution={}, chapters_covered=[],
            ),
            theme_delta=ThemeDelta(
                new_themes=[], dropped_themes=[], persistent_themes=[],
            ),
            attention_shifts=[],
            capture_count_change=0,
            granularity_shift={"micro": 5},
        )
        highlights = _build_highlights(analysis)
        assert any("finer-grained" in h for h in highlights)


# ---------------------------------------------------------------------------
# format_delta integration tests
# ---------------------------------------------------------------------------


class TestFormatDelta:
    def test_full_format(self):
        analysis = _make_analysis(capture_change=-2)
        presentation = format_delta(analysis)

        assert presentation.work_id == "guite--faith-hope-poetry"
        assert presentation.work_title == "Faith, Hope and Poetry"
        assert len(presentation.summary) > 0
        assert len(presentation.session_note_section) > 0
        assert len(presentation.source_note_section) > 0
        assert len(presentation.highlights) > 0

    def test_to_dict(self):
        analysis = _make_analysis()
        presentation = format_delta(analysis)
        d = presentation.to_dict()

        assert "summary" in d
        assert "session_note_section" in d
        assert "source_note_section" in d
        assert "highlights" in d

    def test_empty_themes(self):
        analysis = _make_analysis(
            new_themes=[], dropped_themes=[], persistent_themes=[],
        )
        presentation = format_delta(analysis)
        assert len(presentation.summary) > 0
