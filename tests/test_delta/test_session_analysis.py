"""Tests for P5: Session analysis — themes, threads, pick-up-here suggestions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from author_library.delta.session_analysis import (
    PickUpHere,
    SessionAnalysisResult,
    SessionAnalyzer,
    SessionSource,
    SessionThread,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSessionSource:
    def test_creation(self):
        s = SessionSource(
            work_id="guite--faith-hope-poetry",
            title="Faith, Hope and Poetry",
            capture_count=5,
            themes=["imagination", "coleridge"],
        )
        assert s.work_id == "guite--faith-hope-poetry"
        assert s.capture_count == 5
        assert len(s.themes) == 2


class TestSessionThread:
    def test_creation(self):
        t = SessionThread(
            theme="imagination",
            previous_session_id="session-001",
            previous_session_date="2026-01-10T09:00:00",
            connection_description="Both sessions explored 'imagination'.",
        )
        assert t.theme == "imagination"
        assert t.previous_session_id == "session-001"


class TestPickUpHere:
    def test_creation(self):
        p = PickUpHere(
            work_id="w-1",
            work_title="Faith, Hope and Poetry",
            chapter="Ch 5",
            suggestion="Continue with Faith, Hope and Poetry from Ch 5",
            themes_in_progress=["imagination", "liturgy"],
        )
        assert p.work_id == "w-1"
        assert p.chapter == "Ch 5"
        assert len(p.themes_in_progress) == 2


class TestSessionAnalysisResult:
    def test_to_dict(self):
        result = SessionAnalysisResult(
            session_id="sess-1",
            date_start="2026-01-15T10:00:00",
            date_end="2026-01-15T11:30:00",
            duration_minutes=90.0,
            sources=[
                SessionSource(
                    work_id="w-1", title="Work 1",
                    capture_count=5, themes=["t1"],
                ),
            ],
            themes=["t1", "t2"],
            theme_counts={"t1": 3, "t2": 2},
            threads=[
                SessionThread(
                    theme="t1",
                    previous_session_id="sess-0",
                    previous_session_date="2026-01-10",
                    connection_description="Both explored 't1'.",
                ),
            ],
            pick_up_here=[
                PickUpHere(
                    work_id="w-1",
                    work_title="Work 1",
                    chapter="Ch 3",
                    suggestion="Continue with Work 1 from Ch 3",
                    themes_in_progress=["t1"],
                ),
            ],
            capture_count=5,
        )
        d = result.to_dict()

        assert d["session_id"] == "sess-1"
        assert d["duration_minutes"] == 90.0
        assert len(d["sources"]) == 1
        assert d["sources"][0]["work_id"] == "w-1"
        assert d["themes"] == ["t1", "t2"]
        assert d["theme_counts"] == {"t1": 3, "t2": 2}
        assert len(d["threads"]) == 1
        assert d["threads"][0]["theme"] == "t1"
        assert len(d["pick_up_here"]) == 1
        assert d["pick_up_here"][0]["chapter"] == "Ch 3"
        assert d["capture_count"] == 5

    def test_to_dict_empty(self):
        result = SessionAnalysisResult(
            session_id="sess-empty",
            date_start="", date_end="",
            duration_minutes=0,
            sources=[], themes=[], theme_counts={},
            threads=[], pick_up_here=[],
            capture_count=0,
        )
        d = result.to_dict()
        assert d["sources"] == []
        assert d["capture_count"] == 0

    def test_duration_rounded(self):
        result = SessionAnalysisResult(
            session_id="s",
            date_start="", date_end="",
            duration_minutes=45.6789,
            sources=[], themes=[], theme_counts={},
            threads=[], pick_up_here=[],
            capture_count=0,
        )
        d = result.to_dict()
        assert d["duration_minutes"] == 45.7


# ---------------------------------------------------------------------------
# SessionAnalyzer._compute_duration tests
# ---------------------------------------------------------------------------


class TestComputeDuration:
    def test_valid_timestamps(self):
        duration = SessionAnalyzer._compute_duration({
            "date_start": "2026-01-15T10:00:00+00:00",
            "date_end": "2026-01-15T11:30:00+00:00",
        })
        assert duration == 90.0

    def test_missing_start(self):
        duration = SessionAnalyzer._compute_duration({
            "date_start": "",
            "date_end": "2026-01-15T11:30:00",
        })
        assert duration == 0.0

    def test_missing_end(self):
        duration = SessionAnalyzer._compute_duration({
            "date_start": "2026-01-15T10:00:00",
            "date_end": "",
        })
        assert duration == 0.0

    def test_missing_both(self):
        duration = SessionAnalyzer._compute_duration({})
        assert duration == 0.0

    def test_invalid_format(self):
        duration = SessionAnalyzer._compute_duration({
            "date_start": "not-a-date",
            "date_end": "also-not-a-date",
        })
        assert duration == 0.0

    def test_fractional_minutes(self):
        duration = SessionAnalyzer._compute_duration({
            "date_start": "2026-01-15T10:00:00",
            "date_end": "2026-01-15T10:15:30",
        })
        assert duration == 15.5


# ---------------------------------------------------------------------------
# SessionAnalyzer._generate_pick_up_suggestions tests
# ---------------------------------------------------------------------------


class TestGeneratePickUpSuggestions:
    def test_empty_captures_no_suggestions(self):
        result = SessionAnalyzer._generate_pick_up_suggestions([], [], [])
        assert result == []

    def test_single_capture_with_chapter(self):
        captures = [
            {
                "work_id": "w-1",
                "chapter": "Ch 5",
                "chunk_id": "c-1",
            },
        ]
        sources = [
            SessionSource(
                work_id="w-1", title="Faith, Hope and Poetry",
                capture_count=1, themes=["imagination"],
            ),
        ]
        result = SessionAnalyzer._generate_pick_up_suggestions(
            captures, sources, ["imagination"],
        )
        assert len(result) == 1
        assert result[0].work_id == "w-1"
        assert result[0].work_title == "Faith, Hope and Poetry"
        assert result[0].chapter == "Ch 5"
        assert "Faith, Hope and Poetry" in result[0].suggestion
        assert "Ch 5" in result[0].suggestion

    def test_last_capture_used(self):
        """Pick-up-here suggestion uses the LAST capture, not the first."""
        captures = [
            {"work_id": "w-1", "chapter": "Ch 1", "chunk_id": "c-1"},
            {"work_id": "w-1", "chapter": "Ch 3", "chunk_id": "c-2"},
            {"work_id": "w-2", "chapter": "Ch 7", "chunk_id": "c-3"},
        ]
        sources = [
            SessionSource(
                work_id="w-1", title="Work 1",
                capture_count=2, themes=[],
            ),
            SessionSource(
                work_id="w-2", title="Work 2",
                capture_count=1, themes=[],
            ),
        ]
        result = SessionAnalyzer._generate_pick_up_suggestions(
            captures, sources, [],
        )
        assert len(result) == 1
        assert result[0].work_id == "w-2"
        assert result[0].chapter == "Ch 7"

    def test_no_chapter_still_suggests(self):
        captures = [
            {"work_id": "w-1", "chunk_id": "c-1"},
        ]
        sources = [
            SessionSource(
                work_id="w-1", title="Work 1",
                capture_count=1, themes=[],
            ),
        ]
        result = SessionAnalyzer._generate_pick_up_suggestions(
            captures, sources, [],
        )
        assert len(result) == 1
        assert result[0].chapter == ""
        assert "Work 1" in result[0].suggestion

    def test_themes_in_progress_capped_at_three(self):
        captures = [{"work_id": "w-1", "chapter": "Ch 1", "chunk_id": "c-1"}]
        sources = [
            SessionSource(
                work_id="w-1", title="W", capture_count=1, themes=[],
            ),
        ]
        themes = ["t1", "t2", "t3", "t4", "t5"]
        result = SessionAnalyzer._generate_pick_up_suggestions(
            captures, sources, themes,
        )
        assert len(result[0].themes_in_progress) == 3

    def test_unknown_source_uses_work_id_as_title(self):
        """When source info doesn't match, work_id is used as title."""
        captures = [
            {"work_id": "unknown-work", "chapter": "Ch 1", "chunk_id": "c-1"},
        ]
        sources = [
            SessionSource(
                work_id="different-work", title="Different",
                capture_count=1, themes=[],
            ),
        ]
        result = SessionAnalyzer._generate_pick_up_suggestions(
            captures, sources, [],
        )
        assert result[0].work_title == "unknown-work"


# ---------------------------------------------------------------------------
# SessionAnalyzer.analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.fixture()
    def mock_storage(self):
        storage = MagicMock()
        storage.pg = MagicMock()
        storage.neo4j = MagicMock()
        storage.works = MagicMock()
        return storage

    @pytest.fixture()
    def analyzer(self, mock_storage):
        return SessionAnalyzer(storage=mock_storage)

    @pytest.mark.asyncio()
    async def test_session_not_found_returns_none(self, analyzer, mock_storage):
        mock_storage.pg.fetch_one = AsyncMock(return_value=None)

        result = await analyzer.analyze("nonexistent-session")
        assert result is None

    @pytest.mark.asyncio()
    async def test_no_captures_returns_empty_result(self, analyzer, mock_storage):
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "session_id": "sess-1",
            "user_id": "user-1",
            "date_start": "2026-01-15T10:00:00",
            "date_end": "2026-01-15T11:00:00",
            "title": "Study Session",
            "metadata": {},
        })
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        result = await analyzer.analyze("sess-1")
        assert result is not None
        assert result.session_id == "sess-1"
        assert result.capture_count == 0
        assert result.sources == []
        assert result.themes == []

    @pytest.mark.asyncio()
    async def test_full_analysis(self, analyzer, mock_storage):
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "session_id": "sess-1",
            "user_id": "user-1",
            "date_start": "2026-01-15T10:00:00+00:00",
            "date_end": "2026-01-15T11:30:00+00:00",
            "title": "Study Session",
            "metadata": {},
        })

        # Session captures
        mock_storage.pg.fetch_all = AsyncMock(side_effect=[
            # First call: _get_session_captures
            [
                {
                    "chunk_id": "c-1", "work_id": "w-1",
                    "text": "Imagination passage",
                    "granularity": "meso", "source_class": "primary",
                    "chapter": "Ch 3", "date_created": "2026-01-15T10:15:00",
                    "metadata": {},
                },
                {
                    "chunk_id": "c-2", "work_id": "w-1",
                    "text": "Coleridge reflection",
                    "granularity": "micro", "source_class": "primary",
                    "chapter": "Ch 3", "date_created": "2026-01-15T10:30:00",
                    "metadata": {},
                },
            ],
            # Subsequent calls: _find_threads queries (for each theme)
            [],  # First theme query
        ])

        # Works lookup
        mock_storage.works.get = AsyncMock(return_value={
            "title": "Faith, Hope and Poetry",
        })

        # Graph queries for themes
        mock_storage.neo4j.execute_read = AsyncMock(side_effect=[
            # _get_themes_for_chunks for work w-1
            [{"theme": "imagination"}, {"theme": "coleridge"}],
            # _analyze_themes for all chunk_ids
            [
                {"theme": "imagination", "count": 2},
                {"theme": "coleridge", "count": 1},
            ],
            # _get_themes_for_chunks called again (internal)
        ])

        result = await analyzer.analyze("sess-1")

        assert result is not None
        assert result.session_id == "sess-1"
        assert result.capture_count == 2
        assert result.duration_minutes == 90.0
        assert len(result.sources) >= 1
        assert result.sources[0].title == "Faith, Hope and Poetry"
        assert result.sources[0].capture_count == 2

    @pytest.mark.asyncio()
    async def test_analyze_with_missing_chunk_ids(self, analyzer, mock_storage):
        """Captures without chunk_id are handled gracefully."""
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "session_id": "sess-1",
            "user_id": "u-1",
            "date_start": "2026-01-15T10:00:00",
            "date_end": "2026-01-15T10:30:00",
            "title": "S",
            "metadata": {},
        })

        mock_storage.pg.fetch_all = AsyncMock(side_effect=[
            # Captures — some with empty chunk_id
            [
                {
                    "chunk_id": "", "work_id": "w-1",
                    "text": "Text", "granularity": "meso",
                    "source_class": "primary", "chapter": "Ch 1",
                    "date_created": "2026-01-15", "metadata": {},
                },
            ],
            # _find_threads
            [],
        ])

        mock_storage.works.get = AsyncMock(return_value={"title": "W1"})
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[])

        result = await analyzer.analyze("sess-1")
        assert result is not None
        assert result.capture_count == 1
        # Empty chunk_ids are filtered out for theme analysis
        assert result.themes == []
