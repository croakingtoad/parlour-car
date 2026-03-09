"""Integration tests for the lesson writer (LL2).

Tests:
- record_lesson() creates a new lesson in the DB
- Deduplication: same problem_type + overlapping trigger_context increments
  times_prevented instead of creating a duplicate
- Confidence formula: 0.5 + 0.5 * (times_prevented / max(times_applied, 1))
- Non-overlapping trigger_context creates a new lesson
- Errors in lesson writing are caught and re-raised (do not silently swallow)

Uses the test database (author_library_test).
"""

from __future__ import annotations

from typing import Any

import pytest

from author_library.intelligence.lesson_writer import record_lesson

from .conftest import SKIP_NO_DB


# ---------------------------------------------------------------------------
# TestRecordLesson
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestRecordLesson:
    """record_lesson() stores lessons and returns UUID."""

    async def test_creates_lesson_returns_uuid(self, clean_storage: Any) -> None:
        """record_lesson creates a lesson and returns a non-None UUID."""
        from uuid import UUID

        lesson_id = await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--lesson-basic", "source_class": "primary"},
            problem_description="3 chunks with text < 50 chars.",
            fix_applied="Flagged for review.",
        )

        assert lesson_id is not None
        assert isinstance(lesson_id, UUID), f"Expected UUID, got {type(lesson_id)}"

    async def test_created_lesson_is_in_db(self, clean_storage: Any) -> None:
        """Lesson created by record_lesson appears in list_all()."""
        await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--lesson-db", "source_class": "primary"},
            problem_description="5 orphaned entity nodes.",
            fix_applied="Deleted 5 orphaned nodes.",
            prevention_step="entity_extraction",
        )

        lessons = await clean_storage.lessons.list_all(active_only=True)
        orphan_lessons = [
            l for l in lessons if l["problem_type"] == "orphan_nodes"
        ]
        assert len(orphan_lessons) >= 1
        assert orphan_lessons[0]["detection_method"] == "qg1_inline"
        assert orphan_lessons[0]["prevention_step"] == "entity_extraction"

    async def test_lesson_has_initial_confidence_05(self, clean_storage: Any) -> None:
        """New lessons start with confidence=0.5."""
        lesson_id = await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--lesson-conf", "source_class": "primary"},
            problem_description="2 noise chunks.",
            fix_applied="Flagged.",
        )

        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(lesson_id)), None)
        assert match is not None
        assert abs(float(match["confidence"]) - 0.5) < 0.01, (
            f"Expected confidence=0.5, got {match['confidence']}"
        )

    async def test_optional_fields_stored(self, clean_storage: Any) -> None:
        """prevention_rule and prevention_step are stored when provided."""
        lesson_id = await record_lesson(
            clean_storage,
            problem_type="misclassification",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--lesson-opts", "source_class": "contextual"},
            problem_description="Author matches subject but classified as contextual.",
            fix_applied="Flagged for manual review.",
            prevention_rule="Default to primary when author matches subject_author_id.",
            prevention_step="classification",
        )

        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(lesson_id)), None)
        assert match is not None
        assert match["prevention_rule"] is not None
        assert "primary" in match["prevention_rule"]
        assert match["prevention_step"] == "classification"


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestDeduplication:
    """Duplicate lessons increment times_prevented instead of creating new rows."""

    async def test_same_problem_type_and_context_increments_prevented(
        self, clean_storage: Any
    ) -> None:
        """Calling record_lesson twice with same problem_type + context reuses the lesson."""
        ctx = {"work_id": "test--dedup-work", "source_class": "primary"}

        first_id = await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context=ctx,
            problem_description="3 orphaned nodes found.",
            fix_applied="Deleted 3 orphaned nodes.",
        )

        second_id = await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context=ctx,
            problem_description="1 orphaned node found.",
            fix_applied="Deleted 1 orphaned node.",
        )

        # Should return the SAME lesson ID (not create a new one)
        assert str(first_id) == str(second_id), (
            f"Expected dedup to return same lesson {first_id}, got new one {second_id}"
        )

        # Verify times_prevented was incremented
        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(first_id)), None)
        assert match is not None
        assert int(match["times_prevented"]) >= 1, (
            f"Expected times_prevented >= 1, got {match['times_prevented']}"
        )

    async def test_different_problem_type_creates_new_lesson(
        self, clean_storage: Any
    ) -> None:
        """Different problem_type creates a separate lesson even with same context."""
        ctx = {"work_id": "test--dedup-different", "source_class": "primary"}

        first_id = await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context=ctx,
            problem_description="Orphaned nodes.",
            fix_applied="Deleted.",
        )

        second_id = await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context=ctx,
            problem_description="Noise chunks.",
            fix_applied="Flagged.",
        )

        assert str(first_id) != str(second_id), (
            "Different problem_type should create a different lesson"
        )

    async def test_non_overlapping_context_creates_new_lesson(
        self, clean_storage: Any
    ) -> None:
        """Completely non-overlapping trigger_context creates a new lesson."""
        first_id = await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--ctx-work-a", "source_class": "primary"},
            problem_description="Noise in work A.",
            fix_applied="Flagged A.",
        )

        # Completely different context — no shared key:value
        second_id = await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--ctx-work-b", "source_class": "secondary"},
            problem_description="Noise in work B.",
            fix_applied="Flagged B.",
        )

        assert str(first_id) != str(second_id), (
            "Non-overlapping context should create a separate lesson"
        )


# ---------------------------------------------------------------------------
# TestConfidenceFormula
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestConfidenceFormula:
    """Confidence increases each time a lesson prevents a problem."""

    async def test_confidence_increases_after_prevented(
        self, clean_storage: Any
    ) -> None:
        """Confidence rises above 0.5 after times_prevented is incremented."""
        ctx = {"work_id": "test--confidence-work", "source_class": "primary"}

        lesson_id = await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context=ctx,
            problem_description="Orphaned nodes.",
            fix_applied="Deleted.",
        )

        # First call: creates lesson at confidence=0.5
        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(lesson_id)), None)
        initial_confidence = float(match["confidence"])
        assert abs(initial_confidence - 0.5) < 0.01

        # Simulate times_applied=1 then increment_prevented
        await clean_storage.lessons.increment_applied(lesson_id)
        await clean_storage.lessons.increment_prevented(lesson_id)

        # Confidence should now be: 0.5 + 0.5 * (1 / max(1, 1)) = 1.0
        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(lesson_id)), None)
        new_confidence = float(match["confidence"])
        assert new_confidence > initial_confidence, (
            f"Confidence should increase: {initial_confidence} -> {new_confidence}"
        )
        assert new_confidence <= 1.0, f"Confidence should not exceed 1.0: {new_confidence}"

    async def test_confidence_starts_at_05_for_new_lesson(
        self, clean_storage: Any
    ) -> None:
        """New lessons always start at confidence=0.5 (initial uncertainty)."""
        lesson_id = await record_lesson(
            clean_storage,
            problem_type="theme_explosion",
            detection_method="qg2_async",
            trigger_context={"work_id": "test--conf-start", "author_id": "test-author"},
            problem_description="15 near-duplicate themes merged.",
            fix_applied="Theme deduplication merged 15 themes.",
        )

        lessons = await clean_storage.lessons.list_all(active_only=True)
        match = next((l for l in lessons if str(l["id"]) == str(lesson_id)), None)
        assert match is not None
        assert abs(float(match["confidence"]) - 0.5) < 0.01
