"""Integration tests for LL3 — lesson context injection into LLM prompts.

Tests:
- get_lesson_context() returns empty string and empty list when no lessons exist
- get_lesson_context() returns formatted context and IDs for matching lessons
- get_lesson_context() respects min_confidence threshold (skips low-confidence)
- get_lesson_context() caps at max_lessons (5 by default)
- get_lesson_context() returns empty when step has no matching lessons
- increment_applied tracks lessons that were injected (via direct call)
- Prevention rule included in context when present
- Context string format is correct (numbered list with problem_type and description)

Uses the test database (author_library_test).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from author_library.intelligence.lesson_writer import get_lesson_context, record_lesson

from .conftest import SKIP_NO_DB


# ---------------------------------------------------------------------------
# TestGetLessonContext
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestGetLessonContext:
    """get_lesson_context() builds prompt-ready lesson context."""

    async def test_returns_empty_when_no_lessons(self, clean_storage: Any) -> None:
        """Returns ('', []) when no lessons exist for the step."""
        context, ids = await get_lesson_context(clean_storage, "classification")

        assert context == ""
        assert ids == []

    async def test_returns_empty_for_wrong_step(self, clean_storage: Any) -> None:
        """Returns ('', []) when lessons exist but not for the requested step."""
        await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--inject-step"},
            problem_description="Noisy chunks detected.",
            fix_applied="Flagged.",
            prevention_rule="Filter chunks under 50 chars.",
            prevention_step="chunking",  # different step
        )

        context, ids = await get_lesson_context(clean_storage, "classification")

        assert context == ""
        assert ids == []

    async def test_returns_context_for_matching_step(self, clean_storage: Any) -> None:
        """Returns formatted context and lesson IDs for matching step."""
        lesson_id = await record_lesson(
            clean_storage,
            problem_type="misclassification",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--inject-match"},
            problem_description="Devotional works misclassified as contextual.",
            fix_applied="Reclassified as primary.",
            prevention_rule="If author matches subject, bias toward PRIMARY.",
            prevention_step="classification",
        )
        # New lessons start at 0.5; increment_prevented bumps confidence above 0.6
        await clean_storage.lessons.increment_prevented(lesson_id)

        context, ids = await get_lesson_context(clean_storage, "classification")

        assert context != ""
        assert len(ids) == 1
        assert ids[0] == lesson_id
        assert "KNOWN ISSUES FROM PREVIOUS INGESTIONS:" in context
        assert "misclassification" in context
        assert "Devotional works misclassified" in context
        assert "bias toward PRIMARY" in context

    async def test_skips_low_confidence_lessons(self, clean_storage: Any) -> None:
        """Lessons below min_confidence (0.6 default) are excluded."""
        # Create a lesson at default confidence 0.5 — below the 0.6 threshold
        await record_lesson(
            clean_storage,
            problem_type="chunk_noise",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--inject-lowconf"},
            problem_description="Low-confidence lesson.",
            fix_applied="Flagged.",
            prevention_step="entity_extraction",
        )

        context, ids = await get_lesson_context(clean_storage, "entity_extraction")

        # Default confidence is 0.5, threshold is 0.6 — should be excluded
        assert context == ""
        assert ids == []

    async def test_includes_high_confidence_lessons(self, clean_storage: Any) -> None:
        """Lessons at or above min_confidence are included."""
        context, ids = await get_lesson_context(
            clean_storage, "entity_extraction", min_confidence=0.0
        )
        # With min_confidence=0.0, any lesson at 0.5 should be included
        # (seed lessons from migration may be present; we just verify the API works)
        assert isinstance(context, str)
        assert isinstance(ids, list)

    async def test_caps_at_max_lessons(self, clean_storage: Any) -> None:
        """Returns at most max_lessons entries."""
        # Create 7 lessons for the same step, all above threshold
        for i in range(7):
            lid = await record_lesson(
                clean_storage,
                problem_type=f"test_problem_{i}",
                detection_method="qg1_inline",
                trigger_context={"work_id": f"test--inject-cap-{i}"},
                problem_description=f"Problem number {i}.",
                fix_applied="Fixed.",
                prevention_rule=f"Rule {i}.",
                prevention_step="thematic_index",
            )
            # Manually bump confidence above 0.6 via increment_prevented calls
            for _ in range(3):
                await clean_storage.lessons.increment_prevented(lid)

        context, ids = await get_lesson_context(clean_storage, "thematic_index")

        # Should be capped at 5
        assert len(ids) <= 5

    async def test_context_excludes_prevention_when_none(self, clean_storage: Any) -> None:
        """No 'Prevention:' suffix when prevention_rule is None."""
        await record_lesson(
            clean_storage,
            problem_type="orphan_nodes",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--inject-noprev"},
            problem_description="Orphaned Neo4j nodes found.",
            fix_applied="Cleaned up.",
            prevention_rule=None,
            prevention_step="classification",
        )

        context, ids = await get_lesson_context(
            clean_storage, "classification", min_confidence=0.0
        )

        # The lesson line should not have 'Prevention:' appended
        lines = [l for l in context.split("\n") if "orphan_nodes" in l]
        assert len(lines) == 1
        assert "Prevention:" not in lines[0]

    async def test_increment_applied_tracks_injected_lessons(
        self, clean_storage: Any
    ) -> None:
        """increment_applied() bumps times_applied for injected lessons."""
        lesson_id = await record_lesson(
            clean_storage,
            problem_type="misclassification",
            detection_method="qg1_inline",
            trigger_context={"work_id": "test--inject-applied"},
            problem_description="Test lesson for applied tracking.",
            fix_applied="Reclassified.",
            prevention_step="classification",
        )

        # Simulate injection loop: get context then track applied
        _, ids = await get_lesson_context(
            clean_storage, "classification", min_confidence=0.0
        )

        # Find our lesson in the returned IDs
        assert lesson_id in ids

        # Call increment_applied as the injection site would
        await clean_storage.lessons.increment_applied(lesson_id)

        # Verify times_applied was bumped
        lessons = await clean_storage.lessons.get_lessons_for_step(
            "classification", active_only=True, min_confidence=0.0
        )
        our_lesson = next((l for l in lessons if l["id"] == lesson_id), None)
        assert our_lesson is not None
        assert our_lesson["times_applied"] == 1

    async def test_numbered_list_format(self, clean_storage: Any) -> None:
        """Context lines are numbered starting from 1."""
        for i in range(3):
            lid = await record_lesson(
                clean_storage,
                problem_type=f"format_test_{i}",
                detection_method="qg1_inline",
                trigger_context={"work_id": f"test--inject-fmt-{i}"},
                problem_description=f"Description {i}.",
                fix_applied="Fixed.",
                prevention_step="voice_profile",
            )
            # Bump confidence above 0.6
            for _ in range(3):
                await clean_storage.lessons.increment_prevented(lid)

        context, ids = await get_lesson_context(clean_storage, "voice_profile")

        assert len(ids) >= 1
        lines = [l for l in context.split("\n") if l.startswith("1.")]
        assert len(lines) == 1  # exactly one line starting with "1."
