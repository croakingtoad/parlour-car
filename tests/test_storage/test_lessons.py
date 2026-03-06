"""Tests for the ingestion lessons repository."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from author_library.storage.lessons import LessonRepository
from author_library.storage.migrations.runner import run_migrations

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool


@pytest.fixture
async def repo(pg_pool: PostgresPool) -> LessonRepository:
    """Provide a LessonRepository with migrations applied."""
    await run_migrations(pg_pool)
    # Clear seeded lessons so tests start clean
    await pg_pool.execute("DELETE FROM ingestion_lessons")
    return LessonRepository(pg_pool)


def _sample_lesson(**overrides: object) -> dict:
    """Build a sample lesson dict with sensible defaults."""
    data: dict = {
        "problem_type": "misclassification",
        "detection_method": "manual_review",
        "trigger_context": {"source_class": "contextual", "expected": "primary"},
        "problem_description": "Devotional works misclassified as contextual.",
        "fix_applied": "Reclassified to primary.",
        "prevention_rule": "Match author field to subject author.",
        "prevention_step": "classification",
        "confidence": 0.8,
    }
    data.update(overrides)
    return data


async def test_create_lesson(repo: LessonRepository) -> None:
    """Creating a lesson returns a valid UUID."""
    lesson_id = await repo.create_lesson(_sample_lesson())
    assert isinstance(lesson_id, UUID)


async def test_list_all_returns_created(repo: LessonRepository) -> None:
    """list_all includes a newly created lesson."""
    lesson_id = await repo.create_lesson(_sample_lesson())
    lessons = await repo.list_all()
    ids = [row["id"] for row in lessons]
    assert lesson_id in ids


async def test_list_all_active_only(repo: LessonRepository) -> None:
    """list_all with active_only=True excludes deactivated lessons."""
    lid = await repo.create_lesson(_sample_lesson())
    await repo.deactivate_lesson(lid)
    active = await repo.list_all(active_only=True)
    ids = [row["id"] for row in active]
    assert lid not in ids

    all_lessons = await repo.list_all(active_only=False)
    ids_all = [row["id"] for row in all_lessons]
    assert lid in ids_all


async def test_get_lessons_for_step(repo: LessonRepository) -> None:
    """get_lessons_for_step filters by prevention_step."""
    await repo.create_lesson(_sample_lesson(prevention_step="classification"))
    await repo.create_lesson(_sample_lesson(prevention_step="chunking"))

    classification_lessons = await repo.get_lessons_for_step("classification")
    assert len(classification_lessons) == 1
    assert classification_lessons[0]["prevention_step"] == "classification"

    chunking_lessons = await repo.get_lessons_for_step("chunking")
    assert len(chunking_lessons) == 1
    assert chunking_lessons[0]["prevention_step"] == "chunking"


async def test_get_lessons_for_step_min_confidence(repo: LessonRepository) -> None:
    """get_lessons_for_step respects min_confidence threshold."""
    await repo.create_lesson(
        _sample_lesson(prevention_step="classification", confidence=0.3)
    )
    await repo.create_lesson(
        _sample_lesson(prevention_step="classification", confidence=0.9)
    )

    high_conf = await repo.get_lessons_for_step("classification", min_confidence=0.5)
    assert len(high_conf) == 1
    assert high_conf[0]["confidence"] >= 0.5

    all_conf = await repo.get_lessons_for_step("classification", min_confidence=0.0)
    assert len(all_conf) == 2


async def test_get_lessons_for_step_active_only(repo: LessonRepository) -> None:
    """get_lessons_for_step excludes inactive lessons by default."""
    lid = await repo.create_lesson(_sample_lesson(prevention_step="classification"))
    await repo.deactivate_lesson(lid)

    active = await repo.get_lessons_for_step("classification")
    assert len(active) == 0

    including_inactive = await repo.get_lessons_for_step(
        "classification", active_only=False
    )
    assert len(including_inactive) == 1


async def test_increment_applied(repo: LessonRepository) -> None:
    """increment_applied bumps times_applied and sets last_applied_at."""
    lid = await repo.create_lesson(_sample_lesson())
    await repo.increment_applied(lid)
    await repo.increment_applied(lid)

    lessons = await repo.list_all()
    lesson = next(r for r in lessons if r["id"] == lid)
    assert lesson["times_applied"] == 2
    assert lesson["last_applied_at"] is not None


async def test_increment_prevented_updates_confidence(repo: LessonRepository) -> None:
    """increment_prevented recalculates confidence correctly.

    Formula: confidence = 0.5 + 0.5 * (times_prevented / max(times_applied, 1))

    Starting: times_applied=0, times_prevented=0
    After increment_prevented: times_prevented=1, times_applied=0
        confidence = 0.5 + 0.5 * (1 / max(0, 1)) = 0.5 + 0.5 = 1.0

    After increment_applied x3 then increment_prevented:
        times_applied=3, times_prevented=2
        confidence = 0.5 + 0.5 * (2 / max(3, 1)) = 0.5 + 0.333 = 0.833
    """
    lid = await repo.create_lesson(_sample_lesson(confidence=0.5))

    # First prevention with no applications: max(0, 1) = 1
    await repo.increment_prevented(lid)
    lessons = await repo.list_all()
    lesson = next(r for r in lessons if r["id"] == lid)
    assert lesson["times_prevented"] == 1
    assert abs(lesson["confidence"] - 1.0) < 0.01

    # Now apply 3 times and prevent once more
    await repo.increment_applied(lid)
    await repo.increment_applied(lid)
    await repo.increment_applied(lid)
    await repo.increment_prevented(lid)

    lessons = await repo.list_all()
    lesson = next(r for r in lessons if r["id"] == lid)
    assert lesson["times_prevented"] == 2
    assert lesson["times_applied"] == 3
    # confidence = 0.5 + 0.5 * (2/3) = 0.833...
    expected = 0.5 + 0.5 * (2.0 / 3.0)
    assert abs(lesson["confidence"] - expected) < 0.01


async def test_deactivate_lesson(repo: LessonRepository) -> None:
    """deactivate_lesson sets is_active to False."""
    lid = await repo.create_lesson(_sample_lesson())
    await repo.deactivate_lesson(lid)

    all_lessons = await repo.list_all(active_only=False)
    lesson = next(r for r in all_lessons if r["id"] == lid)
    assert lesson["is_active"] is False


async def test_increment_applied_nonexistent_raises(repo: LessonRepository) -> None:
    """increment_applied raises StorageError for nonexistent lesson."""
    from uuid import uuid4

    from author_library.errors import StorageError

    with pytest.raises(StorageError):
        await repo.increment_applied(uuid4())


async def test_increment_prevented_nonexistent_raises(repo: LessonRepository) -> None:
    """increment_prevented raises StorageError for nonexistent lesson."""
    from uuid import uuid4

    from author_library.errors import StorageError

    with pytest.raises(StorageError):
        await repo.increment_prevented(uuid4())


async def test_deactivate_nonexistent_raises(repo: LessonRepository) -> None:
    """deactivate_lesson raises StorageError for nonexistent lesson."""
    from uuid import uuid4

    from author_library.errors import StorageError

    with pytest.raises(StorageError):
        await repo.deactivate_lesson(uuid4())


async def test_create_lesson_minimal_fields(repo: LessonRepository) -> None:
    """Creating a lesson with only required fields works (no prevention_rule/step)."""
    data = {
        "problem_type": "unknown_issue",
        "detection_method": "automated",
        "trigger_context": {},
        "problem_description": "Something went wrong.",
        "fix_applied": "Fixed it.",
    }
    lid = await repo.create_lesson(data)
    assert isinstance(lid, UUID)

    lessons = await repo.list_all()
    lesson = next(r for r in lessons if r["id"] == lid)
    assert lesson["prevention_rule"] is None
    assert lesson["prevention_step"] is None
    assert abs(lesson["confidence"] - 0.5) < 0.01


async def test_seeded_lessons_exist(pg_pool: PostgresPool) -> None:
    """Migration 013 seeds 5 lessons into the table.

    The autouse cleanup fixture wipes the table between tests, and the
    migration runner is idempotent (won't re-insert seeds). So we re-run
    the seed SQL directly to verify its content.
    """
    from pathlib import Path

    await run_migrations(pg_pool)
    # Re-seed: the autouse cleanup deletes rows but the idempotent migration
    # won't re-insert. Execute just the INSERT portion of the migration.
    seed_sql = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "author_library"
        / "storage"
        / "migrations"
        / "013_ingestion_lessons.sql"
    ).read_text()
    # Extract only the INSERT statement (everything after the last CREATE INDEX)
    insert_start = seed_sql.index("INSERT INTO ingestion_lessons")
    await pg_pool.execute(seed_sql[insert_start:])

    repo = LessonRepository(pg_pool)
    lessons = await repo.list_all()
    assert len(lessons) >= 5

    steps = {r["prevention_step"] for r in lessons}
    assert "classification" in steps
    assert "entity_extraction" in steps
    assert "thematic_index" in steps
    assert "chunking" in steps

    types = {r["problem_type"] for r in lessons}
    assert "misclassification" in types
    assert "theme_explosion" in types
    assert "llm_format_violation" in types
    assert "orphaned_nodes" in types
    assert "micro_chunk_pollution" in types


async def test_lessons_ordered_by_confidence(repo: LessonRepository) -> None:
    """list_all returns lessons ordered by confidence descending."""
    await repo.create_lesson(_sample_lesson(confidence=0.3, prevention_step="a"))
    await repo.create_lesson(_sample_lesson(confidence=0.9, prevention_step="b"))
    await repo.create_lesson(_sample_lesson(confidence=0.6, prevention_step="c"))

    lessons = await repo.list_all()
    confidences = [r["confidence"] for r in lessons]
    assert confidences == sorted(confidences, reverse=True)
