"""Repository for ingestion lessons — recurring quality problems and fixes.

Lessons capture what went wrong during ingestion (misclassification, theme
explosion, orphaned nodes, etc.) and how to prevent it. The pipeline can
query active lessons for a given step to apply learned prevention rules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from author_library.errors import StorageError

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)


class LessonRepository:
    """PostgreSQL-backed repository for ingestion lessons."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create_lesson(self, data: dict[str, Any]) -> UUID:
        """Insert a new lesson and return its UUID.

        Required keys: problem_type, detection_method, trigger_context,
        problem_description, fix_applied.

        Optional keys: prevention_rule, prevention_step, confidence.
        """
        row = await self._pool.fetch_one(
            """INSERT INTO ingestion_lessons (
                problem_type, detection_method, trigger_context,
                problem_description, fix_applied,
                prevention_rule, prevention_step, confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id""",
            data["problem_type"],
            data["detection_method"],
            json.dumps(data["trigger_context"]),
            data["problem_description"],
            data["fix_applied"],
            data.get("prevention_rule"),
            data.get("prevention_step"),
            data.get("confidence", 0.5),
        )
        if row is None:
            raise StorageError("Failed to create ingestion lesson — no id returned")
        lesson_id: UUID = row["id"]
        log.info(
            "lesson_created",
            lesson_id=str(lesson_id),
            problem_type=data["problem_type"],
            step=data.get("prevention_step"),
        )
        return lesson_id

    async def get_lessons_for_step(
        self,
        step: str,
        *,
        active_only: bool = True,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return lessons applicable to a pipeline step.

        Args:
            step: The pipeline step name (e.g. 'classification', 'chunking').
            active_only: If True, only return active lessons.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of lesson dicts ordered by confidence descending.
        """
        conditions = ["prevention_step = $1", "confidence >= $2"]
        params: list[Any] = [step, min_confidence]

        if active_only:
            conditions.append("is_active = TRUE")

        where = " AND ".join(conditions)
        rows = await self._pool.fetch_all(
            f"""SELECT id, problem_type, detection_method, trigger_context,
                       problem_description, fix_applied, prevention_rule,
                       prevention_step, confidence, times_applied,
                       times_prevented, is_active, created_at, last_applied_at
                FROM ingestion_lessons
                WHERE {where}
                ORDER BY confidence DESC""",
            *params,
        )
        return [dict(r) for r in rows]

    async def increment_applied(self, lesson_id: UUID) -> None:
        """Record that a lesson was applied during ingestion.

        Bumps times_applied and updates last_applied_at.
        """
        result = await self._pool.execute(
            """UPDATE ingestion_lessons
               SET times_applied = times_applied + 1,
                   last_applied_at = NOW()
               WHERE id = $1""",
            lesson_id,
        )
        if not result.endswith("1"):
            raise StorageError(
                "Lesson not found for increment_applied",
                context={"lesson_id": str(lesson_id)},
            )

    async def increment_prevented(self, lesson_id: UUID) -> None:
        """Record that a lesson successfully prevented a problem.

        Bumps times_prevented and recalculates confidence using:
            confidence = 0.5 + 0.5 * (times_prevented / max(times_applied, 1))
        """
        result = await self._pool.execute(
            """UPDATE ingestion_lessons
               SET times_prevented = times_prevented + 1,
                   confidence = 0.5 + 0.5 * (
                       (times_prevented + 1)::FLOAT
                       / GREATEST(times_applied, 1)::FLOAT
                   )
               WHERE id = $1""",
            lesson_id,
        )
        if not result.endswith("1"):
            raise StorageError(
                "Lesson not found for increment_prevented",
                context={"lesson_id": str(lesson_id)},
            )

    async def deactivate_lesson(self, lesson_id: UUID) -> None:
        """Mark a lesson as inactive so it stops being applied."""
        result = await self._pool.execute(
            "UPDATE ingestion_lessons SET is_active = FALSE WHERE id = $1",
            lesson_id,
        )
        if not result.endswith("1"):
            raise StorageError(
                "Lesson not found for deactivation",
                context={"lesson_id": str(lesson_id)},
            )

    async def list_all(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        """List all lessons, optionally filtered to active only.

        Returns:
            List of lesson dicts ordered by confidence descending.
        """
        if active_only:
            rows = await self._pool.fetch_all(
                """SELECT id, problem_type, detection_method, trigger_context,
                          problem_description, fix_applied, prevention_rule,
                          prevention_step, confidence, times_applied,
                          times_prevented, is_active, created_at, last_applied_at
                   FROM ingestion_lessons
                   WHERE is_active = TRUE
                   ORDER BY confidence DESC"""
            )
        else:
            rows = await self._pool.fetch_all(
                """SELECT id, problem_type, detection_method, trigger_context,
                          problem_description, fix_applied, prevention_rule,
                          prevention_step, confidence, times_applied,
                          times_prevented, is_active, created_at, last_applied_at
                   FROM ingestion_lessons
                   ORDER BY confidence DESC"""
            )
        return [dict(r) for r in rows]
