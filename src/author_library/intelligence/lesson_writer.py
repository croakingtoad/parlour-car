"""Lesson writer — auto-store quality gate findings as ingestion lessons.

Called by QG1 (inline checks in ingestion_pipeline.py) and QG2 (async
task_quality_gate in tasks.py) whenever a problem is detected and fixed.

Implements:
- Deduplication: if a lesson with the same problem_type + similar
  trigger_context already exists, increment its times_prevented counter
  and update confidence rather than creating a duplicate.
- Confidence formula: 0.5 + 0.5 * (times_prevented / max(times_applied, 1))
  Starts at 0.5, approaches 1.0 as the lesson repeatedly proves useful.

Designed to be lightweight — called inline during quality checks, must
not add significant latency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def record_lesson(
    storage: StorageManager,
    *,
    problem_type: str,
    detection_method: str,
    trigger_context: dict[str, Any],
    problem_description: str,
    fix_applied: str,
    prevention_rule: str | None = None,
    prevention_step: str | None = None,
) -> UUID:
    """Record a quality gate finding as an ingestion lesson.

    Before creating a new lesson, checks for an existing active lesson with
    the same problem_type and overlapping trigger_context keys. If found,
    increments times_prevented and updates confidence instead of creating
    a duplicate.

    Args:
        storage: StorageManager with access to the lessons repository.
        problem_type: Category of the problem (e.g. 'orphan_nodes',
            'misclassification', 'chunk_noise', 'theme_explosion',
            'pg_neo4j_desync').
        detection_method: How the problem was detected (e.g. 'qg1_inline',
            'qg2_async', 'theme_dedup', 'consistency_check').
        trigger_context: Dict of context that triggered this lesson — genre,
            source_class, work_id, etc. Used for dedup matching.
        problem_description: Human-readable description of what went wrong.
        fix_applied: What the pipeline did to fix it.
        prevention_rule: Optional rule to prevent this in future runs.
        prevention_step: Optional pipeline step where prevention applies.

    Returns:
        UUID of the created or updated lesson.
    """
    lessons = storage.lessons

    # --- Deduplication check ---
    # Find existing active lessons with the same problem_type.
    # If trigger_context shares any key:value pairs with an existing lesson,
    # treat it as the same recurring problem.
    try:
        existing = await lessons._pool.fetch_all(
            """SELECT id, trigger_context, times_applied, times_prevented
               FROM ingestion_lessons
               WHERE problem_type = $1 AND is_active = TRUE
               ORDER BY confidence DESC""",
            problem_type,
        )
    except Exception as exc:
        log.warning("lesson_dedup_query_failed", problem_type=problem_type, error=str(exc))
        existing = []

    match_id: UUID | None = None
    for row in existing:
        stored_ctx_raw = row["trigger_context"]
        try:
            if isinstance(stored_ctx_raw, str):
                stored_ctx = json.loads(stored_ctx_raw)
            else:
                stored_ctx = dict(stored_ctx_raw)
        except (ValueError, TypeError):
            stored_ctx = {}

        # Match if any key:value in trigger_context appears in the stored lesson
        if any(
            stored_ctx.get(k) == v
            for k, v in trigger_context.items()
            if k in stored_ctx
        ):
            match_id = UUID(str(row["id"]))
            break

    if match_id is not None:
        # Existing lesson — increment prevented counter (updates confidence)
        try:
            await lessons.increment_prevented(match_id)
            log.info(
                "lesson_prevented_incremented",
                lesson_id=str(match_id),
                problem_type=problem_type,
            )
            return match_id
        except Exception as exc:
            log.warning(
                "lesson_increment_failed",
                lesson_id=str(match_id),
                error=str(exc),
            )
            # Fall through to create a new lesson if increment fails

    # --- Create new lesson ---
    try:
        lesson_id = await lessons.create_lesson({
            "problem_type": problem_type,
            "detection_method": detection_method,
            "trigger_context": trigger_context,
            "problem_description": problem_description,
            "fix_applied": fix_applied,
            "prevention_rule": prevention_rule,
            "prevention_step": prevention_step,
            "confidence": 0.5,
        })
        log.info(
            "lesson_created",
            lesson_id=str(lesson_id),
            problem_type=problem_type,
            step=prevention_step,
        )
        return lesson_id
    except Exception as exc:
        # Lesson writing must NEVER crash the quality gate
        log.error(
            "lesson_write_failed",
            problem_type=problem_type,
            error=str(exc),
        )
        raise
