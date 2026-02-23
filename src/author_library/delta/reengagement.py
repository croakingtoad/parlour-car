"""P1: Re-engagement detector.

Detects when the user captures from a source that already has captures,
indicating a re-reading or re-watching. Uses the pass_number field from
Epic A7 — new captures from a re-engagement get an incremented pass_number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReengagementInfo:
    """Information about a re-engagement with a source."""

    work_id: str
    current_pass: int
    previous_pass: int
    is_reengagement: bool
    previous_capture_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def work_title(self) -> str:
        return self.metadata.get("work_title", "")

    @property
    def first_engagement_date(self) -> str:
        return self.metadata.get("first_engagement_date", "")

    @property
    def last_engagement_date(self) -> str:
        return self.metadata.get("last_engagement_date", "")


class ReengagementDetector:
    """Detects re-engagement with previously captured sources.

    When the user captures from a source that already has captures, this
    constitutes a re-engagement. The detector determines the appropriate
    pass_number for new captures and provides metadata about the previous
    engagement(s).
    """

    def __init__(self, *, storage: StorageManager) -> None:
        self._storage = storage

    async def detect(self, work_id: str) -> ReengagementInfo:
        """Check if capturing from this work constitutes a re-engagement.

        Args:
            work_id: The work being captured from.

        Returns:
            ReengagementInfo with pass number and engagement history.
        """
        # Get maximum pass_number for existing chunks of this work
        max_pass = await self._storage.chunks.get_max_pass_number(work_id)

        # Count existing captures (non-personal chunks for this work)
        capture_count = await self._count_captures(work_id)

        # Get work metadata
        work_info = await self._storage.works.get(work_id)
        metadata: dict[str, Any] = {}
        if work_info:
            metadata["work_title"] = work_info.get("title", "")
            metadata["author"] = work_info.get("author", "")
            metadata["media"] = work_info.get("media", "")

        # Get engagement date range
        date_info = await self._get_engagement_dates(work_id)
        metadata.update(date_info)

        is_reengagement = max_pass > 0 and capture_count > 0

        return ReengagementInfo(
            work_id=work_id,
            current_pass=max_pass + 1 if is_reengagement else max(max_pass, 1),
            previous_pass=max_pass,
            is_reengagement=is_reengagement,
            previous_capture_count=capture_count,
            metadata=metadata,
        )

    async def get_pass_history(self, work_id: str) -> list[dict[str, Any]]:
        """Get a summary of captures per pass for a work.

        Returns a list of dicts with pass_number, count, and date range
        for each engagement pass.

        Args:
            work_id: The work to get pass history for.

        Returns:
            List of pass summaries, ordered by pass_number.
        """
        rows = await self._storage.pg.fetch_all(
            """SELECT pass_number,
                      COUNT(*) AS capture_count,
                      MIN(created_at)::text AS first_capture,
                      MAX(created_at)::text AS last_capture
            FROM chunks
            WHERE work_id = $1
              AND source_class != 'personal'
              AND pass_number IS NOT NULL
            GROUP BY pass_number
            ORDER BY pass_number""",
            work_id,
        )

        return [
            {
                "pass_number": row["pass_number"],
                "capture_count": row["capture_count"],
                "first_capture": row["first_capture"],
                "last_capture": row["last_capture"],
            }
            for row in rows
        ]

    async def get_captures_by_pass(
        self,
        work_id: str,
        pass_number: int,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all captures for a specific pass of a work.

        Args:
            work_id: The work ID.
            pass_number: The engagement pass number.
            limit: Maximum captures to return.

        Returns:
            List of chunk dicts for the specified pass.
        """
        rows = await self._storage.pg.fetch_all(
            """SELECT id::text AS chunk_id, work_id, text, granularity,
                      source_class, chapter, section, position,
                      created_at::text AS date_created, metadata
            FROM chunks
            WHERE work_id = $1
              AND pass_number = $2
            ORDER BY position
            LIMIT $3""",
            work_id,
            pass_number,
            limit,
        )

        return [dict(row) for row in rows]

    async def _count_captures(self, work_id: str) -> int:
        """Count non-personal chunks for a work."""
        result = await self._storage.pg.fetch_val(
            """SELECT COUNT(*) FROM chunks
            WHERE work_id = $1 AND source_class != 'personal'""",
            work_id,
        )
        return int(result)

    async def _get_engagement_dates(self, work_id: str) -> dict[str, str]:
        """Get first and last capture dates for a work."""
        row = await self._storage.pg.fetch_one(
            """SELECT MIN(created_at)::text AS first_date,
                      MAX(created_at)::text AS last_date
            FROM chunks
            WHERE work_id = $1 AND source_class != 'personal'""",
            work_id,
        )
        if row and row["first_date"]:
            return {
                "first_engagement_date": row["first_date"],
                "last_engagement_date": row["last_date"],
            }
        return {}
