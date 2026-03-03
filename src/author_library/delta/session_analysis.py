"""P5: Session analysis — themes, threads, "pick up here" suggestions.

After a session ends, generates analysis including:
  - What sources were engaged
  - What themes emerged across captures
  - What threads connect this session to previous sessions
  - Suggested "pick up here next time" note (ADHD recovery)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionSource:
    """A source engaged during the session."""

    work_id: str
    title: str
    capture_count: int
    themes: list[str]


@dataclass(frozen=True, slots=True)
class SessionThread:
    """A thread connecting this session to a previous session."""

    theme: str
    previous_session_id: str
    previous_session_date: str
    connection_description: str


@dataclass(frozen=True, slots=True)
class PickUpHere:
    """Suggested re-entry point for next session."""

    work_id: str
    work_title: str
    chapter: str
    suggestion: str
    themes_in_progress: list[str]


@dataclass(frozen=True, slots=True)
class SessionAnalysisResult:
    """Complete analysis of a study session."""

    session_id: str
    date_start: str
    date_end: str
    duration_minutes: float
    sources: list[SessionSource]
    themes: list[str]
    theme_counts: dict[str, int]
    threads: list[SessionThread]
    pick_up_here: list[PickUpHere]
    capture_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/MCP output."""
        return {
            "session_id": self.session_id,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "duration_minutes": round(self.duration_minutes, 1),
            "sources": [
                {
                    "work_id": s.work_id,
                    "title": s.title,
                    "capture_count": s.capture_count,
                    "themes": s.themes,
                }
                for s in self.sources
            ],
            "themes": self.themes,
            "theme_counts": self.theme_counts,
            "threads": [
                {
                    "theme": t.theme,
                    "previous_session_id": t.previous_session_id,
                    "previous_session_date": t.previous_session_date,
                    "connection_description": t.connection_description,
                }
                for t in self.threads
            ],
            "pick_up_here": [
                {
                    "work_id": p.work_id,
                    "work_title": p.work_title,
                    "chapter": p.chapter,
                    "suggestion": p.suggestion,
                    "themes_in_progress": p.themes_in_progress,
                }
                for p in self.pick_up_here
            ],
            "capture_count": self.capture_count,
        }


class SessionAnalyzer:
    """Analyzes completed study sessions.

    After a session ends, builds a comprehensive analysis of what happened,
    what themes emerged, how it connects to previous sessions, and where
    the user should pick up next time.
    """

    def __init__(self, *, storage: StorageManager) -> None:
        self._storage = storage

    async def analyze(self, session_id: str) -> SessionAnalysisResult | None:
        """Analyze a completed session.

        Args:
            session_id: The session to analyze.

        Returns:
            SessionAnalysisResult or None if session not found.
        """
        # Get session data
        session = await self._get_session(session_id)
        if not session:
            return None

        # Get captures for this session
        captures = await self._get_session_captures(session_id)
        if not captures:
            return SessionAnalysisResult(
                session_id=session_id,
                date_start=session.get("date_start", ""),
                date_end=session.get("date_end", ""),
                duration_minutes=0,
                sources=[],
                themes=[],
                theme_counts={},
                threads=[],
                pick_up_here=[],
                capture_count=0,
            )

        # Analyze sources
        sources = await self._analyze_sources(captures)

        # Analyze themes
        chunk_ids = [c["chunk_id"] for c in captures if c.get("chunk_id")]
        themes, theme_counts = await self._analyze_themes(chunk_ids)

        # Find threads to previous sessions
        threads = await self._find_threads(session_id, themes)

        # Generate "pick up here" suggestions
        pick_up_here = self._generate_pick_up_suggestions(captures, sources, themes)

        # Compute duration
        duration = self._compute_duration(session)

        return SessionAnalysisResult(
            session_id=session_id,
            date_start=session.get("date_start", ""),
            date_end=session.get("date_end", ""),
            duration_minutes=duration,
            sources=sources,
            themes=themes,
            theme_counts=theme_counts,
            threads=threads,
            pick_up_here=pick_up_here,
            capture_count=len(captures),
        )

    async def _get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session data from the database."""
        row = await self._storage.pg.fetch_one(
            """SELECT id::text AS session_id, user_id,
                      date_start::text, date_end::text,
                      title, metadata
            FROM sessions WHERE id::text = $1""",
            session_id,
        )
        return dict(row) if row else None

    async def _get_session_captures(
        self, session_id: str,
    ) -> list[dict[str, Any]]:
        """Get all captures associated with a session."""
        rows = await self._storage.pg.fetch_all(
            """SELECT c.id::text AS chunk_id, c.work_id, c.text,
                      c.granularity, c.source_class, c.chapter,
                      c.created_at::text AS date_created, c.metadata
            FROM session_captures sc
            JOIN chunks c ON c.id = sc.chunk_id
            WHERE sc.session_id::text = $1
            ORDER BY sc.capture_order""",
            session_id,
        )
        return [dict(row) for row in rows]

    async def _analyze_sources(
        self, captures: list[dict[str, Any]],
    ) -> list[SessionSource]:
        """Analyze which sources were engaged."""
        work_counts: dict[str, int] = {}
        for cap in captures:
            wid = cap.get("work_id", "")
            work_counts[wid] = work_counts.get(wid, 0) + 1

        sources: list[SessionSource] = []
        for work_id, count in sorted(work_counts.items(), key=lambda x: -x[1]):
            work_info = await self._storage.works.get(work_id)
            title = work_info.get("title", work_id) if work_info else work_id

            # Get themes for this work's captures
            chunk_ids = [
                c["chunk_id"] for c in captures
                if c.get("work_id") == work_id and c.get("chunk_id")
            ]
            themes = await self._get_themes_for_chunks(chunk_ids)

            sources.append(SessionSource(
                work_id=work_id,
                title=title,
                capture_count=count,
                themes=themes,
            ))

        return sources

    async def _analyze_themes(
        self, chunk_ids: list[str],
    ) -> tuple[list[str], dict[str, int]]:
        """Analyze themes across all session captures."""
        if not chunk_ids:
            return [], {}

        try:
            records = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE c.chunk_id IN $chunk_ids
                RETURN t.canonical_name AS theme, count(*) AS count
                ORDER BY count DESC""",
                {"chunk_ids": chunk_ids},
            )
            themes = [r["theme"] for r in records if r.get("theme")]
            theme_counts = {
                r["theme"]: r["count"]
                for r in records
                if r.get("theme")
            }
            return themes, theme_counts
        except Exception:
            log.debug("session_theme_analysis_failed")
            return [], {}

    async def _find_threads(
        self,
        session_id: str,
        themes: list[str],
    ) -> list[SessionThread]:
        """Find threads connecting this session to previous sessions."""
        if not themes:
            return []

        threads: list[SessionThread] = []

        try:
            # Find previous sessions that share themes
            for theme in themes[:5]:  # Cap at 5 themes
                rows = await self._storage.pg.fetch_all(
                    """SELECT DISTINCT s.id::text AS session_id,
                              s.date_start::text AS session_date
                    FROM sessions s
                    JOIN session_captures sc ON sc.session_id = s.id
                    JOIN chunks c ON c.id = sc.chunk_id
                    JOIN LATERAL (
                        SELECT 1 FROM unnest(ARRAY[$2]) AS t
                    ) themes ON true
                    WHERE s.id::text != $1
                      AND s.date_end IS NOT NULL
                    ORDER BY s.date_start DESC
                    LIMIT 3""",
                    session_id,
                    theme,
                )

                for row in rows:
                    threads.append(SessionThread(
                        theme=theme,
                        previous_session_id=row["session_id"],
                        previous_session_date=row["session_date"],
                        connection_description=f"Both sessions explored '{theme}'.",
                    ))
        except Exception:
            log.debug("session_thread_finding_failed")

        return threads

    async def _get_themes_for_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Get themes for a set of chunks from the graph."""
        if not chunk_ids:
            return []

        try:
            records = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE c.chunk_id IN $chunk_ids
                RETURN DISTINCT t.canonical_name AS theme
                ORDER BY theme""",
                {"chunk_ids": chunk_ids},
            )
            return [r["theme"] for r in records if r.get("theme")]
        except Exception:
            return []

    @staticmethod
    def _generate_pick_up_suggestions(
        captures: list[dict[str, Any]],
        sources: list[SessionSource],
        themes: list[str],
    ) -> list[PickUpHere]:
        """Generate "pick up here" suggestions for next session."""
        suggestions: list[PickUpHere] = []

        if not captures:
            return suggestions

        # Suggest continuing from the last captured source/chapter
        last_capture = captures[-1]
        last_work = last_capture.get("work_id", "")
        last_chapter = last_capture.get("chapter", "")

        # Find the source info
        source_title = last_work
        for src in sources:
            if src.work_id == last_work:
                source_title = src.title
                break

        if last_work:
            suggestion_text = f"Continue with {source_title}"
            if last_chapter:
                suggestion_text += f" from {last_chapter}"

            suggestions.append(PickUpHere(
                work_id=last_work,
                work_title=source_title,
                chapter=last_chapter or "",
                suggestion=suggestion_text,
                themes_in_progress=themes[:3],
            ))

        return suggestions

    @staticmethod
    def _compute_duration(session: dict[str, Any]) -> float:
        """Compute session duration in minutes."""
        start = session.get("date_start", "")
        end = session.get("date_end", "")

        if not start or not end:
            return 0.0

        try:
            from datetime import datetime

            dt_start = datetime.fromisoformat(start)
            dt_end = datetime.fromisoformat(end)
            return (dt_end - dt_start).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.0
