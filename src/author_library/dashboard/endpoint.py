"""Dashboard HTTP endpoints.

GET /dashboard         — serve the single-page HTML template
GET /dashboard/stats   — JSON: library overview + per-work details + graph stats
GET /dashboard/health  — JSON: health check results
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from author_library.dashboard.health import run_all_checks
from author_library.dashboard.queries import (
    get_all_themes,
    get_author_health,
    get_pipeline_status,
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
    get_theme_detail,
    get_voice_profiles,
    get_work_detail,
)

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

_TEMPLATE = Path(__file__).parent / "template.html"


def _storage(request: Request) -> "StorageManager":
    return request.app.state.dashboard_state["storage"]


async def handle_dashboard(request: Request) -> FileResponse:
    """Serve the dashboard HTML."""
    return FileResponse(_TEMPLATE, media_type="text/html")


async def handle_stats(request: Request) -> JSONResponse:
    """Return all library stats as JSON."""
    storage = _storage(request)
    try:
        library, works, graph = await asyncio.gather(
            get_library_overview(storage.pg),
            get_per_work_details(storage.pg),
            get_graph_stats(storage.neo4j),
        )
        library["by_source_class"] = {
            "primary":    library.get("primary_works", 0),
            "secondary":  library.get("secondary_works", 0),
            "contextual": library.get("contextual_works", 0),
            "tertiary":   library.get("tertiary_works", 0),
            "personal":   library.get("personal_works", 0),
            "reference":  library.get("reference_works", 0),
        }
        return JSONResponse({"library": library, "works": works, "graph": graph})
    except Exception as exc:
        log.error("dashboard_stats_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_health(request: Request) -> JSONResponse:
    """Return health check results as JSON."""
    storage = _storage(request)
    try:
        checks = await run_all_checks(storage.pg, storage.neo4j)
        statuses = [c.status for c in checks]
        overall = (
            "error" if "error" in statuses
            else "warn" if "warn" in statuses
            else "ok"
        )
        return JSONResponse({
            "checks": [dataclasses.asdict(c) for c in checks],
            "overall": overall,
        })
    except Exception as exc:
        log.error("dashboard_health_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_voice_profiles(request: Request) -> JSONResponse:
    """Return all current voice profiles."""
    storage = _storage(request)
    try:
        profiles = await get_voice_profiles(storage.pg)
        return JSONResponse({"profiles": profiles})
    except Exception as exc:
        log.error("dashboard_voice_profiles_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_blend_studio_authors(request: Request) -> JSONResponse:
    """Return current voice profiles for Voice Blend Studio.

    Called server-to-server by Blend Studio — no auth required
    since both services run on the same Tailscale node.
    """
    storage = _storage(request)
    try:
        profiles = await get_voice_profiles(storage.pg)
        authors = []
        for p in profiles:
            d = dict(p)
            d["confidence"] = (
                d["profile"].get("confidence", 0.0)
                if isinstance(d.get("profile"), dict)
                else 0.0
            )
            authors.append(d)
        return JSONResponse({"authors": authors})
    except Exception as exc:
        log.error("blend_studio_authors_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_work_detail(request: Request) -> JSONResponse:
    """Return full detail for a single work."""
    storage = _storage(request)
    work_id = request.path_params["work_id"]
    try:
        detail = await get_work_detail(storage.pg, storage.neo4j, work_id)
        if detail is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(detail)
    except Exception as exc:
        log.error("dashboard_work_detail_error", work_id=work_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_themes(request: Request) -> JSONResponse:
    """Return all themes with appearance counts."""
    storage = _storage(request)
    try:
        themes = await get_all_themes(storage.pg)
        return JSONResponse({"themes": themes})
    except Exception as exc:
        log.error("dashboard_themes_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_theme_detail(request: Request) -> JSONResponse:
    """Return full theme detail including per-work appearances and quotes."""
    storage = _storage(request)
    entry_id = request.path_params["entry_id"]
    try:
        detail = await get_theme_detail(storage.pg, storage.neo4j, entry_id)
        if detail is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(detail)
    except Exception as exc:
        log.error("dashboard_theme_detail_error", entry_id=entry_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_author_health(request: Request) -> JSONResponse:
    """Return author health rows for the dashboard integrity table."""
    storage = _storage(request)
    try:
        rows = await get_author_health(storage.pg)
        return JSONResponse({"authors": rows})
    except Exception as exc:
        log.error("dashboard_author_health_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_pipeline(request: Request) -> JSONResponse:
    """Return per-work pipeline completion status."""
    storage = _storage(request)
    try:
        data = await get_pipeline_status(storage.pg, storage.neo4j)
        return JSONResponse(data)
    except Exception as exc:
        log.error("dashboard_pipeline_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
