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
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
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
