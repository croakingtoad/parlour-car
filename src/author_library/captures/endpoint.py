"""HTTP capture endpoint for the Chrome extension.

Provides POST /api/v1/captures with Bearer token authentication.
Validates the capture payload and enqueues it for background processing
via arq. Returns immediately with a job_id for status polling.
"""

from __future__ import annotations

import json
import secrets
from typing import TYPE_CHECKING, Any

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse

from author_library.captures.models import CapturePayload

if TYPE_CHECKING:
    from author_library.queue import TaskQueue

log = structlog.get_logger(__name__)


def _authenticate(request: Request, expected_key: str) -> bool:
    """Validate Bearer token from Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header[7:]
    return secrets.compare_digest(token, expected_key)


async def handle_capture(request: Request) -> JSONResponse:
    """Handle POST /api/v1/captures from the Chrome extension.

    Authenticates via Bearer token, validates the capture payload,
    and enqueues it for background processing.

    The request's app state must contain:
        - api_key: str — the expected PARLOUR_API_KEY
        - task_queue: TaskQueue — for enqueuing capture jobs
    """
    state: dict[str, Any] = request.app.state._state  # type: ignore[union-attr]

    # Authenticate
    api_key = state.get("api_key", "")
    if not api_key:
        log.error("capture_endpoint_no_api_key_configured")
        return JSONResponse(
            {"error": "Server misconfigured: no API key set"},
            status_code=500,
        )

    if not _authenticate(request, api_key):
        log.warning("capture_endpoint_auth_failed", remote=request.client.host if request.client else "unknown")
        return JSONResponse(
            {"error": "Unauthorized"},
            status_code=401,
        )

    # Parse and validate payload
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"},
            status_code=400,
        )

    try:
        payload = CapturePayload(**body)
    except Exception as exc:
        log.warning("capture_endpoint_validation_failed", error=str(exc))
        return JSONResponse(
            {"error": f"Validation error: {exc}"},
            status_code=422,
        )

    # Enqueue for background processing
    task_queue: TaskQueue | None = state.get("task_queue")
    if task_queue is None or not task_queue.available:
        log.error("capture_endpoint_queue_unavailable")
        return JSONResponse(
            {"error": "Task queue unavailable. Ensure Redis is running."},
            status_code=503,
        )

    job_id = await task_queue.enqueue_capture(payload=payload.model_dump(mode="json"))
    if job_id is None:
        return JSONResponse(
            {"error": "Failed to enqueue capture"},
            status_code=500,
        )

    log.info(
        "capture_enqueued",
        job_id=job_id,
        source_url=payload.source_url,
        mode=payload.mode.value,
        timestamp=payload.timestamp_seconds,
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "message": "Capture queued for processing. Poll /api/v1/captures/status/{job_id}.",
        },
        status_code=202,
    )


async def handle_capture_status(request: Request) -> JSONResponse:
    """Handle GET /api/v1/captures/status/{job_id}.

    Returns the current status of a capture processing job.
    """
    state: dict[str, Any] = request.app.state._state  # type: ignore[union-attr]

    # Authenticate
    api_key = state.get("api_key", "")
    if api_key and not _authenticate(request, api_key):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = request.path_params.get("job_id", "")
    if not job_id:
        return JSONResponse({"error": "job_id is required"}, status_code=400)

    task_queue: TaskQueue | None = state.get("task_queue")
    if task_queue is None or not task_queue.available:
        return JSONResponse(
            {"error": "Task queue unavailable"},
            status_code=503,
        )

    status = await task_queue.get_job_status(job_id)
    return JSONResponse(status)
