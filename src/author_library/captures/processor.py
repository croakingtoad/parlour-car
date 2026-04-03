"""Capture event processor — orchestrates the full capture pipeline.

This is the main entry point for background processing of captures
from the Chrome extension. Called as an arq background task.

Processing flow:
1. Validate and parse the capture payload
2. Fetch/cache transcript for the source URL
3. If first capture from this URL: generate source overview
4. Route to Quick/Deep/Visual processor based on capture mode
5. Return capture result with chunk_id for status polling
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

from author_library.captures.models import CaptureMode, CapturePayload, CaptureResult
from author_library.captures.overview import generate_source_overview
from author_library.captures.transcript import (
    extract_transcript_window,
    extract_video_id,
    get_transcript,
)

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def process_capture(
    payload: CapturePayload,
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> CaptureResult:
    """Process a single capture event from the Chrome extension.

    This is the top-level orchestrator that coordinates transcript
    retrieval, source overview generation, and mode-specific processing.

    Args:
        payload: Validated capture payload.
        settings: Application settings.
        storage: Storage manager.
        embedding_provider: Embedding provider.

    Returns:
        CaptureResult with processing outcome.
    """
    capture_id = uuid.uuid4().hex[:16]
    errors: list[str] = []

    log.info(
        "capture_processing_start",
        capture_id=capture_id,
        source_url=payload.source_url,
        mode=payload.mode.value,
        timestamp=payload.timestamp_seconds,
    )

    # Step 1: Determine work_id
    video_id = extract_video_id(payload.source_url)
    work_id = (
        f"video--{video_id}"
        if video_id
        else f"media--{hash(payload.source_url) & 0xFFFFFFFF:08x}"
    )

    # Step 2: Get transcript — prefer the one extracted by the extension (runs on
    # the YouTube page, bypasses cloud IP blocks), fall back to server-side fetch.
    if payload.transcript:
        transcript = payload.transcript
        log.info(
            "capture_transcript_from_extension",
            capture_id=capture_id,
            chars=len(transcript),
        )
    else:
        transcript = await get_transcript(
            payload.source_url,
            cache_repo=storage.transcript_cache,
        )

    if not transcript:
        error_msg = f"No transcript available for {payload.source_url}"
        log.warning("capture_no_transcript", source_url=payload.source_url)
        errors.append(error_msg)
        return CaptureResult(
            capture_id=capture_id,
            source_url=payload.source_url,
            work_id=work_id,
            mode=payload.mode.value,
            errors=errors,
        )

    # Step 3: Check if this is the first capture from this source
    existing_work = await storage.works.get(work_id)
    if not existing_work:
        log.info("capture_new_source", work_id=work_id)
        overview = await generate_source_overview(
            payload.source_url,
            payload.source_title,
            settings=settings,
            storage=storage,
        )
        if overview:
            log.info(
                "capture_overview_generated",
                work_id=work_id,
                speakers=overview.speakers,
                content_type=overview.content_type,
            )

    # Step 4: Route to mode-specific processor
    chunk_id: str | None = None

    if payload.mode == CaptureMode.QUICK:
        result = await _process_quick(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        chunk_id = result.get("chunk_id")
        errors.extend(result.get("errors", []))

    elif payload.mode == CaptureMode.DEEP:
        result = await _process_deep(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        chunk_id = result.get("chunk_id")
        errors.extend(result.get("errors", []))

    elif payload.mode == CaptureMode.VISUAL_QUICK:
        result = await _process_visual(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            capture_id=capture_id,
            is_deep=False,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        chunk_id = result.get("chunk_id")
        errors.extend(result.get("errors", []))

    elif payload.mode == CaptureMode.VISUAL_DEEP:
        result = await _process_visual(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            capture_id=capture_id,
            is_deep=True,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        chunk_id = result.get("chunk_id")
        errors.extend(result.get("errors", []))

    # Step 5: Add to session tracking
    if chunk_id:
        try:
            session = await storage.sessions.get_active("marty")
            if session:
                from uuid import UUID

                session_id = session["id"]
                captures = await storage.sessions.list_captures(session_id)
                capture_order = len(captures) + 1
                await storage.sessions.add_capture(
                    session_id, UUID(chunk_id), capture_order
                )
                await storage.sessions.add_source(session_id, work_id)
        except Exception as exc:
            log.warning("capture_session_tracking_failed", error=str(exc))

    log.info(
        "capture_processing_complete",
        capture_id=capture_id,
        chunk_id=chunk_id,
        mode=payload.mode.value,
        errors=len(errors),
    )

    return CaptureResult(
        capture_id=capture_id,
        source_url=payload.source_url,
        work_id=work_id,
        chunk_id=chunk_id,
        mode=payload.mode.value,
        errors=errors,
    )


async def _process_quick(
    *,
    work_id: str,
    transcript: str,
    payload: CapturePayload,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    """Delegate to quick capture processor."""
    from author_library.captures.quick import process_quick_capture

    return await process_quick_capture(
        work_id=work_id,
        transcript=transcript,
        timestamp_seconds=payload.timestamp_seconds,
        source_url=payload.source_url,
        source_title=payload.source_title,
        annotation=payload.annotation,
        speaker_override=payload.speaker_override,
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )


async def _process_deep(
    *,
    work_id: str,
    transcript: str,
    payload: CapturePayload,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    """Delegate to deep capture processor."""
    from author_library.captures.deep import process_deep_capture

    return await process_deep_capture(
        work_id=work_id,
        transcript=transcript,
        timestamp_seconds=payload.timestamp_seconds,
        source_url=payload.source_url,
        source_title=payload.source_title,
        annotation=payload.annotation,
        speaker_override=payload.speaker_override,
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )


async def _process_visual(
    *,
    work_id: str,
    transcript: str,
    payload: CapturePayload,
    capture_id: str,
    is_deep: bool,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    """Process a visual capture (visual_quick or visual_deep).

    Combines screenshot analysis with Quick or Deep transcript processing.
    """
    from author_library.captures.visual import analyze_screenshot, save_screenshot

    errors: list[str] = []

    # First, run the base transcript processing (Quick or Deep)
    if is_deep:
        base_result = await _process_deep(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
    else:
        base_result = await _process_quick(
            work_id=work_id,
            transcript=transcript,
            payload=payload,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

    errors.extend(base_result.get("errors", []))
    chunk_id = base_result.get("chunk_id")

    # Process screenshot if provided
    if payload.screenshot_base64 and chunk_id:
        # Analyze screenshot with vision model
        window_seconds = 30.0 if is_deep else 15.0
        window_text = extract_transcript_window(
            transcript,
            payload.timestamp_seconds,
            window_seconds=window_seconds,
        )

        visual_analysis = await analyze_screenshot(
            payload.screenshot_base64,
            transcript_window=window_text,
            source_title=payload.source_title,
            annotation=payload.annotation,
            settings=settings,
        )

        # Save screenshot to filesystem
        screenshot_path = save_screenshot(
            payload.screenshot_base64,
            capture_id=capture_id,
        )

        # Update chunk metadata with visual analysis
        if chunk_id:
            from uuid import UUID

            existing_chunk = await storage.chunks.get(UUID(chunk_id))
            if existing_chunk:
                metadata = existing_chunk.get("metadata", {})
                if isinstance(metadata, str):
                    import json

                    metadata = json.loads(metadata)
                metadata["visual_analysis"] = visual_analysis
                if screenshot_path:
                    metadata["screenshot_path"] = screenshot_path

                # Update the chunk's metadata — use the text column to append
                # visual context for richer retrieval
                visual_text = visual_analysis.get("context_synthesis", "")
                if visual_text:
                    updated_text = (
                        existing_chunk.get("text", "")
                        + f"\n\n[Visual context: {visual_text}]"
                    )
                    import json

                    await storage.pg.execute(
                        "UPDATE chunks SET metadata = $1, text = $2 WHERE id = $3",
                        json.dumps(metadata),
                        updated_text,
                        UUID(chunk_id),
                    )
    elif payload.screenshot_base64 and not chunk_id:
        errors.append("Screenshot provided but no chunk created for visual analysis")

    return {
        "chunk_id": chunk_id,
        "errors": errors,
    }
