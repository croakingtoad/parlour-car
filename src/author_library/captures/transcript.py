"""YouTube transcript retrieval and caching.

Fetches transcripts via the YouTube Captions API (timedtext endpoint),
caches them in the transcript_cache table, and provides transcript
window extraction for capture processing.

For V1: YouTube only. Whisper fallback is a future TODO.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import httpx
import structlog

if TYPE_CHECKING:
    from author_library.storage.repositories import TranscriptCacheRepository

log = structlog.get_logger(__name__)

# Default TTL for cached transcripts (24 hours)
_DEFAULT_TTL_SECONDS = 86400

# YouTube URL patterns
_YT_VIDEO_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)"
    r"([a-zA-Z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from a URL.

    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
    """
    match = _YT_VIDEO_RE.search(url)
    if match:
        return match.group(1)

    # Fallback: check query params
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    v_list = params.get("v")
    if v_list:
        return v_list[0]

    return None


async def fetch_youtube_transcript(
    video_id: str,
    *,
    lang: str = "en",
) -> str | None:
    """Fetch a YouTube transcript via the timedtext API.

    Tries manual captions first, then falls back to auto-generated.
    Returns the full transcript text with timestamps, or None if
    no captions are available.

    Args:
        video_id: YouTube video ID (11 characters).
        lang: Language code for captions (default: English).

    Returns:
        Transcript text with speaker labels and timestamps, or None.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Try to get the video page to extract caption track info
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            response = await client.get(video_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(
                "youtube_page_fetch_failed",
                video_id=video_id,
                error=str(exc),
            )
            return None

        page_content = response.text

        # Extract caption tracks from the page data
        caption_tracks = _extract_caption_tracks(page_content)
        if not caption_tracks:
            log.info("youtube_no_captions", video_id=video_id)
            return None

        # Find the best caption track: prefer manual, then auto-generated
        track_url = _select_caption_track(caption_tracks, lang=lang)
        if not track_url:
            log.info(
                "youtube_no_matching_captions",
                video_id=video_id,
                lang=lang,
                available=[t.get("languageCode") for t in caption_tracks],
            )
            return None

        # Fetch the actual transcript
        try:
            # Request in srv3 (JSON) format for structured data
            fetch_url = track_url
            if "&fmt=" not in fetch_url:
                fetch_url += "&fmt=srv3"
            transcript_response = await client.get(fetch_url)
            transcript_response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(
                "youtube_transcript_fetch_failed",
                video_id=video_id,
                error=str(exc),
            )
            return None

        transcript_text = _parse_transcript_response(
            transcript_response.text,
            content_type=transcript_response.headers.get("content-type", ""),
        )

        if transcript_text:
            log.info(
                "youtube_transcript_fetched",
                video_id=video_id,
                length=len(transcript_text),
                word_count=len(transcript_text.split()),
            )

        return transcript_text


def _extract_caption_tracks(page_html: str) -> list[dict[str, Any]]:
    """Extract caption track URLs from YouTube page HTML.

    YouTube embeds caption data in the page's JavaScript as
    'captionTracks' in the player response.
    """
    import json

    # Look for captionTracks in the ytInitialPlayerResponse
    pattern = r'"captionTracks"\s*:\s*(\[.*?\])'
    match = re.search(pattern, page_html)
    if not match:
        return []

    try:
        tracks = json.loads(match.group(1))
        return tracks  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return []


def _select_caption_track(
    tracks: list[dict[str, Any]],
    *,
    lang: str = "en",
) -> str | None:
    """Select the best caption track URL.

    Preference order:
    1. Manual captions in requested language
    2. Auto-generated captions in requested language
    3. Manual captions in any language
    4. Auto-generated captions in any language
    """
    manual_match = None
    auto_match = None
    manual_any = None
    auto_any = None

    for track in tracks:
        track_lang = track.get("languageCode", "")
        base_url = track.get("baseUrl", "")
        is_auto = track.get("kind") == "asr"

        if not base_url:
            continue

        if track_lang.startswith(lang):
            if is_auto:
                auto_match = base_url
            else:
                manual_match = base_url
        else:
            if is_auto:
                if auto_any is None:
                    auto_any = base_url
            elif manual_any is None:
                manual_any = base_url

    return manual_match or auto_match or manual_any or auto_any


def _parse_transcript_response(content: str, *, content_type: str) -> str | None:
    """Parse YouTube transcript response into text with timestamps.

    Handles both XML (srv3) and JSON formats.
    """
    if "xml" in content_type or content.strip().startswith("<?xml") or content.strip().startswith("<"):
        return _parse_xml_transcript(content)
    return _parse_json_transcript(content)


def _parse_xml_transcript(xml_content: str) -> str | None:
    """Parse XML (srv3) transcript format."""
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml_content)
    except ElementTree.ParseError:
        return None

    lines: list[str] = []
    # srv3 format uses <p> or <text> elements with t (start) and d (duration) attrs
    for elem in root.iter():
        if elem.tag in ("p", "text"):
            start_ms = int(elem.get("t", "0"))
            text = (elem.text or "").strip()
            if not text:
                # Some elements have text in child <s> elements
                text = "".join(
                    (s.text or "") for s in elem.iter() if s.text
                ).strip()

            if text:
                # Clean up HTML entities
                text = text.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
                timestamp = _format_seconds(start_ms / 1000)
                lines.append(f"[{timestamp}] {text}")

    return "\n".join(lines) if lines else None


def _parse_json_transcript(json_content: str) -> str | None:
    """Parse JSON transcript format (timedtext API v3)."""
    import json as json_module

    try:
        data = json_module.loads(json_content)
    except (json_module.JSONDecodeError, TypeError):
        return None

    events = data.get("events", [])
    lines: list[str] = []

    for event in events:
        start_ms = event.get("tStartMs", 0)
        segs = event.get("segs", [])
        text = "".join(seg.get("utf8", "") for seg in segs).strip()
        if text and text != "\n":
            timestamp = _format_seconds(start_ms / 1000)
            lines.append(f"[{timestamp}] {text}")

    return "\n".join(lines) if lines else None


def _format_seconds(total_seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    total = int(total_seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


async def get_transcript(
    source_url: str,
    *,
    cache_repo: TranscriptCacheRepository,
    lang: str = "en",
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str | None:
    """Get transcript for a source URL, using cache when available.

    On first access: fetches, parses, and caches the transcript.
    On subsequent access: returns the cached version.

    Args:
        source_url: URL of the video/audio source.
        cache_repo: Transcript cache repository.
        lang: Language preference for captions.
        ttl_seconds: Cache TTL in seconds.

    Returns:
        Transcript text with timestamps, or None if unavailable.
    """
    # Check cache first
    cached = await cache_repo.get_cached(source_url)
    if cached is not None:
        log.info("transcript_cache_hit", source_url=source_url)
        return cached

    # Extract video ID and fetch
    video_id = extract_video_id(source_url)
    if not video_id:
        log.warning("transcript_unsupported_url", source_url=source_url)
        return None

    transcript = await fetch_youtube_transcript(video_id, lang=lang)
    if transcript is None:
        log.info("transcript_not_available", source_url=source_url)
        return None

    # Cache the transcript
    await cache_repo.cache(source_url, transcript, ttl_seconds)
    log.info(
        "transcript_cached",
        source_url=source_url,
        length=len(transcript),
    )

    return transcript


def extract_transcript_window(
    transcript: str,
    timestamp_seconds: float,
    *,
    window_seconds: float = 15.0,
) -> str:
    """Extract a window of transcript text around a timestamp.

    Finds lines within [timestamp - window, timestamp + window] seconds.

    Args:
        transcript: Full transcript text with [MM:SS] or [HH:MM:SS] timestamps.
        timestamp_seconds: Center timestamp in seconds.
        window_seconds: Half-width of the extraction window.

    Returns:
        Extracted transcript text (may be empty if no lines match).
    """
    window_start = max(0, timestamp_seconds - window_seconds)
    window_end = timestamp_seconds + window_seconds

    lines = transcript.split("\n")
    window_lines: list[str] = []

    for line in lines:
        line_ts = _parse_line_timestamp(line)
        if line_ts is not None and window_start <= line_ts <= window_end:
            window_lines.append(line)

    return "\n".join(window_lines)


def _parse_line_timestamp(line: str) -> float | None:
    """Parse the timestamp from a transcript line like '[1:23] text'."""
    match = re.match(r"\[(\d+):(\d{2})(?::(\d{2}))?\]", line)
    if not match:
        return None

    parts = match.groups()
    if parts[2] is not None:
        # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    # MM:SS
    return int(parts[0]) * 60 + int(parts[1])
