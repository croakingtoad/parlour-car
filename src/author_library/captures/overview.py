"""Source overview generation for first captures from a new URL.

On the first capture from a new source URL:
1. Retrieves/caches the full transcript
2. Generates an overview via LLM (title, speakers, content type, topic summary, structural arc)
3. Creates a source record in the works table and graph
4. Creates/updates speaker notes for detected speakers
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.captures.models import SourceOverview
from author_library.captures.transcript import extract_video_id, get_transcript

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def generate_source_overview(
    source_url: str,
    source_title: str,
    *,
    settings: Settings,
    storage: StorageManager,
) -> SourceOverview | None:
    """Generate an overview for a new source URL.

    Called on the first capture from a source. Fetches the full transcript,
    generates an LLM overview, and creates the source record in the database.

    Args:
        source_url: URL of the video/audio source.
        source_title: Title from the Chrome extension.
        settings: Application settings.
        storage: Storage manager for database access.

    Returns:
        SourceOverview with generated metadata, or None if transcript unavailable.
    """
    # Get (or fetch + cache) the transcript
    transcript = await get_transcript(
        source_url,
        cache_repo=storage.transcript_cache,
    )
    if not transcript:
        log.warning("overview_no_transcript", source_url=source_url)
        return None

    # Generate overview via LLM
    overview = await _generate_overview_llm(
        transcript=transcript,
        source_title=source_title,
        source_url=source_url,
        settings=settings,
    )

    # Create work record
    video_id = extract_video_id(source_url)
    work_id = f"video--{video_id}" if video_id else f"media--{hash(source_url) & 0xFFFFFFFF:08x}"

    work_data: dict[str, Any] = {
        "work_id": work_id,
        "title": overview.title or source_title,
        "author": ", ".join(overview.speakers) if overview.speakers else "Unknown",
        "source_class": "contextual",
        "source_class_note": f"Video source: {source_url}",
        "publication_year": None,
        "publisher": "YouTube" if video_id else "",
        "format_ingested": "video-transcript",
        "word_count": len(transcript.split()),
        "genre_tags": ["transcript", "youtube-captions"],
        "subject_headings": [],
        "url": source_url,
        "duration": None,
        "speakers": overview.speakers,
        "transcript_cached": True,
        "media": "video",
        "source_metadata": json.dumps({
            "content_type": overview.content_type,
            "topic_summary": overview.topic_summary,
            "structural_arc": overview.structural_arc,
            "video_id": video_id,
        }),
    }

    # Check if work already exists
    existing = await storage.works.get(work_id)
    if existing:
        log.info("overview_work_exists", work_id=work_id)
    else:
        await storage.works.create(work_data)
        log.info("overview_work_created", work_id=work_id, title=overview.title)

        # Upsert work node in Neo4j
        await storage.graph.upsert_work_node({
            "work_id": work_id,
            "title": overview.title or source_title,
            "author": work_data["author"],
            "source_class": "contextual",
            "publication_year": None,
        })

    return overview


async def _generate_overview_llm(
    transcript: str,
    source_title: str,
    source_url: str,
    *,
    settings: Settings,
) -> SourceOverview:
    """Use LLM to generate a source overview from the transcript.

    Analyzes the full transcript to produce:
    - Corrected/enriched title
    - List of speakers
    - Content type classification
    - Topic summary
    - Structural arc description
    """
    import anthropic

    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Truncate very long transcripts to fit context
    max_transcript_chars = 50000
    transcript_sample = transcript[:max_transcript_chars]
    if len(transcript) > max_transcript_chars:
        transcript_sample += f"\n\n[... truncated, total {len(transcript)} chars]"

    system_prompt = (
        "You are analyzing a video/audio transcript to generate a structured overview. "
        "Respond with a JSON object containing these fields:\n"
        '- "title": string — the best title for this content\n'
        '- "speakers": array of strings — names of speakers identified in the transcript\n'
        '- "content_type": string — one of: lecture, interview, conversation, panel, '
        "sermon, podcast, documentary, tutorial, debate, other\n"
        '- "topic_summary": string — 2-3 sentence summary of the main topics covered\n'
        '- "structural_arc": string — brief description of the content structure '
        "(e.g., introduction, main arguments, conclusion)\n\n"
        "Respond ONLY with the JSON object, no other text."
    )

    user_content = (
        f"Source URL: {source_url}\n"
        f"Page title: {source_title}\n\n"
        f"Transcript:\n{transcript_sample}"
    )

    try:
        response = await client.messages.create(
            model=settings.llm.ingestion_model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text if response.content else ""
        if not response_text:
            raise ValueError("Empty response from Anthropic API")

        # Parse JSON response
        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        data = json.loads(cleaned)

        return SourceOverview(
            source_url=source_url,
            title=data.get("title", source_title),
            speakers=data.get("speakers", []),
            content_type=data.get("content_type", "other"),
            topic_summary=data.get("topic_summary", ""),
            structural_arc=data.get("structural_arc", ""),
        )

    except Exception as exc:
        log.error(
            "overview_llm_failed",
            source_url=source_url,
            error=str(exc),
        )
        # Return a minimal overview on failure
        return SourceOverview(
            source_url=source_url,
            title=source_title,
            speakers=[],
            content_type="other",
            topic_summary="Overview generation failed",
            structural_arc="",
        )
