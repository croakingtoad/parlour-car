"""Quick capture processing.

Extracts a ±15s transcript window around the capture timestamp,
generates a synopsis (2-3 sentences) and theme tags from the
controlled vocabulary, creates a micro-granularity chunk in the
graph, and creates passage links to the existing corpus.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.captures.transcript import extract_transcript_window

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# Quick capture window: ±15 seconds
_QUICK_WINDOW_SECONDS = 15.0


async def process_quick_capture(
    *,
    work_id: str,
    transcript: str,
    timestamp_seconds: float,
    source_url: str,
    source_title: str,
    annotation: str | None,
    speaker_override: str | None,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    """Process a quick capture.

    1. Extract ±15s transcript window
    2. Generate synopsis + theme tags via LLM
    3. Create micro chunk in PostgreSQL
    4. Create chunk node in Neo4j
    5. Embed the chunk
    6. Create passage links to existing corpus

    Args:
        work_id: The work ID for the source.
        transcript: Full transcript text.
        timestamp_seconds: Capture timestamp in seconds.
        source_url: Source URL.
        source_title: Source title.
        annotation: Optional user annotation.
        speaker_override: Optional speaker name override.
        settings: Application settings.
        storage: Storage manager.
        embedding_provider: Embedding provider.

    Returns:
        dict with chunk_id, synopsis, tags, and errors.
    """
    errors: list[str] = []

    # 1. Extract transcript window
    window_text = extract_transcript_window(
        transcript,
        timestamp_seconds,
        window_seconds=_QUICK_WINDOW_SECONDS,
    )

    if not window_text.strip():
        errors.append(f"No transcript text found around {timestamp_seconds}s")
        return {
            "chunk_id": None,
            "synopsis": None,
            "tags": [],
            "errors": errors,
        }

    # 2. Generate synopsis + theme tags via LLM
    analysis = await _generate_quick_analysis(
        window_text=window_text,
        source_title=source_title,
        annotation=annotation,
        settings=settings,
    )

    # 3. Build chunk text
    chunk_text = window_text
    if annotation:
        chunk_text = f"[User note: {annotation}]\n\n{chunk_text}"

    # Determine speaker
    speaker = speaker_override or _detect_speaker(window_text)

    # 4. Create micro chunk in PostgreSQL
    chunk_data: dict[str, Any] = {
        "work_id": work_id,
        "text": chunk_text,
        "annotation": analysis.get("synopsis", ""),
        "granularity": "micro",
        "source_class": "contextual",
        "position": int(timestamp_seconds),
        "metadata": {
            "genre": "transcript",
            "capture_mode": "quick",
            "timestamp_seconds": timestamp_seconds,
            "source_url": source_url,
            "tags": analysis.get("tags", []),
        },
        "raw_content": window_text,
        "raw_content_window": f"±{_QUICK_WINDOW_SECONDS}s around {timestamp_seconds}s",
        "pass_number": 1,
    }
    if speaker:
        chunk_data["metadata"]["speaker"] = speaker

    chunk_id = await storage.chunks.create(chunk_data)
    log.info(
        "quick_capture_chunk_created",
        chunk_id=str(chunk_id),
        work_id=work_id,
        timestamp=timestamp_seconds,
    )

    # 5. Create chunk node in Neo4j
    await storage.graph.upsert_chunk_node({
        "chunk_id": str(chunk_id),
        "work_id": work_id,
        "text_preview": chunk_text[:200],
        "granularity": "micro",
        "source_class": "contextual",
    })

    # 6. Embed the chunk
    try:
        embed_text = f"{analysis.get('synopsis', '')}\n\n{chunk_text}"
        result = await embedding_provider.embed_batch([embed_text])
        if result.vectors:
            await storage.embeddings.store(
                chunk_id,
                result.vectors[0],
                embedding_provider.provider_name,
                embedding_provider.model_name,
                embedding_provider.dimensions,
            )
    except Exception as exc:
        error_msg = f"Embedding failed: {exc}"
        log.error("quick_capture_embedding_failed", error=error_msg)
        errors.append(error_msg)

    return {
        "chunk_id": str(chunk_id),
        "synopsis": analysis.get("synopsis"),
        "tags": analysis.get("tags", []),
        "speaker": speaker,
        "errors": errors,
    }


async def _generate_quick_analysis(
    *,
    window_text: str,
    source_title: str,
    annotation: str | None,
    settings: Settings,
) -> dict[str, Any]:
    """Generate synopsis and theme tags for a quick capture via LLM."""
    import anthropic

    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    system_prompt = (
        "You are analyzing a brief excerpt from a video/audio transcript. "
        "Generate a concise analysis. Respond with a JSON object:\n"
        '- "synopsis": string — 2-3 sentence summary of what is being discussed\n'
        '- "tags": array of strings — 2-5 theme tags from this controlled vocabulary: '
        "theology, philosophy, literature, apologetics, ethics, imagination, "
        "sacramental, myth, reason, faith, suffering, joy, friendship, education, "
        "culture, science, nature, prayer, conversion, scripture, tradition, "
        "modernity, romanticism, medieval, classical\n\n"
        "Respond ONLY with the JSON object."
    )

    user_content = f"Source: {source_title}\n"
    if annotation:
        user_content += f"User note: {annotation}\n"
    user_content += f"\nTranscript excerpt:\n{window_text}"

    try:
        response = await client.messages.create(
            model=settings.llm.ingestion_model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text if response.content else ""
        if not response_text:
            return {"synopsis": "", "tags": []}

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        return json.loads(cleaned)

    except Exception as exc:
        log.error("quick_analysis_llm_failed", error=str(exc))
        return {"synopsis": "", "tags": []}


def _detect_speaker(window_text: str) -> str | None:
    """Detect the primary speaker in a transcript window."""
    import re

    speaker_re = re.compile(r"^(?:\[)?([A-Z][A-Za-z\s.'-]+?)(?:\])?\s*:", re.MULTILINE)
    speakers: dict[str, int] = {}
    for match in speaker_re.finditer(window_text):
        name = match.group(1).strip()
        speakers[name] = speakers.get(name, 0) + 1

    if not speakers:
        return None

    # Return the most frequent speaker
    return max(speakers, key=lambda s: speakers[s])
