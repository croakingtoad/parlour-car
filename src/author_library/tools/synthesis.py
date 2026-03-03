"""O5: synthesize_my_thinking MCP tool implementation.

Orchestrates the full synthesis pipeline:
  1. Gather Personal reflections (O1)
  2. Draft position statement via LLM (O2)
  3. Enrich citations with provenance (O3)
  4. Detect open tensions (O4)
  5. Return structured result

CRITICAL RULES:
  - Only user's words become Personal data
  - AI/LLM dialogue is NEVER stored as Personal source class
  - Synthesis delivered as proposal for user review
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.synthesis.citation import CitationEnricher
from author_library.synthesis.gatherer import PersonalReflectionGatherer
from author_library.synthesis.prompt_engine import SynthesisPromptEngine
from author_library.synthesis.tension_detector import TensionDetector

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def handle_synthesize_my_thinking(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the synthesize_my_thinking MCP tool call.

    Orchestrates the full synthesis pipeline from gathering reflections
    through drafting, citation enrichment, and tension detection.

    Args:
        arguments: Tool input parameters.
        settings: Application settings.
        storage: Storage manager.
        embedding_provider: Embedding provider.
        cache_manager: Optional cache manager.

    Returns:
        JSON string with synthesis result.
    """
    theme = arguments.get("theme", "")
    speaker = arguments.get("speaker")
    date_range = arguments.get("date_range") or {}
    date_after = date_range.get("after")
    date_before = date_range.get("before")
    prompt = arguments.get("prompt", "")

    if not any([theme, speaker, date_after, prompt]):
        return json.dumps({
            "error": (
                "At least one of theme, speaker, date_range, or prompt "
                "is required to guide the synthesis."
            ),
        })

    # Step 1: Gather Personal reflections (O1)
    gatherer = PersonalReflectionGatherer(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
        cache_manager=cache_manager,
    )

    gathered = await gatherer.gather(
        theme=theme or None,
        speaker=speaker,
        date_after=date_after,
        date_before=date_before,
        prompt=prompt or None,
    )

    if not gathered.reflections:
        return json.dumps({
            "synthesis": "",
            "sources_used": [],
            "confidence": "tentative",
            "open_tensions": [],
            "reflection_count": 0,
            "message": "No Personal reflections found matching the criteria.",
        })

    if gathered.total_found < 3:
        log.info(
            "synthesis_few_reflections",
            count=gathered.total_found,
            theme=theme,
        )

    # Step 2: Draft position statement (O2)
    engine = SynthesisPromptEngine(settings=settings)
    synthesis_result = await engine.synthesize(
        gathered,
        theme=theme,
        prompt=prompt,
    )

    # Step 3: Enrich citations (O3)
    enricher = CitationEnricher(storage=storage)
    citation_report = await enricher.enrich(synthesis_result, gathered.reflections)

    # Step 4: Detect open tensions (O4)
    tension_detector = TensionDetector(settings=settings)
    tension_analysis = await tension_detector.detect(gathered, theme=theme)

    # Merge tension results — prefer LLM-detected tensions from O4 if available,
    # fall back to synthesis result tensions from O2
    open_tensions = (
        [t.description for t in tension_analysis.tensions]
        if tension_analysis.tensions
        else synthesis_result.open_tensions
    )

    # Build response matching Design-Specifications §3.5
    result: dict[str, Any] = {
        "synthesis": synthesis_result.synthesis,
        "sources_used": [
            {
                "capture_id": c.capture_id,
                "note_path": c.note_path,
                "excerpt": c.excerpt,
                "date": c.date,
            }
            for c in citation_report.citations
        ],
        "confidence": synthesis_result.confidence.value,
        "open_tensions": open_tensions,
        "reflection_count": gathered.total_found,
        "theme_counts": gathered.theme_counts,
        "date_range": list(gathered.date_range) if gathered.date_range else None,
        "citation_verification": {
            "verified": citation_report.verified_count,
            "unverified": citation_report.unverified_count,
            "unique_works": citation_report.unique_works,
        },
    }

    return json.dumps(result, indent=2)
