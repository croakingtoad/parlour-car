"""Context assembly and voice calibration for LLM consumption.

Assembles retrieved chunks, graph context, and summaries into a coherent
context window with voice calibration system prompts. Manages token budget
and prioritizes by relevance score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from author_library.errors import RetrievalError
from author_library.retrieval.models import (
    ContextPassage,
    ContextWindow,
    GraphExpansionResult,
    RetrievalResult,
)

if TYPE_CHECKING:
    from author_library.intelligence.thematic_index import ThematicEntry
    from author_library.intelligence.voice_profile import VoiceProfile
    from author_library.retrieval.orchestrator import OrchestratedResult

log = structlog.get_logger(__name__)


# Rough token estimation: 1 token ~ 4 characters
CHARS_PER_TOKEN = 4

# Default context window budget (in tokens)
DEFAULT_TOKEN_BUDGET = 100_000

# Voice calibration system prompt template
VOICE_SYSTEM_TEMPLATE = """\
You are channeling the voice of {author_name}. Based on their voice profile:
- Register: {register}
- Characteristic patterns: {patterns}
- Example of their voice: {example_passage}

Respond to the user's question drawing ONLY from the provided passages.
Always cite your sources as [Work Title, Chapter].
When you reference material from a contextual source, clearly attribute it:
"In {{original_author}}'s {{work}}, which {author_name} engages with..."

Source classification labels:
- PRIMARY: Written by {author_name} — treat as the authoritative voice
- SECONDARY: Written about {author_name} — use for factual context only
- CONTEXTUAL: Works {author_name} engages with — always attribute to original author
- TERTIARY: Reference material — use only for factual verification

CRITICAL: Never present SECONDARY or CONTEXTUAL material as if {author_name} wrote it.
Always preserve the source_class label when citing any passage.\
"""


def build_voice_system_prompt(
    voice_profile: VoiceProfile,
    *,
    author_name: str,
) -> str:
    """Build the voice calibration system prompt from a VoiceProfile.

    Args:
        voice_profile: The author's extracted voice profile.
        author_name: Display name of the author.

    Returns:
        Formatted system prompt string.
    """
    patterns = "; ".join(voice_profile.sentence_patterns[:5])
    example = voice_profile.example_passages[0] if voice_profile.example_passages else ""

    return VOICE_SYSTEM_TEMPLATE.format(
        author_name=author_name,
        register=voice_profile.register,
        patterns=patterns,
        example_passage=example,
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _format_passage(
    text: str,
    *,
    work_id: str,
    source_class: str,
    source: str,
    relationship_info: str | None = None,
) -> str:
    """Format a passage with source-class label and citation metadata."""
    header = f"[{source_class.upper()}] [{work_id}]"
    if relationship_info:
        header += f" (via {relationship_info})"
    return f"{header}\n{text}"


def assemble_context(
    orchestrated: OrchestratedResult,
    *,
    voice_profile: VoiceProfile | None = None,
    author_name: str = "the author",
    thematic_entries: list[ThematicEntry] | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContextWindow:
    """Assemble retrieved results into a coherent context window.

    Prioritizes content by relevance score and manages token budget.
    Always includes voice profile and source classification labels.

    Args:
        orchestrated: Result from the RetrievalOrchestrator.
        voice_profile: Optional voice profile for calibration.
        author_name: Display name of the author.
        thematic_entries: Optional thematic summaries to include.
        token_budget: Maximum tokens for the context window.

    Returns:
        Assembled ContextWindow ready for LLM consumption.

    Raises:
        RetrievalError: If assembly fails.
    """
    # 1. Build voice system prompt
    if voice_profile is not None:
        system_prompt = build_voice_system_prompt(
            voice_profile, author_name=author_name
        )
        voice_text = _format_voice_profile_text(voice_profile)
    else:
        system_prompt = (
            f"Respond based on the provided passages about {author_name}. "
            "Always cite sources and respect source classification labels."
        )
        voice_text = ""

    # Reserve budget for system prompt and voice profile
    system_tokens = _estimate_tokens(system_prompt)
    voice_tokens = _estimate_tokens(voice_text)
    remaining_budget = token_budget - system_tokens - voice_tokens

    if remaining_budget <= 0:
        raise RetrievalError(
            "Token budget exhausted by system prompt and voice profile",
            context={"budget": token_budget, "system": system_tokens, "voice": voice_tokens},
        )

    # 2. Build thematic summaries
    thematic_summaries: list[str] = []
    if thematic_entries:
        for entry in thematic_entries:
            summary = _format_thematic_summary(entry)
            summary_tokens = _estimate_tokens(summary)
            if summary_tokens <= remaining_budget:
                thematic_summaries.append(summary)
                remaining_budget -= summary_tokens
            else:
                break  # No more budget for summaries

    # 3. Assemble passages in priority order
    passages: list[ContextPassage] = []
    total_passage_tokens = 0

    # Primary results first (highest priority)
    for result in orchestrated.primary_results:
        passage = _result_to_passage(result)
        p_tokens = _estimate_tokens(passage.text)
        if total_passage_tokens + p_tokens > remaining_budget:
            break
        passages.append(passage)
        total_passage_tokens += p_tokens

    # Graph expansions second (context enrichment)
    for expansion in orchestrated.graph_expansions:
        passage = _expansion_to_passage(expansion)
        p_tokens = _estimate_tokens(passage.text)
        if total_passage_tokens + p_tokens > remaining_budget:
            break
        passages.append(passage)
        total_passage_tokens += p_tokens

    # Supporting evidence last (micro-chunk quotes)
    for result in orchestrated.supporting_evidence:
        passage = _result_to_passage(result, source_label="supporting_quote")
        p_tokens = _estimate_tokens(passage.text)
        if total_passage_tokens + p_tokens > remaining_budget:
            break
        passages.append(passage)
        total_passage_tokens += p_tokens

    total_tokens = system_tokens + voice_tokens + total_passage_tokens
    for s in thematic_summaries:
        total_tokens += _estimate_tokens(s)

    log.info(
        "context_assembled",
        passages=len(passages),
        thematic_summaries=len(thematic_summaries),
        total_tokens_estimate=total_tokens,
        token_budget=token_budget,
    )

    return ContextWindow(
        voice_profile_text=voice_text,
        system_prompt=system_prompt,
        passages=passages,
        thematic_summaries=thematic_summaries,
        total_tokens_estimate=total_tokens,
        token_budget=token_budget,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_voice_profile_text(profile: VoiceProfile) -> str:
    """Format a voice profile as readable text for the context window."""
    parts = [
        f"Voice Profile for {profile.author_id}:",
        f"  Register: {profile.register}",
        f"  Sentence patterns: {'; '.join(profile.sentence_patterns[:5])}",
        f"  Vocabulary tendencies: {'; '.join(profile.vocabulary_tendencies[:5])}",
        f"  Rhetorical moves: {'; '.join(profile.rhetorical_moves[:5])}",
        f"  Characteristic phrases: {'; '.join(profile.characteristic_phrases[:5])}",
    ]
    if profile.humor_style:
        parts.append(f"  Humor style: {profile.humor_style}")
    parts.append(f"  Confidence: {profile.confidence:.2f}")
    return "\n".join(parts)


def _format_thematic_summary(entry: ThematicEntry) -> str:
    """Format a thematic entry as a summary string."""
    parts = [f"Theme: {entry.theme}", f"  Author's stance: {entry.author_stance}"]
    if entry.appearances:
        for appearance in entry.appearances[:3]:
            parts.append(f"  In {appearance.work_id}: {appearance.treatment_summary}")
    if entry.related_themes:
        parts.append(f"  Related themes: {', '.join(entry.related_themes[:5])}")
    return "\n".join(parts)


def _result_to_passage(
    result: RetrievalResult,
    source_label: str | None = None,
) -> ContextPassage:
    """Convert a RetrievalResult to a ContextPassage with citation."""
    label = source_label or result.source
    formatted_text = _format_passage(
        result.text,
        work_id=result.work_id,
        source_class=result.source_class,
        source=label,
    )
    citation = f"[{result.work_id}]"

    return ContextPassage(
        text=formatted_text,
        work_id=result.work_id,
        source_class=result.source_class,
        relevance_score=result.score,
        citation_label=citation,
        source=label,
    )


def _expansion_to_passage(expansion: GraphExpansionResult) -> ContextPassage:
    """Convert a GraphExpansionResult to a ContextPassage."""
    relationship_info = (
        f"{expansion.relationship_type}, confidence={expansion.confidence}"
    )
    formatted_text = _format_passage(
        expansion.text_preview,
        work_id=expansion.work_id,
        source_class=expansion.source_class,
        source="graph",
        relationship_info=relationship_info,
    )
    citation = f"[{expansion.work_id}] (via {expansion.relationship_type})"

    return ContextPassage(
        text=formatted_text,
        work_id=expansion.work_id,
        source_class=expansion.source_class,
        relevance_score=0.5,  # Graph expansions get a baseline relevance
        citation_label=citation,
        source="graph",
    )
