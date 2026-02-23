"""O2: Synthesis prompt engineering.

Drafts a position statement from the user's scattered Personal reflections.
Uses an LLM to weave reflections together into a coherent synthesis, identifying:
  - Central positions the user holds
  - Evidence from specific captures
  - Open tensions and contradictions
  - Confidence level (tentative / developing / coherent)

CRITICAL RULES:
  - Only the user's own words appear as quoted evidence
  - AI/LLM dialogue is NEVER stored as Personal source data
  - Synthesis is proposed, never imposed — delivered as PR for user review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from author_library.synthesis.gatherer import GatheredReflections, PersonalReflection

if TYPE_CHECKING:
    from author_library.config import Settings

log = structlog.get_logger(__name__)


class SynthesisConfidence(StrEnum):
    """How coherent the user's position appears across reflections."""

    TENTATIVE = "tentative"
    DEVELOPING = "developing"
    COHERENT = "coherent"


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A reference to a specific Personal reflection used in synthesis."""

    capture_id: str
    note_path: str
    excerpt: str
    date: str


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """The output of a synthesis operation."""

    synthesis: str
    sources_used: list[SourceCitation]
    confidence: SynthesisConfidence
    open_tensions: list[str]
    theme: str
    prompt: str
    reflection_count: int
    date_range: tuple[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/MCP output."""
        return {
            "synthesis": self.synthesis,
            "sources_used": [
                {
                    "capture_id": s.capture_id,
                    "note_path": s.note_path,
                    "excerpt": s.excerpt,
                    "date": s.date,
                }
                for s in self.sources_used
            ],
            "confidence": self.confidence.value,
            "open_tensions": self.open_tensions,
            "theme": self.theme,
            "prompt": self.prompt,
            "reflection_count": self.reflection_count,
            "date_range": list(self.date_range) if self.date_range else None,
        }


class SynthesisPromptEngine:
    """Drafts position statements from Personal reflections via LLM.

    Takes gathered reflections and constructs a prompt for the LLM to
    synthesize them into a coherent position statement. The synthesis
    identifies what the user thinks, how their thinking has evolved,
    and where tensions remain.
    """

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    async def synthesize(
        self,
        gathered: GatheredReflections,
        *,
        theme: str = "",
        prompt: str = "",
    ) -> SynthesisResult:
        """Draft a position statement from gathered reflections.

        Args:
            gathered: The collected Personal reflections.
            theme: The theme being synthesized.
            prompt: The user's framing question.

        Returns:
            SynthesisResult with drafted synthesis and citations.
        """
        if not gathered.reflections:
            return SynthesisResult(
                synthesis="",
                sources_used=[],
                confidence=SynthesisConfidence.TENTATIVE,
                open_tensions=[],
                theme=theme,
                prompt=prompt,
                reflection_count=0,
            )

        # Build the prompt
        system_prompt = self._build_system_prompt(theme, prompt)
        user_prompt = self._build_user_prompt(gathered.reflections, theme, prompt)

        # Call the LLM
        raw_response = await self._call_llm(system_prompt, user_prompt)

        # Parse the response
        return self._parse_response(
            raw_response,
            reflections=gathered.reflections,
            theme=theme,
            prompt=prompt,
            date_range=gathered.date_range,
        )

    def _build_system_prompt(self, theme: str, prompt: str) -> str:
        """Build the system prompt for the synthesis LLM call."""
        return f"""You are a thinking partner helping a reader understand their own intellectual development.

You will be given a collection of the reader's personal reflections — their own words captured while engaging with various sources over time. Your task is to synthesize these scattered reflections into a coherent position statement.

RULES:
1. Only quote the reader's own words. Never fabricate quotes.
2. Cite specific reflections by their [REF-N] markers.
3. Identify the reader's central position on the topic.
4. Note how their thinking has evolved chronologically if dates span a meaningful period.
5. Explicitly call out tensions or contradictions — places where the reader seems to hold conflicting views.
6. Assign a confidence level:
   - "tentative": Few reflections, early-stage thinking, no clear position yet
   - "developing": Multiple reflections showing direction but not yet resolved
   - "coherent": Clear, consistent position across multiple reflections over time
7. Write in second person ("You appear to think..." not "The user thinks...").
8. Be honest about uncertainty. If the reflections don't clearly support a position, say so.

OUTPUT FORMAT:
Respond with exactly these sections, using these exact headers:

## SYNTHESIS
[Your drafted position statement — 2-5 paragraphs of freeform prose weaving the reflections together]

## SOURCES
[List each cited source as: REF-N: "excerpt" (date)]

## CONFIDENCE
[One of: tentative, developing, coherent]

## OPEN_TENSIONS
[Bulleted list of unresolved contradictions or tensions. If none, write "None identified."]"""

    def _build_user_prompt(
        self,
        reflections: list[PersonalReflection],
        theme: str,
        prompt: str,
    ) -> str:
        """Build the user prompt containing all reflections."""
        parts: list[str] = []

        if prompt:
            parts.append(f"The reader asks: \"{prompt}\"\n")
        if theme:
            parts.append(f"Topic: {theme}\n")

        parts.append(f"The reader has {len(reflections)} personal reflections on this topic:\n")

        for i, ref in enumerate(reflections, 1):
            date_str = ref.date_created[:10] if ref.date_created else "unknown date"
            source = ref.source_note or ref.work_id
            section = ref.section_type

            parts.append(f"[REF-{i}] ({date_str}, {section}, from: {source})")
            # Truncate very long reflections to stay within token limits
            text = ref.text[:1500].strip()
            if len(ref.text) > 1500:
                text += "..."
            parts.append(f"{text}\n")

        return "\n".join(parts)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM to generate the synthesis."""
        import anthropic

        api_key = self._settings.api_keys.anthropic_api_key.get_secret_value()
        model = self._settings.llm.query_model

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text

    def _parse_response(
        self,
        raw: str,
        *,
        reflections: list[PersonalReflection],
        theme: str,
        prompt: str,
        date_range: tuple[str, str] | None,
    ) -> SynthesisResult:
        """Parse the LLM's structured response into a SynthesisResult."""
        synthesis = ""
        confidence = SynthesisConfidence.TENTATIVE
        open_tensions: list[str] = []
        sources_used: list[SourceCitation] = []

        # Parse sections
        sections = _split_sections(raw)

        synthesis = sections.get("SYNTHESIS", "").strip()

        # Parse confidence
        raw_confidence = sections.get("CONFIDENCE", "").strip().lower()
        if "coherent" in raw_confidence:
            confidence = SynthesisConfidence.COHERENT
        elif "developing" in raw_confidence:
            confidence = SynthesisConfidence.DEVELOPING
        else:
            confidence = SynthesisConfidence.TENTATIVE

        # Parse open tensions
        tensions_text = sections.get("OPEN_TENSIONS", "")
        if tensions_text and "none identified" not in tensions_text.lower():
            for line in tensions_text.strip().split("\n"):
                line = line.strip().lstrip("-•* ")
                if line:
                    open_tensions.append(line)

        # Build source citations from the REF markers referenced in synthesis
        sources_used = _extract_citations(synthesis, reflections)

        return SynthesisResult(
            synthesis=synthesis,
            sources_used=sources_used,
            confidence=confidence,
            open_tensions=open_tensions,
            theme=theme,
            prompt=prompt,
            reflection_count=len(reflections),
            date_range=date_range,
        )


def _split_sections(raw: str) -> dict[str, str]:
    """Split the LLM response into named sections."""
    sections: dict[str, str] = {}
    current_section = ""
    current_lines: list[str] = []

    for line in raw.split("\n"):
        stripped = line.strip()
        # Detect section headers
        if stripped.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = stripped[3:].strip().upper()
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines)

    return sections


def _extract_citations(
    synthesis: str,
    reflections: list[PersonalReflection],
) -> list[SourceCitation]:
    """Extract source citations from REF markers in the synthesis text."""
    import re

    citations: list[SourceCitation] = []
    seen_refs: set[int] = set()

    # Find all REF-N markers in the synthesis
    for match in re.finditer(r"\[REF-(\d+)\]", synthesis):
        ref_num = int(match.group(1))
        if ref_num in seen_refs:
            continue
        seen_refs.add(ref_num)

        idx = ref_num - 1  # Convert 1-based to 0-based
        if 0 <= idx < len(reflections):
            ref = reflections[idx]
            # Build a short excerpt (first 200 chars)
            excerpt = ref.text[:200].strip()
            if len(ref.text) > 200:
                excerpt += "..."

            citations.append(SourceCitation(
                capture_id=ref.chunk_id,
                note_path=ref.source_note or ref.work_id,
                excerpt=excerpt,
                date=ref.date_created[:10] if ref.date_created else "",
            ))

    return citations
