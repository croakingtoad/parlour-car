"""O4: Open tension detector — identify contradictions in user's thinking.

Analyzes gathered Personal reflections to find places where the user
holds conflicting or evolving views. This feeds into the synthesis
to honestly represent the user's intellectual state, including
unresolved tensions and shifting positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.synthesis.gatherer import GatheredReflections, PersonalReflection

if TYPE_CHECKING:
    from author_library.config import Settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Tension:
    """A detected tension or contradiction in the user's thinking."""

    description: str
    reflection_a_id: str
    reflection_b_id: str
    tension_type: str  # "contradiction", "evolution", "uncertainty", "qualification"
    evidence_a: str
    evidence_b: str
    date_a: str
    date_b: str


@dataclass(frozen=True, slots=True)
class TensionAnalysis:
    """Complete tension analysis of a set of reflections."""

    tensions: list[Tension]
    reflection_count: int
    theme: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "tensions": [
                {
                    "description": t.description,
                    "reflection_a_id": t.reflection_a_id,
                    "reflection_b_id": t.reflection_b_id,
                    "tension_type": t.tension_type,
                    "evidence_a": t.evidence_a,
                    "evidence_b": t.evidence_b,
                    "date_a": t.date_a,
                    "date_b": t.date_b,
                }
                for t in self.tensions
            ],
            "tension_count": len(self.tensions),
            "reflection_count": self.reflection_count,
            "theme": self.theme,
        }


class TensionDetector:
    """Detects contradictions and evolving positions in reflections.

    Uses an LLM to compare pairs of reflections and identify where
    the user's thinking contains tensions, contradictions, or
    meaningful evolution over time.
    """

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    async def detect(
        self,
        gathered: GatheredReflections,
        *,
        theme: str = "",
    ) -> TensionAnalysis:
        """Detect tensions in gathered reflections.

        Args:
            gathered: The collected Personal reflections.
            theme: The theme being analyzed.

        Returns:
            TensionAnalysis with detected tensions.
        """
        reflections = gathered.reflections
        if len(reflections) < 2:
            return TensionAnalysis(
                tensions=[],
                reflection_count=len(reflections),
                theme=theme,
            )

        # Build prompt for tension detection
        system_prompt = self._build_system_prompt(theme)
        user_prompt = self._build_user_prompt(reflections, theme)

        # Call LLM
        raw_response = await self._call_llm(system_prompt, user_prompt)

        # Parse response
        tensions = self._parse_response(raw_response, reflections)

        return TensionAnalysis(
            tensions=tensions,
            reflection_count=len(reflections),
            theme=theme,
        )

    def _build_system_prompt(self, theme: str) -> str:
        """Build the system prompt for tension detection."""
        return f"""You are analyzing a reader's personal reflections to identify tensions, contradictions, and evolution in their thinking.

Look for:
1. CONTRADICTIONS: Places where the reader explicitly holds conflicting views
2. EVOLUTION: Where the reader's position has shifted over time (note dates)
3. UNCERTAINTY: Where the reader expresses doubt or qualification
4. QUALIFICATION: Where a later reflection qualifies or limits an earlier one

RULES:
1. Only identify genuine tensions — don't manufacture disagreement from compatible ideas.
2. Reference specific reflections by their [REF-N] markers.
3. Be specific about what the tension IS, not just that it exists.
4. Distinguish between contradiction (incompatible views) and evolution (changing views).

OUTPUT FORMAT:
List each tension as a block:

TENSION: [brief description]
TYPE: [contradiction|evolution|uncertainty|qualification]
REF_A: [N] — [short quote from reflection A]
REF_B: [N] — [short quote from reflection B]

If no tensions found, respond with: NO_TENSIONS_FOUND"""

    def _build_user_prompt(
        self,
        reflections: list[PersonalReflection],
        theme: str,
    ) -> str:
        """Build the user prompt with reflections."""
        parts: list[str] = []

        if theme:
            parts.append(f"Topic: {theme}\n")

        parts.append(f"Analyzing {len(reflections)} reflections for tensions:\n")

        for i, ref in enumerate(reflections, 1):
            date_str = ref.date_created[:10] if ref.date_created else "unknown"
            text = ref.text[:800].strip()
            if len(ref.text) > 800:
                text += "..."
            parts.append(f"[REF-{i}] ({date_str})\n{text}\n")

        return "\n".join(parts)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM for tension detection."""
        import anthropic

        api_key = self._settings.api_keys.anthropic_api_key.get_secret_value()
        model = self._settings.llm.query_model

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text

    def _parse_response(
        self,
        raw: str,
        reflections: list[PersonalReflection],
    ) -> list[Tension]:
        """Parse the LLM response into Tension objects."""
        if "NO_TENSIONS_FOUND" in raw:
            return []

        tensions: list[Tension] = []
        blocks = raw.split("TENSION:")

        for block in blocks[1:]:  # Skip the first (pre-TENSION) part
            tension = self._parse_tension_block(block.strip(), reflections)
            if tension:
                tensions.append(tension)

        return tensions

    def _parse_tension_block(
        self,
        block: str,
        reflections: list[PersonalReflection],
    ) -> Tension | None:
        """Parse a single tension block."""
        import re

        lines = block.strip().split("\n")
        if not lines:
            return None

        description = lines[0].strip()
        tension_type = "uncertainty"
        ref_a_num = 0
        ref_b_num = 0
        evidence_a = ""
        evidence_b = ""

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("TYPE:"):
                raw_type = line[5:].strip().lower()
                if raw_type in ("contradiction", "evolution", "uncertainty", "qualification"):
                    tension_type = raw_type
            elif line.startswith("REF_A:"):
                match = re.match(r"REF_A:\s*(\d+)\s*[—-]\s*(.*)", line)
                if match:
                    ref_a_num = int(match.group(1))
                    evidence_a = match.group(2).strip()
            elif line.startswith("REF_B:"):
                match = re.match(r"REF_B:\s*(\d+)\s*[—-]\s*(.*)", line)
                if match:
                    ref_b_num = int(match.group(1))
                    evidence_b = match.group(2).strip()

        # Resolve reflection IDs
        ref_a = reflections[ref_a_num - 1] if 0 < ref_a_num <= len(reflections) else None
        ref_b = reflections[ref_b_num - 1] if 0 < ref_b_num <= len(reflections) else None

        if not ref_a or not ref_b:
            return None

        return Tension(
            description=description,
            reflection_a_id=ref_a.chunk_id,
            reflection_b_id=ref_b.chunk_id,
            tension_type=tension_type,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            date_a=ref_a.date_created[:10] if ref_a.date_created else "",
            date_b=ref_b.date_created[:10] if ref_b.date_created else "",
        )
