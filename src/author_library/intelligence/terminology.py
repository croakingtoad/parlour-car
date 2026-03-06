"""Terminology normalization for author corpus.

Establishes a controlled vocabulary that maps variant terms to canonical
concepts. Authors often use different words for the same concept across
works, or the same word with shifting meaning over time. This module
uses LLM-assisted analysis to identify synonyms, related terms, and
establish canonical forms.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IntelligenceError

if TYPE_CHECKING:
    from author_library.config import Settings

from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Terminology data models
# ---------------------------------------------------------------------------


class TermMapping(BaseModel):
    """A mapping from a variant term to its canonical form."""

    variant: str
    canonical: str
    relationship: str = "synonym"  # synonym, narrower, broader, related
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None


class AmbiguousTerm(BaseModel):
    """A term flagged as ambiguous, requiring human review."""

    term: str
    possible_canonicals: list[str]
    context_examples: list[str] = Field(default_factory=list)
    reason: str


class TerminologyMap(BaseModel):
    """Complete terminology normalization output."""

    author_id: str
    mappings: list[TermMapping] = Field(default_factory=list)
    ambiguous: list[AmbiguousTerm] = Field(default_factory=list)
    canonical_terms: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TERMINOLOGY_SYSTEM = """\
You are a literary and theological terminology specialist. You analyze \
terms extracted from an author's corpus and identify synonyms, variant \
spellings, and related terms that should map to canonical concepts.

For each cluster of related terms, establish one canonical form and map \
all variants to it. Flag genuinely ambiguous cases where a term could \
map to multiple distinct concepts.

Respond with valid JSON matching this schema:
{
  "canonical_terms": ["<term 1>", "<term 2>", ...],
  "mappings": [
    {
      "variant": "<variant term as used in text>",
      "canonical": "<canonical form>",
      "relationship": "synonym" | "narrower" | "broader" | "related",
      "confidence": <float 0.0-1.0>,
      "notes": "<optional explanation>"
    },
    ...
  ],
  "ambiguous": [
    {
      "term": "<ambiguous term>",
      "possible_canonicals": ["<option 1>", "<option 2>"],
      "reason": "<why this is ambiguous>"
    },
    ...
  ]
}

Guidelines:
- Canonical forms should be the most common or most precise usage
- "synonym": interchangeable terms (e.g., "imagination" / "imaginative faculty")
- "narrower": more specific than canonical \
(e.g., "Secondary Imagination" narrower than "Imagination")
- "broader": more general than canonical
- "related": conceptually linked but not interchangeable
- Flag terms where the author uses the same word with genuinely different \
meanings in different contexts as ambiguous
- Confidence: 0.9+ for clear synonyms, 0.7-0.9 for likely matches, \
below 0.7 should be flagged as ambiguous\
"""


# ---------------------------------------------------------------------------
# Term extraction helpers
# ---------------------------------------------------------------------------

MAX_TERMS_PER_BATCH = 100


def _extract_terms_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract candidate terms from chunk texts.

    Identifies capitalized multi-word phrases, quoted terms, and
    recurring significant words that may represent domain concepts.
    """
    import re

    term_counts: dict[str, int] = {}

    for chunk in chunks:
        text = chunk.get("text", "")

        # Capitalized phrases (2-4 words, likely concept names)
        cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        for phrase in cap_phrases:
            # Skip common sentence-starting patterns
            if phrase.split()[0] in {"The", "This", "That", "These", "Those", "In", "On", "At"}:
                continue
            term_counts[phrase] = term_counts.get(phrase, 0) + 1

        # Quoted terms (single or double quotes, 1-5 words)
        quoted = re.findall(r"""['"]([A-Za-z][A-Za-z\s]{2,40})['"]""", text)
        for term in quoted:
            term = term.strip()
            if len(term.split()) <= 5:
                term_counts[term] = term_counts.get(term, 0) + 1

    # Return terms that appear at least twice, sorted by frequency
    significant = [
        term for term, count in term_counts.items() if count >= 2
    ]
    significant.sort(key=lambda t: term_counts[t], reverse=True)

    return significant


# ---------------------------------------------------------------------------
# Terminology normalizer
# ---------------------------------------------------------------------------


class TerminologyNormalizer:
    """Normalizes terminology across an author's corpus.

    Identifies variant terms for the same concepts and establishes
    canonical forms, flagging ambiguous cases for human review.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.api_keys.anthropic_api_key.get_secret_value()
        self._model = settings.llm.ingestion_model

    async def normalize(
        self,
        *,
        author_id: str,
        author_name: str,
        chunks: list[dict[str, Any]],
    ) -> TerminologyMap:
        """Build a terminology normalization map from corpus chunks.

        Args:
            author_id: The author's slug identifier.
            author_name: The author's canonical display name.
            chunks: Meso-level chunks from the author's primary corpus.

        Returns:
            A TerminologyMap with canonical terms, mappings, and ambiguous flags.

        Raises:
            IntelligenceError: If normalization fails.
        """
        if not self._api_key:
            raise IntelligenceError(
                "Anthropic API key is required for terminology normalization",
                context={"author_id": author_id},
            )

        # Extract candidate terms from corpus
        terms = _extract_terms_from_chunks(chunks)

        if not terms:
            log.warning(
                "no_terms_extracted",
                author_id=author_id,
                n_chunks=len(chunks),
            )
            return TerminologyMap(author_id=author_id)

        log.info(
            "normalizing_terminology",
            author_id=author_id,
            n_candidate_terms=len(terms),
            model=self._model,
        )

        # Process in batches
        all_mappings: list[TermMapping] = []
        all_ambiguous: list[AmbiguousTerm] = []
        all_canonical: set[str] = set()

        for batch_start in range(0, len(terms), MAX_TERMS_PER_BATCH):
            batch = terms[batch_start : batch_start + MAX_TERMS_PER_BATCH]

            # Include some chunk context for disambiguation
            context_chunks = chunks[:10]
            context_text = "\n\n".join(
                c.get("text", "")[:300] for c in context_chunks
            )

            result = await self._normalize_batch(
                author_name=author_name,
                terms=batch,
                context_text=context_text,
            )

            all_canonical.update(result.get("canonical_terms", []))

            for m in result.get("mappings", []):
                all_mappings.append(
                    TermMapping(
                        variant=m["variant"],
                        canonical=m["canonical"],
                        relationship=m.get("relationship", "synonym"),
                        confidence=m.get("confidence", 0.8),
                        notes=m.get("notes"),
                    )
                )

            for a in result.get("ambiguous", []):
                all_ambiguous.append(
                    AmbiguousTerm(
                        term=a["term"],
                        possible_canonicals=a.get("possible_canonicals", []),
                        reason=a.get("reason", ""),
                    )
                )

        terminology_map = TerminologyMap(
            author_id=author_id,
            mappings=all_mappings,
            ambiguous=all_ambiguous,
            canonical_terms=sorted(all_canonical),
        )

        log.info(
            "terminology_normalized",
            author_id=author_id,
            n_canonical=len(terminology_map.canonical_terms),
            n_mappings=len(terminology_map.mappings),
            n_ambiguous=len(terminology_map.ambiguous),
        )

        return terminology_map

    def lookup(self, term: str, terminology_map: TerminologyMap) -> str:
        """Look up the canonical form for a term.

        Returns the canonical term if a mapping exists, otherwise
        returns the original term unchanged.
        """
        term_lower = term.lower()
        for mapping in terminology_map.mappings:
            if mapping.variant.lower() == term_lower:
                return mapping.canonical
        return term

    async def _normalize_batch(
        self,
        *,
        author_name: str,
        terms: list[str],
        context_text: str,
    ) -> dict[str, Any]:
        """Normalize a batch of terms using LLM analysis."""
        user_prompt = (
            f"Author: {author_name}\n\n"
            f"Terms extracted from the author's corpus:\n"
            + "\n".join(f"  - {t}" for t in terms)
            + "\n\nContext samples from the corpus:\n"
            + context_text
            + "\n\nNormalize these terms into canonical concepts. "
            "Respond with JSON only."
        )

        return await self._call_anthropic(user_prompt)

    async def _call_anthropic(self, user_prompt: str) -> dict[str, Any]:
        """Call the Anthropic API and parse JSON response."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=TERMINOLOGY_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise IntelligenceError(
                f"Anthropic API error during terminology normalization: {exc}",
                context={"model": self._model, "status_code": getattr(exc, "status_code", None)},
                cause=exc,
            ) from exc

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text
                break

        if not response_text:
            raise IntelligenceError(
                "Empty response from Anthropic API during terminology normalization",
                context={"model": self._model},
            )

        return self._parse_json_response(response_text)

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse a JSON response from an LLM, handling common malformations."""
        from author_library.intelligence.json_parser import extract_json

        try:
            return extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise IntelligenceError(
                f"Failed to parse terminology response as JSON: {exc}",
                context={"response_text": response_text[:500]},
                cause=exc,
            ) from exc
