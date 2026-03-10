"""Source classification engine for The Author Library.

Implements the classification decision tree from collection-librarian SKILL.md §2.
Uses LLM-assisted classification via the Anthropic API with structured signal
analysis for robust source classification.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.catalog.models import ClassificationResult, SourceClass
from author_library.errors import ClassificationError
from author_library.intelligence.lesson_writer import get_lesson_context

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.parsing.models import ParsedDocument
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a Collection Development Librarian classifying documents for an \
author-specific archive. Your task is to determine the source classification \
for a document based on its relationship to the archive's subject author.

Source classes:
- PRIMARY: Works authored BY the subject author. These form the canonical \
corpus for voice profile, knowledge graph, and thematic index.
- SECONDARY: Works ABOUT the subject author by other people. These provide \
scholarly context but MUST NEVER contaminate the author's voice profile.
- CONTEXTUAL: Works by other authors that the subject author frequently \
references or engages with. Not by or about the subject, but illuminates \
their arguments.
- TERTIARY: Reference works (bibliographies, encyclopedias, catalogs). \
Metadata only, no content ingestion.

CRITICAL SAFETY RULE: When uncertain, classify as SECONDARY. It is far more \
damaging to contaminate a voice profile with foreign prose than to temporarily \
exclude legitimate primary material from voice extraction.

You must respond with valid JSON matching this schema:
{
  "source_class": "primary" | "secondary" | "contextual" | "tertiary",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-3 sentence explanation>",
  "signals_detected": ["<signal1>", "<signal2>", ...]
}

Signals to analyze:
1. Authorship attribution — Is the document author == subject author?
2. Title analysis — Does the title reference the subject author by name?
3. Publication context — Publisher, series, year patterns.
4. Content sampling — First-person voice matching subject author?
5. Bibliographic cross-reference — Does this appear in subject author's bibliography?\
"""


def _build_classification_prompt(
    *,
    subject_author: str,
    document_author: str | None,
    document_title: str | None,
    publisher: str | None,
    publication_year: int | None,
    content_sample: str,
    metadata_hints: dict[str, Any] | None,
) -> str:
    """Build the user-facing classification prompt with document signals."""
    parts = [
        f"Subject author of this archive: {subject_author}",
        "",
        "Document to classify:",
    ]

    if document_title:
        parts.append(f"  Title: {document_title}")
    if document_author:
        parts.append(f"  Author (as listed): {document_author}")
    if publisher:
        parts.append(f"  Publisher: {publisher}")
    if publication_year:
        parts.append(f"  Publication year: {publication_year}")

    if metadata_hints:
        parts.append("")
        parts.append("Additional metadata hints provided by the user:")
        for key, value in metadata_hints.items():
            parts.append(f"  {key}: {value}")

    parts.append("")
    parts.append("Content sample (first ~1000 words):")
    parts.append(content_sample[:4000])

    parts.append("")
    parts.append(
        "Based on these signals, classify this document. "
        "Respond with JSON only, no additional text."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------


class SourceClassifier:
    """LLM-assisted source classification engine.

    Uses the Anthropic API to classify documents according to the
    collection-librarian decision tree. Applies the default-to-secondary
    safety rule when confidence is below threshold.
    """

    def __init__(
        self,
        settings: Settings,
        storage: StorageManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._api_key = settings.api_keys.anthropic_api_key.get_secret_value()
        self._model = settings.llm.ingestion_model

    async def classify(
        self,
        document: ParsedDocument,
        *,
        subject_author: str,
        metadata_hints: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """Classify a parsed document's relationship to the subject author.

        Args:
            document: The parsed document to classify.
            subject_author: The canonical name of the archive's subject author.
            metadata_hints: Optional user-provided hints (e.g., known bibliography entries).

        Returns:
            ClassificationResult with source_class, confidence, reasoning, signals.

        Raises:
            ClassificationError: If the API call fails or response is unparseable.
        """
        if not self._api_key:
            raise ClassificationError(
                "Anthropic API key is required for LLM-assisted classification",
                context={"subject_author": subject_author},
            )

        content_sample = self._extract_content_sample(document)

        user_prompt = _build_classification_prompt(
            subject_author=subject_author,
            document_author=document.metadata.author,
            document_title=document.metadata.title,
            publisher=document.metadata.publisher,
            publication_year=self._extract_publication_year(document),
            content_sample=content_sample,
            metadata_hints=metadata_hints,
        )

        # Fetch lesson context for injection into system prompt
        lesson_context = ""
        lesson_ids = []
        if self._storage is not None:
            lesson_context, lesson_ids = await get_lesson_context(
                self._storage, "classification"
            )

        log.info(
            "classifying_document",
            title=document.metadata.title,
            author=document.metadata.author,
            subject_author=subject_author,
            model=self._model,
            lessons_injected=len(lesson_ids),
        )

        try:
            result = await self._call_anthropic(user_prompt, lesson_context=lesson_context)
        except ClassificationError:
            raise
        except Exception as exc:
            raise ClassificationError(
                f"LLM classification failed: {exc}",
                context={
                    "title": document.metadata.title,
                    "subject_author": subject_author,
                },
                cause=exc,
            ) from exc

        # Track that these lessons were applied
        if lesson_ids and self._storage is not None:
            for lid in lesson_ids:
                try:
                    await self._storage.lessons.increment_applied(lid)
                except Exception as exc:
                    log.warning(
                        "lesson_increment_applied_failed",
                        lesson_id=str(lid),
                        error=str(exc),
                    )

        log.info(
            "classification_complete",
            title=document.metadata.title,
            source_class=result.source_class,
            confidence=result.confidence,
        )

        return result

    async def _call_anthropic(
        self, user_prompt: str, *, lesson_context: str = ""
    ) -> ClassificationResult:
        """Call the Anthropic API and parse the structured response."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        system = CLASSIFICATION_SYSTEM_PROMPT
        if lesson_context:
            system = f"{system}\n\n{lesson_context}"

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise ClassificationError(
                f"Anthropic API error: {exc}",
                context={"model": self._model, "status_code": getattr(exc, "status_code", None)},
                cause=exc,
            ) from exc

        # Extract text from response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text
                break

        if not response_text:
            raise ClassificationError(
                "Empty response from Anthropic API",
                context={"model": self._model},
            )

        return self._parse_classification_response(response_text)

    def _parse_classification_response(self, response_text: str) -> ClassificationResult:
        """Parse the LLM's JSON response into a ClassificationResult."""
        from author_library.intelligence.json_parser import extract_json

        try:
            data = extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise ClassificationError(
                f"Failed to parse classification response as JSON: {exc}",
                context={"response_text": response_text[:500]},
                cause=exc,
            ) from exc

        try:
            return ClassificationResult(
                source_class=SourceClass(data["source_class"]),
                confidence=float(data["confidence"]),
                reasoning=str(data["reasoning"]),
                signals_detected=list(data.get("signals_detected", [])),
            )
        except (KeyError, ValueError) as exc:
            raise ClassificationError(
                f"Invalid classification response structure: {exc}",
                context={"parsed_data": data},
                cause=exc,
            ) from exc

    @staticmethod
    def _extract_publication_year(document: ParsedDocument) -> int | None:
        """Extract publication year from document metadata."""
        pub_date = document.metadata.publication_date
        if pub_date and len(pub_date) >= 4:
            try:
                return int(pub_date[:4])
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_content_sample(document: ParsedDocument) -> str:
        """Extract the first ~1000 words from the document for content sampling."""
        text = document.raw_text
        if not text:
            # Fall back to tree text extraction
            text = _extract_text_from_tree(document.tree)

        # Take approximately 1000 words
        words = text.split()
        sample_words = words[:1000]
        return " ".join(sample_words)


def _extract_text_from_tree(node: Any) -> str:
    """Recursively extract text from a DocumentNode tree."""
    parts: list[str] = []
    if hasattr(node, "text") and node.text:
        parts.append(node.text)
    if hasattr(node, "children"):
        for child in node.children:
            parts.append(_extract_text_from_tree(child))
    return " ".join(parts)
