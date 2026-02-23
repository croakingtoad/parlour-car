"""M2: Confidence scoring for surfaced connections.

Classifies connection strength into three tiers:
  - High: Explicit citation, exact term match, passage link with confidence > 0.8
  - Medium: Implicit engagement, thematic overlap, confidence 0.5-0.8
  - Low: Thematic parallel, weak semantic similarity, confidence < 0.5

Each tier carries a presentation label that guides how Claude presents
the connection to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from author_library.surfacing.related_content import ConnectionType, RelatedItem


class ConfidenceLevel(StrEnum):
    """Connection confidence tiers."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Presentation labels per PRD §3
CONFIDENCE_LABELS: dict[ConfidenceLevel, str] = {
    ConfidenceLevel.HIGH: "This directly engages with",
    ConfidenceLevel.MEDIUM: "This appears to connect to",
    ConfidenceLevel.LOW: "You might find this relevant",
}

# Extended label variants for richer presentation
CONFIDENCE_LABEL_VARIANTS: dict[ConfidenceLevel, list[str]] = {
    ConfidenceLevel.HIGH: [
        "This directly engages with",
        "This explicitly references",
        "This is closely tied to",
    ],
    ConfidenceLevel.MEDIUM: [
        "This appears to connect to",
        "This seems related to",
        "This likely connects to",
    ],
    ConfidenceLevel.LOW: [
        "You might find this relevant",
        "This may be tangentially related",
        "You might see a connection here",
    ],
}


@dataclass(frozen=True, slots=True)
class ScoredConnection:
    """A connection with classified confidence and presentation label."""

    item: RelatedItem
    confidence_level: ConfidenceLevel
    label: str
    raw_score: float


def classify_confidence(item: RelatedItem) -> ScoredConnection:
    """Classify the confidence level of a related item.

    The classification uses the relevance_score combined with the
    connection_type to determine the appropriate confidence tier.

    Scoring logic:
    - Passage links with high confidence → HIGH
    - Passage links with medium confidence → MEDIUM
    - Direct personal reflections → HIGH (user's own reflection is always relevant)
    - Thematic parallels → MEDIUM or LOW depending on score
    - Vector similarity → scored by threshold
    - Everything else → scored by threshold

    Args:
        item: The RelatedItem to classify.

    Returns:
        ScoredConnection with confidence level and presentation label.
    """
    from author_library.surfacing.related_content import ConnectionType

    score = item.relevance_score
    conn_type = item.connection_type
    link_confidence = item.metadata.get("confidence", "")

    # Connection-type-specific logic
    if conn_type == ConnectionType.PASSAGE_LINK:
        if link_confidence == "high" or score > 0.8:
            level = ConfidenceLevel.HIGH
        elif link_confidence == "medium" or score > 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

    elif conn_type == ConnectionType.PERSONAL_REFLECTION:
        # Direct reflections from USER_REFLECTS_ON are high confidence
        if item.metadata.get("target_type") or score > 0.7:
            level = ConfidenceLevel.HIGH
        elif score > 0.4:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

    elif conn_type == ConnectionType.THEMATIC_PARALLEL:
        if score > 0.75:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

    elif conn_type == ConnectionType.VECTOR_SIMILARITY:
        if score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

    else:
        # Temporal proximity or unknown
        if score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

    label = CONFIDENCE_LABELS[level]

    return ScoredConnection(
        item=item,
        confidence_level=level,
        label=label,
        raw_score=score,
    )


def classify_batch(items: list[RelatedItem]) -> list[ScoredConnection]:
    """Classify confidence for a batch of related items.

    Returns results sorted by confidence level (HIGH first) then
    by raw score within each level.

    Args:
        items: List of RelatedItems to classify.

    Returns:
        Sorted list of ScoredConnections.
    """
    scored = [classify_confidence(item) for item in items]

    # Sort: HIGH > MEDIUM > LOW, then by score within level
    level_order = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2}
    scored.sort(key=lambda s: (level_order[s.confidence_level], -s.raw_score))

    return scored
