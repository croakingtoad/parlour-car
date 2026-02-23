"""Timestamped transcript chunking strategy.

Implements chunking for timestamped transcripts (video, audio, podcasts):
- Primary split: on speaker change
- Fallback: split at ~2 minutes if single speaker runs long
- Preserves timestamp metadata per chunk

Granularity tiers:
- Macro: entire transcript or major segment
- Meso: speaker turn or ~2-minute window (primary retrieval unit)
- Micro: individual speaker utterance within a turn
- Nano: raw capture moments (timestamp + brief text, internal only)

Speaker changes are the natural boundary. When a single speaker continues
for more than ~2 minutes, the strategy introduces fallback splits to keep
meso chunks within retrievable bounds.
"""

from __future__ import annotations

import re

import structlog

from author_library.chunking._tree_utils import collect_text, find_nodes
from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.parsing.models import DocumentNode, NodeType, ParsedDocument

logger = structlog.get_logger()

# Pattern to detect timestamps in transcript text (HH:MM:SS or MM:SS)
_TIMESTAMP_RE = re.compile(
    r"\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?"
)

# Pattern to detect speaker labels (e.g., "Speaker Name:", "SPEAKER:", "[Speaker]:")
_SPEAKER_RE = re.compile(
    r"^(?:\[)?([A-Z][A-Za-z\s.'-]+?)(?:\])?\s*:\s*",
    re.MULTILINE,
)

# ~2 minutes of speech ≈ 300 words at normal speaking pace (~150 wpm)
_FALLBACK_SPLIT_WORDS = 300

# Maximum words before forced split
_MAX_MESO_WORDS = 600


class TranscriptChunkingStrategy(ChunkingStrategy):
    """Chunking strategy for timestamped transcripts.

    Primary split: on speaker change.
    Fallback: split at ~2 minutes (300 words) if single speaker runs long.
    Preserves timestamp and speaker metadata per chunk.
    """

    def supported_genres(self) -> list[str]:
        return [
            "transcript",
            "video-transcript",
            "audio-transcript",
            "podcast-transcript",
            "youtube-captions",
            "interview-transcript",
        ]

    def chunk(
        self,
        document: ParsedDocument,
        work_id: str,
        source_class: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        position_counters: dict[ChunkGranularity, int] = {
            g: 0 for g in ChunkGranularity
        }

        # Get the full transcript text
        full_text = document.raw_text or collect_text(document.tree)
        if not full_text.strip():
            return chunks

        # Parse the transcript into speaker segments
        segments = _parse_transcript_segments(full_text)

        if not segments:
            # Fallback: treat entire text as a single segment
            segments = [TranscriptSegment(
                speaker=None,
                text=full_text.strip(),
                timestamp=_extract_first_timestamp(full_text),
            )]

        # Extract document-level metadata
        doc_title = document.metadata.title or ""
        all_speakers = list(dict.fromkeys(
            seg.speaker for seg in segments if seg.speaker
        ))

        # --- MACRO: entire transcript ---
        macro_meta: dict[str, str | int | bool | list[str]] = {
            "genre": "transcript",
        }
        if all_speakers:
            macro_meta["speakers"] = all_speakers
        if doc_title:
            macro_meta["title"] = doc_title

        macro_chunk = Chunk(
            text=full_text.strip(),
            granularity=ChunkGranularity.MACRO,
            work_id=work_id,
            source_class=source_class,
            position=position_counters[ChunkGranularity.MACRO],
            metadata=macro_meta,
        )
        position_counters[ChunkGranularity.MACRO] += 1
        chunks.append(macro_chunk)

        # --- MESO: speaker turns with ~2-minute fallback splits ---
        meso_segments = _build_meso_segments(segments)

        for meso_seg in meso_segments:
            meso_meta: dict[str, str | int | bool | list[str]] = {
                "genre": "transcript",
            }
            if meso_seg.speaker:
                meso_meta["speaker"] = meso_seg.speaker
            if meso_seg.timestamp:
                meso_meta["timestamp"] = meso_seg.timestamp
            if meso_seg.end_timestamp:
                meso_meta["end_timestamp"] = meso_seg.end_timestamp

            meso_chunk = Chunk(
                text=meso_seg.text,
                granularity=ChunkGranularity.MESO,
                work_id=work_id,
                source_class=source_class,
                position=position_counters[ChunkGranularity.MESO],
                parent_chunk_id=macro_chunk.id,
                metadata=meso_meta,
            )
            position_counters[ChunkGranularity.MESO] += 1
            chunks.append(meso_chunk)

            # --- MICRO: individual utterances within the meso turn ---
            utterances = _split_into_utterances(meso_seg.text)
            for utterance in utterances:
                if not utterance.strip():
                    continue
                micro_meta: dict[str, str | int | bool | list[str]] = {
                    "genre": "transcript",
                }
                if meso_seg.speaker:
                    micro_meta["speaker"] = meso_seg.speaker
                ts = _extract_first_timestamp(utterance)
                if ts:
                    micro_meta["timestamp"] = ts

                micro_chunk = Chunk(
                    text=utterance.strip(),
                    granularity=ChunkGranularity.MICRO,
                    work_id=work_id,
                    source_class=source_class,
                    position=position_counters[ChunkGranularity.MICRO],
                    parent_chunk_id=meso_chunk.id,
                    metadata=micro_meta,
                )
                position_counters[ChunkGranularity.MICRO] += 1
                chunks.append(micro_chunk)

        logger.info(
            "transcript_chunking_complete",
            work_id=work_id,
            total_chunks=len(chunks),
            macro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MACRO),
            meso=sum(1 for c in chunks if c.granularity == ChunkGranularity.MESO),
            micro=sum(1 for c in chunks if c.granularity == ChunkGranularity.MICRO),
            speakers=len(all_speakers),
        )
        return chunks


# ------------------------------------------------------------------
# Internal data structures
# ------------------------------------------------------------------


class TranscriptSegment:
    """A segment of transcript text attributed to a speaker."""

    __slots__ = ("end_timestamp", "speaker", "text", "timestamp")

    def __init__(
        self,
        *,
        speaker: str | None,
        text: str,
        timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> None:
        self.speaker = speaker
        self.text = text
        self.timestamp = timestamp
        self.end_timestamp = end_timestamp

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_transcript_segments(text: str) -> list[TranscriptSegment]:
    """Parse transcript text into speaker-attributed segments.

    Detects speaker labels and timestamps, splitting on speaker changes.
    """
    segments: list[TranscriptSegment] = []
    lines = text.strip().split("\n")

    current_speaker: str | None = None
    current_lines: list[str] = []
    current_timestamp: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        # Check for speaker change
        speaker_match = _SPEAKER_RE.match(stripped)
        if speaker_match:
            # Save previous segment
            if current_lines:
                segment_text = "\n".join(current_lines).strip()
                if segment_text:
                    segments.append(TranscriptSegment(
                        speaker=current_speaker,
                        text=segment_text,
                        timestamp=current_timestamp,
                    ))

            current_speaker = speaker_match.group(1).strip()
            remaining = stripped[speaker_match.end():].strip()
            current_lines = [remaining] if remaining else []
            current_timestamp = _extract_first_timestamp(stripped)
        else:
            # Check for timestamp at start of line (no speaker change)
            ts = _extract_first_timestamp(stripped)
            if ts and not current_timestamp:
                current_timestamp = ts
            current_lines.append(stripped)

    # Final segment
    if current_lines:
        segment_text = "\n".join(current_lines).strip()
        if segment_text:
            segments.append(TranscriptSegment(
                speaker=current_speaker,
                text=segment_text,
                timestamp=current_timestamp,
            ))

    return segments


def _build_meso_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Build meso-level segments from raw transcript segments.

    Groups consecutive same-speaker segments and applies the ~2-minute
    fallback split when a single speaker runs longer than _FALLBACK_SPLIT_WORDS.
    """
    if not segments:
        return []

    meso_segments: list[TranscriptSegment] = []

    # Group consecutive same-speaker segments
    grouped: list[TranscriptSegment] = []
    current_group_speaker: str | None = segments[0].speaker
    current_group_lines: list[str] = [segments[0].text]
    current_group_ts = segments[0].timestamp

    for seg in segments[1:]:
        if seg.speaker == current_group_speaker:
            current_group_lines.append(seg.text)
        else:
            # Speaker changed — flush current group
            grouped.append(TranscriptSegment(
                speaker=current_group_speaker,
                text="\n".join(current_group_lines).strip(),
                timestamp=current_group_ts,
            ))
            current_group_speaker = seg.speaker
            current_group_lines = [seg.text]
            current_group_ts = seg.timestamp

    # Flush final group
    grouped.append(TranscriptSegment(
        speaker=current_group_speaker,
        text="\n".join(current_group_lines).strip(),
        timestamp=current_group_ts,
    ))

    # Apply fallback splits for long segments
    for group in grouped:
        if group.word_count <= _MAX_MESO_WORDS:
            meso_segments.append(group)
        else:
            # Split at ~2-minute boundaries
            split_segments = _split_by_word_count(
                group.text,
                target_words=_FALLBACK_SPLIT_WORDS,
                speaker=group.speaker,
                base_timestamp=group.timestamp,
            )
            meso_segments.extend(split_segments)

    return meso_segments


def _split_by_word_count(
    text: str,
    target_words: int,
    speaker: str | None,
    base_timestamp: str | None,
) -> list[TranscriptSegment]:
    """Split text into segments of approximately target_words words.

    Tries to split at sentence boundaries for natural breaks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[TranscriptSegment] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_word_count + sentence_words > target_words and current_sentences:
            segment_text = " ".join(current_sentences)
            ts = _extract_first_timestamp(segment_text) or base_timestamp
            segments.append(TranscriptSegment(
                speaker=speaker,
                text=segment_text,
                timestamp=ts,
            ))
            current_sentences = [sentence]
            current_word_count = sentence_words
        else:
            current_sentences.append(sentence)
            current_word_count += sentence_words

    # Final segment
    if current_sentences:
        segment_text = " ".join(current_sentences)
        ts = _extract_first_timestamp(segment_text) or base_timestamp
        segments.append(TranscriptSegment(
            speaker=speaker,
            text=segment_text,
            timestamp=ts,
        ))

    return segments


def _split_into_utterances(text: str) -> list[str]:
    """Split a meso segment into micro-level utterances.

    Splits on paragraph breaks or sentence clusters (2-3 sentences).
    """
    # First try paragraph splits
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return [p.strip() for p in paragraphs if p.strip()]

    # Single paragraph — split into sentence clusters
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 3:
        return [text]

    # Group into clusters of 2-3 sentences
    clusters: list[str] = []
    cluster_size = 3
    for i in range(0, len(sentences), cluster_size):
        cluster = " ".join(sentences[i:i + cluster_size])
        if cluster.strip():
            clusters.append(cluster)

    return clusters


def _extract_first_timestamp(text: str) -> str | None:
    """Extract the first timestamp found in text."""
    match = _TIMESTAMP_RE.search(text)
    return match.group(1) if match else None
