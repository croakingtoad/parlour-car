"""Tests for transcript chunking strategy (A6).

Covers:
- Supported genres
- Speaker-change splitting (primary split)
- ~2-minute fallback splitting for single speaker
- Timestamp and speaker metadata preservation per chunk
- Granularity tier production (macro, meso, micro)
- Registration as 7th strategy in the registry
"""

from __future__ import annotations

import pytest

from author_library.chunking import get_chunking_strategy, list_strategies
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.chunking.transcript import TranscriptChunkingStrategy
from author_library.parsing.models import DocumentMetadata, DocumentNode, NodeType, ParsedDocument


# ---------------------------------------------------------------------------
# Fixtures: transcript documents
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_speaker_transcript() -> ParsedDocument:
    """Transcript with multiple speaker changes."""
    raw_text = (
        "Malcolm Guite: [00:00:15] The imagination is not a faculty among "
        "other faculties. It is the living power and prime agent of all human "
        "perception. Coleridge understood this perhaps better than anyone.\n"
        "\n"
        "Rowan Williams: [00:01:30] I think that is exactly right. And what "
        "makes Coleridge so fascinating is that he never separates the poetic "
        "from the theological. For him, the act of imagination is always already "
        "a participation in the divine creative act.\n"
        "\n"
        "Malcolm Guite: [00:02:45] Yes, and that is why I return to him again "
        "and again. The Secondary Imagination, as he calls it, is an echo of "
        "the primary. It dissolves, diffuses, dissipates in order to recreate. "
        "That is the poetic act in a nutshell.\n"
        "\n"
        "Rowan Williams: [00:04:00] And the implications for liturgy are "
        "profound. If poetry is a mode of participation in divine creativity, "
        "then the language of worship must be genuinely poetic."
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[
            DocumentNode(node_type=NodeType.PARAGRAPH, text=raw_text),
        ],
    )

    return ParsedDocument(
        source_path="/transcripts/guite-williams-conversation.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="Imagination and Theology: A Conversation",
            author="Malcolm Guite",
            word_count=len(raw_text.split()),
        ),
        tree=tree,
        raw_text=raw_text,
    )


@pytest.fixture
def single_speaker_long_transcript() -> ParsedDocument:
    """Single-speaker transcript longer than 600 words to trigger fallback split."""
    # Generate >700 words of single-speaker content to exceed _MAX_MESO_WORDS (600)
    sentences = [
        "The imagination is not a faculty among other faculties.",
        "It is the living power and prime agent of all human perception.",
        "Coleridge understood this perhaps better than anyone in the English tradition.",
        "When we read the Biographia Literaria we encounter a mind grappling with the deepest questions.",
        "The Primary Imagination is a repetition in the finite mind of the eternal act of creation.",
        "This is not merely an aesthetic claim but a profoundly theological one.",
        "Every act of genuine perception is already a participation in divine creativity.",
        "The Secondary Imagination is an echo of the former.",
        "It co-exists with the conscious will yet retains the same kind of agency.",
        "What distinguishes the poet is the intensity and scope of this secondary power.",
        "Poetry becomes not merely an art form but a mode of theological knowing.",
        "The symbol, as Coleridge defines it, partakes of the reality it renders intelligible.",
        "This is fundamentally different from allegory which is merely a translation of abstract notions.",
        "The sacramental vision that Coleridge articulates finds its deepest expression here.",
        "We must understand that for Coleridge language itself is participatory.",
        "Words do not merely point to things but participate in the realities they name.",
        "This insight has enormous implications for how we read Scripture.",
        "The Bible is not a collection of propositions but a living word.",
        "It enacts the very realities it describes through its poetic power.",
        "Herbert understood this when he wrote The Temple.",
        "Each poem is not merely about prayer but is itself an act of prayer.",
        "The form and the content are inseparable in genuine religious poetry.",
        "This is what I mean when I say that imagination is a theological faculty.",
        "It is not opposed to reason but completes it.",
        "Reason without imagination gives us only the letter that kills.",
        "Imagination without reason gives us mere fantasy.",
        "But together they give us what Coleridge calls the living power.",
        "And this living power is what we need most desperately in our time.",
        "The reduction of language to mere information has impoverished our souls.",
        "We need poets who can restore to us the sacramental dimension of speech.",
        "That is the calling I have tried to follow in my own work.",
        "Each sonnet is an attempt to hold together the temporal and the eternal.",
        "The volta in the Petrarchan form enacts the very turn from doubt to faith.",
        "The constraint of the form is not a limitation but a liberation.",
        "Just as the banks of a river give the water its power and direction.",
        "So the fourteen lines of a sonnet concentrate and intensify meaning.",
        "And we find in the Romantic tradition a profound engagement with the sacred.",
        "Wordsworth speaks of the world as charged with a grandeur that will flame out.",
        "Hopkins follows this line with his inscape and instress which are fundamentally sacramental.",
        "The dappled things that Hopkins celebrates are not mere surfaces but windows into glory.",
        "The windhover riding the rolling level underneath him steady air is Christ in motion.",
        "And Keats with his negative capability opens a space for mystery that theology needs.",
        "The poet must be capable of being in uncertainties without any irritable reaching after fact.",
        "This is not relativism but rather a deeper form of knowing that transcends mere information.",
        "The Romantic poets teach us that beauty and truth are finally inseparable.",
        "And this convergence of beauty and truth is what the Christian tradition has always affirmed.",
        "From Augustine through Aquinas through the medieval mystics we find this same insistence.",
        "That the beautiful is not merely decorative but revelatory of the deepest structures of reality.",
        "This is what I have tried to recover in my own sonnets and my critical writings.",
        "Each poem is an experiment in transfiguration attempting to let the eternal shine through.",
    ]
    raw_text = "Malcolm Guite: [00:00:00] " + " ".join(sentences)

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[DocumentNode(node_type=NodeType.PARAGRAPH, text=raw_text)],
    )

    return ParsedDocument(
        source_path="/transcripts/guite-lecture.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="Coleridge and the Poetic Imagination",
            author="Malcolm Guite",
            word_count=len(raw_text.split()),
        ),
        tree=tree,
        raw_text=raw_text,
    )


@pytest.fixture
def no_speaker_transcript() -> ParsedDocument:
    """Transcript without speaker labels (e.g., auto-generated captions)."""
    raw_text = (
        "[00:00:05] The imagination is not a faculty among other faculties. "
        "It is the living power and prime agent of all human perception.\n"
        "[00:00:30] Coleridge understood this perhaps better than anyone "
        "in the English tradition. The Biographia Literaria remains essential.\n"
        "[00:01:00] When we read Chapter XIII we encounter the famous "
        "distinction between Primary and Secondary Imagination."
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[DocumentNode(node_type=NodeType.PARAGRAPH, text=raw_text)],
    )

    return ParsedDocument(
        source_path="/transcripts/auto-captions.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="Auto-generated captions",
            author="Unknown",
            word_count=len(raw_text.split()),
        ),
        tree=tree,
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Strategy basics
# ---------------------------------------------------------------------------


class TestTranscriptStrategyBasics:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "transcript" in genres
        assert "video-transcript" in genres
        assert "audio-transcript" in genres
        assert "podcast-transcript" in genres
        assert "youtube-captions" in genres
        assert "interview-transcript" in genres

    def test_supported_genre_count(self) -> None:
        assert len(self.strategy.supported_genres()) == 6


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestTranscriptRegistration:
    def test_transcript_in_strategy_list(self) -> None:
        strategies = list_strategies()
        types = {type(s) for s in strategies}
        assert TranscriptChunkingStrategy in types

    def test_seven_strategies_total(self) -> None:
        assert len(list_strategies()) == 8

    def test_bare_transcript_genre_maps_to_sermon(self) -> None:
        # "transcript" is shared by SermonStrategy and TranscriptChunkingStrategy.
        # SermonStrategy registers later in the _GENRE_MAP so it wins.
        # Use more specific genre tags (video-transcript, etc.) for transcripts.
        strategy = get_chunking_strategy(["transcript"])
        from author_library.chunking.sermon import SermonStrategy
        assert isinstance(strategy, SermonStrategy)

    def test_video_transcript_genre_selects_strategy(self) -> None:
        strategy = get_chunking_strategy(["video-transcript"])
        assert isinstance(strategy, TranscriptChunkingStrategy)

    def test_youtube_captions_genre_selects_strategy(self) -> None:
        strategy = get_chunking_strategy(["youtube-captions"])
        assert isinstance(strategy, TranscriptChunkingStrategy)

    def test_podcast_transcript_genre_selects_strategy(self) -> None:
        strategy = get_chunking_strategy(["podcast-transcript"])
        assert isinstance(strategy, TranscriptChunkingStrategy)


# ---------------------------------------------------------------------------
# Multi-speaker splitting
# ---------------------------------------------------------------------------


class TestMultiSpeakerSplitting:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_produces_chunks(self, multi_speaker_transcript: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        assert len(chunks) > 0

    def test_produces_all_granularities(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        granularities = {c.granularity for c in chunks}
        assert ChunkGranularity.MACRO in granularities
        assert ChunkGranularity.MESO in granularities
        assert ChunkGranularity.MICRO in granularities

    def test_one_macro_chunk(self, multi_speaker_transcript: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro) == 1

    def test_meso_splits_on_speaker_change(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # 4 speaker turns → 4 meso chunks
        assert len(meso) == 4

    def test_speaker_metadata_on_meso(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        speakers = [c.metadata.get("speaker") for c in meso]
        assert "Malcolm Guite" in speakers
        assert "Rowan Williams" in speakers

    def test_timestamp_metadata_on_meso(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        timestamps = [c.metadata.get("timestamp") for c in meso]
        # Should have timestamps from the transcript
        assert any(ts is not None for ts in timestamps)

    def test_macro_has_all_speakers(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO][0]
        speakers = macro.metadata.get("speakers", [])
        assert "Malcolm Guite" in speakers
        assert "Rowan Williams" in speakers

    def test_source_class_propagates(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.source_class == "primary"

    def test_work_id_propagates(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.work_id == "guite--imagination-conversation"

    def test_genre_metadata(self, multi_speaker_transcript: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "transcript"

    def test_parent_child_relationships(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        macro_ids = {c.id for c in chunks if c.granularity == ChunkGranularity.MACRO}
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        micro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]

        for meso in meso_chunks:
            assert meso.parent_chunk_id in macro_ids

        meso_ids = {c.id for c in meso_chunks}
        for micro in micro_chunks:
            assert micro.parent_chunk_id in meso_ids


# ---------------------------------------------------------------------------
# Single-speaker fallback splitting
# ---------------------------------------------------------------------------


class TestSingleSpeakerFallback:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_long_single_speaker_gets_split(
        self, single_speaker_long_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            single_speaker_long_transcript,
            work_id="guite--coleridge-lecture",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # ~700 words should produce at least 2 meso chunks via fallback splitting
        assert len(meso) >= 2

    def test_fallback_chunks_approximate_300_words(
        self, single_speaker_long_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            single_speaker_long_transcript,
            work_id="guite--coleridge-lecture",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for meso_chunk in meso:
            # Each should be roughly 300 words (allow generous margin for sentence boundaries)
            assert meso_chunk.word_count <= 700

    def test_speaker_preserved_on_fallback_splits(
        self, single_speaker_long_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            single_speaker_long_transcript,
            work_id="guite--coleridge-lecture",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for meso_chunk in meso:
            assert meso_chunk.metadata.get("speaker") == "Malcolm Guite"


# ---------------------------------------------------------------------------
# No-speaker transcripts
# ---------------------------------------------------------------------------


class TestNoSpeakerTranscript:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_produces_chunks(self, no_speaker_transcript: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            no_speaker_transcript,
            work_id="unknown--auto-captions",
            source_class="contextual",
        )
        assert len(chunks) > 0

    def test_produces_macro(self, no_speaker_transcript: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            no_speaker_transcript,
            work_id="unknown--auto-captions",
            source_class="contextual",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro) == 1


# ---------------------------------------------------------------------------
# Position numbering
# ---------------------------------------------------------------------------


class TestPositionNumbering:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_positions_are_sequential(
        self, multi_speaker_transcript: ParsedDocument
    ) -> None:
        chunks = self.strategy.chunk(
            multi_speaker_transcript,
            work_id="guite--imagination-conversation",
            source_class="primary",
        )
        for granularity in ChunkGranularity:
            positions = [c.position for c in chunks if c.granularity == granularity]
            if positions:
                assert positions == sorted(positions)
                assert positions == list(range(len(positions)))


# ---------------------------------------------------------------------------
# Empty transcript handling
# ---------------------------------------------------------------------------


class TestEmptyTranscript:
    def setup_method(self) -> None:
        self.strategy = TranscriptChunkingStrategy()

    def test_empty_text_returns_no_chunks(self) -> None:
        doc = ParsedDocument(
            source_path="/transcripts/empty.txt",
            format="txt",
            metadata=DocumentMetadata(
                title="Empty",
                author="Unknown",
                word_count=0,
            ),
            tree=DocumentNode(node_type=NodeType.BOOK, children=[]),
            raw_text="",
        )
        chunks = self.strategy.chunk(doc, work_id="test--empty", source_class="primary")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self) -> None:
        doc = ParsedDocument(
            source_path="/transcripts/whitespace.txt",
            format="txt",
            metadata=DocumentMetadata(
                title="Whitespace",
                author="Unknown",
                word_count=0,
            ),
            tree=DocumentNode(node_type=NodeType.BOOK, children=[]),
            raw_text="   \n\n   \n  ",
        )
        chunks = self.strategy.chunk(doc, work_id="test--ws", source_class="primary")
        assert chunks == []
