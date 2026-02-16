"""Shared fixtures for chunking tests.

Provides pre-built ParsedDocument trees for each genre, avoiding any
mock data — all trees are constructed from real-structured content.
"""

from __future__ import annotations

import pytest

from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

# ------------------------------------------------------------------
# Scholarly prose document
# ------------------------------------------------------------------


@pytest.fixture
def scholarly_document() -> ParsedDocument:
    """A scholarly monograph with chapters, sections, paragraphs, footnotes."""
    para1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Coleridge's distinction between Primary and Secondary Imagination forms "
            "the cornerstone of his entire poetic philosophy. The Primary Imagination, "
            "as he articulates it in Chapter 13 of the Biographia Literaria, is the "
            "living power and prime agent of all human perception. It is a repetition "
            "in the finite mind of the eternal act of creation in the infinite I AM. "
            "This theological grounding is precisely what makes Coleridge's theory so "
            "distinctive among Romantic accounts of creativity. Where Wordsworth speaks "
            "of the mind as fitted to the external world, Coleridge insists that the "
            "mind participates in the divine creative act itself."
        ),
        metadata={"footnote_refs": ["fn1"]},
    )
    para2 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "The Secondary Imagination, by contrast, is described as an echo of the "
            "former, co-existing with the conscious will. It dissolves, diffuses, "
            "dissipates, in order to re-create. This is the specifically poetic faculty, "
            "the power that enables the poet to shape and reshape experience into new "
            "unities. The poet, as Coleridge conceives the role, does not merely "
            "observe or record but actively participates in bringing forth the meaning "
            "of what is perceived."
        ),
    )
    para3 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "The implications of this framework for theology are profound. If the "
            "Primary Imagination is truly a participation in divine creativity, then "
            "every act of genuine perception is already a theological event. Poetry, "
            "which exercises the Secondary Imagination most fully, becomes not merely "
            "an aesthetic enterprise but a mode of theological knowing. This insight "
            "underpins the entire argument that follows in subsequent chapters."
        ),
    )
    block_quote = DocumentNode(
        node_type=NodeType.BLOCK_QUOTE,
        text=(
            "The primary IMAGINATION I hold to be the living Power and prime Agent "
            "of all human Perception, and as a repetition in the finite mind of the "
            "eternal act of creation in the infinite I AM."
        ),
        metadata={"quoted_author": "Samuel Taylor Coleridge"},
    )
    footnote = DocumentNode(
        node_type=NodeType.FOOTNOTE,
        text=(
            "Coleridge, Biographia Literaria, ed. James Engell and W. Jackson Bate, "
            "vol. 7 of The Collected Works of Samuel Taylor Coleridge (Princeton: "
            "Princeton University Press, 1983), Ch. 13, pp. 295-296. This passage "
            "has been the subject of extensive scholarly debate regarding whether "
            "Coleridge is describing an actual ontological participation or merely "
            "an analogical relationship between human and divine creativity."
        ),
        metadata={"ref": "fn1"},
    )

    section1 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[para1, block_quote, para2],
        metadata={"title": "Imagination as Theological Faculty"},
    )
    section2 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[para3],
        metadata={"title": "Implications for Theology"},
    )
    bibliography = DocumentNode(
        node_type=NodeType.BIBLIOGRAPHY,
        children=[
            DocumentNode(
                node_type=NodeType.BIB_ENTRY,
                text="Coleridge, Samuel Taylor. Biographia Literaria. 1817.",
            ),
        ],
    )

    chapter = DocumentNode(
        node_type=NodeType.CHAPTER,
        children=[section1, section2, footnote, bibliography],
        metadata={"title": "The Poetic Imagination"},
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[chapter],
    )

    all_text = (
        f"{para1.text}\n{block_quote.text}\n{para2.text}\n{para3.text}\n"
        f"{footnote.text}\n{bibliography.children[0].text}"
    )

    return ParsedDocument(
        source_path="/books/faith-hope-poetry.epub",
        format="epub",
        metadata=DocumentMetadata(
            title="Faith, Hope and Poetry",
            author="Malcolm Guite",
            publication_date="2010",
            word_count=len(all_text.split()),
        ),
        tree=tree,
        raw_text=all_text,
    )


# ------------------------------------------------------------------
# Poetry document
# ------------------------------------------------------------------


def _make_poem_lines(count: int) -> str:
    """Generate *count* lines of verse-like text."""
    lines = []
    themes = [
        "The morning light breaks through the ancient glass",
        "And casts a shadow where the altar stands",
        "The choir lifts its voice in solemn mass",
        "While pilgrims gather from the distant lands",
        "Each stone remembers what the builders knew",
        "That faith is carved in more than mortar's bond",
        "The arches reach for what the heart holds true",
        "A grace that stretches far beyond, beyond",
        "In every window lives a storied saint",
        "Whose colors catch the slowly turning day",
        "They speak of joy without the least complaint",
        "And guide the wanderer upon the way",
    ]
    for i in range(count):
        lines.append(themes[i % len(themes)])
    return "\n".join(lines)


@pytest.fixture
def poetry_document() -> ParsedDocument:
    """A poetry collection with short and long poems."""
    short_poem = DocumentNode(
        node_type=NodeType.POEM,
        children=[
            DocumentNode(
                node_type=NodeType.STANZA,
                text=(
                    "The morning light breaks through the ancient glass\n"
                    "And casts a shadow where the altar stands\n"
                    "The choir lifts its voice in solemn mass\n"
                    "While pilgrims gather from the distant lands"
                ),
            ),
            DocumentNode(
                node_type=NodeType.STANZA,
                text=(
                    "Each stone remembers what the builders knew\n"
                    "That faith is carved in more than mortar's bond\n"
                    "The arches reach for what the heart holds true\n"
                    "A grace that stretches far beyond, beyond"
                ),
            ),
        ],
        metadata={"title": "Cathedral", "epigraph": "For the builders"},
    )

    # Long poem: 50 lines across 5 stanzas
    long_stanzas = []
    for _stanza_idx in range(5):
        stanza_lines = _make_poem_lines(10)
        long_stanzas.append(
            DocumentNode(
                node_type=NodeType.STANZA,
                text=stanza_lines,
            )
        )
    long_poem = DocumentNode(
        node_type=NodeType.POEM,
        children=long_stanzas,
        metadata={"title": "The Long Pilgrimage"},
    )

    section = DocumentNode(
        node_type=NodeType.SECTION,
        children=[short_poem, long_poem],
        metadata={"title": "Part I: Sacred Spaces"},
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[section],
    )

    raw_text = "\n\n".join(
        "\n".join(s.text for s in poem.children)
        for poem in [short_poem, long_poem]
    )

    return ParsedDocument(
        source_path="/books/sounding-the-seasons.epub",
        format="epub",
        metadata=DocumentMetadata(
            title="Sounding the Seasons",
            author="Malcolm Guite",
            publication_date="2012",
            word_count=len(raw_text.split()),
        ),
        tree=tree,
        raw_text=raw_text,
    )


# ------------------------------------------------------------------
# Sermon document
# ------------------------------------------------------------------


@pytest.fixture
def sermon_document() -> ParsedDocument:
    """A sermon with movements and scripture references."""
    intro = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "We gather this morning in the season of Advent, a time of waiting "
            "and expectation. Our text today comes from John 1:14, which tells us "
            "that the Word became flesh and dwelt among us. This is the central "
            "mystery of the Incarnation, and it speaks directly to our theme of "
            "how poetry and theology converge."
        ),
    )
    development = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "When we say that the Word became flesh, we are saying something about "
            "language itself. Language is not merely a tool for communication but a "
            "mode of incarnation. Every genuine poem enacts a kind of incarnation, "
            "giving flesh to what would otherwise remain abstract and inaccessible. "
            "As Coleridge understood, the symbol partakes of the reality it renders "
            "intelligible. The bread and wine of Communion are not mere signs pointing "
            "elsewhere; they are the reality they signify. So too, the best poetry "
            "does not merely describe experience but embodies it."
        ),
    )
    illustration = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Consider the opening of Genesis 1:1, 'In the beginning God created the "
            "heavens and the earth.' This is not merely a historical claim. It is "
            "itself a creative act of language, a poem that brings into being the very "
            "reality it describes. The Hebrew word 'bara' carries this double force: "
            "God speaks, and the world is spoken into existence. Matthew 5:3 reminds "
            "us that the poor in spirit are blessed precisely because they remain open "
            "to this creative word."
        ),
    )
    conclusion = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "So let us go forth this Advent with ears attuned to the Word that is "
            "always becoming flesh in our midst. Let us read poetry as a spiritual "
            "discipline, and let us receive the Incarnation not as a doctrine to "
            "be defended but as a reality to be inhabited."
        ),
    )

    movement1 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[intro],
        metadata={"title": "Introduction: Advent and the Word"},
    )
    movement2 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[development],
        metadata={"title": "Language as Incarnation"},
    )
    movement3 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[illustration],
        metadata={"title": "Genesis and Creative Language"},
    )
    movement4 = DocumentNode(
        node_type=NodeType.SECTION,
        children=[conclusion],
        metadata={"title": "Going Forth"},
    )

    chapter = DocumentNode(
        node_type=NodeType.CHAPTER,
        children=[movement1, movement2, movement3, movement4],
        metadata={
            "title": "The Word Made Flesh",
            "occasion": "Advent Sunday",
            "venue": "Girton College Chapel",
            "date": "2018-12-02",
        },
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[chapter],
    )

    raw_text = "\n\n".join([intro.text, development.text, illustration.text, conclusion.text])

    return ParsedDocument(
        source_path="/sermons/advent-word.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="The Word Made Flesh",
            author="Malcolm Guite",
            word_count=len(raw_text.split()),
        ),
        tree=tree,
        raw_text=raw_text,
    )


# ------------------------------------------------------------------
# Letter document
# ------------------------------------------------------------------


@pytest.fixture
def letter_document() -> ParsedDocument:
    """A collection of letters."""
    letter1_para1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Dear Rowan, Thank you for your kind words about the new collection. "
            "I have been thinking deeply about what you said regarding the "
            "relationship between form and meaning in the sonnets. You are right "
            "that the constraint of the form is not a limitation but a liberation."
        ),
    )
    letter1_para2 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "I have enclosed a new sonnet sequence on the Beatitudes which I "
            "think takes the conversation further. The paradox at the heart of "
            "each Beatitude seemed to demand the volta structure of the Petrarchan "
            "form. With warmest regards, Malcolm"
        ),
    )

    letter1 = DocumentNode(
        node_type=NodeType.CHAPTER,
        children=[letter1_para1, letter1_para2],
        metadata={
            "title": "Letter to Rowan Williams",
            "recipient": "Rowan Williams",
            "date": "2015-03-14",
        },
    )

    letter2_para1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Dear Luci, Your reflection on the Transfiguration poems opened up "
            "something I had not seen in my own work. You pointed out that the "
            "movement from glory to suffering mirrors the liturgical year itself."
        ),
    )

    letter2 = DocumentNode(
        node_type=NodeType.CHAPTER,
        children=[letter2_para1],
        metadata={
            "title": "Letter to Luci Shaw",
            "recipient": "Luci Shaw",
            "date": "2016-07-22",
        },
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[letter1, letter2],
    )

    raw = "\n\n".join([letter1_para1.text, letter1_para2.text, letter2_para1.text])

    return ParsedDocument(
        source_path="/letters/collected-letters.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="Collected Letters",
            author="Malcolm Guite",
            word_count=len(raw.split()),
        ),
        tree=tree,
        raw_text=raw,
    )


# ------------------------------------------------------------------
# Blog post document
# ------------------------------------------------------------------


@pytest.fixture
def blog_document() -> ParsedDocument:
    """A blog post."""
    para1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "I have been reading George Herbert's The Temple again this Lent and "
            "finding new depths in poems I thought I knew well. Herbert's capacity "
            "to hold doubt and faith in the same poetic breath never ceases to "
            "astonish me. Each reading reveals layers I had missed before."
        ),
    )
    para2 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Take 'The Collar,' for instance. The speaker's rebellion against "
            "divine constraint is rendered with such visceral energy that we feel "
            "the genuine force of the temptation. But the poem's resolution in "
            "its final four words — 'My child. / My Lord.' — transforms everything "
            "that has come before. This is not a suppression of doubt but a "
            "transfiguration of it."
        ),
    )

    chapter = DocumentNode(
        node_type=NodeType.CHAPTER,
        children=[para1, para2],
        metadata={
            "title": "Re-reading Herbert in Lent",
            "url": "https://blog.malcolmguite.com/rereading-herbert",
            "date": "2019-03-15",
        },
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[chapter],
    )

    raw = f"{para1.text}\n\n{para2.text}"

    return ParsedDocument(
        source_path="/blog/herbert-lent.html",
        format="html",
        metadata=DocumentMetadata(
            title="Re-reading Herbert in Lent",
            author="Malcolm Guite",
            word_count=len(raw.split()),
        ),
        tree=tree,
        raw_text=raw,
    )


# ------------------------------------------------------------------
# Interview document
# ------------------------------------------------------------------


@pytest.fixture
def interview_document() -> ParsedDocument:
    """An interview with Q&A pairs."""
    q1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Your work often draws on Coleridge's theory of imagination. Could "
            "you explain why his ideas remain relevant for contemporary theology?"
        ),
        metadata={"role": "question"},
    )
    a1 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Coleridge understood something that most systematic theologians still "
            "miss: that imagination is not a decorative add-on to reason but is "
            "itself a mode of knowing. When he describes the Primary Imagination as "
            "a repetition of the divine creative act, he is making a deeply "
            "theological claim about human consciousness. Every act of genuine "
            "perception is already a participation in God's creativity. This matters "
            "enormously for how we think about revelation, inspiration, and the "
            "nature of religious language."
        ),
        metadata={"role": "answer"},
    )
    q2 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "How does poetry fit into this picture? Is it just an illustration "
            "of these ideas, or does it play a more fundamental role?"
        ),
        metadata={"role": "question"},
    )
    a2 = DocumentNode(
        node_type=NodeType.PARAGRAPH,
        text=(
            "Poetry is absolutely fundamental. A poem is not merely an illustration "
            "of a theological truth; it is an enactment of one. When a poem works, "
            "it performs the very transfiguration of language that theology talks "
            "about. It takes ordinary words and charges them with meaning that "
            "exceeds their dictionary definitions. In that sense, every good poem "
            "is a small incarnation."
        ),
        metadata={"role": "answer"},
    )

    tree = DocumentNode(
        node_type=NodeType.BOOK,
        children=[q1, a1, q2, a2],
        metadata={"interviewer": "Jane Smith"},
    )

    raw = "\n\n".join([q1.text, a1.text, q2.text, a2.text])

    return ParsedDocument(
        source_path="/interviews/imagination-theology.txt",
        format="txt",
        metadata=DocumentMetadata(
            title="Imagination and Theology: A Conversation",
            author="Malcolm Guite",
            word_count=len(raw.split()),
        ),
        tree=tree,
        raw_text=raw,
    )
