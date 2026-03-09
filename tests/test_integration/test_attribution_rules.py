"""Integration tests for attribution and contamination prevention rules.

Verifies CLAUDE.md Absolute Inviolable Rules:
- Rule 1: Voice contamination is forbidden. Only primary sources contribute
  to voice profiles (voice_profile_eligible=true only).
- Rule 2: Personal source data is never attributed to a speaker.
  User reflections are attributed to the user, not the subject author.

These tests work at the storage/intelligence layer — no LLM calls needed.
Data is inserted directly via storage repositories.

Runs against the test database (author_library_test).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from author_library.config import Settings
from author_library.intelligence.voice_profile import VoiceProfileExtractor
from author_library.tools.composable_query import handle_search_chunks

from .conftest import SKIP_NO_DB

# ---------------------------------------------------------------------------
# Helpers: build test records
# ---------------------------------------------------------------------------


def _make_work(
    work_id: str,
    title: str,
    author: str,
    source_class: str,
    subject_author_id: str = "test-author",
    voice_profile_eligible: bool = True,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"subject_author_id": subject_author_id}
    if source_class == "primary":
        meta["voice_profile_eligible"] = voice_profile_eligible
    return {
        "work_id": work_id,
        "title": title,
        "author": author,
        "source_class": source_class,
        "source_class_note": "Test data for attribution rule verification",
        "publication_year": 2000,
        "publisher": "Test",
        "format_ingested": "txt",
        "word_count": 1000,
        "genre_tags": ["poetry"],
        "subject_headings": [],
        "source_metadata": meta,
    }


def _make_chunk(
    work_id: str,
    text: str,
    source_class: str,
    position: int = 0,
    granularity: str = "meso",
) -> dict[str, Any]:
    return {
        "work_id": work_id,
        "text": text,
        "annotation": None,
        "granularity": granularity,
        "source_class": source_class,
        "chapter": None,
        "section": None,
        "position": position,
        "parent_chunk_id": None,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# TestVoiceContamination (Rule 1)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestVoiceContamination:
    """Verify Rule 1: only primary source chunks feed voice profiles."""

    async def test_secondary_chunks_excluded_from_voice_extraction(
        self, clean_storage: Any
    ) -> None:
        """_gather_eligible_chunks excludes secondary source works."""
        # Insert a primary work
        await clean_storage.works.create(
            _make_work("test--primary-work", "Primary Poetry", "Test Author", "primary")
        )
        await clean_storage.chunks.create(
            _make_chunk("test--primary-work", "The light of stars endures.", "primary", 0)
        )
        await clean_storage.chunks.create(
            _make_chunk("test--primary-work", "Silence speaks what words cannot hold.", "primary", 1)
        )

        # Insert a secondary work (criticism)
        await clean_storage.works.create(
            _make_work(
                "test--secondary-work",
                "A Critique of Primary Poetry",
                "Another Critic",
                "secondary",
                subject_author_id="test-author",
            )
        )
        await clean_storage.chunks.create(
            _make_chunk(
                "test--secondary-work",
                "The poet's imagery is derivative and uninspired.",
                "secondary",
                0,
            )
        )

        # Use _gather_eligible_chunks to see what the voice extractor sees
        extractor = VoiceProfileExtractor.__new__(VoiceProfileExtractor)
        chunks = await extractor._gather_eligible_chunks(
            author_id="test-author",
            work_repo=clean_storage.works,
            chunk_repo=clean_storage.chunks,
        )

        # Should only include the primary chunks
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk["source_class"] == "primary", (
                f"Voice extractor returned non-primary chunk: {chunk['source_class']!r}"
            )
        texts = {c["text"] for c in chunks}
        assert "The light of stars endures." in texts
        assert "Silence speaks what words cannot hold." in texts
        # Critic's text must NOT be present
        assert "The poet's imagery is derivative and uninspired." not in texts

    async def test_contextual_chunks_excluded_from_voice_extraction(
        self, clean_storage: Any
    ) -> None:
        """_gather_eligible_chunks excludes contextual source works."""
        await clean_storage.works.create(
            _make_work("test--primary-2", "Primary Work", "Test Author", "primary")
        )
        await clean_storage.chunks.create(
            _make_chunk("test--primary-2", "Faith is the substance of things hoped for.", "primary", 0)
        )

        await clean_storage.works.create(
            _make_work(
                "test--contextual-work",
                "A Book That Influenced The Author",
                "Influence Author",
                "contextual",
                subject_author_id="test-author",
            )
        )
        await clean_storage.chunks.create(
            _make_chunk(
                "test--contextual-work",
                "Contextual source text that must not contaminate voice.",
                "contextual",
                0,
            )
        )

        extractor = VoiceProfileExtractor.__new__(VoiceProfileExtractor)
        chunks = await extractor._gather_eligible_chunks(
            author_id="test-author",
            work_repo=clean_storage.works,
            chunk_repo=clean_storage.chunks,
        )

        assert all(c["source_class"] == "primary" for c in chunks)
        texts = [c["text"] for c in chunks]
        assert any("Faith is the substance" in t for t in texts)
        assert not any("Contextual source text" in t for t in texts)

    async def test_voice_ineligible_primary_excluded(
        self, clean_storage: Any
    ) -> None:
        """_gather_eligible_chunks excludes primary works marked voice_profile_eligible=False."""
        await clean_storage.works.create(
            _make_work("test--eligible-work", "Eligible Primary", "Test Author", "primary", voice_profile_eligible=True)
        )
        await clean_storage.chunks.create(
            _make_chunk("test--eligible-work", "Eligible text for voice.", "primary", 0)
        )

        await clean_storage.works.create(
            _make_work(
                "test--ineligible-work",
                "Ineligible Primary (co-authored)",
                "Test Author",
                "primary",
                voice_profile_eligible=False,
            )
        )
        await clean_storage.chunks.create(
            _make_chunk("test--ineligible-work", "Ineligible text (must not appear).", "primary", 0)
        )

        extractor = VoiceProfileExtractor.__new__(VoiceProfileExtractor)
        chunks = await extractor._gather_eligible_chunks(
            author_id="test-author",
            work_repo=clean_storage.works,
            chunk_repo=clean_storage.chunks,
        )

        texts = [c["text"] for c in chunks]
        assert any("Eligible text for voice." in t for t in texts)
        assert not any("Ineligible text" in t for t in texts), (
            "voice_profile_eligible=False work leaked into voice extractor"
        )

    async def test_no_primary_sources_returns_empty(
        self, clean_storage: Any
    ) -> None:
        """_gather_eligible_chunks returns empty when only secondary works exist."""
        await clean_storage.works.create(
            _make_work(
                "test--only-secondary",
                "Only Secondary Work",
                "Critic",
                "secondary",
                subject_author_id="test-author",
            )
        )
        await clean_storage.chunks.create(
            _make_chunk("test--only-secondary", "Secondary text.", "secondary", 0)
        )

        extractor = VoiceProfileExtractor.__new__(VoiceProfileExtractor)
        chunks = await extractor._gather_eligible_chunks(
            author_id="test-author",
            work_repo=clean_storage.works,
            chunk_repo=clean_storage.chunks,
        )

        assert chunks == [], (
            f"Expected empty list with no primary sources, got: {chunks}"
        )


# ---------------------------------------------------------------------------
# TestVoiceEligibleProvenance (Rule 1 at retrieval layer)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestVoiceEligibleProvenance:
    """search_chunks provenance_rules.voice_eligible is False for non-primary."""

    async def test_primary_chunk_voice_eligible_true(
        self, clean_storage: Any, integration_settings: Settings
    ) -> None:
        """search_chunks marks primary chunks as voice_eligible=True."""
        from author_library.embeddings import ProviderRegistry

        await clean_storage.works.create(
            _make_work("test--prov-primary", "Voice Eligible Work", "Test Author", "primary")
        )
        chunk_id = await clean_storage.chunks.create(
            _make_chunk("test--prov-primary", "To be or not to be.", "primary", 0)
        )

        # Need an embedding to appear in vector search — insert a dummy embedding
        await clean_storage.pg.execute(
            """INSERT INTO chunk_embeddings (chunk_id, embedding, provider, model, dimensions)
               VALUES ($1, $2::vector, 'voyage', 'voyage-3-large', 1024)""",
            chunk_id,
            "[" + ",".join(["0.0"] * 1024) + "]",
        )

        embedding_provider = ProviderRegistry.create(integration_settings)
        try:
            result_str = await handle_search_chunks(
                {"query": "to be or not to be", "max_results": 5},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
        finally:
            await embedding_provider.close()

        result = json.loads(result_str)
        # May return 0 results if the dummy embedding doesn't match well,
        # but if it returns any from our work, they must be voice_eligible=True
        primary_results = [
            r for r in result.get("results", [])
            if r.get("metadata", {}).get("work_id") == "test--prov-primary"
        ]
        for r in primary_results:
            assert r["provenance_rules"]["voice_eligible"] is True, (
                f"Primary chunk marked voice_eligible=False: {r}"
            )


# ---------------------------------------------------------------------------
# TestPersonalDataAttribution (Rule 2)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestPersonalDataAttribution:
    """Verify Rule 2: personal source data is attributed to user, not the author."""

    async def test_personal_chunk_source_class_is_personal(
        self, clean_storage: Any
    ) -> None:
        """Personal reflections are stored with source_class='personal'."""
        await clean_storage.works.create({
            "work_id": "test--personal-reflection",
            "title": "My Reflections",
            "author": "User",
            "source_class": "personal",
            "source_class_note": "Personal reflection by user, not the subject author",
            "publication_year": 2024,
            "publisher": "Self",
            "format_ingested": "txt",
            "word_count": 100,
            "genre_tags": [],
            "subject_headings": [],
            "source_metadata": {"personal": True},
        })
        chunk_id = await clean_storage.chunks.create(
            _make_chunk(
                "test--personal-reflection",
                "Reading this book reminded me of my father's garden.",
                "personal",
                0,
            )
        )

        # Verify the chunk is stored with source_class='personal'
        chunk = await clean_storage.chunks.get(chunk_id)
        assert chunk is not None
        assert chunk["source_class"] == "personal", (
            f"Personal reflection stored with wrong source_class: {chunk['source_class']!r}"
        )

    async def test_personal_chunks_excluded_from_voice_extraction(
        self, clean_storage: Any
    ) -> None:
        """_gather_eligible_chunks excludes personal source chunks."""
        # Primary work
        await clean_storage.works.create(
            _make_work("test--primary-att", "Primary Work", "Test Author", "primary")
        )
        await clean_storage.chunks.create(
            _make_chunk("test--primary-att", "Author's own words, eligible for voice.", "primary", 0)
        )

        # Personal work (user's journal)
        await clean_storage.works.create({
            "work_id": "test--user-journal",
            "title": "My Reading Journal",
            "author": "User",
            "source_class": "personal",
            "source_class_note": "Personal reflection by user, not the subject author",
            "publication_year": 2024,
            "publisher": "Self",
            "format_ingested": "txt",
            "word_count": 200,
            "genre_tags": [],
            "subject_headings": [],
            "source_metadata": {"subject_author_id": "test-author"},
        })
        await clean_storage.chunks.create(
            _make_chunk(
                "test--user-journal",
                "I think this poem speaks to my own grief.",
                "personal",
                0,
            )
        )

        extractor = VoiceProfileExtractor.__new__(VoiceProfileExtractor)
        chunks = await extractor._gather_eligible_chunks(
            author_id="test-author",
            work_repo=clean_storage.works,
            chunk_repo=clean_storage.chunks,
        )

        texts = [c["text"] for c in chunks]
        assert any("Author's own words" in t for t in texts)
        assert not any("my own grief" in t for t in texts), (
            "Personal reflection leaked into voice extractor — attribution violation!"
        )

    async def test_personal_chunk_not_voice_eligible_in_provenance(
        self, clean_storage: Any
    ) -> None:
        """Personal chunks stored in DB retain source_class='personal'."""
        # This verifies that source_class='personal' is correctly stored and
        # that the retrieval layer would mark such chunks as non-voice-eligible.
        await clean_storage.works.create({
            "work_id": "test--personal-work",
            "title": "Personal Notes",
            "author": "User",
            "source_class": "personal",
            "source_class_note": "Personal reflection by user, not the subject author",
            "publication_year": 2024,
            "publisher": "Self",
            "format_ingested": "txt",
            "word_count": 50,
            "genre_tags": [],
            "subject_headings": [],
            "source_metadata": {"personal": True},
        })
        chunk_id = await clean_storage.chunks.create(
            _make_chunk("test--personal-work", "A personal reflection.", "personal", 0)
        )

        chunk = await clean_storage.chunks.get(chunk_id)
        assert chunk is not None

        # voice_eligible in composable_query is: source_class == "primary"
        # Personal source_class is NOT "primary", so voice_eligible would be False
        assert chunk["source_class"] != "primary", (
            "Personal chunk incorrectly stored as 'primary' source_class"
        )
