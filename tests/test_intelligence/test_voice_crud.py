"""Tests for voice profile CRUD and versioning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.intelligence.voice_crud import VoiceProfileManager
from author_library.intelligence.voice_profile import VoiceProfile
from author_library.storage.repositories import PgVoiceProfileRepository

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.postgres import PostgresPool

from tests.test_intelligence.conftest import (
    insert_sample_data,
    requires_anthropic_key,
)

# ---------------------------------------------------------------------------
# Profile diff tests
# ---------------------------------------------------------------------------


class TestProfileDiff:
    """Test voice profile version diffing."""

    def test_diff_detects_register_change(self) -> None:
        """Should detect changes in register."""
        manager = VoiceProfileManager.__new__(VoiceProfileManager)

        old = VoiceProfile(
            author_id="test",
            register="academic scholarly",
            sentence_patterns=["pattern1"],
            vocabulary_tendencies=["vocab1"],
            rhetorical_moves=["move1"],
            characteristic_phrases=["phrase1", "phrase2"],
            confidence=0.7,
        )
        new = VoiceProfile(
            author_id="test",
            register="conversational scholarly",
            sentence_patterns=["pattern1"],
            vocabulary_tendencies=["vocab1"],
            rhetorical_moves=["move1"],
            characteristic_phrases=["phrase1", "phrase3"],
            confidence=0.85,
        )

        diff = manager._compute_diff(
            old_profile=old,
            new_profile=new,
            old_version=1,
            new_version=2,
        )

        assert diff.has_changes
        assert "register" in diff.changes
        assert "confidence" in diff.changes
        assert "phrase3" in diff.added_phrases
        assert "phrase2" in diff.removed_phrases

    def test_diff_no_changes(self) -> None:
        """Should report no changes when profiles are identical."""
        manager = VoiceProfileManager.__new__(VoiceProfileManager)

        profile = VoiceProfile(
            author_id="test",
            register="scholarly",
            sentence_patterns=["p1"],
            vocabulary_tendencies=["v1"],
            rhetorical_moves=["m1"],
            characteristic_phrases=["cp1"],
            confidence=0.8,
        )

        diff = manager._compute_diff(
            old_profile=profile,
            new_profile=profile,
            old_version=1,
            new_version=2,
        )

        assert not diff.has_changes
        assert diff.added_phrases == []
        assert diff.removed_phrases == []


# ---------------------------------------------------------------------------
# CRUD tests against real PostgreSQL
# ---------------------------------------------------------------------------


async def test_store_and_retrieve_profile(pg_pool: PostgresPool, app_settings: Settings) -> None:
    """Store a voice profile and retrieve it."""
    await insert_sample_data(pg_pool)

    voice_repo = PgVoiceProfileRepository(pg_pool)
    manager = VoiceProfileManager(app_settings)

    profile = VoiceProfile(
        author_id="malcolm-guite",
        register="academic but accessible, conversational scholarly",
        sentence_patterns=[
            "favors complex sentences with embedded clauses",
            "alternates between analytical exposition and poetic quotation",
        ],
        vocabulary_tendencies=[
            "frequently uses 'sacramental', 'incarnational'",
        ],
        rhetorical_moves=[
            "builds arguments through close reading of poetry",
        ],
        characteristic_phrases=[
            "outward and visible sign",
            "the poet's task",
        ],
        humor_style="dry wit, self-deprecating",
        example_passages=["A representative passage of the author's work."],
        confidence=0.85,
    )

    profile_id = await manager.store_profile(
        profile=profile,
        voice_repo=voice_repo,
    )
    assert profile_id is not None

    # Retrieve
    current = await manager.get_current(
        author_id="malcolm-guite",
        voice_repo=voice_repo,
    )
    assert current is not None
    assert current.author_id == "malcolm-guite"
    assert current.register == "academic but accessible, conversational scholarly"
    assert current.confidence == 0.85


async def test_versioning(pg_pool: PostgresPool, app_settings: Settings) -> None:
    """Store multiple versions and verify is_current flag."""
    await insert_sample_data(pg_pool)

    voice_repo = PgVoiceProfileRepository(pg_pool)
    manager = VoiceProfileManager(app_settings)

    # Version 1
    profile_v1 = VoiceProfile(
        author_id="malcolm-guite",
        register="scholarly",
        sentence_patterns=["pattern1"],
        vocabulary_tendencies=["vocab1"],
        rhetorical_moves=["move1"],
        characteristic_phrases=["phrase1"],
        confidence=0.7,
    )
    await manager.store_profile(profile=profile_v1, voice_repo=voice_repo)

    # Version 2
    profile_v2 = VoiceProfile(
        author_id="malcolm-guite",
        register="conversational scholarly",
        sentence_patterns=["pattern1", "pattern2"],
        vocabulary_tendencies=["vocab1", "vocab2"],
        rhetorical_moves=["move1"],
        characteristic_phrases=["phrase1", "phrase2"],
        confidence=0.85,
    )
    await manager.store_profile(profile=profile_v2, voice_repo=voice_repo)

    # Current should be v2
    current = await manager.get_current(
        author_id="malcolm-guite",
        voice_repo=voice_repo,
    )
    assert current is not None
    assert current.register == "conversational scholarly"
    assert current.confidence == 0.85

    # Should have 2 versions
    versions = await manager.list_versions(
        author_id="malcolm-guite",
        voice_repo=voice_repo,
    )
    assert len(versions) == 2

    # Only one should be current
    current_versions = [v for v in versions if v["is_current"]]
    assert len(current_versions) == 1
    assert current_versions[0]["version"] == 2


async def test_get_specific_version(pg_pool: PostgresPool, app_settings: Settings) -> None:
    """Retrieve a specific version by number."""
    await insert_sample_data(pg_pool)

    voice_repo = PgVoiceProfileRepository(pg_pool)
    manager = VoiceProfileManager(app_settings)

    # Store two versions
    for i in range(2):
        profile = VoiceProfile(
            author_id="malcolm-guite",
            register=f"register-v{i + 1}",
            sentence_patterns=[],
            vocabulary_tendencies=[],
            rhetorical_moves=[],
            characteristic_phrases=[],
            confidence=0.5 + (i * 0.2),
        )
        await manager.store_profile(profile=profile, voice_repo=voice_repo)

    # Get version 1
    v1 = await manager.get_version(
        author_id="malcolm-guite",
        version=1,
        voice_repo=voice_repo,
    )
    assert v1 is not None
    assert v1.register == "register-v1"

    # Get version 2
    v2 = await manager.get_version(
        author_id="malcolm-guite",
        version=2,
        voice_repo=voice_repo,
    )
    assert v2 is not None
    assert v2.register == "register-v2"

    # Non-existent version
    v99 = await manager.get_version(
        author_id="malcolm-guite",
        version=99,
        voice_repo=voice_repo,
    )
    assert v99 is None


async def test_no_profile_returns_none(pg_pool: PostgresPool, app_settings: Settings) -> None:
    """Getting a profile for an author with no profiles should return None."""
    await insert_sample_data(pg_pool)

    voice_repo = PgVoiceProfileRepository(pg_pool)
    manager = VoiceProfileManager(app_settings)

    result = await manager.get_current(
        author_id="malcolm-guite",
        voice_repo=voice_repo,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Refresh integration test (requires API key)
# ---------------------------------------------------------------------------


@requires_anthropic_key
async def test_refresh_profile_integration(
    pg_pool: PostgresPool,
    app_settings: Settings,
) -> None:
    """End-to-end profile refresh with diff against real API."""
    await insert_sample_data(pg_pool)

    voice_repo = PgVoiceProfileRepository(pg_pool)
    manager = VoiceProfileManager(app_settings)

    from author_library.storage.repositories import PgChunkRepository, PgWorkRepository

    work_repo = PgWorkRepository(pg_pool)
    chunk_repo = PgChunkRepository(pg_pool)

    # Store an initial profile
    initial = VoiceProfile(
        author_id="malcolm-guite",
        register="scholarly",
        sentence_patterns=["pattern1"],
        vocabulary_tendencies=["vocab1"],
        rhetorical_moves=["move1"],
        characteristic_phrases=["old phrase"],
        confidence=0.6,
    )
    await manager.store_profile(profile=initial, voice_repo=voice_repo)

    # Refresh (re-extract from corpus)
    new_profile, diff = await manager.refresh_profile(
        author_id="malcolm-guite",
        author_name="Malcolm Guite",
        work_repo=work_repo,
        chunk_repo=chunk_repo,
        voice_repo=voice_repo,
    )

    assert new_profile.author_id == "malcolm-guite"
    assert new_profile.confidence > 0.0
    assert diff is not None
    # The LLM-extracted profile should differ from our hand-crafted initial
    assert diff.has_changes

    # Should now have 2 versions
    versions = await manager.list_versions(
        author_id="malcolm-guite",
        voice_repo=voice_repo,
    )
    assert len(versions) == 2
