"""Voice profile CRUD and versioning.

Manages storage, retrieval, updating, and versioning of voice profiles
via VoiceProfileRepository. When new primary works are ingested,
profiles are re-extracted and stored as new versions with proper
is_current flag management.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IntelligenceError
from author_library.intelligence.voice_profile import VoiceProfile

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.config import Settings
    from author_library.storage.repositories import (
        ChunkRepository,
        VoiceProfileRepository,
        WorkRepository,
    )

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Version diff model
# ---------------------------------------------------------------------------


class ProfileDiff:
    """Diff between two voice profile versions."""

    def __init__(
        self,
        *,
        old_version: int,
        new_version: int,
        changes: dict[str, tuple[Any, Any]],
        added_phrases: list[str],
        removed_phrases: list[str],
    ) -> None:
        self.old_version = old_version
        self.new_version = new_version
        self.changes = changes
        self.added_phrases = added_phrases
        self.removed_phrases = removed_phrases

    @property
    def has_changes(self) -> bool:
        """Whether there are any differences between versions."""
        return bool(self.changes or self.added_phrases or self.removed_phrases)


# ---------------------------------------------------------------------------
# Voice profile manager
# ---------------------------------------------------------------------------


class VoiceProfileManager:
    """Manages voice profile lifecycle: store, retrieve, update, version.

    Provides CRUD operations against VoiceProfileRepository with
    automatic versioning. Only one version is active (is_current=True)
    at any time.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def store_profile(
        self,
        *,
        profile: VoiceProfile,
        voice_repo: VoiceProfileRepository,
    ) -> UUID:
        """Store a new voice profile as the next version.

        Determines the next version number, marks previous versions
        as non-current, and stores the new profile as current.

        Args:
            profile: The voice profile to store.
            voice_repo: Voice profile repository.

        Returns:
            UUID of the stored profile record.

        Raises:
            IntelligenceError: If storage fails.
        """
        try:
            # Determine next version number
            versions = await voice_repo.list_versions(profile.author_id)
            next_version = max((v.get("version", 0) for v in versions), default=0) + 1

            profile_id = await voice_repo.store(
                profile.author_id,
                profile.model_dump(exclude={"author_id"}),
                next_version,
            )

            log.info(
                "voice_profile_stored",
                author_id=profile.author_id,
                version=next_version,
                profile_id=str(profile_id),
            )

            return profile_id

        except IntelligenceError:
            raise
        except Exception as exc:
            raise IntelligenceError(
                f"Failed to store voice profile: {exc}",
                context={"author_id": profile.author_id},
                cause=exc,
            ) from exc

    async def get_current(
        self,
        *,
        author_id: str,
        voice_repo: VoiceProfileRepository,
    ) -> VoiceProfile | None:
        """Retrieve the current (active) voice profile for an author.

        Args:
            author_id: The author's slug identifier.
            voice_repo: Voice profile repository.

        Returns:
            The current VoiceProfile, or None if no profile exists.
        """
        row = await voice_repo.get_current(author_id)
        if row is None:
            return None

        return self._row_to_profile(author_id, row)

    async def list_versions(
        self,
        *,
        author_id: str,
        voice_repo: VoiceProfileRepository,
    ) -> list[dict[str, Any]]:
        """List all voice profile versions for an author.

        Returns a list of version metadata (version number, is_current,
        created_at, confidence).
        """
        versions = await voice_repo.list_versions(author_id)
        result: list[dict[str, Any]] = []

        for v in versions:
            profile_data = v.get("profile", {})
            if isinstance(profile_data, str):
                profile_data = json.loads(profile_data)

            result.append(
                {
                    "id": str(v.get("id", "")),
                    "version": v.get("version", 0),
                    "is_current": v.get("is_current", False),
                    "created_at": v.get("created_at"),
                    "confidence": profile_data.get("confidence", 0.0),
                }
            )

        return result

    async def get_version(
        self,
        *,
        author_id: str,
        version: int,
        voice_repo: VoiceProfileRepository,
    ) -> VoiceProfile | None:
        """Retrieve a specific version of a voice profile.

        Args:
            author_id: The author's slug identifier.
            version: The version number to retrieve.
            voice_repo: Voice profile repository.

        Returns:
            The VoiceProfile at that version, or None if not found.
        """
        versions = await voice_repo.list_versions(author_id)
        for v in versions:
            if v.get("version") == version:
                return self._row_to_profile(author_id, v)
        return None

    async def refresh_profile(
        self,
        *,
        author_id: str,
        author_name: str,
        work_repo: WorkRepository,
        chunk_repo: ChunkRepository,
        voice_repo: VoiceProfileRepository,
    ) -> tuple[VoiceProfile, ProfileDiff | None]:
        """Re-extract voice profile and store as new version.

        Used when new primary works are ingested to update the profile.
        Returns the new profile and a diff against the previous version
        (if one exists).

        Args:
            author_id: The author's slug identifier.
            author_name: The author's canonical display name.
            work_repo: Repository for accessing work metadata.
            chunk_repo: Repository for accessing corpus chunks.
            voice_repo: Voice profile repository.

        Returns:
            Tuple of (new_profile, diff_or_none).

        Raises:
            IntelligenceError: If extraction or storage fails.
        """
        from author_library.intelligence.voice_profile import VoiceProfileExtractor

        # Get the old profile for comparison
        old_profile = await self.get_current(
            author_id=author_id,
            voice_repo=voice_repo,
        )
        old_version = 0
        if old_profile:
            versions = await voice_repo.list_versions(author_id)
            current_versions = [v for v in versions if v.get("is_current", False)]
            if current_versions:
                old_version = current_versions[0].get("version", 0)

        # Extract new profile
        extractor = VoiceProfileExtractor(self._settings)
        new_profile = await extractor.extract(
            author_id=author_id,
            author_name=author_name,
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        # Store as new version
        await self.store_profile(
            profile=new_profile,
            voice_repo=voice_repo,
        )

        # Compute diff
        diff: ProfileDiff | None = None
        if old_profile:
            diff = self._compute_diff(
                old_profile=old_profile,
                new_profile=new_profile,
                old_version=old_version,
                new_version=old_version + 1,
            )

        log.info(
            "voice_profile_refreshed",
            author_id=author_id,
            old_version=old_version,
            new_version=old_version + 1,
            has_changes=diff.has_changes if diff else True,
        )

        return new_profile, diff

    def _compute_diff(
        self,
        *,
        old_profile: VoiceProfile,
        new_profile: VoiceProfile,
        old_version: int,
        new_version: int,
    ) -> ProfileDiff:
        """Compute the differences between two voice profile versions."""
        changes: dict[str, tuple[Any, Any]] = {}

        # Compare scalar fields
        for field in ["register", "humor_style", "confidence"]:
            old_val = getattr(old_profile, field)
            new_val = getattr(new_profile, field)
            if old_val != new_val:
                changes[field] = (old_val, new_val)

        # Compare list fields
        for field in [
            "sentence_patterns",
            "vocabulary_tendencies",
            "rhetorical_moves",
            "example_passages",
        ]:
            old_val = getattr(old_profile, field)
            new_val = getattr(new_profile, field)
            if set(old_val) != set(new_val):
                changes[field] = (old_val, new_val)

        # Characteristic phrases diff
        old_phrases = set(old_profile.characteristic_phrases)
        new_phrases = set(new_profile.characteristic_phrases)

        return ProfileDiff(
            old_version=old_version,
            new_version=new_version,
            changes=changes,
            added_phrases=sorted(new_phrases - old_phrases),
            removed_phrases=sorted(old_phrases - new_phrases),
        )

    def _row_to_profile(self, author_id: str, row: dict[str, Any]) -> VoiceProfile:
        """Convert a database row to a VoiceProfile model."""
        profile_data = row.get("profile", {})
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)

        return VoiceProfile(
            author_id=author_id,
            register=profile_data.get("register", ""),
            sentence_patterns=profile_data.get("sentence_patterns", []),
            vocabulary_tendencies=profile_data.get("vocabulary_tendencies", []),
            rhetorical_moves=profile_data.get("rhetorical_moves", []),
            characteristic_phrases=profile_data.get("characteristic_phrases", []),
            humor_style=profile_data.get("humor_style"),
            example_passages=profile_data.get("example_passages", []),
            confidence=profile_data.get("confidence", 0.0),
        )
