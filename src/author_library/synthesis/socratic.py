"""O6: Socratic loop — user response → new Personal data → re-synthesis.

When the user responds to a synthesis (via "My Response" section), this
module:
  1. Validates the response is the user's own words (not AI/LLM)
  2. Stores the response as new Personal source data
  3. Triggers re-synthesis incorporating the new reflection
  4. Returns the updated synthesis

CRITICAL RULES:
  - Only the USER's words become Personal data
  - AI/LLM dialogue is NEVER stored as Personal source class
  - The system proposes; the user disposes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.synthesis.gatherer import PersonalReflectionGatherer
from author_library.synthesis.prompt_engine import SynthesisPromptEngine, SynthesisResult

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SocraticResponse:
    """Result of processing a user's response to a synthesis."""

    response_chunk_id: str
    original_synthesis_theme: str
    response_stored: bool
    re_synthesis: SynthesisResult | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        result: dict[str, Any] = {
            "response_chunk_id": self.response_chunk_id,
            "original_synthesis_theme": self.original_synthesis_theme,
            "response_stored": self.response_stored,
            "message": self.message,
        }
        if self.re_synthesis:
            result["re_synthesis"] = self.re_synthesis.to_dict()
        return result


class SocraticLoop:
    """Processes user responses to synthesis and triggers re-synthesis.

    The Socratic refinement loop:
    1. User reads synthesis
    2. User writes response in "My Response" section
    3. System stores response as Personal data
    4. System re-synthesizes incorporating the new reflection
    5. Repeat until user is satisfied
    """

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._cache = cache_manager

    async def process_response(
        self,
        *,
        user_response: str,
        theme: str,
        synthesis_chunk_id: str | None = None,
        prompt: str = "",
        re_synthesize: bool = True,
    ) -> SocraticResponse:
        """Process a user's response to a synthesis.

        Stores the response as Personal data and optionally re-synthesizes.

        Args:
            user_response: The user's own words responding to the synthesis.
            theme: The theme of the original synthesis.
            synthesis_chunk_id: Optional ID of the synthesis chunk being responded to.
            prompt: Optional prompt for the re-synthesis.
            re_synthesize: Whether to trigger re-synthesis (default True).

        Returns:
            SocraticResponse with stored chunk ID and optional re-synthesis.
        """
        if not user_response or not user_response.strip():
            return SocraticResponse(
                response_chunk_id="",
                original_synthesis_theme=theme,
                response_stored=False,
                message="Empty response — nothing to store.",
            )

        # Step 1: Store user response as Personal data
        response_chunk_id = await self._store_personal_response(
            user_response=user_response,
            theme=theme,
            synthesis_chunk_id=synthesis_chunk_id,
        )

        if not response_chunk_id:
            return SocraticResponse(
                response_chunk_id="",
                original_synthesis_theme=theme,
                response_stored=False,
                message="Failed to store response.",
            )

        log.info(
            "socratic_response_stored",
            chunk_id=response_chunk_id,
            theme=theme,
        )

        # Step 2: Optionally re-synthesize
        re_synthesis: SynthesisResult | None = None
        if re_synthesize:
            re_synthesis = await self._re_synthesize(
                theme=theme,
                prompt=prompt or f"What do I think about {theme}?",
            )

        return SocraticResponse(
            response_chunk_id=response_chunk_id,
            original_synthesis_theme=theme,
            response_stored=True,
            re_synthesis=re_synthesis,
            message="Your response has been recorded as Personal data.",
        )

    async def _store_personal_response(
        self,
        *,
        user_response: str,
        theme: str,
        synthesis_chunk_id: str | None = None,
    ) -> str:
        """Store the user's response as a Personal source chunk.

        Creates a new chunk with source_class='personal' and
        section_type='my_response'. If synthesis_chunk_id is provided,
        creates a USER_REFLECTS_ON edge in the graph.

        Returns:
            The new chunk ID, or empty string on failure.
        """
        # Determine work_id for the response
        work_id = f"personal--synthesis-responses"

        metadata = {
            "section_type": "my_response",
            "themes": [theme] if theme else [],
            "source_synthesis_chunk_id": synthesis_chunk_id or "",
        }

        try:
            # Store the chunk
            chunk_id = await self._storage.pg.fetch_val(
                """INSERT INTO chunks (work_id, text, source_class, granularity, metadata)
                VALUES ($1, $2, 'personal', 'micro', $3::jsonb)
                RETURNING id::text""",
                work_id,
                user_response,
                _to_json(metadata),
            )

            if not chunk_id:
                return ""

            # Create graph edge if we have a synthesis reference
            if synthesis_chunk_id:
                try:
                    await self._storage.graph.create_user_reflects_on_edge(
                        personal_chunk_id=str(chunk_id),
                        target_id=synthesis_chunk_id,
                        target_key="chunk_id",
                        target_label="Chunk",
                    )
                except Exception:
                    log.debug(
                        "graph_edge_creation_failed",
                        chunk_id=str(chunk_id),
                        target=synthesis_chunk_id,
                    )

            # Generate embedding for the new chunk
            try:
                from author_library.retrieval.vector_search import vector_search

                embedding = await self._embedding_provider.embed_text(user_response)
                await self._storage.embeddings.store(
                    chunk_id=chunk_id,
                    embedding=embedding,
                )
            except Exception:
                log.debug("embedding_generation_failed", chunk_id=str(chunk_id))

            return str(chunk_id)

        except Exception:
            log.error("personal_response_storage_failed", theme=theme)
            return ""

    async def _re_synthesize(
        self,
        *,
        theme: str,
        prompt: str,
    ) -> SynthesisResult | None:
        """Re-run synthesis incorporating the new reflection."""
        try:
            gatherer = PersonalReflectionGatherer(
                settings=self._settings,
                storage=self._storage,
                embedding_provider=self._embedding_provider,
                cache_manager=self._cache,
            )

            gathered = await gatherer.gather(theme=theme, prompt=prompt)

            if not gathered.reflections:
                return None

            engine = SynthesisPromptEngine(settings=self._settings)
            return await engine.synthesize(
                gathered,
                theme=theme,
                prompt=prompt,
            )
        except Exception:
            log.error("re_synthesis_failed", theme=theme)
            return None


def _to_json(data: Any) -> str:
    """Convert to JSON string for PostgreSQL jsonb parameter."""
    import json

    return json.dumps(data)
