"""Shared fixtures for retrieval engine tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from author_library.embeddings.base import (
    BatchEmbeddingResult,
    EmbeddingProvider,
    EmbeddingResult,
)
from author_library.retrieval.models import (
    GraphExpansionResult,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Deterministic embedding provider for tests
# ---------------------------------------------------------------------------


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that returns deterministic vectors for testing.

    Generates a stable vector based on the hash of the input text,
    so the same text always produces the same embedding.
    """

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def dimensions(self) -> int:
        return 8

    async def embed_text(self, text: str) -> EmbeddingResult:
        vector = self._deterministic_vector(text)
        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            provider=self.provider_name,
            dimensions=self.dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        vectors = [self._deterministic_vector(t) for t in texts]
        return BatchEmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            provider=self.provider_name,
            dimensions=self.dimensions,
        )

    async def embed_query(self, text: str) -> EmbeddingResult:
        return await self.embed_text(text)

    def _deterministic_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit vector from text hash."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h[: self.dimensions]]
        # Normalize to unit vector
        magnitude = sum(x * x for x in raw) ** 0.5
        if magnitude > 0:
            raw = [x / magnitude for x in raw]
        return raw


# ---------------------------------------------------------------------------
# In-memory embedding repository for tests
# ---------------------------------------------------------------------------


class InMemoryEmbeddingRepository:
    """In-memory embedding repository for testing vector search.

    Stores embeddings in a dict and implements cosine distance search.
    """

    def __init__(self) -> None:
        self._embeddings: dict[UUID, dict[str, Any]] = {}

    def add_embedding(
        self,
        chunk_id: UUID,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
        text: str,
        work_id: str,
        granularity: str,
        source_class: str,
        pass_number: int = 1,
        speaker: str | None = None,
    ) -> UUID:
        """Store an embedding for testing."""
        emb_id = uuid4()
        self._embeddings[emb_id] = {
            "chunk_id": chunk_id,
            "embedding": embedding,
            "provider": provider,
            "model": model,
            "dimensions": dimensions,
            "text": text,
            "work_id": work_id,
            "granularity": granularity,
            "source_class": source_class,
            "pass_number": pass_number,
            "speaker": speaker,
        }
        return emb_id

    async def store(
        self,
        chunk_id: UUID,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
    ) -> UUID:
        # Simplified store — in real tests the chunk text is pre-loaded
        emb_id = uuid4()
        self._embeddings[emb_id] = {
            "chunk_id": chunk_id,
            "embedding": embedding,
            "provider": provider,
            "model": model,
            "dimensions": dimensions,
            "text": "",
            "work_id": "",
            "granularity": "meso",
            "source_class": "primary",
        }
        return emb_id

    async def get_by_chunk(self, chunk_id: UUID) -> list[dict[str, Any]]:
        return [
            v for v in self._embeddings.values() if v["chunk_id"] == chunk_id
        ]

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        provider: str,
        model: str,
        limit: int = 20,
        source_class_filter: str | None = None,
        subject_headings_filter: list[str] | None = None,
        genre_tags_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine distance search over in-memory embeddings."""
        candidates: list[tuple[float, dict[str, Any]]] = []

        for emb_data in self._embeddings.values():
            if emb_data["provider"] != provider or emb_data["model"] != model:
                continue
            if source_class_filter and emb_data["source_class"] != source_class_filter:
                continue

            distance = _cosine_distance(query_embedding, emb_data["embedding"])
            candidates.append((distance, emb_data))

        candidates.sort(key=lambda x: x[0])

        return [
            {
                "chunk_id": c["chunk_id"],
                "work_id": c["work_id"],
                "text": c["text"],
                "granularity": c["granularity"],
                "source_class": c["source_class"],
                "pass_number": c.get("pass_number", 1),
                "speaker": c.get("speaker"),
                "distance": dist,
            }
            for dist, c in candidates[:limit]
        ]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 1.0
    similarity = dot / (mag_a * mag_b)
    return 1.0 - similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNK_IDS = [uuid4() for _ in range(10)]


@pytest.fixture()
def embedding_provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider()


@pytest.fixture()
def embedding_repo(
    embedding_provider: DeterministicEmbeddingProvider,
) -> InMemoryEmbeddingRepository:
    """Pre-populated in-memory embedding repo with literary test data."""
    import asyncio

    repo = InMemoryEmbeddingRepository()

    chunks = [
        {
            "id": CHUNK_IDS[0],
            "text": (
                "The Weight of Glory is perhaps Lewis's most eloquent "
                "sermon, delivered at the Church of St Mary the Virgin."
            ),
            "work_id": "lewis--weight-of-glory",
            "granularity": "meso",
            "source_class": "primary",
        },
        {
            "id": CHUNK_IDS[1],
            "text": (
                "In Mere Christianity Lewis argues that the moral law "
                "points to a lawgiver beyond nature."
            ),
            "work_id": "lewis--mere-christianity",
            "granularity": "meso",
            "source_class": "primary",
        },
        {
            "id": CHUNK_IDS[2],
            "text": (
                "Tolkien's influence on Lewis's conversion is well documented "
                "in their correspondence."
            ),
            "work_id": "mcgrath--cs-lewis-biography",
            "granularity": "meso",
            "source_class": "secondary",
        },
        {
            "id": CHUNK_IDS[3],
            "text": (
                "George MacDonald's Phantastes baptized Lewis's imagination "
                "long before his intellect was converted."
            ),
            "work_id": "macdonald--phantastes",
            "granularity": "meso",
            "source_class": "contextual",
        },
        {
            "id": CHUNK_IDS[4],
            "text": "Mere Christianity was published in 1952 by Geoffrey Bles.",
            "work_id": "lewis--mere-christianity",
            "granularity": "micro",
            "source_class": "primary",
        },
        {
            "id": CHUNK_IDS[5],
            "text": (
                "Joy is the serious business of Heaven. "
                "Lewis returns to this theme across many works."
            ),
            "work_id": "lewis--letters-to-malcolm",
            "granularity": "micro",
            "source_class": "primary",
        },
        {
            "id": CHUNK_IDS[6],
            "text": (
                "Lewis's treatment of the imagination draws heavily on "
                "the Romantic tradition, particularly Coleridge and Wordsworth."
            ),
            "work_id": "lewis--surprised-by-joy",
            "granularity": "macro",
            "source_class": "primary",
        },
    ]

    for chunk in chunks:
        result = asyncio.get_event_loop().run_until_complete(
            embedding_provider.embed_text(chunk["text"])
        )
        repo.add_embedding(
            chunk_id=chunk["id"],
            embedding=result.vector,
            provider=result.provider,
            model=result.model,
            dimensions=result.dimensions,
            text=chunk["text"],
            work_id=chunk["work_id"],
            granularity=chunk["granularity"],
            source_class=chunk["source_class"],
        )

    return repo


@pytest.fixture()
def sample_retrieval_results() -> list[RetrievalResult]:
    """Sample retrieval results for fusion and orchestration tests."""
    return [
        RetrievalResult(
            chunk_id=CHUNK_IDS[0],
            work_id="lewis--weight-of-glory",
            text=(
                "The Weight of Glory is perhaps Lewis's most eloquent "
                "sermon, delivered at the Church of St Mary the Virgin."
            ),
            score=0.92,
            granularity="meso",
            source_class="primary",
            source="vector",
        ),
        RetrievalResult(
            chunk_id=CHUNK_IDS[1],
            work_id="lewis--mere-christianity",
            text=(
                "In Mere Christianity Lewis argues that the moral law "
                "points to a lawgiver beyond nature."
            ),
            score=0.88,
            granularity="meso",
            source_class="primary",
            source="vector",
        ),
        RetrievalResult(
            chunk_id=CHUNK_IDS[2],
            work_id="mcgrath--cs-lewis-biography",
            text=(
                "Tolkien's influence on Lewis's conversion is well documented "
                "in their correspondence."
            ),
            score=0.75,
            granularity="meso",
            source_class="secondary",
            source="vector",
        ),
    ]


@pytest.fixture()
def sample_graph_expansions() -> list[GraphExpansionResult]:
    """Sample graph expansion results."""
    return [
        GraphExpansionResult(
            chunk_id=str(CHUNK_IDS[3]),
            work_id="macdonald--phantastes",
            text_preview=(
                "George MacDonald's Phantastes baptized Lewis's imagination "
                "long before his intellect was converted."
            ),
            granularity="meso",
            source_class="contextual",
            relationship_type="ENGAGES_WITH",
            confidence="high",
            evidence="Lewis explicitly cites MacDonald as formative influence",
        ),
    ]
