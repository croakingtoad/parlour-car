"""Backfill missing embeddings for a specific work."""

import asyncio
import sys

from author_library.config import get_settings
from author_library.embeddings import ProviderRegistry
from author_library.embeddings.base import build_token_aware_batches
from author_library.storage import StorageManager


async def backfill(work_id: str) -> None:
    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)

    provider = ProviderRegistry.create(settings)

    rows = await storage.pg.fetch_all(
        """
        SELECT c.id,
               CASE WHEN c.annotation IS NOT NULL AND c.annotation != ''
                    THEN c.annotation || E'\\n\\n' || c.text
                    ELSE c.text
               END AS embed_text
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        WHERE c.work_id = $1 AND ce.id IS NULL
        ORDER BY c.position
        """,
        work_id,
    )

    print(f"Found {len(rows)} chunks to embed for {work_id}")
    if not rows:
        await storage.close()
        return

    texts = [r["embed_text"] for r in rows]
    chunk_ids = [r["id"] for r in rows]
    batches = build_token_aware_batches(texts)

    total_embedded = 0
    offset = 0
    for i, batch_texts in enumerate(batches):
        batch_ids = chunk_ids[offset : offset + len(batch_texts)]
        offset += len(batch_texts)

        result = await provider.embed_batch(batch_texts)

        for cid, vec in zip(batch_ids, result.vectors):
            await storage.embeddings.store(
                cid,
                vec,
                settings.embedding.provider,
                settings.embedding.model,
                settings.embedding.dimensions,
            )
            total_embedded += 1

        print(
            f"  Batch {i + 1}/{len(batches)}: {len(batch_texts)} chunks "
            f"(total: {total_embedded}/{len(rows)})"
        )

    print(f"Done! Embedded {total_embedded} chunks")
    await storage.close()


if __name__ == "__main__":
    work_id = sys.argv[1] if len(sys.argv) > 1 else (
        "ewan-james-jones--coleridge-and-the-philosophy-of-poetic-form"
    )
    asyncio.run(backfill(work_id))
