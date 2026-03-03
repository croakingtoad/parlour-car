-- 009_transcript_cache.sql  (renamed from 004_transcript_cache.sql)
-- Transcript cache for Parlour Chrome video/audio sources.
-- Stores fetched transcripts keyed by source URL with TTL-based invalidation.
-- Uses IF NOT EXISTS for compatibility with parallel epic branches.

CREATE TABLE IF NOT EXISTS transcript_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url TEXT NOT NULL UNIQUE,
    transcript_text TEXT NOT NULL,
    cached_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ttl_seconds INT NOT NULL DEFAULT 86400
);

-- Add TTL columns if table exists from another branch without them
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'transcript_cache' AND column_name = 'cached_at'
    ) THEN
        ALTER TABLE transcript_cache ADD COLUMN cached_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'transcript_cache' AND column_name = 'ttl_seconds'
    ) THEN
        ALTER TABLE transcript_cache ADD COLUMN ttl_seconds INT NOT NULL DEFAULT 86400;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_transcript_cache_source_url ON transcript_cache (source_url);
