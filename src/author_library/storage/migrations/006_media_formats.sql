-- 006_media_formats.sql
-- Add media format types (video, audio, transcript, youtube_captions) to
-- format_ingested CHECK constraint, add source/media fields to works table,
-- and create transcript cache storage.

-- Update format_ingested CHECK constraint
ALTER TABLE works DROP CONSTRAINT IF EXISTS works_format_ingested_check;
ALTER TABLE works ADD CONSTRAINT works_format_ingested_check
    CHECK (format_ingested IN (
        'epub', 'pdf', 'txt', 'html', 'docx',
        'video', 'audio', 'transcript', 'youtube_captions'
    ));

-- Add source/media fields to works table
ALTER TABLE works ADD COLUMN IF NOT EXISTS url TEXT;
ALTER TABLE works ADD COLUMN IF NOT EXISTS duration INT;  -- seconds
ALTER TABLE works ADD COLUMN IF NOT EXISTS speakers TEXT[];
ALTER TABLE works ADD COLUMN IF NOT EXISTS date_published DATE;
ALTER TABLE works ADD COLUMN IF NOT EXISTS date_consumed DATE;
ALTER TABLE works ADD COLUMN IF NOT EXISTS transcript_cached BOOLEAN DEFAULT FALSE;
ALTER TABLE works ADD COLUMN IF NOT EXISTS media TEXT
    CHECK (media IS NULL OR media IN ('book', 'video', 'audio', 'podcast', 'article'));

-- Transcript cache storage keyed by source URL
CREATE TABLE IF NOT EXISTS transcript_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url TEXT NOT NULL UNIQUE,
    work_id TEXT REFERENCES works(work_id),
    transcript_text TEXT NOT NULL,
    format TEXT NOT NULL CHECK (format IN ('transcript', 'youtube_captions', 'srt', 'vtt')),
    language TEXT NOT NULL DEFAULT 'en',
    speaker_labels BOOLEAN DEFAULT FALSE,
    duration INT,  -- seconds
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcript_cache_work_id ON transcript_cache(work_id);
