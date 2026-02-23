-- 007_pass_number.sql
-- Add pass_number to chunks for multi-pass engagement tracking.
-- Each re-ingestion or new capture increments the pass number.
-- engagement_passes on works is the max pass_number across its chunks.

-- Add pass_number to chunks (default 1 for existing rows)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS pass_number INT NOT NULL DEFAULT 1;

-- Add engagement_passes to works (derived from max pass_number)
ALTER TABLE works ADD COLUMN IF NOT EXISTS engagement_passes INT NOT NULL DEFAULT 1;

-- Index for efficient pass_number queries
CREATE INDEX IF NOT EXISTS idx_chunks_pass_number ON chunks(work_id, pass_number);
