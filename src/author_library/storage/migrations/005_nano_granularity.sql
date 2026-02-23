-- 005_nano_granularity.sql
-- Add 'nano' granularity tier and raw_content fields to chunks table.
-- Nano chunks are internal pipeline artifacts (raw captures, timestamps, moments)
-- that never get their own note — they are embedded in the consuming capture's
-- <details> block.

-- Update granularity CHECK constraint to include 'nano'
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_granularity_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_granularity_check
    CHECK (granularity IN ('macro', 'meso', 'micro', 'nano'));

-- Add raw_content fields for nano chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS raw_content TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS raw_content_window TEXT;
