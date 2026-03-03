-- 010_search_vector_annotation.sql
-- Update full-text search vector to include annotation content.
-- The annotation contains source-class markers and contextual metadata
-- that improve retrieval quality when searching for thematic or
-- source-specific content.

-- Drop existing index and generated column, then recreate with annotation.
DROP INDEX IF EXISTS idx_chunks_search;
ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector;

ALTER TABLE chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(annotation, '') || ' ' || text)
    ) STORED;

CREATE INDEX idx_chunks_search ON chunks USING GIN(search_vector);
