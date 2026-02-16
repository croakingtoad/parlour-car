-- 003_fulltext.sql
-- Full-text search support via tsvector generated column and GIN index.

ALTER TABLE chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX idx_chunks_search ON chunks USING GIN(search_vector);
