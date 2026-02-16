-- 002_indexes.sql
-- B-tree and HNSW indexes for query performance.

-- Works indexes
CREATE INDEX idx_works_source_class ON works(source_class);
CREATE INDEX idx_works_author ON works(author);

-- Chunks indexes
CREATE INDEX idx_chunks_work_id ON chunks(work_id);
CREATE INDEX idx_chunks_granularity ON chunks(granularity);
CREATE INDEX idx_chunks_source_class ON chunks(source_class);

-- Chunk embeddings indexes
CREATE INDEX idx_chunk_embeddings_chunk_id ON chunk_embeddings(chunk_id);

-- HNSW index for vector cosine similarity search
CREATE INDEX idx_chunk_embeddings_vector ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
