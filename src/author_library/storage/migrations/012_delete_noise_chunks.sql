-- 011_delete_noise_chunks.sql
-- Remove noise micro/nano chunks that predate the section_type and
-- min_chunk_size filters added to the ingestion pipeline.

-- Step 1: Delete micro/nano chunks from non-content sections
DELETE FROM chunks
WHERE granularity IN ('micro', 'nano')
  AND metadata->>'section_type' IN ('index', 'bibliography', 'toc', 'front_matter');

-- Step 2: Delete remaining tiny fragments (footnote orphans, section markers, page numbers)
DELETE FROM chunks
WHERE granularity IN ('micro', 'nano')
  AND length(text) < 50;
