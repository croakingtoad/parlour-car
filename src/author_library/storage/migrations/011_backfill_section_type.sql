-- 010_backfill_section_type.sql
-- Backfill section_type into chunk metadata for pre-existing chunks.
-- New chunks already get section_type via the ingestion pipeline.

UPDATE chunks
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{section_type}',
    CASE
        WHEN chapter ILIKE '%index%' THEN '"index"'
        WHEN chapter ILIKE '%bibliograph%' OR chapter ILIKE '%works cited%'
             OR chapter ILIKE '%references%' OR chapter ILIKE '%further reading%' THEN '"bibliography"'
        WHEN chapter ILIKE '%contents%' THEN '"toc"'
        WHEN chapter ILIKE '%copyright%' OR chapter ILIKE '%dedication%'
             OR chapter ILIKE '%acknowledgement%' THEN '"front_matter"'
        WHEN chapter ILIKE '%preface%' OR chapter ILIKE '%foreword%' THEN '"preface"'
        ELSE '"chapter"'
    END::jsonb
)
WHERE metadata IS NULL
   OR NOT (metadata ? 'section_type');
