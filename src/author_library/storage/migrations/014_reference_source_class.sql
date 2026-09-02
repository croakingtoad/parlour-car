-- Rollback (valid only while no works use 'reference'):
-- ALTER TABLE works DROP CONSTRAINT IF EXISTS works_source_class_check;
-- ALTER TABLE works ADD CONSTRAINT works_source_class_check
--     CHECK (source_class IN ('primary', 'secondary', 'contextual', 'tertiary', 'personal'));
--
-- 014_reference_source_class.sql
-- Add 'reference' to source_class CHECK constraint on works table.

-- Drop and recreate the CHECK constraint to include 'reference'
ALTER TABLE works DROP CONSTRAINT IF EXISTS works_source_class_check;
ALTER TABLE works ADD CONSTRAINT works_source_class_check
    CHECK (source_class IN ('primary', 'secondary', 'contextual', 'tertiary', 'personal', 'reference'));
