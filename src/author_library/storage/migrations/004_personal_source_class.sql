-- 004_personal_source_class.sql
-- Add 'personal' to source_class CHECK constraint on works table.
-- Personal sources are user reflections/notes that NEVER contribute to voice
-- profiles and are NEVER attributed to the subject author.

-- Drop and recreate the CHECK constraint to include 'personal'
ALTER TABLE works DROP CONSTRAINT IF EXISTS works_source_class_check;
ALTER TABLE works ADD CONSTRAINT works_source_class_check
    CHECK (source_class IN ('primary', 'secondary', 'contextual', 'tertiary', 'personal'));
