-- 015_vocabulary_terms.sql
-- Durable schema for the controlled vocabulary previously created lazily by
-- VocabularyManager.
--
-- Locks: CREATE TABLE IF NOT EXISTS and each ADD CONSTRAINT take an
-- ACCESS EXCLUSIVE lock on vocabulary_terms. At the observed production size
-- (1,300 rows) the expected duration is under one second; apply only after
-- the normal production lock-window check.
--
-- Reversible: yes. Tested inverse: drop vocabulary_terms_status_check and
-- vocabulary_terms_merged_into_check. Drop the table only when this migration
-- created it; never drop a pre-existing production table as rollback.
--
-- Production preflight on 2026-09-03: 1,300 rows, all status='proposed', and
-- no merged rows. Both constraints therefore accept every existing row.

CREATE TABLE IF NOT EXISTS vocabulary_terms (
    id          SERIAL PRIMARY KEY,
    term        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'proposed',
    merged_into TEXT,
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'public.vocabulary_terms'::regclass
          AND conname = 'vocabulary_terms_status_check'
    ) THEN
        ALTER TABLE vocabulary_terms
            ADD CONSTRAINT vocabulary_terms_status_check
            CHECK (status IN ('proposed', 'canonical', 'deprecated', 'merged'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'public.vocabulary_terms'::regclass
          AND conname = 'vocabulary_terms_merged_into_check'
    ) THEN
        ALTER TABLE vocabulary_terms
            ADD CONSTRAINT vocabulary_terms_merged_into_check
            CHECK ((status = 'merged') = (merged_into IS NOT NULL));
    END IF;
END
$$;

-- Rollback (tested against the disposable migration database):
-- ALTER TABLE vocabulary_terms DROP CONSTRAINT vocabulary_terms_merged_into_check;
-- ALTER TABLE vocabulary_terms DROP CONSTRAINT vocabulary_terms_status_check;
