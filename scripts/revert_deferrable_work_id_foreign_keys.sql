-- Revert 016_deferrable_work_id_foreign_keys.sql.
-- Lock: ACCESS EXCLUSIVE briefly on each existing child table, one at a time.
-- Duration: catalog-only constraint metadata changes; no table rewrite or validation scan.
-- Preconditions: run only after all work_id rename transactions using deferred checks have ended.

DO $$
DECLARE
    target record;
BEGIN
    FOR target IN
        SELECT *
        FROM (VALUES
            ('public', 'chunks', 'chunks_work_id_fkey'),
            ('public', 'session_sources', 'session_sources_work_id_fkey'),
            ('public', 'thematic_appearances', 'thematic_appearances_work_id_fkey'),
            ('public', 'transcript_cache', 'transcript_cache_work_id_fkey')
        ) AS constraints(schema_name, table_name, constraint_name)
    LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_constraint AS con
            JOIN pg_class AS rel ON rel.oid = con.conrelid
            JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace
            JOIN pg_class AS parent ON parent.oid = con.confrelid
            WHERE con.contype = 'f'
              AND ns.nspname = target.schema_name
              AND rel.relname = target.table_name
              AND con.conname = target.constraint_name
              AND parent.oid = 'works'::regclass
              AND con.condeferrable
        ) THEN
            EXECUTE format(
                'ALTER TABLE %I.%I ALTER CONSTRAINT %I NOT DEFERRABLE',
                target.schema_name,
                target.table_name,
                target.constraint_name
            );
        END IF;
    END LOOP;
END
$$;
