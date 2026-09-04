-- Permit honest NULLs when a publication year is absent or PDF Producer
-- metadata is rejected as a publisher.
-- Lock: ACCESS EXCLUSIVE on works for two metadata-only column changes.
-- Duration: expected to be brief at production scale; no table rewrite or row update.
-- Reversible: restore values for any NULL rows before applying SET NOT NULL.

ALTER TABLE works
    ALTER COLUMN publication_year DROP NOT NULL,
    ALTER COLUMN publisher DROP NOT NULL;
