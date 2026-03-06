-- Ingestion lessons: store recurring quality problems and their fixes
-- so the pipeline can learn from past mistakes.

CREATE TABLE IF NOT EXISTS ingestion_lessons (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_type    TEXT NOT NULL,
    detection_method TEXT NOT NULL,
    trigger_context JSONB NOT NULL DEFAULT '{}',
    problem_description TEXT NOT NULL,
    fix_applied     TEXT NOT NULL,
    prevention_rule TEXT,
    prevention_step TEXT,
    confidence      FLOAT DEFAULT 0.5,
    times_applied   INT DEFAULT 0,
    times_prevented INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_applied_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ingestion_lessons_problem_type
    ON ingestion_lessons (problem_type);

CREATE INDEX IF NOT EXISTS idx_ingestion_lessons_prevention_step
    ON ingestion_lessons (prevention_step);

CREATE INDEX IF NOT EXISTS idx_ingestion_lessons_active_confidence
    ON ingestion_lessons (is_active, confidence DESC);

-- Seed lessons from recent ingestion sessions
INSERT INTO ingestion_lessons (
    problem_type, detection_method, trigger_context,
    problem_description, fix_applied,
    prevention_rule, prevention_step, confidence
) VALUES
(
    'misclassification',
    'manual_review',
    '{"source_class": "contextual", "expected": "primary", "pattern": "devotional works by subject author"}',
    'Devotional works written by the subject author were misclassified as contextual instead of primary. The classifier treated religious/devotional content as secondary material when the author field matched the subject author.',
    'Reclassified affected works to primary source class and re-ran voice profile extraction to include their content.',
    'When author field matches the subject author, always classify as primary regardless of genre or content type.',
    'classification',
    0.8
),
(
    'theme_explosion',
    'graph_inspection',
    '{"node_type": "Theme", "symptom": "duplicate canonical_names with minor variations"}',
    'Theme canonical_name values proliferated without deduplication — near-identical themes like "grace" and "Grace of God" and "divine grace" each got separate Theme nodes instead of merging under a single canonical form.',
    'Ran deduplicate_themes() with cosine similarity threshold 0.85 to merge near-duplicate Theme nodes.',
    'Run theme deduplication after every entity extraction batch. Use cosine similarity >= 0.85 to detect duplicates.',
    'entity_extraction',
    0.8
),
(
    'llm_format_violation',
    'response_parsing',
    '{"expected_format": "JSON", "actual_format": "prose", "tool": "thematic_index_mapper"}',
    'LLM returned prose essays instead of JSON when mapping chunks to pervasive themes. The prompt was too open-ended, allowing the model to explain its reasoning in paragraph form rather than structured output.',
    'Tightened the thematic index mapping prompt with explicit JSON schema, added response format enforcement, and wrapped output in json_repair.',
    'Always include explicit JSON schema in thematic mapping prompts. Use json_repair as a safety net for malformed responses.',
    'thematic_index',
    0.9
),
(
    'orphaned_nodes',
    'graph_consistency_check',
    '{"node_type": "Argument", "source_class": "contextual", "symptom": "nodes with zero edges"}',
    'Argument nodes were created for contextual sources but never connected to any other nodes via edges. Contextual sources lack the deep analysis that produces meaningful argument relationships, resulting in orphaned graph nodes.',
    'Deleted orphaned Argument nodes with zero relationships. Reclassified contextual-source entity extraction to skip Argument creation.',
    'Skip Argument node creation for contextual source class. Only primary and secondary sources should produce Argument entities.',
    'entity_extraction',
    0.7
),
(
    'micro_chunk_pollution',
    'search_quality_review',
    '{"chunk_type": "index/bibliography", "char_count_threshold": 50, "symptom": "low-quality search results"}',
    'Index entries, bibliography references, and other structural micro-chunks under 50 characters were polluting vector search results. These fragments have high lexical overlap with real content but carry no semantic value.',
    'Applied filter_min_chunk_size(50) and section_type filters in both composable and pipeline retrieval paths.',
    'Filter chunks under 50 characters at ingestion time. Apply section_type exclusions for index, bibliography, and front-matter sections.',
    'chunking',
    0.85
);
