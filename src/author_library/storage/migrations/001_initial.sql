-- 001_initial.sql
-- Core schema for The Author Library: authors, works, chunks, embeddings,
-- thematic index, and voice profiles.

-- Authors
CREATE TABLE authors (
    id TEXT PRIMARY KEY,  -- slug format: malcolm-guite
    canonical_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Works (catalog entries per catalog-schema.md)
CREATE TABLE works (
    work_id TEXT PRIMARY KEY,  -- format: author-slug--title-slug
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    source_class TEXT NOT NULL CHECK (source_class IN ('primary', 'secondary', 'contextual', 'tertiary')),
    source_class_note TEXT NOT NULL CHECK (LENGTH(source_class_note) >= 10),
    publication_year INT NOT NULL,
    original_publication_year INT,
    edition TEXT,
    publisher TEXT NOT NULL,
    isbn TEXT,
    format_ingested TEXT NOT NULL CHECK (format_ingested IN ('epub', 'pdf', 'txt', 'html', 'docx')),
    language TEXT NOT NULL DEFAULT 'en',
    word_count INT NOT NULL,
    genre_tags TEXT[] NOT NULL CHECK (array_length(genre_tags, 1) >= 1),
    subject_headings TEXT[] NOT NULL CHECK (array_length(subject_headings, 1) >= 1),
    ocr_quality TEXT CHECK (ocr_quality IN ('high', 'medium', 'low', 'not-applicable')),
    ingestion_date DATE NOT NULL DEFAULT CURRENT_DATE,
    notes TEXT,
    -- Source-class specific fields stored as JSONB
    source_metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks (multi-granularity)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id TEXT NOT NULL REFERENCES works(work_id),
    text TEXT NOT NULL,
    annotation TEXT,  -- contextual annotation prepended before embedding
    granularity TEXT NOT NULL CHECK (granularity IN ('macro', 'meso', 'micro')),
    source_class TEXT NOT NULL,
    chapter TEXT,
    section TEXT,
    position INT NOT NULL,  -- ordering within the work at this granularity
    parent_chunk_id UUID REFERENCES chunks(id),  -- parent in granularity hierarchy
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunk embeddings (pgvector)
CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(1024),  -- default dimension; variable dimensions supported
    provider TEXT NOT NULL,  -- voyage, openai, ollama
    model TEXT NOT NULL,
    dimensions INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, provider, model)
);

-- Thematic index entries
CREATE TABLE thematic_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_id TEXT NOT NULL REFERENCES authors(id),
    theme TEXT NOT NULL,
    author_stance TEXT,
    related_themes TEXT[],
    key_passages JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Thematic appearances (junction: theme × work)
CREATE TABLE thematic_appearances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id UUID NOT NULL REFERENCES thematic_entries(id) ON DELETE CASCADE,
    work_id TEXT NOT NULL REFERENCES works(work_id),
    chapters TEXT[],
    treatment_summary TEXT,
    UNIQUE(entry_id, work_id)
);

-- Voice profiles
CREATE TABLE voice_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_id TEXT NOT NULL REFERENCES authors(id),
    version INT NOT NULL DEFAULT 1,
    profile JSONB NOT NULL,  -- structured voice profile JSON
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(author_id, version)
);
