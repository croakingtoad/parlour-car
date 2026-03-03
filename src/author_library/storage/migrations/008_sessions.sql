-- 008_sessions.sql
-- Sessions table for tracking engagement sessions (reading/viewing/listening).
-- Auto-start: first capture event starts a session.
-- Auto-end: 60min inactivity (configurable) or theme change + >30min gap.

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    date_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date_end TIMESTAMPTZ,
    duration_minutes INT,
    user_id TEXT NOT NULL DEFAULT 'marty',  -- single-user V1, schema-ready for multi
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Junction: session ↔ chunks (captures within a session)
CREATE TABLE IF NOT EXISTS session_captures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    capture_order INT NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, chunk_id)
);

-- Junction: session ↔ works (sources engaged during a session)
CREATE TABLE IF NOT EXISTS session_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    work_id TEXT NOT NULL REFERENCES works(work_id) ON DELETE CASCADE,
    UNIQUE(session_id, work_id)
);

-- Indexes for efficient session queries
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date_start ON sessions(date_start);
CREATE INDEX IF NOT EXISTS idx_session_captures_session_id ON session_captures(session_id);
CREATE INDEX IF NOT EXISTS idx_session_captures_chunk_id ON session_captures(chunk_id);
CREATE INDEX IF NOT EXISTS idx_session_sources_session_id ON session_sources(session_id);
CREATE INDEX IF NOT EXISTS idx_session_sources_work_id ON session_sources(work_id);
