PRAGMA foreign_keys = ON;

CREATE TABLE sessions (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    audio_source TEXT NOT NULL,
    audio_profile TEXT NOT NULL,
    inference_mode TEXT NOT NULL,
    asr_backend TEXT NOT NULL,
    translation_backend TEXT NOT NULL,
    route_reason TEXT NOT NULL DEFAULT '',
    privacy_json TEXT NOT NULL,
    model_config_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE segments (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start_ms REAL NOT NULL,
    end_ms REAL NOT NULL,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL DEFAULT '',
    source_final INTEGER NOT NULL DEFAULT 0,
    target_final INTEGER NOT NULL DEFAULT 0,
    speaker_id TEXT,
    asr_confidence REAL,
    source_revision INTEGER NOT NULL,
    target_revision INTEGER NOT NULL DEFAULT 0,
    timestamp_quality TEXT NOT NULL DEFAULT 'none',
    word_timings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, source_revision)
);

CREATE INDEX segments_session_ordinal ON segments(session_id, ordinal);

CREATE TABLE revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    segment_id TEXT REFERENCES segments(id) ON DELETE CASCADE,
    stream TEXT NOT NULL,
    revision_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    backend TEXT NOT NULL,
    latency_json TEXT NOT NULL DEFAULT '{}',
    emitted_at TEXT NOT NULL
);

CREATE TABLE metric_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    captured_audio_ms REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE session_search USING fts5(
    session_id UNINDEXED,
    title,
    source_text,
    translated_text,
    tokenize = 'unicode61'
);
