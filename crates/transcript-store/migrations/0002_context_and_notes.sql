-- Session context, title provenance and AI notes.

ALTER TABLE sessions ADD COLUMN title_source TEXT NOT NULL DEFAULT 'default';
ALTER TABLE sessions ADD COLUMN context TEXT NOT NULL DEFAULT '';

-- Titles that differ from the generated default were typed by the user; AI
-- titles must never replace them.
UPDATE sessions
SET title_source = 'user'
WHERE title <> source_language || ' → ' || target_language || ' lecture';

CREATE TABLE session_notes (
    session_id TEXT PRIMARY KEY NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    markdown TEXT NOT NULL,
    language TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    attachments_json TEXT NOT NULL DEFAULT '[]',
    usage_json TEXT NOT NULL DEFAULT '{}',
    prompt_version TEXT NOT NULL DEFAULT '',
    source_chars INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
