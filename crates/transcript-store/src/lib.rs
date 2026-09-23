//! SQLite/WAL transcript persistence and revision-aware export.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{FromRow, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;
use thiserror::Error;
use uuid::Uuid;

pub const SCHEMA_VERSION: u32 = 2;

/// Where a session title came from (`sessions.title_source`).
pub const TITLE_SOURCE_DEFAULT: &str = "default";
/// Typed by the user; AI titles never replace it.
pub const TITLE_SOURCE_USER: &str = "user";
/// Written by the AI assistant.
pub const TITLE_SOURCE_AI: &str = "ai";

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("SQLite error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("SQLite migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),
    #[error("cannot create transcript store directory: {0}")]
    Directory(#[from] std::io::Error),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("cannot encode stored JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("this session has no AI notes yet")]
    NotesNotFound(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDraft {
    pub id: Uuid,
    pub title: String,
    pub source_language: String,
    pub target_language: String,
    pub audio_source: String,
    pub audio_profile: String,
    pub inference_mode: String,
    pub asr_backend: String,
    pub translation_backend: String,
    pub route_reason: String,
    pub privacy: serde_json::Value,
    pub model_config: serde_json::Value,
    /// Per-lecture topic and terms the session ran with.
    #[serde(default)]
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDraft {
    pub id: Uuid,
    pub session_id: Uuid,
    pub ordinal: i64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub source_text: String,
    pub source_final: bool,
    pub asr_confidence: Option<f64>,
    pub source_revision: i64,
    pub timestamp_quality: String,
    pub word_timings: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SessionRecord {
    pub id: String,
    pub title: String,
    pub status: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub source_language: String,
    pub target_language: String,
    pub audio_source: String,
    pub audio_profile: String,
    pub inference_mode: String,
    pub asr_backend: String,
    pub translation_backend: String,
    pub route_reason: String,
    pub privacy_json: String,
    pub model_config_json: String,
    pub warnings_json: String,
    pub created_at: String,
    pub updated_at: String,
    /// `default` (generated), `user` (renamed) or `ai` (assistant title).
    pub title_source: String,
    /// The lecture context the session ran with.
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SegmentRecord {
    pub id: String,
    pub session_id: String,
    pub ordinal: i64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub source_text: String,
    pub translated_text: String,
    pub source_final: bool,
    pub target_final: bool,
    pub speaker_id: Option<String>,
    pub asr_confidence: Option<f64>,
    pub source_revision: i64,
    pub target_revision: i64,
    pub timestamp_quality: String,
    pub word_timings_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDetail {
    pub session: SessionRecord,
    pub segments: Vec<SegmentRecord>,
}

/// AI notes to store for one session (one row per session; saving replaces).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionNotesDraft {
    pub session_id: Uuid,
    pub markdown: String,
    /// Language the notes are written in (the session's target language).
    pub language: String,
    pub provider: String,
    pub model: String,
    /// Per-attachment reports from the sidecar (`[{id, name, pages, ...}]`).
    #[serde(default)]
    pub attachments: serde_json::Value,
    /// `{prompt_tokens, completion_tokens}` as reported by the provider.
    #[serde(default)]
    pub usage: serde_json::Value,
    #[serde(default)]
    pub prompt_version: String,
    #[serde(default)]
    pub source_chars: i64,
}

/// Saved AI notes, shaped for the UI (`SessionNotes` in `types.ts`):
/// attachments and usage are decoded JSON rather than strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionNotes {
    pub session_id: String,
    pub markdown: String,
    pub language: String,
    pub provider: String,
    pub model: String,
    pub attachments: serde_json::Value,
    pub usage: serde_json::Value,
    pub prompt_version: String,
    pub source_chars: i64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow)]
struct SessionNotesRow {
    session_id: String,
    markdown: String,
    language: String,
    provider: String,
    model: String,
    attachments_json: String,
    usage_json: String,
    prompt_version: String,
    source_chars: i64,
    created_at: String,
    updated_at: String,
}

impl From<SessionNotesRow> for SessionNotes {
    fn from(row: SessionNotesRow) -> Self {
        // Stored JSON was written by `save_notes`; a damaged value degrades
        // to an empty container instead of hiding the notes themselves.
        let decode = |text: &str, fallback: serde_json::Value| {
            serde_json::from_str(text).unwrap_or(fallback)
        };
        Self {
            attachments: decode(&row.attachments_json, serde_json::json!([])),
            usage: decode(&row.usage_json, serde_json::json!({})),
            session_id: row.session_id,
            markdown: row.markdown,
            language: row.language,
            provider: row.provider,
            model: row.model,
            prompt_version: row.prompt_version,
            source_chars: row.source_chars,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Markdown,
    Txt,
    Json,
    Srt,
    Vtt,
    /// The saved AI notes alone, as Markdown.
    Notes,
}

#[derive(Debug, Clone)]
pub struct TranscriptStore {
    pool: SqlitePool,
}

impl TranscriptStore {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = format!("sqlite://{}", path.to_string_lossy());
        let options = SqliteConnectOptions::from_str(&url)?
            .create_if_missing(true)
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        sqlx::migrate!().run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn create_session(&self, draft: &SessionDraft) -> Result<SessionRecord, StoreError> {
        let now = timestamp(Utc::now());
        sqlx::query(
            "INSERT INTO sessions (id,title,status,started_at,source_language,target_language,audio_source,audio_profile,inference_mode,asr_backend,translation_backend,route_reason,privacy_json,model_config_json,created_at,updated_at,title_source,context) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        .bind(draft.id.to_string()).bind(&draft.title).bind("active").bind(&now)
        .bind(&draft.source_language).bind(&draft.target_language).bind(&draft.audio_source)
        .bind(&draft.audio_profile).bind(&draft.inference_mode).bind(&draft.asr_backend)
        .bind(&draft.translation_backend).bind(&draft.route_reason)
        .bind(serde_json::to_string(&draft.privacy)?).bind(serde_json::to_string(&draft.model_config)?)
        .bind(&now).bind(&now).bind(TITLE_SOURCE_DEFAULT).bind(&draft.context)
        .execute(&self.pool).await?;
        self.refresh_search(draft.id).await?;
        self.session(draft.id).await
    }

    pub async fn complete_session(&self, id: Uuid, warnings: &[String]) -> Result<(), StoreError> {
        let now = timestamp(Utc::now());
        let result = sqlx::query(
            "UPDATE sessions SET status=?, ended_at=?, warnings_json=?, updated_at=? WHERE id=?",
        )
        .bind(if warnings.is_empty() {
            "completed"
        } else {
            "completed_with_warnings"
        })
        .bind(&now)
        .bind(serde_json::to_string(warnings)?)
        .bind(&now)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::SessionNotFound(id.to_string()));
        }
        Ok(())
    }

    pub async fn recover_active_sessions(&self) -> Result<u64, StoreError> {
        let now = timestamp(Utc::now());
        let warning = serde_json::to_string(&["Application stopped before the session completed"])?;
        let result = sqlx::query(
            "UPDATE sessions SET status='interrupted',ended_at=?,warnings_json=?,updated_at=? WHERE status='active'",
        )
        .bind(&now)
        .bind(warning)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    pub async fn upsert_segment(
        &self,
        segment: &SegmentDraft,
    ) -> Result<SegmentRecord, StoreError> {
        let now = timestamp(Utc::now());
        sqlx::query(
            "INSERT INTO segments (id,session_id,ordinal,start_ms,end_ms,source_text,source_final,asr_confidence,source_revision,timestamp_quality,word_timings_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,source_revision) DO UPDATE SET ordinal=excluded.ordinal,start_ms=excluded.start_ms,end_ms=excluded.end_ms,source_text=excluded.source_text,source_final=excluded.source_final,asr_confidence=excluded.asr_confidence,timestamp_quality=excluded.timestamp_quality,word_timings_json=excluded.word_timings_json,updated_at=excluded.updated_at"
        ).bind(segment.id.to_string()).bind(segment.session_id.to_string()).bind(segment.ordinal)
            .bind(segment.start_ms).bind(segment.end_ms).bind(&segment.source_text)
            .bind(segment.source_final).bind(segment.asr_confidence).bind(segment.source_revision)
            .bind(&segment.timestamp_quality).bind(serde_json::to_string(&segment.word_timings)?)
            .bind(&now).bind(&now).execute(&self.pool).await?;
        self.refresh_search(segment.session_id).await?;
        self.segment_by_revision(segment.session_id, segment.source_revision)
            .await
    }

    pub async fn update_translation(
        &self,
        session_id: Uuid,
        source_revision: i64,
        target_revision: i64,
        text: &str,
        final_value: bool,
    ) -> Result<(), StoreError> {
        let result = sqlx::query(
            "UPDATE segments SET translated_text=?,target_final=?,target_revision=?,updated_at=? WHERE session_id=? AND source_revision=?"
        ).bind(text).bind(final_value).bind(target_revision).bind(timestamp(Utc::now()))
            .bind(session_id.to_string()).bind(source_revision).execute(&self.pool).await?;
        if result.rows_affected() == 0 {
            return Ok(());
        }
        self.refresh_search(session_id).await
    }

    pub async fn update_alignment(
        &self,
        session_id: Uuid,
        source_revision: i64,
        start_ms: f64,
        end_ms: f64,
        word_timings: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let result = sqlx::query(
            "UPDATE segments SET start_ms=?,end_ms=?,timestamp_quality='forced',word_timings_json=?,updated_at=? WHERE session_id=? AND source_revision=?",
        )
        .bind(start_ms.max(0.0))
        .bind(end_ms.max(start_ms))
        .bind(serde_json::to_string(word_timings)?)
        .bind(timestamp(Utc::now()))
        .bind(session_id.to_string())
        .bind(source_revision)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn append_revision(
        &self,
        session_id: Uuid,
        segment_id: Option<Uuid>,
        stream: &str,
        revision_id: i64,
        kind: &str,
        text: &str,
        backend: &str,
        latency: &serde_json::Value,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO revisions(session_id,segment_id,stream,revision_id,kind,text,backend,latency_json,emitted_at) VALUES(?,?,?,?,?,?,?,?,?)",
        )
        .bind(session_id.to_string())
        .bind(segment_id.map(|id| id.to_string()))
        .bind(stream)
        .bind(revision_id)
        .bind(kind)
        .bind(text)
        .bind(backend)
        .bind(serde_json::to_string(latency)?)
        .bind(timestamp(Utc::now()))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn record_metrics(
        &self,
        session_id: Uuid,
        captured_audio_ms: f64,
        metrics: &serde_json::Value,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO metric_samples(session_id,captured_audio_ms,metrics_json,created_at) VALUES(?,?,?,?)",
        )
        .bind(session_id.to_string())
        .bind(captured_audio_ms)
        .bind(serde_json::to_string(metrics)?)
        .bind(timestamp(Utc::now()))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn session(&self, id: Uuid) -> Result<SessionRecord, StoreError> {
        sqlx::query_as::<_, SessionRecord>("SELECT * FROM sessions WHERE id=?")
            .bind(id.to_string())
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| StoreError::SessionNotFound(id.to_string()))
    }

    pub async fn detail(&self, id: Uuid) -> Result<SessionDetail, StoreError> {
        let session = self.session(id).await?;
        let segments = sqlx::query_as::<_, SegmentRecord>(
            "SELECT * FROM segments WHERE session_id=? ORDER BY ordinal,start_ms",
        )
        .bind(id.to_string())
        .fetch_all(&self.pool)
        .await?;
        Ok(SessionDetail { session, segments })
    }

    pub async fn has_segments(&self, session_id: Uuid) -> Result<bool, StoreError> {
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM segments WHERE session_id=?")
            .bind(session_id.to_string())
            .fetch_one(&self.pool)
            .await?;
        Ok(count > 0)
    }

    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<SessionRecord>, StoreError> {
        if query.trim().is_empty() {
            return Ok(sqlx::query_as::<_, SessionRecord>(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            )
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?);
        }
        let escaped = format!("\"{}\"", query.trim().replace('"', "\"\""));
        Ok(sqlx::query_as::<_, SessionRecord>(
            "SELECT s.* FROM sessions s JOIN session_search f ON f.session_id=s.id WHERE session_search MATCH ? ORDER BY s.started_at DESC LIMIT ?"
        ).bind(escaped).bind(limit as i64).fetch_all(&self.pool).await?)
    }

    /// A user rename. The title becomes `user`-owned, so AI titles never
    /// replace it afterwards.
    pub async fn rename(&self, id: Uuid, title: &str) -> Result<(), StoreError> {
        let result =
            sqlx::query("UPDATE sessions SET title=?,title_source=?,updated_at=? WHERE id=?")
                .bind(title.trim())
                .bind(TITLE_SOURCE_USER)
                .bind(timestamp(Utc::now()))
                .bind(id.to_string())
                .execute(&self.pool)
                .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::SessionNotFound(id.to_string()));
        }
        self.refresh_search(id).await
    }

    /// Apply an assistant title unless the user named the session. Returns
    /// whether the title changed hands (`false` for a `user` title or a blank
    /// candidate); an unknown session is an error.
    pub async fn set_ai_title(&self, id: Uuid, title: &str) -> Result<bool, StoreError> {
        let title = title.trim();
        if title.is_empty() {
            self.session(id).await?;
            return Ok(false);
        }
        let result = sqlx::query(
            "UPDATE sessions SET title=?,title_source=?,updated_at=? WHERE id=? AND title_source<>?",
        )
        .bind(title)
        .bind(TITLE_SOURCE_AI)
        .bind(timestamp(Utc::now()))
        .bind(id.to_string())
        .bind(TITLE_SOURCE_USER)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            // Distinguish "user owns the title" from "no such session".
            self.session(id).await?;
            return Ok(false);
        }
        self.refresh_search(id).await?;
        Ok(true)
    }

    /// Insert or replace the AI notes of a session.
    pub async fn save_notes(&self, draft: &SessionNotesDraft) -> Result<SessionNotes, StoreError> {
        let now = timestamp(Utc::now());
        let attachments = if draft.attachments.is_null() {
            serde_json::json!([])
        } else {
            draft.attachments.clone()
        };
        let usage = if draft.usage.is_null() {
            serde_json::json!({})
        } else {
            draft.usage.clone()
        };
        let result = sqlx::query(
            "INSERT INTO session_notes(session_id,markdown,language,provider,model,attachments_json,usage_json,prompt_version,source_chars,created_at,updated_at) SELECT id,?,?,?,?,?,?,?,?,?,? FROM sessions WHERE id=? ON CONFLICT(session_id) DO UPDATE SET markdown=excluded.markdown,language=excluded.language,provider=excluded.provider,model=excluded.model,attachments_json=excluded.attachments_json,usage_json=excluded.usage_json,prompt_version=excluded.prompt_version,source_chars=excluded.source_chars,updated_at=excluded.updated_at",
        )
        .bind(&draft.markdown)
        .bind(&draft.language)
        .bind(&draft.provider)
        .bind(&draft.model)
        .bind(serde_json::to_string(&attachments)?)
        .bind(serde_json::to_string(&usage)?)
        .bind(&draft.prompt_version)
        .bind(draft.source_chars.max(0))
        .bind(&now)
        .bind(&now)
        .bind(draft.session_id.to_string())
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::SessionNotFound(draft.session_id.to_string()));
        }
        self.notes(draft.session_id)
            .await?
            .ok_or_else(|| StoreError::SessionNotFound(draft.session_id.to_string()))
    }

    /// The saved AI notes of a session, if any.
    pub async fn notes(&self, session_id: Uuid) -> Result<Option<SessionNotes>, StoreError> {
        Ok(sqlx::query_as::<_, SessionNotesRow>(
            "SELECT session_id,markdown,language,provider,model,attachments_json,usage_json,prompt_version,source_chars,created_at,updated_at FROM session_notes WHERE session_id=?",
        )
        .bind(session_id.to_string())
        .fetch_optional(&self.pool)
        .await?
        .map(SessionNotes::from))
    }

    pub async fn delete(&self, id: Uuid) -> Result<(), StoreError> {
        let mut transaction = self.pool.begin().await?;
        sqlx::query("DELETE FROM session_search WHERE session_id=?")
            .bind(id.to_string())
            .execute(&mut *transaction)
            .await?;
        let result = sqlx::query("DELETE FROM sessions WHERE id=?")
            .bind(id.to_string())
            .execute(&mut *transaction)
            .await?;
        transaction.commit().await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::SessionNotFound(id.to_string()));
        }
        Ok(())
    }

    pub async fn export(&self, id: Uuid, format: ExportFormat) -> Result<String, StoreError> {
        let detail = self.detail(id).await?;
        Ok(match format {
            ExportFormat::Markdown => export_markdown(&detail, self.notes(id).await?.as_ref()),
            ExportFormat::Txt => export_txt(&detail),
            ExportFormat::Json => serde_json::to_string_pretty(&detail)?,
            ExportFormat::Srt => export_subtitles(&detail.segments, false),
            ExportFormat::Vtt => export_subtitles(&detail.segments, true),
            ExportFormat::Notes => {
                let notes = self
                    .notes(id)
                    .await?
                    .ok_or_else(|| StoreError::NotesNotFound(id.to_string()))?;
                let mut markdown = notes.markdown.trim_end().to_string();
                markdown.push('\n');
                markdown
            }
        })
    }

    async fn segment_by_revision(
        &self,
        session_id: Uuid,
        revision: i64,
    ) -> Result<SegmentRecord, StoreError> {
        sqlx::query_as::<_, SegmentRecord>(
            "SELECT * FROM segments WHERE session_id=? AND source_revision=?",
        )
        .bind(session_id.to_string())
        .bind(revision)
        .fetch_one(&self.pool)
        .await
        .map_err(Into::into)
    }

    async fn refresh_search(&self, session_id: Uuid) -> Result<(), StoreError> {
        let id = session_id.to_string();
        sqlx::query("DELETE FROM session_search WHERE session_id=?")
            .bind(&id)
            .execute(&self.pool)
            .await?;
        sqlx::query(
            "INSERT INTO session_search(session_id,title,source_text,translated_text) SELECT s.id,s.title,COALESCE(group_concat(g.source_text,' '),''),COALESCE(group_concat(g.translated_text,' '),'') FROM sessions s LEFT JOIN segments g ON g.session_id=s.id WHERE s.id=? GROUP BY s.id"
        ).bind(&id).execute(&self.pool).await?;
        Ok(())
    }
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn clock(milliseconds: f64, decimal: char) -> String {
    let total = milliseconds.max(0.0).round() as u64;
    let hours = total / 3_600_000;
    let minutes = total / 60_000 % 60;
    let seconds = total / 1_000 % 60;
    let millis = total % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{decimal}{millis:03}")
}

fn export_markdown(detail: &SessionDetail, notes: Option<&SessionNotes>) -> String {
    let mut output = format!("# {}\n\n", detail.session.title);
    if let Some(notes) = notes.filter(|notes| !notes.markdown.trim().is_empty()) {
        output.push_str("## AI notes\n\n");
        output.push_str(demote_headings(notes.markdown.trim(), 2).trim_end());
        output.push_str("\n\n");
    }
    for segment in &detail.segments {
        output.push_str(&format!(
            "## {}\n\n**Original**\n\n{}\n\n**Translation**\n\n{}\n\n",
            clock(segment.start_ms, ':'),
            segment.source_text,
            segment.translated_text
        ));
    }
    output
}

/// Push every ATX heading outside fenced code blocks down by `levels` (capped
/// at `######`) so embedded notes nest under the export's own headings.
fn demote_headings(markdown: &str, levels: usize) -> String {
    let mut fence: Option<&str> = None;
    let mut output = String::with_capacity(markdown.len() + 64);
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        let marker = if trimmed.starts_with("```") {
            Some("```")
        } else if trimmed.starts_with("~~~") {
            Some("~~~")
        } else {
            None
        };
        if let Some(marker) = marker {
            match fence {
                Some(open) if open == marker => fence = None,
                None => fence = Some(marker),
                _ => {}
            }
            output.push_str(line);
        } else if fence.is_none() && line.starts_with('#') {
            let hashes = line.chars().take_while(|c| *c == '#').count();
            let rest = &line[hashes..];
            if hashes <= 6 && (rest.is_empty() || rest.starts_with([' ', '\t'])) {
                output.push_str(&"#".repeat((hashes + levels).min(6)));
                output.push_str(rest);
            } else {
                output.push_str(line);
            }
        } else {
            output.push_str(line);
        }
        output.push('\n');
    }
    output
}

fn export_txt(detail: &SessionDetail) -> String {
    detail
        .segments
        .iter()
        .map(|segment| {
            format!(
                "[{}] {}\n{}",
                clock(segment.start_ms, ':'),
                segment.source_text,
                segment.translated_text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn export_subtitles(segments: &[SegmentRecord], webvtt: bool) -> String {
    let mut output = if webvtt {
        "WEBVTT\n\n".to_string()
    } else {
        String::new()
    };
    for (index, segment) in segments.iter().enumerate() {
        let start = segment.start_ms.max(0.0);
        let mut end = segment.end_ms.max(start + 500.0);
        if let Some(next) = segments.get(index + 1) {
            end = end.min(next.start_ms.max(start + 500.0));
        }
        if !webvtt {
            output.push_str(&format!("{}\n", index + 1));
        }
        let decimal = if webvtt { '.' } else { ',' };
        output.push_str(&format!(
            "{} --> {}\n{}\n{}\n\n",
            clock(start, decimal),
            clock(end, decimal),
            segment.source_text,
            segment.translated_text
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_draft(id: Uuid) -> SessionDraft {
        SessionDraft {
            id,
            title: "Robotics lecture".into(),
            source_language: "en".into(),
            target_language: "zh".into(),
            audio_source: "microphone".into(),
            audio_profile: "lecture".into(),
            inference_mode: "auto".into(),
            asr_backend: "mock".into(),
            translation_backend: "mock".into(),
            route_reason: "test".into(),
            privacy: serde_json::json!({}),
            model_config: serde_json::json!({}),
            context: String::new(),
        }
    }

    fn segment(session_id: Uuid, revision: i64, text: &str) -> SegmentDraft {
        SegmentDraft {
            id: Uuid::new_v4(),
            session_id,
            ordinal: revision,
            start_ms: revision as f64 * 1_000.0,
            end_ms: revision as f64 * 1_000.0 + 900.0,
            source_text: text.into(),
            source_final: false,
            asr_confidence: None,
            source_revision: revision,
            timestamp_quality: "interpolated".into(),
            word_timings: serde_json::json!([]),
        }
    }

    fn notes_draft(session_id: Uuid, markdown: &str) -> SessionNotesDraft {
        SessionNotesDraft {
            session_id,
            markdown: markdown.into(),
            language: "zh".into(),
            provider: "dashscope".into(),
            model: "qwen-plus".into(),
            attachments: serde_json::json!([{"id": "a1", "name": "slides.pdf", "pages": 12}]),
            usage: serde_json::json!({"prompt_tokens": 1200, "completion_tokens": 800}),
            prompt_version: "notes.v1".into(),
            source_chars: 5_400,
        }
    }

    /// Insert a session the way a build that only knew migration 0001 did.
    async fn insert_legacy_session(pool: &SqlitePool, id: Uuid, title: &str) {
        let now = timestamp(Utc::now());
        sqlx::query(
            "INSERT INTO sessions (id,title,status,started_at,source_language,target_language,audio_source,audio_profile,inference_mode,asr_backend,translation_backend,route_reason,privacy_json,model_config_json,created_at,updated_at) VALUES (?,?,'completed',?,'en','zh','microphone','lecture','auto','mock','mock','','{}','{}',?,?)",
        )
        .bind(id.to_string())
        .bind(title)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn migration_0002_backfills_title_sources_on_a_0001_database() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history.sqlite");
        let default_id = Uuid::new_v4();
        let renamed_id = Uuid::new_v4();
        let other_pair_id = Uuid::new_v4();
        {
            let options = SqliteConnectOptions::from_str(&format!(
                "sqlite://{}",
                path.to_string_lossy()
            ))
            .unwrap()
            .create_if_missing(true)
            .foreign_keys(true);
            let pool = SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(options)
                .await
                .unwrap();
            // Only the first migration, recorded exactly as an old build would.
            sqlx::migrate!().run_to(1, &pool).await.unwrap();
            let columns: Vec<String> =
                sqlx::query_scalar("SELECT name FROM pragma_table_info('sessions')")
                    .fetch_all(&pool)
                    .await
                    .unwrap();
            assert!(!columns.iter().any(|name| name == "title_source"));
            insert_legacy_session(&pool, default_id, "en → zh lecture").await;
            insert_legacy_session(&pool, renamed_id, "Visual perception, week 3").await;
            // A default title for another pair is still recognised per row.
            insert_legacy_session(&pool, other_pair_id, "en → ja lecture").await;
            pool.close().await;
        }

        let store = TranscriptStore::open(&path).await.unwrap();
        let default = store.session(default_id).await.unwrap();
        assert_eq!(default.title_source, TITLE_SOURCE_DEFAULT);
        assert_eq!(default.context, "");
        let renamed = store.session(renamed_id).await.unwrap();
        assert_eq!(renamed.title_source, TITLE_SOURCE_USER);
        // `en → ja lecture` differs from its own `en → zh` default: the user typed it.
        assert_eq!(
            store.session(other_pair_id).await.unwrap().title_source,
            TITLE_SOURCE_USER
        );
        assert!(store.notes(default_id).await.unwrap().is_none());

        // AI titles apply to default titles only, and search finds them.
        assert!(store
            .set_ai_title(default_id, "  Texture perception and pre-attentive vision ")
            .await
            .unwrap());
        let titled = store.session(default_id).await.unwrap();
        assert_eq!(titled.title, "Texture perception and pre-attentive vision");
        assert_eq!(titled.title_source, TITLE_SOURCE_AI);
        assert!(!store.set_ai_title(renamed_id, "AI rewrite").await.unwrap());
        assert_eq!(
            store.session(renamed_id).await.unwrap().title,
            "Visual perception, week 3"
        );
        let found = store.search("pre-attentive", 10).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, default_id.to_string());
        assert_eq!(found[0].title_source, TITLE_SOURCE_AI);

        // Reopening an already-migrated database is a no-op.
        drop(store);
        let reopened = TranscriptStore::open(&path).await.unwrap();
        assert_eq!(
            reopened.session(default_id).await.unwrap().title_source,
            TITLE_SOURCE_AI
        );
    }

    #[tokio::test]
    async fn ai_titles_never_replace_user_titles() {
        let directory = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(directory.path().join("history.sqlite"))
            .await
            .unwrap();
        let id = Uuid::new_v4();
        let mut draft = session_draft(id);
        draft.title = "en → zh lecture".into();
        draft.context = "Topic: early vision\nJulesz = 尤列斯".into();
        let created = store.create_session(&draft).await.unwrap();
        assert_eq!(created.title_source, TITLE_SOURCE_DEFAULT);
        assert_eq!(created.context, "Topic: early vision\nJulesz = 尤列斯");

        assert!(store.set_ai_title(id, "Early vision").await.unwrap());
        // A later AI title may replace an earlier AI title.
        assert!(store.set_ai_title(id, "Early vision and textons").await.unwrap());
        // Blank candidates change nothing.
        assert!(!store.set_ai_title(id, "   ").await.unwrap());
        assert_eq!(store.session(id).await.unwrap().title, "Early vision and textons");

        store.rename(id, "My own title").await.unwrap();
        let renamed = store.session(id).await.unwrap();
        assert_eq!(renamed.title_source, TITLE_SOURCE_USER);
        assert!(!store.set_ai_title(id, "Should not apply").await.unwrap());
        assert_eq!(store.session(id).await.unwrap().title, "My own title");
        assert!(store.search("apply", 10).await.unwrap().is_empty());

        assert!(matches!(
            store.set_ai_title(Uuid::new_v4(), "Nobody").await,
            Err(StoreError::SessionNotFound(_))
        ));
    }

    #[tokio::test]
    async fn notes_upsert_export_and_cascade_with_the_session() {
        let directory = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(directory.path().join("history.sqlite"))
            .await
            .unwrap();
        let id = Uuid::new_v4();
        store.create_session(&session_draft(id)).await.unwrap();
        store
            .upsert_segment(&segment(id, 1, "Textures just pop out."))
            .await
            .unwrap();
        let before = store.export(id, ExportFormat::Markdown).await.unwrap();
        assert!(!before.contains("## AI notes"));
        assert!(matches!(
            store.export(id, ExportFormat::Notes).await,
            Err(StoreError::NotesNotFound(_))
        ));

        let first = store
            .save_notes(&notes_draft(id, "# 纹理感知\n\n## 前注意视觉 (00:00–05:00)\n\n- 纹理基元"))
            .await
            .unwrap();
        assert_eq!(first.session_id, id.to_string());
        assert_eq!(first.attachments[0]["name"], "slides.pdf");
        assert_eq!(first.usage["completion_tokens"], 800);
        assert_eq!(first.source_chars, 5_400);

        let mut revised = notes_draft(id, "# 纹理感知（修订）\n\n```python\n# not a heading\n```\n");
        revised.attachments = serde_json::Value::Null;
        revised.usage = serde_json::Value::Null;
        let second = store.save_notes(&revised).await.unwrap();
        assert_eq!(second.created_at, first.created_at);
        assert_eq!(second.markdown, revised.markdown);
        assert_eq!(second.attachments, serde_json::json!([]));
        assert_eq!(second.usage, serde_json::json!({}));
        assert_eq!(store.notes(id).await.unwrap(), Some(second.clone()));

        let markdown = store.export(id, ExportFormat::Markdown).await.unwrap();
        assert!(markdown.starts_with("# Robotics lecture\n\n## AI notes\n\n### 纹理感知（修订）"));
        assert!(markdown.contains("```python\n# not a heading\n```"));
        assert!(markdown.contains("**Original**\n\nTextures just pop out."));
        let txt = store.export(id, ExportFormat::Txt).await.unwrap();
        assert!(!txt.contains("纹理感知") && txt.contains("Textures just pop out."));
        let notes_only = store.export(id, ExportFormat::Notes).await.unwrap();
        assert_eq!(notes_only, format!("{}\n", revised.markdown.trim_end()));
        let srt = store.export(id, ExportFormat::Srt).await.unwrap();
        assert!(!srt.contains("纹理感知"));

        assert!(matches!(
            store.save_notes(&notes_draft(Uuid::new_v4(), "orphan")).await,
            Err(StoreError::SessionNotFound(_))
        ));

        store.delete(id).await.unwrap();
        assert!(store.notes(id).await.unwrap().is_none());
        let orphans: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM session_notes")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(orphans, 0);
    }

    #[test]
    fn heading_demotion_skips_code_fences_and_caps_depth() {
        let demoted = demote_headings("# A\n##### Deep\n#hashtag\n~~~\n# code\n~~~\n## B", 2);
        assert_eq!(demoted, "### A\n###### Deep\n#hashtag\n~~~\n# code\n~~~\n#### B\n");
    }

    #[tokio::test]
    async fn session_history_search_and_exports_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(directory.path().join("history.sqlite"))
            .await
            .unwrap();
        let session_id = Uuid::new_v4();
        store
            .create_session(&session_draft(session_id))
            .await
            .unwrap();
        store
            .upsert_segment(&SegmentDraft {
                id: Uuid::new_v4(),
                session_id,
                ordinal: 1,
                start_ms: 1_000.0,
                end_ms: 3_250.0,
                source_text: "A robotics framework.".into(),
                source_final: true,
                asr_confidence: Some(0.9),
                source_revision: 1,
                timestamp_quality: "interpolated".into(),
                word_timings: serde_json::json!([]),
            })
            .await
            .unwrap();
        store
            .update_translation(session_id, 1, 2, "一个机器人框架。", true)
            .await
            .unwrap();
        assert_eq!(store.search("robotics", 20).await.unwrap().len(), 1);
        store.rename(session_id, "Updated lecture").await.unwrap();
        let markdown = store
            .export(session_id, ExportFormat::Markdown)
            .await
            .unwrap();
        assert!(markdown.contains("Updated lecture") && markdown.contains("一个机器人框架"));
        let srt = store.export(session_id, ExportFormat::Srt).await.unwrap();
        assert!(srt.contains("00:00:01,000 --> 00:00:03,250"));
        let vtt = store.export(session_id, ExportFormat::Vtt).await.unwrap();
        assert!(vtt.starts_with("WEBVTT") && vtt.contains("00:00:01.000"));
        assert!(store
            .update_alignment(
                session_id,
                1,
                1_120.0,
                3_100.0,
                &serde_json::json!([
                    {"text": "robotics", "start_ms": 1120.0, "end_ms": 1600.0}
                ]),
            )
            .await
            .unwrap());
        let aligned = store.detail(session_id).await.unwrap().segments.remove(0);
        assert_eq!(aligned.timestamp_quality, "forced");
        assert_eq!(aligned.start_ms, 1_120.0);
        assert!(aligned.word_timings_json.contains("robotics"));
        store.complete_session(session_id, &[]).await.unwrap();
        assert_eq!(store.session(session_id).await.unwrap().status, "completed");
        store.delete(session_id).await.unwrap();
        assert!(matches!(
            store.session(session_id).await,
            Err(StoreError::SessionNotFound(_))
        ));
    }

    #[tokio::test]
    async fn active_session_is_preserved_and_marked_interrupted_on_restart() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history.sqlite");
        let session_id = Uuid::new_v4();
        {
            let store = TranscriptStore::open(&path).await.unwrap();
            store
                .create_session(&session_draft(session_id))
                .await
                .unwrap();
            store
                .upsert_segment(&SegmentDraft {
                    id: Uuid::new_v4(),
                    session_id,
                    ordinal: 1,
                    start_ms: 0.0,
                    end_ms: 500.0,
                    source_text: "Preserved before crash".into(),
                    source_final: false,
                    asr_confidence: None,
                    source_revision: 1,
                    timestamp_quality: "interpolated".into(),
                    word_timings: serde_json::json!([]),
                })
                .await
                .unwrap();
        }
        let reopened = TranscriptStore::open(&path).await.unwrap();
        assert_eq!(reopened.recover_active_sessions().await.unwrap(), 1);
        let detail = reopened.detail(session_id).await.unwrap();
        assert_eq!(detail.session.status, "interrupted");
        assert!(detail.session.ended_at.is_some());
        assert_eq!(detail.segments[0].source_text, "Preserved before crash");
    }
}