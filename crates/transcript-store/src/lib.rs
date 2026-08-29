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

pub const SCHEMA_VERSION: u32 = 1;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Markdown,
    Txt,
    Json,
    Srt,
    Vtt,
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
            "INSERT INTO sessions (id,title,status,started_at,source_language,target_language,audio_source,audio_profile,inference_mode,asr_backend,translation_backend,route_reason,privacy_json,model_config_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        .bind(draft.id.to_string()).bind(&draft.title).bind("active").bind(&now)
        .bind(&draft.source_language).bind(&draft.target_language).bind(&draft.audio_source)
        .bind(&draft.audio_profile).bind(&draft.inference_mode).bind(&draft.asr_backend)
        .bind(&draft.translation_backend).bind(&draft.route_reason)
        .bind(serde_json::to_string(&draft.privacy)?).bind(serde_json::to_string(&draft.model_config)?)
        .bind(&now).bind(&now).execute(&self.pool).await?;
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

    pub async fn rename(&self, id: Uuid, title: &str) -> Result<(), StoreError> {
        let result = sqlx::query("UPDATE sessions SET title=?,updated_at=? WHERE id=?")
            .bind(title.trim())
            .bind(timestamp(Utc::now()))
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::SessionNotFound(id.to_string()));
        }
        self.refresh_search(id).await
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
            ExportFormat::Markdown => export_markdown(&detail),
            ExportFormat::Txt => export_txt(&detail),
            ExportFormat::Json => serde_json::to_string_pretty(&detail)?,
            ExportFormat::Srt => export_subtitles(&detail.segments, false),
            ExportFormat::Vtt => export_subtitles(&detail.segments, true),
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

fn export_markdown(detail: &SessionDetail) -> String {
    let mut output = format!("# {}\n\n", detail.session.title);
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
        }
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
}
