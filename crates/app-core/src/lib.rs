//! Rust-owned desktop application state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

pub mod providers;
pub use providers::{
    catalog, catalog_digest, catalog_value, CredentialField, CredentialGroup, Locality,
    ProviderCatalog, ProviderKind, ProviderSetting, ProviderSpec,
};

pub const PRODUCT_PHASE: &str = "phase4";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionPhase {
    Idle,
    Starting,
    Listening,
    Paused,
    Stopping,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSourceKind {
    Microphone,
    SystemAudio,
    SystemAudioAndMicrophone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMode {
    Auto,
    Local,
    Cloud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendHealth {
    Starting,
    Connected,
    Reconnecting,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PrivacyPolicy {
    pub audio_upload_allowed: bool,
    pub transcript_upload_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartSessionRequest {
    pub expected_state_revision: u64,
    pub source_language: String,
    pub target_language: String,
    pub audio_source: AudioSourceKind,
    pub audio_device_id: Option<String>,
    pub audio_profile: String,
    pub inference_mode: InferenceMode,
    pub asr_provider: String,
    pub translation_provider: String,
    /// Which cloud ASR provider the Auto/Cloud route may fall over to. Only
    /// consulted when the route leaves the device and consent was given.
    #[serde(default = "default_cloud_preference")]
    pub cloud_asr_preference: String,
    /// Which cloud translation provider the Auto/Cloud route may use.
    #[serde(default = "default_cloud_preference")]
    pub cloud_translation_preference: String,
    /// Per-lecture topic and terms (docs/adr/0006). Sent to recognition and
    /// translation providers only under the session's upload flags.
    #[serde(default)]
    pub session_context: String,
    /// Standing terminology, one `term = translation` or `term` per line.
    #[serde(default)]
    pub glossary: String,
    pub privacy: PrivacyPolicy,
}

/// Upper bounds shared by the shell's validation and the UI counters.
pub const SESSION_CONTEXT_MAX_CHARS: usize = 2000;
pub const GLOSSARY_MAX_CHARS: usize = 4000;

fn default_cloud_preference() -> String {
    "qwen_cloud".into()
}

impl Default for StartSessionRequest {
    fn default() -> Self {
        Self {
            expected_state_revision: 0,
            source_language: "en".into(),
            target_language: "zh".into(),
            audio_source: AudioSourceKind::Microphone,
            audio_device_id: None,
            audio_profile: "lecture".into(),
            inference_mode: InferenceMode::Auto,
            asr_provider: "auto".into(),
            translation_provider: "auto".into(),
            cloud_asr_preference: default_cloud_preference(),
            cloud_translation_preference: default_cloud_preference(),
            session_context: String::new(),
            glossary: String::new(),
            privacy: PrivacyPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteStatus {
    pub asr_provider: String,
    pub asr_model: Option<String>,
    pub asr_locality: String,
    #[serde(default)]
    pub asr_display_name: Option<String>,
    pub asr_health: BackendHealth,
    pub translation_provider: String,
    pub translation_model: Option<String>,
    pub translation_locality: String,
    #[serde(default)]
    pub translation_display_name: Option<String>,
    pub translation_health: BackendHealth,
    pub deployment: String,
    pub reason: String,
}

impl RouteStatus {
    pub fn starting(request: &StartSessionRequest) -> Self {
        let display_name = |kind: ProviderKind, id: &str| {
            catalog()
                .find(kind, id)
                .map(|spec| spec.display_name.clone())
        };
        Self {
            asr_provider: request.asr_provider.clone(),
            asr_model: None,
            asr_locality: "pending".into(),
            asr_display_name: display_name(ProviderKind::Asr, &request.asr_provider),
            asr_health: BackendHealth::Starting,
            translation_provider: request.translation_provider.clone(),
            translation_model: None,
            translation_locality: "pending".into(),
            translation_display_name: display_name(
                ProviderKind::Translation,
                &request.translation_provider,
            ),
            translation_health: BackendHealth::Starting,
            deployment: "pending".into(),
            reason: "Runtime calibration and privacy policy are being evaluated".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LiveMetrics {
    pub input_rms_dbfs: Option<f32>,
    pub enhanced_rms_dbfs: Option<f32>,
    pub vad_probability: Option<f32>,
    pub speech_detected: bool,
    pub frontend_latency_ms: Option<f32>,
    pub asr_first_partial_latency_ms: Option<f32>,
    pub asr_commit_latency_ms: Option<f32>,
    pub translation_latency_ms: Option<f32>,
    pub end_to_end_latency_ms: Option<f32>,
    pub cloud_roundtrip_latency_ms: Option<f32>,
    pub network_jitter_ms: Option<f32>,
    pub reconnect_count: u32,
    pub buffered_audio_ms: f32,
    pub dropped_audio_ms: f32,
    pub translation_queue_depth: u32,
    pub translation_backlog_ms: f32,
    pub translation_first_delta_ms: Option<f32>,
    pub translation_dropped_partials: u32,
    pub translation_cancelled_requests: u32,
    pub translation_errors: u32,
}

/// Three tiers of live text, mirroring the streaming policy: committed rows
/// live in `previous_segments`; `open_text` is recognizer-committed text that
/// has not closed into a row yet; `original_unstable` is the revisable tail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LiveTranscript {
    pub original_committed: String,
    pub open_text: String,
    pub original_unstable: String,
    pub translation_committed: String,
    pub translation_editable: String,
    pub source_revision_id: u64,
    pub translation_revision_id: u64,
    pub translation_source_revision_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentSummary {
    pub id: Uuid,
    pub ordinal: u32,
    pub start_ms: f64,
    pub end_ms: f64,
    pub original: String,
    pub translation: String,
    /// pending | streaming | done | unavailable
    #[serde(default = "default_translation_status")]
    pub translation_status: String,
}

fn default_translation_status() -> String {
    "pending".into()
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscriptProjection {
    pub segment: Option<SegmentSummary>,
    pub live_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub state_revision: u64,
    pub phase: SessionPhase,
    pub session_id: Option<Uuid>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub config: Option<StartSessionRequest>,
    pub route: Option<RouteStatus>,
    pub live: LiveTranscript,
    pub previous_segments: Vec<SegmentSummary>,
    pub metrics: LiveMetrics,
    pub recoverable_error: Option<String>,
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            state_revision: 0,
            phase: SessionPhase::Idle,
            session_id: None,
            started_at: None,
            ended_at: None,
            config: None,
            route: None,
            live: LiveTranscript::default(),
            previous_segments: Vec::new(),
            metrics: LiveMetrics::default(),
            recoverable_error: None,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionError {
    #[error("stale session command: expected revision {expected}, current revision {actual}")]
    StaleRevision { expected: u64, actual: u64 },
    #[error("cannot {action} while session is {phase:?}")]
    InvalidTransition {
        action: &'static str,
        phase: SessionPhase,
    },
    #[error("cloud ASR requires explicit audio upload consent")]
    AudioUploadNotAllowed,
    #[error("cloud translation requires explicit transcript upload consent")]
    TranscriptUploadNotAllowed,
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
}

#[derive(Debug, Default)]
pub struct AppCore {
    snapshot: SessionSnapshot,
}

impl AppCore {
    pub fn snapshot(&self) -> SessionSnapshot {
        self.snapshot.clone()
    }

    pub fn start(&mut self, request: StartSessionRequest) -> Result<SessionSnapshot, SessionError> {
        self.check_revision(request.expected_state_revision)?;
        if !matches!(
            self.snapshot.phase,
            SessionPhase::Idle | SessionPhase::Completed
        ) {
            return Err(self.invalid("start"));
        }
        validate_privacy(&request)?;
        let route = RouteStatus::starting(&request);
        self.snapshot = SessionSnapshot {
            state_revision: self.snapshot.state_revision + 1,
            phase: SessionPhase::Starting,
            session_id: Some(Uuid::new_v4()),
            started_at: Some(Utc::now()),
            ended_at: None,
            config: Some(request),
            route: Some(route),
            live: LiveTranscript::default(),
            previous_segments: Vec::new(),
            metrics: LiveMetrics::default(),
            recoverable_error: None,
        };
        Ok(self.snapshot())
    }

    pub fn mark_listening(&mut self, route: RouteStatus) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("mark listening", &[SessionPhase::Starting])?;
        self.snapshot.phase = SessionPhase::Listening;
        self.snapshot.route = Some(route);
        self.bump();
        Ok(self.snapshot())
    }

    pub fn pause(&mut self, expected: u64) -> Result<SessionSnapshot, SessionError> {
        self.check_revision(expected)?;
        self.require_phase("pause", &[SessionPhase::Listening])?;
        self.snapshot.phase = SessionPhase::Paused;
        self.bump();
        Ok(self.snapshot())
    }

    pub fn resume(&mut self, expected: u64) -> Result<SessionSnapshot, SessionError> {
        self.check_revision(expected)?;
        self.require_phase("resume", &[SessionPhase::Paused])?;
        self.snapshot.phase = SessionPhase::Listening;
        self.snapshot.recoverable_error = None;
        self.bump();
        Ok(self.snapshot())
    }

    pub fn begin_stop(&mut self, expected: u64) -> Result<SessionSnapshot, SessionError> {
        self.check_revision(expected)?;
        self.require_phase(
            "stop",
            &[
                SessionPhase::Starting,
                SessionPhase::Listening,
                SessionPhase::Paused,
            ],
        )?;
        self.snapshot.phase = SessionPhase::Stopping;
        self.bump();
        Ok(self.snapshot())
    }

    pub fn complete(&mut self) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("complete", &[SessionPhase::Stopping])?;
        self.snapshot.phase = SessionPhase::Completed;
        self.snapshot.ended_at = Some(Utc::now());
        self.bump();
        Ok(self.snapshot())
    }

    pub fn fail_start(
        &mut self,
        message: impl Into<String>,
    ) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("fail start", &[SessionPhase::Starting])?;
        let revision = self.snapshot.state_revision + 1;
        self.snapshot = SessionSnapshot::default();
        self.snapshot.state_revision = revision;
        self.snapshot.recoverable_error = Some(message.into());
        Ok(self.snapshot())
    }

    pub fn backend_disconnected(
        &mut self,
        message: impl Into<String>,
    ) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("report backend disconnect", &[SessionPhase::Listening])?;
        if let Some(route) = self.snapshot.route.as_mut() {
            route.asr_health = BackendHealth::Reconnecting;
            route.deployment = "degraded".into();
            route.reason = message.into();
        }
        self.snapshot.recoverable_error =
            Some("Backend reconnecting; local audio is buffered".into());
        self.bump();
        Ok(self.snapshot())
    }

    pub fn pause_for_recovery(
        &mut self,
        message: impl Into<String>,
    ) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("pause for recovery", &[SessionPhase::Listening])?;
        self.snapshot.phase = SessionPhase::Paused;
        self.snapshot.recoverable_error = Some(message.into());
        self.bump();
        Ok(self.snapshot())
    }

    pub fn audio_device_removed(
        &mut self,
        name: impl Into<String>,
    ) -> Result<SessionSnapshot, SessionError> {
        self.require_phase("remove audio device", &[SessionPhase::Listening])?;
        self.snapshot.phase = SessionPhase::Paused;
        self.snapshot.recoverable_error = Some(format!("Audio device removed: {}", name.into()));
        self.bump();
        Ok(self.snapshot())
    }

    pub fn update_metrics(&mut self, metrics: LiveMetrics) {
        self.snapshot.metrics = metrics;
    }
    pub fn update_live_transcript(&mut self, live: LiveTranscript) {
        self.snapshot.live = live;
    }
    pub fn commit_segment(&mut self, segment: SegmentSummary) {
        if let Some(existing) = self
            .snapshot
            .previous_segments
            .iter_mut()
            .find(|existing| existing.ordinal == segment.ordinal)
        {
            *existing = segment;
        } else {
            self.snapshot.previous_segments.push(segment);
        }
        if self.snapshot.previous_segments.len() > 200 {
            self.snapshot.previous_segments.remove(0);
        }
    }

    pub fn update_segment_translation(
        &mut self,
        ordinal: u32,
        translation: String,
    ) -> Option<SegmentSummary> {
        self.update_segment_translation_status(ordinal, translation, "done")
    }

    pub fn update_segment_translation_status(
        &mut self,
        ordinal: u32,
        translation: String,
        status: &str,
    ) -> Option<SegmentSummary> {
        if let Some(segment) = self
            .snapshot
            .previous_segments
            .iter_mut()
            .find(|segment| segment.ordinal == ordinal)
        {
            if !translation.is_empty() || status == "unavailable" {
                segment.translation = translation;
            }
            segment.translation_status = status.to_string();
            return Some(segment.clone());
        }
        None
    }

    /// Project a canonical transcript event onto live text and rows.
    ///
    /// * `partial` updates the open (recognizer-committed) and unstable tiers.
    /// * `stable` closes a sentence unit into a row and clears the live tiers.
    /// * `final` only carries text that never closed into a unit.
    /// * `alignment_update` and `error` never touch live text; they belong to
    ///   persistence and diagnostics respectively.
    pub fn apply_transcript_event(&mut self, payload: &Value) -> TranscriptProjection {
        let kind = payload["kind"].as_str().unwrap_or_default();
        if kind == "alignment_update" || kind == "error" {
            return TranscriptProjection::default();
        }
        let text = payload["text"].as_str().unwrap_or_default().to_string();
        let committed = payload["committed_text"].as_str().unwrap_or_default().to_string();
        let revision = payload["revision_id"].as_u64();
        let mut live = self.snapshot.live.clone();
        let mut segment = None;
        if let Some(value) = payload["first_token_latency_ms"].as_f64() {
            self.snapshot.metrics.asr_first_partial_latency_ms = Some(value as f32);
        }
        match kind {
            "partial" => {
                live.original_unstable = payload["unstable_text"]
                    .as_str()
                    .unwrap_or(&text)
                    .to_string();
                live.open_text = payload["stable_text"].as_str().unwrap_or_default().to_string();
            }
            "stable" => {
                if !committed.is_empty() {
                    live.original_committed = committed;
                }
                live.open_text.clear();
                live.original_unstable.clear();
                live.translation_editable.clear();
                if let Some(value) = payload["commit_latency_ms"].as_f64() {
                    self.snapshot.metrics.asr_commit_latency_ms = Some(value as f32);
                }
                if !text.trim().is_empty() {
                    let ordinal = revision.unwrap_or(0) as u32;
                    let end_ms = payload["end_ms"]
                        .as_f64()
                        .or_else(|| payload["audio_cursor_ms"].as_f64())
                        .unwrap_or(0.0);
                    let start_ms = payload["start_ms"]
                        .as_f64()
                        .unwrap_or((end_ms - 1_000.0).max(0.0));
                    let summary = SegmentSummary {
                        id: payload["event_id"]
                            .as_str()
                            .and_then(|value| value.parse().ok())
                            .unwrap_or_else(Uuid::new_v4),
                        ordinal,
                        start_ms,
                        end_ms: end_ms.max(start_ms),
                        original: text,
                        translation: String::new(),
                        translation_status: "pending".into(),
                    };
                    self.commit_segment(summary.clone());
                    segment = Some(summary);
                }
            }
            _ => {
                // final
                if !committed.is_empty() {
                    live.original_committed = committed.clone();
                }
                live.open_text.clear();
                live.original_unstable.clear();
                live.translation_editable.clear();
                if self.snapshot.previous_segments.is_empty() {
                    let original = if committed.is_empty() { text } else { committed };
                    if !original.trim().is_empty() {
                        let end_ms = payload["audio_cursor_ms"].as_f64().unwrap_or(0.0);
                        let summary = SegmentSummary {
                            id: payload["event_id"]
                                .as_str()
                                .and_then(|value| value.parse().ok())
                                .unwrap_or_else(Uuid::new_v4),
                            ordinal: revision.unwrap_or(0) as u32,
                            start_ms: 0.0,
                            end_ms,
                            original,
                            translation: String::new(),
                            translation_status: "pending".into(),
                        };
                        self.commit_segment(summary.clone());
                        segment = Some(summary);
                    }
                }
            }
        }
        if let Some(revision) = revision {
            live.source_revision_id = revision;
        }
        let live_changed = live != self.snapshot.live;
        self.snapshot.live = live;
        TranscriptProjection {
            segment,
            live_changed,
        }
    }

    /// Project a canonical translation event.
    ///
    /// Events for a committed source span (`source_committed`) only ever update
    /// the row with that ordinal; events for the provisional tail only ever
    /// update the live editable translation, and only while unstable or open
    /// source text is still on screen. Errors never blank a row.
    pub fn apply_translation_event(&mut self, payload: &Value) -> Option<SegmentSummary> {
        let kind = payload["kind"].as_str().unwrap_or("partial");
        let text = payload["text"].as_str().unwrap_or_default().to_string();
        let editable = payload["editable_text"].as_str().unwrap_or_default().to_string();
        let committed = payload["committed_text"].as_str().unwrap_or_default().to_string();
        let source_committed = payload["source_committed"].as_bool().unwrap_or(false);
        let source_revision = payload["source_revision_id"].as_u64();
        let mut live = self.snapshot.live.clone();
        if let Some(revision) = payload["revision_id"].as_u64() {
            live.translation_revision_id = revision;
        }
        if let Some(revision) = source_revision {
            live.translation_source_revision_id = revision;
        }
        let mut segment = None;
        if source_committed {
            if !committed.is_empty() && kind == "final" {
                live.translation_committed = committed;
            }
            if let Some(ordinal) = source_revision {
                let status = match kind {
                    "final" => "done",
                    "error" => "unavailable",
                    _ => "streaming",
                };
                let row_text = if kind == "error" { String::new() } else { text.clone() };
                segment = self.update_segment_translation_status(ordinal as u32, row_text, status);
            }
            if kind == "final" {
                if let Some(total) = payload["total_latency_ms"].as_f64() {
                    self.snapshot.metrics.translation_latency_ms = Some(total as f32);
                    if let Some(commit) = self.snapshot.metrics.asr_commit_latency_ms {
                        self.snapshot.metrics.end_to_end_latency_ms = Some(commit + total as f32);
                    }
                }
            }
        } else if kind != "error" {
            let has_live_source =
                !live.original_unstable.is_empty() || !live.open_text.is_empty();
            live.translation_editable = if has_live_source {
                if editable.is_empty() { text } else { editable }
            } else {
                String::new()
            };
        }
        self.snapshot.live = live;
        segment
    }

    fn check_revision(&self, expected: u64) -> Result<(), SessionError> {
        if expected != self.snapshot.state_revision {
            return Err(SessionError::StaleRevision {
                expected,
                actual: self.snapshot.state_revision,
            });
        }
        Ok(())
    }

    fn require_phase(
        &self,
        action: &'static str,
        phases: &[SessionPhase],
    ) -> Result<(), SessionError> {
        if phases.contains(&self.snapshot.phase) {
            Ok(())
        } else {
            Err(self.invalid(action))
        }
    }
    fn invalid(&self, action: &'static str) -> SessionError {
        SessionError::InvalidTransition {
            action,
            phase: self.snapshot.phase,
        }
    }
    fn bump(&mut self) {
        self.snapshot.state_revision += 1;
    }
}

/// Gate explicit provider selections on the catalog's privacy flags.
///
/// `auto` defers to the sidecar's route planner, which enforces the same
/// flags against the resolved provider; every other id must exist in the
/// catalog for its kind. Cloud preferences only name which cloud provider the
/// planner may pick, so they must be cloud ids but carry no consent of their
/// own: without the privacy flag the planner simply stays local.
fn validate_privacy(request: &StartSessionRequest) -> Result<(), SessionError> {
    let catalog = catalog();
    for (kind, id) in [
        (ProviderKind::Asr, request.asr_provider.as_str()),
        (ProviderKind::Translation, request.translation_provider.as_str()),
    ] {
        if id == "auto" {
            continue;
        }
        let spec = catalog
            .find(kind, id)
            .ok_or_else(|| SessionError::UnknownProvider(format!("{kind}:{id}")))?;
        if spec.audio_upload_required && !request.privacy.audio_upload_allowed {
            return Err(SessionError::AudioUploadNotAllowed);
        }
        if spec.transcript_upload_required && !request.privacy.transcript_upload_allowed {
            return Err(SessionError::TranscriptUploadNotAllowed);
        }
    }
    for (kind, id) in [
        (ProviderKind::Asr, request.cloud_asr_preference.as_str()),
        (
            ProviderKind::Translation,
            request.cloud_translation_preference.as_str(),
        ),
    ] {
        match catalog.find(kind, id) {
            Some(spec) if spec.is_cloud() => {}
            _ => return Err(SessionError::UnknownProvider(format!("{kind}:{id}"))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected_route() -> RouteStatus {
        RouteStatus {
            asr_provider: "qwen_local".into(),
            asr_model: Some("qwen3-asr-0.6b".into()),
            asr_locality: "local".into(),
            asr_display_name: Some("Qwen3-ASR (local)".into()),
            asr_health: BackendHealth::Connected,
            translation_provider: "hymt_local".into(),
            translation_model: Some("hymt2-1.8b".into()),
            translation_locality: "local".into(),
            translation_display_name: Some("Hy-MT2 (local)".into()),
            translation_health: BackendHealth::Connected,
            deployment: "local".into(),
            reason: "Local calibration meets realtime SLA".into(),
        }
    }

    #[test]
    fn lifecycle_is_core_owned_and_revisioned() {
        let mut core = AppCore::default();
        let starting = core.start(StartSessionRequest::default()).unwrap();
        assert_eq!(starting.phase, SessionPhase::Starting);
        let listening = core.mark_listening(connected_route()).unwrap();
        let paused = core.pause(listening.state_revision).unwrap();
        let resumed = core.resume(paused.state_revision).unwrap();
        let stopping = core.begin_stop(resumed.state_revision).unwrap();
        assert_eq!(stopping.phase, SessionPhase::Stopping);
        let completed = core.complete().unwrap();
        assert_eq!(completed.phase, SessionPhase::Completed);
        assert!(completed.ended_at.is_some());
    }

    #[test]
    fn stale_and_illegal_commands_are_rejected() {
        let mut core = AppCore::default();
        assert!(matches!(
            core.pause(0),
            Err(SessionError::InvalidTransition { .. })
        ));
        let starting = core.start(StartSessionRequest::default()).unwrap();
        let current = core.mark_listening(connected_route()).unwrap();
        assert_eq!(
            core.pause(starting.state_revision),
            Err(SessionError::StaleRevision {
                expected: starting.state_revision,
                actual: current.state_revision,
            })
        );
    }

    #[test]
    fn cloud_routes_require_explicit_consent() {
        let mut request = StartSessionRequest::default();
        request.asr_provider = "qwen_cloud".into();
        let mut core = AppCore::default();
        assert_eq!(
            core.start(request),
            Err(SessionError::AudioUploadNotAllowed)
        );
        let mut request = StartSessionRequest::default();
        request.translation_provider = "qwen_cloud".into();
        assert_eq!(
            core.start(request),
            Err(SessionError::TranscriptUploadNotAllowed)
        );
        // Every cloud adapter in the catalog is gated the same way.
        for spec in catalog().specs(ProviderKind::Asr) {
            if spec.is_cloud() {
                let mut request = StartSessionRequest::default();
                request.asr_provider = spec.id.clone();
                assert_eq!(core.start(request), Err(SessionError::AudioUploadNotAllowed));
            }
        }
        for spec in catalog().specs(ProviderKind::Translation) {
            if spec.is_cloud() {
                let mut request = StartSessionRequest::default();
                request.translation_provider = spec.id.clone();
                assert_eq!(
                    core.start(request),
                    Err(SessionError::TranscriptUploadNotAllowed)
                );
            }
        }
        let mut request = StartSessionRequest::default();
        request.asr_provider = "deepgram".into();
        request.translation_provider = "deepl".into();
        request.privacy = PrivacyPolicy {
            audio_upload_allowed: true,
            transcript_upload_allowed: true,
        };
        assert_eq!(core.start(request).unwrap().phase, SessionPhase::Starting);
    }

    #[test]
    fn unknown_providers_and_non_cloud_preferences_are_rejected() {
        let mut core = AppCore::default();
        let mut request = StartSessionRequest::default();
        request.asr_provider = "whisper_cloud".into();
        assert_eq!(
            core.start(request),
            Err(SessionError::UnknownProvider("asr:whisper_cloud".into()))
        );
        let mut request = StartSessionRequest::default();
        request.translation_provider = "qwen_local".into();
        assert_eq!(
            core.start(request),
            Err(SessionError::UnknownProvider("translation:qwen_local".into()))
        );
        let mut request = StartSessionRequest::default();
        request.cloud_asr_preference = "qwen_local".into();
        assert_eq!(
            core.start(request),
            Err(SessionError::UnknownProvider("asr:qwen_local".into()))
        );
        // Auto with a cloud preference needs no consent up front: the planner
        // stays local until the privacy flag is set.
        let mut request = StartSessionRequest::default();
        request.cloud_asr_preference = "deepgram".into();
        request.cloud_translation_preference = "deepl".into();
        let starting = core.start(request).unwrap();
        assert_eq!(starting.phase, SessionPhase::Starting);
        let route = starting.route.unwrap();
        assert_eq!(route.asr_display_name, None);
        assert_eq!(route.asr_locality, "pending");
    }

    #[test]
    fn requests_and_routes_deserialize_without_the_new_fields() {
        let request: StartSessionRequest = serde_json::from_value(serde_json::json!({
            "expected_state_revision": 0,
            "source_language": "en",
            "target_language": "zh",
            "audio_source": "microphone",
            "audio_device_id": null,
            "audio_profile": "lecture",
            "inference_mode": "auto",
            "asr_provider": "qwen_local",
            "translation_provider": "hymt_local",
            "privacy": {"audio_upload_allowed": false, "transcript_upload_allowed": false}
        }))
        .unwrap();
        assert_eq!(request.cloud_asr_preference, "qwen_cloud");
        assert_eq!(request.cloud_translation_preference, "qwen_cloud");
        assert_eq!(request.session_context, "");
        assert_eq!(request.glossary, "");
        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized["cloud_asr_preference"], "qwen_cloud");
        let route: RouteStatus = serde_json::from_value(serde_json::json!({
            "asr_provider": "qwen_local",
            "asr_model": null,
            "asr_locality": "local",
            "asr_health": "connected",
            "translation_provider": "hymt_local",
            "translation_model": null,
            "translation_locality": "local",
            "translation_health": "connected",
            "deployment": "local",
            "reason": "ok"
        }))
        .unwrap();
        assert_eq!(route.asr_display_name, None);
        let starting = RouteStatus::starting(&request);
        assert_eq!(starting.asr_display_name.as_deref(), Some("Qwen3-ASR (local)"));
        assert_eq!(
            starting.translation_display_name.as_deref(),
            Some("Hy-MT2 (local)")
        );
    }

    #[test]
    fn device_removal_pauses_without_losing_session() {
        let mut core = AppCore::default();
        core.start(StartSessionRequest::default()).unwrap();
        let listening = core.mark_listening(connected_route()).unwrap();
        let paused = core.audio_device_removed("USB microphone").unwrap();
        assert_eq!(paused.phase, SessionPhase::Paused);
        assert_eq!(paused.session_id, listening.session_id);
    }

    #[test]
    fn live_events_do_not_invalidate_lifecycle_commands() {
        let mut core = AppCore::default();
        core.start(StartSessionRequest::default()).unwrap();
        let listening = core.mark_listening(connected_route()).unwrap();
        let revision = listening.state_revision;
        core.update_metrics(LiveMetrics {
            input_rms_dbfs: Some(-31.4),
            ..LiveMetrics::default()
        });
        core.update_live_transcript(LiveTranscript {
            original_unstable: "live words".into(),
            ..LiveTranscript::default()
        });
        core.commit_segment(SegmentSummary {
            id: Uuid::new_v4(),
            ordinal: 1,
            start_ms: 0.0,
            end_ms: 500.0,
            original: "stable words".into(),
            translation: String::new(),
            translation_status: "pending".into(),
        });
        assert_eq!(core.snapshot().state_revision, revision);
        assert_eq!(core.pause(revision).unwrap().phase, SessionPhase::Paused);
    }

    #[test]
    fn disconnect_degrades_before_recovery_pause() {
        let mut core = AppCore::default();
        core.start(StartSessionRequest::default()).unwrap();
        core.mark_listening(connected_route()).unwrap();
        let reconnecting = core.backend_disconnected("network timeout").unwrap();
        assert_eq!(reconnecting.phase, SessionPhase::Listening);
        assert_eq!(
            reconnecting.route.unwrap().asr_health,
            BackendHealth::Reconnecting
        );
        assert_eq!(
            core.pause_for_recovery("fallback unavailable")
                .unwrap()
                .phase,
            SessionPhase::Paused
        );
    }

    #[test]
    fn segment_translation_returns_the_updated_bilingual_projection() {
        let mut core = AppCore::default();
        let id = Uuid::new_v4();
        core.commit_segment(SegmentSummary {
            id,
            ordinal: 7,
            start_ms: 1_000.0,
            end_ms: 2_000.0,
            original: "Good morning.".into(),
            translation: String::new(),
            translation_status: "pending".into(),
        });

        let segment = core
            .update_segment_translation(7, "早上好。".into())
            .expect("known source revision");

        assert_eq!(segment.id, id);
        assert_eq!(segment.translation, "早上好。");
        assert_eq!(core.snapshot().previous_segments, vec![segment]);
    }

    fn transcript(kind: &str, revision: u64, text: &str, stable: &str, unstable: &str) -> Value {
        serde_json::json!({
            "kind": kind,
            "revision_id": revision,
            "event_id": Uuid::new_v4().to_string(),
            "text": text,
            "committed_text": if kind == "stable" || kind == "final" { text } else { "" },
            "stable_text": stable,
            "unstable_text": unstable,
            "start_ms": 1000.0,
            "end_ms": 2500.0,
            "audio_cursor_ms": 2600.0,
            "commit_latency_ms": 3200.0,
        })
    }

    fn translation(kind: &str, source_revision: u64, committed: bool, text: &str) -> Value {
        serde_json::json!({
            "kind": kind,
            "revision_id": 3,
            "source_revision_id": source_revision,
            "source_committed": committed,
            "text": text,
            "editable_text": if committed { "" } else { text },
            "committed_text": if committed && kind == "final" { text } else { "" },
            "total_latency_ms": 700.0,
        })
    }

    #[test]
    fn transcript_projection_keeps_three_text_tiers_and_makes_rows_from_units() {
        let mut core = AppCore::default();
        let partial = core.apply_transcript_event(&transcript("partial", 1, "we propose a new", "we propose", "a new"));
        assert!(partial.live_changed && partial.segment.is_none());
        assert_eq!(core.snapshot().live.open_text, "we propose");
        assert_eq!(core.snapshot().live.original_unstable, "a new");

        let stable = core.apply_transcript_event(&transcript("stable", 2, "We propose a new framework.", "", ""));
        let row = stable.segment.expect("unit becomes a row");
        assert_eq!(row.ordinal, 2);
        assert_eq!(row.original, "We propose a new framework.");
        assert_eq!(row.translation_status, "pending");
        assert_eq!((row.start_ms, row.end_ms), (1000.0, 2500.0));
        let live = core.snapshot().live;
        assert!(live.open_text.is_empty() && live.original_unstable.is_empty());
        assert_eq!(live.source_revision_id, 2);
        assert_eq!(core.snapshot().metrics.asr_commit_latency_ms, Some(3200.0));

        // Alignment and error events never touch live text or rows.
        core.apply_transcript_event(&transcript("partial", 3, "tail", "", "tail"));
        let alignment = core.apply_transcript_event(&transcript("alignment_update", 2, "old text", "", ""));
        assert!(!alignment.live_changed && alignment.segment.is_none());
        assert_eq!(core.snapshot().live.original_unstable, "tail");
        assert_eq!(core.snapshot().previous_segments.len(), 1);
    }

    #[test]
    fn translation_projection_pairs_committed_spans_with_rows_only() {
        let mut core = AppCore::default();
        core.apply_transcript_event(&transcript("stable", 2, "We propose a new framework.", "", ""));
        core.apply_transcript_event(&transcript("partial", 3, "it uses", "", "it uses"));

        // Streaming deltas for the committed row update that row, not the live tail.
        let streaming = core.apply_translation_event(&translation("partial", 2, true, "我们提出")).unwrap();
        assert_eq!(streaming.translation_status, "streaming");
        assert_eq!(streaming.translation, "我们提出");
        assert_eq!(core.snapshot().live.translation_editable, "");

        let done = core.apply_translation_event(&translation("final", 2, true, "我们提出了一个新的框架。")).unwrap();
        assert_eq!(done.translation_status, "done");
        assert_eq!(core.snapshot().live.translation_committed, "我们提出了一个新的框架。");
        assert_eq!(core.snapshot().metrics.translation_latency_ms, Some(700.0));

        // Provisional text only ever reaches the live tail.
        assert!(core.apply_translation_event(&translation("partial", 3, false, "它使用")).is_none());
        assert_eq!(core.snapshot().live.translation_editable, "它使用");
        assert_eq!(core.snapshot().previous_segments[0].translation, "我们提出了一个新的框架。");

        // Errors mark the row unavailable without blanking committed text elsewhere.
        core.apply_transcript_event(&transcript("stable", 4, "Second unit.", "", ""));
        let failed = core.apply_translation_event(&translation("error", 4, true, "timeout")).unwrap();
        assert_eq!(failed.translation_status, "unavailable");
        assert_eq!(failed.translation, "");
        assert_eq!(core.snapshot().previous_segments[0].translation, "我们提出了一个新的框架。");

        // A late provisional event with no live source text is dropped.
        assert!(core.apply_translation_event(&translation("partial", 5, false, "迟到")).is_none());
        assert_eq!(core.snapshot().live.translation_editable, "");
        // Unknown ordinals are ignored rather than mis-attached.
        assert!(core.apply_translation_event(&translation("final", 99, true, "孤儿")).is_none());
    }
}
