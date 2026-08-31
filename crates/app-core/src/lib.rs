//! Rust-owned desktop application state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

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
    pub privacy: PrivacyPolicy,
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
            privacy: PrivacyPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteStatus {
    pub asr_provider: String,
    pub asr_model: Option<String>,
    pub asr_locality: String,
    pub asr_health: BackendHealth,
    pub translation_provider: String,
    pub translation_model: Option<String>,
    pub translation_locality: String,
    pub translation_health: BackendHealth,
    pub deployment: String,
    pub reason: String,
}

impl RouteStatus {
    pub fn starting(request: &StartSessionRequest) -> Self {
        Self {
            asr_provider: request.asr_provider.clone(),
            asr_model: None,
            asr_locality: "pending".into(),
            asr_health: BackendHealth::Starting,
            translation_provider: request.translation_provider.clone(),
            translation_model: None,
            translation_locality: "pending".into(),
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LiveTranscript {
    pub original_committed: String,
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
        if let Some(segment) = self
            .snapshot
            .previous_segments
            .iter_mut()
            .find(|segment| segment.ordinal == ordinal)
        {
            segment.translation = translation;
            return Some(segment.clone());
        }
        None
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

fn validate_privacy(request: &StartSessionRequest) -> Result<(), SessionError> {
    if request.asr_provider == "qwen_cloud" && !request.privacy.audio_upload_allowed {
        return Err(SessionError::AudioUploadNotAllowed);
    }
    if request.translation_provider == "qwen_cloud" && !request.privacy.transcript_upload_allowed {
        return Err(SessionError::TranscriptUploadNotAllowed);
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
            asr_health: BackendHealth::Connected,
            translation_provider: "hymt_local".into(),
            translation_model: Some("hymt2-1.8b".into()),
            translation_locality: "local".into(),
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
        });

        let segment = core
            .update_segment_translation(7, "早上好。".into())
            .expect("known source revision");

        assert_eq!(segment.id, id);
        assert_eq!(segment.translation, "早上好。");
        assert_eq!(core.snapshot().previous_segments, vec![segment]);
    }
}
