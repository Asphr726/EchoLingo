use app_core::{
    AppCore, AudioSourceKind as AppAudioSourceKind, BackendHealth, LiveMetrics, RouteStatus,
    SegmentSummary, SessionPhase, SessionSnapshot, StartSessionRequest,
};
use audio_core::{
    audio_permission_status as native_permission_status,
    list_audio_devices as enumerate_audio_devices, preferred_capture_format,
    request_audio_permission as request_native_audio_permission, start_microphone,
    start_system_audio, AudioCaptureSession, AudioDevice, AudioPermissionStatus, AudioSourceEvent,
    PermissionKind, PermissionState,
};
use inference_ipc::{
    AudioFrameHeader, InferenceSupervisor, SidecarCommand, SidecarEvent, SidecarLaunchConfig,
    UiEventEnvelope, UiEventKind, PROTOCOL_VERSION,
};
#[cfg(target_os = "macos")]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use runtime_manager::{ModelManager, ModelProgress, ModelStatus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use transcript_store::{
    ExportFormat, SegmentDraft, SessionDetail, SessionDraft, SessionRecord, TranscriptStore,
};

const UI_EVENT_CHANNEL: &str = "echolingo://ui-event";
const MODEL_PROGRESS_CHANNEL: &str = "echolingo://model-progress";
const KEYCHAIN_SERVICE: &str = "app.echolingo.desktop";
const DASHSCOPE_API_KEY_ACCOUNT: &str = "dashscope-api-key";
const DASHSCOPE_WORKSPACE_ACCOUNT: &str = "dashscope-workspace-id";

#[derive(Default)]
struct CredentialStore;

#[derive(Debug, Clone, Serialize)]
struct CloudCredentialStatus {
    api_key_available: bool,
    workspace_id_available: bool,
    source: String,
}

#[derive(Debug, Deserialize)]
struct CloudCredentialInput {
    api_key: String,
    workspace_id: String,
}

impl CredentialStore {
    fn entry(account: &str) -> Result<keyring::Entry, String> {
        keyring::Entry::new(KEYCHAIN_SERVICE, account).map_err(|error| error.to_string())
    }

    fn get(account: &str) -> Result<Option<String>, String> {
        match Self::entry(account)?.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }

    fn set(account: &str, value: &str) -> Result<(), String> {
        Self::entry(account)?
            .set_password(value)
            .map_err(|error| error.to_string())
    }

    fn delete(account: &str) -> Result<(), String> {
        match Self::entry(account)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn status(&self) -> Result<CloudCredentialStatus, String> {
        let keychain_key = Self::get(DASHSCOPE_API_KEY_ACCOUNT)?.is_some();
        let keychain_workspace = Self::get(DASHSCOPE_WORKSPACE_ACCOUNT)?.is_some();
        let environment_key = std::env::var_os("DASHSCOPE_API_KEY").is_some();
        let environment_workspace = std::env::var_os("DASHSCOPE_WORKSPACE_ID").is_some();
        let source = if keychain_key || keychain_workspace {
            "macos_keychain"
        } else if environment_key || environment_workspace {
            "environment"
        } else {
            "none"
        };
        Ok(CloudCredentialStatus {
            api_key_available: keychain_key || environment_key,
            workspace_id_available: keychain_workspace || environment_workspace,
            source: source.into(),
        })
    }

    fn sidecar_environment(&self) -> Result<HashMap<String, String>, String> {
        let mut values = HashMap::new();
        let api_key = Self::get(DASHSCOPE_API_KEY_ACCOUNT)?
            .or_else(|| std::env::var("DASHSCOPE_API_KEY").ok());
        let workspace = Self::get(DASHSCOPE_WORKSPACE_ACCOUNT)?
            .or_else(|| std::env::var("DASHSCOPE_WORKSPACE_ID").ok());
        if let Some(value) = api_key {
            values.insert("DASHSCOPE_API_KEY".into(), value);
        }
        if let Some(value) = workspace {
            values.insert("DASHSCOPE_WORKSPACE_ID".into(), value);
        }
        Ok(values)
    }
}

fn validate_cloud_credentials(input: &CloudCredentialInput) -> Result<(), String> {
    if input.api_key.trim().len() < 8 {
        return Err("DashScope API key is too short".into());
    }
    if input.workspace_id.trim().len() < 3 {
        return Err("DashScope workspace ID is too short".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaptionDisplayMode {
    Both,
    Original,
    Translation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CaptionPreferences {
    display: CaptionDisplayMode,
    recent_segments: u8,
    font_size_px: u16,
    opacity: f64,
}

impl Default for CaptionPreferences {
    fn default() -> Self {
        Self {
            display: CaptionDisplayMode::Both,
            recent_segments: 2,
            font_size_px: 30,
            opacity: 0.92,
        }
    }
}

struct RuntimeState {
    core: Mutex<AppCore>,
    event_sequence: AtomicU64,
    audio_sequence: AtomicU64,
    audio: tokio::sync::Mutex<Option<AudioCaptureSession>>,
    shutting_down: std::sync::atomic::AtomicBool,
    caption_preferences: Mutex<CaptionPreferences>,
    session_defaults: Mutex<StartSessionRequest>,
    supervisor: Arc<InferenceSupervisor>,
    store: tokio::sync::OnceCell<TranscriptStore>,
    preferences_path: std::sync::OnceLock<PathBuf>,
    credentials: CredentialStore,
    models: std::sync::OnceLock<ModelManager>,
    onboarding_complete: std::sync::atomic::AtomicBool,
}

impl Default for RuntimeState {
    fn default() -> Self {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        Self {
            core: Mutex::new(AppCore::default()),
            event_sequence: AtomicU64::new(0),
            audio_sequence: AtomicU64::new(0),
            audio: tokio::sync::Mutex::new(None),
            shutting_down: std::sync::atomic::AtomicBool::new(false),
            caption_preferences: Mutex::new(CaptionPreferences::default()),
            session_defaults: Mutex::new(StartSessionRequest::default()),
            supervisor: InferenceSupervisor::new(SidecarLaunchConfig::development(project_root)),
            store: tokio::sync::OnceCell::new(),
            preferences_path: std::sync::OnceLock::new(),
            credentials: CredentialStore,
            models: std::sync::OnceLock::new(),
            onboarding_complete: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl RuntimeState {
    fn snapshot(&self) -> Result<SessionSnapshot, String> {
        self.core
            .lock()
            .map_err(|_| "app core lock poisoned".to_string())
            .map(|core| core.snapshot())
    }

    fn emit_snapshot(&self, app: &AppHandle, snapshot: &SessionSnapshot) -> Result<(), String> {
        self.emit_event(
            app,
            snapshot.session_id,
            UiEventKind::SessionState,
            serde_json::to_value(snapshot).map_err(|error| error.to_string())?,
        )
    }

    fn emit_event(
        &self,
        app: &AppHandle,
        session_id: Option<uuid::Uuid>,
        kind: UiEventKind,
        payload: Value,
    ) -> Result<(), String> {
        let event = UiEventEnvelope {
            schema_version: PROTOCOL_VERSION,
            sequence: self.event_sequence.fetch_add(1, Ordering::Relaxed) + 1,
            session_id,
            kind,
            emitted_at_unix_ms: chrono::Utc::now().timestamp_millis(),
            payload,
        };
        app.emit(UI_EVENT_CHANNEL, event)
            .map_err(|error| error.to_string())
    }

    fn store(&self) -> Result<&TranscriptStore, String> {
        self.store
            .get()
            .ok_or_else(|| "transcript store is not initialized".to_string())
    }

    fn preferences(&self) -> Result<DesktopPreferences, String> {
        Ok(DesktopPreferences {
            schema_version: 1,
            onboarding_complete: self.onboarding_complete.load(Ordering::Acquire),
            session: self
                .session_defaults
                .lock()
                .map_err(|_| "session defaults lock poisoned".to_string())?
                .clone(),
            caption: self
                .caption_preferences
                .lock()
                .map_err(|_| "caption preferences lock poisoned".to_string())?
                .clone(),
        })
    }

    fn persist_preferences(&self) -> Result<(), String> {
        let path = self
            .preferences_path
            .get()
            .ok_or_else(|| "preferences path is not initialized".to_string())?;
        save_preferences(path, &self.preferences()?)
    }

    fn models(&self) -> Result<&ModelManager, String> {
        self.models
            .get()
            .ok_or_else(|| "model manager is not initialized".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct DesktopPreferences {
    schema_version: u16,
    onboarding_complete: bool,
    session: StartSessionRequest,
    caption: CaptionPreferences,
}

impl Default for DesktopPreferences {
    fn default() -> Self {
        Self {
            schema_version: 1,
            onboarding_complete: false,
            session: StartSessionRequest::default(),
            caption: CaptionPreferences::default(),
        }
    }
}

fn validate_session_defaults(defaults: &StartSessionRequest) -> Result<(), String> {
    if !matches!(defaults.source_language.as_str(), "en" | "zh" | "ja" | "ko")
        || !matches!(defaults.target_language.as_str(), "en" | "zh" | "ja" | "ko")
    {
        return Err("session languages must be en, zh, ja, or ko".into());
    }
    if defaults.source_language == defaults.target_language {
        return Err("source and target languages must differ".into());
    }
    if !matches!(
        defaults.audio_profile.as_str(),
        "lecture" | "conversation" | "raw"
    ) {
        return Err("unknown audio profile".into());
    }
    if !matches!(
        defaults.asr_provider.as_str(),
        "auto" | "qwen_local" | "qwen_cloud" | "simulstreaming" | "mock"
    ) {
        return Err("unknown ASR provider".into());
    }
    if !matches!(
        defaults.translation_provider.as_str(),
        "auto" | "hymt_local" | "qwen_cloud" | "mock" | "none"
    ) {
        return Err("unknown translation provider".into());
    }
    Ok(())
}

fn validate_caption_preferences(preferences: &CaptionPreferences) -> Result<(), String> {
    if !(18..=72).contains(&preferences.font_size_px) {
        return Err("caption font size must be between 18 and 72 px".into());
    }
    if !(0.35..=1.0).contains(&preferences.opacity) || !preferences.opacity.is_finite() {
        return Err("caption opacity must be between 0.35 and 1.0".into());
    }
    if !(1..=3).contains(&preferences.recent_segments) {
        return Err("caption recent segments must be between 1 and 3".into());
    }
    Ok(())
}

fn load_preferences(path: &std::path::Path) -> DesktopPreferences {
    let preferences = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<DesktopPreferences>(&bytes).ok())
        .unwrap_or_default();
    if preferences.schema_version != 1
        || validate_session_defaults(&preferences.session).is_err()
        || validate_caption_preferences(&preferences.caption).is_err()
    {
        DesktopPreferences::default()
    } else {
        preferences
    }
}

fn save_preferences(
    path: &std::path::Path,
    preferences: &DesktopPreferences,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary_path = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(preferences).map_err(|error| error.to_string())?;
    std::fs::write(&temporary_path, bytes).map_err(|error| error.to_string())?;
    #[cfg(target_os = "windows")]
    if path.exists() {
        std::fs::remove_file(path).map_err(|error| error.to_string())?;
    }
    std::fs::rename(&temporary_path, path).map_err(|error| error.to_string())
}

#[tauri::command]
fn get_app_snapshot(state: State<'_, RuntimeState>) -> Result<SessionSnapshot, String> {
    state.snapshot()
}

#[derive(Debug, Clone, Serialize)]
struct AudioTestResult {
    source: String,
    sample_rate_hz: u32,
    channels: u16,
    peak_rms_dbfs: f32,
    frames_observed: u64,
}

#[tauri::command]
fn onboarding_status(state: State<'_, RuntimeState>) -> Result<bool, String> {
    Ok(state.preferences()?.onboarding_complete)
}

#[tauri::command]
fn complete_onboarding(state: State<'_, RuntimeState>) -> Result<bool, String> {
    let path = state
        .preferences_path
        .get()
        .ok_or_else(|| "preferences path is not initialized".to_string())?;
    let mut preferences = state.preferences()?;
    preferences.onboarding_complete = true;
    save_preferences(path, &preferences)?;
    state.onboarding_complete.store(true, Ordering::Release);
    Ok(true)
}

#[tauri::command]
async fn audio_permission_status() -> Result<AudioPermissionStatus, String> {
    tauri::async_runtime::spawn_blocking(native_permission_status)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn request_audio_permission(kind: PermissionKind) -> Result<PermissionState, String> {
    tauri::async_runtime::spawn_blocking(move || request_native_audio_permission(kind))
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn test_audio_input(
    state: State<'_, RuntimeState>,
    source: AppAudioSourceKind,
    device_id: Option<String>,
) -> Result<AudioTestResult, String> {
    if !matches!(
        state.snapshot()?.phase,
        SessionPhase::Idle | SessionPhase::Completed
    ) {
        return Err("audio test is unavailable during a session".into());
    }
    let mut capture = match source {
        AppAudioSourceKind::Microphone => start_microphone(device_id.as_deref()),
        AppAudioSourceKind::SystemAudio => start_system_audio(),
        AppAudioSourceKind::SystemAudioAndMicrophone => {
            return Err("combined audio testing is not implemented".into())
        }
    }
    .map_err(|error| error.to_string())?;
    let mut events = capture
        .take_events()
        .ok_or_else(|| "audio test event stream is unavailable".to_string())?;
    let mut frames = capture
        .take_frames()
        .ok_or_else(|| "audio test frame stream is unavailable".to_string())?;
    let result = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        while let Some(event) = events.recv().await {
            match event {
                AudioSourceEvent::Started => break,
                AudioSourceEvent::PickerCancelled => {
                    return Err("system audio selection was cancelled".into())
                }
                AudioSourceEvent::Error { message, .. }
                | AudioSourceEvent::DeviceRemoved { message } => return Err(message),
                _ => {}
            }
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
        let mut peak = -120.0_f32;
        let mut observed = 0_u64;
        let mut format = None;
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(std::time::Duration::from_millis(500), frames.recv()).await {
                Ok(Some(frame)) => {
                    peak = peak.max(frame.rms_dbfs());
                    observed += frame.frame_count() as u64;
                    format = Some((frame.sample_rate_hz, frame.channels));
                }
                Ok(None) => break,
                Err(_) if observed > 0 => break,
                Err(_) => continue,
            }
        }
        let (sample_rate_hz, channels) =
            format.ok_or_else(|| "audio source started but produced no samples".to_string())?;
        Ok(AudioTestResult {
            source: value_name(&source),
            sample_rate_hz,
            channels,
            peak_rms_dbfs: peak,
            frames_observed: observed,
        })
    })
    .await
    .map_err(|_| "audio test timed out".to_string())?;
    let _ = capture.stop();
    result
}

#[tauri::command]
fn credential_status(state: State<'_, RuntimeState>) -> Result<CloudCredentialStatus, String> {
    state.credentials.status()
}

#[tauri::command]
async fn set_cloud_credentials(
    state: State<'_, RuntimeState>,
    input: CloudCredentialInput,
) -> Result<CloudCredentialStatus, String> {
    validate_cloud_credentials(&input)?;
    CredentialStore::set(DASHSCOPE_API_KEY_ACCOUNT, input.api_key.trim())?;
    if let Err(error) = CredentialStore::set(DASHSCOPE_WORKSPACE_ACCOUNT, input.workspace_id.trim())
    {
        let _ = CredentialStore::delete(DASHSCOPE_API_KEY_ACCOUNT);
        return Err(error);
    }
    state.supervisor.shutdown().await;
    state.credentials.status()
}

#[tauri::command]
async fn clear_cloud_credentials(
    state: State<'_, RuntimeState>,
) -> Result<CloudCredentialStatus, String> {
    CredentialStore::delete(DASHSCOPE_API_KEY_ACCOUNT)?;
    CredentialStore::delete(DASHSCOPE_WORKSPACE_ACCOUNT)?;
    state.supervisor.shutdown().await;
    state.credentials.status()
}

#[tauri::command]
fn list_models(state: State<'_, RuntimeState>) -> Result<Vec<ModelStatus>, String> {
    Ok(state.models()?.list())
}

#[tauri::command]
async fn install_model(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    model_id: String,
) -> Result<ModelStatus, String> {
    let progress_app = app.clone();
    let callback = Arc::new(move |progress: ModelProgress| {
        let _ = progress_app.emit(MODEL_PROGRESS_CHANNEL, progress);
    });
    state
        .models()?
        .install(&model_id, callback)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn verify_model(
    state: State<'_, RuntimeState>,
    model_id: String,
) -> Result<ModelStatus, String> {
    let root = state.models()?.root().to_path_buf();
    tauri::async_runtime::spawn_blocking(move || ModelManager::new(root).verify(&model_id))
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_model(state: State<'_, RuntimeState>, model_id: String) -> Result<ModelStatus, String> {
    state
        .models()?
        .delete(&model_id)
        .map_err(|error| error.to_string())
}

fn route_status(value: &Value) -> RouteStatus {
    let asr = value["asr_provider"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let translation = value["translation_provider"]
        .as_str()
        .unwrap_or("none")
        .to_string();
    let degraded = value["degraded"].as_bool().unwrap_or(false);
    let locality = |provider: &str| {
        if provider.contains("cloud") {
            "cloud"
        } else {
            "local"
        }
    };
    let reasons = value["reasons"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "Runtime route selected".into());
    RouteStatus {
        asr_model: match asr.as_str() {
            "qwen_local" => Some("qwen3-asr-0.6b".into()),
            "qwen_cloud" => Some("qwen3-asr-flash-realtime".into()),
            _ => None,
        },
        asr_locality: locality(&asr).into(),
        asr_health: if degraded {
            BackendHealth::Degraded
        } else {
            BackendHealth::Connected
        },
        translation_model: match translation.as_str() {
            "hymt_local" => Some("tencent/Hy-MT2-1.8B".into()),
            "qwen_cloud" => Some("qwen-mt-flash".into()),
            _ => None,
        },
        translation_locality: locality(&translation).into(),
        translation_health: if translation == "none" {
            BackendHealth::Unavailable
        } else if degraded {
            BackendHealth::Degraded
        } else {
            BackendHealth::Connected
        },
        deployment: value["status"].as_str().unwrap_or("degraded").to_string(),
        reason: reasons,
        asr_provider: asr,
        translation_provider: translation,
    }
}

fn value_name<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".into())
}

async fn persist_sidecar_event(
    state: &RuntimeState,
    session_id: uuid::Uuid,
    event: &SidecarEvent,
) -> Result<(), String> {
    let store = state.store()?;
    match event {
        SidecarEvent::Transcript(payload) => {
            let revision = payload["revision_id"].as_i64().unwrap_or(0);
            let kind = payload["kind"].as_str().unwrap_or("partial");
            let text = payload["text"].as_str().unwrap_or_default();
            let segment_id = payload["event_id"].as_str().and_then(|id| id.parse().ok());
            store
                .append_revision(
                    session_id,
                    None,
                    "source",
                    revision,
                    kind,
                    text,
                    payload["backend"].as_str().unwrap_or("unknown"),
                    &json!({
                        "first_token_latency_ms": payload["first_token_latency_ms"],
                        "commit_latency_ms": payload["commit_latency_ms"]
                    }),
                )
                .await
                .map_err(|error| error.to_string())?;
            let should_persist_segment = kind == "stable"
                || (kind == "final"
                    && !store
                        .has_segments(session_id)
                        .await
                        .map_err(|error| error.to_string())?);
            if should_persist_segment && !text.is_empty() {
                let end = payload["end_ms"]
                    .as_f64()
                    .unwrap_or_else(|| payload["audio_cursor_ms"].as_f64().unwrap_or(0.0));
                let start = payload["start_ms"]
                    .as_f64()
                    .unwrap_or((end - 1_000.0).max(0.0));
                store
                    .upsert_segment(&SegmentDraft {
                        id: segment_id.unwrap_or_else(uuid::Uuid::new_v4),
                        session_id,
                        ordinal: revision,
                        start_ms: start,
                        end_ms: end.max(start + 500.0),
                        source_text: if kind == "stable" {
                            text.to_string()
                        } else {
                            payload["committed_text"]
                                .as_str()
                                .filter(|value| !value.is_empty())
                                .unwrap_or(text)
                                .to_string()
                        },
                        source_final: kind == "final",
                        asr_confidence: payload["confidence"].as_f64(),
                        source_revision: revision,
                        timestamp_quality: payload["timestamp_quality"]
                            .as_str()
                            .unwrap_or("none")
                            .to_string(),
                        word_timings: payload["words"].clone(),
                    })
                    .await
                    .map_err(|error| error.to_string())?;
            }
        }
        SidecarEvent::Translation(payload) => {
            let source_revision = payload["source_revision_id"].as_i64().unwrap_or(0);
            let revision = payload["revision_id"].as_i64().unwrap_or(0);
            let kind = payload["kind"].as_str().unwrap_or("partial");
            let text = payload["text"].as_str().unwrap_or_default();
            store
                .append_revision(
                    session_id,
                    None,
                    "target",
                    revision,
                    kind,
                    text,
                    payload["provider"].as_str().unwrap_or("unknown"),
                    &json!({
                        "first_delta_latency_ms": payload["first_delta_latency_ms"],
                        "total_latency_ms": payload["total_latency_ms"]
                    }),
                )
                .await
                .map_err(|error| error.to_string())?;
            store
                .update_translation(session_id, source_revision, revision, text, kind == "final")
                .await
                .map_err(|error| error.to_string())?;
        }
        SidecarEvent::Metrics(payload) => {
            store
                .record_metrics(
                    session_id,
                    payload["captured_audio_ms"].as_f64().unwrap_or(0.0),
                    payload,
                )
                .await
                .map_err(|error| error.to_string())?;
        }
        SidecarEvent::AlignmentUpdate(payload) => {
            let source_revision = payload["revision_id"].as_i64().unwrap_or(0);
            let start_ms = payload["start_ms"].as_f64().unwrap_or(0.0);
            let end_ms = payload["end_ms"].as_f64().unwrap_or(start_ms);
            store
                .append_revision(
                    session_id,
                    None,
                    "source",
                    source_revision,
                    "alignment_update",
                    payload["text"].as_str().unwrap_or_default(),
                    payload["model"].as_str().unwrap_or("qwen_forced_aligner"),
                    &json!({
                        "alignment_processing_ms": payload["alignment_processing_ms"]
                    }),
                )
                .await
                .map_err(|error| error.to_string())?;
            store
                .update_alignment(
                    session_id,
                    source_revision,
                    start_ms,
                    end_ms,
                    &payload["words"],
                )
                .await
                .map_err(|error| error.to_string())?;
        }
        _ => {}
    }
    Ok(())
}

fn forward_sidecar_events(app: AppHandle) {
    let mut receiver = app.state::<RuntimeState>().supervisor.subscribe();
    tauri::async_runtime::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            let state = app.state::<RuntimeState>();
            let persist_event = event.clone();
            let (kind, payload) = match event {
                SidecarEvent::Transcript(payload) => {
                    if let Ok(mut core) = state.core.lock() {
                        let mut live = core.snapshot().live;
                        let event_kind = payload["kind"].as_str().unwrap_or_default();
                        let text = payload["text"].as_str().unwrap_or_default().to_string();
                        let committed = payload["committed_text"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string();
                        if event_kind == "partial" {
                            live.original_unstable = payload["unstable_text"]
                                .as_str()
                                .unwrap_or(&text)
                                .to_string();
                        } else {
                            live.original_committed = if committed.is_empty() {
                                text
                            } else {
                                committed
                            };
                            live.original_unstable.clear();
                        }
                        live.source_revision_id = payload["revision_id"]
                            .as_u64()
                            .unwrap_or(live.source_revision_id);
                        core.update_live_transcript(live);
                        let should_commit_segment = event_kind == "stable"
                            || (event_kind == "final"
                                && core.snapshot().previous_segments.is_empty());
                        if should_commit_segment {
                            let revision = payload["revision_id"].as_u64().unwrap_or(0) as u32;
                            let end_ms = payload["end_ms"]
                                .as_f64()
                                .or_else(|| payload["audio_cursor_ms"].as_f64())
                                .unwrap_or(0.0);
                            let start_ms = payload["start_ms"]
                                .as_f64()
                                .unwrap_or((end_ms - 1_000.0).max(0.0));
                            core.commit_segment(SegmentSummary {
                                id: payload["event_id"]
                                    .as_str()
                                    .and_then(|value| value.parse().ok())
                                    .unwrap_or_else(uuid::Uuid::new_v4),
                                ordinal: revision,
                                start_ms,
                                end_ms,
                                original: if event_kind == "stable" {
                                    payload["text"].as_str().unwrap_or_default().to_string()
                                } else {
                                    payload["committed_text"]
                                        .as_str()
                                        .filter(|value| !value.is_empty())
                                        .unwrap_or(payload["text"].as_str().unwrap_or_default())
                                        .to_string()
                                },
                                translation: String::new(),
                            });
                        }
                    }
                    (UiEventKind::TranscriptRevision, payload)
                }
                SidecarEvent::Translation(payload) => {
                    if let Ok(mut core) = state.core.lock() {
                        let mut live = core.snapshot().live;
                        let committed = payload["committed_text"].as_str().unwrap_or_default();
                        let editable = payload["editable_text"].as_str().unwrap_or_default();
                        if !committed.is_empty() {
                            live.translation_committed = committed.into();
                        }
                        live.translation_editable = editable.into();
                        live.translation_revision_id = payload["revision_id"]
                            .as_u64()
                            .unwrap_or(live.translation_revision_id);
                        core.update_live_transcript(live);
                        if let Some(source_revision) = payload["source_revision_id"].as_u64() {
                            let text = payload["committed_text"]
                                .as_str()
                                .filter(|value| !value.is_empty())
                                .or_else(|| payload["editable_text"].as_str())
                                .or_else(|| payload["text"].as_str())
                                .unwrap_or_default()
                                .to_string();
                            core.update_segment_translation(source_revision as u32, text);
                        }
                    }
                    (UiEventKind::TranslationRevision, payload)
                }
                SidecarEvent::Metrics(payload) => {
                    if let Ok(metrics) = serde_json::from_value::<LiveMetrics>(payload.clone()) {
                        if let Ok(mut core) = state.core.lock() {
                            core.update_metrics(metrics);
                        }
                    }
                    (UiEventKind::Metrics, payload)
                }
                SidecarEvent::SegmentCommitted(payload) => (UiEventKind::SegmentCommitted, payload),
                SidecarEvent::AlignmentUpdate(payload) => {
                    (UiEventKind::TranscriptRevision, payload)
                }
                SidecarEvent::BackendHealth(payload) => (UiEventKind::BackendHealth, payload),
                SidecarEvent::Error {
                    code,
                    message,
                    recoverable,
                } => (
                    UiEventKind::Error,
                    json!({"code": code, "message": message, "recoverable": recoverable}),
                ),
                SidecarEvent::Ready { route, .. } => (UiEventKind::RouteDecision, route),
                SidecarEvent::HelloAccepted { .. } | SidecarEvent::SessionFinished { .. } => {
                    continue
                }
            };
            let active_session_id = state
                .snapshot()
                .ok()
                .and_then(|snapshot| snapshot.session_id);
            let event_session_id = payload["session_id"]
                .as_str()
                .and_then(|value| value.parse().ok())
                .or(active_session_id);
            let _ = state.emit_event(&app, event_session_id, kind, payload);
            if let Some(session_id) = event_session_id {
                if let Err(error) = persist_sidecar_event(&state, session_id, &persist_event).await
                {
                    let _ = state.emit_event(
                        &app,
                        Some(session_id),
                        UiEventKind::Error,
                        json!({"code": "storage_error", "message": error, "recoverable": true}),
                    );
                }
            }
        }
    });
}

fn emit_audio_event(
    app: &AppHandle,
    state: &RuntimeState,
    event: &AudioSourceEvent,
) -> Result<(), String> {
    state.emit_event(
        app,
        state.snapshot()?.session_id,
        UiEventKind::AudioDeviceChange,
        serde_json::to_value(event).map_err(|error| error.to_string())?,
    )
}

async fn start_audio_capture(
    app: &AppHandle,
    state: &RuntimeState,
    request: &StartSessionRequest,
) -> Result<(), String> {
    let mut capture = match request.audio_source {
        AppAudioSourceKind::Microphone => start_microphone(request.audio_device_id.as_deref()),
        AppAudioSourceKind::SystemAudio => start_system_audio(),
        AppAudioSourceKind::SystemAudioAndMicrophone => {
            return Err(
                "combined system audio and microphone capture is not implemented yet".into(),
            )
        }
    }
    .map_err(|error| error.to_string())?;
    let mut events = capture
        .take_events()
        .ok_or_else(|| "audio event receiver is unavailable".to_string())?;
    let ready = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        while let Some(event) = events.recv().await {
            emit_audio_event(app, state, &event)?;
            match event {
                AudioSourceEvent::Started => return Ok(()),
                AudioSourceEvent::PickerCancelled => {
                    return Err("system audio selection was cancelled".into())
                }
                AudioSourceEvent::DeviceRemoved { message } => return Err(message),
                AudioSourceEvent::Error { message, .. } => return Err(message),
                _ => {}
            }
        }
        Err("audio source closed before starting".into())
    })
    .await
    .map_err(|_| "audio source start timed out".to_string())?;
    ready?;

    let frames = capture
        .take_frames()
        .ok_or_else(|| "audio frame receiver is unavailable".to_string())?;
    let mut audio = state.audio.lock().await;
    if audio.is_some() {
        return Err("an audio source is already running".into());
    }
    *audio = Some(capture);
    drop(audio);
    pump_audio_frames(app.clone(), frames);
    forward_audio_events(app.clone(), events);
    Ok(())
}

fn pump_audio_frames(
    app: AppHandle,
    mut frames: tokio::sync::mpsc::Receiver<audio_core::AudioFrame>,
) {
    tauri::async_runtime::spawn(async move {
        while let Some(frame) = frames.recv().await {
            let state = app.state::<RuntimeState>();
            let max_frames = usize::from(u16::MAX);
            let channels = usize::from(frame.channels.max(1));
            for (part, samples) in frame.samples.chunks(max_frames * channels).enumerate() {
                let sequence = state.audio_sequence.fetch_add(1, Ordering::Relaxed) + 1;
                let header = AudioFrameHeader {
                    flags: u16::from(frame.overflow && part == 0),
                    sequence,
                    capture_monotonic_ns: frame.capture_monotonic_ns,
                    sample_rate_hz: frame.sample_rate_hz,
                    channels: frame.channels,
                    frame_count: (samples.len() / channels) as u16,
                };
                let mut packet = Vec::with_capacity(32 + samples.len() * size_of::<f32>());
                packet.extend_from_slice(&header.encode());
                for sample in samples {
                    packet.extend_from_slice(&sample.to_le_bytes());
                }
                if let Err(error) = state.supervisor.send_audio(packet).await {
                    let _ = state.emit_event(
                        &app,
                        state
                            .snapshot()
                            .ok()
                            .and_then(|snapshot| snapshot.session_id),
                        UiEventKind::Error,
                        json!({
                            "code": "audio_transport_error",
                            "message": error.to_string(),
                            "recoverable": true
                        }),
                    );
                    break;
                }
            }
        }
    });
}

fn forward_audio_events(app: AppHandle, mut events: tokio::sync::mpsc::Receiver<AudioSourceEvent>) {
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            let state = app.state::<RuntimeState>();
            let _ = emit_audio_event(&app, &state, &event);
            if let AudioSourceEvent::DeviceRemoved { message } = event {
                let recovery = state
                    .core
                    .lock()
                    .ok()
                    .and_then(|mut core| core.pause_for_recovery(message).ok());
                if let Some(snapshot) = recovery {
                    if let Some(session_id) = snapshot.session_id {
                        let _ = state
                            .supervisor
                            .send_command(SidecarCommand::Pause {
                                session_id,
                                epoch: snapshot.state_revision as u32,
                            })
                            .await;
                    }
                    let _ = state.emit_snapshot(&app, &snapshot);
                }
            }
        }
    });
}

async fn stop_audio_capture(state: &RuntimeState) -> Result<(), String> {
    let mut capture = state.audio.lock().await.take();
    if let Some(capture) = capture.as_mut() {
        capture.stop().map_err(|error| error.to_string())?;
    }
    Ok(())
}

async fn shutdown_application(app: &AppHandle) {
    let state = app.state::<RuntimeState>();
    let snapshot = state.snapshot().ok();
    let active = snapshot.as_ref().is_some_and(|snapshot| {
        matches!(
            snapshot.phase,
            SessionPhase::Starting | SessionPhase::Listening | SessionPhase::Paused
        )
    });
    if active {
        if let Some(snapshot) = snapshot {
            if let Ok(stopping) = state
                .core
                .lock()
                .map_err(|_| ())
                .and_then(|mut core| core.begin_stop(snapshot.state_revision).map_err(|_| ()))
            {
                let _ = state.emit_snapshot(app, &stopping);
                let _ = stop_audio_capture(&state).await;
                if let Some(session_id) = stopping.session_id {
                    let mut receiver = state.supervisor.subscribe();
                    if state
                        .supervisor
                        .send_command(SidecarCommand::FinishSession { session_id })
                        .await
                        .is_ok()
                    {
                        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                            while let Ok(event) = receiver.recv().await {
                                if matches!(event, SidecarEvent::SessionFinished { session_id: finished } if finished == session_id) {
                                    break;
                                }
                            }
                        })
                        .await;
                    }
                    if let Ok(completed) = state
                        .core
                        .lock()
                        .map_err(|_| ())
                        .and_then(|mut core| core.complete().map_err(|_| ()))
                    {
                        if let Ok(store) = state.store() {
                            let _ = store.complete_session(session_id, &[]).await;
                        }
                        let _ = state.emit_snapshot(app, &completed);
                    }
                }
            }
        }
    } else {
        let _ = stop_audio_capture(&state).await;
    }
    state.supervisor.shutdown().await;
}

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    tauri::async_runtime::spawn_blocking(enumerate_audio_devices)
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn get_caption_preferences(state: State<'_, RuntimeState>) -> Result<CaptionPreferences, String> {
    state
        .caption_preferences
        .lock()
        .map_err(|_| "caption preferences lock poisoned".to_string())
        .map(|preferences| preferences.clone())
}

#[tauri::command]
fn get_session_defaults(state: State<'_, RuntimeState>) -> Result<StartSessionRequest, String> {
    let mut defaults = state
        .session_defaults
        .lock()
        .map_err(|_| "session defaults lock poisoned".to_string())?
        .clone();
    defaults.expected_state_revision = state.snapshot()?.state_revision;
    Ok(defaults)
}

#[tauri::command]
fn update_session_defaults(
    state: State<'_, RuntimeState>,
    mut defaults: StartSessionRequest,
) -> Result<StartSessionRequest, String> {
    validate_session_defaults(&defaults)?;
    defaults.expected_state_revision = state.snapshot()?.state_revision;
    *state
        .session_defaults
        .lock()
        .map_err(|_| "session defaults lock poisoned".to_string())? = defaults.clone();
    state.persist_preferences()?;
    Ok(defaults)
}

#[tauri::command]
fn update_caption_preferences(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    preferences: CaptionPreferences,
) -> Result<CaptionPreferences, String> {
    validate_caption_preferences(&preferences)?;
    *state
        .caption_preferences
        .lock()
        .map_err(|_| "caption preferences lock poisoned".to_string())? = preferences.clone();
    state.persist_preferences()?;
    state.emit_event(
        &app,
        state.snapshot()?.session_id,
        UiEventKind::SettingsChanged,
        serde_json::to_value(&preferences).map_err(|error| error.to_string())?,
    )?;
    if let Some(window) = app.get_webview_window("caption") {
        apply_caption_opacity(&window, preferences.opacity)?;
    }
    Ok(preferences)
}

#[cfg(target_os = "macos")]
fn apply_caption_opacity(window: &tauri::WebviewWindow, opacity: f64) -> Result<(), String> {
    let handle = window.window_handle().map_err(|error| error.to_string())?;
    match handle.as_raw() {
        RawWindowHandle::AppKit(handle) => {
            unsafe {
                audio_core::set_macos_window_opacity(handle.ns_view.as_ptr(), opacity);
            }
            Ok(())
        }
        _ => Err("caption window does not expose an AppKit handle".into()),
    }
}

#[cfg(not(target_os = "macos"))]
fn apply_caption_opacity(_window: &tauri::WebviewWindow, _opacity: f64) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
fn show_caption_window(app: AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("caption")
        .ok_or_else(|| "caption window is unavailable".to_string())?;
    let opacity = app
        .state::<RuntimeState>()
        .caption_preferences
        .lock()
        .map_err(|_| "caption preferences lock poisoned".to_string())?
        .opacity;
    apply_caption_opacity(&window, opacity)?;
    window.show().map_err(|error| error.to_string())?;
    window.set_focus().map_err(|error| error.to_string())
}

#[tauri::command]
fn hide_caption_window(app: AppHandle) -> Result<(), String> {
    app.get_webview_window("caption")
        .ok_or_else(|| "caption window is unavailable".to_string())?
        .hide()
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn start_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    request: StartSessionRequest,
) -> Result<SessionSnapshot, String> {
    let starting = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .start(request)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &starting)?;
    let result = async {
        let session_config = starting
            .config
            .as_ref()
            .ok_or_else(|| "missing session config".to_string())?;
        let source_kind = match session_config.audio_source {
            AppAudioSourceKind::Microphone => audio_core::AudioSourceKind::Microphone,
            AppAudioSourceKind::SystemAudio => audio_core::AudioSourceKind::SystemAudio,
            AppAudioSourceKind::SystemAudioAndMicrophone => {
                return Err("combined audio capture is not implemented".into())
            }
        };
        let capture_format =
            preferred_capture_format(source_kind, session_config.audio_device_id.as_deref())
                .map_err(|error| error.to_string())?;
        let mut sidecar_environment = state.credentials.sidecar_environment()?;
        sidecar_environment.insert(
            "ECHOLINGO_MODEL_ROOT".into(),
            state.models()?.root().to_string_lossy().into_owned(),
        );
        state
            .supervisor
            .configure_secret_environment(sidecar_environment)
            .await;
        state
            .supervisor
            .ensure_started()
            .await
            .map_err(|error| error.to_string())?;
        let mut receiver = state.supervisor.subscribe();
        let mut payload = serde_json::to_value(
            starting
                .config
                .as_ref()
                .ok_or_else(|| "missing session config".to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let object = payload
            .as_object_mut()
            .ok_or_else(|| "invalid session config".to_string())?;
        object.insert("session_id".into(), json!(starting.session_id));
        object.insert(
            "sample_rate_hz".into(),
            json!(capture_format.sample_rate_hz),
        );
        object.insert("channels".into(), json!(capture_format.channels));
        state
            .supervisor
            .send_command(SidecarCommand::StartSession(payload))
            .await
            .map_err(|error| error.to_string())?;
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(20), receiver.recv()).await {
                Ok(Ok(SidecarEvent::Ready { session_id, route }))
                    if Some(session_id) == starting.session_id =>
                {
                    start_audio_capture(&app, &state, starting.config.as_ref().unwrap()).await?;
                    break Ok(route);
                }
                Ok(Ok(SidecarEvent::Error { message, .. })) => break Err(message),
                Ok(Ok(_)) => continue,
                Ok(Err(error)) => break Err(error.to_string()),
                Err(_) => break Err("inference sidecar ready timeout".into()),
            }
        }
    }
    .await;
    match result {
        Ok(route) => {
            let selected_route = route_status(&route);
            let snapshot = state
                .core
                .lock()
                .map_err(|_| "app core lock poisoned".to_string())?
                .mark_listening(selected_route.clone())
                .map_err(|error| error.to_string())?;
            let config = snapshot
                .config
                .as_ref()
                .ok_or_else(|| "missing session config".to_string())?;
            state
                .store()?
                .create_session(&SessionDraft {
                    id: snapshot
                        .session_id
                        .ok_or_else(|| "missing session id".to_string())?,
                    title: format!(
                        "{} → {} lecture",
                        config.source_language, config.target_language
                    ),
                    source_language: config.source_language.clone(),
                    target_language: config.target_language.clone(),
                    audio_source: value_name(&config.audio_source),
                    audio_profile: config.audio_profile.clone(),
                    inference_mode: value_name(&config.inference_mode),
                    asr_backend: selected_route.asr_provider,
                    translation_backend: selected_route.translation_provider,
                    route_reason: selected_route.reason,
                    privacy: serde_json::to_value(&config.privacy)
                        .map_err(|error| error.to_string())?,
                    model_config: route,
                })
                .await
                .map_err(|error| error.to_string())?;
            state.emit_snapshot(&app, &snapshot)?;
            Ok(snapshot)
        }
        Err(error) => {
            let _ = stop_audio_capture(&state).await;
            let snapshot = state
                .core
                .lock()
                .map_err(|_| "app core lock poisoned".to_string())?
                .fail_start(&error)
                .map_err(|failure| failure.to_string())?;
            state.emit_snapshot(&app, &snapshot)?;
            Err(error)
        }
    }
}

#[tauri::command]
async fn pause_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
    if let Some(capture) = state.audio.lock().await.as_ref() {
        capture.pause().map_err(|error| error.to_string())?;
    }
    let session_id = state
        .snapshot()?
        .session_id
        .ok_or_else(|| "no active session".to_string())?;
    state
        .supervisor
        .send_command(SidecarCommand::Pause {
            session_id,
            epoch: 0,
        })
        .await
        .map_err(|error| error.to_string())?;
    let snapshot = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .pause(expected_state_revision)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &snapshot)?;
    Ok(snapshot)
}

#[tauri::command]
async fn resume_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
    let session_id = state
        .snapshot()?
        .session_id
        .ok_or_else(|| "no active session".to_string())?;
    state
        .supervisor
        .send_command(SidecarCommand::Resume {
            session_id,
            epoch: 1,
        })
        .await
        .map_err(|error| error.to_string())?;
    if let Some(capture) = state.audio.lock().await.as_ref() {
        capture.resume().map_err(|error| error.to_string())?;
    }
    let snapshot = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .resume(expected_state_revision)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &snapshot)?;
    Ok(snapshot)
}

#[tauri::command]
async fn stop_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
    let stopping = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .begin_stop(expected_state_revision)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &stopping)?;
    stop_audio_capture(&state).await?;
    if let Some(session_id) = stopping.session_id {
        let mut receiver = state.supervisor.subscribe();
        state
            .supervisor
            .send_command(SidecarCommand::FinishSession { session_id })
            .await
            .map_err(|error| error.to_string())?;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(10), async move {
            while let Ok(event) = receiver.recv().await {
                if matches!(event, SidecarEvent::SessionFinished { session_id: finished } if finished == session_id) { break; }
            }
        }).await;
    }
    let completed = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .complete()
        .map_err(|error| error.to_string())?;
    if let Some(session_id) = completed.session_id {
        state
            .store()?
            .complete_session(session_id, &[])
            .await
            .map_err(|error| error.to_string())?;
    }
    state.emit_snapshot(&app, &completed)?;
    Ok(completed)
}

#[tauri::command]
async fn history_search(
    state: State<'_, RuntimeState>,
    query: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<SessionRecord>, String> {
    state
        .store()?
        .search(query.as_deref().unwrap_or(""), limit.unwrap_or(100))
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn history_open(
    state: State<'_, RuntimeState>,
    session_id: String,
) -> Result<SessionDetail, String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    state
        .store()?
        .detail(id)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn history_rename(
    state: State<'_, RuntimeState>,
    session_id: String,
    title: String,
) -> Result<(), String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    state
        .store()?
        .rename(id, &title)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn history_delete(state: State<'_, RuntimeState>, session_id: String) -> Result<(), String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    state
        .store()?
        .delete(id)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn history_export(
    state: State<'_, RuntimeState>,
    session_id: String,
    format: ExportFormat,
) -> Result<String, String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    state
        .store()?
        .export(id, format)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn history_export_to_path(
    state: State<'_, RuntimeState>,
    session_id: String,
    format: ExportFormat,
    path: String,
) -> Result<(), String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    let contents = state
        .store()?
        .export(id, format)
        .await
        .map_err(|error| error.to_string())?;
    let destination = PathBuf::from(path);
    let parent = destination
        .parent()
        .ok_or_else(|| "export destination has no parent".to_string())?;
    if !parent.is_dir() {
        return Err("export destination directory is unavailable".into());
    }
    let temporary = destination.with_extension("echolingo-export.tmp");
    std::fs::write(&temporary, contents).map_err(|error| error.to_string())?;
    std::fs::rename(&temporary, &destination).map_err(|error| error.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![
            get_app_snapshot,
            onboarding_status,
            complete_onboarding,
            audio_permission_status,
            request_audio_permission,
            test_audio_input,
            credential_status,
            set_cloud_credentials,
            clear_cloud_credentials,
            list_models,
            install_model,
            verify_model,
            delete_model,
            list_audio_devices,
            get_caption_preferences,
            get_session_defaults,
            update_session_defaults,
            update_caption_preferences,
            show_caption_window,
            hide_caption_window,
            start_session,
            pause_session,
            resume_session,
            stop_session,
            history_search,
            history_open,
            history_rename,
            history_delete,
            history_export,
            history_export_to_path,
        ])
        .setup(|app| {
            let state = app.state::<RuntimeState>();
            let app_data_directory = app.path().app_data_dir()?;
            let database_path = app_data_directory.join("history.sqlite");
            let store = tauri::async_runtime::block_on(TranscriptStore::open(database_path))
                .map_err(std::io::Error::other)?;
            state
                .store
                .set(store)
                .map_err(|_| std::io::Error::other("store already initialized"))?;
            let preferences_path = app_data_directory.join("preferences.json");
            state
                .models
                .set(ModelManager::new(app_data_directory.join("models")))
                .map_err(|_| std::io::Error::other("model manager already initialized"))?;
            let preferences = load_preferences(&preferences_path);
            state
                .onboarding_complete
                .store(preferences.onboarding_complete, Ordering::Release);
            *state
                .session_defaults
                .lock()
                .map_err(|_| std::io::Error::other("session defaults lock poisoned"))? =
                preferences.session;
            *state
                .caption_preferences
                .lock()
                .map_err(|_| std::io::Error::other("caption preferences lock poisoned"))? =
                preferences.caption;
            state
                .preferences_path
                .set(preferences_path)
                .map_err(|_| std::io::Error::other("preferences path already initialized"))?;
            let snapshot = state.snapshot().map_err(std::io::Error::other)?;
            state
                .emit_snapshot(app.handle(), &snapshot)
                .map_err(std::io::Error::other)?;
            forward_sidecar_events(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<RuntimeState>();
                if state.shutting_down.swap(true, Ordering::SeqCst) {
                    return;
                }
                api.prevent_close();
                let window = window.clone();
                tauri::async_runtime::spawn(async move {
                    shutdown_application(window.app_handle()).await;
                    let _ = window.destroy();
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run EchoLingo desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_preferences_round_trip_and_reject_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let mut preferences = DesktopPreferences::default();
        preferences.session.source_language = "ja".into();
        preferences.session.target_language = "en".into();
        preferences.session.audio_profile = "lecture".into();
        preferences.caption.opacity = 0.75;
        preferences.onboarding_complete = true;
        save_preferences(&path, &preferences).unwrap();
        let serialized = std::fs::read_to_string(&path).unwrap();
        assert!(!serialized.contains("API_KEY") && !serialized.contains("DASHSCOPE"));

        let loaded = load_preferences(&path);
        assert_eq!(loaded.session.source_language, "ja");
        assert_eq!(loaded.caption.opacity, 0.75);
        assert!(loaded.onboarding_complete);

        std::fs::write(&path, b"not-json").unwrap();
        let fallback = load_preferences(&path);
        assert_eq!(fallback.session.source_language, "en");
        assert_eq!(fallback.caption, CaptionPreferences::default());
    }

    #[test]
    fn desktop_preferences_reject_invalid_language_pair() {
        let mut defaults = StartSessionRequest::default();
        defaults.target_language = defaults.source_language.clone();
        assert!(validate_session_defaults(&defaults).is_err());
    }

    #[test]
    fn cloud_credential_validation_never_echoes_secret() {
        let input = CloudCredentialInput {
            api_key: "sekrit".into(),
            workspace_id: "workspace".into(),
        };
        let error = validate_cloud_credentials(&input).unwrap_err();
        assert!(!error.contains("sekrit"));
    }
}
