use app_core::{
    AppCore, AudioSourceKind as AppAudioSourceKind, BackendHealth, LiveMetrics, RouteStatus,
    SessionSnapshot, StartSessionRequest,
};
use audio_core::{
    list_audio_devices as enumerate_audio_devices, start_microphone, start_system_audio,
    AudioCaptureSession, AudioDevice, AudioSourceEvent,
};
use inference_ipc::{
    AudioFrameHeader, InferenceSupervisor, SidecarCommand, SidecarEvent, SidecarLaunchConfig,
    UiEventEnvelope, UiEventKind, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use transcript_store::{
    ExportFormat, SegmentDraft, SessionDetail, SessionDraft, SessionRecord, TranscriptStore,
};

const UI_EVENT_CHANNEL: &str = "echolingo://ui-event";

struct RuntimeState {
    core: Mutex<AppCore>,
    event_sequence: AtomicU64,
    audio_sequence: AtomicU64,
    audio: tokio::sync::Mutex<Option<AudioCaptureSession>>,
    shutting_down: std::sync::atomic::AtomicBool,
    supervisor: Arc<InferenceSupervisor>,
    store: tokio::sync::OnceCell<TranscriptStore>,
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
            supervisor: InferenceSupervisor::new(SidecarLaunchConfig::development(project_root)),
            store: tokio::sync::OnceCell::new(),
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
}

#[tauri::command]
fn get_app_snapshot(state: State<'_, RuntimeState>) -> Result<SessionSnapshot, String> {
    state.snapshot()
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
            if matches!(kind, "stable" | "final") && !text.is_empty() {
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
                        source_text: payload["committed_text"]
                            .as_str()
                            .filter(|value| !value.is_empty())
                            .unwrap_or(text)
                            .to_string(),
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
            let session_id = state
                .snapshot()
                .ok()
                .and_then(|snapshot| snapshot.session_id);
            let _ = state.emit_event(&app, session_id, kind, payload);
            if let Some(session_id) = session_id {
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
            for (part, samples) in frame.samples.chunks(max_frames).enumerate() {
                let sequence = state.audio_sequence.fetch_add(1, Ordering::Relaxed) + 1;
                let header = AudioFrameHeader {
                    flags: u16::from(frame.overflow && part == 0),
                    sequence,
                    capture_monotonic_ns: frame.capture_monotonic_ns,
                    sample_rate_hz: frame.sample_rate_hz,
                    channels: frame.channels,
                    frame_count: samples.len() as u16,
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

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    tauri::async_runtime::spawn_blocking(enumerate_audio_devices)
        .await
        .map_err(|error| error.to_string())?
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
        object.insert("sample_rate_hz".into(), json!(48_000));
        object.insert("channels".into(), json!(1));
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![
            get_app_snapshot,
            list_audio_devices,
            start_session,
            pause_session,
            resume_session,
            stop_session,
            history_search,
            history_open,
            history_rename,
            history_delete,
            history_export,
        ])
        .setup(|app| {
            let state = app.state::<RuntimeState>();
            let database_path = app.path().app_data_dir()?.join("history.sqlite");
            let store = tauri::async_runtime::block_on(TranscriptStore::open(database_path))
                .map_err(std::io::Error::other)?;
            state
                .store
                .set(store)
                .map_err(|_| std::io::Error::other("store already initialized"))?;
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
                    let state = window.state::<RuntimeState>();
                    let _ = stop_audio_capture(&state).await;
                    state.supervisor.shutdown().await;
                    let _ = window.destroy();
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run EchoLingo desktop application");
}
