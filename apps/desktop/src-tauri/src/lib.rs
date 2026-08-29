use app_core::{
    AppCore, BackendHealth, LiveMetrics, RouteStatus, SessionSnapshot, StartSessionRequest,
};
use inference_ipc::{
    InferenceSupervisor, SidecarCommand, SidecarEvent, SidecarLaunchConfig, UiEventEnvelope,
    UiEventKind, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

const UI_EVENT_CHANNEL: &str = "echolingo://ui-event";

struct RuntimeState {
    core: Mutex<AppCore>,
    event_sequence: AtomicU64,
    supervisor: Arc<InferenceSupervisor>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        Self {
            core: Mutex::new(AppCore::default()),
            event_sequence: AtomicU64::new(0),
            supervisor: InferenceSupervisor::new(SidecarLaunchConfig::development(project_root)),
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

fn forward_sidecar_events(app: AppHandle) {
    let mut receiver = app.state::<RuntimeState>().supervisor.subscribe();
    tauri::async_runtime::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            let state = app.state::<RuntimeState>();
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
        }
    });
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
                    break Ok(route)
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
            let snapshot = state
                .core
                .lock()
                .map_err(|_| "app core lock poisoned".to_string())?
                .mark_listening(route_status(&route))
                .map_err(|error| error.to_string())?;
            state.emit_snapshot(&app, &snapshot)?;
            Ok(snapshot)
        }
        Err(error) => {
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
    state.emit_snapshot(&app, &completed)?;
    Ok(completed)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![
            get_app_snapshot,
            start_session,
            pause_session,
            resume_session,
            stop_session,
        ])
        .setup(|app| {
            let state = app.state::<RuntimeState>();
            let snapshot = state.snapshot().map_err(std::io::Error::other)?;
            state
                .emit_snapshot(app.handle(), &snapshot)
                .map_err(std::io::Error::other)?;
            forward_sidecar_events(app.handle().clone());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run EchoLingo desktop application");
}
