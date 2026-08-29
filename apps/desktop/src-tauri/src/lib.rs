use app_core::{AppCore, SessionSnapshot, StartSessionRequest};
use inference_ipc::{UiEventEnvelope, UiEventKind, PROTOCOL_VERSION};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

const UI_EVENT_CHANNEL: &str = "echolingo://ui-event";

#[derive(Default)]
struct RuntimeState {
    core: Mutex<AppCore>,
    event_sequence: AtomicU64,
}

impl RuntimeState {
    fn snapshot(&self) -> Result<SessionSnapshot, String> {
        self.core
            .lock()
            .map_err(|_| "app core lock poisoned".to_string())
            .map(|core| core.snapshot())
    }

    fn emit_snapshot(&self, app: &AppHandle, snapshot: &SessionSnapshot) -> Result<(), String> {
        let event = UiEventEnvelope {
            schema_version: PROTOCOL_VERSION,
            sequence: self.event_sequence.fetch_add(1, Ordering::Relaxed) + 1,
            session_id: snapshot.session_id,
            kind: UiEventKind::SessionState,
            emitted_at_unix_ms: chrono::Utc::now().timestamp_millis(),
            payload: serde_json::to_value(snapshot).map_err(|error| error.to_string())?,
        };
        app.emit(UI_EVENT_CHANNEL, event)
            .map_err(|error| error.to_string())
    }
}

#[tauri::command]
fn get_app_snapshot(state: State<'_, RuntimeState>) -> Result<SessionSnapshot, String> {
    state.snapshot()
}

#[tauri::command]
fn start_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    request: StartSessionRequest,
) -> Result<SessionSnapshot, String> {
    let snapshot = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?
        .start(request)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &snapshot)?;
    Ok(snapshot)
}

#[tauri::command]
fn pause_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
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
fn resume_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
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
fn stop_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    expected_state_revision: u64,
) -> Result<SessionSnapshot, String> {
    let mut core = state
        .core
        .lock()
        .map_err(|_| "app core lock poisoned".to_string())?;
    let stopping = core
        .begin_stop(expected_state_revision)
        .map_err(|error| error.to_string())?;
    state.emit_snapshot(&app, &stopping)?;
    // Replaced by the bounded sidecar drain once the supervisor is connected.
    let completed = core.complete().map_err(|error| error.to_string())?;
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
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run EchoLingo desktop application");
}
