//! AI assistant jobs driven by the shell: session notes,
//! session titles, lecture-context import and the provider probe.
//!
//! The assistant runs inside the Python sidecar. This module gates each
//! request on the user's provider choice and consent, gathers the transcript
//! from SQLite, sends one `assistant_request`, follows that request's events
//! and persists the result. React sees only the commands below and the
//! `assistant_update` / `history_changed` UI events; it never receives file
//! paths. Transcript text, file contents and keys are never logged: the
//! desktop log records job ids, tasks, states, error codes and durations.

use crate::{
    emit_history_changed, missing_credential_labels, CredentialGroupStatus, RuntimeState,
    SidecarUser,
};
use app_core::{catalog, AssistantPreferences, ProviderCatalog, SESSION_CONTEXT_MAX_CHARS};
use inference_ipc::{SidecarCommand, SidecarEvent, UiEventKind};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tokio::sync::{broadcast, watch};
use tokio::time::Instant;
use transcript_store::{
    SessionDetail, SessionNotes, SessionNotesDraft, TranscriptStore, TITLE_SOURCE_USER,
};
use uuid::Uuid;

/// Returned by operations that would restart the sidecar under a running job.
pub(crate) const ASSISTANT_BUSY: &str = "An AI assistant task is running";
pub(crate) const MAX_ATTACHMENTS: usize = 5;
pub(crate) const MAX_ATTACHMENT_BYTES: u64 = 25 * 1024 * 1024;
pub(crate) const ATTACHMENT_EXTENSIONS: &[&str] =
    &["pdf", "pptx", "docx", "txt", "md", "markdown", "tex", "csv"];
/// Picked files remembered for a later `create_session_notes`; oldest go first.
const MAX_REMEMBERED_ATTACHMENTS: usize = 64;
/// A session is named automatically only once its transcript is this long.
pub(crate) const AUTO_TITLE_MIN_WORDS: usize = 30;
const TITLE_MAX_CHARS: usize = 120;
/// Minimum spacing of `assistant_update` emits that only add text.
pub(crate) const TEXT_UI_INTERVAL: Duration = Duration::from_millis(100);
const NO_TRANSCRIPT: &str = "This session has no transcript yet";
const CONSENT_REQUIRED: &str =
    "Allow the AI assistant to read transcripts and files in Settings → AI assistant first";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AssistantTask {
    Notes,
    Title,
    Context,
    Probe,
}

impl AssistantTask {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Notes => "notes",
            Self::Title => "title",
            Self::Context => "context",
            Self::Probe => "probe",
        }
    }

    /// How long the sidecar may take once the request was sent (launching
    /// the sidecar has its own startup timeout).
    pub(crate) fn timeout(self) -> Duration {
        match self {
            Self::Notes => Duration::from_secs(15 * 60),
            Self::Title => Duration::from_secs(90),
            Self::Context => Duration::from_secs(3 * 60),
            Self::Probe => Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JobState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// `{code, message}` of a failed job, as the sidecar reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct AssistantError {
    pub code: String,
    pub message: String,
}

impl AssistantError {
    pub(crate) fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    fn cancelled() -> Self {
        Self::new("cancelled", "The AI assistant task was cancelled")
    }

    fn storage(error: impl std::fmt::Display) -> Self {
        Self::new("storage_error", error.to_string())
    }

    /// The `result` of an `assistant_result` with `ok: false`.
    fn from_result(result: &Value) -> Self {
        Self::new(
            text_field(result, "code").unwrap_or("assistant_failed"),
            text_field(result, "message")
                .unwrap_or("The AI assistant could not complete the task"),
        )
    }
}

fn text_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value[key]
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

/// Payload of the `assistant_update` UI event and of `get_session_notes().job`
/// (`AssistantJob` in `types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AssistantJobView {
    pub job_id: Uuid,
    pub session_id: Option<Uuid>,
    pub task: AssistantTask,
    pub state: JobState,
    pub stage: Option<String>,
    pub progress: Option<Value>,
    /// Full markdown produced so far.
    pub text: String,
    pub result: Option<Value>,
    pub error: Option<AssistantError>,
}

struct JobEntry {
    session_id: Option<Uuid>,
    task: AssistantTask,
    /// Sends transcripts or files to the model (stopped when consent ends).
    uploads: bool,
    stage: Option<String>,
    progress: Option<Value>,
    text: String,
    started: std::time::Instant,
    /// The request reached the sidecar; restarting it now would cut it off.
    dispatched: bool,
    cancel: watch::Sender<bool>,
}

impl JobEntry {
    fn view(&self, job_id: Uuid) -> AssistantJobView {
        AssistantJobView {
            job_id,
            session_id: self.session_id,
            task: self.task,
            state: JobState::Running,
            stage: self.stage.clone(),
            progress: self.progress.clone(),
            text: self.text.clone(),
            result: None,
            error: None,
        }
    }
}

/// A file the user picked; its path never leaves Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PickedFile {
    pub path: String,
    pub name: String,
    pub size_bytes: u64,
    pub extension: String,
}

/// What the webview learns about a picked file (`NoteAttachment`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct NoteAttachment {
    pub id: Uuid,
    pub name: String,
    pub size_bytes: u64,
    pub extension: String,
}

/// Running assistant jobs (finished jobs are removed) and picked files.
#[derive(Default)]
pub(crate) struct AssistantRegistry {
    jobs: Mutex<HashMap<Uuid, JobEntry>>,
    attachments: Mutex<Vec<(Uuid, PickedFile)>>,
}

impl AssistantRegistry {
    fn jobs(&self) -> MutexGuard<'_, HashMap<Uuid, JobEntry>> {
        self.jobs.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn attachments(&self) -> MutexGuard<'_, Vec<(Uuid, PickedFile)>> {
        self.attachments
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Register a running job. A session has at most one notes job and one
    /// title job at a time.
    pub(crate) fn register(
        &self,
        task: AssistantTask,
        session_id: Option<Uuid>,
        uploads: bool,
    ) -> Result<(Uuid, watch::Receiver<bool>, AssistantJobView), String> {
        let mut jobs = self.jobs();
        if session_id.is_some()
            && jobs
                .values()
                .any(|job| job.session_id == session_id && job.task == task)
        {
            return Err(match task {
                AssistantTask::Notes => "AI notes are already being written for this session",
                AssistantTask::Title => "A title is already being generated for this session",
                _ => ASSISTANT_BUSY,
            }
            .into());
        }
        let job_id = Uuid::new_v4();
        let (cancel, cancelled) = watch::channel(false);
        let entry = JobEntry {
            session_id,
            task,
            uploads,
            stage: None,
            progress: None,
            text: String::new(),
            started: std::time::Instant::now(),
            dispatched: false,
            cancel,
        };
        let view = entry.view(job_id);
        jobs.insert(job_id, entry);
        Ok((job_id, cancelled, view))
    }

    /// Whether a registered job sends transcripts or files to the model.
    pub(crate) fn uploads(&self, job_id: Uuid) -> bool {
        self.jobs().get(&job_id).is_some_and(|job| job.uploads)
    }

    pub(crate) fn mark_dispatched(&self, job_id: Uuid) {
        if let Some(job) = self.jobs().get_mut(&job_id) {
            job.dispatched = true;
        }
    }

    /// Any job registered, including one still waiting for the sidecar.
    pub(crate) fn any_running(&self) -> bool {
        !self.jobs().is_empty()
    }

    /// Any job whose request the sidecar is working on.
    pub(crate) fn any_in_flight(&self) -> bool {
        self.jobs().values().any(|job| job.dispatched)
    }

    /// Record the latest progress and text; returns the view to emit.
    pub(crate) fn update(&self, job_id: Uuid, progress: &JobProgress) -> Option<AssistantJobView> {
        let mut jobs = self.jobs();
        let job = jobs.get_mut(&job_id)?;
        job.stage.clone_from(&progress.stage);
        job.progress.clone_from(&progress.progress);
        job.text.clone_from(&progress.text);
        Some(job.view(job_id))
    }

    pub(crate) fn running(
        &self,
        session_id: Uuid,
        task: AssistantTask,
    ) -> Option<AssistantJobView> {
        self.jobs()
            .iter()
            .find(|(_, job)| job.session_id == Some(session_id) && job.task == task)
            .map(|(job_id, job)| job.view(*job_id))
    }

    /// Ask a job to stop; `false` when it is no longer running.
    pub(crate) fn cancel(&self, job_id: Uuid) -> bool {
        self.jobs()
            .get(&job_id)
            .map(|job| job.cancel.send_replace(true))
            .is_some()
    }

    pub(crate) fn cancel_session_jobs(&self, session_id: Uuid) {
        for job in self.jobs().values() {
            if job.session_id == Some(session_id) {
                job.cancel.send_replace(true);
            }
        }
    }

    /// Consent was withdrawn: stop every job that sends transcripts or files.
    pub(crate) fn cancel_uploading_jobs(&self) {
        for job in self.jobs().values() {
            if job.uploads {
                job.cancel.send_replace(true);
            }
        }
    }

    /// Drop a job without a final update (see [`JobGuard`]).
    fn forget(&self, job_id: Uuid) {
        self.jobs().remove(&job_id);
    }

    /// Remove a job and return its final view and how long it ran. `text`
    /// replaces the streamed text (the saved notes, for example).
    pub(crate) fn finish(
        &self,
        job_id: Uuid,
        state: JobState,
        text: Option<String>,
        result: Option<Value>,
        error: Option<AssistantError>,
    ) -> Option<(AssistantJobView, Duration)> {
        let job = self.jobs().remove(&job_id)?;
        let mut view = job.view(job_id);
        view.state = state;
        if let Some(text) = text {
            view.text = text;
        }
        view.result = result;
        view.error = error;
        Some((view, job.started.elapsed()))
    }

    /// Remember validated files under fresh opaque ids.
    pub(crate) fn remember_attachments(&self, files: Vec<PickedFile>) -> Vec<NoteAttachment> {
        let mut remembered = self.attachments();
        let picked: Vec<(Uuid, PickedFile)> = files
            .into_iter()
            .map(|file| (Uuid::new_v4(), file))
            .collect();
        remembered.extend(picked.iter().cloned());
        let excess = remembered.len().saturating_sub(MAX_REMEMBERED_ATTACHMENTS);
        remembered.drain(..excess);
        picked
            .into_iter()
            .map(|(id, file)| NoteAttachment {
                id,
                name: file.name,
                size_bytes: file.size_bytes,
                extension: file.extension,
            })
            .collect()
    }

    /// Turn attachment ids from the webview back into files, checking each
    /// file again (it may have grown or gone since it was picked).
    pub(crate) fn resolve_attachments(
        &self,
        ids: &[String],
    ) -> Result<Vec<(Uuid, PickedFile)>, String> {
        let mut unique: Vec<Uuid> = Vec::new();
        for id in ids {
            let id: Uuid = id
                .parse()
                .map_err(|_| "invalid attachment id".to_string())?;
            if !unique.contains(&id) {
                unique.push(id);
            }
        }
        if unique.len() > MAX_ATTACHMENTS {
            return Err(format!("Attach at most {MAX_ATTACHMENTS} files"));
        }
        let remembered = self.attachments().clone();
        unique
            .into_iter()
            .map(|id| {
                let (_, file) = remembered
                    .iter()
                    .find(|(known, _)| *known == id)
                    .ok_or_else(|| {
                        "An attached file is no longer available; choose the files again"
                            .to_string()
                    })?;
                inspect_attachment(Path::new(&file.path)).map(|file| (id, file))
            })
            .collect()
    }
}

/// Removes a job that never reached [`finish_job`] (an unwinding or dropped
/// task), so a lost job can never keep the sidecar pinned or block probes.
struct JobGuard<'a> {
    registry: &'a AssistantRegistry,
    job_id: Uuid,
}

impl Drop for JobGuard<'_> {
    fn drop(&mut self) {
        self.registry.forget(self.job_id);
    }
}

/// Check one picked file: whitelisted extension, a regular file, at most
/// 25 MB and a UTF-8 path. Messages name the file, never its path.
pub(crate) fn inspect_attachment(path: &Path) -> Result<PickedFile, String> {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "The file".into());
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if !ATTACHMENT_EXTENSIONS.contains(&extension.as_str()) {
        return Err(format!(
            "{name} is not a supported file (PDF, PPTX, DOCX, TXT, Markdown, TeX or CSV)"
        ));
    }
    let metadata = std::fs::metadata(path).map_err(|_| format!("{name} cannot be read"))?;
    if !metadata.is_file() {
        return Err(format!("{name} is not a file"));
    }
    if metadata.len() > MAX_ATTACHMENT_BYTES {
        return Err(format!("{name} is larger than 25 MB"));
    }
    let path = path
        .to_str()
        .ok_or_else(|| format!("{name} has a path the assistant cannot open"))?;
    Ok(PickedFile {
        path: path.to_string(),
        name,
        size_bytes: metadata.len(),
        extension,
    })
}

pub(crate) fn inspect_attachments(paths: &[PathBuf]) -> Result<Vec<PickedFile>, String> {
    if paths.len() > MAX_ATTACHMENTS {
        return Err(format!("Choose at most {MAX_ATTACHMENTS} files"));
    }
    paths.iter().map(|path| inspect_attachment(path)).collect()
}

/// Progress and markdown accumulated from one request's sidecar events.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct JobProgress {
    pub stage: Option<String>,
    pub progress: Option<Value>,
    pub text: String,
    last_seq: Option<u64>,
    /// The next delta belongs at the top of the document: long sessions
    /// stream their sections first and the title/overview last
    /// (`assistant_progress.detail.placement == "prepend"`).
    prepend_next: bool,
    /// The broadcast receiver fell behind; `text` may miss deltas (the final
    /// result carries the authoritative markdown).
    pub lagged: bool,
}

/// Follow the events of `request_id` until its `assistant_result`, the
/// deadline, cancellation or a sidecar disconnect. `emit` sees every
/// progress change at once and text growth at most every
/// [`TEXT_UI_INTERVAL`] (the latest text is flushed when the interval ends).
/// Subscribe `events` before sending the request.
pub(crate) async fn consume_assistant_events(
    events: &mut broadcast::Receiver<SidecarEvent>,
    request_id: Uuid,
    deadline: Instant,
    cancel: &mut watch::Receiver<bool>,
    mut emit: impl FnMut(&JobProgress),
) -> (JobProgress, Result<Value, AssistantError>) {
    let mut progress = JobProgress::default();
    let mut last_emit: Option<Instant> = None;
    let mut pending = false;
    let mut cancel_open = true;
    loop {
        if *cancel.borrow() {
            return (progress, Err(AssistantError::cancelled()));
        }
        let flush_at = last_emit.map(|at| at + TEXT_UI_INTERVAL).unwrap_or(deadline);
        tokio::select! {
            biased;
            changed = cancel.changed(), if cancel_open => {
                if changed.is_err() {
                    cancel_open = false;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                return (
                    progress,
                    Err(AssistantError::new(
                        "timeout",
                        "The AI assistant took too long and was stopped",
                    )),
                );
            }
            _ = tokio::time::sleep_until(flush_at), if pending => {
                pending = false;
                last_emit = Some(Instant::now());
                emit(&progress);
            }
            received = events.recv() => match received {
                Ok(SidecarEvent::AssistantProgress { request_id: id, stage, detail })
                    if id == request_id =>
                {
                    progress.prepend_next = detail["placement"] == "prepend";
                    progress.stage = Some(stage);
                    progress.progress = (!detail.is_null()).then_some(detail);
                    pending = false;
                    last_emit = Some(Instant::now());
                    emit(&progress);
                }
                Ok(SidecarEvent::AssistantDelta { request_id: id, seq, text })
                    if id == request_id =>
                {
                    // Deltas are append-only in `seq` order; drop replays.
                    if progress.last_seq.is_some_and(|last| seq <= last) {
                        continue;
                    }
                    progress.last_seq = Some(seq);
                    if std::mem::take(&mut progress.prepend_next) {
                        progress.text.insert_str(0, &text);
                    } else {
                        progress.text.push_str(&text);
                    }
                    let now = Instant::now();
                    if last_emit.is_none_or(|at| now >= at + TEXT_UI_INTERVAL) {
                        pending = false;
                        last_emit = Some(now);
                        emit(&progress);
                    } else {
                        pending = true;
                    }
                }
                Ok(SidecarEvent::AssistantResult { request_id: id, ok, result })
                    if id == request_id =>
                {
                    let outcome = if ok {
                        Ok(result)
                    } else {
                        Err(AssistantError::from_result(&result))
                    };
                    return (progress, outcome);
                }
                Ok(SidecarEvent::Error { code, .. }) if code == "sidecar_disconnected" => {
                    return (
                        progress,
                        Err(AssistantError::new(
                            "sidecar_disconnected",
                            "The inference process stopped before the AI assistant finished",
                        )),
                    );
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(_)) => progress.lagged = true,
                Err(broadcast::error::RecvError::Closed) => {
                    return (
                        progress,
                        Err(AssistantError::new(
                            "sidecar_disconnected",
                            "The inference event stream closed",
                        )),
                    );
                }
            }
        }
    }
}

/// The provider and model a job uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LlmChoice {
    pub group: String,
    pub model: String,
    pub display_name: String,
}

impl LlmChoice {
    fn payload(&self) -> Value {
        json!({"group": self.group, "model": self.model})
    }
}

/// The assistant preference resolved against the catalog and credentials.
pub(crate) struct Readiness {
    pub prefs: AssistantPreferences,
    /// `None` when no (known) provider is chosen.
    pub llm: Option<LlmChoice>,
    /// Credential labels the chosen group still needs.
    pub missing: Vec<String>,
}

/// Resolve the preference: the preset of the chosen group, its credential
/// status and the model (preference, else preset default, else the group's
/// `chat_model` setting, else empty for the sidecar to decide).
pub(crate) fn readiness_from(
    prefs: AssistantPreferences,
    catalog: &ProviderCatalog,
    status_for: impl FnOnce(&str) -> Result<CredentialGroupStatus, String>,
) -> Result<Readiness, String> {
    let preset = prefs.preset(catalog);
    let Some((preset, group)) =
        preset.and_then(|preset| catalog.group(&preset.group_id).map(|group| (preset, group)))
    else {
        return Ok(Readiness {
            prefs,
            llm: None,
            missing: Vec::new(),
        });
    };
    let status = status_for(&group.id)?;
    let missing = missing_credential_labels(group, &status);
    let mut model = prefs.effective_model(catalog);
    if model.is_empty() {
        model = status
            .settings
            .get("chat_model")
            .map(|value| value.trim().to_string())
            .unwrap_or_default();
    }
    let llm = LlmChoice {
        group: group.id.clone(),
        model,
        display_name: preset.display_name.clone(),
    };
    Ok(Readiness {
        prefs,
        llm: Some(llm),
        missing,
    })
}

fn readiness(state: &RuntimeState) -> Result<Readiness, String> {
    let prefs = state.assistant_preferences()?;
    let providers = state.provider_settings()?;
    readiness_from(prefs, catalog(), |group_id| {
        state.credentials.group_status(group_id, &providers)
    })
}

/// The provider to call, or why the assistant cannot run. `need_consent`
/// for every task that sends transcripts or files.
pub(crate) fn gate(readiness: &Readiness, need_consent: bool) -> Result<LlmChoice, String> {
    let llm = readiness.llm.as_ref().ok_or_else(|| {
        "Choose an AI assistant provider in Settings → AI assistant first".to_string()
    })?;
    if !readiness.missing.is_empty() {
        return Err(format!(
            "Save the {} credentials first (missing: {})",
            llm.display_name,
            readiness.missing.join(", ")
        ));
    }
    if need_consent && !readiness.prefs.transcript_upload_allowed {
        return Err(CONSENT_REQUIRED.into());
    }
    Ok(llm.clone())
}

/// `assistant_status` result (`AssistantStatus` in `types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct AssistantStatus {
    pub configured: bool,
    pub consent: bool,
    pub provider_group: String,
    pub model: String,
    pub display_name: String,
    pub key_available: bool,
    pub auto_title: bool,
}

impl From<&Readiness> for AssistantStatus {
    fn from(readiness: &Readiness) -> Self {
        let llm = readiness.llm.as_ref();
        Self {
            configured: llm.is_some(),
            consent: readiness.prefs.transcript_upload_allowed,
            provider_group: readiness.prefs.provider_group.clone(),
            model: llm.map(|llm| llm.model.clone()).unwrap_or_default(),
            display_name: llm.map(|llm| llm.display_name.clone()).unwrap_or_default(),
            key_available: llm.is_some() && readiness.missing.is_empty(),
            auto_title: readiness.prefs.auto_title,
        }
    }
}

/// Committed source units in order, `[{start_ms, end_ms, text}]`.
pub(crate) fn transcript_lines(detail: &SessionDetail) -> Vec<Value> {
    detail
        .segments
        .iter()
        .filter(|segment| !segment.source_text.trim().is_empty())
        .map(|segment| {
            json!({
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.source_text.trim(),
            })
        })
        .collect()
}

/// `session` of a notes/title payload. `duration_ms` is the transcript's
/// audio extent, else the wall-clock length of the session.
pub(crate) fn session_payload(detail: &SessionDetail) -> Value {
    let session = &detail.session;
    let audio_extent = detail
        .segments
        .iter()
        .map(|segment| segment.end_ms)
        .fold(0.0_f64, f64::max);
    let wall_clock = session.ended_at.as_deref().and_then(|ended| {
        let started = chrono::DateTime::parse_from_rfc3339(&session.started_at).ok()?;
        let ended = chrono::DateTime::parse_from_rfc3339(ended).ok()?;
        Some((ended - started).num_milliseconds().max(0) as f64)
    });
    let duration_ms = if audio_extent > 0.0 {
        audio_extent
    } else {
        wall_clock.unwrap_or(0.0)
    };
    json!({
        "id": session.id,
        "title": session.title,
        "source_language": session.source_language,
        "target_language": session.target_language,
        "started_at": session.started_at,
        "duration_ms": duration_ms.round() as i64,
        "context": session.context,
    })
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F
    )
}

/// Words in `text`: whitespace-separated tokens, where every Chinese or
/// Japanese character counts as a word (those scripts do not use spaces).
pub(crate) fn word_count(text: &str) -> usize {
    text.split_whitespace()
        .map(|token| {
            let cjk = token.chars().filter(|character| is_cjk(*character)).count();
            let other = token
                .chars()
                .any(|character| character.is_alphanumeric() && !is_cjk(character));
            cjk + usize::from(other)
        })
        .sum()
}

/// Whether a finished session should be named automatically (the caller has
/// already checked the preference, consent and credentials).
pub(crate) fn auto_title_wanted(detail: &SessionDetail) -> bool {
    detail.session.title_source != TITLE_SOURCE_USER
        && detail
            .segments
            .iter()
            .map(|segment| word_count(&segment.source_text))
            .sum::<usize>()
            >= AUTO_TITLE_MIN_WORDS
}

fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

/// A single-line title from model output: first non-empty line, without a
/// Markdown heading marker, one pair of wrapping quotes or brackets, control
/// characters or repeated spaces, at most 120 characters.
pub(crate) fn clean_title(raw: &str) -> String {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    let mut title = line.trim_start_matches('#').trim();
    for (open, close) in [
        ("\"", "\""),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
        ("**", "**"),
        ("*", "*"),
        ("`", "`"),
    ] {
        if title.len() > open.len() + close.len()
            && title.starts_with(open)
            && title.ends_with(close)
        {
            title = title[open.len()..title.len() - close.len()].trim();
        }
    }
    let collapsed = title
        .split(|character: char| character.is_whitespace() || character.is_control())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    truncate_chars(&collapsed, TITLE_MAX_CHARS).trim_end().to_string()
}

/// `generate_session_title` result. `applied` is `false` when the user
/// named the session (AI titles never replace a user title).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TitleOutcome {
    pub title: String,
    pub applied: bool,
}

pub(crate) async fn apply_title_result(
    store: &TranscriptStore,
    session_id: Uuid,
    result: &Value,
) -> Result<TitleOutcome, AssistantError> {
    let title = clean_title(result["title"].as_str().unwrap_or(""));
    if title.is_empty() {
        return Err(AssistantError::new(
            "empty_result",
            "The AI assistant returned no title",
        ));
    }
    let applied = store
        .set_ai_title(session_id, &title)
        .await
        .map_err(AssistantError::storage)?;
    Ok(TitleOutcome { title, applied })
}

/// Saved notes plus the title the notes proposed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoredNotes {
    pub notes: SessionNotes,
    pub title: String,
    pub title_applied: bool,
}

/// Persist a notes result: the notes (replacing earlier ones), then its title
/// unless the user named the session.
pub(crate) async fn store_notes_result(
    store: &TranscriptStore,
    session_id: Uuid,
    language: &str,
    llm: &LlmChoice,
    result: &Value,
) -> Result<StoredNotes, AssistantError> {
    let markdown = result["markdown"].as_str().map(str::trim).unwrap_or("");
    if markdown.is_empty() {
        return Err(AssistantError::new(
            "empty_result",
            "The AI assistant returned empty notes",
        ));
    }
    // Attachment reports are shown in the UI; a path must never reach it.
    let attachments = result["attachments"]
        .as_array()
        .map(|reports| {
            reports
                .iter()
                .filter_map(Value::as_object)
                .map(|report| {
                    let mut report = report.clone();
                    report.remove("path");
                    Value::Object(report)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let draft = SessionNotesDraft {
        session_id,
        markdown: markdown.to_string(),
        language: language.to_string(),
        provider: text_field(result, "provider")
            .unwrap_or(&llm.group)
            .to_string(),
        model: text_field(result, "model").unwrap_or(&llm.model).to_string(),
        attachments: Value::Array(attachments),
        usage: Some(result["usage"].clone())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({})),
        prompt_version: text_field(result, "prompt_version")
            .unwrap_or("")
            .to_string(),
        source_chars: result["source_chars"].as_i64().unwrap_or(0),
    };
    let notes = store
        .save_notes(&draft)
        .await
        .map_err(AssistantError::storage)?;
    let title = clean_title(result["title"].as_str().unwrap_or(""));
    let title_applied = if title.is_empty() {
        false
    } else {
        store
            .set_ai_title(session_id, &title)
            .await
            .map_err(AssistantError::storage)?
    };
    Ok(StoredNotes {
        notes,
        title,
        title_applied,
    })
}

/// `import_context_files` result (`ContextImportResult`): the sidecar's
/// context bounded to the session-context limit, terms, glossary pairs and
/// warnings (the sidecar's first, then the shell's).
pub(crate) fn context_import_result(result: &Value, shell_warnings: Vec<String>) -> Value {
    let strings = |key: &str| -> Vec<String> {
        result[key]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut warnings = strings("warnings");
    let raw_context = result["context"].as_str().unwrap_or("").trim();
    let context = truncate_chars(raw_context, SESSION_CONTEXT_MAX_CHARS).trim_end();
    if context.len() < raw_context.len() {
        warnings.push(format!(
            "The imported context was shortened to {SESSION_CONTEXT_MAX_CHARS} characters"
        ));
    }
    let glossary: Vec<Value> = result["glossary"]
        .as_array()
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(|pair| {
                    let source = text_field(pair, "source")?;
                    Some(json!({
                        "source": source,
                        "target": text_field(pair, "target").unwrap_or(""),
                    }))
                })
                .collect()
        })
        .unwrap_or_default();
    warnings.extend(shell_warnings);
    json!({
        "context": context,
        "terms": strings("terms"),
        "glossary": glossary,
        "warnings": warnings,
    })
}

fn empty_context_import() -> Value {
    json!({"context": "", "terms": [], "glossary": [], "warnings": []})
}

fn parse_session_id(session_id: &str) -> Result<Uuid, String> {
    session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())
}

fn emit_job(app: &AppHandle, state: &RuntimeState, view: &AssistantJobView) {
    if let Ok(payload) = serde_json::to_value(view) {
        let _ = state.emit_event(app, view.session_id, UiEventKind::AssistantUpdate, payload);
    }
}

/// Emit the first `running` update and log the start (counts only).
fn announce(app: &AppHandle, state: &RuntimeState, view: &AssistantJobView, detail: &str) {
    emit_job(app, state, view);
    state.log_desktop_event(&format!(
        "assistant_job job={} task={} state=running {detail}",
        view.job_id,
        view.task.as_str()
    ));
}

/// Launch the sidecar if needed, send the request and follow its events.
/// The request is sent under the sidecar lifecycle lock, so no restart can
/// fall between the launch and the send; from then on the job counts as in
/// flight and restarts wait for it.
async fn dispatch(
    app: &AppHandle,
    job_id: Uuid,
    task: AssistantTask,
    payload: Value,
    cancel: watch::Receiver<bool>,
) -> (JobProgress, Result<Value, AssistantError>) {
    let state = app.state::<RuntimeState>();
    dispatch_request(&state, Some(app), job_id, task, payload, cancel, |view| {
        emit_job(app, &state, view)
    })
    .await
}

/// [`dispatch`] without an app handle: `on_update` receives each running
/// view to emit.
pub(crate) async fn dispatch_request(
    state: &RuntimeState,
    app: Option<&AppHandle>,
    job_id: Uuid,
    task: AssistantTask,
    payload: Value,
    mut cancel: watch::Receiver<bool>,
    mut on_update: impl FnMut(&AssistantJobView),
) -> (JobProgress, Result<Value, AssistantError>) {
    let mut events = {
        let _lifecycle = state.sidecar_lifecycle.lock().await;
        if state.shutting_down.load(Ordering::Acquire) {
            // Never relaunch the sidecar while the application closes.
            return (
                JobProgress::default(),
                Err(AssistantError::new("shutting_down", "EchoLingo is closing")),
            );
        }
        if let Err(message) = state.start_sidecar_locked(app, SidecarUser::Background).await {
            return (
                JobProgress::default(),
                Err(AssistantError::new("sidecar_unavailable", message)),
            );
        }
        let events = state.supervisor().subscribe();
        if *cancel.borrow() {
            return (JobProgress::default(), Err(AssistantError::cancelled()));
        }
        // Consent may have been withdrawn after the command checked it but
        // before the job was registered (withdrawal cancels registered jobs
        // only); never send a transcript or file without it.
        let consented = state
            .assistant_preferences()
            .is_ok_and(|prefs| prefs.transcript_upload_allowed);
        if state.assistant.uploads(job_id) && !consented {
            return (
                JobProgress::default(),
                Err(AssistantError::new("privacy_policy_denied", CONSENT_REQUIRED)),
            );
        }
        if let Err(error) = state
            .supervisor()
            .send_command(SidecarCommand::AssistantRequest {
                request_id: job_id,
                task: task.as_str().into(),
                payload,
            })
            .await
        {
            return (
                JobProgress::default(),
                Err(AssistantError::new("sidecar_unavailable", error.to_string())),
            );
        }
        state.assistant.mark_dispatched(job_id);
        events
    };
    let deadline = Instant::now() + task.timeout();
    let (progress, outcome) =
        consume_assistant_events(&mut events, job_id, deadline, &mut cancel, |progress| {
            if let Some(view) = state.assistant.update(job_id, progress) {
                on_update(&view);
            }
        })
        .await;
    if progress.lagged {
        state.log_desktop_event(&format!(
            "assistant_job job={job_id} lagged=true (streamed text may be incomplete)"
        ));
    }
    if outcome
        .as_ref()
        .is_err_and(|error| matches!(error.code.as_str(), "cancelled" | "timeout"))
    {
        // Stop the model call; a late result for this id is ignored.
        let _ = state
            .supervisor()
            .send_command(SidecarCommand::AssistantCancel { request_id: job_id })
            .await;
    }
    (progress, outcome)
}

/// Remove the job, emit its final update, log it and run a restart the job
/// deferred. `log_message` adds the error message to the log; only tasks
/// that carry no transcript (the probe) may set it.
async fn finish_job(
    app: &AppHandle,
    job_id: Uuid,
    outcome: &Result<Value, AssistantError>,
    text: Option<String>,
    log_message: bool,
) {
    let state = app.state::<RuntimeState>();
    let (job_state, result, error) = match outcome {
        Ok(result) => (JobState::Completed, Some(result.clone()), None),
        Err(error) if error.code == "cancelled" => (JobState::Cancelled, None, None),
        Err(error) => (JobState::Failed, None, Some(error.clone())),
    };
    if let Some((view, elapsed)) = state
        .assistant
        .finish(job_id, job_state, text, result, error)
    {
        emit_job(app, &state, &view);
        let mut line = format!(
            "assistant_job job={job_id} task={} state={} duration_ms={}",
            view.task.as_str(),
            view.state.as_str(),
            elapsed.as_millis()
        );
        if let Some(error) = &view.error {
            line.push_str(&format!(" code={}", error.code));
            if log_message {
                line.push_str(&format!(" message={}", error.message));
            }
        }
        state.log_desktop_event(&line);
    }
    // Run a restart this job deferred without delaying the caller.
    if state.sidecar_environment_stale.load(Ordering::Acquire) {
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            app.state::<RuntimeState>()
                .restart_stale_sidecar_if_quiet()
                .await;
        });
    }
}

/// Ask the assistant for a title and apply it unless the user named the
/// session. Used by `generate_session_title` and the auto title.
async fn request_title(
    app: &AppHandle,
    session_id: Uuid,
    detail: Option<SessionDetail>,
) -> Result<TitleOutcome, String> {
    let state = app.state::<RuntimeState>();
    let llm = gate(&readiness(&state)?, true)?;
    let detail = match detail {
        Some(detail) => detail,
        None => state
            .store()?
            .detail(session_id)
            .await
            .map_err(|error| error.to_string())?,
    };
    let transcript = transcript_lines(&detail);
    if transcript.is_empty() {
        return Err(NO_TRANSCRIPT.into());
    }
    let units = transcript.len();
    let glossary = state.preferences()?.session.glossary;
    let payload = json!({
        "llm": llm.payload(),
        "consent": {"transcript_upload_allowed": true},
        "output_language": detail.session.target_language,
        "session": session_payload(&detail),
        "glossary": glossary,
        "transcript": transcript,
    });
    let (job_id, cancel, view) =
        state
            .assistant
            .register(AssistantTask::Title, Some(session_id), true)?;
    let _guard = JobGuard {
        registry: &state.assistant,
        job_id,
    };
    announce(app, &state, &view, &format!("units={units}"));
    let (_, outcome) = dispatch(app, job_id, AssistantTask::Title, payload, cancel).await;
    let outcome = match outcome {
        Ok(result) => match state.store() {
            Ok(store) => apply_title_result(store, session_id, &result).await,
            Err(error) => Err(AssistantError::storage(error)),
        },
        Err(error) => Err(error),
    };
    let reported = outcome
        .as_ref()
        .map(|title| json!(title))
        .map_err(Clone::clone);
    finish_job(app, job_id, &reported, None, false).await;
    let title = outcome.map_err(|error| error.message)?;
    if title.applied {
        emit_history_changed(app, &state, session_id, "title");
    }
    Ok(title)
}

/// Name a finished session in the background when the user enabled
/// auto titles, consented and configured a provider, the transcript has at
/// least [`AUTO_TITLE_MIN_WORDS`] words and the user has not named it.
pub(crate) fn spawn_auto_title(app: AppHandle, session_id: Uuid) {
    tauri::async_runtime::spawn(async move {
        let state = app.state::<RuntimeState>();
        if state.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let Ok(readiness) = readiness(&state) else {
            return;
        };
        if !readiness.prefs.auto_title || gate(&readiness, true).is_err() {
            return;
        }
        let Ok(store) = state.store() else {
            return;
        };
        let Ok(detail) = store.detail(session_id).await else {
            return;
        };
        if !auto_title_wanted(&detail) {
            return;
        }
        // Failures are logged by `finish_job`; the session keeps its title.
        let _ = request_title(&app, session_id, Some(detail)).await;
    });
}

/// Show the native picker for course files on a blocking thread; `None`
/// when the user cancelled.
async fn pick_files(app: &AppHandle, title: &'static str) -> Result<Option<Vec<PathBuf>>, String> {
    let app = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title(title)
            .add_filter("Documents", ATTACHMENT_EXTENSIONS)
            .blocking_pick_files()
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(picked.map(|files| {
        files
            .into_iter()
            .filter_map(|file| file.into_path().ok())
            .collect()
    }))
}

/// Async so the keychain lookup never runs on the main thread.
#[tauri::command]
pub async fn assistant_status(state: State<'_, RuntimeState>) -> Result<AssistantStatus, String> {
    Ok(AssistantStatus::from(&readiness(&state)?))
}

/// Send one fixed prompt (no transcript) to the configured provider.
#[tauri::command]
pub async fn assistant_probe(
    app: AppHandle,
    state: State<'_, RuntimeState>,
) -> Result<Value, String> {
    let llm = gate(&readiness(&state)?, false)?;
    let (job_id, cancel, view) = state.assistant.register(AssistantTask::Probe, None, false)?;
    let _guard = JobGuard {
        registry: &state.assistant,
        job_id,
    };
    announce(&app, &state, &view, &format!("group={}", llm.group));
    let payload = json!({"llm": llm.payload()});
    let (_, outcome) = dispatch(&app, job_id, AssistantTask::Probe, payload, cancel).await;
    let outcome = outcome.map(|result| {
        json!({
            "provider": text_field(&result, "provider").unwrap_or(&llm.group),
            "model": text_field(&result, "model").unwrap_or(&llm.model),
            "latency_ms": result["latency_ms"],
        })
    });
    finish_job(&app, job_id, &outcome, None, true).await;
    outcome.map_err(|error| error.message)
}

/// `get_session_notes` result (`SessionNotesState`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SessionNotesState {
    pub notes: Option<SessionNotes>,
    /// The running notes job of this session, with its text so far.
    pub job: Option<AssistantJobView>,
}

#[tauri::command]
pub async fn get_session_notes(
    state: State<'_, RuntimeState>,
    session_id: String,
) -> Result<SessionNotesState, String> {
    let session_id = parse_session_id(&session_id)?;
    let notes = state
        .store()?
        .notes(session_id)
        .await
        .map_err(|error| error.to_string())?;
    Ok(SessionNotesState {
        notes,
        job: state.assistant.running(session_id, AssistantTask::Notes),
    })
}

/// Let the user pick up to five course files; returns opaque ids (an empty
/// list when the picker was cancelled).
#[tauri::command]
pub async fn pick_note_attachments(
    app: AppHandle,
    state: State<'_, RuntimeState>,
) -> Result<Vec<NoteAttachment>, String> {
    let Some(paths) = pick_files(&app, "Attach course materials").await? else {
        return Ok(Vec::new());
    };
    let files = inspect_attachments(&paths)?;
    Ok(state.assistant.remember_attachments(files))
}

/// Start writing AI notes for a session; progress arrives as
/// `assistant_update` events and the notes are saved when they complete.
#[tauri::command]
pub async fn create_session_notes(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    session_id: String,
    attachment_ids: Vec<String>,
) -> Result<Value, String> {
    let session_id = parse_session_id(&session_id)?;
    let llm = gate(&readiness(&state)?, true)?;
    let attachments = state.assistant.resolve_attachments(&attachment_ids)?;
    let detail = state
        .store()?
        .detail(session_id)
        .await
        .map_err(|error| error.to_string())?;
    let transcript = transcript_lines(&detail);
    if transcript.is_empty() {
        return Err(NO_TRANSCRIPT.into());
    }
    let language = detail.session.target_language.clone();
    let counts = format!(
        "units={} attachments={}",
        transcript.len(),
        attachments.len()
    );
    let glossary = state.preferences()?.session.glossary;
    let payload = json!({
        "llm": llm.payload(),
        "consent": {"transcript_upload_allowed": true},
        "output_language": language,
        "session": session_payload(&detail),
        "glossary": glossary,
        "transcript": transcript,
        "attachments": attachments
            .iter()
            .map(|(id, file)| json!({"id": id, "path": file.path, "name": file.name}))
            .collect::<Vec<_>>(),
    });
    let (job_id, cancel, view) =
        state
            .assistant
            .register(AssistantTask::Notes, Some(session_id), true)?;
    announce(&app, &state, &view, &counts);
    let job_app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = job_app.state::<RuntimeState>();
        let _guard = JobGuard {
            registry: &state.assistant,
            job_id,
        };
        let (progress, outcome) =
            dispatch(&job_app, job_id, AssistantTask::Notes, payload, cancel).await;
        let stored = match outcome {
            Ok(result) => match state.store() {
                Ok(store) => store_notes_result(store, session_id, &language, &llm, &result).await,
                Err(error) => Err(AssistantError::storage(error)),
            },
            Err(error) => Err(error),
        };
        match stored {
            Ok(stored) => {
                let result = json!({
                    "notes": stored.notes,
                    "title": stored.title,
                    "title_applied": stored.title_applied,
                });
                finish_job(
                    &job_app,
                    job_id,
                    &Ok(result),
                    Some(stored.notes.markdown.clone()),
                    false,
                )
                .await;
                emit_history_changed(&job_app, &state, session_id, "notes");
            }
            Err(error) => {
                finish_job(&job_app, job_id, &Err(error), Some(progress.text), false).await;
            }
        }
    });
    Ok(json!({"job_id": job_id}))
}

/// Stop a running assistant job (notes, title or context import). A job
/// that already finished is not an error.
#[tauri::command]
pub fn cancel_session_notes(state: State<'_, RuntimeState>, job_id: String) -> Result<(), String> {
    let job_id: Uuid = job_id.parse().map_err(|_| "invalid job id".to_string())?;
    state.assistant.cancel(job_id);
    Ok(())
}

#[tauri::command]
pub async fn generate_session_title(
    app: AppHandle,
    session_id: String,
) -> Result<TitleOutcome, String> {
    let session_id = parse_session_id(&session_id)?;
    request_title(&app, session_id, None).await
}

/// Pick course files and turn them into lecture context, terms and glossary
/// pairs. Files are read locally; the model sees them only when `use_llm`
/// is set and the assistant is configured and allowed to read files.
#[tauri::command]
pub async fn import_context_files(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    use_llm: bool,
) -> Result<Value, String> {
    let Some(paths) = pick_files(&app, "Import lecture context").await? else {
        return Ok(empty_context_import());
    };
    let files = inspect_attachments(&paths)?;
    if files.is_empty() {
        return Ok(empty_context_import());
    }
    let mut warnings = Vec::new();
    let llm = if use_llm {
        match gate(&readiness(&state)?, true) {
            Ok(llm) => Some(llm),
            Err(reason) => {
                warnings.push(format!("AI term extraction was skipped. {reason}"));
                None
            }
        }
    } else {
        None
    };
    let defaults = state
        .session_defaults
        .lock()
        .map_err(|_| "session defaults lock poisoned".to_string())?
        .clone();
    let mut payload = json!({
        "attachments": files
            .iter()
            .map(|file| json!({"id": Uuid::new_v4(), "path": file.path, "name": file.name}))
            .collect::<Vec<_>>(),
        "source_language": defaults.source_language,
        "target_language": defaults.target_language,
        "output_language": defaults.target_language,
        "use_llm": llm.is_some(),
    });
    if let Some(llm) = &llm {
        payload["llm"] = llm.payload();
        payload["consent"] = json!({"transcript_upload_allowed": true});
    }
    let (job_id, cancel, view) =
        state
            .assistant
            .register(AssistantTask::Context, None, llm.is_some())?;
    let _guard = JobGuard {
        registry: &state.assistant,
        job_id,
    };
    announce(
        &app,
        &state,
        &view,
        &format!("attachments={} use_llm={}", files.len(), llm.is_some()),
    );
    let (_, outcome) = dispatch(&app, job_id, AssistantTask::Context, payload, cancel).await;
    let outcome = outcome.map(|result| context_import_result(&result, warnings));
    finish_job(&app, job_id, &outcome, None, false).await;
    outcome.map_err(|error| error.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CredentialFieldStatus;
    use transcript_store::{SegmentDraft, SessionDraft, SessionRecord};

    fn delta(request_id: Uuid, seq: u64, text: &str) -> SidecarEvent {
        SidecarEvent::AssistantDelta {
            request_id,
            seq,
            text: text.into(),
        }
    }

    fn result(request_id: Uuid, ok: bool, result: Value) -> SidecarEvent {
        SidecarEvent::AssistantResult {
            request_id,
            ok,
            result,
        }
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(600)
    }

    #[tokio::test(start_paused = true)]
    async fn prepend_placement_puts_the_header_of_long_notes_on_top() {
        let (sender, mut events) = broadcast::channel(64);
        let (_cancel, mut cancelled) = watch::channel(false);
        let id = Uuid::new_v4();
        sender.send(delta(id, 1, "## Part 1 (00:00–14:00)\n")).unwrap();
        sender.send(delta(id, 2, "## Part 2 (14:00–28:00)\n")).unwrap();
        sender
            .send(SidecarEvent::AssistantProgress {
                request_id: id,
                stage: "finishing".into(),
                detail: json!({"placement": "prepend"}),
            })
            .unwrap();
        sender.send(delta(id, 3, "# Title\n\nOverview.\n\n")).unwrap();
        sender.send(delta(id, 4, "## Key terms\n")).unwrap();
        sender.send(result(id, true, json!({"markdown": "done"}))).unwrap();
        let (progress, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {}).await;
        assert!(outcome.is_ok());
        assert_eq!(
            progress.text,
            "# Title\n\nOverview.\n\n## Part 1 (00:00–14:00)\n## Part 2 (14:00–28:00)\n## Key terms\n"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn consumer_follows_one_request_in_seq_order() {
        let (sender, mut events) = broadcast::channel(64);
        let (_cancel, mut cancelled) = watch::channel(false);
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        sender.send(delta(id, 1, "# Notes\n")).unwrap();
        sender.send(delta(other, 1, "not ours")).unwrap();
        sender.send(delta(id, 2, "## Part 1")).unwrap();
        sender.send(delta(id, 2, "## Part 1")).unwrap();
        sender
            .send(SidecarEvent::AssistantProgress {
                request_id: id,
                stage: "section".into(),
                detail: json!({"index": 1, "total": 3, "start_ms": 0, "end_ms": 840000}),
            })
            .unwrap();
        sender.send(result(other, false, json!({"code": "x"}))).unwrap();
        sender
            .send(result(id, true, json!({"markdown": "# Notes\n## Part 1"})))
            .unwrap();
        let mut emitted = Vec::new();
        let (progress, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |p| {
                emitted.push((p.stage.clone(), p.text.clone()))
            })
            .await;
        assert_eq!(outcome.unwrap()["markdown"], "# Notes\n## Part 1");
        assert_eq!(progress.text, "# Notes\n## Part 1");
        assert_eq!(progress.stage.as_deref(), Some("section"));
        assert_eq!(progress.progress.as_ref().unwrap()["total"], 3);
        assert!(!progress.lagged);
        // The first delta is shown at once; the second waits for the
        // interval but rides along with the (immediate) progress update.
        assert_eq!(
            emitted,
            vec![
                (None, "# Notes\n".to_string()),
                (Some("section".to_string()), "# Notes\n## Part 1".to_string()),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn text_updates_are_throttled_and_the_tail_is_flushed() {
        let (sender, mut events) = broadcast::channel(64);
        let (_cancel, mut cancelled) = watch::channel(false);
        let id = Uuid::new_v4();
        let producer = tokio::spawn(async move {
            sender.send(delta(id, 1, "a")).unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
            sender.send(delta(id, 2, "b")).unwrap();
            sender.send(delta(id, 3, "c")).unwrap();
            tokio::time::sleep(Duration::from_millis(400)).await;
            sender.send(result(id, true, json!({}))).unwrap();
            sender
        });
        let started = Instant::now();
        let mut emitted = Vec::new();
        let (progress, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |p| {
                emitted.push((started.elapsed(), p.text.clone()))
            })
            .await;
        producer.await.unwrap();
        assert!(outcome.is_ok());
        assert_eq!(progress.text, "abc");
        assert_eq!(emitted.len(), 2, "{emitted:?}");
        assert_eq!(emitted[0].1, "a");
        assert_eq!(emitted[1].1, "abc");
        assert!(emitted[1].0 >= TEXT_UI_INTERVAL && emitted[1].0 < Duration::from_millis(400));
    }

    #[tokio::test(start_paused = true)]
    async fn consumer_reports_failures_cancellation_timeouts_and_disconnects() {
        let id = Uuid::new_v4();
        let (sender, mut events) = broadcast::channel(64);
        let (_cancel, mut cancelled) = watch::channel(false);
        sender
            .send(result(
                id,
                false,
                json!({"code": "authentication_failed", "message": "Check the key"}),
            ))
            .unwrap();
        let (_, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert_eq!(
            outcome.unwrap_err(),
            AssistantError::new("authentication_failed", "Check the key")
        );
        sender.send(result(id, false, json!(null))).unwrap();
        let (_, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert_eq!(outcome.unwrap_err().code, "assistant_failed");

        // Cancellation while waiting.
        let (cancel, mut cancelled) = watch::channel(false);
        let canceller = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            cancel.send_replace(true);
            cancel
        });
        sender.send(delta(id, 1, "partial")).unwrap();
        let (progress, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert_eq!(outcome.unwrap_err().code, "cancelled");
        assert_eq!(progress.text, "partial");
        drop(canceller.await.unwrap());

        // A dropped cancel sender never cancels (and never spins).
        let (cancel, mut cancelled) = watch::channel(false);
        drop(cancel);
        let started = Instant::now();
        let (_, outcome) = consume_assistant_events(
            &mut events,
            id,
            Instant::now() + AssistantTask::Title.timeout(),
            &mut cancelled,
            |_| {},
        )
        .await;
        assert_eq!(outcome.unwrap_err().code, "timeout");
        assert_eq!(started.elapsed(), Duration::from_secs(90));

        let (_cancel, mut cancelled) = watch::channel(false);
        sender
            .send(SidecarEvent::Error {
                code: "session_error".into(),
                message: "unrelated".into(),
                recoverable: true,
            })
            .unwrap();
        sender
            .send(SidecarEvent::Error {
                code: "sidecar_disconnected".into(),
                message: "gone".into(),
                recoverable: true,
            })
            .unwrap();
        let (_, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert_eq!(outcome.unwrap_err().code, "sidecar_disconnected");
        drop(sender);
        let (_, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert_eq!(outcome.unwrap_err().code, "sidecar_disconnected");
    }

    #[tokio::test(start_paused = true)]
    async fn a_lagging_receiver_is_flagged_and_still_sees_the_result() {
        let (sender, mut events) = broadcast::channel(2);
        let (_cancel, mut cancelled) = watch::channel(false);
        let id = Uuid::new_v4();
        for seq in 1..=4 {
            sender.send(delta(id, seq, "x")).unwrap();
        }
        sender.send(result(id, true, json!({"markdown": "xxxx"}))).unwrap();
        let (progress, outcome) =
            consume_assistant_events(&mut events, id, far_deadline(), &mut cancelled, |_| {})
                .await;
        assert!(progress.lagged);
        assert_eq!(outcome.unwrap()["markdown"], "xxxx");
    }

    #[test]
    fn registry_tracks_running_jobs_dispatch_and_cancellation() {
        let registry = AssistantRegistry::default();
        assert!(!registry.any_running() && !registry.any_in_flight());
        let session = Uuid::new_v4();
        let (notes_id, notes_cancel, view) = registry
            .register(AssistantTask::Notes, Some(session), true)
            .unwrap();
        assert_eq!(view.state, JobState::Running);
        assert_eq!(serde_json::to_value(&view).unwrap()["task"], "notes");
        assert!(registry.any_running() && !registry.any_in_flight());
        assert!(registry
            .register(AssistantTask::Notes, Some(session), true)
            .is_err());
        // A title job for the same session and a context job may run too.
        let (title_id, title_cancel, _) = registry
            .register(AssistantTask::Title, Some(session), true)
            .unwrap();
        let (probe_id, probe_cancel, _) =
            registry.register(AssistantTask::Probe, None, false).unwrap();
        registry.mark_dispatched(notes_id);
        assert!(registry.any_in_flight());

        let progress = JobProgress {
            stage: Some("writing".into()),
            text: "## A".into(),
            ..JobProgress::default()
        };
        let updated = registry.update(notes_id, &progress).unwrap();
        assert_eq!(updated.text, "## A");
        assert_eq!(
            registry.running(session, AssistantTask::Notes).unwrap().stage.as_deref(),
            Some("writing")
        );
        assert!(registry.running(Uuid::new_v4(), AssistantTask::Notes).is_none());

        registry.cancel_uploading_jobs();
        assert!(*notes_cancel.borrow() && *title_cancel.borrow());
        assert!(!*probe_cancel.borrow());
        assert!(registry.cancel(probe_id));

        let (view, _) = registry
            .finish(
                notes_id,
                JobState::Completed,
                Some("# Final".into()),
                Some(json!({"ok": true})),
                None,
            )
            .unwrap();
        assert_eq!(view.state, JobState::Completed);
        assert_eq!(view.text, "# Final");
        assert!(!registry.any_in_flight());
        assert!(!registry.cancel(notes_id));
        assert!(registry.finish(notes_id, JobState::Failed, None, None, None).is_none());
        registry.cancel_session_jobs(session);
        registry.finish(title_id, JobState::Cancelled, None, None, None);
        registry.finish(probe_id, JobState::Cancelled, None, None, None);
        assert!(!registry.any_running());
    }

    #[test]
    fn a_dropped_job_never_stays_registered() {
        let registry = AssistantRegistry::default();
        let (job_id, _cancel, _) = registry.register(AssistantTask::Probe, None, false).unwrap();
        registry.mark_dispatched(job_id);
        {
            let _guard = JobGuard {
                registry: &registry,
                job_id,
            };
        }
        assert!(!registry.any_running() && !registry.any_in_flight());
    }

    #[test]
    fn attachments_are_validated_and_resolved_by_opaque_id() {
        let directory = tempfile::tempdir().unwrap();
        let slides = directory.path().join("Slides.PDF");
        std::fs::write(&slides, b"%PDF-1.7").unwrap();
        let notes = directory.path().join("notes.md");
        std::fs::write(&notes, b"# Week 3").unwrap();
        let huge = directory.path().join("huge.pdf");
        std::fs::File::create(&huge)
            .unwrap()
            .set_len(MAX_ATTACHMENT_BYTES + 1)
            .unwrap();
        let script = directory.path().join("run.sh");
        std::fs::write(&script, b"echo").unwrap();

        let picked = inspect_attachment(&slides).unwrap();
        assert_eq!(picked.extension, "pdf");
        assert_eq!(picked.size_bytes, 8);
        assert_eq!(picked.name, "Slides.PDF");
        let error = inspect_attachment(&huge).unwrap_err();
        assert!(error.contains("huge.pdf") && error.contains("25 MB"));
        assert!(!error.contains(directory.path().to_str().unwrap()));
        assert!(inspect_attachment(&script).unwrap_err().contains("not a supported"));
        assert!(inspect_attachment(&directory.path().join("missing.txt")).is_err());
        let folder = directory.path().join("folder.md");
        std::fs::create_dir(&folder).unwrap();
        assert!(inspect_attachment(&folder).unwrap_err().contains("not a file"));
        assert!(inspect_attachments(&vec![slides.clone(); MAX_ATTACHMENTS + 1]).is_err());

        let registry = AssistantRegistry::default();
        let attachments = registry.remember_attachments(
            inspect_attachments(&[slides.clone(), notes.clone()]).unwrap(),
        );
        assert_eq!(attachments.len(), 2);
        let serialized = serde_json::to_string(&attachments).unwrap();
        assert!(!serialized.contains(directory.path().to_str().unwrap()));
        let ids: Vec<String> = attachments.iter().map(|a| a.id.to_string()).collect();
        let resolved = registry
            .resolve_attachments(&[ids[1].clone(), ids[0].clone(), ids[1].clone()])
            .unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].1.name, "notes.md");
        assert_eq!(resolved[1].1.path, slides.to_str().unwrap());
        assert!(registry
            .resolve_attachments(&[Uuid::new_v4().to_string()])
            .unwrap_err()
            .contains("choose the files again"));
        assert!(registry.resolve_attachments(&["x".into()]).is_err());
        // A file that grew past the limit after it was picked is refused.
        std::fs::File::options()
            .write(true)
            .open(&notes)
            .unwrap()
            .set_len(MAX_ATTACHMENT_BYTES + 1)
            .unwrap();
        assert!(registry.resolve_attachments(&[ids[1].clone()]).is_err());
        let many: Vec<String> = (0..=MAX_ATTACHMENTS).map(|_| Uuid::new_v4().to_string()).collect();
        assert!(registry.resolve_attachments(&many).unwrap_err().contains("at most"));

        // Old picks are forgotten beyond the bound.
        for _ in 0..MAX_REMEMBERED_ATTACHMENTS {
            registry.remember_attachments(vec![picked.clone()]);
        }
        assert!(registry.resolve_attachments(&[ids[0].clone()]).is_err());
    }

    #[test]
    fn words_titles_and_context_results_are_normalised() {
        assert_eq!(word_count("Textures just pop out."), 4);
        assert_eq!(word_count("纹理 感知很重要"), 7);
        assert_eq!(word_count("Julesz的纹理基元 theory"), 7);
        assert_eq!(word_count("  "), 0);

        assert_eq!(clean_title("# \"Texture perception\"\nextra"), "Texture perception");
        assert_eq!(clean_title("《视觉》与纹理"), "《视觉》与纹理");
        assert_eq!(clean_title("《早期视觉》"), "早期视觉");
        assert_eq!(clean_title("  **Pre-attentive   vision**  "), "Pre-attentive vision");
        assert_eq!(clean_title("\n\n"), "");
        assert_eq!(clean_title(&"字".repeat(200)).chars().count(), TITLE_MAX_CHARS);

        let result = context_import_result(
            &json!({
                "context": "x".repeat(SESSION_CONTEXT_MAX_CHARS + 10),
                "terms": ["saccade", " ", 3, "texton"],
                "glossary": [{"source": "saccade", "target": "扫视"}, {"target": "orphan"}, {"source": "Julesz"}],
                "warnings": ["scan.pdf has no text layer"]
            }),
            vec!["AI term extraction was skipped.".into()],
        );
        assert_eq!(
            result["context"].as_str().unwrap().chars().count(),
            SESSION_CONTEXT_MAX_CHARS
        );
        assert_eq!(result["terms"], json!(["saccade", "texton"]));
        assert_eq!(
            result["glossary"],
            json!([{"source": "saccade", "target": "扫视"}, {"source": "Julesz", "target": ""}])
        );
        let warnings = result["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 3);
        assert_eq!(warnings[0], "scan.pdf has no text layer");
        assert_eq!(empty_context_import()["terms"], json!([]));
    }

    fn status(api_key: bool, settings: &[(&str, &str)]) -> CredentialGroupStatus {
        CredentialGroupStatus {
            group_id: String::new(),
            fields: vec![CredentialFieldStatus {
                key: "api_key".into(),
                available: api_key,
                source: if api_key { "keychain" } else { "none" }.into(),
            }],
            settings: settings
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        }
    }

    #[test]
    fn assistant_gating_requires_a_provider_credentials_and_consent() {
        let off = readiness_from(AssistantPreferences::default(), catalog(), |_| {
            panic!("no credential lookup without a provider")
        })
        .unwrap();
        assert!(gate(&off, false).unwrap_err().contains("Choose an AI assistant"));
        let status_off = AssistantStatus::from(&off);
        assert!(!status_off.configured && !status_off.key_available && status_off.auto_title);

        let dashscope = AssistantPreferences {
            provider_group: "dashscope".into(),
            ..AssistantPreferences::default()
        };
        let no_key = readiness_from(dashscope.clone(), catalog(), |group| {
            assert_eq!(group, "dashscope");
            Ok(status(false, &[]))
        })
        .unwrap();
        assert!(gate(&no_key, false).unwrap_err().contains("API key"));
        assert!(!AssistantStatus::from(&no_key).key_available);

        let no_consent =
            readiness_from(dashscope.clone(), catalog(), |_| Ok(status(true, &[]))).unwrap();
        let llm = gate(&no_consent, false).unwrap();
        assert_eq!(llm.model, "qwen-plus");
        assert_eq!(llm.payload(), json!({"group": "dashscope", "model": "qwen-plus"}));
        assert!(gate(&no_consent, true).unwrap_err().contains("Allow the AI assistant"));

        let consented = AssistantPreferences {
            transcript_upload_allowed: true,
            model: "qwen-max".into(),
            ..dashscope
        };
        let ready = readiness_from(consented, catalog(), |_| Ok(status(true, &[]))).unwrap();
        assert_eq!(gate(&ready, true).unwrap().model, "qwen-max");
        let summary = AssistantStatus::from(&ready);
        assert!(summary.configured && summary.consent && summary.key_available);
        assert_eq!(summary.display_name, "Qwen (Alibaba Model Studio)");

        // The custom endpoint needs its base URL; its model may come from the
        // group's chat-model setting.
        let custom = AssistantPreferences {
            provider_group: "custom_openai".into(),
            transcript_upload_allowed: true,
            ..AssistantPreferences::default()
        };
        let no_endpoint =
            readiness_from(custom.clone(), catalog(), |_| Ok(status(false, &[]))).unwrap();
        assert!(gate(&no_endpoint, true).is_err());
        let endpoint = readiness_from(custom, catalog(), |_| {
            Ok(status(
                false,
                &[("base_url", "http://127.0.0.1:8080/v1"), ("chat_model", "llama3")],
            ))
        })
        .unwrap();
        assert_eq!(gate(&endpoint, true).unwrap().model, "llama3");
    }

    fn session_record(title_source: &str) -> SessionRecord {
        SessionRecord {
            id: Uuid::nil().to_string(),
            title: "en → zh lecture".into(),
            status: "completed".into(),
            started_at: "2026-09-22T10:00:00.000Z".into(),
            ended_at: Some("2026-09-22T11:04:00.000Z".into()),
            source_language: "en".into(),
            target_language: "zh".into(),
            audio_source: "microphone".into(),
            audio_profile: "lecture".into(),
            inference_mode: "auto".into(),
            asr_backend: "qwen_local".into(),
            translation_backend: "hymt_local".into(),
            route_reason: String::new(),
            privacy_json: "{}".into(),
            model_config_json: "{}".into(),
            warnings_json: "[]".into(),
            created_at: String::new(),
            updated_at: String::new(),
            title_source: title_source.into(),
            context: "Topic: early vision".into(),
        }
    }

    fn segment_record(ordinal: i64, text: &str) -> transcript_store::SegmentRecord {
        transcript_store::SegmentRecord {
            id: Uuid::new_v4().to_string(),
            session_id: Uuid::nil().to_string(),
            ordinal,
            start_ms: ordinal as f64 * 1_000.0,
            end_ms: ordinal as f64 * 1_000.0 + 900.0,
            source_text: text.into(),
            translated_text: String::new(),
            source_final: false,
            target_final: false,
            speaker_id: None,
            asr_confidence: None,
            source_revision: ordinal,
            target_revision: 0,
            timestamp_quality: "interpolated".into(),
            word_timings_json: "[]".into(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn transcript_payloads_and_the_auto_title_threshold() {
        let sentence = "Textures just pop out when the statistics differ enough.";
        let mut detail = SessionDetail {
            session: session_record("default"),
            segments: vec![
                segment_record(1, sentence),
                segment_record(2, "   "),
                segment_record(3, sentence),
            ],
        };
        let lines = transcript_lines(&detail);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], json!({"start_ms": 3000.0, "end_ms": 3900.0, "text": sentence}));
        let session = session_payload(&detail);
        assert_eq!(session["duration_ms"], 3_900);
        assert_eq!(session["context"], "Topic: early vision");
        assert_eq!(session["target_language"], "zh");
        // 18 words: below the threshold.
        assert!(!auto_title_wanted(&detail));
        detail.segments.push(segment_record(4, &format!("{sentence} {sentence}")));
        assert!(auto_title_wanted(&detail));
        detail.session.title_source = TITLE_SOURCE_USER.into();
        assert!(!auto_title_wanted(&detail));
        // Without segments the wall-clock length is used.
        detail.segments.clear();
        assert_eq!(session_payload(&detail)["duration_ms"], 64 * 60_000);
    }

    async fn store_with_session(title: &str) -> (tempfile::TempDir, TranscriptStore, Uuid) {
        let directory = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(directory.path().join("history.sqlite"))
            .await
            .unwrap();
        let id = Uuid::new_v4();
        store
            .create_session(&SessionDraft {
                id,
                title: "en → zh lecture".into(),
                source_language: "en".into(),
                target_language: "zh".into(),
                audio_source: "microphone".into(),
                audio_profile: "lecture".into(),
                inference_mode: "auto".into(),
                asr_backend: "mock".into(),
                translation_backend: "mock".into(),
                route_reason: String::new(),
                privacy: json!({}),
                model_config: json!({}),
                context: "Topic: early vision".into(),
            })
            .await
            .unwrap();
        store
            .upsert_segment(&SegmentDraft {
                id: Uuid::new_v4(),
                session_id: id,
                ordinal: 1,
                start_ms: 0.0,
                end_ms: 900.0,
                source_text: "Textures just pop out.".into(),
                source_final: false,
                asr_confidence: None,
                source_revision: 1,
                timestamp_quality: "interpolated".into(),
                word_timings: json!([]),
            })
            .await
            .unwrap();
        if title != "en → zh lecture" {
            store.rename(id, title).await.unwrap();
        }
        (directory, store, id)
    }

    fn qwen() -> LlmChoice {
        LlmChoice {
            group: "dashscope".into(),
            model: "qwen-plus".into(),
            display_name: "Qwen".into(),
        }
    }

    #[tokio::test]
    async fn notes_results_are_saved_and_title_the_session_unless_the_user_named_it() {
        let (_directory, store, id) = store_with_session("en → zh lecture").await;
        let result = json!({
            "markdown": "# 纹理感知\n\n## 前注意视觉 (00:00–05:00)",
            "title": "\"纹理感知与前注意视觉\"",
            "model": "qwen-plus-2025",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "attachments": [{"id": "a", "name": "slides.pdf", "path": "/Users/x/slides.pdf", "pages": 3}],
            "prompt_version": "notes.v1",
            "source_chars": 22
        });
        let stored = store_notes_result(&store, id, "zh", &qwen(), &result).await.unwrap();
        assert!(stored.title_applied);
        assert_eq!(stored.title, "纹理感知与前注意视觉");
        assert_eq!(stored.notes.provider, "dashscope");
        assert_eq!(stored.notes.model, "qwen-plus-2025");
        assert_eq!(stored.notes.language, "zh");
        assert_eq!(stored.notes.attachments, json!([{"id": "a", "name": "slides.pdf", "pages": 3}]));
        assert_eq!(stored.notes.usage["completion_tokens"], 20);
        let session = store.session(id).await.unwrap();
        assert_eq!(session.title, "纹理感知与前注意视觉");
        assert_eq!(session.title_source, "ai");
        assert_eq!(store.search("纹理感知与前注意视觉", 5).await.unwrap().len(), 1);

        assert_eq!(
            store_notes_result(&store, id, "zh", &qwen(), &json!({"markdown": "  "}))
                .await
                .unwrap_err()
                .code,
            "empty_result"
        );
        // A missing session is a storage error, not a panic.
        assert_eq!(
            store_notes_result(&store, Uuid::new_v4(), "zh", &qwen(), &result)
                .await
                .unwrap_err()
                .code,
            "storage_error"
        );

        let (_directory, store, id) = store_with_session("Week 3: textures").await;
        let stored = store_notes_result(&store, id, "zh", &qwen(), &result).await.unwrap();
        assert!(!stored.title_applied);
        assert_eq!(stored.notes.model, "qwen-plus-2025");
        assert_eq!(store.session(id).await.unwrap().title, "Week 3: textures");
        let title = apply_title_result(&store, id, &json!({"title": "AI title"}))
            .await
            .unwrap();
        assert_eq!(title, TitleOutcome { title: "AI title".into(), applied: false });
        assert_eq!(
            apply_title_result(&store, id, &json!({"title": ""}))
                .await
                .unwrap_err()
                .code,
            "empty_result"
        );
    }

    /// A loopback stand-in for the Python sidecar: accepts the hello, answers
    /// `notes` requests with progress, two deltas and a result, leaves other
    /// tasks running, and reports every message it receives.
    async fn fake_sidecar() -> (String, tokio::sync::mpsc::UnboundedReceiver<Value>) {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let (seen, received) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let seen = seen.clone();
                tokio::spawn(async move {
                    let Ok(mut socket) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    while let Some(Ok(message)) = socket.next().await {
                        let Ok(value) = serde_json::from_str::<Value>(message.to_text().unwrap_or(""))
                        else {
                            continue;
                        };
                        let _ = seen.send(value.clone());
                        let payload = &value["payload"];
                        let replies = match value["type"].as_str() {
                            Some("hello") => vec![json!({
                                "type": "hello_accepted",
                                "payload": {"protocol_version": inference_ipc::PROTOCOL_VERSION}
                            })],
                            Some("assistant_request") if payload["task"] == "notes" => {
                                let id = payload["request_id"].clone();
                                vec![
                                    json!({"type": "assistant_progress", "payload": {"request_id": id, "stage": "writing"}}),
                                    json!({"type": "assistant_delta", "payload": {"request_id": id, "seq": 1, "text": "## A\n"}}),
                                    json!({"type": "assistant_delta", "payload": {"request_id": id, "seq": 2, "text": "## B"}}),
                                    json!({"type": "assistant_result", "payload": {"request_id": id, "ok": true, "result": {"markdown": "## A\n## B", "title": "T"}}}),
                                ]
                            }
                            _ => Vec::new(),
                        };
                        for reply in replies {
                            if socket.send(Message::Text(reply.to_string().into())).await.is_err() {
                                return;
                            }
                        }
                    }
                });
            }
        });
        (url, received)
    }

    async fn next_of_type(
        received: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
        kind: &str,
    ) -> Value {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let value = received.recv().await.expect("fake sidecar closed");
                if value["type"] == kind {
                    return value;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("no {kind} message reached the sidecar"))
    }

    /// Runtime state whose supervisor talks to the fake sidecar at `url`,
    /// with the model and local-runtime managers the full environment needs.
    fn state_with_sidecar(url: String, directory: &Path, consent: bool) -> RuntimeState {
        let state = RuntimeState::default();
        let mut launch = inference_ipc::SidecarLaunchConfig::development(crate::project_root());
        launch.configured_url = Some(url);
        launch.startup_timeout = Duration::from_secs(5);
        assert!(state
            .supervisor
            .set(inference_ipc::InferenceSupervisor::new(launch))
            .is_ok());
        let model_root = directory.join("models");
        assert!(state
            .models
            .set(runtime_manager::ModelManager::new(model_root.clone()))
            .is_ok());
        assert!(state
            .local_runtimes
            .set(runtime_manager::LocalRuntimeManager::new(crate::local_runtime_layout(
                model_root, directory, directory, 1, 2,
            )))
            .is_ok());
        state
            .runtime_preferences
            .lock()
            .unwrap()
            .assistant
            .transcript_upload_allowed = consent;
        state
    }

    #[tokio::test]
    async fn dispatch_talks_to_the_sidecar_and_restarts_wait_for_in_flight_jobs() {
        let (url, mut received) = fake_sidecar().await;
        let directory = tempfile::tempdir().unwrap();
        let state = state_with_sidecar(url, directory.path(), true);

        // A notes request: launched with the full environment, sent once,
        // followed to its result.
        let session = Uuid::new_v4();
        let (notes_id, cancel, _) = state
            .assistant
            .register(AssistantTask::Notes, Some(session), true)
            .unwrap();
        let mut updates = Vec::new();
        let (progress, outcome) = dispatch_request(
            &state,
            None,
            notes_id,
            AssistantTask::Notes,
            json!({"output_language": "zh", "transcript": [{"start_ms": 0, "end_ms": 900, "text": "hi"}]}),
            cancel,
            |view| updates.push(view.clone()),
        )
        .await;
        assert_eq!(outcome.unwrap()["markdown"], "## A
## B");
        assert_eq!(progress.text, "## A
## B");
        assert!(updates
            .iter()
            .all(|view| view.job_id == notes_id && view.state == JobState::Running));
        assert_eq!(updates.first().unwrap().stage.as_deref(), Some("writing"));
        next_of_type(&mut received, "hello").await;
        let request = next_of_type(&mut received, "assistant_request").await;
        assert_eq!(request["payload"]["request_id"], notes_id.to_string());
        assert_eq!(request["payload"]["task"], "notes");
        assert_eq!(request["payload"]["payload"]["output_language"], "zh");
        let environment = state.full_sidecar_environment().unwrap();
        assert!(environment.contains_key("ECHOLINGO_MODEL_ROOT"));
        assert!(environment.contains_key("ECHOLINGO_LOCAL_QWEN_URL"));
        state
            .assistant
            .finish(notes_id, JobState::Completed, None, None, None);

        // A title request that stays open: credential changes must not
        // restart the sidecar under it, and cancelling tells the sidecar.
        let (title_id, cancel, _) = state
            .assistant
            .register(AssistantTask::Title, Some(session), true)
            .unwrap();
        let stale = || state.sidecar_environment_stale.load(Ordering::Acquire);
        let driver = async {
            let request = next_of_type(&mut received, "assistant_request").await;
            assert_eq!(request["payload"]["request_id"], title_id.to_string());
            assert!(state.assistant.any_in_flight());
            state.invalidate_sidecar_environment().await;
            assert!(stale());
            state
                .ensure_sidecar(None, SidecarUser::Background)
                .await
                .unwrap();
            assert!(stale(), "the restart must wait for the running job");
            assert!(state.supervisor().is_connected().await);
            assert!(state.assistant.cancel(title_id));
            let cancel = next_of_type(&mut received, "assistant_cancel").await;
            assert_eq!(cancel["payload"]["request_id"], title_id.to_string());
        };
        let ((_, outcome), ()) = tokio::join!(
            dispatch_request(
                &state,
                None,
                title_id,
                AssistantTask::Title,
                json!({}),
                cancel,
                |_| {}
            ),
            driver
        );
        assert_eq!(outcome.unwrap_err().code, "cancelled");
        state
            .assistant
            .finish(title_id, JobState::Cancelled, None, None, None);
        // With the job gone the deferred restart happens.
        state.restart_stale_sidecar_if_quiet().await;
        assert!(!stale());
        next_of_type(&mut received, "shutdown").await;
        assert!(!state.supervisor().is_connected().await);
    }

    /// Every message the fake sidecar has reported so far.
    async fn drain(received: &mut tokio::sync::mpsc::UnboundedReceiver<Value>) -> Vec<String> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut kinds = Vec::new();
        while let Ok(message) = received.try_recv() {
            kinds.push(message["type"].as_str().unwrap_or("").to_string());
        }
        kinds
    }

    #[tokio::test]
    async fn jobs_never_restart_a_live_session_nor_upload_without_consent() {
        let (url, mut received) = fake_sidecar().await;
        let directory = tempfile::tempdir().unwrap();
        let state = state_with_sidecar(url, directory.path(), false);
        let stale = || state.sidecar_environment_stale.load(Ordering::Acquire);
        let transcript = json!({"transcript": [{"start_ms": 0, "end_ms": 900, "text": "hi"}]});

        // Consent is off: an uploading job is refused before anything is sent.
        let (job_id, cancel, _) = state
            .assistant
            .register(AssistantTask::Notes, Some(Uuid::new_v4()), true)
            .unwrap();
        let (_, outcome) = dispatch_request(
            &state,
            None,
            job_id,
            AssistantTask::Notes,
            transcript.clone(),
            cancel,
            |_| {},
        )
        .await;
        assert_eq!(outcome.unwrap_err().code, "privacy_policy_denied");
        state.assistant.finish(job_id, JobState::Failed, None, None, None);
        assert_eq!(drain(&mut received).await, vec!["hello"]);

        // A live session runs and credentials change under it.
        state
            .runtime_preferences
            .lock()
            .unwrap()
            .assistant
            .transcript_upload_allowed = true;
        state
            .core
            .lock()
            .unwrap()
            .start(app_core::StartSessionRequest::default())
            .unwrap();
        state.invalidate_sidecar_environment().await;
        assert!(stale());
        let (job_id, cancel, _) = state
            .assistant
            .register(AssistantTask::Notes, Some(Uuid::new_v4()), true)
            .unwrap();
        let (_, outcome) = dispatch_request(
            &state,
            None,
            job_id,
            AssistantTask::Notes,
            transcript,
            cancel,
            |_| {},
        )
        .await;
        assert_eq!(outcome.unwrap()["markdown"], "## A\n## B");
        state.assistant.finish(job_id, JobState::Completed, None, None, None);
        // The job reused the session's sidecar: no shutdown, no new hello.
        assert_eq!(drain(&mut received).await, vec!["assistant_request"]);
        assert!(stale(), "the restart waits for the live session to end");
        assert!(state.supervisor().is_connected().await);
        state.restart_stale_sidecar_if_quiet().await;
        assert!(stale() && state.supervisor().is_connected().await);
    }

    #[test]
    fn task_timeouts_match_the_contract() {
        assert_eq!(AssistantTask::Notes.timeout(), Duration::from_secs(900));
        assert_eq!(AssistantTask::Title.timeout(), Duration::from_secs(90));
        assert_eq!(AssistantTask::Context.timeout(), Duration::from_secs(180));
        assert_eq!(AssistantTask::Probe.timeout(), Duration::from_secs(30));
        let view = AssistantJobView {
            job_id: Uuid::nil(),
            session_id: None,
            task: AssistantTask::Context,
            state: JobState::Failed,
            stage: None,
            progress: None,
            text: String::new(),
            result: None,
            error: Some(AssistantError::new("timeout", "slow")),
        };
        let value = serde_json::to_value(view).unwrap();
        assert_eq!(value["state"], "failed");
        assert_eq!(value["error"], json!({"code": "timeout", "message": "slow"}));
        assert!(value["session_id"].is_null());
    }
}
