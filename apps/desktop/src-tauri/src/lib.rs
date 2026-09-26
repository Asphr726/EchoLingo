use app_core::{
    catalog, catalog_digest, catalog_value, validate_session_text, AppCore, AssistantPreferences,
    AudioSourceKind as AppAudioSourceKind, BackendHealth, CredentialGroup, LiveMetrics,
    ProviderCatalog, ProviderKind, ProviderSetting, RouteStatus, SegmentSummary, SessionPhase,
    SessionSnapshot, StartSessionRequest,
};
use audio_core::{
    audio_permission_status as native_permission_status,
    list_audio_devices as enumerate_audio_devices, preferred_capture_format,
    request_audio_permission as request_native_audio_permission, start_microphone,
    start_system_audio, AudioCaptureSession, AudioDevice, AudioPermissionStatus, AudioSourceEvent,
    CaptureFormat, PermissionKind, PermissionState,
};
use inference_ipc::{
    append_log_chunk, AudioFrameHeader, InferenceSupervisor, SidecarCommand, SidecarEvent,
    SidecarLaunchConfig, UiEventEnvelope, UiEventKind, PROTOCOL_VERSION,
    SIDECAR_LOG_ROTATE_BYTES,
};
#[cfg(target_os = "macos")]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use runtime_manager::gpu_pack::{GpuAccelerationStatus, GpuPackManager, GpuPackState};
use runtime_manager::{
    GpuRuntimeLayout, LocalRuntimeLayout, LocalRuntimeManager, ModelInstallState, ModelManager,
    ModelProgress, ModelStatus, QwenStreamingProfile, RuntimeCommand,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use transcript_store::{
    ExportFormat, SegmentDraft, SessionDetail, SessionDraft, SessionRecord, TranscriptStore,
};

mod assistant;
mod updater;

const UI_EVENT_CHANNEL: &str = "echolingo://ui-event";
const MODEL_PROGRESS_CHANNEL: &str = "echolingo://model-progress";
const DESKTOP_LOG_FILE: &str = "desktop.log";
/// Forced alignment runs after a session finished and reports no end; it
/// counts as running until no alignment update arrived for this long.
const ALIGNMENT_QUIET_PERIOD: std::time::Duration = std::time::Duration::from_secs(90);
const SIDECAR_LOG_FILE: &str = "sidecar.log";

/// Non-secret provider settings, keyed by credential group id and then by
/// `ProviderSetting.key` (for example `{"dashscope": {"region": "beijing"}}`).
type ProviderSettings = HashMap<String, HashMap<String, String>>;

/// OS secure-store access. Account names come from the catalog
/// (`CredentialField.keychain_account`) so adding a provider never touches
/// this code. Secret values are never logged, never returned to the webview
/// and never included in error messages.
#[derive(Default)]
struct CredentialStore {
    /// The secure-store service name, set in `setup` from the bundle
    /// identifier (see [`keychain_service`]). Until then every lookup fails.
    service: std::sync::OnceLock<String>,
}

/// The secure-store service name for a bundle identifier: the identifier
/// itself. The production app (`app.echolingo.desktop`) therefore keeps the
/// service every earlier version saved keys under, and a differently
/// identified build (the `.updatetest` smoke-test app) can never read them.
fn keychain_service(identifier: &str) -> String {
    identifier.to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CredentialFieldStatus {
    key: String,
    available: bool,
    /// `keychain` | `environment` | `none`
    source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CredentialGroupStatus {
    group_id: String,
    fields: Vec<CredentialFieldStatus>,
    /// Effective non-secret settings: the stored preference, else the process
    /// environment, else the catalog default (empty defaults are omitted).
    settings: HashMap<String, String>,
    /// `false` when the OS secure store (Keychain, Windows Credential
    /// Manager, Secret Service) could not be read; `fields` then only
    /// reflect environment variables.
    store_available: bool,
    store_error: Option<String>,
}

impl CredentialGroupStatus {
    fn with_store_failure(mut self, failure: Option<&str>) -> Self {
        if let Some(error) = failure {
            self.store_available = false;
            self.store_error = Some(error.to_string());
        }
        self
    }
}

impl CredentialStore {
    /// Use `service` from now on; a service that is already set stays.
    fn set_service(&self, service: String) {
        let _ = self.service.set(service);
    }

    fn entry(&self, account: &str) -> Result<keyring::Entry, String> {
        let service = self
            .service
            .get()
            .ok_or_else(|| "the secure store is not initialized".to_string())?;
        keyring::Entry::new(service, account).map_err(|error| error.to_string())
    }

    fn get(&self, account: &str) -> Result<Option<String>, String> {
        // Unit tests never read (or prompt for) the developer's secure store.
        if cfg!(test) {
            return Ok(None);
        }
        match self.entry(account)?.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }

    fn set(&self, account: &str, value: &str) -> Result<(), String> {
        if cfg!(test) {
            return Err("the secure store is not available in unit tests".into());
        }
        self.entry(account)?
            .set_password(value)
            .map_err(|error| error.to_string())
    }

    fn delete(&self, account: &str) -> Result<(), String> {
        if cfg!(test) {
            return Ok(());
        }
        match self.entry(account)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }

    /// A keychain lookup that records the first failure instead of aborting,
    /// so the composition helpers stay pure functions of the catalog.
    fn lookup_recording<'a>(
        &'a self,
        failure: &'a std::cell::RefCell<Option<String>>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |account| match self.get(account) {
            Ok(value) => value,
            Err(error) => {
                failure.borrow_mut().get_or_insert(error);
                None
            }
        }
    }

    /// Every group's status. A secure store that cannot be read does not
    /// fail the call: the statuses say so and report environment values.
    fn status(&self, providers: &ProviderSettings) -> Vec<CredentialGroupStatus> {
        let failure = std::cell::RefCell::new(None);
        let statuses: Vec<CredentialGroupStatus> = catalog()
            .credential_groups
            .iter()
            .map(|group| {
                group_status(
                    group,
                    self.lookup_recording(&failure),
                    process_env,
                    providers,
                )
            })
            .collect();
        let failure = failure.into_inner();
        statuses
            .into_iter()
            .map(|status| status.with_store_failure(failure.as_deref()))
            .collect()
    }

    /// One group's status; fails only for an unknown group.
    fn group_status(
        &self,
        group_id: &str,
        providers: &ProviderSettings,
    ) -> Result<CredentialGroupStatus, String> {
        let group = credential_group(group_id)?;
        let failure = std::cell::RefCell::new(None);
        let status = group_status(
            group,
            self.lookup_recording(&failure),
            process_env,
            providers,
        );
        Ok(status.with_store_failure(failure.into_inner().as_deref()))
    }

    /// The provider environment and, when the secure store could not be
    /// read, why (the values then come from the process environment only).
    fn sidecar_environment(
        &self,
        providers: &ProviderSettings,
    ) -> (HashMap<String, String>, Option<String>) {
        let failure = std::cell::RefCell::new(None);
        let values = compose_sidecar_environment(
            catalog(),
            self.lookup_recording(&failure),
            process_env,
            providers,
        );
        (values, failure.into_inner())
    }
}

fn credential_group(group_id: &str) -> Result<&'static CredentialGroup, String> {
    catalog()
        .group(group_id)
        .ok_or_else(|| format!("unknown credential group '{group_id}'"))
}

fn process_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// The value a setting takes for the sidecar and the UI: the stored preference
/// when it is valid, else the process environment, else the catalog default.
fn effective_setting(
    group_id: &str,
    setting: &ProviderSetting,
    env: &impl Fn(&str) -> Option<String>,
    providers: &ProviderSettings,
) -> Option<String> {
    providers
        .get(group_id)
        .and_then(|group| group.get(&setting.key))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && setting.accepts(value))
        .or_else(|| {
            setting
                .env_var()
                .and_then(env)
                .filter(|value| setting.accepts(value))
        })
        .or_else(|| (!setting.default.is_empty()).then(|| setting.default.clone()))
}

fn group_status(
    group: &CredentialGroup,
    lookup: impl Fn(&str) -> Option<String>,
    env: impl Fn(&str) -> Option<String>,
    providers: &ProviderSettings,
) -> CredentialGroupStatus {
    let fields = group
        .fields
        .iter()
        .map(|field| {
            let (available, source) = if lookup(&field.keychain_account)
                .is_some_and(|value| !value.is_empty())
            {
                (true, "keychain")
            } else if env(&field.env_var).is_some() {
                (true, "environment")
            } else {
                (false, "none")
            };
            CredentialFieldStatus {
                key: field.key.clone(),
                available,
                source: source.into(),
            }
        })
        .collect();
    let settings = group
        .settings
        .iter()
        .filter_map(|setting| {
            effective_setting(&group.id, setting, &env, providers)
                .map(|value| (setting.key.clone(), value))
        })
        .collect();
    CredentialGroupStatus {
        group_id: group.id.clone(),
        fields,
        settings,
        store_available: true,
        store_error: None,
    }
}

/// Every environment variable the sidecar needs for cloud providers: each
/// credential field from the keychain (`lookup` by keychain account) or else
/// the process environment (`env` by variable name), and each provider setting
/// with an `env_var` from the stored preference, the process environment or
/// the catalog default. Pure so it can be unit-tested without a keychain.
fn compose_sidecar_environment(
    catalog: &ProviderCatalog,
    lookup: impl Fn(&str) -> Option<String>,
    env: impl Fn(&str) -> Option<String>,
    providers: &ProviderSettings,
) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for group in &catalog.credential_groups {
        for field in &group.fields {
            let value = lookup(&field.keychain_account)
                .filter(|value| !value.is_empty())
                .or_else(|| env(&field.env_var));
            if let Some(value) = value {
                values.insert(field.env_var.clone(), value);
            }
        }
        for setting in &group.settings {
            let Some(name) = setting.env_var() else {
                continue;
            };
            if let Some(value) = effective_setting(&group.id, setting, &env, providers) {
                values.insert(name.to_string(), value);
            }
        }
    }
    values
}

/// Add the model root, the alignment spool and the local runtime
/// capabilities to the provider environment. Pure so it can be tested.
fn compose_full_sidecar_environment(
    mut values: HashMap<String, String>,
    model_root: &std::path::Path,
    capabilities: HashMap<String, String>,
) -> Result<HashMap<String, String>, String> {
    values.insert(
        "ECHOLINGO_MODEL_ROOT".into(),
        model_root.to_string_lossy().into_owned(),
    );
    let alignment_spool = model_root
        .parent()
        .ok_or_else(|| "model directory has no application-data parent".to_string())?
        .join("alignment-spool");
    values.insert(
        "ECHOLINGO_ALIGNMENT_SPOOL_ROOT".into(),
        alignment_spool.to_string_lossy().into_owned(),
    );
    values.extend(capabilities);
    Ok(values)
}

/// Labels of what a credential group still needs before it can be used:
/// required fields without a keychain or environment value, and an empty
/// endpoint (`base_url`) setting. Never includes values.
fn missing_credential_labels(
    group: &CredentialGroup,
    status: &CredentialGroupStatus,
) -> Vec<String> {
    let mut missing: Vec<String> = group
        .fields
        .iter()
        .filter(|field| field.required)
        .filter(|field| {
            !status
                .fields
                .iter()
                .any(|entry| entry.key == field.key && entry.available)
        })
        .map(|field| field.label.clone())
        .collect();
    if let Some(endpoint) = group.setting("base_url") {
        if status
            .settings
            .get("base_url")
            .is_none_or(|value| value.trim().is_empty())
        {
            missing.push(endpoint.label.clone());
        }
    }
    missing
}

/// One keychain mutation planned by [`plan_credential_update`]. `Debug` never
/// prints the value.
#[derive(Clone, PartialEq, Eq)]
enum CredentialWrite {
    Set { account: String, value: String },
    Delete { account: String },
}

impl std::fmt::Debug for CredentialWrite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredentialWrite::Set { account, .. } => formatter
                .debug_struct("Set")
                .field("account", account)
                .field("value", &"[redacted]")
                .finish(),
            CredentialWrite::Delete { account } => formatter
                .debug_struct("Delete")
                .field("account", account)
                .finish(),
        }
    }
}

/// Validate a `set_credentials` payload against the catalog and turn it into
/// keychain writes. Provided values are trimmed and checked against
/// `min_len`; a required field must be provided unless it is already stored
/// (`stored` by keychain account); an empty string deletes an optional field
/// and keeps a stored required one. Error messages carry labels only.
fn plan_credential_update(
    group: &CredentialGroup,
    fields: &HashMap<String, String>,
    stored: impl Fn(&str) -> bool,
) -> Result<Vec<CredentialWrite>, String> {
    for key in fields.keys() {
        if group.field(key).is_none() {
            return Err(format!(
                "{} has no credential field '{key}'",
                group.display_name
            ));
        }
    }
    let mut writes = Vec::new();
    for field in &group.fields {
        let provided = fields.get(&field.key).map(|value| value.trim());
        match provided {
            Some(value) if !value.is_empty() => {
                if value.chars().count() < field.min_len {
                    return Err(format!(
                        "{} is too short (at least {} characters)",
                        field.label, field.min_len
                    ));
                }
                if value.chars().any(char::is_control) {
                    return Err(format!("{} contains control characters", field.label));
                }
                writes.push(CredentialWrite::Set {
                    account: field.keychain_account.clone(),
                    value: value.to_string(),
                });
            }
            Some(_) if !field.required => writes.push(CredentialWrite::Delete {
                account: field.keychain_account.clone(),
            }),
            _ => {
                if field.required && !stored(&field.keychain_account) {
                    return Err(format!("{} is required", field.label));
                }
            }
        }
    }
    Ok(writes)
}

/// Validate and normalise provider settings for one group: unknown keys and
/// invalid `select` values are rejected, values are trimmed and empty values
/// are dropped (the catalog default applies again).
fn normalize_provider_settings(
    group: &CredentialGroup,
    settings: &HashMap<String, String>,
) -> Result<HashMap<String, String>, String> {
    let mut normalized = HashMap::new();
    for (key, value) in settings {
        let setting = group
            .setting(key)
            .ok_or_else(|| format!("{} has no setting '{key}'", group.display_name))?;
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        if value.chars().any(char::is_control) {
            return Err(format!("{} contains control characters", setting.label));
        }
        if !setting.accepts(value) {
            let options = setting
                .options
                .iter()
                .map(|(option, _)| option.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("{} must be one of: {options}", setting.label));
        }
        normalized.insert(key.clone(), value.to_string());
    }
    Ok(normalized)
}

/// Validate a whole `providers` map (every group and key against the catalog).
fn validate_provider_settings(providers: &ProviderSettings) -> Result<ProviderSettings, String> {
    let mut normalized = ProviderSettings::new();
    for (group_id, settings) in providers {
        let group = credential_group(group_id)?;
        let values = normalize_provider_settings(group, settings)?;
        if !values.is_empty() {
            normalized.insert(group_id.clone(), values);
        }
    }
    Ok(normalized)
}

/// Replace every known secret value in `text` so a diagnostic line can never
/// leak a credential even if a provider echoes it back.
fn redact_secrets<'a>(text: &str, secrets: impl IntoIterator<Item = &'a str>) -> String {
    let mut redacted = text.to_string();
    for secret in secrets {
        if secret.len() >= 4 && redacted.contains(secret) {
            redacted = redacted.replace(secret, "[redacted]");
        }
    }
    redacted
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

/// Who asks for the sidecar. The live-session path (Start and the launch
/// warm-up) owns the session, so it may restart a stale sidecar; background
/// users (assistant jobs, credential probes) may run during a live session
/// and must never restart the sidecar under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidecarUser {
    LiveSession,
    Background,
}

struct RuntimeState {
    core: Mutex<AppCore>,
    event_sequence: AtomicU64,
    audio_sequence: AtomicU64,
    audio: tokio::sync::Mutex<Option<AudioCaptureSession>>,
    shutting_down: std::sync::atomic::AtomicBool,
    caption_preferences: Mutex<CaptionPreferences>,
    session_defaults: Mutex<StartSessionRequest>,
    /// Set in `setup` with the app-data log path; falls back to the plain
    /// desktop launch config when first used outside a Tauri app (tests).
    supervisor: std::sync::OnceLock<Arc<InferenceSupervisor>>,
    store: tokio::sync::OnceCell<TranscriptStore>,
    preferences_path: std::sync::OnceLock<PathBuf>,
    logs_directory: std::sync::OnceLock<PathBuf>,
    credentials: CredentialStore,
    /// Credentials or provider settings changed since the sidecar was
    /// launched; the next runtime preparation restarts it with a fresh
    /// environment.
    sidecar_environment_stale: std::sync::atomic::AtomicBool,
    catalog_mismatch_reported: std::sync::atomic::AtomicBool,
    models: std::sync::OnceLock<ModelManager>,
    local_runtimes: std::sync::OnceLock<LocalRuntimeManager>,
    /// The optional NVIDIA acceleration pack (Windows/Linux x64).
    gpu_pack: std::sync::OnceLock<GpuPackManager>,
    /// The secure-store failure has been logged once this run.
    credential_store_failure_reported: std::sync::atomic::AtomicBool,
    /// Models being installed or deleted right now; their services are not
    /// started meanwhile.
    changing_models: Mutex<HashSet<String>>,
    /// Finished sessions whose forced alignment may still run, with the time
    /// of the last sign of it.
    alignment_activity: Mutex<HashMap<uuid::Uuid, std::time::Instant>>,
    onboarding_complete: std::sync::atomic::AtomicBool,
    persistence_throttle: Mutex<PersistenceThrottle>,
    runtime_preferences: Mutex<RuntimePreferences>,
    warmup_in_progress: std::sync::atomic::AtomicBool,
    /// Running AI assistant jobs and picked note attachments.
    assistant: assistant::AssistantRegistry,
    /// Serialises every decision to launch, restart or stop the sidecar, so a
    /// restart can never slip between an assistant job's launch and its
    /// request, and two callers never launch two sidecars.
    sidecar_lifecycle: tokio::sync::Mutex<()>,
    /// An update is being downloaded or installed; sessions, model and GPU
    /// pack changes and sidecar launches are refused until it ends (it
    /// normally ends with a restart). Set under the `core` lock, see
    /// `updater::claim_install`.
    updating: std::sync::atomic::AtomicBool,
    /// The last update check and the download in progress.
    updater: updater::UpdateRegistry,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// `echolingo-sidecar` with this platform's executable suffix.
fn sidecar_file_name() -> String {
    format!("echolingo-sidecar{}", std::env::consts::EXE_SUFFIX)
}

/// Where a packaged sidecar lives, in lookup order: the onedir build shipped
/// as the `sidecar/` resource (Windows, Linux), then next to the app
/// executable (the macOS onefile `externalBin`).
fn bundled_sidecar_candidates(
    resource_directory: Option<&Path>,
    executable_directory: Option<&Path>,
) -> Vec<PathBuf> {
    let name = sidecar_file_name();
    resource_directory
        .map(|directory| directory.join("sidecar").join(&name))
        .into_iter()
        .chain(executable_directory.map(|directory| directory.join(&name)))
        .collect()
}

fn find_bundled_sidecar(
    resource_directory: Option<&Path>,
    executable_directory: Option<&Path>,
) -> Option<PathBuf> {
    bundled_sidecar_candidates(resource_directory, executable_directory)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn executable_directory() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

/// The sidecar executable for the supervisor and the local runtimes:
/// `ECHOLINGO_SIDECAR_EXECUTABLE`, else the bundled one; `None` selects the
/// development Conda environment.
fn resolve_sidecar_executable(resource_directory: Option<&Path>) -> Option<PathBuf> {
    std::env::var_os("ECHOLINGO_SIDECAR_EXECUTABLE")
        .map(PathBuf::from)
        .or_else(|| find_bundled_sidecar(resource_directory, executable_directory().as_deref()))
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            core: Mutex::new(AppCore::default()),
            event_sequence: AtomicU64::new(0),
            audio_sequence: AtomicU64::new(0),
            audio: tokio::sync::Mutex::new(None),
            shutting_down: std::sync::atomic::AtomicBool::new(false),
            caption_preferences: Mutex::new(CaptionPreferences::default()),
            session_defaults: Mutex::new(StartSessionRequest::default()),
            supervisor: std::sync::OnceLock::new(),
            store: tokio::sync::OnceCell::new(),
            preferences_path: std::sync::OnceLock::new(),
            logs_directory: std::sync::OnceLock::new(),
            credentials: CredentialStore::default(),
            sidecar_environment_stale: std::sync::atomic::AtomicBool::new(false),
            catalog_mismatch_reported: std::sync::atomic::AtomicBool::new(false),
            models: std::sync::OnceLock::new(),
            local_runtimes: std::sync::OnceLock::new(),
            gpu_pack: std::sync::OnceLock::new(),
            credential_store_failure_reported: std::sync::atomic::AtomicBool::new(false),
            changing_models: Mutex::new(HashSet::new()),
            alignment_activity: Mutex::new(HashMap::new()),
            onboarding_complete: std::sync::atomic::AtomicBool::new(false),
            persistence_throttle: Mutex::new(PersistenceThrottle::default()),
            runtime_preferences: Mutex::new(RuntimePreferences::default()),
            warmup_in_progress: std::sync::atomic::AtomicBool::new(false),
            assistant: assistant::AssistantRegistry::default(),
            sidecar_lifecycle: tokio::sync::Mutex::new(()),
            updating: std::sync::atomic::AtomicBool::new(false),
            updater: updater::UpdateRegistry::default(),
        }
    }
}

#[derive(Default)]
struct PersistenceThrottle {
    last_metrics_ms: HashMap<uuid::Uuid, f64>,
    last_partial_at: HashMap<(uuid::Uuid, &'static str), std::time::Instant>,
}

impl PersistenceThrottle {
    fn allow_metrics(&mut self, session_id: uuid::Uuid, captured_ms: f64) -> bool {
        let previous = self.last_metrics_ms.entry(session_id).or_insert(-1_000.0);
        if captured_ms - *previous < 1_000.0 {
            return false;
        }
        *previous = captured_ms;
        true
    }

    fn allow_revision(&mut self, session_id: uuid::Uuid, stream: &'static str, kind: &str) -> bool {
        if kind != "partial" {
            return true;
        }
        let now = std::time::Instant::now();
        let previous = self.last_partial_at.entry((session_id, stream)).or_insert(
            now.checked_sub(std::time::Duration::from_millis(500))
                .unwrap_or(now),
        );
        if now.duration_since(*previous) < std::time::Duration::from_millis(500) {
            return false;
        }
        *previous = now;
        true
    }

    fn finish(&mut self, session_id: uuid::Uuid) {
        self.last_metrics_ms.remove(&session_id);
        self.last_partial_at
            .retain(|(stored_session, _), _| *stored_session != session_id);
    }
}

impl RuntimeState {
    fn supervisor(&self) -> &Arc<InferenceSupervisor> {
        self.supervisor.get_or_init(|| {
            InferenceSupervisor::new(SidecarLaunchConfig::desktop(
                project_root(),
                find_bundled_sidecar(None, executable_directory().as_deref()),
            ))
        })
    }

    fn provider_settings(&self) -> Result<ProviderSettings, String> {
        Ok(self
            .runtime_preferences
            .lock()
            .map_err(|_| "runtime preferences lock poisoned".to_string())?
            .providers
            .clone())
    }

    /// Provider credentials and settings for the sidecar. An unreadable
    /// secure store is logged once and the environment-variable keys are
    /// used, so the sidecar and local providers still start.
    fn sidecar_environment(&self) -> Result<HashMap<String, String>, String> {
        let (values, failure) = self
            .credentials
            .sidecar_environment(&self.provider_settings()?);
        if let Some(error) = failure {
            if !self
                .credential_store_failure_reported
                .swap(true, Ordering::AcqRel)
            {
                eprintln!("secure credential store unavailable: {error}");
                self.log_desktop_event(&format!("credential_store_unavailable error={error}"));
            }
        }
        Ok(values)
    }

    /// The complete sidecar launch environment: provider credentials and
    /// settings, the model root, the alignment spool and the local runtime
    /// capabilities. Every path that launches the sidecar (sessions, warm-up,
    /// credential probes, assistant jobs) uses this, so none of them can
    /// leave a partially configured sidecar running.
    fn full_sidecar_environment(&self) -> Result<HashMap<String, String>, String> {
        compose_full_sidecar_environment(
            self.sidecar_environment()?,
            self.models()?.root(),
            self.local_runtimes()?.capability_environment(),
        )
    }

    /// Whether the live session is idle. Assistant jobs are deliberately not
    /// considered: they may run while idle and never block starting a session.
    fn is_idle(&self) -> bool {
        self.snapshot()
            .map(|snapshot| {
                matches!(
                    snapshot.phase,
                    SessionPhase::Idle | SessionPhase::Completed
                )
            })
            .unwrap_or(false)
    }

    /// Launch the sidecar with the full environment if it is not running.
    /// A stale environment (credentials changed since launch) restarts it
    /// first, unless an assistant request is in flight or, for a background
    /// user, a live session runs: then the restart is deferred until the
    /// jobs finish or the session ends. Callers hold `sidecar_lifecycle`.
    async fn start_sidecar_locked(
        &self,
        app: Option<&AppHandle>,
        user: SidecarUser,
    ) -> Result<(), String> {
        // The update shuts the sidecar down and replaces its files; nothing
        // may bring it back meanwhile.
        if self.updating.load(Ordering::Acquire) {
            return Err(updater::UPDATE_IN_PROGRESS.into());
        }
        if self.sidecar_environment_stale.load(Ordering::Acquire) {
            if self.assistant.any_in_flight() {
                self.log_desktop_event("sidecar_restart_deferred reason=assistant_job_running");
            } else if user == SidecarUser::Background && !self.is_idle() {
                // Restarting now would cut off the live session.
                self.log_desktop_event("sidecar_restart_deferred reason=live_session");
            } else {
                // Environment variables only apply at launch. The flag is
                // cleared before the environment is read, so a change that
                // lands in between marks it stale again.
                self.sidecar_environment_stale
                    .store(false, Ordering::Release);
                self.supervisor().shutdown().await;
            }
        }
        let environment = self.full_sidecar_environment()?;
        self.supervisor()
            .configure_secret_environment(environment)
            .await;
        self.supervisor()
            .ensure_started()
            .await
            .map_err(|error| error.to_string())?;
        self.check_catalog_digest(app).await;
        Ok(())
    }

    async fn ensure_sidecar(&self, app: Option<&AppHandle>, user: SidecarUser) -> Result<(), String> {
        let _lifecycle = self.sidecar_lifecycle.lock().await;
        self.start_sidecar_locked(app, user).await
    }

    /// Credentials or settings changed: restart the sidecar now when nothing
    /// is running, otherwise before the next session (or once the running
    /// assistant jobs finish) so neither a live session nor a job is cut off.
    async fn invalidate_sidecar_environment(&self) {
        self.sidecar_environment_stale
            .store(true, Ordering::Release);
        self.restart_stale_sidecar_if_quiet().await;
    }

    /// Stop a sidecar whose environment is stale when no live session and no
    /// assistant request uses it; the next user launches it fresh. Never
    /// waits: while a launch holds the lifecycle lock, the stale flag makes
    /// that launch (or the next one) restart instead.
    async fn restart_stale_sidecar_if_quiet(&self) {
        let Ok(_lifecycle) = self.sidecar_lifecycle.try_lock() else {
            return;
        };
        if self.sidecar_environment_stale.load(Ordering::Acquire)
            && self.is_idle()
            && !self.assistant.any_in_flight()
        {
            self.supervisor().shutdown().await;
            self.sidecar_environment_stale
                .store(false, Ordering::Release);
        }
    }

    fn assistant_preferences(&self) -> Result<AssistantPreferences, String> {
        Ok(self
            .runtime_preferences
            .lock()
            .map_err(|_| "runtime preferences lock poisoned".to_string())?
            .assistant
            .clone())
    }

    /// Compare the sidecar's provider catalog with the one embedded in this
    /// binary. A mismatch means `configs/providers.json` was not regenerated
    /// after a registry change; report it once so Settings can be trusted.
    async fn check_catalog_digest(&self, app: Option<&AppHandle>) {
        let Some(sidecar) = self.supervisor().providers_digest().await else {
            return;
        };
        if sidecar == catalog_digest() {
            return;
        }
        if self
            .catalog_mismatch_reported
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let message = format!(
            "Provider catalog mismatch: sidecar {} vs embedded {}; regenerate configs/providers.json",
            &sidecar[..sidecar.len().min(12)],
            &catalog_digest()[..12]
        );
        eprintln!("{message}");
        self.log_desktop_event(&format!("providers_catalog_mismatch {message}"));
        if let Some(app) = app {
            let _ = self.emit_event(
                app,
                None,
                UiEventKind::Error,
                json!({
                    "code": "providers_catalog_mismatch",
                    "message": message,
                    "recoverable": true
                }),
            );
        }
    }

    /// Append one redacted diagnostic line to `<app_data>/logs/desktop.log`.
    /// Callers pass text that never contains credentials; known secret values
    /// are scrubbed again here as a last line of defence.
    fn log_desktop_event(&self, line: &str) {
        let Some(directory) = self.logs_directory.get() else {
            return;
        };
        let secrets: Vec<String> = self
            .sidecar_environment()
            .map(|values| {
                let secret_names: std::collections::HashSet<&str> = catalog()
                    .credential_groups
                    .iter()
                    .flat_map(|group| group.fields.iter())
                    .filter(|field| field.secret)
                    .map(|field| field.env_var.as_str())
                    .collect();
                values
                    .into_iter()
                    .filter(|(name, _)| secret_names.contains(name.as_str()))
                    .map(|(_, value)| value)
                    .collect()
            })
            .unwrap_or_default();
        let line = redact_secrets(line, secrets.iter().map(String::as_str));
        let entry = format!(
            "{} {}\n",
            chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            line.replace(['\n', '\r'], " ")
        );
        if let Err(error) = append_log_chunk(
            &directory.join(DESKTOP_LOG_FILE),
            entry.as_bytes(),
            SIDECAR_LOG_ROTATE_BYTES,
        ) {
            eprintln!("desktop log unavailable: {error}");
        }
    }

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
            runtime: self
                .runtime_preferences
                .lock()
                .map_err(|_| "runtime preferences lock poisoned".to_string())?
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

    fn local_runtimes(&self) -> Result<&LocalRuntimeManager, String> {
        self.local_runtimes
            .get()
            .ok_or_else(|| "local runtime manager is not initialized".to_string())
    }

    /// The model being installed or deleted that `service` needs, if any.
    fn changing_model_for(&self, service: &str) -> Option<String> {
        let changing = self.changing_models.lock().ok()?;
        changing
            .iter()
            .find(|model| services_using_model(model).contains(&service))
            .cloned()
    }

    fn model_changing(&self, model_id: &str) -> bool {
        self.changing_models
            .lock()
            .is_ok_and(|changing| changing.contains(model_id))
    }

    /// Note sidecar activity for a finished session's forced alignment;
    /// `None` forgets all of it (the alignment failed or the sidecar went
    /// away).
    fn note_alignment_activity(&self, session_id: Option<uuid::Uuid>) {
        if let Ok(mut activity) = self.alignment_activity.lock() {
            match session_id {
                Some(session_id) => {
                    activity.insert(session_id, std::time::Instant::now());
                }
                None => activity.clear(),
            }
        }
    }

    fn alignment_running(&self) -> bool {
        self.alignment_activity.lock().is_ok_and(|mut activity| {
            activity.retain(|_, at| at.elapsed() < ALIGNMENT_QUIET_PERIOD);
            !activity.is_empty()
        })
    }

    fn gpu_pack(&self) -> Result<&GpuPackManager, String> {
        self.gpu_pack
            .get()
            .ok_or_else(|| "GPU acceleration pack manager is not initialized".to_string())
    }

    fn gpu_acceleration_enabled(&self) -> bool {
        self.runtime_preferences
            .lock()
            .is_ok_and(|preferences| preferences.gpu_acceleration)
    }

    fn check_updates_at_launch(&self) -> bool {
        self.runtime_preferences
            .lock()
            .is_ok_and(|preferences| preferences.check_updates_at_launch)
    }

    /// The pack runtime when the user wants it and the installed pack has a
    /// usable CUDA device; `None` selects the bundled CPU runtime.
    fn desired_gpu_runtime(&self) -> Option<GpuRuntimeLayout> {
        if !self.gpu_acceleration_enabled() {
            return None;
        }
        self.gpu_pack.get()?.runtime_layout()
    }

    /// Point the local model services at the desired runtime, restarting
    /// them on a change. Callers make sure no session uses them.
    async fn apply_gpu_runtime(&self) -> Result<(), String> {
        let desired = self.desired_gpu_runtime();
        let accelerated = desired.is_some();
        if self
            .local_runtimes()?
            .select_gpu_runtime(desired)
            .await
        {
            self.log_desktop_event(&format!("local_runtime_selected gpu={accelerated}"));
        }
        Ok(())
    }

    async fn gpu_status(&self) -> Result<GpuAccelerationStatus, String> {
        let runtimes = self.local_runtimes()?;
        Ok(self
            .gpu_pack()?
            .status(
                self.gpu_acceleration_enabled(),
                runtimes.gpu_active(),
                runtimes.gpu_fallback_reason(),
            )
            .await)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct DesktopPreferences {
    schema_version: u16,
    onboarding_complete: bool,
    session: StartSessionRequest,
    caption: CaptionPreferences,
    runtime: RuntimePreferences,
}

impl Default for DesktopPreferences {
    fn default() -> Self {
        Self {
            schema_version: 1,
            onboarding_complete: false,
            session: StartSessionRequest::default(),
            caption: CaptionPreferences::default(),
            runtime: RuntimePreferences::default(),
        }
    }
}

/// Non-secret runtime behaviour the user controls from Settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct RuntimePreferences {
    /// Start the inference sidecar and the local model services right after
    /// launch so the first Start does not wait ~1-2 minutes for model load.
    preload_local_models: bool,
    /// Non-secret `ProviderSetting` values per credential group, for example
    /// `{"dashscope": {"region": "beijing"}}`. Secrets never live here.
    providers: ProviderSettings,
    /// AI assistant for session notes and titles.
    assistant: AssistantPreferences,
    /// Run the local models on the NVIDIA acceleration pack when it is
    /// installed and usable. Changed only through `set_gpu_acceleration`.
    gpu_acceleration: bool,
    /// Ask the update server for a newer version shortly after launch.
    /// Files written before updates existed load with it on.
    check_updates_at_launch: bool,
}

impl Default for RuntimePreferences {
    fn default() -> Self {
        Self {
            preload_local_models: true,
            providers: ProviderSettings::new(),
            assistant: AssistantPreferences::default(),
            gpu_acceleration: true,
            check_updates_at_launch: true,
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
    validate_session_text(defaults).map_err(|error| error.to_string())?;
    if !matches!(
        defaults.audio_profile.as_str(),
        "lecture" | "conversation" | "raw"
    ) {
        return Err("unknown audio profile".into());
    }
    let catalog = catalog();
    let selectable = |kind: ProviderKind, id: &str| {
        id == "auto"
            || catalog
                .find(kind, id)
                .is_some_and(|spec| spec.selectable)
    };
    if !selectable(ProviderKind::Asr, &defaults.asr_provider) {
        return Err("unknown ASR provider".into());
    }
    if !selectable(ProviderKind::Translation, &defaults.translation_provider) {
        return Err("unknown translation provider".into());
    }
    let cloud = |kind: ProviderKind, id: &str| {
        catalog
            .find(kind, id)
            .is_some_and(|spec| spec.is_cloud())
    };
    if !cloud(ProviderKind::Asr, &defaults.cloud_asr_preference) {
        return Err("cloud ASR preference must name a cloud provider".into());
    }
    if !cloud(
        ProviderKind::Translation,
        &defaults.cloud_translation_preference,
    ) {
        return Err("cloud translation preference must name a cloud provider".into());
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

/// `name` (plus the platform's executable suffix) on `PATH`.
fn executable_on_path(name: &str) -> Option<PathBuf> {
    let name = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|directory| directory.join(&name))
            .find(|candidate| candidate.is_file())
    })
}

/// The local runtime layout. `bundled_sidecar` is the resolved sidecar
/// executable (see [`resolve_sidecar_executable`]); it serves
/// `qwen-asr-server` and wraps llama.cpp in its owner watchdog.
fn local_runtime_layout(
    model_root: PathBuf,
    app_data_directory: &Path,
    resource_directory: &Path,
    bundled_sidecar: Option<PathBuf>,
    qwen_port: u16,
    hymt_port: u16,
) -> LocalRuntimeLayout {
    let worker_wrapper = bundled_sidecar.clone();
    let qwen_command = if let Some(executable) =
        std::env::var_os("ECHOLINGO_QWEN_ASR_COMMAND").map(PathBuf::from)
    {
        RuntimeCommand {
            executable,
            args: Vec::new(),
            environment: HashMap::new(),
        }
    } else if let Some(executable) = bundled_sidecar {
        RuntimeCommand {
            executable,
            args: vec!["qwen-asr-server".into()],
            environment: HashMap::new(),
        }
    } else {
        RuntimeCommand {
            executable: PathBuf::from("conda"),
            args: vec![
                "run".into(),
                "--no-capture-output".into(),
                "-n".into(),
                "echolingo-spike1".into(),
                "python".into(),
                "-m".into(),
                "echolingo.service.qwen_server".into(),
            ],
            environment: HashMap::new(),
        }
    };
    let bundled_llama = resource_directory
        .join("runtimes")
        .join("llama.cpp")
        .join(format!("llama-server{}", std::env::consts::EXE_SUFFIX));
    let llama_server = std::env::var_os("ECHOLINGO_LLAMA_SERVER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| bundled_llama.is_file().then_some(bundled_llama))
        .or_else(|| executable_on_path("llama-server"));
    LocalRuntimeLayout {
        qwen_command,
        llama_server,
        worker_wrapper,
        model_root,
        log_root: app_data_directory.join("logs"),
        qwen_device: if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "mps".into()
        } else {
            "auto".into()
        },
        qwen_streaming: QwenStreamingProfile::from_environment(),
        local_api_key: format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        ),
        qwen_port,
        hymt_port,
        startup_timeout: std::time::Duration::from_secs(180),
    }
}

fn reserve_local_runtime_ports() -> std::io::Result<(std::net::TcpListener, std::net::TcpListener)>
{
    let qwen = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let hymt = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok((qwen, hymt))
}

fn load_preferences(path: &std::path::Path) -> DesktopPreferences {
    let mut preferences = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<DesktopPreferences>(&bytes).ok())
        .unwrap_or_default();
    if validate_session_text(&preferences.session).is_err() {
        // Only a hand-edited file can hold over-long text; drop the text,
        // not the whole file.
        preferences.session.session_context.clear();
        preferences.session.glossary.clear();
    }
    if preferences.schema_version != 1
        || validate_session_defaults(&preferences.session).is_err()
        || validate_caption_preferences(&preferences.caption).is_err()
    {
        return DesktopPreferences::default();
    }
    // Provider settings are dropped on corruption rather than discarding the
    // whole file; the catalog defaults apply again.
    preferences.runtime.providers =
        validate_provider_settings(&preferences.runtime.providers).unwrap_or_default();
    // An assistant choice the catalog no longer offers resets to "off"
    // (which also withdraws the upload consent).
    preferences.runtime.assistant = preferences
        .runtime
        .assistant
        .normalized(catalog())
        .unwrap_or_default();
    preferences
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

/// The embedded provider catalog (`configs/providers.json`), verbatim.
#[tauri::command]
fn list_providers() -> Result<Value, String> {
    Ok(catalog_value())
}

/// Keychain reads can block (or prompt), so they run off the async runtime.
#[tauri::command]
async fn credential_status(
    app: AppHandle,
    state: State<'_, RuntimeState>,
) -> Result<Vec<CredentialGroupStatus>, String> {
    let providers = state.provider_settings()?;
    tauri::async_runtime::spawn_blocking(move || {
        app.state::<RuntimeState>().credentials.status(&providers)
    })
    .await
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn logs_directory(state: State<'_, RuntimeState>) -> Result<String, String> {
    state
        .logs_directory
        .get()
        .map(|path| path.to_string_lossy().into_owned())
        .ok_or_else(|| "logs directory is not initialized".to_string())
}

/// Store credential fields for one catalog group in the OS secure store.
/// Values are validated against the catalog (`min_len`, required fields),
/// written per field, and never echoed back, logged or kept in memory.
#[tauri::command]
async fn set_credentials(
    state: State<'_, RuntimeState>,
    group_id: String,
    fields: HashMap<String, String>,
) -> Result<CredentialGroupStatus, String> {
    let group = credential_group(&group_id)?;
    let mut stored = HashMap::new();
    for field in &group.fields {
        stored.insert(
            field.keychain_account.as_str(),
            state.credentials.get(&field.keychain_account)?.is_some(),
        );
    }
    let writes = plan_credential_update(group, &fields, |account| {
        stored.get(account).copied().unwrap_or(false)
    })?;
    drop(fields);
    let mut created = Vec::new();
    for write in &writes {
        let result = match write {
            CredentialWrite::Set { account, value } => state.credentials.set(account, value),
            CredentialWrite::Delete { account } => state.credentials.delete(account),
        };
        if let Err(error) = result {
            // Never leave a group half-saved: undo entries this call created.
            for account in created {
                let _ = state.credentials.delete(account);
            }
            return Err(error);
        }
        if let CredentialWrite::Set { account, .. } = write {
            if !stored.get(account.as_str()).copied().unwrap_or(false) {
                created.push(account.as_str());
            }
        }
    }
    state.invalidate_sidecar_environment().await;
    state
        .credentials
        .group_status(&group_id, &state.provider_settings()?)
}

#[tauri::command]
async fn clear_credentials(
    state: State<'_, RuntimeState>,
    group_id: String,
) -> Result<CredentialGroupStatus, String> {
    let group = credential_group(&group_id)?;
    for field in &group.fields {
        state.credentials.delete(&field.keychain_account)?;
    }
    state.invalidate_sidecar_environment().await;
    state
        .credentials
        .group_status(&group_id, &state.provider_settings()?)
}

/// Merge non-secret settings for one credential group into the runtime
/// preferences (an empty value resets that key to the catalog default).
#[tauri::command]
async fn update_provider_settings(
    state: State<'_, RuntimeState>,
    group_id: String,
    settings: HashMap<String, String>,
) -> Result<RuntimePreferences, String> {
    let group = credential_group(&group_id)?;
    let mut cleared = Vec::new();
    for (key, value) in &settings {
        if group.setting(key).is_none() {
            return Err(format!("{} has no setting '{key}'", group.display_name));
        }
        if value.trim().is_empty() {
            cleared.push(key.clone());
        }
    }
    let normalized = normalize_provider_settings(group, &settings)?;
    let (preferences, changed) = {
        let mut guard = state
            .runtime_preferences
            .lock()
            .map_err(|_| "runtime preferences lock poisoned".to_string())?;
        let previous = guard.providers.clone();
        let entry = guard.providers.entry(group_id.clone()).or_default();
        for key in cleared {
            entry.remove(&key);
        }
        entry.extend(normalized);
        if entry.is_empty() {
            guard.providers.remove(&group_id);
        }
        let changed = guard.providers != previous;
        (guard.clone(), changed)
    };
    state.persist_preferences()?;
    if changed {
        state.invalidate_sidecar_environment().await;
    }
    Ok(preferences)
}

/// Validate cloud credentials for the requested providers by asking the
/// sidecar (launched with its complete environment) to probe them. ASR probes perform the
/// authenticated handshake only (no audio is uploaded); translation probes
/// translate one fixed sentence. Failures are appended, redacted, to
/// `<app_data>/logs/desktop.log`.
#[tauri::command]
async fn probe_cloud(
    state: State<'_, RuntimeState>,
    asr_provider: Option<String>,
    translation_provider: Option<String>,
) -> Result<Value, String> {
    if !state.is_idle() {
        return Err("Stop the current session before testing cloud credentials".into());
    }
    if state.assistant.any_running() {
        // A probe may restart the sidecar for fresh credentials, which would
        // cut the job off.
        return Err(assistant::ASSISTANT_BUSY.into());
    }
    let asr_provider = asr_provider.filter(|id| !id.trim().is_empty());
    let translation_provider = translation_provider.filter(|id| !id.trim().is_empty());
    if asr_provider.is_none() && translation_provider.is_none() {
        return Err("Choose at least one cloud provider to test".into());
    }
    let providers = state.provider_settings()?;
    for (kind, id) in [
        (ProviderKind::Asr, asr_provider.as_deref()),
        (ProviderKind::Translation, translation_provider.as_deref()),
    ] {
        let Some(id) = id else {
            continue;
        };
        let spec = catalog()
            .find(kind, id)
            .ok_or_else(|| format!("unknown {kind} provider '{id}'"))?;
        if !spec.is_cloud() {
            return Err(format!("{} is not a cloud provider", spec.display_name));
        }
        let Some(group_id) = spec.credential_group.as_deref() else {
            continue;
        };
        let group = credential_group(group_id)?;
        let status = state.credentials.group_status(group_id, &providers)?;
        let missing = missing_credential_labels(group, &status);
        if !missing.is_empty() {
            return Err(format!(
                "Save the {} credentials first (missing: {})",
                group.display_name,
                missing.join(", ")
            ));
        }
    }

    let describe = format!(
        "asr={} translation={}",
        asr_provider.as_deref().unwrap_or("-"),
        translation_provider.as_deref().unwrap_or("-")
    );
    let supervisor = state.supervisor().clone();
    let outcome = async {
        // Launch (or restart, when credentials changed since launch) with the
        // complete environment, so the sidecar stays usable for the next
        // session and for assistant jobs after the probe.
        state.ensure_sidecar(None, SidecarUser::Background).await?;
        let mut events = supervisor.subscribe();
        let request_id = uuid::Uuid::new_v4();
        supervisor
            .send_command(SidecarCommand::ProbeCloud {
                request_id,
                asr_provider: asr_provider.clone(),
                translation_provider: translation_provider.clone(),
            })
            .await
            .map_err(|error| error.to_string())?;
        tokio::time::timeout(std::time::Duration::from_secs(30), async move {
            loop {
                match events.recv().await.map_err(|error| error.to_string())? {
                    SidecarEvent::CloudProbeResult {
                        request_id: response_id,
                        result,
                    } if response_id == request_id => return Ok(result),
                    SidecarEvent::Error { message, .. } => return Err(message),
                    _ => {}
                }
            }
        })
        .await
        .map_err(|_| "Cloud provider validation timed out".to_string())?
    }
    .await;
    match &outcome {
        Ok(result) if result["ok"].as_bool() == Some(true) => {}
        Ok(result) => state.log_desktop_event(&format!(
            "probe_cloud failed {describe} code={} message={}",
            result["code"].as_str().unwrap_or("cloud_probe_failed"),
            result["message"].as_str().unwrap_or("no message")
        )),
        Err(message) => state.log_desktop_event(&format!(
            "probe_cloud failed {describe} code=probe_transport message={message}"
        )),
    }
    outcome
}

#[tauri::command]
fn list_models(state: State<'_, RuntimeState>) -> Result<Vec<ModelStatus>, String> {
    Ok(state.models()?.list())
}

/// The local model services that keep files of `model_id` open; the
/// forced aligner runs inside the sidecar itself.
fn services_using_model(model_id: &str) -> &'static [&'static str] {
    match model_id {
        "qwen3-asr-0.6b" => &["qwen_asr"],
        "hymt2-1.8b" => &["hymt"],
        _ => &[],
    }
}

const MODEL_SESSION_BUSY: &str = "Stop the current session before changing local models";

/// A model install or delete in progress; while it lives, the services
/// that need the model are not started.
struct ModelChange<'a> {
    state: &'a RuntimeState,
    model_id: String,
}

impl Drop for ModelChange<'_> {
    fn drop(&mut self) {
        if let Ok(mut changing) = self.state.changing_models.lock() {
            changing.remove(&self.model_id);
        }
    }
}

/// Claim `model_id` for an install or delete and stop every process that
/// has its files open, so they can be replaced or deleted (Windows refuses
/// to while they are open). Refused during a session, a background warm-up,
/// another change of the same model, and, for the aligner, while assistant
/// jobs or a forced alignment use the sidecar.
async fn begin_model_change<'a>(
    state: &'a RuntimeState,
    model_id: &str,
) -> Result<ModelChange<'a>, String> {
    let _lifecycle = state.sidecar_lifecycle.lock().await;
    if !state.is_idle() {
        return Err(MODEL_SESSION_BUSY.into());
    }
    if state.warmup_in_progress.load(Ordering::Acquire) {
        return Err(
            "Local models are loading in the background; try again when they are ready".into(),
        );
    }
    let status = state
        .models()?
        .status(model_id)
        .map_err(|error| error.to_string())?;
    let busy = format!("{} is already being installed or removed", status.display_name);
    if status.state == ModelInstallState::Installing {
        return Err(busy);
    }
    if !state
        .changing_models
        .lock()
        .map_err(|_| "model change lock poisoned".to_string())?
        .insert(model_id.to_string())
    {
        return Err(busy);
    }
    let change = ModelChange {
        state,
        model_id: model_id.to_string(),
    };
    // Checked after registering the change: an update that starts in
    // between sees the registration and refuses instead.
    if state.updating.load(Ordering::Acquire) {
        return Err(updater::UPDATE_IN_PROGRESS.into());
    }
    if status.role == runtime_manager::ModelRole::Alignment {
        if state.assistant.any_running() {
            return Err(assistant::ASSISTANT_BUSY.into());
        }
        if state.alignment_running() {
            return Err(
                "Word timings of the last session are still being aligned; try again in a minute"
                    .into(),
            );
        }
        state.supervisor().shutdown().await;
    }
    for service in services_using_model(model_id) {
        state.local_runtimes()?.stop_service(service).await;
    }
    Ok(change)
}

#[tauri::command]
async fn install_model(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    model_id: String,
) -> Result<ModelStatus, String> {
    let _change = begin_model_change(&state, &model_id).await?;
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
async fn delete_model(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    model_id: String,
) -> Result<ModelStatus, String> {
    let _change = begin_model_change(&state, &model_id).await?;
    // Deleting gigabytes can take a while; keep it off the async runtime.
    tauri::async_runtime::spawn_blocking(move || {
        app.state::<RuntimeState>()
            .models()?
            .delete(&model_id)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

const GPU_SESSION_BUSY: &str = "Stop the current session before changing GPU acceleration";

#[tauri::command]
async fn gpu_acceleration_status(
    state: State<'_, RuntimeState>,
) -> Result<GpuAccelerationStatus, String> {
    state.gpu_status().await
}

/// Download, verify, self-test and activate the NVIDIA acceleration pack.
/// Long-running; progress goes to the model-progress channel with
/// `model_id: "gpu-pack"`.
#[tauri::command]
async fn install_gpu_pack(
    app: AppHandle,
    state: State<'_, RuntimeState>,
) -> Result<GpuAccelerationStatus, String> {
    if !state.is_idle() {
        return Err(GPU_SESSION_BUSY.into());
    }
    if state.updating.load(Ordering::Acquire) {
        return Err(updater::UPDATE_IN_PROGRESS.into());
    }
    let status = state.gpu_status().await?;
    if !status.supported_platform || !status.eligible {
        return Err(status
            .ineligible_reason
            .unwrap_or_else(|| "GPU acceleration is not available on this computer".into()));
    }
    if status.pack_state != GpuPackState::NotInstalled {
        // Replacing a pack: move the services off its files first.
        state.local_runtimes()?.select_gpu_runtime(None).await;
    }
    let progress_app = app.clone();
    let callback = Arc::new(move |progress: ModelProgress| {
        let _ = progress_app.emit(MODEL_PROGRESS_CHANNEL, progress);
    });
    let installed = state.gpu_pack()?.install(callback).await;
    match &installed {
        Ok(record) => {
            state.local_runtimes()?.clear_gpu_fallback();
            state.log_desktop_event(&format!(
                "gpu_pack_installed version={} cuda_available={:?}",
                record.manifest.app_version,
                record.cuda_available()
            ));
        }
        Err(error) => state.log_desktop_event(&format!("gpu_pack_install_failed error={error}")),
    }
    // A session started during the download picks the pack up at its end.
    if state.is_idle() {
        state.apply_gpu_runtime().await?;
    }
    installed.map_err(|error| error.to_string())?;
    state.gpu_status().await
}

#[tauri::command]
async fn remove_gpu_pack(
    app: AppHandle,
    state: State<'_, RuntimeState>,
) -> Result<GpuAccelerationStatus, String> {
    if !state.is_idle() {
        return Err(GPU_SESSION_BUSY.into());
    }
    state.local_runtimes()?.select_gpu_runtime(None).await;
    let removal_app = app.clone();
    let removed = tauri::async_runtime::spawn_blocking(move || {
        removal_app
            .state::<RuntimeState>()
            .gpu_pack()?
            .remove()
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?;
    state.apply_gpu_runtime().await?;
    removed?;
    state.gpu_status().await
}

#[tauri::command]
async fn set_gpu_acceleration(
    state: State<'_, RuntimeState>,
    enabled: bool,
) -> Result<GpuAccelerationStatus, String> {
    if !state.is_idle() {
        return Err(GPU_SESSION_BUSY.into());
    }
    state
        .runtime_preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_string())?
        .gpu_acceleration = enabled;
    state.persist_preferences()?;
    state.apply_gpu_runtime().await?;
    state.gpu_status().await
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
    let catalog = catalog();
    let asr_spec = catalog.find(ProviderKind::Asr, &asr);
    let translation_spec = catalog.find(ProviderKind::Translation, &translation);
    let text = |key: &str| {
        value[key]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    // Prefer what the sidecar reports (registry locality/model/display name);
    // fall back to the catalog, then to the historical id heuristics.
    let locality = |key: &str, provider: &str, spec: Option<&app_core::ProviderSpec>| {
        text(key)
            .or_else(|| spec.map(|spec| spec.locality.as_str().to_string()))
            .unwrap_or_else(|| {
                if provider.contains("cloud") {
                    "cloud".into()
                } else {
                    "local".into()
                }
            })
    };
    let display_name = |key: &str, spec: Option<&app_core::ProviderSpec>| {
        text(key).or_else(|| spec.map(|spec| spec.display_name.clone()))
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
        asr_model: text("asr_model").or_else(|| match asr.as_str() {
            "qwen_local" => Some("qwen3-asr-0.6b".into()),
            "qwen_cloud" => Some("qwen3-asr-flash-realtime".into()),
            _ => None,
        }),
        asr_locality: locality("asr_locality", &asr, asr_spec),
        asr_display_name: display_name("asr_display_name", asr_spec),
        asr_health: if degraded {
            BackendHealth::Degraded
        } else {
            BackendHealth::Connected
        },
        translation_model: text("translation_model").or_else(|| match translation.as_str() {
            "hymt_local" => Some("tencent/Hy-MT2-1.8B".into()),
            "qwen_cloud" => Some("qwen-mt-flash".into()),
            _ => None,
        }),
        translation_locality: locality("translation_locality", &translation, translation_spec),
        translation_display_name: display_name("translation_display_name", translation_spec),
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
            let persist_revision = state
                .persistence_throttle
                .lock()
                .map_err(|_| "persistence throttle lock poisoned".to_string())?
                .allow_revision(session_id, "source", kind);
            if !persist_revision {
                return Ok(());
            }
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
            if kind == "error" {
                return Ok(());
            }
            let persist_revision = state
                .persistence_throttle
                .lock()
                .map_err(|_| "persistence throttle lock poisoned".to_string())?
                .allow_revision(session_id, "target", kind);
            if !persist_revision {
                return Ok(());
            }
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
            let source_committed = payload["source_committed"].as_bool().unwrap_or(false);
            if source_committed && kind == "final" && !text.is_empty() {
                store
                    .update_translation(session_id, source_revision, revision, text, true)
                    .await
                    .map_err(|error| error.to_string())?;
            }
        }
        SidecarEvent::Metrics(payload) => {
            let captured_audio_ms = payload["captured_audio_ms"].as_f64().unwrap_or(0.0);
            let persist_metrics = state
                .persistence_throttle
                .lock()
                .map_err(|_| "persistence throttle lock poisoned".to_string())?
                .allow_metrics(session_id, captured_audio_ms);
            if !persist_metrics {
                return Ok(());
            }
            store
                .record_metrics(session_id, captured_audio_ms, payload)
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

/// Minimum spacing between UI emits of streaming translation deltas for one
/// request. Core state is always updated; only the webview fan-out is paced.
const TRANSLATION_DELTA_UI_INTERVAL: std::time::Duration = std::time::Duration::from_millis(66);

/// One item of the FIFO persistence queue: the session it belongs to and the
/// sidecar event to store.
type PersistItem = (uuid::Uuid, SidecarEvent);

/// Drain the persistence queue in order. `SessionFinished` is not stored: it
/// reaches `on_session_finished` only after every earlier event of the queue
/// has been written, so the hook (the auto title) sees the whole transcript.
async fn run_persistence_queue<Report, Finished, Hook>(
    state: &RuntimeState,
    mut queue: tokio::sync::mpsc::Receiver<PersistItem>,
    mut report_error: Report,
    mut on_session_finished: Finished,
) where
    Report: FnMut(uuid::Uuid, String),
    Finished: FnMut(uuid::Uuid) -> Hook,
    Hook: std::future::Future<Output = ()>,
{
    while let Some((session_id, event)) = queue.recv().await {
        if let SidecarEvent::SessionFinished {
            session_id: finished,
        } = event
        {
            on_session_finished(finished).await;
            continue;
        }
        if let Err(error) = persist_sidecar_event(state, session_id, &event).await {
            report_error(session_id, error);
        }
    }
}

fn forward_sidecar_events(app: AppHandle) {
    let mut receiver = app.state::<RuntimeState>().supervisor().subscribe();
    // Persistence runs on its own task so SQLite latency never delays the
    // canonical UI event stream. The channel is bounded; the forwarder awaits
    // when it fills, which only happens if storage is far behind.
    let (persist_tx, persist_rx) = tokio::sync::mpsc::channel::<PersistItem>(4_096);
    let persist_app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = persist_app.state::<RuntimeState>();
        run_persistence_queue(
            &state,
            persist_rx,
            |session_id, error| {
                let _ = state.emit_event(
                    &persist_app,
                    Some(session_id),
                    UiEventKind::Error,
                    json!({"code": "storage_error", "message": error, "recoverable": true}),
                );
            },
            |session_id| {
                // The title job runs on its own task; the queue keeps draining.
                assistant::spawn_auto_title(persist_app.clone(), session_id);
                std::future::ready(())
            },
        )
        .await;
    });
    tauri::async_runtime::spawn(async move {
        let mut last_delta_emit: HashMap<String, std::time::Instant> = HashMap::new();
        loop {
            let event = match receiver.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    eprintln!("sidecar event forwarder lagged; skipped {skipped} events");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
            let state = app.state::<RuntimeState>();
            if let SidecarEvent::SessionFinished { session_id } = &event {
                // Forced alignment starts in the sidecar now.
                state.note_alignment_activity(Some(*session_id));
                // Queued behind the session's last transcript/translation
                // events (auto title); never shown in the UI.
                if persist_tx.send((*session_id, event.clone())).await.is_err() {
                    break;
                }
                continue;
            }
            let persist_event = event.clone();
            let mut segment_update: Option<SegmentSummary> = None;
            let mut suppress_ui = false;
            let (kind, payload) = match event {
                SidecarEvent::Transcript(payload) => {
                    if let Ok(mut core) = state.core.lock() {
                        let projection = core.apply_transcript_event(&payload);
                        segment_update = projection.segment;
                    }
                    (UiEventKind::TranscriptRevision, payload)
                }
                SidecarEvent::Translation(payload) => {
                    if let Ok(mut core) = state.core.lock() {
                        segment_update = core.apply_translation_event(&payload);
                    }
                    if payload["kind"].as_str() == Some("partial") {
                        let request = payload["request_id"].as_str().unwrap_or_default().to_string();
                        let now = std::time::Instant::now();
                        match last_delta_emit.get(&request) {
                            Some(previous) if now.duration_since(*previous) < TRANSLATION_DELTA_UI_INTERVAL => {
                                suppress_ui = true;
                            }
                            _ => {
                                last_delta_emit.insert(request, now);
                                if last_delta_emit.len() > 64 {
                                    last_delta_emit.retain(|_, at| now.duration_since(*at) < std::time::Duration::from_secs(30));
                                }
                            }
                        }
                    }
                    (UiEventKind::TranslationRevision, payload)
                }
                SidecarEvent::Metrics(payload) => {
                    if let Ok(metrics) = serde_json::from_value::<LiveMetrics>(payload.clone()) {
                        if let Ok(mut core) = state.core.lock() {
                            // Latencies derived from transcript/translation
                            // events live in core; keep them across samples.
                            let current = core.snapshot().metrics;
                            core.update_metrics(LiveMetrics {
                                asr_first_partial_latency_ms: current.asr_first_partial_latency_ms,
                                asr_commit_latency_ms: current.asr_commit_latency_ms,
                                end_to_end_latency_ms: current.end_to_end_latency_ms,
                                translation_latency_ms: metrics
                                    .translation_latency_ms
                                    .or(current.translation_latency_ms),
                                ..metrics
                            });
                        }
                    }
                    (UiEventKind::Metrics, payload)
                }
                SidecarEvent::SegmentCommitted(payload) => (UiEventKind::SegmentCommitted, payload),
                SidecarEvent::AlignmentUpdate(payload) => {
                    if let Some(session_id) = payload["session_id"]
                        .as_str()
                        .and_then(|value| value.parse().ok())
                    {
                        state.note_alignment_activity(Some(session_id));
                    }
                    // Alignment refines persisted timings only; it is not a
                    // live transcript revision and must never touch live text.
                    suppress_ui = true;
                    (UiEventKind::TranscriptRevision, payload)
                }
                SidecarEvent::BackendHealth(payload) => (UiEventKind::BackendHealth, payload),
                SidecarEvent::Error {
                    code,
                    message,
                    recoverable,
                } => {
                    if code == "sidecar_disconnected" || code == "alignment_failed" {
                        state.note_alignment_activity(None);
                    }
                    if code == "sidecar_disconnected" {
                        let recovery = state.core.lock().ok().and_then(|mut core| {
                            if core.snapshot().phase == SessionPhase::Listening {
                                core.backend_disconnected(&message).ok()
                            } else {
                                None
                            }
                        });
                        if let Some(snapshot) = recovery {
                            let _ = state.emit_snapshot(&app, &snapshot);
                        }
                    }
                    (
                        UiEventKind::Error,
                        json!({"code": code, "message": message, "recoverable": recoverable}),
                    )
                }
                SidecarEvent::Ready { route, .. } => (UiEventKind::RouteDecision, route),
                SidecarEvent::HelloAccepted { .. }
                | SidecarEvent::RoutePlan { .. }
                | SidecarEvent::CloudProbeResult { .. }
                | SidecarEvent::SessionFinished { .. }
                // Assistant events are consumed by the job that sent the
                // request; they never reach the live stream.
                | SidecarEvent::AssistantProgress { .. }
                | SidecarEvent::AssistantDelta { .. }
                | SidecarEvent::AssistantResult { .. } => continue,
            };
            let active_session_id = state
                .snapshot()
                .ok()
                .and_then(|snapshot| snapshot.session_id);
            let event_session_id = payload["session_id"]
                .as_str()
                .and_then(|value| value.parse().ok())
                .or(active_session_id);
            if !suppress_ui {
                let _ = state.emit_event(&app, event_session_id, kind, payload);
            }
            if let Some(segment) = segment_update {
                if let Ok(payload) = serde_json::to_value(segment) {
                    let _ = state.emit_event(
                        &app,
                        event_session_id,
                        UiEventKind::SegmentCommitted,
                        payload,
                    );
                }
            }
            if let Some(session_id) = event_session_id {
                if persist_tx.send((session_id, persist_event)).await.is_err() {
                    break;
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

/// Start the session's audio source. `planned` is the format the sidecar
/// session was set up for; a source that opens in another format (the
/// output device's mix format changed in between) is refused, because the
/// sidecar cannot switch formats mid-session.
async fn start_audio_capture(
    app: &AppHandle,
    state: &RuntimeState,
    request: &StartSessionRequest,
    planned: CaptureFormat,
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
    let actual = capture.format();
    if actual != planned {
        let _ = capture.stop();
        return Err(format!(
            "The audio device changed its format ({} Hz, {} channels instead of {} Hz, {} channels); stop and start the session again",
            actual.sample_rate_hz, actual.channels, planned.sample_rate_hz, planned.channels
        ));
    }
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
                if let Err(error) = state.supervisor().send_audio(packet).await {
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
                    return;
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
                            .supervisor()
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

/// Stop and release the audio source. Best effort: a source whose device
/// failed may not stop cleanly, and that must never keep a session from
/// finishing.
async fn stop_audio_capture(state: &RuntimeState) {
    let capture = state.audio.lock().await.take();
    if let Some(mut capture) = capture {
        if let Err(error) = capture.stop() {
            eprintln!("audio source did not stop cleanly: {error}");
            state.log_desktop_event(&format!("audio_stop_failed error={error}"));
        }
    }
}

/// Replace an audio source that failed (its device was removed, or the
/// default output of a loopback capture changed) with a new one before the
/// session resumes. The new source must deliver the session's format.
async fn restart_failed_audio_capture(app: &AppHandle, state: &RuntimeState) -> Result<(), String> {
    let planned = match state.audio.lock().await.as_ref() {
        Some(capture) if capture.is_invalidated() => capture.format(),
        _ => return Ok(()),
    };
    stop_audio_capture(state).await;
    let config = state
        .snapshot()?
        .config
        .ok_or_else(|| "missing session config".to_string())?;
    start_audio_capture(app, state, &config, planned).await
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
                stop_audio_capture(&state).await;
                if let Some(session_id) = stopping.session_id {
                    let mut receiver = state.supervisor().subscribe();
                    if state
                        .supervisor()
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
        stop_audio_capture(&state).await;
    }
    state.supervisor().shutdown().await;
    if let Ok(runtimes) = state.local_runtimes() {
        runtimes.shutdown().await;
    }
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
fn get_runtime_preferences(state: State<'_, RuntimeState>) -> Result<RuntimePreferences, String> {
    Ok(state
        .runtime_preferences
        .lock()
        .map_err(|_| "runtime preferences lock poisoned".to_string())?
        .clone())
}

#[tauri::command]
async fn update_runtime_preferences(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    mut preferences: RuntimePreferences,
) -> Result<RuntimePreferences, String> {
    preferences.providers = validate_provider_settings(&preferences.providers)?;
    preferences.assistant = preferences.assistant.normalized(catalog())?;
    let previous = {
        let mut guard = state
            .runtime_preferences
            .lock()
            .map_err(|_| "runtime preferences lock poisoned".to_string())?;
        // Only `set_gpu_acceleration` changes it (it also switches runtimes).
        preferences.gpu_acceleration = guard.gpu_acceleration;
        std::mem::replace(&mut *guard, preferences.clone())
    };
    state.persist_preferences()?;
    if previous.assistant.transcript_upload_allowed
        && !preferences.assistant.transcript_upload_allowed
    {
        // Consent withdrawn: stop every job that sends transcripts or files.
        state.assistant.cancel_uploading_jobs();
    }
    if preferences.providers != previous.providers {
        state.invalidate_sidecar_environment().await;
    }
    if preferences.preload_local_models && !previous.preload_local_models {
        warm_local_runtimes(app);
    }
    Ok(preferences)
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

/// Start the sidecar, plan the route for `payload` and bring every local model
/// service the plan needs to a healthy state, emitting `backend_health` events
/// so the UI can show model startup. Idempotent: healthy services return
/// immediately, so a warm-up at launch and a later Start share the work.
async fn prepare_inference_runtimes(
    app: &AppHandle,
    state: &RuntimeState,
    session_id: uuid::Uuid,
    payload: &Value,
) -> Result<Vec<String>, String> {
    state.ensure_sidecar(Some(app), SidecarUser::LiveSession).await?;
    let mut receiver = state.supervisor().subscribe();
    state
        .supervisor()
        .send_command(SidecarCommand::PlanSession(payload.clone()))
        .await
        .map_err(|error| error.to_string())?;
    let services_to_start = tokio::time::timeout(std::time::Duration::from_secs(20), async {
        loop {
            match receiver.recv().await {
                Ok(SidecarEvent::RoutePlan {
                    session_id: planned,
                    services_to_start,
                    ..
                }) if planned == session_id => return Ok(services_to_start),
                Ok(SidecarEvent::Error { message, .. }) => return Err(message),
                Ok(_) => continue,
                Err(error) => return Err(error.to_string()),
            }
        }
    })
    .await
    .map_err(|_| "inference route planning timed out".to_string())??;
    // No session uses the services yet: apply a GPU runtime change that
    // arrived while one did.
    state.apply_gpu_runtime().await?;
    for service in &services_to_start {
        if let Some(model_id) = state.changing_model_for(service) {
            let name = state
                .models()?
                .status(&model_id)
                .map(|status| status.display_name)
                .unwrap_or(model_id);
            return Err(format!(
                "{name} is being installed or removed; start again when it has finished"
            ));
        }
        state.emit_event(
            app,
            Some(session_id),
            UiEventKind::BackendHealth,
            json!({"service": service, "state": "starting"}),
        )?;
        let runtimes = state.local_runtimes()?;
        let accelerated = runtimes.uses_gpu(service);
        let started = runtimes.ensure_service(service).await;
        if accelerated {
            if let Some(reason) = runtimes.gpu_fallback_for(service) {
                state.log_desktop_event(&format!("gpu_fallback {reason}"));
                let _ = state.emit_event(
                    app,
                    Some(session_id),
                    UiEventKind::Error,
                    json!({
                        "code": "gpu_fallback",
                        "message": format!(
                            "GPU acceleration could not start for this model, so it runs on the CPU for now. {reason}"
                        ),
                        "recoverable": true
                    }),
                );
            }
        }
        started.map_err(|error| error.to_string())?;
        state.emit_event(
            app,
            Some(session_id),
            UiEventKind::BackendHealth,
            json!({"service": service, "state": "connected"}),
        )?;
    }
    Ok(services_to_start)
}

/// Pre-warm the inference sidecar and local models for the saved session
/// defaults so the first Start is immediate. Runs in the background; failures
/// are reported as a recoverable warning and never block the UI.
fn warm_local_runtimes(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let state = app.state::<RuntimeState>();
        if state
            .warmup_in_progress
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let result = async {
            let enabled = state
                .runtime_preferences
                .lock()
                .map_err(|_| "runtime preferences lock poisoned".to_string())?
                .preload_local_models;
            if !enabled
                || !state.onboarding_complete.load(Ordering::Acquire)
                || state.updating.load(Ordering::Acquire)
            {
                return Ok(Vec::new());
            }
            if state.snapshot()?.phase != SessionPhase::Idle {
                return Ok(Vec::new());
            }
            let defaults = state
                .session_defaults
                .lock()
                .map_err(|_| "session defaults lock poisoned".to_string())?
                .clone();
            let warmup_id = uuid::Uuid::new_v4();
            let mut payload = serde_json::to_value(&defaults).map_err(|error| error.to_string())?;
            let object = payload
                .as_object_mut()
                .ok_or_else(|| "invalid session defaults".to_string())?;
            object.insert("session_id".into(), json!(warmup_id));
            object.insert("sample_rate_hz".into(), json!(48_000));
            object.insert("channels".into(), json!(1));
            state.emit_event(
                &app,
                None,
                UiEventKind::BackendHealth,
                json!({"service": "warmup", "state": "starting"}),
            )?;
            let services = prepare_inference_runtimes(&app, &state, warmup_id, &payload).await?;
            state.emit_event(
                &app,
                None,
                UiEventKind::BackendHealth,
                json!({"service": "warmup", "state": "connected", "services": services}),
            )?;
            Ok::<Vec<String>, String>(services)
        }
        .await;
        state.warmup_in_progress.store(false, Ordering::Release);
        if let Err(error) = result {
            eprintln!("local runtime warm-up skipped: {error}");
            let _ = state.emit_event(
                &app,
                None,
                UiEventKind::BackendHealth,
                json!({"service": "warmup", "state": "unavailable", "message": error}),
            );
        }
    });
}

#[tauri::command]
async fn start_session(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    request: StartSessionRequest,
) -> Result<SessionSnapshot, String> {
    let starting = {
        let mut core = state
            .core
            .lock()
            .map_err(|_| "app core lock poisoned".to_string())?;
        // An update sets `updating` under this lock, so no session can
        // start once it has checked that none runs.
        if state.updating.load(Ordering::Acquire) {
            return Err(updater::UPDATE_IN_PROGRESS.into());
        }
        core.start(request).map_err(|error| error.to_string())?
    };
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
        let session_id = starting
            .session_id
            .ok_or_else(|| "missing session id".to_string())?;
        let mut payload = serde_json::to_value(session_config).map_err(|error| error.to_string())?;
        let object = payload
            .as_object_mut()
            .ok_or_else(|| "invalid session config".to_string())?;
        object.insert("session_id".into(), json!(session_id));
        object.insert(
            "sample_rate_hz".into(),
            json!(capture_format.sample_rate_hz),
        );
        object.insert("channels".into(), json!(capture_format.channels));
        if state.model_changing("qwen3-forced-aligner-0.6b") {
            // The aligner's files are being replaced or deleted.
            object.insert("alignment_enabled".into(), json!(false));
        }
        let mut receiver = state.supervisor().subscribe();
        prepare_inference_runtimes(&app, &state, session_id, &payload).await?;
        state
            .supervisor()
            .send_command(SidecarCommand::StartSession(payload))
            .await
            .map_err(|error| error.to_string())?;
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(20), receiver.recv()).await {
                Ok(Ok(SidecarEvent::Ready { session_id, route }))
                    if Some(session_id) == starting.session_id =>
                {
                    start_audio_capture(
                        &app,
                        &state,
                        starting.config.as_ref().unwrap(),
                        capture_format,
                    )
                    .await?;
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
                    context: config.session_context.clone(),
                })
                .await
                .map_err(|error| error.to_string())?;
            state.emit_snapshot(&app, &snapshot)?;
            Ok(snapshot)
        }
        Err(error) => {
            stop_audio_capture(&state).await;
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
        .supervisor()
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
    restart_failed_audio_capture(&app, &state).await?;
    state
        .supervisor()
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
    stop_audio_capture(&state).await;
    let mut warnings = Vec::new();
    if let Some(session_id) = stopping.session_id {
        let mut receiver = state.supervisor().subscribe();
        match state
            .supervisor()
            .send_command(SidecarCommand::FinishSession { session_id })
            .await
        {
            Ok(()) => {
                let finished = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    async move {
                        while let Ok(event) = receiver.recv().await {
                            if matches!(event, SidecarEvent::SessionFinished { session_id: finished } if finished == session_id) {
                                return true;
                            }
                        }
                        false
                    },
                )
                .await
                .unwrap_or(false);
                if !finished {
                    warnings.push("Inference backend did not acknowledge session finish".into());
                }
            }
            Err(error) => warnings.push(format!(
                "Inference backend was unavailable during Stop: {error}"
            )),
        }
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
            .complete_session(session_id, &warnings)
            .await
            .map_err(|error| error.to_string())?;
        if let Ok(mut throttle) = state.persistence_throttle.lock() {
            throttle.finish(session_id);
        }
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

/// Tell every window that a History entry changed (`history_changed`,
/// reason `title | notes | renamed | deleted`).
fn emit_history_changed(app: &AppHandle, state: &RuntimeState, session_id: uuid::Uuid, reason: &str) {
    let _ = state.emit_event(
        app,
        Some(session_id),
        UiEventKind::HistoryChanged,
        json!({"session_id": session_id, "reason": reason}),
    );
}

#[tauri::command]
async fn history_rename(
    app: AppHandle,
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
        .map_err(|error| error.to_string())?;
    emit_history_changed(&app, &state, id, "renamed");
    Ok(())
}

#[tauri::command]
async fn history_delete(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    session_id: String,
) -> Result<(), String> {
    let id = session_id
        .parse()
        .map_err(|_| "invalid session id".to_string())?;
    // Jobs for a deleted session could only fail when they save.
    state.assistant.cancel_session_jobs(id);
    state
        .store()?
        .delete(id)
        .await
        .map_err(|error| error.to_string())?;
    emit_history_changed(&app, &state, id, "deleted");
    Ok(())
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
    // The window is often minimized or behind slides during a lecture, and
    // lecture halls often mean battery power: keep Windows from throttling
    // the process that captures audio and drives the live captions.
    if let Err(error) = process_support::disable_power_throttling_for_current_process() {
        eprintln!("could not disable power throttling: {error}");
    }
    // WebKitGTK's DMA-BUF renderer shows blank windows with several GPU
    // drivers (notably NVIDIA); the plain renderer works everywhere.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Driven from Rust only (see `updater`); the webview has no updater
        // permission.
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![
            get_app_snapshot,
            onboarding_status,
            complete_onboarding,
            audio_permission_status,
            request_audio_permission,
            test_audio_input,
            list_providers,
            credential_status,
            set_credentials,
            clear_credentials,
            update_provider_settings,
            probe_cloud,
            logs_directory,
            list_models,
            install_model,
            verify_model,
            delete_model,
            gpu_acceleration_status,
            install_gpu_pack,
            remove_gpu_pack,
            set_gpu_acceleration,
            list_audio_devices,
            get_caption_preferences,
            get_session_defaults,
            update_session_defaults,
            get_runtime_preferences,
            update_runtime_preferences,
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
            assistant::assistant_status,
            assistant::assistant_probe,
            assistant::get_session_notes,
            assistant::pick_note_attachments,
            assistant::create_session_notes,
            assistant::cancel_session_notes,
            assistant::generate_session_title,
            assistant::import_context_files,
            updater::get_update_status,
            updater::check_for_update,
            updater::install_update,
        ])
        .setup(|app| {
            let state = app.state::<RuntimeState>();
            // Before anything reads the secure store (the desktop log's
            // redaction does).
            state
                .credentials
                .set_service(keychain_service(&app.config().identifier));
            // Local (not roaming) data: models and runtimes are gigabytes.
            // Identical to the app data directory on macOS and Linux.
            let app_data_directory = app.path().app_local_data_dir()?;
            let resource_directory = app.path().resource_dir()?;
            let logs_directory = app_data_directory.join("logs");
            let mut launch = SidecarLaunchConfig::desktop(
                project_root(),
                find_bundled_sidecar(
                    Some(&resource_directory),
                    executable_directory().as_deref(),
                ),
            );
            launch.log_path = Some(logs_directory.join(SIDECAR_LOG_FILE));
            state
                .supervisor
                .set(InferenceSupervisor::new(launch))
                .map_err(|_| std::io::Error::other("inference supervisor already initialized"))?;
            state
                .logs_directory
                .set(logs_directory)
                .map_err(|_| std::io::Error::other("logs directory already initialized"))?;
            let database_path = app_data_directory.join("history.sqlite");
            let store = tauri::async_runtime::block_on(TranscriptStore::open(database_path))
                .map_err(std::io::Error::other)?;
            tauri::async_runtime::block_on(store.recover_active_sessions())
                .map_err(std::io::Error::other)?;
            state
                .store
                .set(store)
                .map_err(|_| std::io::Error::other("store already initialized"))?;
            let preferences_path = app_data_directory.join("preferences.json");
            let model_root = app_data_directory.join("models");
            state
                .models
                .set(ModelManager::new(model_root.clone()))
                .map_err(|_| std::io::Error::other("model manager already initialized"))?;
            let (qwen_reservation, hymt_reservation) = reserve_local_runtime_ports()?;
            let qwen_port = qwen_reservation.local_addr()?.port();
            let hymt_port = hymt_reservation.local_addr()?.port();
            state
                .local_runtimes
                .set(LocalRuntimeManager::new_with_reserved_ports(
                    local_runtime_layout(
                        model_root,
                        &app_data_directory,
                        &resource_directory,
                        resolve_sidecar_executable(Some(&resource_directory)),
                        qwen_port,
                        hymt_port,
                    ),
                    qwen_reservation,
                    hymt_reservation,
                ))
                .map_err(|_| std::io::Error::other("local runtime manager already initialized"))?;
            state
                .gpu_pack
                .set(GpuPackManager::new(
                    app_data_directory.join("runtimes"),
                    env!("CARGO_PKG_VERSION"),
                ))
                .map_err(|_| std::io::Error::other("GPU pack manager already initialized"))?;
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
            if let Ok(mut runtime) = state.runtime_preferences.lock() {
                *runtime = preferences.runtime;
            }
            // Before any warm-up starts a service. (No desktop.log line here:
            // its redaction reads the secure store, which may block setup.)
            let gpu_runtime = state.desired_gpu_runtime();
            if gpu_runtime.is_some() {
                eprintln!("local model services use the GPU acceleration pack");
            }
            tauri::async_runtime::block_on(
                state
                    .local_runtimes()
                    .map_err(std::io::Error::other)?
                    .select_gpu_runtime(gpu_runtime),
            );
            if std::env::var("ECHOLINGO_PRELOAD_LOCAL_MODELS").as_deref() != Ok("0") {
                warm_local_runtimes(app.handle().clone());
            }
            updater::schedule_launch_check(app.handle().clone());
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
                    let app = window.app_handle().clone();
                    shutdown_application(&app).await;
                    let _ = window.destroy();
                    // Only macOS apps outlive their last window; elsewhere the
                    // hidden caption window would keep the process running.
                    #[cfg(not(target_os = "macos"))]
                    app.exit(0);
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run EchoLingo desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct SoakCounts {
        transcripts: u64,
        translations: u64,
        metrics: u64,
        alignments: u64,
        errors: u64,
        ui_updates: u64,
        max_dropped_audio_ms: f64,
    }

    async fn consume_soak_event(
        state: &RuntimeState,
        fallback_session_id: uuid::Uuid,
        event: &SidecarEvent,
        counts: &mut SoakCounts,
    ) -> Result<(), String> {
        let payload = match event {
            SidecarEvent::Transcript(payload) => {
                counts.transcripts += 1;
                counts.ui_updates += 1;
                payload
            }
            SidecarEvent::Translation(payload) => {
                counts.translations += 1;
                counts.ui_updates += 1;
                payload
            }
            SidecarEvent::Metrics(payload) => {
                counts.metrics += 1;
                counts.ui_updates += 1;
                counts.max_dropped_audio_ms = counts
                    .max_dropped_audio_ms
                    .max(payload["dropped_audio_ms"].as_f64().unwrap_or(0.0));
                payload
            }
            SidecarEvent::AlignmentUpdate(payload) => {
                counts.alignments += 1;
                counts.ui_updates += 1;
                payload
            }
            SidecarEvent::Error { .. } => {
                counts.errors += 1;
                return Ok(());
            }
            _ => return Ok(()),
        };
        let event_session_id = payload["session_id"]
            .as_str()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback_session_id);
        persist_sidecar_event(state, event_session_id, event).await
    }

    #[cfg(not(unix))]
    fn resident_set_bytes() -> Option<u64> {
        None
    }

    #[cfg(unix)]
    fn resident_set_bytes() -> Option<u64> {
        let output = std::process::Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let kib = String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kib * 1024)
    }

    fn mock_session_payload(session_id: uuid::Uuid, alignment_enabled: bool) -> Value {
        json!({
            "session_id": session_id,
            "source_language": "en",
            "target_language": "zh",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "audio_profile": "raw",
            "inference_mode": "auto",
            "asr_provider": "mock",
            "translation_provider": "mock",
            "alignment_enabled": alignment_enabled,
            "alignment_provider": if alignment_enabled { "mock" } else { "none" },
            "privacy": {
                "audio_upload_allowed": false,
                "transcript_upload_allowed": false
            }
        })
    }

    async fn create_soak_history(
        state: &RuntimeState,
        session_id: uuid::Uuid,
        route: &Value,
        ordinal: usize,
    ) -> Result<(), String> {
        state
            .store()?
            .create_session(&SessionDraft {
                id: session_id,
                title: format!("Desktop soak session {ordinal}"),
                source_language: "en".into(),
                target_language: "zh".into(),
                audio_source: "synthetic".into(),
                audio_profile: "lecture".into(),
                inference_mode: "auto".into(),
                asr_backend: "mock".into(),
                translation_backend: "mock".into(),
                route_reason: "deterministic release soak".into(),
                privacy: json!({
                    "audio_upload_allowed": false,
                    "transcript_upload_allowed": false
                }),
                model_config: route.clone(),
                context: String::new(),
            })
            .await
            .map_err(|error| error.to_string())?;
        Ok(())
    }

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
    fn desktop_preferences_accept_every_selectable_catalog_provider() {
        for spec in catalog().specs(ProviderKind::Asr) {
            let mut defaults = StartSessionRequest::default();
            defaults.asr_provider = spec.id.clone();
            assert_eq!(
                validate_session_defaults(&defaults).is_ok(),
                spec.selectable,
                "{}",
                spec.id
            );
        }
        for spec in catalog().specs(ProviderKind::Translation) {
            let mut defaults = StartSessionRequest::default();
            defaults.translation_provider = spec.id.clone();
            assert_eq!(
                validate_session_defaults(&defaults).is_ok(),
                spec.selectable,
                "{}",
                spec.id
            );
        }
        let mut defaults = StartSessionRequest::default();
        defaults.asr_provider = "whisper_cloud".into();
        assert!(validate_session_defaults(&defaults).is_err());
        let mut defaults = StartSessionRequest::default();
        defaults.cloud_asr_preference = "qwen_local".into();
        assert!(validate_session_defaults(&defaults).is_err());
        let mut defaults = StartSessionRequest::default();
        defaults.cloud_asr_preference = "deepgram".into();
        defaults.cloud_translation_preference = "deepl".into();
        assert!(validate_session_defaults(&defaults).is_ok());
    }

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn sidecar_environment_uses_setting_defaults_and_preferences() {
        let none = |_: &str| None;
        let empty = ProviderSettings::new();
        let values = compose_sidecar_environment(catalog(), none, none, &empty);
        assert_eq!(
            values.get("ECHOLINGO_QWEN_REGION").map(String::as_str),
            Some("singapore")
        );
        assert_eq!(
            values.get("ECHOLINGO_DEEPL_TIER").map(String::as_str),
            Some("free")
        );
        assert_eq!(
            values.get("ECHOLINGO_OPENAI_CHAT_MODEL").map(String::as_str),
            Some("gpt-4o-mini")
        );
        // Empty defaults (custom endpoint base URL) are never exported.
        assert!(!values.contains_key("ECHOLINGO_CUSTOM_OPENAI_BASE_URL"));
        // No credential without a source.
        assert!(!values.contains_key("DASHSCOPE_API_KEY"));
        assert!(!values.contains_key("DASHSCOPE_WORKSPACE_ID"));

        let mut providers = ProviderSettings::new();
        providers.insert(
            "dashscope".into(),
            HashMap::from([("region".to_string(), "beijing".to_string())]),
        );
        let values = compose_sidecar_environment(catalog(), none, none, &providers);
        assert_eq!(
            values.get("ECHOLINGO_QWEN_REGION").map(String::as_str),
            Some("beijing")
        );

        // An invalid stored select value falls back to the default.
        providers.insert(
            "dashscope".into(),
            HashMap::from([("region".to_string(), "mars".to_string())]),
        );
        let values = compose_sidecar_environment(catalog(), none, none, &providers);
        assert_eq!(
            values.get("ECHOLINGO_QWEN_REGION").map(String::as_str),
            Some("singapore")
        );
    }

    #[test]
    fn sidecar_environment_prefers_keychain_over_environment_and_omits_optional_fields() {
        let keychain = |account: &str| match account {
            "dashscope-api-key" => Some("keychain-dashscope-key".to_string()),
            "openai-api-key" => Some("keychain-openai-key".to_string()),
            _ => None,
        };
        let env_pairs = [
            ("DASHSCOPE_API_KEY", "env-dashscope-key"),
            ("DEEPGRAM_API_KEY", "env-deepgram-key"),
            ("ECHOLINGO_QWEN_REGION", "beijing"),
        ];
        let empty = ProviderSettings::new();
        let values =
            compose_sidecar_environment(catalog(), keychain, env_from(&env_pairs), &empty);
        assert_eq!(
            values.get("DASHSCOPE_API_KEY").map(String::as_str),
            Some("keychain-dashscope-key")
        );
        assert_eq!(
            values.get("OPENAI_API_KEY").map(String::as_str),
            Some("keychain-openai-key")
        );
        assert_eq!(
            values.get("DEEPGRAM_API_KEY").map(String::as_str),
            Some("env-deepgram-key")
        );
        // The optional workspace id is omitted when nothing provides it.
        assert!(!values.contains_key("DASHSCOPE_WORKSPACE_ID"));
        // A developer's shell region applies when no preference is stored...
        assert_eq!(
            values.get("ECHOLINGO_QWEN_REGION").map(String::as_str),
            Some("beijing")
        );
        // ...and a stored preference wins over the shell.
        let mut providers = ProviderSettings::new();
        providers.insert(
            "dashscope".into(),
            HashMap::from([("region".to_string(), "singapore".to_string())]),
        );
        let values =
            compose_sidecar_environment(catalog(), keychain, env_from(&env_pairs), &providers);
        assert_eq!(
            values.get("ECHOLINGO_QWEN_REGION").map(String::as_str),
            Some("singapore")
        );
    }

    #[test]
    fn credential_group_status_reports_sources_without_values() {
        let group = catalog().group("dashscope").unwrap();
        let keychain =
            |account: &str| (account == "dashscope-api-key").then(|| "sekrit-key-value".to_string());
        let env_pairs = [("DASHSCOPE_WORKSPACE_ID", "ws-123")];
        let status = group_status(
            group,
            keychain,
            env_from(&env_pairs),
            &ProviderSettings::new(),
        );
        assert_eq!(status.group_id, "dashscope");
        assert_eq!(
            status.fields,
            vec![
                CredentialFieldStatus {
                    key: "api_key".into(),
                    available: true,
                    source: "keychain".into()
                },
                CredentialFieldStatus {
                    key: "workspace_id".into(),
                    available: true,
                    source: "environment".into()
                },
            ]
        );
        assert_eq!(
            status.settings.get("region").map(String::as_str),
            Some("singapore")
        );
        let serialized = serde_json::to_string(&status).unwrap();
        assert!(!serialized.contains("sekrit") && !serialized.contains("ws-123"));
        assert!(status.store_available && status.store_error.is_none());
        let unavailable = status.with_store_failure(Some("Secret Service is not running"));
        let serialized = serde_json::to_value(&unavailable).unwrap();
        assert_eq!(serialized["store_available"], false);
        assert_eq!(serialized["store_error"], "Secret Service is not running");

        let none = |_: &str| None;
        let status = group_status(group, none, none, &ProviderSettings::new());
        assert!(status
            .fields
            .iter()
            .all(|field| !field.available && field.source == "none"));
    }

    #[test]
    fn credential_updates_are_validated_without_echoing_values() {
        let group = catalog().group("dashscope").unwrap();
        let nothing_stored = |_: &str| false;
        let fields = HashMap::from([("api_key".to_string(), "zq7-tiny".to_string()[..5].to_string())]);
        let error = plan_credential_update(group, &fields, nothing_stored).unwrap_err();
        assert!(error.contains("API key") && error.contains("at least 8"));
        assert!(!error.contains("zq7-t"));

        let fields = HashMap::from([("workspace_id".to_string(), "ws-only".to_string())]);
        let error = plan_credential_update(group, &fields, nothing_stored).unwrap_err();
        assert!(error.contains("required") && !error.contains("ws-only"));

        let fields = HashMap::from([("token".to_string(), "x".to_string())]);
        assert!(plan_credential_update(group, &fields, nothing_stored).is_err());

        let fields = HashMap::from([
            ("api_key".to_string(), " sk-valid-key-value ".to_string()),
            ("workspace_id".to_string(), String::new()),
        ]);
        let writes = plan_credential_update(group, &fields, nothing_stored).unwrap();
        assert_eq!(
            writes,
            vec![
                CredentialWrite::Set {
                    account: "dashscope-api-key".into(),
                    value: "sk-valid-key-value".into()
                },
                CredentialWrite::Delete {
                    account: "dashscope-workspace-id".into()
                },
            ]
        );
        assert!(!format!("{writes:?}").contains("sk-valid"));

        // A stored required key may be left out or blank (kept as is).
        let stored = |account: &str| account == "dashscope-api-key";
        let fields = HashMap::from([("workspace_id".to_string(), "workspace-1".to_string())]);
        let writes = plan_credential_update(group, &fields, stored).unwrap();
        assert_eq!(writes.len(), 1);
        let fields = HashMap::from([("api_key".to_string(), String::new())]);
        assert!(plan_credential_update(group, &fields, stored)
            .unwrap()
            .is_empty());

        // Optional-only groups accept an empty payload.
        let custom = catalog().group("custom_openai").unwrap();
        assert!(plan_credential_update(custom, &HashMap::new(), nothing_stored)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn provider_settings_are_validated_against_the_catalog() {
        let dashscope = catalog().group("dashscope").unwrap();
        let ok = normalize_provider_settings(
            dashscope,
            &HashMap::from([("region".to_string(), " beijing ".to_string())]),
        )
        .unwrap();
        assert_eq!(ok.get("region").map(String::as_str), Some("beijing"));
        let error = normalize_provider_settings(
            dashscope,
            &HashMap::from([("region".to_string(), "mars".to_string())]),
        )
        .unwrap_err();
        assert!(error.contains("singapore, beijing"));
        assert!(normalize_provider_settings(
            dashscope,
            &HashMap::from([("model".to_string(), "x".to_string())])
        )
        .is_err());
        let mut providers = ProviderSettings::new();
        providers.insert(
            "openai".into(),
            HashMap::from([("chat_model".to_string(), String::new())]),
        );
        providers.insert(
            "deepl".into(),
            HashMap::from([("tier".to_string(), "pro".to_string())]),
        );
        let normalized = validate_provider_settings(&providers).unwrap();
        assert!(!normalized.contains_key("openai"));
        assert_eq!(normalized["deepl"]["tier"], "pro");
        providers.insert("nope".into(), HashMap::new());
        assert!(validate_provider_settings(&providers).is_err());
    }

    #[test]
    fn runtime_preferences_round_trip_provider_settings_without_secrets() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let mut preferences = DesktopPreferences::default();
        preferences.runtime.providers.insert(
            "dashscope".into(),
            HashMap::from([("region".to_string(), "beijing".to_string())]),
        );
        save_preferences(&path, &preferences).unwrap();
        let loaded = load_preferences(&path);
        assert_eq!(loaded.runtime.providers["dashscope"]["region"], "beijing");
        // Older files without `providers` and corrupt settings both load.
        let legacy: RuntimePreferences =
            serde_json::from_str(r#"{"preload_local_models": false}"#).unwrap();
        assert!(legacy.providers.is_empty() && !legacy.preload_local_models);
        preferences.runtime.providers.insert(
            "dashscope".into(),
            HashMap::from([("region".to_string(), "mars".to_string())]),
        );
        save_preferences(&path, &preferences).unwrap();
        assert!(load_preferences(&path).runtime.providers.is_empty());
    }

    #[test]
    fn route_status_prefers_sidecar_keys_and_falls_back_to_heuristics() {
        let route = route_status(&json!({
            "asr_provider": "deepgram",
            "translation_provider": "deepl",
            "asr_locality": "cloud",
            "asr_model": "nova-3",
            "asr_display_name": "Deepgram streaming (cloud)",
            "translation_locality": "cloud",
            "translation_model": null,
            "translation_display_name": "DeepL (cloud)",
            "status": "cloud",
            "reasons": ["cloud requested"]
        }));
        assert_eq!(route.asr_locality, "cloud");
        assert_eq!(route.asr_model.as_deref(), Some("nova-3"));
        assert_eq!(
            route.asr_display_name.as_deref(),
            Some("Deepgram streaming (cloud)")
        );
        assert_eq!(route.translation_model, None);
        assert_eq!(
            route.translation_display_name.as_deref(),
            Some("DeepL (cloud)")
        );
        assert_eq!(route.translation_health, BackendHealth::Connected);

        let legacy = route_status(&json!({
            "asr_provider": "qwen_local",
            "translation_provider": "qwen_cloud",
            "status": "hybrid",
            "degraded": false
        }));
        assert_eq!(legacy.asr_locality, "local");
        assert_eq!(legacy.asr_model.as_deref(), Some("qwen3-asr-0.6b"));
        assert_eq!(
            legacy.asr_display_name.as_deref(),
            Some("Qwen3-ASR (local)")
        );
        assert_eq!(legacy.translation_locality, "cloud");
        assert_eq!(legacy.translation_model.as_deref(), Some("qwen-mt-flash"));
        assert_eq!(
            legacy.translation_display_name.as_deref(),
            Some("Qwen-MT (cloud)")
        );

        let unknown = route_status(&json!({
            "asr_provider": "mystery_cloud",
            "translation_provider": "none"
        }));
        assert_eq!(unknown.asr_locality, "cloud");
        assert_eq!(unknown.asr_display_name, None);
        assert_eq!(unknown.translation_health, BackendHealth::Unavailable);
    }

    #[test]
    fn packaged_sidecar_lookup_prefers_the_onedir_resource() {
        let directory = tempfile::tempdir().unwrap();
        let resources = directory.path().join("resources");
        let executables = directory.path().join("bin");
        let name = format!("echolingo-sidecar{}", std::env::consts::EXE_SUFFIX);
        let onedir = resources.join("sidecar").join(&name);
        let adjacent = executables.join(&name);
        assert_eq!(
            bundled_sidecar_candidates(Some(&resources), Some(&executables)),
            vec![onedir.clone(), adjacent.clone()]
        );
        assert_eq!(find_bundled_sidecar(Some(&resources), Some(&executables)), None);
        std::fs::create_dir_all(&executables).unwrap();
        std::fs::write(&adjacent, b"onefile").unwrap();
        assert_eq!(
            find_bundled_sidecar(Some(&resources), Some(&executables)),
            Some(adjacent.clone())
        );
        std::fs::create_dir_all(onedir.parent().unwrap()).unwrap();
        std::fs::write(&onedir, b"onedir").unwrap();
        assert_eq!(
            find_bundled_sidecar(Some(&resources), Some(&executables)),
            Some(onedir)
        );
        assert_eq!(find_bundled_sidecar(None, Some(&executables)), Some(adjacent));
    }

    #[test]
    fn local_runtime_layout_uses_the_resolved_sidecar_and_bundled_llama() {
        let directory = tempfile::tempdir().unwrap();
        let llama = directory
            .path()
            .join("runtimes")
            .join("llama.cpp")
            .join(format!("llama-server{}", std::env::consts::EXE_SUFFIX));
        std::fs::create_dir_all(llama.parent().unwrap()).unwrap();
        std::fs::write(&llama, b"llama").unwrap();
        let sidecar = directory.path().join("sidecar").join("echolingo-sidecar");
        let layout = local_runtime_layout(
            directory.path().join("models"),
            directory.path(),
            directory.path(),
            Some(sidecar.clone()),
            1,
            2,
        );
        if std::env::var_os("ECHOLINGO_QWEN_ASR_COMMAND").is_none() {
            assert_eq!(layout.qwen_command.executable, sidecar);
            assert_eq!(layout.qwen_command.args, ["qwen-asr-server"]);
        }
        assert_eq!(layout.worker_wrapper.as_ref(), Some(&sidecar));
        if std::env::var_os("ECHOLINGO_LLAMA_SERVER").is_none() {
            assert_eq!(layout.llama_server, Some(llama));
        }
        assert_eq!(layout.log_root, directory.path().join("logs"));
    }

    #[test]
    fn gpu_acceleration_defaults_on_and_survives_older_files() {
        let legacy: RuntimePreferences =
            serde_json::from_str(r#"{"preload_local_models": false}"#).unwrap();
        assert!(legacy.gpu_acceleration);
        let off: RuntimePreferences =
            serde_json::from_str(r#"{"gpu_acceleration": false}"#).unwrap();
        assert!(!off.gpu_acceleration && off.preload_local_models);
        let serialized = serde_json::to_value(RuntimePreferences::default()).unwrap();
        assert_eq!(serialized["gpu_acceleration"], true);
    }

    #[test]
    fn update_checks_at_launch_default_on_and_survive_older_files() {
        let legacy: RuntimePreferences =
            serde_json::from_str(r#"{"preload_local_models": false, "gpu_acceleration": false}"#)
                .unwrap();
        assert!(legacy.check_updates_at_launch);
        let off: RuntimePreferences =
            serde_json::from_str(r#"{"check_updates_at_launch": false}"#).unwrap();
        assert!(!off.check_updates_at_launch && off.preload_local_models);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let mut preferences = DesktopPreferences::default();
        preferences.runtime.check_updates_at_launch = false;
        save_preferences(&path, &preferences).unwrap();
        assert!(!load_preferences(&path).runtime.check_updates_at_launch);
    }

    #[test]
    fn keychain_service_keeps_the_production_name_and_isolates_test_builds() {
        // Every earlier release stored keys under this service name.
        const PRODUCTION_SERVICE: &str = "app.echolingo.desktop";
        let config: Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let identifier = config["identifier"].as_str().unwrap();
        assert_eq!(keychain_service(identifier), PRODUCTION_SERVICE);
        let test_build = format!("{identifier}{}", updater::UPDATE_TEST_IDENTIFIER_SUFFIX);
        assert_ne!(keychain_service(&test_build), PRODUCTION_SERVICE);
        assert_eq!(
            keychain_service(&test_build),
            "app.echolingo.desktop.updatetest"
        );

        // Nothing reaches a secure store before `setup` names the service.
        let store = CredentialStore::default();
        assert!(store.entry("dashscope_api_key").is_err());
        store.set_service(keychain_service(identifier));
        store.set_service("another".into());
        assert_eq!(store.service.get().map(String::as_str), Some(PRODUCTION_SERVICE));
    }

    #[test]
    fn model_changes_stop_the_services_that_hold_their_files() {
        assert_eq!(services_using_model("qwen3-asr-0.6b"), ["qwen_asr"]);
        assert_eq!(services_using_model("hymt2-1.8b"), ["hymt"]);
        assert!(services_using_model("qwen3-forced-aligner-0.6b").is_empty());
        for spec in runtime_manager::catalog() {
            let covered = !services_using_model(&spec.id).is_empty()
                || spec.role == runtime_manager::ModelRole::Alignment;
            assert!(covered, "{} has no owner to stop", spec.id);
        }
    }

    #[tokio::test]
    async fn model_changes_are_exclusive_and_hold_back_the_services_that_need_them() {
        let directory = tempfile::tempdir().unwrap();
        let state = RuntimeState::default();
        let model_root = directory.path().join("models");
        assert!(state.models.set(ModelManager::new(model_root.clone())).is_ok());
        assert!(state
            .local_runtimes
            .set(LocalRuntimeManager::new(local_runtime_layout(
                model_root,
                directory.path(),
                directory.path(),
                None,
                1,
                2,
            )))
            .is_ok());

        let change = begin_model_change(&state, "hymt2-1.8b").await.unwrap();
        assert_eq!(state.changing_model_for("hymt").as_deref(), Some("hymt2-1.8b"));
        assert_eq!(state.changing_model_for("qwen_asr"), None);
        assert!(begin_model_change(&state, "hymt2-1.8b")
            .await
            .err()
            .unwrap()
            .contains("already being installed or removed"));
        drop(change);
        assert_eq!(state.changing_model_for("hymt"), None);

        state.warmup_in_progress.store(true, Ordering::Release);
        assert!(begin_model_change(&state, "hymt2-1.8b")
            .await
            .err()
            .unwrap()
            .contains("loading in the background"));
        state.warmup_in_progress.store(false, Ordering::Release);

        // Forced alignment of a finished session keeps the aligner busy.
        let aligner = "qwen3-forced-aligner-0.6b";
        state.note_alignment_activity(Some(uuid::Uuid::new_v4()));
        assert!(state.alignment_running());
        assert!(begin_model_change(&state, aligner)
            .await
            .err()
            .unwrap()
            .contains("still being aligned"));
        assert!(!state.model_changing(aligner));
        state.note_alignment_activity(None);
        assert!(!state.alignment_running());
        let change = begin_model_change(&state, aligner).await.unwrap();
        assert!(state.model_changing(aligner));
        drop(change);

        // An update in progress refuses model changes and leaves none
        // registered.
        state.updating.store(true, Ordering::Release);
        assert_eq!(
            begin_model_change(&state, "hymt2-1.8b").await.err().unwrap(),
            updater::UPDATE_IN_PROGRESS
        );
        assert!(!state.model_changing("hymt2-1.8b"));
        state.updating.store(false, Ordering::Release);

        state
            .core
            .lock()
            .unwrap()
            .start(StartSessionRequest::default())
            .unwrap();
        assert_eq!(
            begin_model_change(&state, "hymt2-1.8b").await.err().unwrap(),
            MODEL_SESSION_BUSY
        );
    }

    #[test]
    fn redaction_scrubs_known_secrets_only() {
        let secrets = ["sk-live-abcdef", "abc"];
        assert_eq!(
            redact_secrets("auth failed for sk-live-abcdef (abc)", secrets),
            "auth failed for [redacted] (abc)"
        );
        assert_eq!(redact_secrets("clean", secrets), "clean");
    }

    #[test]
    fn persistence_throttle_bounds_hot_path_writes() {
        let session = uuid::Uuid::new_v4();
        let mut throttle = PersistenceThrottle::default();
        assert!(throttle.allow_metrics(session, 0.0));
        assert!(!throttle.allow_metrics(session, 10.0));
        assert!(throttle.allow_metrics(session, 1_000.0));
        assert!(throttle.allow_revision(session, "source", "partial"));
        assert!(!throttle.allow_revision(session, "source", "partial"));
        assert!(throttle.allow_revision(session, "source", "stable"));
        throttle.finish(session);
        assert!(throttle.allow_metrics(session, 20.0));
    }

    #[test]
    fn every_sidecar_launch_gets_the_complete_environment() {
        let secrets = HashMap::from([("DASHSCOPE_API_KEY".to_string(), "sk-test-1".to_string())]);
        let capabilities = HashMap::from([(
            "ECHOLINGO_LOCAL_QWEN_URL".to_string(),
            "http://127.0.0.1:4100".to_string(),
        )]);
        let data = PathBuf::from("/data").join("EchoLingo");
        let root = data.join("models");
        let values = compose_full_sidecar_environment(secrets, &root, capabilities).unwrap();
        assert_eq!(values["DASHSCOPE_API_KEY"], "sk-test-1");
        assert_eq!(Path::new(&values["ECHOLINGO_MODEL_ROOT"]), root);
        assert_eq!(
            Path::new(&values["ECHOLINGO_ALIGNMENT_SPOOL_ROOT"]),
            data.join("alignment-spool")
        );
        assert_eq!(values["ECHOLINGO_LOCAL_QWEN_URL"], "http://127.0.0.1:4100");
        assert!(compose_full_sidecar_environment(
            HashMap::new(),
            std::path::Path::new("/"),
            HashMap::new()
        )
        .is_err());
    }

    #[test]
    fn missing_credentials_name_required_keys_and_an_empty_endpoint() {
        let dashscope = catalog().group("dashscope").unwrap();
        let none = |_: &str| None;
        let status = group_status(dashscope, none, none, &ProviderSettings::new());
        let missing = missing_credential_labels(dashscope, &status);
        assert_eq!(missing, vec![dashscope.field("api_key").unwrap().label.clone()]);
        let keychain = |account: &str| (account == "dashscope-api-key").then(|| "sk-key-123".into());
        let status = group_status(dashscope, keychain, none, &ProviderSettings::new());
        assert!(missing_credential_labels(dashscope, &status).is_empty());

        let custom = catalog().group("custom_openai").unwrap();
        let status = group_status(custom, none, none, &ProviderSettings::new());
        assert_eq!(
            missing_credential_labels(custom, &status),
            vec![custom.setting("base_url").unwrap().label.clone()]
        );
        let mut providers = ProviderSettings::new();
        providers.insert(
            "custom_openai".into(),
            HashMap::from([("base_url".to_string(), "http://127.0.0.1:8080/v1".to_string())]),
        );
        let status = group_status(custom, none, none, &providers);
        assert!(missing_credential_labels(custom, &status).is_empty());
    }

    #[test]
    fn session_defaults_bound_the_context_and_glossary() {
        let mut defaults = StartSessionRequest::default();
        defaults.session_context = "Topic: early vision\nsaccade = 扫视".into();
        defaults.glossary = "Julesz\ntexton = 纹理基元".into();
        assert!(validate_session_defaults(&defaults).is_ok());
        defaults.session_context = "x".repeat(app_core::SESSION_CONTEXT_MAX_CHARS + 1);
        assert!(validate_session_defaults(&defaults)
            .unwrap_err()
            .contains("session context"));
        defaults.session_context.clear();
        defaults.glossary = "x".repeat(app_core::GLOSSARY_MAX_CHARS + 1);
        assert!(validate_session_defaults(&defaults)
            .unwrap_err()
            .contains("glossary"));

        // A hand-edited file with over-long text keeps everything else.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let mut preferences = DesktopPreferences::default();
        preferences.onboarding_complete = true;
        preferences.session.target_language = "ja".into();
        preferences.session.glossary = defaults.glossary.clone();
        save_preferences(&path, &preferences).unwrap();
        let loaded = load_preferences(&path);
        assert!(loaded.onboarding_complete);
        assert_eq!(loaded.session.target_language, "ja");
        assert!(loaded.session.glossary.is_empty());
    }

    #[test]
    fn runtime_preferences_keep_a_valid_assistant_and_reset_an_unknown_one() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let mut preferences = DesktopPreferences::default();
        assert_eq!(preferences.runtime.assistant, AssistantPreferences::default());
        preferences.runtime.assistant = AssistantPreferences {
            provider_group: "dashscope".into(),
            model: "qwen-max".into(),
            transcript_upload_allowed: true,
            auto_title: false,
        };
        preferences.session.session_context = "Topic: textures".into();
        save_preferences(&path, &preferences).unwrap();
        let loaded = load_preferences(&path);
        assert_eq!(loaded.runtime.assistant, preferences.runtime.assistant);
        assert_eq!(loaded.session.session_context, "Topic: textures");

        // Files written before AI notes existed have no `assistant`.
        let legacy: RuntimePreferences =
            serde_json::from_str(r#"{"preload_local_models": true, "providers": {}}"#).unwrap();
        assert_eq!(legacy.assistant, AssistantPreferences::default());
        assert!(legacy.assistant.auto_title && !legacy.assistant.transcript_upload_allowed);

        // A group that is not an assistant preset resets the choice and the
        // consent, without discarding the rest of the file.
        preferences.runtime.assistant.provider_group = "deepl".into();
        save_preferences(&path, &preferences).unwrap();
        let loaded = load_preferences(&path);
        assert_eq!(loaded.runtime.assistant, AssistantPreferences::default());
        assert_eq!(loaded.session.session_context, "Topic: textures");
    }

    #[tokio::test]
    async fn stale_environment_restart_waits_for_in_flight_assistant_jobs() {
        let state = RuntimeState::default();
        let stale = |state: &RuntimeState| state.sidecar_environment_stale.load(Ordering::Acquire);
        let (job_id, _cancel, _) = state
            .assistant
            .register(assistant::AssistantTask::Probe, None, false)
            .unwrap();
        // Registered but not yet sent: restarting is still harmless.
        state.invalidate_sidecar_environment().await;
        assert!(!stale(&state));
        state.assistant.mark_dispatched(job_id);
        state.invalidate_sidecar_environment().await;
        assert!(stale(&state), "restart must wait for the in-flight job");
        // Jobs never make the live session busy.
        assert!(state.is_idle());
        state
            .assistant
            .finish(job_id, assistant::JobState::Completed, None, None, None);
        state.restart_stale_sidecar_if_quiet().await;
        assert!(!stale(&state));

        // A live session defers the restart as before.
        state
            .core
            .lock()
            .unwrap()
            .start(StartSessionRequest::default())
            .unwrap();
        state.invalidate_sidecar_environment().await;
        assert!(stale(&state));
    }

    fn transcript_event(session_id: uuid::Uuid, revision: i64, text: &str) -> SidecarEvent {
        SidecarEvent::Transcript(json!({
            "session_id": session_id,
            "event_id": uuid::Uuid::new_v4(),
            "revision_id": revision,
            "kind": "stable",
            "text": text,
            "start_ms": revision as f64 * 1_000.0,
            "end_ms": revision as f64 * 1_000.0 + 900.0,
            "backend": "mock",
        }))
    }

    #[tokio::test]
    async fn persistence_queue_stores_every_segment_before_the_finished_hook() {
        let directory = tempfile::tempdir().unwrap();
        let state = RuntimeState::default();
        state
            .store
            .set(
                TranscriptStore::open(directory.path().join("history.sqlite"))
                    .await
                    .unwrap(),
            )
            .unwrap();
        let session_id = uuid::Uuid::new_v4();
        create_soak_history(&state, session_id, &json!({}), 1)
            .await
            .unwrap();
        let (sender, queue) = tokio::sync::mpsc::channel::<PersistItem>(16);
        for revision in 1..=3 {
            sender
                .send((session_id, transcript_event(session_id, revision, "Textures pop out.")))
                .await
                .unwrap();
        }
        sender
            .send((session_id, SidecarEvent::SessionFinished { session_id }))
            .await
            .unwrap();
        // Alignment and late events after the hook are still stored.
        sender
            .send((session_id, transcript_event(session_id, 4, "Late unit.")))
            .await
            .unwrap();
        drop(sender);

        let store = state.store().unwrap().clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut errors = Vec::new();
        run_persistence_queue(
            &state,
            queue,
            |_, error| errors.push(error),
            |finished| {
                let store = store.clone();
                let seen = Arc::clone(&seen);
                async move {
                    let stored = store.detail(finished).await.unwrap().segments.len();
                    seen.lock().unwrap().push((finished, stored));
                }
            },
        )
        .await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(*seen.lock().unwrap(), vec![(session_id, 3)]);
        assert_eq!(store.detail(session_id).await.unwrap().segments.len(), 4);
    }

    #[tokio::test]
    #[ignore = "60 audio-minute Rust/Desktop/Python/SQLite release soak"]
    async fn desktop_product_soak_60_audio_minutes() {
        let audio_minutes = std::env::var("ECHOLINGO_SOAK_AUDIO_MINUTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(60);
        let realtime = std::env::var("ECHOLINGO_SOAK_REALTIME").as_deref() == Ok("1");
        let frame_ms = 100_u64;
        let frame_count = 1_600_u16;
        let total_frames = audio_minutes * 60_000 / frame_ms;
        let directory = tempfile::tempdir().unwrap();
        let database_path = directory.path().join("history.sqlite");
        let state = Arc::new(RuntimeState::default());
        state
            .store
            .set(TranscriptStore::open(&database_path).await.unwrap())
            .unwrap();
        state.supervisor().ensure_started().await.unwrap();
        let mut receiver = state.supervisor().subscribe();

        let mut request = StartSessionRequest::default();
        request.asr_provider = "mock".into();
        request.translation_provider = "mock".into();
        request.audio_profile = "lecture".into();
        let starting = state.core.lock().unwrap().start(request).unwrap();
        let session_id = starting.session_id.unwrap();
        state
            .supervisor()
            .send_command(SidecarCommand::StartSession(mock_session_payload(
                session_id, true,
            )))
            .await
            .unwrap();
        let route = loop {
            if let SidecarEvent::Ready {
                session_id: ready_id,
                route,
            } = receiver.recv().await.unwrap()
            {
                if ready_id == session_id {
                    break route;
                }
            }
        };
        state
            .core
            .lock()
            .unwrap()
            .mark_listening(route_status(&route))
            .unwrap();
        create_soak_history(&state, session_id, &route, 1)
            .await
            .unwrap();

        let started = std::time::Instant::now();
        let initial_rss = resident_set_bytes();
        let sender_state = Arc::clone(&state);
        let mut audio_task = tokio::spawn(async move {
            let pcm = 0.08_f32.to_le_bytes().repeat(usize::from(frame_count));
            for sequence in 0..total_frames {
                if sequence == total_frames / 3 {
                    let paused = {
                        let mut core = sender_state.core.lock().unwrap();
                        let revision = core.snapshot().state_revision;
                        core.pause(revision).unwrap()
                    };
                    sender_state
                        .supervisor()
                        .send_command(SidecarCommand::Pause {
                            session_id,
                            epoch: paused.state_revision as u32,
                        })
                        .await
                        .unwrap();
                    sender_state
                        .supervisor()
                        .send_command(SidecarCommand::Resume {
                            session_id,
                            epoch: paused.state_revision as u32 + 1,
                        })
                        .await
                        .unwrap();
                    sender_state
                        .core
                        .lock()
                        .unwrap()
                        .resume(paused.state_revision)
                        .unwrap();
                }
                let header = AudioFrameHeader {
                    flags: 0,
                    sequence,
                    capture_monotonic_ns: sequence * frame_ms * 1_000_000,
                    sample_rate_hz: 16_000,
                    channels: 1,
                    frame_count,
                };
                let mut packet = Vec::with_capacity(32 + pcm.len());
                packet.extend_from_slice(&header.encode());
                packet.extend_from_slice(&pcm);
                sender_state.supervisor().send_audio(packet).await.unwrap();
                if realtime {
                    tokio::time::sleep(std::time::Duration::from_millis(frame_ms)).await;
                } else if sequence % 100 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        });
        let mut counts = SoakCounts::default();
        loop {
            tokio::select! {
                result = &mut audio_task => {
                    result.unwrap();
                    break;
                }
                event = receiver.recv() => {
                    consume_soak_event(&state, session_id, &event.unwrap(), &mut counts)
                        .await
                        .unwrap();
                }
            }
        }

        let current_revision = state.core.lock().unwrap().snapshot().state_revision;
        state
            .core
            .lock()
            .unwrap()
            .begin_stop(current_revision)
            .unwrap();
        state
            .supervisor()
            .send_command(SidecarCommand::FinishSession { session_id })
            .await
            .unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut finished = false;
        while tokio::time::Instant::now() < deadline && !(finished && counts.alignments > 0) {
            let event = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
                .await
                .unwrap()
                .unwrap();
            if matches!(event, SidecarEvent::SessionFinished { session_id: finished_id } if finished_id == session_id)
            {
                finished = true;
            }
            consume_soak_event(&state, session_id, &event, &mut counts)
                .await
                .unwrap();
        }
        assert!(finished, "sidecar did not acknowledge the first session");
        assert!(
            counts.alignments > 0,
            "asynchronous alignment did not complete"
        );
        state.core.lock().unwrap().complete().unwrap();
        state
            .store()
            .unwrap()
            .complete_session(session_id, &[])
            .await
            .unwrap();
        let detail = state.store().unwrap().detail(session_id).await.unwrap();
        assert!(!detail.segments.is_empty());
        assert_eq!(detail.segments[0].timestamp_quality, "forced");

        let second_request = StartSessionRequest {
            expected_state_revision: state.core.lock().unwrap().snapshot().state_revision,
            asr_provider: "mock".into(),
            translation_provider: "mock".into(),
            ..StartSessionRequest::default()
        };
        let second = state.core.lock().unwrap().start(second_request).unwrap();
        let second_id = second.session_id.unwrap();
        state
            .supervisor()
            .send_command(SidecarCommand::StartSession(mock_session_payload(
                second_id, false,
            )))
            .await
            .unwrap();
        let second_route = loop {
            if let SidecarEvent::Ready {
                session_id: ready_id,
                route,
            } = receiver.recv().await.unwrap()
            {
                if ready_id == second_id {
                    break route;
                }
            }
        };
        state
            .core
            .lock()
            .unwrap()
            .mark_listening(route_status(&second_route))
            .unwrap();
        create_soak_history(&state, second_id, &second_route, 2)
            .await
            .unwrap();
        let revision = state.core.lock().unwrap().snapshot().state_revision;
        state.core.lock().unwrap().begin_stop(revision).unwrap();
        state
            .supervisor()
            .send_command(SidecarCommand::FinishSession {
                session_id: second_id,
            })
            .await
            .unwrap();
        loop {
            let event = receiver.recv().await.unwrap();
            consume_soak_event(&state, second_id, &event, &mut counts)
                .await
                .unwrap();
            if matches!(event, SidecarEvent::SessionFinished { session_id: finished_id } if finished_id == second_id)
            {
                break;
            }
        }
        state.core.lock().unwrap().complete().unwrap();
        state
            .store()
            .unwrap()
            .complete_session(second_id, &[])
            .await
            .unwrap();
        assert_eq!(
            state.store().unwrap().search("", 10).await.unwrap().len(),
            2
        );
        state.supervisor().shutdown().await;

        let database_bytes = std::fs::metadata(&database_path).unwrap().len()
            + std::fs::metadata(database_path.with_extension("sqlite-wal"))
                .map(|metadata| metadata.len())
                .unwrap_or(0);
        let elapsed = started.elapsed().as_secs_f64();
        let final_rss = resident_set_bytes();
        println!(
            "DESKTOP_SOAK_RESULT={}",
            json!({
                "audio_minutes": audio_minutes,
                "wall_clock_realtime": realtime,
                "wall_seconds": elapsed,
                "processing_rtf": elapsed / (audio_minutes as f64 * 60.0),
                "frames": total_frames,
                "transcript_events": counts.transcripts,
                "translation_events": counts.translations,
                "metric_events": counts.metrics,
                "alignment_events": counts.alignments,
                "ui_updates_projected": counts.ui_updates,
                "recoverable_errors": counts.errors,
                "max_dropped_audio_ms": counts.max_dropped_audio_ms,
                "initial_rss_bytes": initial_rss,
                "final_rss_bytes": final_rss,
                "database_bytes": database_bytes,
                "second_session_completed": true
            })
        );
        assert_eq!(counts.errors, 0);
        assert_eq!(counts.max_dropped_audio_ms, 0.0);
    }
}
