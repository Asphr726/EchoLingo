//! In-app updates: check, download, verify, replace and restart.
//!
//! The Tauri updater plugin fetches `latest.json`, verifies the minisign
//! signature of the download against the public key in `tauri.conf.json`
//! and replaces the installed app. This module decides when that may
//! happen: never while a session runs, a forced alignment or an assistant
//! job is working, local models load or change, or the GPU pack installs.
//! React sees only the commands below and the `echolingo://update-status`
//! event, whose payload is the whole [`UpdateStatus`] (download progress
//! included, at most about ten events a second).
//!
//! Only the download is signed, not `latest.json`, and the plugin's own
//! "newer than the running version" test reads the manifest's `version`.
//! `plugins.updater.requireSignedVersion` therefore stays on: the plugin
//! then also refuses a download whose signature (its minisign trusted
//! comment) does not name that same version, so a tampered manifest cannot
//! pair a higher version number with an older, genuinely signed release to
//! force a downgrade. The Tauri CLI writes the version into the signature
//! from 2.11.5 on; `scripts/update_manifest.py` refuses to publish a
//! manifest whose signatures lack it.
//!
//! `desktop.log` records one `update check result=...` line per check and
//! `update install state=...` lines for every install step. URLs in those
//! lines and in error messages lose their query strings, which carry the
//! signed tokens of release downloads.

use crate::{shutdown_application, RuntimeState};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Error as UpdaterError, Update, UpdaterExt};

pub(crate) const UPDATE_STATUS_CHANNEL: &str = "echolingo://update-status";
/// Replaces the configured `latest.json` URL (smoke tests and staging).
/// Downloads are still verified against the built-in public key.
pub(crate) const UPDATE_ENDPOINT_ENV: &str = "ECHOLINGO_UPDATE_ENDPOINT";
/// `1` makes the launch check of a `.updatetest` build install what it
/// finds (see [`install_on_launch_requested`]).
pub(crate) const UPDATE_INSTALL_ON_LAUNCH_ENV: &str = "ECHOLINGO_UPDATE_INSTALL_ON_LAUNCH";
/// Bundle identifier suffix of the update smoke-test build.
pub(crate) const UPDATE_TEST_IDENTIFIER_SUFFIX: &str = ".updatetest";
/// Returned by everything an update refuses while it runs.
pub(crate) const UPDATE_IN_PROGRESS: &str =
    "EchoLingo is installing an update and restarts when it is done";
const RELEASE_PAGE_PREFIX: &str = "https://github.com/Asphr726/EchoLingo/releases/tag/v";
/// The launch check waits until the first window and the warm-up are under
/// way.
const LAUNCH_CHECK_DELAY: Duration = Duration::from_secs(15);
const CHECK_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Longest pause between two reads of a response. The download as a whole
/// has no deadline: it may be large and the connection slow.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Minimum spacing of progress events.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);
/// How long the smoke-test hook waits for the app to become idle.
const INSTALL_ON_LAUNCH_IDLE_WAIT: Duration = Duration::from_secs(600);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UpdateState {
    /// Not checked in this run (or a launch check failed quietly).
    #[default]
    Idle,
    Checking,
    Available,
    UpToDate,
    Downloading,
    Installing,
    /// A manual check or an install failed; `error` says why.
    Error,
}

/// A newer version the update server offers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AvailableUpdate {
    pub version: String,
    /// Release notes (Markdown), when the release has any.
    pub notes: Option<String>,
    /// Publication time, RFC 3339.
    pub date: Option<String>,
    /// The release page, for platforms that cannot update in place.
    pub release_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct DownloadProgress {
    pub downloaded_bytes: u64,
    /// `None` when the server sends no length.
    pub total_bytes: Option<u64>,
}

/// Everything the Updates settings and the top-bar pill show.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct UpdateStatus {
    pub current_version: String,
    pub state: UpdateState,
    pub available: Option<AvailableUpdate>,
    /// RFC 3339 time of the last check that got an answer.
    pub last_checked_at: Option<String>,
    pub error: Option<String>,
    /// Whether this copy can replace itself; otherwise the release page
    /// offers the download.
    pub in_place_supported: bool,
    pub unsupported_reason: Option<String>,
    /// The `check_updates_at_launch` preference.
    pub check_at_launch: bool,
    /// Set while downloading.
    pub progress: Option<DownloadProgress>,
    /// Why Install is refused right now (a session, an alignment, a job, a
    /// model or GPU pack install), or `None`.
    pub install_blocked_reason: Option<String>,
}

/// What an update would cut off, gathered from `RuntimeState`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Activity {
    pub session: bool,
    pub updating: bool,
    pub warmup: bool,
    pub alignment: bool,
    pub assistant: bool,
    pub models: bool,
    pub gpu_pack: bool,
}

/// Why an install is refused: a stable code for `desktop.log` and the
/// sentence the user reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Blocker {
    pub code: &'static str,
    pub message: &'static str,
}

const BLOCKED_BY_UPDATE: Blocker = Blocker {
    code: "update_running",
    message: "An update is already being installed.",
};
const BLOCKED_BY_SESSION: Blocker = Blocker {
    code: "session_active",
    message: "Stop the current session before updating EchoLingo.",
};
const BLOCKED_BY_ALIGNMENT: Blocker = Blocker {
    code: "alignment_running",
    message: "Word timings of the last session are still being aligned; update in a minute.",
};
const BLOCKED_BY_ASSISTANT: Blocker = Blocker {
    code: "assistant_running",
    message: "An AI assistant task is running; update when it has finished.",
};
const BLOCKED_BY_MODELS: Blocker = Blocker {
    code: "model_change",
    message: "A local model is being downloaded or removed; update when it has finished.",
};
const BLOCKED_BY_GPU_PACK: Blocker = Blocker {
    code: "gpu_pack_installing",
    message: "The GPU acceleration pack is being installed; update when it has finished.",
};
const BLOCKED_BY_WARMUP: Blocker = Blocker {
    code: "warmup",
    message: "Local models are loading in the background; update when they are ready.",
};
const BLOCKED_STATE_UNAVAILABLE: Blocker = Blocker {
    code: "state_unavailable",
    message: "EchoLingo could not read its session state; restart it and try again.",
};

/// Why an update may not be installed now, or `None`. The first match wins,
/// in the order a user would deal with them.
pub(crate) fn install_blocker(activity: Activity) -> Option<Blocker> {
    [
        (activity.updating, BLOCKED_BY_UPDATE),
        (activity.session, BLOCKED_BY_SESSION),
        (activity.alignment, BLOCKED_BY_ALIGNMENT),
        (activity.assistant, BLOCKED_BY_ASSISTANT),
        (activity.models, BLOCKED_BY_MODELS),
        (activity.gpu_pack, BLOCKED_BY_GPU_PACK),
        (activity.warmup, BLOCKED_BY_WARMUP),
    ]
    .into_iter()
    .find_map(|(busy, blocker)| busy.then_some(blocker))
}

/// Everything but the session and the update flag.
fn background_activity(state: &RuntimeState) -> Activity {
    Activity {
        warmup: state.warmup_in_progress.load(Ordering::Acquire),
        alignment: state.alignment_running(),
        assistant: state.assistant.any_running(),
        models: state
            .changing_models
            .lock()
            .map_or(true, |changing| !changing.is_empty()),
        gpu_pack: state
            .gpu_pack
            .get()
            .is_some_and(|pack| pack.is_installing()),
        ..Activity::default()
    }
}

fn current_activity(state: &RuntimeState) -> Activity {
    Activity {
        session: !state.is_idle(),
        updating: state.updating.load(Ordering::Acquire),
        ..background_activity(state)
    }
}

/// Holds `RuntimeState::updating` from a successful [`claim_install`] on.
/// Dropping it clears the flag again, unless [`InstallClaim::keep`] was
/// called because the app is about to restart.
struct InstallClaim<'a> {
    updating: &'a AtomicBool,
    kept: bool,
}

impl InstallClaim<'_> {
    fn keep(mut self) {
        self.kept = true;
    }
}

impl Drop for InstallClaim<'_> {
    fn drop(&mut self) {
        if !self.kept {
            self.updating.store(false, Ordering::Release);
        }
    }
}

/// Set `updating` if nothing an update would cut off is running.
///
/// The session check and the flag happen under the `core` lock, where
/// `start_session` checks the flag, so a session can never start in
/// between. Model changes register first and check the flag second; this
/// sets the flag first and looks for registrations second, so one of the
/// two always sees the other.
fn claim_install(state: &RuntimeState) -> Result<InstallClaim<'_>, Blocker> {
    {
        let core = state.core.lock().map_err(|_| BLOCKED_STATE_UNAVAILABLE)?;
        let activity = Activity {
            session: !matches!(
                core.snapshot().phase,
                app_core::SessionPhase::Idle | app_core::SessionPhase::Completed
            ),
            updating: state.updating.load(Ordering::Acquire),
            ..Activity::default()
        };
        if let Some(blocker) = install_blocker(activity) {
            return Err(blocker);
        }
        state.updating.store(true, Ordering::Release);
    }
    let claim = InstallClaim {
        updating: &state.updating,
        kept: false,
    };
    match install_blocker(background_activity(state)) {
        Some(blocker) => Err(blocker),
        None => Ok(claim),
    }
}

/// Whether this copy of EchoLingo can replace itself, from the operating
/// system (`std::env::consts::OS`) and the running executable's path; the
/// error says what to do instead.
pub(crate) fn in_place_support(os: &str, executable: Option<&str>) -> Result<(), String> {
    match os {
        "macos" => {
            let Some(path) = executable else {
                return Err("EchoLingo could not tell where it is installed. Download the new version from the release page.".into());
            };
            if path.contains("/AppTranslocation/") {
                return Err("macOS runs this copy of EchoLingo from a temporary, read-only location. Move EchoLingo to the Applications folder, open it from there, and update again.".into());
            }
            if path.starts_with("/Volumes/") {
                return Err("EchoLingo is running from a disk image or an external volume. Drag it to the Applications folder, open it from there, and update again.".into());
            }
            if !path.contains(".app/Contents/MacOS/") {
                return Err("Only an installed EchoLingo.app can update itself. Download the new version from the release page.".into());
            }
            Ok(())
        }
        "windows" => Ok(()),
        "linux" => Err("On Linux, install the new .deb package from the release page.".into()),
        _ => Err("This platform cannot update EchoLingo in place. Download the new version from the release page.".into()),
    }
}

fn current_executable() -> Option<String> {
    std::env::current_exe()
        .ok()
        .map(|path| path.to_string_lossy().into_owned())
}

fn local_support() -> Result<(), String> {
    in_place_support(std::env::consts::OS, current_executable().as_deref())
}

/// Whether the launch check should install what it finds: only in a build
/// whose bundle identifier ends in `.updatetest`, and only when both the
/// install switch and the endpoint override are set. The production
/// identifier can never match.
pub(crate) fn install_on_launch_requested(
    identifier: &str,
    env: impl Fn(&str) -> Option<String>,
) -> bool {
    identifier.ends_with(UPDATE_TEST_IDENTIFIER_SUFFIX)
        && env(UPDATE_INSTALL_ON_LAUNCH_ENV).is_some_and(|value| value.trim() == "1")
        && env(UPDATE_ENDPOINT_ENV).is_some_and(|value| !value.trim().is_empty())
}

pub(crate) fn release_url(version: &str) -> String {
    format!(
        "{RELEASE_PAGE_PREFIX}{}",
        version.trim().trim_start_matches('v')
    )
}

/// `text` without the query string or fragment of any http(s) URL in it:
/// release downloads redirect to signed URLs whose query carries a token.
pub(crate) fn redact_urls(text: &str) -> String {
    let mut redacted = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let start = [rest.find("https://"), rest.find("http://")]
            .into_iter()
            .flatten()
            .min();
        let Some(start) = start else {
            redacted.push_str(rest);
            return redacted;
        };
        redacted.push_str(&rest[..start]);
        let tail = &rest[start..];
        let end = tail
            .find(|c: char| c.is_whitespace() || matches!(c, ')' | '(' | '"' | '\'' | '<' | '>'))
            .unwrap_or(tail.len());
        let url = &tail[..end];
        match url.find(['?', '#']) {
            Some(cut) => {
                redacted.push_str(&url[..=cut]);
                redacted.push_str("[redacted]");
            }
            None => redacted.push_str(url),
        }
        rest = &tail[end..];
    }
}

/// An error and its causes on one line, URLs redacted.
fn error_text(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.contains(&cause_text) {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    redact_urls(&text)
}

fn signature_error(error: &UpdaterError) -> bool {
    matches!(
        error,
        UpdaterError::Minisign(_)
            | UpdaterError::Base64(_)
            | UpdaterError::SignatureUtf8(_)
            | UpdaterError::SignedVersionMismatch { .. }
            | UpdaterError::MissingSignedVersion
    )
}

/// The failed step (`download` or `verify`) and the message for the user.
fn download_failure(error: &UpdaterError) -> (&'static str, String) {
    if signature_error(error) {
        (
            "verify",
            "The download does not carry EchoLingo's release signature, so it was not installed."
                .into(),
        )
    } else {
        (
            "download",
            format!("The update could not be downloaded: {}", error_text(error)),
        )
    }
}

fn install_failure(error: &UpdaterError) -> String {
    match error {
        UpdaterError::Io(io) if io.kind() == std::io::ErrorKind::PermissionDenied => {
            "EchoLingo was not allowed to replace itself, so this version keeps running. Update again and allow the change, or download the new version from the release page.".into()
        }
        _ => format!(
            "The update could not be installed, so this version keeps running: {}",
            error_text(error)
        ),
    }
}

/// The last check and the running download. Checks and installs never
/// overlap (`operation`).
#[derive(Default)]
pub(crate) struct UpdateRegistry {
    cache: Mutex<UpdateCache>,
    operation: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct UpdateCache {
    state: UpdateState,
    available: Option<AvailableUpdate>,
    /// What the last check found, ready to download.
    pending: Option<Update>,
    last_checked_at: Option<DateTime<Utc>>,
    error: Option<String>,
    progress: Option<DownloadProgress>,
}

impl UpdateRegistry {
    fn cache(&self) -> MutexGuard<'_, UpdateCache> {
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The `operation` lock for a check, or `None` when the check should be
    /// skipped. A manual check waits for a running launch check and then asks
    /// again, so its own outcome (an error included) is shown; any other
    /// check, or a manual one during an install, is skipped.
    async fn check_operation(
        &self,
        trigger: Trigger,
        updating: &AtomicBool,
    ) -> Option<tokio::sync::MutexGuard<'_, ()>> {
        if let Ok(operation) = self.operation.try_lock() {
            return Some(operation);
        }
        if trigger != Trigger::Manual || updating.load(Ordering::Acquire) {
            return None;
        }
        Some(self.operation.lock().await)
    }

    /// Enter `Checking`; returns the state to go back to after a quiet
    /// failure.
    fn begin_check(&self, clear_error: bool) -> UpdateState {
        let mut cache = self.cache();
        let previous = cache.state;
        cache.state = UpdateState::Checking;
        if clear_error {
            cache.error = None;
        }
        previous
    }

    fn found(&self, update: Update, available: AvailableUpdate, at: DateTime<Utc>) {
        let mut cache = self.cache();
        cache.state = UpdateState::Available;
        cache.available = Some(available);
        cache.pending = Some(update);
        cache.error = None;
        cache.last_checked_at = Some(at);
    }

    fn up_to_date(&self, at: DateTime<Utc>) {
        let mut cache = self.cache();
        cache.state = UpdateState::UpToDate;
        cache.available = None;
        cache.pending = None;
        cache.error = None;
        cache.last_checked_at = Some(at);
    }

    /// A check failed: shown as an error, or (`None`) quietly back to
    /// `previous`.
    fn check_failed(&self, previous: UpdateState, error: Option<String>) {
        let mut cache = self.cache();
        match error {
            Some(error) => {
                cache.state = UpdateState::Error;
                cache.error = Some(error);
            }
            None => cache.state = previous,
        }
    }

    fn pending(&self) -> Option<Update> {
        self.cache().pending.clone()
    }

    fn begin_download(&self) {
        let mut cache = self.cache();
        cache.state = UpdateState::Downloading;
        cache.error = None;
        cache.progress = Some(DownloadProgress {
            downloaded_bytes: 0,
            total_bytes: None,
        });
    }

    fn set_progress(&self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        self.cache().progress = Some(DownloadProgress {
            downloaded_bytes,
            total_bytes,
        });
    }

    fn begin_install(&self) {
        self.cache().state = UpdateState::Installing;
    }

    /// The install failed; the update stays available for another try.
    fn install_failed(&self, message: String) {
        let mut cache = self.cache();
        cache.state = UpdateState::Error;
        cache.error = Some(message);
        cache.progress = None;
    }

    /// Something started during the download; back to `Available`.
    fn install_refused(&self, message: &str) {
        let mut cache = self.cache();
        cache.state = UpdateState::Available;
        cache.error = Some(message.into());
        cache.progress = None;
    }

    fn view(&self) -> CacheView {
        let cache = self.cache();
        CacheView {
            state: cache.state,
            available: cache.available.clone(),
            last_checked_at: cache
                .last_checked_at
                .map(|at| at.to_rfc3339_opts(SecondsFormat::Secs, true)),
            error: cache.error.clone(),
            progress: cache.progress,
        }
    }
}

/// The part of [`UpdateStatus`] the cache holds.
#[derive(Debug, PartialEq)]
struct CacheView {
    state: UpdateState,
    available: Option<AvailableUpdate>,
    last_checked_at: Option<String>,
    error: Option<String>,
    progress: Option<DownloadProgress>,
}

fn describe_update(update: &Update) -> AvailableUpdate {
    AvailableUpdate {
        version: update.version.clone(),
        notes: update
            .body
            .as_deref()
            .map(str::trim)
            .filter(|notes| !notes.is_empty())
            .map(str::to_string),
        date: update
            .date
            .and_then(|date| DateTime::from_timestamp(date.unix_timestamp(), 0))
            .map(|date| date.to_rfc3339_opts(SecondsFormat::Secs, true)),
        release_url: release_url(&update.version),
    }
}

pub(crate) fn status(app: &AppHandle) -> UpdateStatus {
    let state = app.state::<RuntimeState>();
    let blocker = install_blocker(current_activity(&state));
    let support = local_support();
    let cache = state.updater.view();
    UpdateStatus {
        current_version: app.package_info().version.to_string(),
        state: cache.state,
        available: cache.available,
        last_checked_at: cache.last_checked_at,
        error: cache.error,
        in_place_supported: support.is_ok(),
        unsupported_reason: support.err(),
        check_at_launch: state.check_updates_at_launch(),
        progress: cache.progress,
        install_blocked_reason: blocker.map(|blocker| blocker.message.to_string()),
    }
}

fn emit_status(app: &AppHandle) {
    let _ = app.emit(UPDATE_STATUS_CHANNEL, status(app));
}

fn log(app: &AppHandle, line: &str) {
    eprintln!("{line}");
    app.state::<RuntimeState>().log_desktop_event(line);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trigger {
    /// The quiet check shortly after launch.
    Launch,
    /// Settings → Check for updates, or Install.
    Manual,
    /// The `.updatetest` smoke-test hook.
    Test,
}

impl Trigger {
    fn as_str(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::Manual => "manual",
            Self::Test => "test",
        }
    }
}

/// What the update server said about this version.
enum CheckAnswer {
    Available(Box<Update>),
    UpToDate,
    /// `latest.json` lists no build for this platform (the targets asked
    /// for), for example while that platform's release is still a draft.
    /// Shown as up to date and logged apart so it can be told from one.
    NoPlatformEntry(String),
}

fn check_answer(result: Result<Option<Update>, UpdaterError>) -> Result<CheckAnswer, String> {
    match result {
        Ok(Some(update)) => Ok(CheckAnswer::Available(Box::new(update))),
        Ok(None) => Ok(CheckAnswer::UpToDate),
        Err(UpdaterError::TargetNotFound(target)) => Ok(CheckAnswer::NoPlatformEntry(target)),
        Err(UpdaterError::TargetsNotFound(targets)) => {
            Ok(CheckAnswer::NoPlatformEntry(targets.join(",")))
        }
        Err(error) => Err(error_text(&error)),
    }
}

fn no_platform_entry_log_line(current: &str, trigger: Trigger, targets: &str) -> String {
    format!(
        "update check result=no_platform_entry current={current} trigger={} targets={targets}",
        trigger.as_str()
    )
}

/// Ask the update server.
async fn fetch(app: &AppHandle) -> Result<CheckAnswer, String> {
    // Connect and read timeouts only: a whole-request timeout would also
    // cut off a large download on a slow connection.
    let mut builder = app.updater_builder().configure_client(|client| {
        client
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
    });
    if let Some(endpoint) = crate::process_env(UPDATE_ENDPOINT_ENV) {
        let url = endpoint
            .trim()
            .parse::<tauri::Url>()
            .map_err(|_| format!("{UPDATE_ENDPOINT_ENV} is not a valid URL"))?;
        builder = builder
            .endpoints(vec![url])
            .map_err(|error| error_text(&error))?;
    }
    let updater = builder.build().map_err(|error| error_text(&error))?;
    match tokio::time::timeout(CHECK_TIMEOUT, updater.check()).await {
        Err(_) => Err(format!(
            "The update server did not answer within {} seconds.",
            CHECK_TIMEOUT.as_secs()
        )),
        Ok(result) => check_answer(result),
    }
}

/// Check for a newer version and cache the answer. A launch (or test-hook)
/// check that fails is only logged; a manual one shows its error.
async fn check(app: &AppHandle, trigger: Trigger) {
    let state = app.state::<RuntimeState>();
    // Otherwise a check or install is already running; its result arrives
    // as an event.
    let Some(_operation) = state
        .updater
        .check_operation(trigger, &state.updating)
        .await
    else {
        return;
    };
    if state.updating.load(Ordering::Acquire) {
        return;
    }
    let previous = state.updater.begin_check(trigger == Trigger::Manual);
    emit_status(app);
    let current = app.package_info().version.to_string();
    let outcome = fetch(app).await;
    let now = Utc::now();
    match outcome {
        Ok(CheckAnswer::Available(update)) => {
            let available = describe_update(&update);
            log(
                app,
                &format!(
                    "update check result=available version={} current={current} trigger={}",
                    available.version,
                    trigger.as_str()
                ),
            );
            state.updater.found(*update, available, now);
        }
        Ok(CheckAnswer::UpToDate) => {
            log(
                app,
                &format!(
                    "update check result=up_to_date version={current} current={current} trigger={}",
                    trigger.as_str()
                ),
            );
            state.updater.up_to_date(now);
        }
        Ok(CheckAnswer::NoPlatformEntry(targets)) => {
            log(app, &no_platform_entry_log_line(&current, trigger, &targets));
            state.updater.up_to_date(now);
        }
        Err(message) => {
            log(
                app,
                &format!(
                    "update check result=error current={current} trigger={} error={message}",
                    trigger.as_str()
                ),
            );
            let shown = (trigger == Trigger::Manual)
                .then(|| format!("Could not check for updates: {message}"));
            state.updater.check_failed(previous, shown);
        }
    }
    emit_status(app);
}

/// Download, verify and install the update the last check found, then
/// restart. Returns only when nothing was replaced (an error) or, on macOS,
/// once the restart has been requested; on Windows a successful install
/// ends the process.
async fn install(app: &AppHandle, trigger: Trigger) -> Result<(), String> {
    let state = app.state::<RuntimeState>();
    let Ok(_operation) = state.updater.operation.try_lock() else {
        return Err("EchoLingo is checking for updates; try again in a moment.".into());
    };
    let update = state
        .updater
        .pending()
        .ok_or_else(|| "No update is ready to install; check for updates first.".to_string())?;
    let version = update.version.clone();
    if let Err(reason) = local_support() {
        log(
            app,
            &format!("update install state=refused version={version} reason=unsupported_location"),
        );
        return Err(reason);
    }
    let claim = match claim_install(&state) {
        Ok(claim) => claim,
        Err(blocker) => {
            log(
                app,
                &format!(
                    "update install state=refused version={version} reason={}",
                    blocker.code
                ),
            );
            return Err(blocker.message.into());
        }
    };

    log(
        app,
        &format!(
            "update install state=downloading version={version} trigger={}",
            trigger.as_str()
        ),
    );
    state.updater.begin_download();
    emit_status(app);
    let progress_app = app.clone();
    let mut downloaded = 0_u64;
    let mut last_emit: Option<Instant> = None;
    let downloaded_bytes = update
        .download(
            move |chunk, total| {
                downloaded += chunk as u64;
                progress_app
                    .state::<RuntimeState>()
                    .updater
                    .set_progress(downloaded, total);
                if last_emit.is_none_or(|at| at.elapsed() >= PROGRESS_INTERVAL) {
                    last_emit = Some(Instant::now());
                    emit_status(&progress_app);
                }
            },
            || {},
        )
        .await;
    let bytes = match downloaded_bytes {
        Ok(bytes) => bytes,
        Err(error) => {
            let (stage, message) = download_failure(&error);
            log(
                app,
                &format!(
                    "update install state=failed version={version} stage={stage} error={}",
                    error_text(&error)
                ),
            );
            drop(claim);
            state.updater.install_failed(message.clone());
            emit_status(app);
            return Err(message);
        }
    };
    log(
        app,
        &format!(
            "update install state=downloaded version={version} bytes={}",
            bytes.len()
        ),
    );

    // Everything else was refused during the download; look again before
    // anything is replaced all the same.
    if let Some(blocker) = install_blocker(Activity {
        session: !state.is_idle(),
        ..background_activity(&state)
    }) {
        log(
            app,
            &format!(
                "update install state=refused version={version} reason={}",
                blocker.code
            ),
        );
        drop(claim);
        state.updater.install_refused(blocker.message);
        emit_status(app);
        return Err(blocker.message.into());
    }

    state.updater.begin_install();
    emit_status(app);
    log(
        app,
        &format!("update install state=installing version={version}"),
    );
    if cfg!(target_os = "windows") {
        // The NSIS installer replaces the sidecar and llama.cpp files and
        // quits this process, so everything stops first.
        state.shutting_down.store(true, Ordering::SeqCst);
        shutdown_application(app).await;
        log(
            app,
            &format!("update install state=installer_started version={version}"),
        );
        let result = tauri::async_runtime::spawn_blocking(move || update.install(bytes)).await;
        // Only reached when the installer could not be started.
        state.shutting_down.store(false, Ordering::SeqCst);
        if let Some(window) = app.get_webview_window("main") {
            // The plugin hides the windows before it starts the installer.
            let _ = window.show();
        }
        let message = match result {
            Ok(Ok(())) => "The installer did not start; this version keeps running.".to_string(),
            Ok(Err(error)) => install_failure(&error),
            Err(error) => format!("The update could not be installed: {error}"),
        };
        log(
            app,
            &format!("update install state=failed version={version} stage=install error={message}"),
        );
        drop(claim);
        state.updater.install_failed(message.clone());
        emit_status(app);
        return Err(message);
    }

    // Replace the app first: when the Applications folder needs an
    // administrator and the prompt is cancelled, the old app keeps running
    // untouched.
    let result = tauri::async_runtime::spawn_blocking(move || update.install(bytes)).await;
    let result = match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(install_failure(&error)),
        Err(error) => Err(format!("The update could not be installed: {error}")),
    };
    if let Err(message) = result {
        log(
            app,
            &format!("update install state=failed version={version} stage=install error={message}"),
        );
        drop(claim);
        state.updater.install_failed(message.clone());
        emit_status(app);
        return Err(message);
    }
    log(
        app,
        &format!("update install state=installed version={version}"),
    );
    claim.keep();
    state.shutting_down.store(true, Ordering::SeqCst);
    shutdown_application(app).await;
    log(
        app,
        &format!("update install state=restarting version={version}"),
    );
    app.request_restart();
    Ok(())
}

/// Smoke-test hook: check now and install what is found once the app is
/// idle. Only called when [`install_on_launch_requested`] holds.
async fn install_on_launch(app: &AppHandle) {
    check(app, Trigger::Test).await;
    let state = app.state::<RuntimeState>();
    let Some(version) = state.updater.pending().map(|update| update.version) else {
        return;
    };
    let deadline = Instant::now() + INSTALL_ON_LAUNCH_IDLE_WAIT;
    let mut reported = None;
    while let Some(blocker) = install_blocker(current_activity(&state)) {
        if Instant::now() >= deadline {
            log(
                app,
                &format!(
                    "update install state=refused version={version} reason={}",
                    blocker.code
                ),
            );
            return;
        }
        if reported != Some(blocker.code) {
            reported = Some(blocker.code);
            log(
                app,
                &format!(
                    "update install state=waiting version={version} reason={}",
                    blocker.code
                ),
            );
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    // `install` logs its own outcome.
    let _ = install(app, Trigger::Test).await;
}

/// Start the launch check in the background: after [`LAUNCH_CHECK_DELAY`]
/// when the preference is on, or at once (and installing) for the
/// smoke-test hook. Runs once per app, not per window.
pub(crate) fn schedule_launch_check(app: AppHandle) {
    let install_now = install_on_launch_requested(&app.config().identifier, crate::process_env);
    tauri::async_runtime::spawn(async move {
        if install_now {
            install_on_launch(&app).await;
            return;
        }
        tokio::time::sleep(LAUNCH_CHECK_DELAY).await;
        // Read after the delay: turning the preference off meanwhile counts.
        if app.state::<RuntimeState>().check_updates_at_launch() {
            check(&app, Trigger::Launch).await;
        }
    });
}

#[tauri::command]
pub async fn get_update_status(app: AppHandle) -> Result<UpdateStatus, String> {
    Ok(status(&app))
}

/// A manual check; its outcome (including a failure) is in the status.
#[tauri::command]
pub async fn check_for_update(app: AppHandle) -> Result<UpdateStatus, String> {
    check(&app, Trigger::Manual).await;
    Ok(status(&app))
}

/// Download, verify and install the update found by the last check, then
/// restart. Refused while anything an update would cut off is running.
#[tauri::command]
pub async fn install_update(app: AppHandle) -> Result<UpdateStatus, String> {
    install(&app, Trigger::Manual).await?;
    Ok(status(&app))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::StartSessionRequest;

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn status_serializes_for_the_webview() {
        let status = UpdateStatus {
            current_version: "0.3.1".into(),
            state: UpdateState::Downloading,
            available: Some(AvailableUpdate {
                version: "0.3.2".into(),
                notes: Some("## Fixes\n\n- Faster start".into()),
                date: Some("2026-09-20T08:00:00Z".into()),
                release_url: release_url("0.3.2"),
            }),
            last_checked_at: Some("2026-09-24T10:00:00Z".into()),
            error: None,
            in_place_supported: true,
            unsupported_reason: None,
            check_at_launch: true,
            progress: Some(DownloadProgress {
                downloaded_bytes: 1_024,
                total_bytes: None,
            }),
            install_blocked_reason: Some(BLOCKED_BY_UPDATE.message.into()),
        };
        let value = serde_json::to_value(&status).unwrap();
        assert_eq!(value["state"], "downloading");
        assert_eq!(value["current_version"], "0.3.1");
        assert_eq!(value["available"]["version"], "0.3.2");
        assert_eq!(
            value["available"]["release_url"],
            "https://github.com/Asphr726/EchoLingo/releases/tag/v0.3.2"
        );
        assert_eq!(value["progress"]["downloaded_bytes"], 1_024);
        assert!(value["progress"]["total_bytes"].is_null());
        assert!(value["error"].is_null());
        assert_eq!(value["check_at_launch"], true);
        assert_eq!(value["in_place_supported"], true);
        assert_eq!(
            value["install_blocked_reason"],
            "An update is already being installed."
        );
        for (state, name) in [
            (UpdateState::Idle, "idle"),
            (UpdateState::Checking, "checking"),
            (UpdateState::Available, "available"),
            (UpdateState::UpToDate, "up_to_date"),
            (UpdateState::Installing, "installing"),
            (UpdateState::Error, "error"),
        ] {
            assert_eq!(serde_json::to_value(state).unwrap(), name);
        }
        assert_eq!(release_url("v0.3.2"), release_url("0.3.2"));
    }

    #[test]
    fn the_shipped_config_requires_the_signed_version() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        // Parsed as the plugin parses it, so a misspelt key cannot pass.
        let updater: tauri_plugin_updater::Config =
            serde_json::from_value(config["plugins"]["updater"].clone()).unwrap();
        assert!(updater.require_signed_version);
        assert!(!updater.allow_downgrades);
        assert!(!updater.dangerous_insecure_transport_protocol);
        assert_eq!(
            updater
                .endpoints
                .iter()
                .map(|url| url.as_str())
                .collect::<Vec<_>>(),
            ["https://github.com/Asphr726/EchoLingo/releases/download/updater/latest.json"]
        );
    }

    #[test]
    fn install_blocker_names_the_first_reason() {
        assert_eq!(install_blocker(Activity::default()), None);
        let cases = [
            (
                Activity {
                    updating: true,
                    ..Activity::default()
                },
                "update_running",
            ),
            (
                Activity {
                    session: true,
                    ..Activity::default()
                },
                "session_active",
            ),
            (
                Activity {
                    alignment: true,
                    ..Activity::default()
                },
                "alignment_running",
            ),
            (
                Activity {
                    assistant: true,
                    ..Activity::default()
                },
                "assistant_running",
            ),
            (
                Activity {
                    models: true,
                    ..Activity::default()
                },
                "model_change",
            ),
            (
                Activity {
                    gpu_pack: true,
                    ..Activity::default()
                },
                "gpu_pack_installing",
            ),
            (
                Activity {
                    warmup: true,
                    ..Activity::default()
                },
                "warmup",
            ),
        ];
        for (activity, code) in cases {
            let blocker = install_blocker(activity).unwrap();
            assert_eq!(blocker.code, code);
            assert!(blocker.message.ends_with('.'), "{}", blocker.message);
        }
        let everything = Activity {
            session: true,
            updating: true,
            warmup: true,
            alignment: true,
            assistant: true,
            models: true,
            gpu_pack: true,
        };
        assert_eq!(install_blocker(everything), Some(BLOCKED_BY_UPDATE));
        let recording_and_aligning = Activity {
            session: true,
            alignment: true,
            ..Activity::default()
        };
        assert_eq!(
            install_blocker(recording_and_aligning).unwrap().message,
            "Stop the current session before updating EchoLingo."
        );
    }

    #[tokio::test]
    async fn claims_are_exclusive_and_refused_while_work_runs() {
        let state = RuntimeState::default();
        let claim = claim_install(&state).unwrap();
        assert!(state.updating.load(Ordering::Acquire));
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_UPDATE));
        drop(claim);
        assert!(!state.updating.load(Ordering::Acquire));

        // A refused claim leaves the flag clear.
        state.note_alignment_activity(Some(uuid::Uuid::new_v4()));
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_ALIGNMENT));
        assert!(!state.updating.load(Ordering::Acquire));
        state.note_alignment_activity(None);

        state
            .changing_models
            .lock()
            .unwrap()
            .insert("hymt2-1.8b".into());
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_MODELS));
        state.changing_models.lock().unwrap().clear();

        state.warmup_in_progress.store(true, Ordering::Release);
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_WARMUP));
        state.warmup_in_progress.store(false, Ordering::Release);

        let (job_id, _cancel, _view) = state
            .assistant
            .register(
                crate::assistant::AssistantTask::Title,
                Some(uuid::Uuid::new_v4()),
                true,
            )
            .unwrap();
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_ASSISTANT));
        state.assistant.finish(
            job_id,
            crate::assistant::JobState::Cancelled,
            None,
            None,
            None,
        );

        // A kept claim survives (the app restarts), and nothing may bring
        // the sidecar back meanwhile.
        claim_install(&state).unwrap().keep();
        assert!(state.updating.load(Ordering::Acquire));
        assert_eq!(
            state
                .ensure_sidecar(None, crate::SidecarUser::Background)
                .await
                .unwrap_err(),
            UPDATE_IN_PROGRESS
        );
        state.updating.store(false, Ordering::Release);

        state
            .core
            .lock()
            .unwrap()
            .start(StartSessionRequest::default())
            .unwrap();
        assert_eq!(claim_install(&state).err(), Some(BLOCKED_BY_SESSION));
        assert!(!state.updating.load(Ordering::Acquire));
        assert_eq!(
            install_blocker(current_activity(&state)),
            Some(BLOCKED_BY_SESSION)
        );
    }

    #[test]
    fn install_on_launch_needs_the_test_identifier_and_both_variables() {
        let both = [
            (UPDATE_INSTALL_ON_LAUNCH_ENV, "1"),
            (UPDATE_ENDPOINT_ENV, "https://127.0.0.1:8443/latest.json"),
        ];
        assert!(install_on_launch_requested(
            "app.echolingo.desktop.updatetest",
            env_from(&both)
        ));
        // Never in the production app, whatever the environment says.
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop",
            env_from(&both)
        ));
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop.updatetest.other",
            env_from(&both)
        ));
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop.updatetest",
            env_from(&both[..1])
        ));
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop.updatetest",
            env_from(&both[1..])
        ));
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop.updatetest",
            env_from(&[
                (UPDATE_INSTALL_ON_LAUNCH_ENV, "0"),
                (UPDATE_ENDPOINT_ENV, "https://127.0.0.1:8443/latest.json"),
            ])
        ));
        assert!(!install_on_launch_requested(
            "app.echolingo.desktop.updatetest",
            env_from(&[
                (UPDATE_INSTALL_ON_LAUNCH_ENV, "1"),
                (UPDATE_ENDPOINT_ENV, "  ")
            ])
        ));
    }

    #[test]
    fn in_place_support_follows_the_platform_and_location() {
        let installed = "/Applications/EchoLingo.app/Contents/MacOS/EchoLingo";
        assert_eq!(in_place_support("macos", Some(installed)), Ok(()));
        assert_eq!(
            in_place_support(
                "macos",
                Some("/Users/ada/Applications/EchoLingo.app/Contents/MacOS/EchoLingo")
            ),
            Ok(())
        );
        let translocated = "/private/var/folders/xy/T/AppTranslocation/0A1B2C3D/d/EchoLingo.app/Contents/MacOS/EchoLingo";
        assert!(in_place_support("macos", Some(translocated))
            .unwrap_err()
            .contains("Applications folder"));
        let disk_image = "/Volumes/EchoLingo 0.3.0/EchoLingo.app/Contents/MacOS/EchoLingo";
        assert!(in_place_support("macos", Some(disk_image))
            .unwrap_err()
            .contains("disk image"));
        // A development binary is not an app bundle.
        assert!(in_place_support(
            "macos",
            Some("/Users/ada/EchoLingo/target/debug/echolingo-desktop")
        )
        .is_err());
        assert!(in_place_support("macos", None).is_err());
        assert_eq!(
            in_place_support(
                "windows",
                Some(r"C:\Users\ada\AppData\Local\EchoLingo\echolingo-desktop.exe")
            ),
            Ok(())
        );
        assert!(
            in_place_support("linux", Some("/usr/bin/echolingo-desktop"))
                .unwrap_err()
                .contains(".deb")
        );
        assert!(in_place_support("freebsd", Some("/usr/local/bin/echolingo")).is_err());
    }

    #[test]
    fn urls_lose_their_query_strings() {
        assert_eq!(
            redact_urls("error sending request for url (https://objects.githubusercontent.com/a/b.tar.gz?X-Amz-Signature=abc&token=xyz): timed out"),
            "error sending request for url (https://objects.githubusercontent.com/a/b.tar.gz?[redacted]): timed out"
        );
        assert_eq!(
            redact_urls("see http://127.0.0.1:8080/latest.json#frag and https://example.com/x"),
            "see http://127.0.0.1:8080/latest.json#[redacted] and https://example.com/x"
        );
        assert_eq!(redact_urls("no links here"), "no links here");
    }

    #[tokio::test]
    async fn only_a_manual_check_waits_for_a_running_check() {
        let registry = UpdateRegistry::default();
        let updating = AtomicBool::new(false);
        assert!(registry
            .check_operation(Trigger::Launch, &updating)
            .await
            .is_some());

        let running = registry.operation.lock().await;
        for trigger in [Trigger::Launch, Trigger::Test] {
            assert!(registry.check_operation(trigger, &updating).await.is_none());
        }
        // An install holds the lock: a manual check does not wait for it.
        updating.store(true, Ordering::Release);
        assert!(registry
            .check_operation(Trigger::Manual, &updating)
            .await
            .is_none());
        updating.store(false, Ordering::Release);

        let mut waiting = std::pin::pin!(registry.check_operation(Trigger::Manual, &updating));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut waiting)
                .await
                .is_err()
        );
        drop(running);
        assert!(waiting.await.is_some());
    }

    #[test]
    fn a_missing_platform_entry_reads_as_up_to_date_but_is_logged_apart() {
        assert!(matches!(check_answer(Ok(None)), Ok(CheckAnswer::UpToDate)));
        assert!(matches!(
            check_answer(Err(UpdaterError::TargetNotFound("windows-x86_64".into()))),
            Ok(CheckAnswer::NoPlatformEntry(targets)) if targets == "windows-x86_64"
        ));
        assert!(matches!(
            check_answer(Err(UpdaterError::TargetsNotFound(vec![
                "windows-x86_64-nsis".into(),
                "windows-x86_64".into(),
            ]))),
            Ok(CheckAnswer::NoPlatformEntry(targets))
                if targets == "windows-x86_64-nsis,windows-x86_64"
        ));
        assert!(check_answer(Err(UpdaterError::MissingSignedVersion)).is_err());
        assert_eq!(
            no_platform_entry_log_line("0.3.1", Trigger::Launch, "windows-x86_64"),
            "update check result=no_platform_entry current=0.3.1 trigger=launch \
             targets=windows-x86_64"
        );
    }

    #[test]
    fn a_quiet_check_failure_keeps_the_previous_state() {
        let registry = UpdateRegistry::default();
        let previous = registry.begin_check(false);
        assert_eq!(previous, UpdateState::Idle);
        assert_eq!(registry.view().state, UpdateState::Checking);
        registry.check_failed(previous, None);
        assert_eq!(
            registry.view(),
            CacheView {
                state: UpdateState::Idle,
                available: None,
                last_checked_at: None,
                error: None,
                progress: None,
            }
        );

        let previous = registry.begin_check(true);
        registry.check_failed(
            previous,
            Some("Could not check for updates: offline".into()),
        );
        let view = registry.view();
        assert_eq!(view.state, UpdateState::Error);
        assert_eq!(
            view.error.as_deref(),
            Some("Could not check for updates: offline")
        );

        // A manual check clears the old error while it runs.
        registry.begin_check(true);
        assert_eq!(registry.view().error, None);
        registry.up_to_date(Utc::now());
        let view = registry.view();
        assert_eq!(view.state, UpdateState::UpToDate);
        assert!(view.available.is_none() && view.last_checked_at.is_some());

        registry.begin_download();
        registry.set_progress(2_048, Some(4_096));
        assert_eq!(
            registry.view().progress,
            Some(DownloadProgress {
                downloaded_bytes: 2_048,
                total_bytes: Some(4_096),
            })
        );
        registry.install_refused(BLOCKED_BY_ASSISTANT.message);
        let view = registry.view();
        assert_eq!(view.state, UpdateState::Available);
        assert_eq!(view.error.as_deref(), Some(BLOCKED_BY_ASSISTANT.message));
        assert_eq!(view.progress, None);
    }
}
