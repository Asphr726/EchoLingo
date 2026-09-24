//! The optional NVIDIA acceleration pack for Windows and Linux x64.
//!
//! The pack holds a CUDA build of the sidecar (for Qwen3-ASR) and a Vulkan
//! build of llama.cpp (for Hy-MT2). It is too large for the installers, so it
//! is published as split, individually hashed parts next to a manifest,
//! downloaded on demand (resumable), verified, streamed through zstd and tar
//! into a staging directory, self-tested and only then activated.

use crate::{activate_directory, remove_scoped_directory, GpuRuntimeLayout, ModelProgress};
use process_support::configure_background;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// `model_id` of the pack's progress events on the model-progress channel.
pub const GPU_PACK_PROGRESS_ID: &str = "gpu-pack";
pub const GPU_PACK_SCHEMA_VERSION: u16 = 1;
/// Overrides where manifests and parts come from: an http(s) URL prefix, a
/// `file://` URL or a plain local directory.
pub const GPU_PACK_BASE_URL_ENV: &str = "ECHOLINGO_GPU_PACK_BASE_URL";
const RELEASE_DOWNLOAD_URL: &str = "https://github.com/Asphr726/EchoLingo/releases/download";
/// CUDA 13.x runs on R580+ drivers (minor-version compatibility) on both
/// Windows and Linux; 580.65 is the first R580 production driver. Applies
/// until a manifest names its own minimum.
pub const DEFAULT_MIN_DRIVER_VERSION: &str = "580.65";
/// Turing (GeForce RTX 20 / GTX 16 series) and newer.
pub const DEFAULT_MIN_COMPUTE_CAPABILITY: &str = "7.5";

const INSTALL_DIRECTORY: &str = "gpu-pack";
const STAGING_DIRECTORY: &str = ".gpu-pack.installing";
const DOWNLOADS_DIRECTORY: &str = ".downloads";
const RECORD_FILE: &str = ".echolingo-gpu-pack.json";
const ARCHIVE_ROOT: &str = "echolingo-gpu-pack";
const MANIFEST_LIMIT_BYTES: usize = 1024 * 1024;
const DOWNLOAD_ATTEMPTS: u32 = 4;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
/// The first start of a frozen CUDA torch can be slow (antivirus scans of
/// every DLL on Windows).
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(900);
const DETECTION_TIMEOUT: Duration = Duration::from_secs(15);

type ProgressCallback = Arc<dyn Fn(ModelProgress) + Send + Sync>;
type SelfTestHook = Arc<dyn Fn(&Path) -> Result<Value, GpuPackError> + Send + Sync>;

/// The pack platform of this build, or `None` where no pack exists (macOS
/// uses Metal through the regular runtime).
pub fn pack_platform() -> Option<&'static str> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("windows-x64")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("linux-x64")
    } else {
        None
    }
}

#[derive(Debug, Error)]
pub enum GpuPackError {
    #[error("GPU acceleration packs are only available for Windows and Linux x64")]
    UnsupportedPlatform,
    #[error("the GPU acceleration pack is already being installed")]
    AlreadyInstalling,
    #[error("GPU acceleration pack manifest is invalid: {0}")]
    Manifest(String),
    #[error("the GPU acceleration pack is for EchoLingo {found}, this is {expected}")]
    VersionMismatch { expected: String, found: String },
    #[error("the GPU acceleration pack is for {found}, this computer needs {expected}")]
    PlatformMismatch { expected: String, found: String },
    #[error("not enough disk space: need {required_bytes} bytes, have {available_bytes} bytes")]
    DiskSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("GPU acceleration pack download failed: {0}")]
    Download(String),
    #[error("GPU acceleration pack part {0} failed verification; try again")]
    CorruptPart(String),
    #[error("GPU acceleration pack archive is invalid: {0}")]
    Archive(String),
    #[error("GPU acceleration pack self-test failed: {0}")]
    SelfTest(String),
    #[error("GPU acceleration pack filesystem operation failed: {0}")]
    Io(#[from] io::Error),
}

impl GpuPackError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::Download(_) | Self::Io(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuPackPart {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// The release manifest `echolingo-gpu-pack-{version}-{platform}.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuPackManifest {
    pub schema_version: u16,
    pub app_version: String,
    pub platform: String,
    pub torch_version: String,
    pub cuda_version: String,
    pub min_driver_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_driver_version_windows: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_driver_version_linux: Option<String>,
    pub min_compute_capability: String,
    pub archive: String,
    pub archive_bytes: u64,
    pub archive_sha256: String,
    pub unpacked_bytes: u64,
    pub parts: Vec<GpuPackPart>,
}

impl GpuPackManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, GpuPackError> {
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| GpuPackError::Manifest(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), GpuPackError> {
        let invalid = |message: String| Err(GpuPackError::Manifest(message));
        if self.schema_version != GPU_PACK_SCHEMA_VERSION {
            return invalid(format!("unsupported schema version {}", self.schema_version));
        }
        if self.parts.is_empty() {
            return invalid("no parts".into());
        }
        if !is_sha256(&self.archive_sha256) {
            return invalid("archive_sha256 is not a SHA-256 digest".into());
        }
        for part in &self.parts {
            if !is_plain_file_name(&part.name) {
                return invalid(format!("part name {:?} is not a plain file name", part.name));
            }
            if !is_sha256(&part.sha256) {
                return invalid(format!("part {} has no SHA-256 digest", part.name));
            }
        }
        let total = self
            .parts
            .iter()
            .try_fold(0_u64, |total, part| total.checked_add(part.bytes));
        if total != Some(self.archive_bytes) {
            return invalid("part sizes do not add up to archive_bytes".into());
        }
        if version_key(&self.min_compute_capability).is_empty() {
            return invalid("min_compute_capability is not a version".into());
        }
        Ok(())
    }

    /// Refuse a pack built for another app version or platform.
    pub fn check_compatible(&self, app_version: &str, platform: &str) -> Result<(), GpuPackError> {
        if self.app_version != app_version {
            return Err(GpuPackError::VersionMismatch {
                expected: app_version.into(),
                found: self.app_version.clone(),
            });
        }
        if self.platform != platform {
            return Err(GpuPackError::PlatformMismatch {
                expected: platform.into(),
                found: self.platform.clone(),
            });
        }
        Ok(())
    }

    /// The minimum NVIDIA driver on `platform`.
    pub fn min_driver_version_for(&self, platform: &str) -> &str {
        let specific = match platform {
            "windows-x64" => self.min_driver_version_windows.as_deref(),
            "linux-x64" => self.min_driver_version_linux.as_deref(),
            _ => None,
        };
        specific.unwrap_or(&self.min_driver_version)
    }

    pub fn download_bytes(&self) -> u64 {
        self.parts.iter().map(|part| part.bytes).sum()
    }
}

/// `.echolingo-gpu-pack.json` in the installed pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuPackRecord {
    pub manifest: GpuPackManifest,
    pub installed_at: String,
    pub self_test: Value,
}

impl GpuPackRecord {
    pub fn cuda_available(&self) -> Option<bool> {
        self.self_test["cuda"]["available"].as_bool()
    }

    pub fn device_name(&self) -> Option<String> {
        self.self_test["cuda"]["device_name"]
            .as_str()
            .map(str::to_string)
    }
}

/// The first NVIDIA GPU as `nvidia-smi` reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub compute_capability: String,
    pub driver_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuPackState {
    NotInstalled,
    Installing,
    Ready,
    UpdateRequired,
    Corrupt,
}

/// What Settings shows about GPU acceleration.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuAccelerationStatus {
    /// `false` on macOS and non-x64 builds; the UI hides the card.
    pub supported_platform: bool,
    pub gpu: Option<GpuInfo>,
    pub eligible: bool,
    pub ineligible_reason: Option<String>,
    pub pack_state: GpuPackState,
    pub pack_version: Option<String>,
    pub download_bytes: Option<u64>,
    pub installed_bytes: Option<u64>,
    pub cuda_available: Option<bool>,
    pub device_name: Option<String>,
    /// The user preference.
    pub enabled: bool,
    /// The local model services start on the pack now.
    pub active: bool,
    /// Set when the pack failed to start and the app fell back to the CPU.
    pub fallback_reason: Option<String>,
}

/// Why `gpu` cannot use the pack, or `None` when it can.
pub fn ineligibility(
    gpu: Option<&GpuInfo>,
    min_compute_capability: &str,
    min_driver_version: &str,
) -> Option<String> {
    let Some(gpu) = gpu else {
        return Some("No NVIDIA GPU detected".into());
    };
    if version_less(&gpu.compute_capability, min_compute_capability) {
        return Some(format!(
            "{} has compute capability {}; GPU acceleration needs {} or newer",
            gpu.name, gpu.compute_capability, min_compute_capability
        ));
    }
    if version_less(&gpu.driver_version, min_driver_version) {
        return Some(format!(
            "Driver {} is older than {}; update the NVIDIA driver",
            gpu.driver_version, min_driver_version
        ));
    }
    None
}

/// Parse `nvidia-smi --query-gpu=name,compute_cap,driver_version
/// --format=csv,noheader` output (first GPU only).
pub fn parse_nvidia_smi(output: &str) -> Option<GpuInfo> {
    let line = output.lines().map(str::trim).find(|line| !line.is_empty())?;
    let mut fields = line.rsplitn(3, ',').map(str::trim);
    let driver_version = fields.next()?.to_string();
    let compute_capability = fields.next()?.to_string();
    let name = fields.next()?.to_string();
    if name.is_empty()
        || version_key(&compute_capability).is_empty()
        || version_key(&driver_version).is_empty()
    {
        return None;
    }
    Some(GpuInfo {
        name,
        compute_capability,
        driver_version,
    })
}

/// Query the first NVIDIA GPU. `Ok(None)` when there is no NVIDIA driver
/// tooling at all; `Err` when `nvidia-smi` exists but cannot answer.
pub async fn detect_nvidia_gpu() -> Result<Option<GpuInfo>, String> {
    let mut command = Command::new(format!("nvidia-smi{}", std::env::consts::EXE_SUFFIX));
    command
        .args([
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_background(&mut command);
    let output = match tokio::time::timeout(DETECTION_TIMEOUT, command.output()).await {
        Err(_) => return Err("nvidia-smi did not answer".into()),
        Ok(Err(error)) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Ok(Err(error)) => return Err(error.to_string()),
        Ok(Ok(output)) => output,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.success() {
        if let Some(gpu) = parse_nvidia_smi(&stdout) {
            return Ok(Some(gpu));
        }
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let detail = [stdout.trim(), stderr.trim()]
        .into_iter()
        .find(|text| !text.is_empty())
        .and_then(|text| text.lines().next())
        .unwrap_or("no output")
        .to_string();
    if stdout.contains("No devices were found") {
        return Ok(None);
    }
    Err(detail)
}

enum PackSource {
    Remote(String),
    Local(PathBuf),
}

fn parse_source(raw: &str) -> Result<PackSource, GpuPackError> {
    let raw = raw.trim();
    if raw.starts_with("http://") || raw.starts_with("https://") {
        let mut base = raw.to_string();
        if !base.ends_with('/') {
            base.push('/');
        }
        return Ok(PackSource::Remote(base));
    }
    if raw.starts_with("file:") {
        let path = url::Url::parse(raw)
            .ok()
            .and_then(|url| url.to_file_path().ok())
            .ok_or_else(|| GpuPackError::Download(format!("invalid file URL {raw}")))?;
        return Ok(PackSource::Local(path));
    }
    Ok(PackSource::Local(PathBuf::from(raw)))
}

#[derive(Default)]
struct RemoteManifest {
    manifest: Option<GpuPackManifest>,
    failed_at: Option<Instant>,
}

/// Installs, reports and removes the pack under `<runtimes_root>/gpu-pack`.
pub struct GpuPackManager {
    runtimes_root: PathBuf,
    app_version: String,
    platform: Option<&'static str>,
    source: Option<String>,
    installing: AtomicBool,
    detection: tokio::sync::OnceCell<Result<Option<GpuInfo>, String>>,
    remote: tokio::sync::Mutex<RemoteManifest>,
    retry_delay: Duration,
    self_test: Option<SelfTestHook>,
}

struct InstallingGuard<'a>(&'a AtomicBool);

impl Drop for InstallingGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl GpuPackManager {
    /// `runtimes_root` is `<app_local_data>/runtimes`. Parts come from the
    /// GitHub release of `app_version` unless [`GPU_PACK_BASE_URL_ENV`] is set.
    pub fn new(runtimes_root: PathBuf, app_version: impl Into<String>) -> Self {
        Self {
            runtimes_root,
            app_version: app_version.into(),
            platform: pack_platform(),
            source: std::env::var(GPU_PACK_BASE_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty()),
            installing: AtomicBool::new(false),
            detection: tokio::sync::OnceCell::new(),
            remote: tokio::sync::Mutex::new(RemoteManifest::default()),
            retry_delay: Duration::from_secs(2),
            self_test: None,
        }
    }

    /// Read manifests and parts from `source` (URL prefix, `file://` URL or
    /// local directory) instead of the release.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Treat this build as `platform` (used by tests on other hosts).
    pub fn with_platform(mut self, platform: &'static str) -> Self {
        self.platform = Some(platform);
        self
    }

    #[cfg(test)]
    fn with_test_hooks(mut self, self_test: Option<SelfTestHook>) -> Self {
        self.retry_delay = Duration::from_millis(10);
        self.self_test = self_test;
        self
    }

    pub fn install_directory(&self) -> PathBuf {
        self.runtimes_root.join(INSTALL_DIRECTORY)
    }

    fn staging_directory(&self) -> PathBuf {
        self.runtimes_root.join(STAGING_DIRECTORY)
    }

    fn downloads_root(&self) -> PathBuf {
        self.runtimes_root.join(DOWNLOADS_DIRECTORY)
    }

    fn download_cache(&self) -> PathBuf {
        self.downloads_root().join(INSTALL_DIRECTORY)
    }

    pub fn is_installing(&self) -> bool {
        self.installing.load(Ordering::Acquire)
    }

    /// The installed record, if the pack directory has a readable one.
    pub fn record(&self) -> Option<GpuPackRecord> {
        read_record(&self.install_directory())
    }

    pub fn pack_state(&self) -> GpuPackState {
        self.installed().0
    }

    fn installed(&self) -> (GpuPackState, Option<GpuPackRecord>) {
        let directory = self.install_directory();
        let record = read_record(&directory);
        if self.is_installing() {
            return (GpuPackState::Installing, record);
        }
        if !directory.exists() {
            return (GpuPackState::NotInstalled, None);
        }
        let Some(record) = record else {
            return (GpuPackState::Corrupt, None);
        };
        let platform = self.platform.unwrap_or_default();
        if record
            .manifest
            .check_compatible(&self.app_version, platform)
            .is_err()
        {
            return (GpuPackState::UpdateRequired, Some(record));
        }
        if !pack_files_present(&directory) {
            return (GpuPackState::Corrupt, Some(record));
        }
        (GpuPackState::Ready, Some(record))
    }

    /// The pack runtime to use when the pack is ready and its self-test found
    /// a usable CUDA device.
    pub fn runtime_layout(&self) -> Option<GpuRuntimeLayout> {
        let (state, record) = self.installed();
        if state != GpuPackState::Ready || record?.cuda_available() != Some(true) {
            return None;
        }
        Some(pack_runtime_layout(&self.install_directory()))
    }

    /// The first NVIDIA GPU, detected once per app run.
    pub async fn detect_gpu(&self) -> Result<Option<GpuInfo>, String> {
        self.detection
            .get_or_init(|| async {
                if self.platform.is_none() {
                    Ok(None)
                } else {
                    detect_nvidia_gpu().await
                }
            })
            .await
            .clone()
    }

    /// The release manifest, fetched once per run (a failure is retried after
    /// five minutes). Never fails; `None` when it is unavailable.
    async fn cached_remote_manifest(&self) -> Option<GpuPackManifest> {
        let platform = self.platform?;
        let mut remote = self.remote.lock().await;
        if remote.manifest.is_some() {
            return remote.manifest.clone();
        }
        if remote
            .failed_at
            .is_some_and(|at| at.elapsed() < Duration::from_secs(300))
        {
            return None;
        }
        let fetched = match self.source() {
            Ok(source) => tokio::time::timeout(
                Duration::from_secs(10),
                self.fetch_manifest(&source, platform),
            )
            .await
            .ok()
            .and_then(Result::ok),
            Err(_) => None,
        };
        match fetched {
            Some(manifest) => remote.manifest = Some(manifest),
            None => remote.failed_at = Some(Instant::now()),
        }
        remote.manifest.clone()
    }

    /// Settings status. `enabled`, `active` and `fallback_reason` come from
    /// the app (preference and the running runtime).
    pub async fn status(
        &self,
        enabled: bool,
        active: bool,
        fallback_reason: Option<String>,
    ) -> GpuAccelerationStatus {
        let Some(platform) = self.platform else {
            return GpuAccelerationStatus {
                supported_platform: false,
                gpu: None,
                eligible: false,
                ineligible_reason: Some(GpuPackError::UnsupportedPlatform.to_string()),
                pack_state: GpuPackState::NotInstalled,
                pack_version: None,
                download_bytes: None,
                installed_bytes: None,
                cuda_available: None,
                device_name: None,
                enabled,
                active: false,
                fallback_reason: None,
            };
        };
        let detection = self.detect_gpu().await;
        let gpu = detection.as_ref().ok().cloned().flatten();
        let (state, record) = self.installed();
        let remote = if gpu.is_some() && state != GpuPackState::Ready {
            self.cached_remote_manifest().await
        } else {
            self.remote.lock().await.manifest.clone()
        };
        let manifest = match state {
            GpuPackState::Ready => record.as_ref().map(|record| &record.manifest),
            _ => remote.as_ref(),
        };
        let min_compute = manifest
            .map(|manifest| manifest.min_compute_capability.as_str())
            .unwrap_or(DEFAULT_MIN_COMPUTE_CAPABILITY);
        let min_driver = manifest
            .map(|manifest| manifest.min_driver_version_for(platform))
            .unwrap_or(DEFAULT_MIN_DRIVER_VERSION);
        let ineligible_reason = match &detection {
            Err(error) => Some(format!("The NVIDIA driver could not be queried: {error}")),
            Ok(gpu) => ineligibility(gpu.as_ref(), min_compute, min_driver),
        };
        GpuAccelerationStatus {
            supported_platform: true,
            gpu,
            eligible: ineligible_reason.is_none(),
            ineligible_reason,
            pack_state: state,
            pack_version: record
                .as_ref()
                .map(|record| record.manifest.app_version.clone()),
            download_bytes: manifest.map(GpuPackManifest::download_bytes),
            installed_bytes: record.as_ref().map(|record| record.manifest.unpacked_bytes),
            cuda_available: record.as_ref().and_then(GpuPackRecord::cuda_available),
            device_name: record.as_ref().and_then(GpuPackRecord::device_name),
            enabled,
            active,
            fallback_reason,
        }
    }

    fn source(&self) -> Result<PackSource, GpuPackError> {
        match &self.source {
            Some(raw) => parse_source(raw),
            None => Ok(PackSource::Remote(format!(
                "{RELEASE_DOWNLOAD_URL}/v{}/",
                self.app_version
            ))),
        }
    }

    fn manifest_name(&self, platform: &str) -> String {
        format!("echolingo-gpu-pack-{}-{platform}.json", self.app_version)
    }

    fn http_client(&self) -> Result<reqwest::Client, GpuPackError> {
        reqwest::Client::builder()
            .user_agent(format!("echolingo/{}", self.app_version))
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(|error| GpuPackError::Download(error.to_string()))
    }

    async fn fetch_manifest(
        &self,
        source: &PackSource,
        platform: &str,
    ) -> Result<GpuPackManifest, GpuPackError> {
        let name = self.manifest_name(platform);
        let bytes = match source {
            PackSource::Remote(base) => {
                let url = format!("{base}{name}");
                let response = self
                    .http_client()?
                    .get(&url)
                    .timeout(Duration::from_secs(60))
                    .send()
                    .await
                    .map_err(|error| GpuPackError::Download(format!("{url}: {error}")))?;
                if !response.status().is_success() {
                    return Err(GpuPackError::Download(format!(
                        "{url}: HTTP {}",
                        response.status()
                    )));
                }
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|error| GpuPackError::Download(format!("{url}: {error}")))?;
                if bytes.len() > MANIFEST_LIMIT_BYTES {
                    return Err(GpuPackError::Manifest("manifest is too large".into()));
                }
                bytes.to_vec()
            }
            PackSource::Local(directory) => tokio::fs::read(directory.join(&name))
                .await
                .map_err(|error| {
                    GpuPackError::Download(format!("{}: {error}", directory.join(&name).display()))
                })?,
        };
        GpuPackManifest::parse(&bytes)
    }

    /// Download (or read), verify, unpack, self-test and activate the pack
    /// for this app version. Progress is reported with `model_id`
    /// [`GPU_PACK_PROGRESS_ID`] and the phases `starting`, `downloading`,
    /// `verifying`, `extracting`, `testing` and `ready`. Downloaded parts are
    /// kept after a failure so the next attempt resumes.
    pub async fn install(&self, callback: ProgressCallback) -> Result<GpuPackRecord, GpuPackError> {
        let platform = self.platform.ok_or(GpuPackError::UnsupportedPlatform)?;
        if self.installing.swap(true, Ordering::AcqRel) {
            return Err(GpuPackError::AlreadyInstalling);
        }
        let _guard = InstallingGuard(&self.installing);
        let staging = self.staging_directory();
        let result = self.install_inner(platform, &staging, callback).await;
        if result.is_err() {
            let _ = remove_scoped_directory(&self.runtimes_root, &staging);
        }
        result
    }

    async fn install_inner(
        &self,
        platform: &'static str,
        staging: &Path,
        callback: ProgressCallback,
    ) -> Result<GpuPackRecord, GpuPackError> {
        let mut progress = Progress::new(callback.clone());
        progress.phase("starting", 0, 0);
        std::fs::create_dir_all(&self.runtimes_root)?;
        let source = self.source()?;
        let manifest = self.fetch_manifest(&source, platform).await?;
        manifest.check_compatible(&self.app_version, platform)?;
        self.remote.lock().await.manifest = Some(manifest.clone());

        let (parts, remaining_download) = match &source {
            PackSource::Local(directory) => (
                manifest
                    .parts
                    .iter()
                    .map(|part| directory.join(&part.name))
                    .collect::<Vec<_>>(),
                0,
            ),
            PackSource::Remote(_) => {
                let cache = self.download_cache();
                std::fs::create_dir_all(&cache)?;
                let downloaded: u64 = manifest
                    .parts
                    .iter()
                    .map(|part| {
                        std::fs::metadata(cache.join(&part.name))
                            .map(|metadata| metadata.len().min(part.bytes))
                            .unwrap_or(0)
                    })
                    .sum();
                (
                    manifest
                        .parts
                        .iter()
                        .map(|part| cache.join(&part.name))
                        .collect(),
                    manifest.archive_bytes.saturating_sub(downloaded),
                )
            }
        };
        let required = remaining_download
            .saturating_add(manifest.unpacked_bytes)
            .saturating_add(manifest.unpacked_bytes / 20);
        let available = fs2::available_space(&self.runtimes_root)?;
        if available < required {
            return Err(GpuPackError::DiskSpace {
                required_bytes: required,
                available_bytes: available,
            });
        }

        if let PackSource::Remote(base) = &source {
            let client = self.http_client()?;
            progress.phase(
                "downloading",
                manifest.archive_bytes - remaining_download,
                manifest.archive_bytes,
            );
            for (part, path) in manifest.parts.iter().zip(&parts) {
                self.download_part(&client, &format!("{base}{}", part.name), path, part, &mut progress)
                    .await?;
            }
        }

        progress.phase("verifying", 0, manifest.archive_bytes);
        for (part, path) in manifest.parts.iter().zip(&parts) {
            let expected = part.sha256.to_ascii_lowercase();
            let path_for_hash = path.clone();
            let size = part.bytes;
            let hashed = tokio::task::spawn_blocking(move || sha256_file(&path_for_hash, size))
                .await
                .map_err(|error| io::Error::other(error.to_string()))?;
            match hashed {
                Ok(digest) if digest == expected => progress.advance(size),
                Err(error) if matches!(source, PackSource::Local(_)) => {
                    return Err(GpuPackError::Download(format!("{}: {error}", path.display())));
                }
                Ok(_) | Err(_) => {
                    // A bad download is fetched again on the next attempt;
                    // local parts are never touched.
                    if matches!(source, PackSource::Remote(_)) {
                        let _ = std::fs::remove_file(path);
                    }
                    return Err(GpuPackError::CorruptPart(part.name.clone()));
                }
            }
        }

        progress.phase("extracting", 0, manifest.archive_bytes);
        remove_scoped_directory(&self.runtimes_root, staging)?;
        std::fs::create_dir_all(staging)?;
        let extract_parts = parts.clone();
        let extract_target = staging.to_path_buf();
        let archive_sha256 = manifest.archive_sha256.to_ascii_lowercase();
        let extract_progress = Arc::new(std::sync::Mutex::new(progress));
        let reporter = extract_progress.clone();
        tokio::task::spawn_blocking(move || {
            extract_archive(extract_parts, &archive_sha256, &extract_target, |bytes| {
                if let Ok(mut progress) = reporter.lock() {
                    progress.advance(bytes);
                }
            })
        })
        .await
        .map_err(|error| io::Error::other(error.to_string()))??;
        let mut progress = Arc::try_unwrap(extract_progress)
            .ok()
            .and_then(|progress| progress.into_inner().ok())
            .unwrap_or_else(|| Progress::new(callback.clone()));
        check_unpacked_pack(staging, &self.app_version, platform)?;

        progress.phase("testing", manifest.archive_bytes, manifest.archive_bytes);
        let self_test = match &self.self_test {
            Some(hook) => hook(staging)?,
            None => run_self_test(staging).await?,
        };
        let record = GpuPackRecord {
            manifest: manifest.clone(),
            installed_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            self_test,
        };
        write_record(staging, &record)?;
        self.activate(staging).await?;
        if matches!(source, PackSource::Remote(_)) {
            let _ = remove_scoped_directory(&self.downloads_root(), &self.download_cache());
        }
        progress.phase("ready", manifest.archive_bytes, manifest.archive_bytes);
        Ok(record)
    }

    /// Move the tested staging directory into place. Right after the
    /// self-test, antivirus scanners on Windows may still hold files of the
    /// staging tree open for a moment, so the rename is retried briefly.
    async fn activate(&self, staging: &Path) -> Result<(), GpuPackError> {
        let mut attempt = 1;
        loop {
            match activate_directory(&self.runtimes_root, staging, &self.install_directory()) {
                Ok(()) => return Ok(()),
                Err(_) if attempt < 10 => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    async fn download_part(
        &self,
        client: &reqwest::Client,
        url: &str,
        path: &Path,
        part: &GpuPackPart,
        progress: &mut Progress,
    ) -> Result<(), GpuPackError> {
        let mut attempt = 1;
        loop {
            match download_attempt(client, url, path, part, progress).await {
                Ok(()) => return Ok(()),
                Err(error) if attempt < DOWNLOAD_ATTEMPTS && error.is_transient() => {
                    attempt += 1;
                    tokio::time::sleep(self.retry_delay * attempt).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Delete the installed pack, an interrupted staging directory and any
    /// downloaded parts. Callers stop the services that use the pack first.
    pub fn remove(&self) -> Result<(), GpuPackError> {
        if self.is_installing() {
            return Err(GpuPackError::AlreadyInstalling);
        }
        if !self.runtimes_root.exists() {
            return Ok(());
        }
        remove_scoped_directory(&self.runtimes_root, &self.install_directory())?;
        remove_scoped_directory(&self.runtimes_root, &self.staging_directory())?;
        remove_scoped_directory(
            &self.runtimes_root,
            &self.runtimes_root.join(format!(".{INSTALL_DIRECTORY}.previous")),
        )?;
        if self.downloads_root().exists() {
            remove_scoped_directory(&self.downloads_root(), &self.download_cache())?;
        }
        Ok(())
    }
}

/// Where the runtime lives inside an unpacked pack.
pub fn pack_runtime_layout(pack: &Path) -> GpuRuntimeLayout {
    let suffix = std::env::consts::EXE_SUFFIX;
    GpuRuntimeLayout {
        sidecar: pack
            .join("sidecar")
            .join(format!("echolingo-sidecar{suffix}")),
        llama_server: pack
            .join("llama.cpp")
            .join(format!("llama-server{suffix}")),
    }
}

fn pack_files_present(pack: &Path) -> bool {
    let layout = pack_runtime_layout(pack);
    layout.sidecar.is_file() && layout.llama_server.is_file()
}

fn read_record(pack: &Path) -> Option<GpuPackRecord> {
    std::fs::read(pack.join(RECORD_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

fn write_record(pack: &Path, record: &GpuPackRecord) -> Result<(), GpuPackError> {
    let temporary = pack.join(format!("{RECORD_FILE}.tmp"));
    let bytes = serde_json::to_vec_pretty(record)
        .map_err(|error| GpuPackError::Manifest(error.to_string()))?;
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, pack.join(RECORD_FILE))?;
    Ok(())
}

/// One download attempt of `part` into `path`, resuming from the bytes
/// already on disk with an HTTP Range request.
async fn download_attempt(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    part: &GpuPackPart,
    progress: &mut Progress,
) -> Result<(), GpuPackError> {
    let mut existing = std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if existing > part.bytes {
        progress.retract(part.bytes);
        std::fs::remove_file(path)?;
        existing = 0;
    }
    if existing == part.bytes {
        return Ok(());
    }
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| GpuPackError::Download(format!("{url}: {error}")))?;
    let status = response.status();
    let resumed = existing > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        progress.retract(existing);
        std::fs::remove_file(path)?;
        return Err(GpuPackError::Download(format!("{url}: resume was refused")));
    }
    if !status.is_success() {
        return Err(GpuPackError::Download(format!("{url}: HTTP {status}")));
    }
    let mut file = if resumed {
        tokio::fs::OpenOptions::new().append(true).open(path).await?
    } else {
        // The server ignored the range: start this part again.
        progress.retract(existing);
        existing = 0;
        tokio::fs::File::create(path).await?
    };
    let mut written = existing;
    loop {
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(error) => {
                file.flush().await?;
                return Err(GpuPackError::Download(format!("{url}: {error}")));
            }
        };
        written += chunk.len() as u64;
        if written > part.bytes {
            drop(file);
            progress.retract(written - chunk.len() as u64);
            std::fs::remove_file(path)?;
            return Err(GpuPackError::CorruptPart(part.name.clone()));
        }
        file.write_all(&chunk).await?;
        progress.advance(chunk.len() as u64);
    }
    file.flush().await?;
    file.sync_all().await?;
    if written != part.bytes {
        return Err(GpuPackError::Download(format!(
            "{url}: connection closed after {written} of {} bytes",
            part.bytes
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path, expected_bytes: u64) -> io::Result<String> {
    let mut file = File::open(path)?;
    if file.metadata()?.len() != expected_bytes {
        return Err(io::Error::other("unexpected size"));
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// The parts read back to back as one archive stream, hashed on the way.
struct PartsReader<F: FnMut(u64)> {
    pending: VecDeque<PathBuf>,
    current: Option<File>,
    digest: Sha256,
    report: F,
}

impl<F: FnMut(u64)> Read for PartsReader<F> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.current.is_none() {
                match self.pending.pop_front() {
                    Some(path) => self.current = Some(File::open(path)?),
                    None => return Ok(0),
                }
            }
            let read = self
                .current
                .as_mut()
                .map_or(Ok(0), |file| file.read(buffer))?;
            if read == 0 {
                self.current = None;
                continue;
            }
            self.digest.update(&buffer[..read]);
            (self.report)(read as u64);
            return Ok(read);
        }
    }
}

/// Stream the parts through zstd and tar into `destination`, dropping the
/// archive's `echolingo-gpu-pack/` root, and check the whole-archive digest.
fn extract_archive(
    parts: Vec<PathBuf>,
    archive_sha256: &str,
    destination: &Path,
    report: impl FnMut(u64),
) -> Result<(), GpuPackError> {
    let reader = PartsReader {
        pending: parts.into(),
        current: None,
        digest: Sha256::new(),
        report,
    };
    let decoder = zstd::stream::read::Decoder::new(reader)
        .map_err(|error| GpuPackError::Archive(error.to_string()))?;
    let mut archive = tar::Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|error| GpuPackError::Archive(error.to_string()))?;
    // Symbolic links are created last, when their targets exist.
    let mut symlinks = Vec::new();
    for entry in entries {
        let mut entry = entry.map_err(|error| GpuPackError::Archive(error.to_string()))?;
        let path = entry
            .path()
            .map_err(|error| GpuPackError::Archive(error.to_string()))?
            .into_owned();
        let kind = entry.header().entry_type();
        if matches!(
            kind,
            tar::EntryType::XGlobalHeader | tar::EntryType::XHeader
        ) {
            continue;
        }
        let relative = pack_relative_path(&path)?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = destination.join(&relative);
        if kind.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if kind.is_hard_link() {
            let link = entry
                .link_name()
                .map_err(|error| GpuPackError::Archive(error.to_string()))?
                .ok_or_else(|| GpuPackError::Archive(format!("{} has no link target", path.display())))?;
            let source = destination.join(pack_relative_path(&link)?);
            if std::fs::hard_link(&source, &target).is_err() {
                std::fs::copy(&source, &target)?;
            }
            continue;
        }
        if kind.is_symlink() {
            let link = entry
                .link_name()
                .map_err(|error| GpuPackError::Archive(error.to_string()))?
                .ok_or_else(|| GpuPackError::Archive(format!("{} has no link target", path.display())))?
                .into_owned();
            // Only links that stay below their own directory (shared-library
            // aliases) are accepted.
            if !link
                .components()
                .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
            {
                return Err(GpuPackError::Archive(format!(
                    "{} links outside the pack",
                    path.display()
                )));
            }
            symlinks.push((target, link));
            continue;
        } else if !kind.is_file() && kind != tar::EntryType::Continuous {
            continue;
        }
        entry
            .unpack(&target)
            .map_err(|error| GpuPackError::Archive(format!("{}: {error}", path.display())))?;
    }
    for (target, link) in symlinks {
        create_symlink(&link, &target)?;
    }
    // Consume whatever follows the tar end marker so the digest covers the
    // whole archive.
    let mut rest = archive.into_inner().finish();
    io::copy(&mut rest, &mut io::sink())?;
    let digest = format!("{:x}", rest.into_inner().digest.finalize());
    if digest != archive_sha256 {
        return Err(GpuPackError::Archive(
            "the reassembled archive does not match its SHA-256 digest".into(),
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn create_symlink(link: &Path, target: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(link, target)
}

/// Symbolic links need a privilege on Windows; a copy of the file works the
/// same for a shared-library alias.
#[cfg(not(unix))]
fn create_symlink(link: &Path, target: &Path) -> io::Result<()> {
    let source = target.parent().unwrap_or(Path::new("")).join(link);
    #[cfg(windows)]
    if std::os::windows::fs::symlink_file(link, target).is_ok() {
        return Ok(());
    }
    std::fs::copy(source, target).map(|_| ())
}

/// The path of an archive entry below the `echolingo-gpu-pack/` root.
fn pack_relative_path(path: &Path) -> Result<PathBuf, GpuPackError> {
    let mut components = path
        .components()
        .filter(|component| !matches!(component, Component::CurDir));
    let outside = || GpuPackError::Archive(format!("{} is outside {ARCHIVE_ROOT}/", path.display()));
    match components.next() {
        Some(Component::Normal(root)) if root == ARCHIVE_ROOT => {}
        _ => return Err(outside()),
    }
    let mut relative = PathBuf::new();
    for component in components {
        match component {
            Component::Normal(part) => relative.push(part),
            _ => return Err(outside()),
        }
    }
    Ok(relative)
}

/// The unpacked pack must be the one this app asked for and contain both
/// runtimes.
fn check_unpacked_pack(pack: &Path, app_version: &str, platform: &str) -> Result<(), GpuPackError> {
    let inner: Value = std::fs::read(pack.join("pack.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .ok_or_else(|| GpuPackError::Archive("pack.json is missing or invalid".into()))?;
    let found_version = inner["app_version"].as_str().unwrap_or_default();
    if found_version != app_version {
        return Err(GpuPackError::VersionMismatch {
            expected: app_version.into(),
            found: found_version.into(),
        });
    }
    let found_platform = inner["platform"].as_str().unwrap_or_default();
    if found_platform != platform {
        return Err(GpuPackError::PlatformMismatch {
            expected: platform.into(),
            found: found_platform.into(),
        });
    }
    if !pack_files_present(pack) {
        return Err(GpuPackError::Archive(
            "the sidecar or llama-server is missing".into(),
        ));
    }
    Ok(())
}

/// Run `sidecar/echolingo-sidecar self-test --cuda` from the unpacked pack.
/// A machine without a usable GPU passes (with `cuda.available: false`); a
/// failed import does not.
async fn run_self_test(pack: &Path) -> Result<Value, GpuPackError> {
    let executable = pack_runtime_layout(pack).sidecar;
    let mut command = Command::new(&executable);
    command
        .args(["self-test", "--cuda"])
        .current_dir(executable.parent().unwrap_or(pack))
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_background(&mut command);
    let output = tokio::time::timeout(SELF_TEST_TIMEOUT, command.output())
        .await
        .map_err(|_| GpuPackError::SelfTest("timed out".into()))?
        .map_err(|error| GpuPackError::SelfTest(format!("cannot start the sidecar: {error}")))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = stdout
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
        .find(Value::is_object);
    let Some(report) = report else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .trim()
            .chars()
            .rev()
            .take(800)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(GpuPackError::SelfTest(format!(
            "no report ({}): {tail}",
            output.status
        )));
    };
    if !output.status.success() || report["ok"].as_bool() != Some(true) {
        let failed = report["checks"]
            .as_object()
            .map(|checks| {
                checks
                    .iter()
                    .filter(|(_, check)| check["ok"].as_bool() != Some(true))
                    .map(|(name, check)| {
                        format!("{name}: {}", check["detail"].as_str().unwrap_or("failed"))
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| format!("exit status {}", output.status));
        return Err(GpuPackError::SelfTest(failed));
    }
    Ok(report)
}

/// Throttled progress events with a transfer rate over the last interval.
struct Progress {
    callback: ProgressCallback,
    phase: &'static str,
    completed: u64,
    total: u64,
    last_emit: Instant,
    rate_window: (Instant, u64),
    bytes_per_second: Option<f64>,
}

impl Progress {
    fn new(callback: ProgressCallback) -> Self {
        let now = Instant::now();
        Self {
            callback,
            phase: "starting",
            completed: 0,
            total: 0,
            last_emit: now,
            rate_window: (now, 0),
            bytes_per_second: None,
        }
    }

    fn phase(&mut self, phase: &'static str, completed: u64, total: u64) {
        self.phase = phase;
        self.completed = completed;
        self.total = total;
        self.rate_window = (Instant::now(), completed);
        self.bytes_per_second = None;
        self.emit();
    }

    fn advance(&mut self, bytes: u64) {
        self.completed = self.completed.saturating_add(bytes).min(self.total);
        if self.last_emit.elapsed() >= PROGRESS_INTERVAL {
            let (since, at) = self.rate_window;
            let elapsed = since.elapsed().as_secs_f64();
            if elapsed >= 1.0 {
                self.bytes_per_second =
                    Some(self.completed.saturating_sub(at) as f64 / elapsed);
                self.rate_window = (Instant::now(), self.completed);
            }
            self.emit();
        }
    }

    /// Bytes counted earlier that have to be downloaded again.
    fn retract(&mut self, bytes: u64) {
        self.completed = self.completed.saturating_sub(bytes);
        self.rate_window.1 = self.rate_window.1.min(self.completed);
    }

    fn emit(&mut self) {
        self.last_emit = Instant::now();
        (self.callback)(ModelProgress {
            model_id: GPU_PACK_PROGRESS_ID.into(),
            bytes_completed: self.completed,
            total_bytes: self.total,
            bytes_per_second: self.bytes_per_second,
            phase: self.phase.into(),
        });
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains(['/', '\\', ':'])
        && Path::new(name)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Numeric components of a dotted version (`"580.65.06"` → `[580, 65, 6]`).
fn version_key(version: &str) -> Vec<u64> {
    version
        .trim()
        .split('.')
        .map_while(|part| {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            digits.parse::<u64>().ok()
        })
        .collect()
}

fn version_less(version: &str, minimum: &str) -> bool {
    let (mut left, mut right) = (version_key(version), version_key(minimum));
    let length = left.len().max(right.len());
    left.resize(length, 0);
    right.resize(length, 0);
    left < right
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    const VERSION: &str = "0.2.0";
    const PLATFORM: &str = "linux-x64";

    struct Fixture {
        manifest: GpuPackManifest,
        parts: Vec<(String, Vec<u8>)>,
        manifest_name: String,
    }

    fn sidecar_name() -> String {
        format!("echolingo-sidecar{}", std::env::consts::EXE_SUFFIX)
    }

    fn llama_name() -> String {
        format!("llama-server{}", std::env::consts::EXE_SUFFIX)
    }

    fn add_file(builder: &mut tar::Builder<Vec<u8>>, path: &str, contents: &[u8], mode: u32) {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
    }

    /// A tiny pack: tar → zstd → parts of `part_bytes`, with the manifest the
    /// release would carry. `sidecar` becomes `sidecar/echolingo-sidecar`.
    fn build_fixture(app_version: &str, platform: &str, part_bytes: usize, sidecar: &[u8]) -> Fixture {
        let mut builder = tar::Builder::new(Vec::new());
        let mut directory = tar::Header::new_gnu();
        directory.set_entry_type(tar::EntryType::Directory);
        directory.set_mode(0o755);
        directory.set_size(0);
        directory.set_cksum();
        builder
            .append_data(&mut directory.clone(), "echolingo-gpu-pack/", io::empty())
            .unwrap();
        builder
            .append_data(&mut directory, "echolingo-gpu-pack/sidecar/", io::empty())
            .unwrap();
        let pack_json = json!({
            "schema_version": 1,
            "app_version": app_version,
            "platform": platform,
            "torch_version": "2.13.0+cu130",
            "cuda_version": "13.0",
        });
        add_file(
            &mut builder,
            "echolingo-gpu-pack/pack.json",
            pack_json.to_string().as_bytes(),
            0o644,
        );
        add_file(
            &mut builder,
            &format!("echolingo-gpu-pack/sidecar/{}", sidecar_name()),
            sidecar,
            0o755,
        );
        // Incompressible payload so the archive spans several parts.
        let mut state = 0x2545_f491_u32;
        let payload: Vec<u8> = (0..24_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        add_file(
            &mut builder,
            "echolingo-gpu-pack/sidecar/_internal/torch_cuda.bin",
            &payload,
            0o644,
        );
        add_file(
            &mut builder,
            &format!("echolingo-gpu-pack/llama.cpp/{}", llama_name()),
            b"llama",
            0o755,
        );
        add_file(
            &mut builder,
            "echolingo-gpu-pack/llama.cpp/libggml-vulkan.so.0",
            b"library",
            0o644,
        );
        let mut link = tar::Header::new_gnu();
        link.set_entry_type(tar::EntryType::Symlink);
        link.set_size(0);
        link.set_mode(0o777);
        builder
            .append_link(
                &mut link,
                "echolingo-gpu-pack/llama.cpp/libggml-vulkan.so",
                "libggml-vulkan.so.0",
            )
            .unwrap();
        let archive = zstd::stream::encode_all(&builder.into_inner().unwrap()[..], 3).unwrap();
        let archive_name = format!("echolingo-gpu-pack-{app_version}-{platform}.tar.zst");
        let parts: Vec<(String, Vec<u8>)> = archive
            .chunks(part_bytes)
            .enumerate()
            .map(|(index, chunk)| (format!("{archive_name}.part{:02}", index + 1), chunk.to_vec()))
            .collect();
        let manifest = GpuPackManifest {
            schema_version: 1,
            app_version: app_version.into(),
            platform: platform.into(),
            torch_version: "2.13.0+cu130".into(),
            cuda_version: "13.0".into(),
            min_driver_version: "580.65".into(),
            min_driver_version_windows: None,
            min_driver_version_linux: None,
            min_compute_capability: "7.5".into(),
            archive: archive_name,
            archive_bytes: archive.len() as u64,
            archive_sha256: format!("{:x}", Sha256::digest(&archive)),
            unpacked_bytes: 64 * 1024,
            parts: parts
                .iter()
                .map(|(name, bytes)| GpuPackPart {
                    name: name.clone(),
                    bytes: bytes.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                })
                .collect(),
        };
        Fixture {
            manifest_name: format!("echolingo-gpu-pack-{app_version}-{platform}.json"),
            manifest,
            parts,
        }
    }

    fn write_fixture(fixture: &Fixture, directory: &Path) {
        std::fs::create_dir_all(directory).unwrap();
        for (name, bytes) in &fixture.parts {
            std::fs::write(directory.join(name), bytes).unwrap();
        }
        std::fs::write(
            directory.join(&fixture.manifest_name),
            serde_json::to_vec(&fixture.manifest).unwrap(),
        )
        .unwrap();
    }

    fn cuda_self_test() -> Option<SelfTestHook> {
        Some(Arc::new(|pack: &Path| {
            assert!(pack_runtime_layout(pack).sidecar.is_file());
            Ok(json!({
                "ok": true,
                "version": VERSION,
                "checks": {"torch": {"ok": true, "detail": "2.13.0+cu130"}},
                "cuda": {"available": true, "device_name": "Test GPU", "capability": "8.6"}
            }))
        }))
    }

    fn recorder() -> (ProgressCallback, Arc<Mutex<Vec<ModelProgress>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        (
            Arc::new(move |progress| sink.lock().unwrap().push(progress)),
            events,
        )
    }

    fn phases(events: &Mutex<Vec<ModelProgress>>) -> Vec<String> {
        let mut phases: Vec<String> = Vec::new();
        for event in events.lock().unwrap().iter() {
            assert_eq!(event.model_id, GPU_PACK_PROGRESS_ID);
            if phases.last() != Some(&event.phase) {
                phases.push(event.phase.clone());
            }
        }
        phases
    }

    #[test]
    fn manifest_parsing_validates_parts_and_digests() {
        let fixture = build_fixture(VERSION, PLATFORM, 4096, b"sidecar");
        let bytes = serde_json::to_vec(&fixture.manifest).unwrap();
        let parsed = GpuPackManifest::parse(&bytes).unwrap();
        assert_eq!(parsed, fixture.manifest);
        assert!(parsed.parts.len() > 2);
        assert_eq!(parsed.download_bytes(), parsed.archive_bytes);
        assert_eq!(parsed.min_driver_version_for("windows-x64"), "580.65");

        let mut value = serde_json::to_value(&fixture.manifest).unwrap();
        value["min_driver_version_windows"] = json!("581.15");
        let parsed = GpuPackManifest::parse(value.to_string().as_bytes()).unwrap();
        assert_eq!(parsed.min_driver_version_for("windows-x64"), "581.15");
        assert_eq!(parsed.min_driver_version_for("linux-x64"), "580.65");

        let invalid = |edit: &dyn Fn(&mut Value)| {
            let mut value = serde_json::to_value(&fixture.manifest).unwrap();
            edit(&mut value);
            GpuPackManifest::parse(value.to_string().as_bytes()).unwrap_err()
        };
        assert!(matches!(
            invalid(&|value| value["parts"][0]["name"] = json!("../escape.part01")),
            GpuPackError::Manifest(_)
        ));
        assert!(matches!(
            invalid(&|value| value["parts"][0]["sha256"] = json!("abc")),
            GpuPackError::Manifest(_)
        ));
        assert!(matches!(
            invalid(&|value| value["archive_bytes"] = json!(1)),
            GpuPackError::Manifest(_)
        ));
        assert!(matches!(
            invalid(&|value| value["schema_version"] = json!(2)),
            GpuPackError::Manifest(_)
        ));
        assert!(matches!(
            invalid(&|value| value["parts"] = json!([])),
            GpuPackError::Manifest(_)
        ));
        assert!(matches!(
            fixture.manifest.check_compatible("0.3.0", PLATFORM),
            Err(GpuPackError::VersionMismatch { .. })
        ));
        assert!(matches!(
            fixture.manifest.check_compatible(VERSION, "windows-x64"),
            Err(GpuPackError::PlatformMismatch { .. })
        ));
    }

    #[test]
    fn eligibility_needs_turing_or_newer_and_an_r580_driver() {
        let gpu = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 8.6, 581.29\n").unwrap();
        assert_eq!(
            gpu,
            GpuInfo {
                name: "NVIDIA GeForce RTX 3060".into(),
                compute_capability: "8.6".into(),
                driver_version: "581.29".into(),
            }
        );
        assert_eq!(ineligibility(Some(&gpu), "7.5", "580.65"), None);
        assert_eq!(
            ineligibility(None, "7.5", "580"),
            Some("No NVIDIA GPU detected".into())
        );
        let pascal = parse_nvidia_smi("NVIDIA GeForce GTX 1080, 6.1, 581.29").unwrap();
        assert!(ineligibility(Some(&pascal), "7.5", "580")
            .unwrap()
            .contains("compute capability 6.1"));
        let old_driver = parse_nvidia_smi("Tesla T4, 7.5, 551.86").unwrap();
        assert_eq!(
            ineligibility(Some(&old_driver), "7.5", "580.65"),
            Some("Driver 551.86 is older than 580.65; update the NVIDIA driver".into())
        );
        // Linux drivers carry three components; Blackwell reports 12.0.
        let linux = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 12.0, 580.65.06").unwrap();
        assert_eq!(ineligibility(Some(&linux), "7.5", "580.65"), None);
        assert!(version_less("580.64.99", "580.65"));
        assert!(!version_less("580.65.06", "580.65"));
        assert!(!version_less("580", "580"));
        assert!(version_less("7.0", "7.5"));
        // A comma in the model name is kept; garbage is rejected.
        let odd = parse_nvidia_smi("NVIDIA RTX A2000, 12GB, 8.6, 581.29").unwrap();
        assert_eq!(odd.name, "NVIDIA RTX A2000, 12GB");
        assert_eq!(parse_nvidia_smi("Field \"compute_cap\" is not a valid field"), None);
        assert_eq!(parse_nvidia_smi(""), None);
    }

    #[tokio::test]
    async fn installs_from_a_local_directory_of_parts() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = build_fixture(VERSION, PLATFORM, 4096, b"sidecar");
        let source = directory.path().join("release parts ü");
        write_fixture(&fixture, &source);
        let runtimes = directory.path().join("runtimes");
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(source.to_string_lossy())
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
        assert!(manager.runtime_layout().is_none());

        let (callback, events) = recorder();
        let record = manager.install(callback).await.unwrap();

        assert_eq!(
            phases(&events),
            ["starting", "verifying", "extracting", "testing", "ready"]
        );
        let last = events.lock().unwrap().last().cloned().unwrap();
        assert_eq!(last.bytes_completed, fixture.manifest.archive_bytes);
        assert_eq!(record.manifest, fixture.manifest);
        assert_eq!(record.cuda_available(), Some(true));
        assert_eq!(manager.pack_state(), GpuPackState::Ready);
        let pack = runtimes.join("gpu-pack");
        assert_eq!(manager.record().unwrap(), record);
        let layout = manager.runtime_layout().unwrap();
        assert_eq!(layout, pack_runtime_layout(&pack));
        assert_eq!(std::fs::read(&layout.sidecar).unwrap(), b"sidecar");
        assert_eq!(std::fs::read(&layout.llama_server).unwrap(), b"llama");
        assert_eq!(
            std::fs::metadata(pack.join("sidecar/_internal/torch_cuda.bin"))
                .unwrap()
                .len(),
            24_000
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::read_link(pack.join("llama.cpp/libggml-vulkan.so")).unwrap(),
                Path::new("libggml-vulkan.so.0")
            );
            let mode = std::fs::metadata(&layout.sidecar).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111);
        }
        assert!(!runtimes.join(".gpu-pack.installing").exists());
        // Local parts are read in place, never copied or deleted.
        assert!(!runtimes.join(".downloads").exists());
        assert!(source.join(&fixture.parts[0].0).is_file());

        let status = manager.status(true, true, None).await;
        assert!(status.supported_platform);
        assert_eq!(status.pack_state, GpuPackState::Ready);
        assert_eq!(status.pack_version.as_deref(), Some(VERSION));
        assert_eq!(status.cuda_available, Some(true));
        assert_eq!(status.device_name.as_deref(), Some("Test GPU"));
        assert_eq!(status.download_bytes, Some(fixture.manifest.archive_bytes));
        let serialized = serde_json::to_value(&status).unwrap();
        assert_eq!(serialized["pack_state"], "ready");
        assert!(serialized.get("fallback_reason").is_some());

        // A newer app needs a new pack.
        let newer = GpuPackManager::new(runtimes.clone(), "0.3.0").with_platform(PLATFORM);
        assert_eq!(newer.pack_state(), GpuPackState::UpdateRequired);
        assert!(newer.runtime_layout().is_none());

        // A pack whose runtime vanished is corrupt.
        std::fs::remove_file(&layout.llama_server).unwrap();
        assert_eq!(manager.pack_state(), GpuPackState::Corrupt);

        manager.remove().unwrap();
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
        assert!(!pack.exists());
    }

    #[tokio::test]
    async fn a_corrupt_part_is_rejected_and_nothing_is_activated() {
        let directory = tempfile::tempdir().unwrap();
        let mut fixture = build_fixture(VERSION, PLATFORM, 4096, b"sidecar");
        fixture.parts[1].1[10] ^= 0xff;
        let source = directory.path().join("parts");
        write_fixture(&fixture, &source);
        let runtimes = directory.path().join("runtimes");
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(url::Url::from_file_path(&source).unwrap().to_string())
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        let (callback, _) = recorder();

        let error = manager.install(callback).await.unwrap_err();

        assert!(
            matches!(&error, GpuPackError::CorruptPart(name) if *name == fixture.parts[1].0),
            "{error}"
        );
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
        assert!(!runtimes.join(".gpu-pack.installing").exists());
        assert!(!manager.is_installing());
    }

    #[tokio::test]
    async fn packs_for_another_version_or_platform_are_refused() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("parts");
        // The release manifest names the wrong app version.
        let fixture = build_fixture("0.1.9", PLATFORM, 4096, b"sidecar");
        write_fixture(&fixture, &source);
        std::fs::rename(
            source.join(&fixture.manifest_name),
            source.join(format!("echolingo-gpu-pack-{VERSION}-{PLATFORM}.json")),
        )
        .unwrap();
        let runtimes = directory.path().join("runtimes");
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(source.to_string_lossy())
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        let (callback, _) = recorder();
        assert!(matches!(
            manager.install(callback.clone()).await,
            Err(GpuPackError::VersionMismatch { found, .. }) if found == "0.1.9"
        ));

        // The manifest is right but the unpacked pack is not.
        let mut fixture = build_fixture(VERSION, "windows-x64", 4096, b"sidecar");
        fixture.manifest.platform = PLATFORM.into();
        fixture.manifest_name = format!("echolingo-gpu-pack-{VERSION}-{PLATFORM}.json");
        let source = directory.path().join("mislabelled");
        write_fixture(&fixture, &source);
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(source.to_string_lossy())
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        assert!(matches!(
            manager.install(callback).await,
            Err(GpuPackError::PlatformMismatch { found, .. }) if found == "windows-x64"
        ));
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
        assert!(!runtimes.join(".gpu-pack.installing").exists());
    }

    type RequestLog = Arc<Mutex<Vec<(String, Option<String>)>>>;

    /// A minimal HTTP/1.1 file server with Range support. The first request
    /// for `cut.0` is closed after `cut.1` body bytes.
    async fn serve(files: HashMap<String, Vec<u8>>, cut: Option<(String, usize)>) -> (String, RequestLog) {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let base = format!("http://{}/", listener.local_addr().unwrap());
        let log: RequestLog = Arc::new(Mutex::new(Vec::new()));
        let files = Arc::new(files);
        let cut = Arc::new(Mutex::new(cut));
        let requests = log.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let files = files.clone();
                let requests = requests.clone();
                let cut = cut.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0_u8; 1024];
                    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                        match stream.read(&mut buffer).await {
                            Ok(0) | Err(_) => return,
                            Ok(read) => request.extend_from_slice(&buffer[..read]),
                        }
                    }
                    let text = String::from_utf8_lossy(&request).to_string();
                    let path = text
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .trim_start_matches('/')
                        .to_string();
                    let range = text.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("range")
                            .then(|| value.trim().to_string())
                    });
                    requests.lock().unwrap().push((path.clone(), range.clone()));
                    let Some(body) = files.get(&path) else {
                        let _ = stream
                            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                            .await;
                        return;
                    };
                    let start = range
                        .as_deref()
                        .and_then(|range| range.strip_prefix("bytes="))
                        .and_then(|range| range.strip_suffix('-'))
                        .and_then(|start| start.parse::<usize>().ok());
                    let mut head = match start {
                        Some(start) => format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\nContent-Length: {}\r\n",
                            body.len() - 1,
                            body.len(),
                            body.len() - start
                        ),
                        None => format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n", body.len()),
                    };
                    head.push_str("Connection: close\r\n\r\n");
                    let body = &body[start.unwrap_or(0)..];
                    let limit = {
                        let mut cut = cut.lock().unwrap();
                        match cut.as_ref() {
                            Some((name, bytes)) if *name == path => {
                                let bytes = *bytes;
                                *cut = None;
                                bytes
                            }
                            _ => body.len(),
                        }
                    };
                    let _ = stream.write_all(head.as_bytes()).await;
                    let _ = stream.write_all(&body[..limit.min(body.len())]).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        (base, log)
    }

    #[tokio::test]
    async fn http_downloads_resume_with_range_requests() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = build_fixture(VERSION, PLATFORM, 4096, b"sidecar");
        let mut files: HashMap<String, Vec<u8>> = fixture.parts.iter().cloned().collect();
        files.insert(
            fixture.manifest_name.clone(),
            serde_json::to_vec(&fixture.manifest).unwrap(),
        );
        let first = fixture.parts[0].0.clone();
        let second = fixture.parts[1].0.clone();
        let (base, requests) = serve(files, Some((first.clone(), 1_000))).await;
        let runtimes = directory.path().join("runtimes");
        // An earlier run left half of part 2 behind.
        let cache = runtimes.join(".downloads").join("gpu-pack");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join(&second), &fixture.parts[1].1[..1_500]).unwrap();
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(base.trim_end_matches('/'))
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        let (callback, events) = recorder();

        let record = manager.install(callback).await.unwrap();

        assert_eq!(record.manifest, fixture.manifest);
        let requests = requests.lock().unwrap().clone();
        let for_part = |name: &str| {
            requests
                .iter()
                .filter(|(path, _)| path == name)
                .map(|(_, range)| range.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(for_part(&first), [None, Some("bytes=1000-".to_string())]);
        assert_eq!(for_part(&second), [Some("bytes=1500-".to_string())]);
        assert_eq!(
            phases(&events),
            ["starting", "downloading", "verifying", "extracting", "testing", "ready"]
        );
        assert_eq!(manager.pack_state(), GpuPackState::Ready);
        // Parts are discarded once the pack is active.
        assert!(!cache.exists());
    }

    #[tokio::test]
    async fn a_missing_release_is_reported_without_touching_the_install() {
        let directory = tempfile::tempdir().unwrap();
        let (base, _) = serve(HashMap::new(), None).await;
        let manager = GpuPackManager::new(directory.path().join("runtimes"), VERSION)
            .with_source(base)
            .with_platform(PLATFORM)
            .with_test_hooks(cuda_self_test());
        let (callback, _) = recorder();
        let error = manager.install(callback).await.unwrap_err();
        assert!(matches!(&error, GpuPackError::Download(message) if message.contains("404")));
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_unpacked_sidecar_runs_its_cuda_self_test() {
        let directory = tempfile::tempdir().unwrap();
        let report = json!({
            "ok": true, "version": VERSION, "platform": "linux", "frozen": true,
            "utf8_mode": true, "ssl_ca_certs": 140,
            "checks": {"torch": {"ok": true, "detail": "2.13.0+cu130"}},
            "cuda": {"available": false, "device_name": null, "error": null}
        });
        let script = format!(
            "#!/bin/sh\n[ \"$1 $2\" = 'self-test --cuda' ] || exit 9\n[ \"$PYTHONUTF8\" = 1 ] || exit 8\necho 'loading torch' >&2\necho '{report}'\n"
        );
        let fixture = build_fixture(VERSION, PLATFORM, 8192, script.as_bytes());
        let source = directory.path().join("parts");
        write_fixture(&fixture, &source);
        let manager = GpuPackManager::new(directory.path().join("runtimes"), VERSION)
            .with_source(source.to_string_lossy())
            .with_platform(PLATFORM);
        let (callback, _) = recorder();
        let record = manager.install(callback).await.unwrap();
        assert_eq!(record.self_test, report);
        assert_eq!(record.cuda_available(), Some(false));
        // Installed, but without a usable GPU the CPU runtime stays in use.
        assert_eq!(manager.pack_state(), GpuPackState::Ready);
        assert!(manager.runtime_layout().is_none());

        let failing = json!({
            "ok": false,
            "checks": {"qwen_asr": {"ok": false, "detail": "No module named qwen_asr"}},
            "cuda": null
        });
        let script = format!("#!/bin/sh\necho '{failing}'\nexit 1\n");
        let fixture = build_fixture(VERSION, PLATFORM, 8192, script.as_bytes());
        let source = directory.path().join("broken");
        write_fixture(&fixture, &source);
        let runtimes = directory.path().join("broken-runtimes");
        let manager = GpuPackManager::new(runtimes.clone(), VERSION)
            .with_source(source.to_string_lossy())
            .with_platform(PLATFORM);
        let (callback, _) = recorder();
        let error = manager.install(callback).await.unwrap_err();
        assert!(
            matches!(&error, GpuPackError::SelfTest(detail) if detail.contains("No module named qwen_asr")),
            "{error}"
        );
        assert_eq!(manager.pack_state(), GpuPackState::NotInstalled);
        assert!(!runtimes.join(".gpu-pack.installing").exists());
    }

    #[tokio::test]
    async fn status_hides_the_pack_where_it_does_not_exist() {
        let directory = tempfile::tempdir().unwrap();
        let mut manager = GpuPackManager::new(directory.path().to_path_buf(), VERSION);
        manager.platform = None;
        let status = manager.status(true, false, None).await;
        assert!(!status.supported_platform);
        assert!(!status.eligible);
        assert_eq!(status.pack_state, GpuPackState::NotInstalled);
        let (callback, _) = recorder();
        assert!(matches!(
            manager.install(callback).await,
            Err(GpuPackError::UnsupportedPlatform)
        ));
    }

    /// End-to-end install of the real CI-built pack: the parts and manifest
    /// of this version for this platform in `ECHOLINGO_GPU_PACK_TEST_DIR`.
    #[tokio::test]
    #[ignore = "needs the CI-built GPU pack parts in ECHOLINGO_GPU_PACK_TEST_DIR"]
    async fn gpu_pack_installs_from_local_parts() {
        let source = std::env::var("ECHOLINGO_GPU_PACK_TEST_DIR")
            .expect("set ECHOLINGO_GPU_PACK_TEST_DIR to the directory with the pack parts");
        assert!(
            pack_platform().is_some(),
            "GPU packs exist for Windows and Linux x64 only"
        );
        let directory = tempfile::tempdir().unwrap();
        let manager = GpuPackManager::new(
            directory.path().join("runtimes"),
            env!("CARGO_PKG_VERSION"),
        )
        .with_source(source);
        let record = manager
            .install(Arc::new(|progress: ModelProgress| {
                eprintln!(
                    "gpu-pack {} {}/{}",
                    progress.phase, progress.bytes_completed, progress.total_bytes
                );
            }))
            .await
            .unwrap();
        println!("GPU_PACK_SELF_TEST={}", record.self_test);
        assert_eq!(record.self_test["ok"], true);
        assert!(record.self_test["cuda"].is_object());
        assert_eq!(manager.pack_state(), GpuPackState::Ready);
        let layout = pack_runtime_layout(&manager.install_directory());
        assert!(layout.sidecar.is_file() && layout.llama_server.is_file());
        assert_eq!(
            manager.runtime_layout().is_some(),
            record.cuda_available() == Some(true)
        );
    }
}
