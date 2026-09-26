//! Managed, pinned local-model installation for the desktop product.

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::HFClient;
use process_support::ProcessTree;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::Mutex as AsyncMutex;

pub mod gpu_pack;

pub const MODEL_CATALOG_VERSION: u16 = 1;
/// SIGTERM-to-SIGKILL grace when a local model service is stopped.
const SERVICE_STOP_GRACE: Duration = Duration::from_secs(2);
/// Model download attempts per install, the first included. Only a stalled
/// attempt is retried; a download error ends the install at once.
const DOWNLOAD_ATTEMPTS: u32 = 3;
/// How long a model download may go without receiving data before the
/// attempt is abandoned (at least; see [`StallClock`]), in seconds: the
/// default and the range `ECHOLINGO_MODEL_STALL_SECONDS` may choose from.
const DEFAULT_STALL_SECONDS: u64 = 60;
const MIN_STALL_SECONDS: u64 = 5;
const MAX_STALL_SECONDS: u64 = 600;
const STALL_SECONDS_ENVIRONMENT: &str = "ECHOLINGO_MODEL_STALL_SECONDS";
/// Xet transfers (most Hugging Face files) report bytes only once a whole
/// term of up to 64 MB is written, and fetch several terms at once, so on a
/// slow link data flows long before the first report. Until the first one
/// the allowed silence is this many stall timeouts.
const FIRST_DATA_STALL_FACTOR: u32 = 5;
/// For the same reason the allowed silence is at least this many times the
/// longest silence the attempt has already come back from.
const STALL_GAP_FACTOR: u32 = 3;
/// How often the stall watchdog looks at the clock at most.
const STALL_CHECK_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Asr,
    Translation,
    Alignment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DownloadKind {
    Snapshot { allow_patterns: Vec<String> },
    File { filename: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    pub role: ModelRole,
    pub repository: String,
    pub revision: String,
    pub expected_bytes: u64,
    pub required_file: String,
    pub required_file_sha256: String,
    download: DownloadKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInstallState {
    NotDownloaded,
    Installing,
    Ready,
    Corrupt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub display_name: String,
    pub role: ModelRole,
    pub size_bytes: u64,
    pub state: ModelInstallState,
    pub path: PathBuf,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelProgress {
    pub model_id: String,
    pub bytes_completed: u64,
    pub total_bytes: u64,
    pub bytes_per_second: Option<f64>,
    pub phase: String,
    /// Why an install failed (`phase: "failed"`), or why a model download
    /// started over (on every event of the new attempt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstallManifest {
    schema_version: u16,
    model_id: String,
    repository: String,
    revision: String,
    required_file: String,
    required_file_sha256: String,
}

#[derive(Debug, Error)]
pub enum ModelManagerError {
    #[error("unknown local model: {0}")]
    UnknownModel(String),
    #[error("model installation is already running: {0}")]
    AlreadyInstalling(String),
    #[error("not enough disk space: need {required_bytes} bytes, have {available_bytes} bytes")]
    DiskSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("model download failed: {0}")]
    Download(String),
    #[error("model integrity check failed for {0}")]
    Integrity(String),
    #[error("model filesystem operation failed: {0}")]
    Io(#[from] io::Error),
    #[error("model metadata is invalid: {0}")]
    Metadata(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCommand {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub environment: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenStreamingProfile {
    pub chunk_seconds: f32,
    pub stable_iterations: u8,
    pub hold_back_words: u16,
    pub left_context_seconds: f32,
    pub right_context_ms: u32,
    pub segment_max_steps: u32,
}

impl Default for QwenStreamingProfile {
    fn default() -> Self {
        Self {
            chunk_seconds: 1.0,
            stable_iterations: 1,
            hold_back_words: 4,
            left_context_seconds: 12.0,
            right_context_ms: 640,
            segment_max_steps: 200,
        }
    }
}

impl QwenStreamingProfile {
    pub fn from_environment() -> Self {
        let mut value = Self::default();
        value.chunk_seconds = environment_number(
            "ECHOLINGO_QWEN_STREAMING_CHUNK_SEC",
            value.chunk_seconds,
            0.5,
            4.0,
        );
        value.stable_iterations = environment_number(
            "ECHOLINGO_QWEN_STREAMING_STABLE_ITERATIONS",
            value.stable_iterations,
            1,
            4,
        );
        value.hold_back_words = environment_number(
            "ECHOLINGO_QWEN_STREAMING_HOLD_BACK_WORDS",
            value.hold_back_words,
            0,
            20,
        );
        value.left_context_seconds = environment_number(
            "ECHOLINGO_QWEN_STREAMING_LEFT_CONTEXT_SEC",
            value.left_context_seconds,
            4.0,
            30.0,
        );
        value.right_context_ms = environment_number(
            "ECHOLINGO_QWEN_STREAMING_RIGHT_CONTEXT_MS",
            value.right_context_ms,
            0,
            2_000,
        );
        value.segment_max_steps = environment_number(
            "ECHOLINGO_QWEN_STREAMING_SEGMENT_MAX_STEPS",
            value.segment_max_steps,
            50,
            400,
        );
        value
    }
}

fn environment_number<T>(name: &str, default: T, minimum: T, maximum: T) -> T
where
    T: std::str::FromStr + PartialOrd + Copy,
{
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<T>().ok())
        .filter(|candidate| *candidate >= minimum && *candidate <= maximum)
        .unwrap_or(default)
}

/// The parts of the NVIDIA acceleration pack (see [`gpu_pack`]) that passed
/// their install checks, used in place of the bundled CPU runtime per
/// service while selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRuntimeLayout {
    /// The pack's CUDA build of the sidecar, run as `qwen-asr-server`.
    pub sidecar: Option<PathBuf>,
    /// The pack's Vulkan `llama-server`.
    pub llama_server: Option<PathBuf>,
}

impl GpuRuntimeLayout {
    fn executable_for(&self, service: &str) -> Option<&Path> {
        match service {
            "qwen_asr" => self.sidecar.as_deref(),
            "hymt" => self.llama_server.as_deref(),
            _ => None,
        }
    }
}

const LOCAL_SERVICES: [&str; 2] = ["qwen_asr", "hymt"];

#[derive(Debug, Clone)]
pub struct LocalRuntimeLayout {
    pub qwen_command: RuntimeCommand,
    pub llama_server: Option<PathBuf>,
    pub worker_wrapper: Option<PathBuf>,
    pub model_root: PathBuf,
    pub log_root: PathBuf,
    pub qwen_device: String,
    pub qwen_streaming: QwenStreamingProfile,
    pub local_api_key: String,
    pub qwen_port: u16,
    pub hymt_port: u16,
    pub startup_timeout: Duration,
}

#[derive(Debug, Error)]
pub enum LocalRuntimeError {
    #[error("unknown local runtime service: {0}")]
    UnknownService(String),
    #[error("local runtime is not bundled: {0}")]
    RuntimeUnavailable(String),
    #[error("required local model is not ready: {0}")]
    ModelUnavailable(String),
    #[error("cannot launch {service}: {source}")]
    Launch { service: String, source: io::Error },
    #[error("{service} exited during startup ({status}); see {log_path}: {diagnostics}")]
    StartupExit {
        service: String,
        status: String,
        log_path: PathBuf,
        diagnostics: String,
    },
    #[error("{service} did not become healthy within {seconds}s; see {log_path}")]
    StartupTimeout {
        service: String,
        seconds: u64,
        log_path: PathBuf,
    },
    #[error("local runtime filesystem operation failed: {0}")]
    Io(#[from] io::Error),
}

impl LocalRuntimeError {
    /// The service process itself failed (as opposed to a missing model or
    /// runtime), so another runtime may still succeed.
    fn is_process_failure(&self) -> bool {
        matches!(
            self,
            Self::Launch { .. } | Self::StartupExit { .. } | Self::StartupTimeout { .. }
        )
    }

    /// One line for the UI: the log tail stays in the log file.
    fn summary(&self) -> String {
        match self {
            Self::StartupExit {
                service,
                status,
                log_path,
                ..
            } => format!(
                "{service} exited during startup ({status}); see {}",
                log_path.display()
            ),
            other => other.to_string(),
        }
    }
}

#[derive(Debug, Default)]
struct GpuSelection {
    runtime: Option<GpuRuntimeLayout>,
    /// Services that failed on the GPU runtime and use the CPU runtime until
    /// the selection changes, with why.
    fallbacks: std::collections::BTreeMap<String, String>,
}

pub struct LocalRuntimeManager {
    layout: LocalRuntimeLayout,
    gpu: Mutex<GpuSelection>,
    services: AsyncMutex<HashMap<String, ProcessTree>>,
    port_reservations: Mutex<HashMap<String, std::net::TcpListener>>,
}

impl LocalRuntimeManager {
    pub fn new(layout: LocalRuntimeLayout) -> Self {
        Self {
            layout,
            gpu: Mutex::new(GpuSelection::default()),
            services: AsyncMutex::new(HashMap::new()),
            port_reservations: Mutex::new(HashMap::new()),
        }
    }

    pub fn new_with_reserved_ports(
        layout: LocalRuntimeLayout,
        qwen: std::net::TcpListener,
        hymt: std::net::TcpListener,
    ) -> Self {
        Self {
            layout,
            gpu: Mutex::new(GpuSelection::default()),
            services: AsyncMutex::new(HashMap::new()),
            port_reservations: Mutex::new(HashMap::from([
                ("qwen_asr".into(), qwen),
                ("hymt".into(), hymt),
            ])),
        }
    }

    pub fn capability_environment(&self) -> HashMap<String, String> {
        let mut values = HashMap::from([(
            "ECHOLINGO_QWEN_ASR_COMMAND".into(),
            self.layout
                .qwen_command
                .executable
                .to_string_lossy()
                .into_owned(),
        )]);
        values.insert("WLK_API_TOKEN".into(), self.layout.local_api_key.clone());
        values.insert(
            "ECHOLINGO_LOCAL_MT_API_KEY".into(),
            self.layout.local_api_key.clone(),
        );
        values.insert(
            "ECHOLINGO_LOCAL_QWEN_URL".into(),
            format!("ws://127.0.0.1:{}/asr", self.layout.qwen_port),
        );
        values.insert(
            "ECHOLINGO_LOCAL_HYMT_URL".into(),
            format!("http://127.0.0.1:{}/v1", self.layout.hymt_port),
        );
        values.insert(
            "ECHOLINGO_LOCAL_QWEN_PORT".into(),
            self.layout.qwen_port.to_string(),
        );
        values.insert(
            "ECHOLINGO_LOCAL_HYMT_PORT".into(),
            self.layout.hymt_port.to_string(),
        );
        if let Some(path) = &self.layout.llama_server {
            values.insert(
                "ECHOLINGO_LLAMA_SERVER".into(),
                path.to_string_lossy().into_owned(),
            );
        }
        values
    }

    /// Use `runtime` (the GPU pack) for the model services from now on, or
    /// the bundled CPU runtime for `None`. A change stops every running
    /// service so the next start uses the new runtime, and gives services
    /// that failed on the GPU earlier another chance; returns whether
    /// anything changed. Callers only switch while no session uses the
    /// services.
    pub async fn select_gpu_runtime(&self, runtime: Option<GpuRuntimeLayout>) -> bool {
        let mut services = self.services.lock().await;
        {
            let Ok(mut selection) = self.gpu.lock() else {
                return false;
            };
            if selection.runtime == runtime {
                return false;
            }
            selection.runtime = runtime;
            selection.fallbacks.clear();
        }
        for (_, mut process) in services.drain() {
            let _ = process.terminate(SERVICE_STOP_GRACE).await;
        }
        true
    }

    /// Forget earlier GPU failures so the next start of each service tries
    /// the GPU runtime again (after the user reinstalled the pack).
    pub fn clear_gpu_fallback(&self) {
        if let Ok(mut selection) = self.gpu.lock() {
            selection.fallbacks.clear();
        }
    }

    /// Whether `service` starts on the GPU runtime now.
    pub fn uses_gpu(&self, service: &str) -> bool {
        self.gpu_executable(service).is_some()
    }

    /// Whether any service starts on the GPU runtime now.
    pub fn gpu_active(&self) -> bool {
        LOCAL_SERVICES.iter().any(|service| self.uses_gpu(service))
    }

    /// Why services fell back to the CPU runtime, if any did.
    pub fn gpu_fallback_reason(&self) -> Option<String> {
        let selection = self.gpu.lock().ok()?;
        (!selection.fallbacks.is_empty())
            .then(|| selection.fallbacks.values().cloned().collect::<Vec<_>>().join("; "))
    }

    /// Why `service` fell back to the CPU runtime, if it did.
    pub fn gpu_fallback_for(&self, service: &str) -> Option<String> {
        self.gpu.lock().ok()?.fallbacks.get(service).cloned()
    }

    fn gpu_executable(&self, service: &str) -> Option<PathBuf> {
        let selection = self.gpu.lock().ok()?;
        if selection.fallbacks.contains_key(service) {
            return None;
        }
        selection
            .runtime
            .as_ref()?
            .executable_for(service)
            .map(Path::to_path_buf)
    }

    fn record_gpu_fallback(&self, service: &str, reason: String) {
        if let Ok(mut selection) = self.gpu.lock() {
            selection.fallbacks.entry(service.into()).or_insert(reason);
        }
    }

    /// The command that starts `service` on the runtime selected now.
    pub fn command_for(&self, service: &str) -> Result<RuntimeCommand, LocalRuntimeError> {
        self.command_on(service, self.gpu_executable(service).as_deref())
    }

    /// The command for `service`; `gpu` is the pack executable to run it
    /// with, `None` for the CPU runtime.
    fn command_on(
        &self,
        service: &str,
        gpu: Option<&Path>,
    ) -> Result<RuntimeCommand, LocalRuntimeError> {
        match service {
            "qwen_asr" => {
                let model = self.layout.model_root.join("qwen3-asr-0.6b");
                if !model.join("model.safetensors").is_file() {
                    return Err(LocalRuntimeError::ModelUnavailable("qwen3-asr-0.6b".into()));
                }
                let (mut command, device) = match gpu {
                    Some(sidecar) => (
                        RuntimeCommand {
                            executable: sidecar.to_path_buf(),
                            args: vec!["qwen-asr-server".into()],
                            environment: HashMap::new(),
                        },
                        "cuda".to_string(),
                    ),
                    None => (
                        self.layout.qwen_command.clone(),
                        self.layout.qwen_device.clone(),
                    ),
                };
                command
                    .environment
                    .insert("WLK_API_TOKEN".into(), self.layout.local_api_key.clone());
                command.args.extend([
                    "--host".into(),
                    "127.0.0.1".into(),
                    "--port".into(),
                    self.layout.qwen_port.to_string(),
                    "--backend".into(),
                    "qwen3-streaming".into(),
                    "--model_dir".into(),
                    model.to_string_lossy().into_owned(),
                    "--lan".into(),
                    "en".into(),
                    "--pcm-input".into(),
                    "--no-vac".into(),
                    "--no-vad".into(),
                    "--warmup-file".into(),
                    "".into(),
                    "--qwen3-streaming-device".into(),
                    device,
                    "--qwen3-streaming-chunk-sec".into(),
                    self.layout.qwen_streaming.chunk_seconds.to_string(),
                    "--qwen3-streaming-stable-iterations".into(),
                    self.layout.qwen_streaming.stable_iterations.to_string(),
                    "--qwen3-streaming-hold-back-words".into(),
                    self.layout.qwen_streaming.hold_back_words.to_string(),
                    "--qwen3-streaming-left-context-sec".into(),
                    self.layout.qwen_streaming.left_context_seconds.to_string(),
                    "--qwen3-streaming-right-context-ms".into(),
                    self.layout.qwen_streaming.right_context_ms.to_string(),
                    "--qwen3-streaming-segment-max-steps".into(),
                    self.layout.qwen_streaming.segment_max_steps.to_string(),
                    "--log-level".into(),
                    "INFO".into(),
                ]);
                Ok(command)
            }
            "hymt" => {
                let executable = gpu
                    .map(Path::to_path_buf)
                    .or_else(|| self.layout.llama_server.clone())
                    .ok_or_else(|| LocalRuntimeError::RuntimeUnavailable("llama-server".into()))?;
                let model = self
                    .layout
                    .model_root
                    .join("hymt2-1.8b")
                    .join("Hy-MT2-1.8B-Q4_K_M.gguf");
                if !model.is_file() {
                    return Err(LocalRuntimeError::ModelUnavailable("hymt2-1.8b".into()));
                }
                let mut args = vec![
                    "--model".into(),
                    model.to_string_lossy().into_owned(),
                    "--host".into(),
                    "127.0.0.1".into(),
                    "--port".into(),
                    self.layout.hymt_port.to_string(),
                    "--alias".into(),
                    "tencent/Hy-MT2-1.8B".into(),
                    "--ctx-size".into(),
                    "4096".into(),
                    "--parallel".into(),
                    "1".into(),
                    "--n-gpu-layers".into(),
                    "99".into(),
                    // The GGUF carries the Hy-MT chat template; apply it explicitly
                    // and cap generation server-side so a runaway request can never
                    // fill the context again.
                    "--jinja".into(),
                    "--n-predict".into(),
                    "400".into(),
                ];
                let executable = if let Some(wrapper) = &self.layout.worker_wrapper {
                    args.insert(0, executable.to_string_lossy().into_owned());
                    args.insert(0, "watch-process".into());
                    wrapper.clone()
                } else {
                    executable
                };
                Ok(RuntimeCommand {
                    executable,
                    args,
                    environment: HashMap::from([(
                        "LLAMA_API_KEY".into(),
                        self.layout.local_api_key.clone(),
                    )]),
                })
            }
            other => Err(LocalRuntimeError::UnknownService(other.into())),
        }
    }

    pub async fn ensure_services(&self, services: &[String]) -> Result<(), LocalRuntimeError> {
        for service in services {
            self.ensure_service(service).await?;
        }
        Ok(())
    }

    pub async fn ensure_service(&self, service: &str) -> Result<(), LocalRuntimeError> {
        let (port, health_path) = self.service_endpoint(service)?;
        self.port_reservations
            .lock()
            .map_err(|_| io::Error::other("local runtime port reservation lock poisoned"))?
            .remove(service);
        if service_healthy(port, health_path).await {
            return Ok(());
        }

        let mut services = self.services.lock().await;
        if service_healthy(port, health_path).await {
            return Ok(());
        }
        if let Some(mut previous) = services.remove(service) {
            let _ = previous.terminate(SERVICE_STOP_GRACE).await;
        }
        let gpu = self.gpu_executable(service);
        let process = match self.launch(service, port, health_path, gpu.as_deref()).await {
            Ok(process) => process,
            Err(error) if gpu.is_some() && error.is_process_failure() => {
                // The GPU runtime is optional: keep the session working on
                // the CPU runtime. Only this service falls back; the other
                // keeps its GPU runtime.
                self.record_gpu_fallback(
                    service,
                    format!(
                        "{service} could not start with GPU acceleration: {}",
                        error.summary()
                    ),
                );
                self.launch(service, port, health_path, None).await?
            }
            Err(error) => return Err(error),
        };
        services.insert(service.into(), process);
        Ok(())
    }

    /// Start `service` on the CPU runtime or `gpu` and wait until it is
    /// healthy. GPU starts log to `<service>.gpu.log` so a CPU fallback does
    /// not overwrite the diagnostics of the failure.
    async fn launch(
        &self,
        service: &str,
        port: u16,
        health_path: &str,
        gpu: Option<&Path>,
    ) -> Result<ProcessTree, LocalRuntimeError> {
        let spec = self.command_on(service, gpu)?;
        std::fs::create_dir_all(&self.layout.log_root)?;
        let log_name = if gpu.is_some() {
            format!("{service}.gpu.log")
        } else {
            format!("{service}.log")
        };
        let log_path = self.layout.log_root.join(log_name);
        let log = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)?;
        let stderr = log.try_clone()?;
        let mut command = Command::new(&spec.executable);
        command
            .args(&spec.args)
            .envs(&spec.environment)
            .env("ECHOLINGO_PARENT_PID", std::process::id().to_string())
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr));
        let mut process =
            ProcessTree::spawn(&mut command).map_err(|source| LocalRuntimeError::Launch {
                service: service.into(),
                source,
            })?;
        let deadline = tokio::time::Instant::now() + self.layout.startup_timeout;
        loop {
            if service_healthy(port, health_path).await {
                return Ok(process);
            }
            if let Some(status) = process.try_wait().map_err(LocalRuntimeError::Io)? {
                return Err(LocalRuntimeError::StartupExit {
                    service: service.into(),
                    status: status.to_string(),
                    diagnostics: log_tail(&log_path, 4096),
                    log_path,
                });
            }
            if tokio::time::Instant::now() >= deadline {
                let _ = process.terminate(SERVICE_STOP_GRACE).await;
                return Err(LocalRuntimeError::StartupTimeout {
                    service: service.into(),
                    seconds: self.layout.startup_timeout.as_secs(),
                    log_path,
                });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Stop one service (and everything it launched) if it is running, for
    /// example before its model files are replaced or deleted.
    pub async fn stop_service(&self, service: &str) {
        let process = self.services.lock().await.remove(service);
        if let Some(mut process) = process {
            let _ = process.terminate(SERVICE_STOP_GRACE).await;
        }
    }

    pub async fn shutdown(&self) {
        let mut services = self.services.lock().await;
        for (_, mut process) in services.drain() {
            let _ = process.terminate(SERVICE_STOP_GRACE).await;
        }
    }

    fn service_endpoint(&self, service: &str) -> Result<(u16, &'static str), LocalRuntimeError> {
        match service {
            "qwen_asr" => Ok((self.layout.qwen_port, "/health")),
            "hymt" => Ok((self.layout.hymt_port, "/health")),
            other => Err(LocalRuntimeError::UnknownService(other.into())),
        }
    }
}

async fn service_healthy(port: u16, path: &str) -> bool {
    let connection = tokio::time::timeout(
        Duration::from_millis(500),
        TcpStream::connect(("127.0.0.1", port)),
    )
    .await;
    let Ok(Ok(mut stream)) = connection else {
        return false;
    };
    let request = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }
    let mut response = vec![0_u8; 4096];
    let read = tokio::time::timeout(Duration::from_millis(500), stream.read(&mut response)).await;
    let Ok(Ok(read)) = read else {
        return false;
    };
    response[..read].starts_with(b"HTTP/1.1 200") || response[..read].starts_with(b"HTTP/1.0 200")
}

fn log_tail(path: &Path, limit: usize) -> String {
    let Ok(bytes) = std::fs::read(path) else {
        return "no diagnostics were produced".into();
    };
    let start = bytes.len().saturating_sub(limit);
    let output = String::from_utf8_lossy(&bytes[start..]).trim().to_string();
    if output.is_empty() {
        "no diagnostics were produced".into()
    } else {
        output
    }
}

type ProgressCallback = Arc<dyn Fn(ModelProgress) + Send + Sync>;

/// One model download attempt: fill `staging` with the model's files and
/// report through `progress`.
struct DownloadRequest {
    spec: ModelSpec,
    staging: PathBuf,
    cache: PathBuf,
    progress: Arc<ModelProgressHandler>,
}

type DownloadFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;
type Downloader = Arc<dyn Fn(DownloadRequest) -> DownloadFuture + Send + Sync>;

/// The Hugging Face download of a pinned model revision.
fn hugging_face_download(request: DownloadRequest) -> DownloadFuture {
    Box::pin(async move {
        let DownloadRequest {
            spec,
            staging,
            cache,
            progress,
        } = request;
        let client = HFClient::builder()
            .cache_dir(&cache)
            .user_agent(format!("echolingo/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| error.to_string())?;
        let (owner, name) = spec
            .repository
            .split_once('/')
            .ok_or_else(|| "invalid repository name".to_string())?;
        let repository = client.model(owner, name);
        let result = match &spec.download {
            DownloadKind::Snapshot { allow_patterns } => repository
                .snapshot_download()
                .revision(spec.revision.clone())
                .allow_patterns(allow_patterns.clone())
                .local_dir(staging)
                .max_workers(4)
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
            DownloadKind::File { filename } => repository
                .download_file()
                .filename(filename.clone())
                .revision(spec.revision.clone())
                .local_dir(staging)
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
        };
        result.map_err(|error| error.to_string())
    })
}

/// `ECHOLINGO_MODEL_STALL_SECONDS` (clamped to its range) or the default.
fn stall_timeout_from(raw: Option<&str>) -> Duration {
    let seconds = raw
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .map(|seconds| seconds.clamp(MIN_STALL_SECONDS, MAX_STALL_SECONDS))
        .unwrap_or(DEFAULT_STALL_SECONDS);
    Duration::from_secs(seconds)
}

pub struct ModelManager {
    root: PathBuf,
    cache: PathBuf,
    active: Mutex<HashSet<String>>,
    downloader: Downloader,
    stall_timeout: Duration,
}

impl ModelManager {
    pub fn new(root: PathBuf) -> Self {
        let stall_timeout =
            stall_timeout_from(std::env::var(STALL_SECONDS_ENVIRONMENT).ok().as_deref());
        Self::with_downloader(root, Arc::new(hugging_face_download), stall_timeout)
    }

    fn with_downloader(root: PathBuf, downloader: Downloader, stall_timeout: Duration) -> Self {
        let cache = root.join(".downloads");
        Self {
            root,
            cache,
            active: Mutex::new(HashSet::new()),
            downloader,
            stall_timeout,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> Vec<ModelStatus> {
        catalog()
            .into_iter()
            .map(|spec| self.status_for(&spec))
            .collect()
    }

    pub fn status(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        Ok(self.status_for(&spec))
    }

    fn status_for(&self, spec: &ModelSpec) -> ModelStatus {
        let path = self.root.join(&spec.id);
        let active = self
            .active
            .lock()
            .is_ok_and(|active| active.contains(&spec.id));
        let state = if active {
            ModelInstallState::Installing
        } else if !path.exists() {
            ModelInstallState::NotDownloaded
        } else if installed_manifest_matches(&path, spec) && required_file_matches_size(&path, spec)
        {
            ModelInstallState::Ready
        } else {
            ModelInstallState::Corrupt
        };
        ModelStatus {
            id: spec.id.clone(),
            display_name: spec.display_name.clone(),
            role: spec.role,
            size_bytes: spec.expected_bytes,
            state,
            path,
            revision: spec.revision.clone(),
        }
    }

    pub async fn install(
        &self,
        model_id: &str,
        callback: ProgressCallback,
    ) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        {
            let mut active = self
                .active
                .lock()
                .map_err(|_| io::Error::other("model manager lock poisoned"))?;
            if !active.insert(spec.id.clone()) {
                return Err(ModelManagerError::AlreadyInstalling(spec.id));
            }
        }
        let result = self.install_inner(&spec, callback).await;
        if let Ok(mut active) = self.active.lock() {
            active.remove(&spec.id);
        }
        result.map(|()| self.status_for(&spec))
    }

    async fn install_inner(
        &self,
        spec: &ModelSpec,
        callback: ProgressCallback,
    ) -> Result<(), ModelManagerError> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::create_dir_all(&self.cache)?;
        let required = spec.expected_bytes.saturating_add(spec.expected_bytes / 10);
        let available = fs2::available_space(&self.root)?;
        if available < required {
            return Err(ModelManagerError::DiskSpace {
                required_bytes: required,
                available_bytes: available,
            });
        }

        let staging = self.download(spec, &callback).await?;

        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: spec.expected_bytes,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "verifying".into(),
            message: None,
        });
        verify_required_file(&staging, spec)?;
        write_manifest(&staging, spec)?;
        activate_directory(&self.root, &staging, &self.root.join(&spec.id))?;
        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: spec.expected_bytes,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "ready".into(),
            message: None,
        });
        Ok(())
    }

    /// Download `spec` into a fresh staging directory and return it. An
    /// attempt that stalls is abandoned and the download starts over, in a
    /// new directory, up to [`DOWNLOAD_ATTEMPTS`] times in all.
    async fn download(
        &self,
        spec: &ModelSpec,
        callback: &ProgressCallback,
    ) -> Result<PathBuf, ModelManagerError> {
        // Retry directories an earlier install could not remove.
        for attempt in 2..=DOWNLOAD_ATTEMPTS {
            self.discard_staging(&self.staging_directory(spec, attempt));
        }
        for attempt in 1..=DOWNLOAD_ATTEMPTS {
            let staging = self.staging_directory(spec, attempt);
            remove_scoped_directory(&self.root, &staging)?;
            std::fs::create_dir_all(&staging)?;
            let note = (attempt > 1)
                .then(|| format!("Download stalled; retrying ({attempt}/{DOWNLOAD_ATTEMPTS})"));
            callback(ModelProgress {
                model_id: spec.id.clone(),
                bytes_completed: 0,
                total_bytes: spec.expected_bytes,
                bytes_per_second: None,
                phase: if attempt == 1 { "starting" } else { "retrying" }.into(),
                message: note.clone(),
            });
            let activity = Arc::new(DownloadActivity::new(self.stall_timeout));
            let request = DownloadRequest {
                spec: spec.clone(),
                staging: staging.clone(),
                cache: self.cache.clone(),
                progress: Arc::new(ModelProgressHandler {
                    model_id: spec.id.clone(),
                    callback: callback.clone(),
                    note,
                    activity: activity.clone(),
                    files: Mutex::new(HashMap::new()),
                }),
            };
            match run_download_attempt(&self.downloader, request, &activity).await? {
                AttemptOutcome::Finished(Ok(())) => return Ok(staging),
                AttemptOutcome::Finished(Err(error)) => {
                    self.discard_staging(&staging);
                    return Err(ModelManagerError::Download(error));
                }
                AttemptOutcome::Stalled => {
                    eprintln!(
                        "model download of {} stalled (attempt {attempt}/{DOWNLOAD_ATTEMPTS})",
                        spec.id
                    );
                    self.discard_staging(&staging);
                }
            }
        }
        Err(ModelManagerError::Download(format!(
            "the download stalled {DOWNLOAD_ATTEMPTS} times without receiving data; \
             check the network connection and try again"
        )))
    }

    /// The staging directory of download attempt `attempt`: each attempt
    /// has its own, so a write that was still in flight when a stalled
    /// attempt was stopped can never land in the next one.
    fn staging_directory(&self, spec: &ModelSpec, attempt: u32) -> PathBuf {
        if attempt <= 1 {
            self.root.join(format!(".{}.installing", spec.id))
        } else {
            self.root.join(format!(".{}.installing-{attempt}", spec.id))
        }
    }

    /// Best-effort removal of an abandoned staging directory. It can fail on
    /// Windows while a file in it is still open; the next install retries.
    fn discard_staging(&self, staging: &Path) {
        if let Err(error) = remove_scoped_directory(&self.root, staging) {
            eprintln!(
                "could not remove {} ({error}); the next install removes it",
                staging.display()
            );
        }
    }

    pub fn verify(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        verify_required_file(&self.root.join(&spec.id), &spec)?;
        Ok(self.status_for(&spec))
    }

    pub fn delete(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        if self
            .active
            .lock()
            .is_ok_and(|active| active.contains(&spec.id))
        {
            return Err(ModelManagerError::AlreadyInstalling(spec.id));
        }
        remove_scoped_directory(&self.root, &self.root.join(&spec.id))?;
        Ok(self.status_for(&spec))
    }
}

enum AttemptOutcome {
    Finished(Result<(), String>),
    Stalled,
}

/// Run one download attempt on a runtime of its own until it finishes or
/// stalls.
///
/// hf-hub runs a Xet transfer (and its progress poller) as detached tasks on
/// the runtime that polls the download, so dropping the download future alone
/// would leave a stalled transfer running, and writing into the staging
/// directory, for the life of the app. Shutting the attempt's runtime down
/// drops every task it spawned; only a blocking file write that is already
/// running finishes, which is why each attempt also has its own staging
/// directory.
async fn run_download_attempt(
    downloader: &Downloader,
    request: DownloadRequest,
    activity: &DownloadActivity,
) -> io::Result<AttemptOutcome> {
    let runtime = AttemptRuntime::new()?;
    let task = runtime.spawn(downloader(request));
    let outcome = tokio::select! {
        joined = task => AttemptOutcome::Finished(
            joined.unwrap_or_else(|error| Err(format!("the download task failed: {error}"))),
        ),
        () = activity.stalled() => AttemptOutcome::Stalled,
    };
    // Late events of this attempt must not reach the UI after the next one
    // has started.
    activity.retire();
    drop(runtime);
    Ok(outcome)
}

/// The multi-thread runtime of one download attempt (Xet requires one with
/// time and I/O drivers), shut down without blocking when dropped, which is
/// safe inside another runtime.
struct AttemptRuntime {
    handle: tokio::runtime::Handle,
    runtime: Option<tokio::runtime::Runtime>,
}

impl AttemptRuntime {
    fn new() -> io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("model-download")
            .build()?;
        Ok(Self {
            handle: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }

    fn spawn(&self, download: DownloadFuture) -> tokio::task::JoinHandle<Result<(), String>> {
        self.handle.spawn(download)
    }
}

impl Drop for AttemptRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// When a download attempt last received data and how long it may stay
/// silent. Only events that change a byte count (or start or finish the
/// transfer) count: hf-hub's Xet poller repeats the same totals ten times a
/// second while a transfer is stuck.
#[derive(Debug)]
struct StallClock {
    stall_timeout: Duration,
    last_advance: Instant,
    longest_silence: Duration,
    file_bytes: u64,
    /// Xet's aggregate byte count, once it has reported one.
    xet_bytes: Option<u64>,
}

impl StallClock {
    fn new(stall_timeout: Duration, now: Instant) -> Self {
        Self {
            stall_timeout,
            last_advance: now,
            longest_silence: Duration::ZERO,
            file_bytes: 0,
            xet_bytes: None,
        }
    }

    fn advance(&mut self, now: Instant) {
        self.longest_silence = self
            .longest_silence
            .max(now.saturating_duration_since(self.last_advance));
        self.last_advance = self.last_advance.max(now);
    }

    /// The sum of the per-file byte counts (plain HTTP transfers and Xet).
    fn file_bytes(&mut self, bytes: u64, now: Instant) {
        if bytes != self.file_bytes {
            self.file_bytes = bytes;
            self.advance(now);
        }
    }

    /// Xet's aggregate byte count.
    fn xet_bytes(&mut self, bytes: u64, now: Instant) {
        if self.xet_bytes != Some(bytes) {
            let first_report = self.xet_bytes.is_none();
            self.xet_bytes = Some(bytes);
            // A first report of zero bytes only says that Xet has started.
            if !(first_report && bytes == 0) {
                self.advance(now);
            }
        }
    }

    /// The stall timeout, stretched for Xet's coarse steps (see
    /// [`FIRST_DATA_STALL_FACTOR`] and [`STALL_GAP_FACTOR`]).
    fn allowed_silence(&self) -> Duration {
        let mut allowed = self
            .stall_timeout
            .max(self.longest_silence * STALL_GAP_FACTOR);
        if self.xet_bytes == Some(0) {
            allowed = allowed.max(self.stall_timeout * FIRST_DATA_STALL_FACTOR);
        }
        allowed
    }

    /// Zero once the attempt has stalled.
    fn time_left(&self, now: Instant) -> Duration {
        (self.last_advance + self.allowed_silence()).saturating_duration_since(now)
    }
}

/// A download attempt's [`StallClock`] shared with its progress handler.
struct DownloadActivity {
    clock: Mutex<StallClock>,
    /// Held while an event is passed on, so that nothing of the attempt is
    /// reported once [`Self::retire`] has returned.
    retired: Mutex<bool>,
}

impl DownloadActivity {
    fn new(stall_timeout: Duration) -> Self {
        Self {
            clock: Mutex::new(StallClock::new(stall_timeout, Instant::now())),
            retired: Mutex::new(false),
        }
    }

    fn record(&self, update: impl FnOnce(&mut StallClock, Instant)) {
        if let Ok(mut clock) = self.clock.lock() {
            update(&mut clock, Instant::now());
        }
    }

    /// Resolves once the attempt has stalled.
    async fn stalled(&self) {
        loop {
            let left = match self.clock.lock() {
                Ok(clock) => clock.time_left(Instant::now()),
                Err(_) => STALL_CHECK_INTERVAL,
            };
            if left.is_zero() {
                return;
            }
            // The allowance can shrink (Xet's first bytes end the longer
            // wait for them), so look again at least every interval.
            tokio::time::sleep(left.min(STALL_CHECK_INTERVAL)).await;
        }
    }

    /// Run `report` unless the attempt is over.
    fn report(&self, report: impl FnOnce()) {
        if let Ok(retired) = self.retired.lock() {
            if !*retired {
                report();
            }
        }
    }

    fn retire(&self) {
        if let Ok(mut retired) = self.retired.lock() {
            *retired = true;
        }
    }
}

struct ModelProgressHandler {
    model_id: String,
    callback: ProgressCallback,
    /// Shown with every event of a retried attempt.
    note: Option<String>,
    activity: Arc<DownloadActivity>,
    files: Mutex<HashMap<String, (u64, u64)>>,
}

impl ProgressHandler for ModelProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let progress = match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => {
                self.activity.record(StallClock::advance);
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed: 0,
                    total_bytes: *total_bytes,
                    bytes_per_second: None,
                    phase: "downloading".into(),
                    message: None,
                }
            }
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                let mut cumulative = match self.files.lock() {
                    Ok(value) => value,
                    Err(_) => return,
                };
                for file in files {
                    cumulative.insert(
                        file.filename.clone(),
                        (file.bytes_completed, file.total_bytes),
                    );
                }
                let bytes_completed = cumulative.values().map(|value| value.0).sum();
                self.activity
                    .record(|clock, now| clock.file_bytes(bytes_completed, now));
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed,
                    total_bytes: cumulative.values().map(|value| value.1).sum(),
                    bytes_per_second: None,
                    phase: "downloading".into(),
                    message: None,
                }
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => {
                self.activity
                    .record(|clock, now| clock.xet_bytes(*bytes_completed, now));
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed: *bytes_completed,
                    total_bytes: *total_bytes,
                    bytes_per_second: *bytes_per_sec,
                    phase: "downloading".into(),
                    message: None,
                }
            }
            ProgressEvent::Download(DownloadEvent::Complete) => {
                self.activity.record(StallClock::advance);
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed: 0,
                    total_bytes: 0,
                    bytes_per_second: None,
                    phase: "downloaded".into(),
                    message: None,
                }
            }
            _ => return,
        };
        self.activity.report(|| {
            (self.callback)(ModelProgress {
                message: self.note.clone(),
                ..progress
            })
        });
    }
}

fn model_spec(model_id: &str) -> Result<ModelSpec, ModelManagerError> {
    catalog()
        .into_iter()
        .find(|spec| spec.id == model_id)
        .ok_or_else(|| ModelManagerError::UnknownModel(model_id.into()))
}

pub fn catalog() -> Vec<ModelSpec> {
    let qwen_files = vec![
        "*.json".into(),
        "*.txt".into(),
        "*.md".into(),
        "model.safetensors".into(),
    ];
    vec![
        ModelSpec {
            id: "qwen3-asr-0.6b".into(),
            display_name: "Qwen3-ASR 0.6B".into(),
            role: ModelRole::Asr,
            repository: "Qwen/Qwen3-ASR-0.6B".into(),
            revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0".into(),
            expected_bytes: 1_876_091_704,
            required_file: "model.safetensors".into(),
            required_file_sha256:
                "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea".into(),
            download: DownloadKind::Snapshot {
                allow_patterns: qwen_files.clone(),
            },
        },
        ModelSpec {
            id: "qwen3-forced-aligner-0.6b".into(),
            display_name: "Qwen3 ForcedAligner 0.6B".into(),
            role: ModelRole::Alignment,
            repository: "Qwen/Qwen3-ForcedAligner-0.6B".into(),
            revision: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7".into(),
            expected_bytes: 1_835_544_544,
            required_file: "model.safetensors".into(),
            required_file_sha256:
                "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7".into(),
            download: DownloadKind::Snapshot {
                allow_patterns: qwen_files,
            },
        },
        ModelSpec {
            id: "hymt2-1.8b".into(),
            display_name: "Hy-MT2 1.8B Q4_K_M".into(),
            role: ModelRole::Translation,
            repository: "tencent/Hy-MT2-1.8B-GGUF".into(),
            revision: "1cd5208700acedef4ef93019b6cfc148b8522d45".into(),
            expected_bytes: 1_133_080_448,
            required_file: "Hy-MT2-1.8B-Q4_K_M.gguf".into(),
            required_file_sha256:
                "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699".into(),
            download: DownloadKind::File {
                filename: "Hy-MT2-1.8B-Q4_K_M.gguf".into(),
            },
        },
    ]
}

fn manifest_path(directory: &Path) -> PathBuf {
    directory.join(".echolingo-model.json")
}

fn installed_manifest_matches(directory: &Path, spec: &ModelSpec) -> bool {
    std::fs::read(manifest_path(directory))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<InstallManifest>(&bytes).ok())
        .is_some_and(|manifest| {
            manifest.schema_version == MODEL_CATALOG_VERSION
                && manifest.model_id == spec.id
                && manifest.repository == spec.repository
                && manifest.revision == spec.revision
                && manifest.required_file == spec.required_file
                && manifest.required_file_sha256 == spec.required_file_sha256
        })
}

fn required_file_matches_size(directory: &Path, spec: &ModelSpec) -> bool {
    std::fs::metadata(directory.join(&spec.required_file))
        .is_ok_and(|metadata| metadata.len() == spec.expected_bytes)
}

fn verify_required_file(directory: &Path, spec: &ModelSpec) -> Result<(), ModelManagerError> {
    let path = directory.join(&spec.required_file);
    if !required_file_matches_size(directory, spec) {
        return Err(ModelManagerError::Integrity(spec.id.clone()));
    }
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    if format!("{:x}", digest.finalize()) != spec.required_file_sha256 {
        return Err(ModelManagerError::Integrity(spec.id.clone()));
    }
    Ok(())
}

fn write_manifest(directory: &Path, spec: &ModelSpec) -> Result<(), ModelManagerError> {
    let manifest = InstallManifest {
        schema_version: MODEL_CATALOG_VERSION,
        model_id: spec.id.clone(),
        repository: spec.repository.clone(),
        revision: spec.revision.clone(),
        required_file: spec.required_file.clone(),
        required_file_sha256: spec.required_file_sha256.clone(),
    };
    let temporary = directory.join(".echolingo-model.json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::rename(temporary, manifest_path(directory))?;
    Ok(())
}

fn activate_directory(root: &Path, staging: &Path, destination: &Path) -> io::Result<()> {
    activate_directory_retrying(root, staging, destination, 1, Duration::ZERO)
}

/// Replace `destination` with `staging`: the current copy moves aside to
/// `.<name>.previous`, `staging` takes its place (the old copy is put back
/// if that fails) and the old copy is deleted last. Each rename is tried up
/// to `attempts` times, `delay` apart, and not again once its source is
/// gone. Deleting the old copy is best effort; the next activation removes
/// a leftover.
fn activate_directory_retrying(
    root: &Path,
    staging: &Path,
    destination: &Path,
    attempts: u32,
    delay: Duration,
) -> io::Result<()> {
    let previous = root.join(format!(
        ".{}.previous",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    ));
    if !staging.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} does not exist", staging.display()),
        ));
    }
    remove_scoped_directory(root, &previous)?;
    let moved_aside = destination.exists();
    if moved_aside {
        retry_rename(destination, &previous, attempts, delay)?;
    }
    if let Err(error) = retry_rename(staging, destination, attempts, delay) {
        if moved_aside && !destination.exists() {
            let _ = std::fs::rename(&previous, destination);
        }
        return Err(error);
    }
    if let Err(error) = remove_scoped_directory(root, &previous) {
        eprintln!(
            "could not remove {} ({error}); the next update removes it",
            previous.display()
        );
    }
    Ok(())
}

fn retry_rename(from: &Path, to: &Path, attempts: u32, delay: Duration) -> io::Result<()> {
    let mut attempt = 1;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(_) if attempt < attempts && from.exists() && !to.exists() => {
                attempt += 1;
                std::thread::sleep(delay);
            }
            Err(error) => return Err(error),
        }
    }
}

fn remove_scoped_directory(root: &Path, target: &Path) -> io::Result<()> {
    if !target.exists() {
        return Ok(());
    }
    let root = root.canonicalize()?;
    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("model path has no parent"))?
        .canonicalize()?;
    if parent != root {
        return Err(io::Error::other(
            "refusing to remove path outside model root",
        ));
    }
    std::fs::remove_dir_all(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn runtime_layout(root: &Path, executable: PathBuf) -> LocalRuntimeLayout {
        LocalRuntimeLayout {
            qwen_command: RuntimeCommand {
                executable,
                args: vec!["qwen-asr-server".into()],
                environment: HashMap::new(),
            },
            llama_server: None,
            model_root: root.join("models"),
            log_root: root.join("logs"),
            qwen_device: "mps".into(),
            qwen_streaming: QwenStreamingProfile::default(),
            worker_wrapper: None,
            local_api_key: "test-local-token".into(),
            qwen_port: 38_123,
            hymt_port: 38_124,
            startup_timeout: Duration::from_secs(2),
        }
    }

    #[test]
    fn catalog_is_pinned_and_has_integrity_metadata() {
        for spec in catalog() {
            assert_eq!(spec.revision.len(), 40);
            assert_eq!(spec.required_file_sha256.len(), 64);
            assert!(spec.expected_bytes > 1_000_000_000);
        }
    }

    #[test]
    fn unknown_model_cannot_escape_managed_root() {
        let directory = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(directory.path().join("models"));
        assert!(matches!(
            manager.delete("../../outside"),
            Err(ModelManagerError::UnknownModel(_))
        ));
    }

    #[test]
    fn incomplete_model_is_reported_as_corrupt_and_can_be_deleted() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let manager = ModelManager::new(root.clone());
        std::fs::create_dir_all(root.join("qwen3-asr-0.6b")).unwrap();
        assert_eq!(
            manager.status("qwen3-asr-0.6b").unwrap().state,
            ModelInstallState::Corrupt
        );
        assert_eq!(
            manager.delete("qwen3-asr-0.6b").unwrap().state,
            ModelInstallState::NotDownloaded
        );
    }

    #[test]
    fn qwen_runtime_command_uses_managed_model_and_local_frontend_contract() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("models").join("qwen3-asr-0.6b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.safetensors"), b"model").unwrap();
        let manager = LocalRuntimeManager::new(runtime_layout(
            directory.path(),
            PathBuf::from("echolingo-sidecar"),
        ));

        let command = manager.command_for("qwen_asr").unwrap();
        assert_eq!(command.args[0], "qwen-asr-server");
        assert!(command
            .args
            .windows(2)
            .any(|pair| { pair[0] == "--model_dir" && Path::new(&pair[1]) == model }));
        assert!(command.args.contains(&"--pcm-input".into()));
        assert!(command.args.contains(&"--no-vad".into()));
        assert!(command
            .args
            .windows(2)
            .any(|pair| { pair[0] == "--qwen3-streaming-device" && pair[1] == "mps" }));
        assert!(command
            .args
            .windows(2)
            .any(|pair| pair[0] == "--lan" && pair[1] == "en"));
        assert!(command
            .args
            .windows(2)
            .any(|pair| pair[0] == "--qwen3-streaming-chunk-sec" && pair[1] == "1"));
        assert!(command
            .args
            .windows(2)
            .any(|pair| { pair[0] == "--qwen3-streaming-stable-iterations" && pair[1] == "1" }));
        assert!(command
            .args
            .windows(2)
            .any(|pair| { pair[0] == "--qwen3-streaming-hold-back-words" && pair[1] == "4" }));
        assert!(command
            .args
            .windows(2)
            .any(|pair| pair[0] == "--port" && pair[1] == "38123"));
        assert!(!command.args.iter().any(|argument| argument == "--language"));
    }

    #[test]
    fn packaged_hymt_runs_inside_the_owner_watchdog_wrapper() {
        let directory = tempfile::tempdir().unwrap();
        let wrapper = directory.path().join("echolingo-sidecar");
        let llama = directory.path().join("llama-server");
        std::fs::write(&wrapper, b"wrapper").unwrap();
        std::fs::write(&llama, b"llama").unwrap();
        let model = directory
            .path()
            .join("models/hymt2-1.8b/Hy-MT2-1.8B-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model").unwrap();
        let mut layout = runtime_layout(directory.path(), wrapper.clone());
        layout.llama_server = Some(llama.clone());
        layout.worker_wrapper = Some(wrapper.clone());

        let command = LocalRuntimeManager::new(layout)
            .command_for("hymt")
            .unwrap();

        assert_eq!(command.executable, wrapper);
        assert_eq!(command.args[0], "watch-process");
        assert_eq!(Path::new(&command.args[1]), llama);
        assert!(!command.args.iter().any(|argument| argument == "--api-key"));
        assert!(command.args.iter().any(|argument| argument == "--jinja"));
        assert!(command
            .args
            .windows(2)
            .any(|pair| pair[0] == "--n-predict" && pair[1] == "400"));
        assert_eq!(
            command.environment.get("LLAMA_API_KEY").map(String::as_str),
            Some("test-local-token")
        );
    }

    #[test]
    fn sidecar_environment_uses_isolated_runtime_endpoints() {
        let directory = tempfile::tempdir().unwrap();
        let manager = LocalRuntimeManager::new(runtime_layout(
            directory.path(),
            PathBuf::from("echolingo-sidecar"),
        ));

        let environment = manager.capability_environment();

        assert_eq!(
            environment
                .get("ECHOLINGO_LOCAL_QWEN_URL")
                .map(String::as_str),
            Some("ws://127.0.0.1:38123/asr")
        );
        assert_eq!(
            environment
                .get("ECHOLINGO_LOCAL_HYMT_URL")
                .map(String::as_str),
            Some("http://127.0.0.1:38124/v1")
        );
        assert_eq!(
            environment
                .get("ECHOLINGO_LOCAL_QWEN_PORT")
                .map(String::as_str),
            Some("38123")
        );
        assert_eq!(
            environment
                .get("ECHOLINGO_LOCAL_HYMT_PORT")
                .map(String::as_str),
            Some("38124")
        );
    }

    #[tokio::test]
    async fn selected_gpu_runtime_runs_the_pack_and_can_be_switched_off() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("models").join("qwen3-asr-0.6b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.safetensors"), b"model").unwrap();
        let gguf = directory.path().join("models").join("hymt2-1.8b");
        std::fs::create_dir_all(&gguf).unwrap();
        std::fs::write(gguf.join("Hy-MT2-1.8B-Q4_K_M.gguf"), b"model").unwrap();
        let cpu_sidecar = directory.path().join("echolingo-sidecar");
        let cpu_llama = directory.path().join("cpu-llama-server");
        let mut layout = runtime_layout(directory.path(), cpu_sidecar.clone());
        layout.worker_wrapper = Some(cpu_sidecar.clone());
        layout.llama_server = Some(cpu_llama.clone());
        let manager = LocalRuntimeManager::new(layout);
        let gpu_sidecar = directory.path().join("gpu").join("echolingo-sidecar");
        let gpu_llama = directory.path().join("gpu").join("llama-server");
        let pack = GpuRuntimeLayout {
            sidecar: Some(gpu_sidecar.clone()),
            llama_server: Some(gpu_llama.clone()),
        };

        assert!(!manager.gpu_active());
        assert!(manager.select_gpu_runtime(Some(pack.clone())).await);
        assert!(!manager.select_gpu_runtime(Some(pack.clone())).await);
        assert!(manager.gpu_active());
        assert!(manager.uses_gpu("qwen_asr") && manager.uses_gpu("hymt"));
        let qwen = manager.command_for("qwen_asr").unwrap();
        assert_eq!(qwen.executable, gpu_sidecar);
        assert_eq!(qwen.args[0], "qwen-asr-server");
        assert!(qwen
            .args
            .windows(2)
            .any(|pair| pair[0] == "--qwen3-streaming-device" && pair[1] == "cuda"));
        let hymt = manager.command_for("hymt").unwrap();
        // The pack's llama-server still runs under the owner watchdog.
        assert_eq!(hymt.executable, cpu_sidecar);
        assert_eq!(Path::new(&hymt.args[1]), gpu_llama);

        // A pack whose llama-server failed its install check keeps CUDA for
        // Qwen and leaves translation on the CPU build.
        assert!(
            manager
                .select_gpu_runtime(Some(GpuRuntimeLayout {
                    sidecar: Some(gpu_sidecar.clone()),
                    llama_server: None,
                }))
                .await
        );
        assert!(manager.uses_gpu("qwen_asr") && !manager.uses_gpu("hymt"));
        assert_eq!(
            Path::new(&manager.command_for("hymt").unwrap().args[1]),
            cpu_llama
        );

        assert!(manager.select_gpu_runtime(None).await);
        assert!(!manager.gpu_active());
        let qwen = manager.command_for("qwen_asr").unwrap();
        assert_eq!(qwen.executable, cpu_sidecar);
        assert!(qwen
            .args
            .windows(2)
            .any(|pair| pair[0] == "--qwen3-streaming-device" && pair[1] == "mps"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_gpu_start_falls_back_for_that_service_only() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("models").join("qwen3-asr-0.6b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.safetensors"), b"model").unwrap();
        let script = |name: &str, body: &str| {
            let path = directory.path().join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&path, permissions).unwrap();
            path
        };
        let gpu_sidecar = script(
            "gpu-sidecar",
            "echo 'CUDA driver version is insufficient' >&2\nexit 3",
        );
        let cpu_sidecar = script("cpu-sidecar", "echo 'cpu runtime ran' >&2\nexit 5");
        let manager = LocalRuntimeManager::new(runtime_layout(directory.path(), cpu_sidecar));
        let gpu_llama = directory.path().join("llama-server");
        let pack = GpuRuntimeLayout {
            sidecar: Some(gpu_sidecar),
            llama_server: Some(gpu_llama.clone()),
        };
        manager.select_gpu_runtime(Some(pack.clone())).await;

        let error = manager.ensure_service("qwen_asr").await.unwrap_err();
        // The CPU runtime was tried after the GPU failure.
        assert!(matches!(
            error,
            LocalRuntimeError::StartupExit { status, diagnostics, .. }
                if status.contains('5') && diagnostics.contains("cpu runtime ran")
        ));
        let reason = manager.gpu_fallback_for("qwen_asr").unwrap();
        assert!(reason.contains("qwen_asr") && reason.contains(".gpu.log"), "{reason}");
        assert_eq!(manager.gpu_fallback_reason(), Some(reason));
        assert_eq!(manager.gpu_fallback_for("hymt"), None);
        // Translation keeps the Vulkan llama-server.
        assert!(!manager.uses_gpu("qwen_asr") && manager.uses_gpu("hymt"));
        assert!(manager.gpu_active());
        assert!(
            std::fs::read_to_string(directory.path().join("logs").join("qwen_asr.gpu.log"))
                .unwrap()
                .contains("CUDA driver version is insufficient")
        );
        // The fallback holds until the user retries explicitly.
        assert_eq!(
            manager.command_for("qwen_asr").unwrap().executable,
            directory.path().join("cpu-sidecar")
        );
        manager.clear_gpu_fallback();
        assert!(manager.uses_gpu("qwen_asr"));
        // Switching the runtime off and on again also retries.
        manager.ensure_service("qwen_asr").await.unwrap_err();
        assert!(manager.gpu_fallback_reason().is_some());
        manager.select_gpu_runtime(None).await;
        manager.select_gpu_runtime(Some(pack)).await;
        assert_eq!(manager.gpu_fallback_reason(), None);
    }

    const MB: u64 = 1024 * 1024;

    #[test]
    fn stall_timeout_comes_from_the_environment_within_its_range() {
        assert_eq!(stall_timeout_from(None), Duration::from_secs(60));
        assert_eq!(stall_timeout_from(Some("90")), Duration::from_secs(90));
        assert_eq!(stall_timeout_from(Some(" 30 ")), Duration::from_secs(30));
        assert_eq!(stall_timeout_from(Some("1")), Duration::from_secs(5));
        assert_eq!(stall_timeout_from(Some("99999")), Duration::from_secs(600));
        assert_eq!(stall_timeout_from(Some("soon")), Duration::from_secs(60));
    }

    #[test]
    fn only_changing_byte_counts_hold_off_the_stall_clock() {
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        let mut clock = StallClock::new(Duration::from_secs(60), start);
        assert_eq!(clock.time_left(start), Duration::from_secs(60));
        clock.file_bytes(MB, at(1));
        clock.xet_bytes(64 * MB, at(2));
        // Xet's poller repeating the same totals is not progress.
        for second in 3..62 {
            clock.xet_bytes(64 * MB, at(second));
            clock.file_bytes(MB, at(second));
        }
        assert_eq!(clock.time_left(at(61)), Duration::from_secs(1));
        assert!(clock.time_left(at(62)).is_zero());
    }

    #[test]
    fn a_slow_xet_transfer_gets_room_for_its_coarse_steps() {
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        let mut clock = StallClock::new(Duration::from_secs(60), start);
        // Small files over plain HTTP, then Xet starts on the large one.
        clock.file_bytes(20_000, at(1));
        clock.xet_bytes(0, at(2));
        // Several 64 MB terms share a slow link before the first is written.
        assert!(!clock.time_left(at(250)).is_zero());
        assert!(clock.time_left(at(301)).is_zero());
        clock.xet_bytes(64 * MB, at(250));
        // One-minute steps from now on are no stall either.
        clock.xet_bytes(128 * MB, at(320));
        assert!(!clock.time_left(at(320 + 600)).is_zero());
        assert!(clock.time_left(at(320 + 3 * 249)).is_zero());
    }

    #[test]
    fn a_fast_transfer_that_stops_stalls_after_the_timeout() {
        let start = Instant::now();
        let at = |millis: u64| start + Duration::from_millis(millis);
        let mut clock = StallClock::new(Duration::from_secs(60), start);
        clock.xet_bytes(0, at(500));
        for step in 1..=6 {
            clock.xet_bytes(step * 64 * MB, at(1_000 + step * 500));
        }
        assert!(clock.time_left(at(4_000 + 59_000)) > Duration::ZERO);
        assert!(clock.time_left(at(4_000 + 60_000)).is_zero());
    }

    #[test]
    fn a_retired_attempt_reports_nothing() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        let activity = Arc::new(DownloadActivity::new(Duration::from_secs(60)));
        let handler = ModelProgressHandler {
            model_id: "test-model".into(),
            callback: Arc::new(move |progress: ModelProgress| sink.lock().unwrap().push(progress)),
            note: Some("Download stalled; retrying (2/3)".into()),
            activity: activity.clone(),
            files: Mutex::new(HashMap::new()),
        };
        let event = ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: MB,
            total_bytes: 4 * MB,
            bytes_per_sec: None,
        });
        handler.on_progress(&event);
        activity.retire();
        handler.on_progress(&event);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].bytes_completed, MB);
        assert_eq!(
            events[0].message.as_deref(),
            Some("Download stalled; retrying (2/3)")
        );
    }

    /// A small model whose one file is `content`.
    fn test_model(content: &[u8]) -> ModelSpec {
        ModelSpec {
            id: "test-model".into(),
            display_name: "Test model".into(),
            role: ModelRole::Asr,
            repository: "test/model".into(),
            revision: "0".repeat(40),
            expected_bytes: content.len() as u64,
            required_file: "model.bin".into(),
            required_file_sha256: format!("{:x}", Sha256::digest(content)),
            download: DownloadKind::File {
                filename: "model.bin".into(),
            },
        }
    }

    type Recorded = Arc<Mutex<Vec<ModelProgress>>>;

    fn recording_callback() -> (ProgressCallback, Recorded) {
        let events: Recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        (
            Arc::new(move |progress: ModelProgress| sink.lock().unwrap().push(progress)),
            events,
        )
    }

    /// Counts its drops: a dropped future or task has really stopped.
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    async fn wait_for_count(counter: &AtomicUsize, expected: usize) -> bool {
        for _ in 0..200 {
            if counter.load(Ordering::SeqCst) == expected {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    fn aggregate(bytes_completed: u64) -> ProgressEvent {
        ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed,
            total_bytes: 100 * MB,
            bytes_per_sec: None,
        })
    }

    #[tokio::test]
    async fn a_download_that_never_progresses_is_stopped_and_retried_then_fails() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let calls = Arc::new(AtomicUsize::new(0));
        let futures_dropped = Arc::new(AtomicUsize::new(0));
        let transfers_dropped = Arc::new(AtomicUsize::new(0));
        let downloader: Downloader = {
            let calls = calls.clone();
            let futures_dropped = futures_dropped.clone();
            let transfers_dropped = transfers_dropped.clone();
            Arc::new(move |request: DownloadRequest| {
                calls.fetch_add(1, Ordering::SeqCst);
                let future_guard = DropCounter(futures_dropped.clone());
                let transfer_guard = DropCounter(transfers_dropped.clone());
                Box::pin(async move {
                    let _future_guard = future_guard;
                    // Like hf-hub's Xet transfer: a detached task that keeps
                    // writing into the staging directory.
                    let partial = request.staging.join("model.bin.incomplete");
                    tokio::spawn(async move {
                        let _transfer_guard = transfer_guard;
                        loop {
                            let _ = std::fs::write(&partial, b"partial");
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    });
                    std::future::pending::<Result<(), String>>().await
                })
            })
        };
        let manager =
            ModelManager::with_downloader(root.clone(), downloader, Duration::from_millis(100));
        let spec = test_model(b"weights");
        let (callback, events) = recording_callback();

        let error = manager.install_inner(&spec, callback).await.unwrap_err();

        assert!(
            matches!(&error, ModelManagerError::Download(message) if message.contains("stalled 3 times")),
            "{error}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        // Every attempt, and the transfer it left behind, really stopped.
        assert!(wait_for_count(&futures_dropped, 3).await);
        assert!(wait_for_count(&transfers_dropped, 3).await);
        for attempt in 1..=3 {
            assert!(!manager.staging_directory(&spec, attempt).exists());
        }
        assert!(!root.join("test-model").exists());
        let events = events.lock().unwrap();
        let retries: Vec<_> = events
            .iter()
            .filter(|event| event.phase == "retrying")
            .map(|event| event.message.clone().unwrap_or_default())
            .collect();
        assert_eq!(
            retries,
            [
                "Download stalled; retrying (2/3)",
                "Download stalled; retrying (3/3)"
            ]
        );
        assert_eq!(events[0].phase, "starting");
        assert_eq!(events[0].message, None);
    }

    #[tokio::test]
    async fn a_download_is_retried_only_once_its_bytes_stop_moving() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let started = Arc::new(Mutex::new(Vec::<Instant>::new()));
        let progress_window = Duration::from_millis(1_000);
        let downloader: Downloader = {
            let started = started.clone();
            Arc::new(move |request: DownloadRequest| {
                let attempt = {
                    let mut started = started.lock().unwrap();
                    started.push(Instant::now());
                    started.len()
                };
                Box::pin(async move {
                    if attempt == 1 {
                        let began = Instant::now();
                        let mut bytes = 0;
                        while began.elapsed() < progress_window {
                            bytes += MB;
                            request.progress.on_progress(&aggregate(bytes));
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                        // Stuck: the poller keeps repeating the same total.
                        loop {
                            request.progress.on_progress(&aggregate(bytes));
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                    }
                    std::future::pending::<Result<(), String>>().await
                })
            })
        };
        let manager = ModelManager::with_downloader(root, downloader, Duration::from_millis(300));
        let (callback, events) = recording_callback();

        let error = manager
            .install_inner(&test_model(b"weights"), callback)
            .await
            .unwrap_err();

        assert!(matches!(error, ModelManagerError::Download(_)));
        let started = started.lock().unwrap();
        assert_eq!(started.len(), 3);
        let first_attempt = started[1] - started[0];
        // Not stopped while bytes arrived, but soon after they stopped.
        assert!(first_attempt >= progress_window, "{first_attempt:?}");
        assert!(
            first_attempt < progress_window + Duration::from_secs(3),
            "{first_attempt:?}"
        );
        let events = events.lock().unwrap();
        let first_retry = events
            .iter()
            .position(|event| event.phase == "retrying")
            .unwrap();
        assert!(events[..first_retry]
            .iter()
            .any(|event| event.phase == "downloading" && event.bytes_completed > 0));
        // Nothing from the stalled attempt arrives after the retry began.
        assert!(events[first_retry..]
            .iter()
            .all(|event| event.bytes_completed == 0));
    }

    #[tokio::test]
    async fn a_stalled_download_succeeds_on_the_next_attempt() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let content = b"model weights".to_vec();
        let calls = Arc::new(AtomicUsize::new(0));
        let downloader: Downloader = {
            let calls = calls.clone();
            let content = content.clone();
            Arc::new(move |request: DownloadRequest| {
                let attempt = calls.fetch_add(1, Ordering::SeqCst) + 1;
                let content = content.clone();
                Box::pin(async move {
                    if attempt == 1 {
                        std::fs::write(request.staging.join("model.bin.incomplete"), b"part")
                            .unwrap();
                        return std::future::pending().await;
                    }
                    assert!(!request.staging.join("model.bin.incomplete").exists());
                    std::fs::write(request.staging.join("model.bin"), &content).unwrap();
                    request.progress.on_progress(&ProgressEvent::Download(
                        DownloadEvent::Progress {
                            files: vec![hf_hub::progress::FileProgress {
                                filename: "model.bin".into(),
                                bytes_completed: content.len() as u64,
                                total_bytes: content.len() as u64,
                                status: hf_hub::progress::FileStatus::Complete,
                            }],
                        },
                    ));
                    Ok(())
                })
            })
        };
        let manager =
            ModelManager::with_downloader(root.clone(), downloader, Duration::from_millis(100));
        let spec = test_model(&content);
        let (callback, events) = recording_callback();

        manager.install_inner(&spec, callback).await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            std::fs::read(root.join("test-model").join("model.bin")).unwrap(),
            content
        );
        assert!(installed_manifest_matches(&root.join("test-model"), &spec));
        assert!(!manager.staging_directory(&spec, 1).exists());
        assert!(!manager.staging_directory(&spec, 2).exists());
        let events = events.lock().unwrap();
        let phases: Vec<_> = events.iter().map(|event| event.phase.as_str()).collect();
        assert_eq!(
            phases,
            ["starting", "retrying", "downloading", "verifying", "ready"]
        );
        assert_eq!(
            events[2].message.as_deref(),
            Some("Download stalled; retrying (2/3)")
        );
        assert_eq!(events[4].message, None);
    }

    #[tokio::test]
    async fn a_failed_download_is_not_retried() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let calls = Arc::new(AtomicUsize::new(0));
        let downloader: Downloader = {
            let calls = calls.clone();
            Arc::new(move |_request: DownloadRequest| {
                calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Err("HTTP 404".to_string()) })
            })
        };
        let manager =
            ModelManager::with_downloader(root.clone(), downloader, Duration::from_millis(100));
        let spec = test_model(b"weights");
        let (callback, _events) = recording_callback();

        let error = manager.install_inner(&spec, callback).await.unwrap_err();

        assert!(matches!(&error, ModelManagerError::Download(message) if message == "HTTP 404"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!manager.staging_directory(&spec, 1).exists());
    }

    #[tokio::test]
    async fn an_install_clears_retry_directories_left_behind() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let spec = test_model(b"weights");
        let downloader: Downloader = Arc::new(|request: DownloadRequest| {
            Box::pin(async move {
                std::fs::write(request.staging.join("model.bin"), b"weights").unwrap();
                Ok(())
            })
        });
        let manager =
            ModelManager::with_downloader(root.clone(), downloader, Duration::from_millis(100));
        let leftover = manager.staging_directory(&spec, 3);
        std::fs::create_dir_all(&leftover).unwrap();
        std::fs::write(leftover.join("model.bin.incomplete"), b"part").unwrap();
        let (callback, _events) = recording_callback();

        manager.install_inner(&spec, callback).await.unwrap();

        assert!(!leftover.exists());
        assert!(root.join("test-model").join("model.bin").is_file());
    }

    #[test]
    fn activation_swaps_directories_and_tolerates_a_stuck_leftover() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("runtimes");
        let staging = root.join(".pack.installing");
        let destination = root.join("pack");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("version"), b"1").unwrap();
        activate_directory(&root, &staging, &destination).unwrap();
        assert_eq!(std::fs::read(destination.join("version")).unwrap(), b"1");

        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("version"), b"2").unwrap();
        activate_directory_retrying(&root, &staging, &destination, 3, Duration::ZERO).unwrap();
        assert_eq!(std::fs::read(destination.join("version")).unwrap(), b"2");
        assert!(!staging.exists() && !root.join(".pack.previous").exists());

        // Without a staging tree nothing is retried and the pack stays.
        let error =
            activate_directory_retrying(&root, &staging, &destination, 5, Duration::from_secs(5))
                .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(std::fs::read(destination.join("version")).unwrap(), b"2");
        assert!(!root.join(".pack.previous").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_runtime_surfaces_process_exit_and_log_tail() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("models/qwen3-asr-0.6b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.safetensors"), b"model").unwrap();
        let executable = directory.path().join("broken-qwen");
        std::fs::write(
            &executable,
            "#!/bin/sh\necho 'model runtime failed to load' >&2\nexit 44\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let manager = LocalRuntimeManager::new(runtime_layout(directory.path(), executable));

        let error = manager.ensure_service("qwen_asr").await.unwrap_err();
        assert!(matches!(
            error,
            LocalRuntimeError::StartupExit { status, diagnostics, .. }
                if status.contains("44") && diagnostics.contains("model runtime failed to load")
        ));
    }
}
