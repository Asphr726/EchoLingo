//! Managed, pinned local-model installation for the desktop product.

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex as AsyncMutex;

pub const MODEL_CATALOG_VERSION: u16 = 1;

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

struct ManagedService {
    child: Child,
}

pub struct LocalRuntimeManager {
    layout: LocalRuntimeLayout,
    services: AsyncMutex<HashMap<String, ManagedService>>,
    port_reservations: Mutex<HashMap<String, std::net::TcpListener>>,
}

impl LocalRuntimeManager {
    pub fn new(layout: LocalRuntimeLayout) -> Self {
        Self {
            layout,
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

    pub fn command_for(&self, service: &str) -> Result<RuntimeCommand, LocalRuntimeError> {
        match service {
            "qwen_asr" => {
                let model = self.layout.model_root.join("qwen3-asr-0.6b");
                if !model.join("model.safetensors").is_file() {
                    return Err(LocalRuntimeError::ModelUnavailable("qwen3-asr-0.6b".into()));
                }
                let mut command = self.layout.qwen_command.clone();
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
                    self.layout.qwen_device.clone(),
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
                let executable =
                    self.layout.llama_server.clone().ok_or_else(|| {
                        LocalRuntimeError::RuntimeUnavailable("llama-server".into())
                    })?;
                let model = self
                    .layout
                    .model_root
                    .join("hymt2-1.8b/Hy-MT2-1.8B-Q4_K_M.gguf");
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
            let _ = previous.child.start_kill();
            let _ = previous.child.wait().await;
        }
        let spec = self.command_for(service)?;
        std::fs::create_dir_all(&self.layout.log_root)?;
        let log_path = self.layout.log_root.join(format!("{service}.log"));
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
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr))
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .map_err(|source| LocalRuntimeError::Launch {
                service: service.into(),
                source,
            })?;
        let deadline = tokio::time::Instant::now() + self.layout.startup_timeout;
        loop {
            if service_healthy(port, health_path).await {
                services.insert(service.into(), ManagedService { child });
                return Ok(());
            }
            if let Some(status) = child.try_wait().map_err(LocalRuntimeError::Io)? {
                return Err(LocalRuntimeError::StartupExit {
                    service: service.into(),
                    status: status.to_string(),
                    diagnostics: log_tail(&log_path, 4096),
                    log_path,
                });
            }
            if tokio::time::Instant::now() >= deadline {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(LocalRuntimeError::StartupTimeout {
                    service: service.into(),
                    seconds: self.layout.startup_timeout.as_secs(),
                    log_path,
                });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    pub async fn shutdown(&self) {
        let mut services = self.services.lock().await;
        for (_, mut service) in services.drain() {
            let _ = service.child.start_kill();
            let _ = service.child.wait().await;
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

pub struct ModelManager {
    root: PathBuf,
    cache: PathBuf,
    active: Mutex<HashSet<String>>,
}

impl ModelManager {
    pub fn new(root: PathBuf) -> Self {
        let cache = root.join(".downloads");
        Self {
            root,
            cache,
            active: Mutex::new(HashSet::new()),
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

        let staging = self.root.join(format!(".{}.installing", spec.id));
        remove_scoped_directory(&self.root, &staging)?;
        std::fs::create_dir_all(&staging)?;
        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: 0,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "starting".into(),
        });

        let client = HFClient::builder()
            .cache_dir(&self.cache)
            .user_agent(format!("echolingo/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| ModelManagerError::Download(error.to_string()))?;
        let (owner, name) = spec
            .repository
            .split_once('/')
            .ok_or_else(|| ModelManagerError::Download("invalid repository name".into()))?;
        let repository = client.model(owner, name);
        let progress = ModelProgressHandler {
            model_id: spec.id.clone(),
            callback: callback.clone(),
            files: Mutex::new(HashMap::new()),
        };
        let download_result = match &spec.download {
            DownloadKind::Snapshot { allow_patterns } => repository
                .snapshot_download()
                .revision(spec.revision.clone())
                .allow_patterns(allow_patterns.clone())
                .local_dir(staging.clone())
                .max_workers(4)
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
            DownloadKind::File { filename } => repository
                .download_file()
                .filename(filename.clone())
                .revision(spec.revision.clone())
                .local_dir(staging.clone())
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
        };
        if let Err(error) = download_result {
            remove_scoped_directory(&self.root, &staging)?;
            return Err(ModelManagerError::Download(error.to_string()));
        }

        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: spec.expected_bytes,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "verifying".into(),
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
        });
        Ok(())
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

struct ModelProgressHandler {
    model_id: String,
    callback: ProgressCallback,
    files: Mutex<HashMap<String, (u64, u64)>>,
}

impl ProgressHandler for ModelProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let progress = match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: 0,
                total_bytes: *total_bytes,
                bytes_per_second: None,
                phase: "downloading".into(),
            },
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
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed: cumulative.values().map(|value| value.0).sum(),
                    total_bytes: cumulative.values().map(|value| value.1).sum(),
                    bytes_per_second: None,
                    phase: "downloading".into(),
                }
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: *bytes_completed,
                total_bytes: *total_bytes,
                bytes_per_second: *bytes_per_sec,
                phase: "downloading".into(),
            },
            ProgressEvent::Download(DownloadEvent::Complete) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: 0,
                total_bytes: 0,
                bytes_per_second: None,
                phase: "downloaded".into(),
            },
            _ => return,
        };
        (self.callback)(progress);
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
    let previous = root.join(format!(
        ".{}.previous",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    ));
    remove_scoped_directory(root, &previous)?;
    if destination.exists() {
        std::fs::rename(destination, &previous)?;
    }
    if let Err(error) = std::fs::rename(staging, destination) {
        if previous.exists() {
            let _ = std::fs::rename(&previous, destination);
        }
        return Err(error);
    }
    remove_scoped_directory(root, &previous)
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
        let model = directory.path().join("models/qwen3-asr-0.6b");
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
            .any(|pair| { pair[0] == "--model_dir" && pair[1] == model.to_string_lossy() }));
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
        assert_eq!(command.args[1], llama.to_string_lossy());
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
