use crate::{Hello, SidecarCommand, SidecarEvent, PROTOCOL_VERSION};
use futures_util::{SinkExt, StreamExt};
use process_support::ProcessTree;
use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use uuid::Uuid;

/// Upper bound for one sidecar stderr log file before it rotates to `<name>.1`.
pub const SIDECAR_LOG_ROTATE_BYTES: u64 = 5 * 1024 * 1024;
const DIAGNOSTICS_RING_BYTES: usize = 16 * 1024;
/// How long a sidecar asked to shut down may take before its process tree is
/// terminated.
const SHUTDOWN_EXIT_GRACE: Duration = Duration::from_secs(2);
/// SIGTERM-to-SIGKILL grace when the process tree is terminated.
const TERMINATE_GRACE: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
pub struct SidecarLaunchConfig {
    pub project_root: PathBuf,
    pub conda_environment: String,
    pub executable: Option<PathBuf>,
    pub configured_url: Option<String>,
    pub startup_timeout: Duration,
    /// When set, every sidecar stderr chunk is appended to this file (parent
    /// directories are created; the file rotates to `<name>.1` past
    /// [`SIDECAR_LOG_ROTATE_BYTES`]) in addition to the in-memory diagnostics
    /// ring used for startup failures. `None` keeps stderr in memory only.
    pub log_path: Option<PathBuf>,
}

impl SidecarLaunchConfig {
    pub fn development(project_root: PathBuf) -> Self {
        Self {
            project_root,
            conda_environment: "echolingo-spike1".into(),
            executable: std::env::var_os("ECHOLINGO_SIDECAR_EXECUTABLE").map(PathBuf::from),
            configured_url: std::env::var("ECHOLINGO_SIDECAR_URL").ok(),
            startup_timeout: Duration::from_secs(15),
            log_path: None,
        }
    }

    /// The packaged-app configuration: `ECHOLINGO_SIDECAR_EXECUTABLE` when
    /// set, else the `bundled` sidecar the shell resolved from its resources,
    /// else the development Conda environment.
    pub fn desktop(project_root: PathBuf, bundled: Option<PathBuf>) -> Self {
        let mut config = Self::development(project_root);
        if config.executable.is_none() {
            config.executable = bundled;
        }
        if config.executable.is_some() {
            config.startup_timeout = Duration::from_secs(120);
        }
        config
    }
}

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("cannot allocate a loopback port: {0}")]
    Port(std::io::Error),
    #[error("cannot launch inference sidecar: {0}")]
    Launch(std::io::Error),
    #[error("inference sidecar startup timed out")]
    StartupTimeout,
    #[error("inference sidecar exited during startup ({status}): {diagnostics}")]
    StartupExit { status: String, diagnostics: String },
    #[error("cannot inspect inference sidecar process: {0}")]
    ProcessStatus(std::io::Error),
    #[error("inference sidecar WebSocket failed: {0}")]
    WebSocket(String),
    #[error("inference sidecar protocol error: {0}")]
    Protocol(String),
    #[error("inference sidecar command channel is closed")]
    ChannelClosed,
}

pub struct InferenceSupervisor {
    config: SidecarLaunchConfig,
    commands: Mutex<Option<mpsc::Sender<Message>>>,
    child: Mutex<Option<ProcessTree>>,
    /// Incremented by every launch (under the `child` lock) so the
    /// disconnect cleanup of an earlier connection can never stop the
    /// sidecar that replaced it.
    generation: AtomicU64,
    events: broadcast::Sender<SidecarEvent>,
    secret_environment: Mutex<HashMap<String, String>>,
    /// `providers_digest` from the last accepted hello, when the sidecar sent one.
    providers_digest: Mutex<Option<String>>,
}

impl InferenceSupervisor {
    pub fn new(config: SidecarLaunchConfig) -> Arc<Self> {
        let (events, _) = broadcast::channel(512);
        Arc::new(Self {
            config,
            commands: Mutex::new(None),
            child: Mutex::new(None),
            generation: AtomicU64::new(0),
            events,
            secret_environment: Mutex::new(HashMap::new()),
            providers_digest: Mutex::new(None),
        })
    }

    /// The provider-catalog digest the running sidecar reported in its hello,
    /// for comparison with the catalog embedded in the shell.
    pub async fn providers_digest(&self) -> Option<String> {
        self.providers_digest.lock().await.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SidecarEvent> {
        self.events.subscribe()
    }

    pub async fn configure_secret_environment(&self, values: HashMap<String, String>) {
        *self.secret_environment.lock().await = values;
    }

    pub async fn is_connected(&self) -> bool {
        self.commands.lock().await.is_some()
    }

    pub async fn ensure_started(self: &Arc<Self>) -> Result<(), SupervisorError> {
        if self.commands.lock().await.is_some() {
            return Ok(());
        }
        let generation = {
            let _child = self.child.lock().await;
            self.generation.fetch_add(1, Ordering::AcqRel) + 1
        };
        let token = if self.config.configured_url.is_some() {
            std::env::var("ECHOLINGO_IPC_TOKEN").unwrap_or_else(|_| {
                format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
            })
        } else {
            format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
        };
        let url = if let Some(url) = &self.config.configured_url {
            url.clone()
        } else {
            let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(SupervisorError::Port)?;
            let port = listener.local_addr().map_err(SupervisorError::Port)?.port();
            drop(listener);
            let (mut command, working_directory) = if let Some(executable) = &self.config.executable
            {
                let mut command = Command::new(executable);
                command.args(["--port", &port.to_string()]);
                let working_directory = executable
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."));
                (command, working_directory)
            } else {
                let mut command = Command::new("conda");
                command.args([
                    "run",
                    "--no-capture-output",
                    "-n",
                    &self.config.conda_environment,
                    "python",
                    "-m",
                    "echolingo.service",
                    "--port",
                    &port.to_string(),
                ]);
                (command, self.config.project_root.clone())
            };
            command
                .current_dir(working_directory)
                .env("ECHOLINGO_IPC_TOKEN", &token)
                .env("ECHOLINGO_PARENT_PID", std::process::id().to_string())
                .env("PYTHONUTF8", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            for (name, value) in self.secret_environment.lock().await.iter() {
                command.env(name, value);
            }
            // The sidecar's own children (the Conda wrapper's Python, model
            // workers) belong to its process group / job and stop with it.
            let mut child = ProcessTree::spawn(&mut command).map_err(SupervisorError::Launch)?;
            let diagnostics = Arc::new(Mutex::new(Vec::new()));
            let log_path = self.config.log_path.clone();
            let stderr_task = child.child_mut().stderr.take().map(|mut stderr| {
                let diagnostics = diagnostics.clone();
                tokio::spawn(async move {
                    let mut chunk = [0_u8; 2048];
                    let mut log_failed = false;
                    while let Ok(read) = stderr.read(&mut chunk).await {
                        if read == 0 {
                            break;
                        }
                        {
                            let mut output = diagnostics.lock().await;
                            output.extend_from_slice(&chunk[..read]);
                            if output.len() > DIAGNOSTICS_RING_BYTES {
                                let excess = output.len() - DIAGNOSTICS_RING_BYTES;
                                output.drain(..excess);
                            }
                        }
                        if let Some(path) = log_path.as_deref() {
                            if !log_failed {
                                if let Err(error) =
                                    append_log_chunk(path, &chunk[..read], SIDECAR_LOG_ROTATE_BYTES)
                                {
                                    // Logging never interferes with the sidecar;
                                    // report once and keep the in-memory ring.
                                    eprintln!(
                                        "sidecar log file unavailable ({}): {error}",
                                        path.display()
                                    );
                                    log_failed = true;
                                }
                            }
                        }
                    }
                })
            });
            *self.child.lock().await = Some(child);
            self.wait_for_websocket_or_exit(
                &format!("ws://127.0.0.1:{port}"),
                diagnostics,
                stderr_task,
            )
            .await?
        };

        let mut websocket = if self.config.configured_url.is_some() {
            self.wait_for_configured_websocket(&url).await?
        } else {
            connect_async(&url)
                .await
                .map_err(|error| SupervisorError::WebSocket(error.to_string()))?
                .0
        };
        let hello = SidecarCommand::Hello(Hello {
            protocol_version: PROTOCOL_VERSION,
            authentication_token: token,
            build: env!("CARGO_PKG_VERSION").into(),
            capabilities: vec!["binary_pcm_f32le".into(), "canonical_events_v2".into()],
        });
        websocket
            .send(Message::Text(
                serde_json::to_string(&hello)
                    .map_err(|error| SupervisorError::Protocol(error.to_string()))?
                    .into(),
            ))
            .await
            .map_err(|error| SupervisorError::WebSocket(error.to_string()))?;
        let accepted = tokio::time::timeout(Duration::from_secs(5), websocket.next())
            .await
            .map_err(|_| SupervisorError::StartupTimeout)?
            .ok_or_else(|| SupervisorError::Protocol("sidecar closed during hello".into()))?
            .map_err(|error| SupervisorError::WebSocket(error.to_string()))?;
        let event: SidecarEvent = serde_json::from_str(
            accepted
                .to_text()
                .map_err(|error| SupervisorError::Protocol(error.to_string()))?,
        )
        .map_err(|error| SupervisorError::Protocol(error.to_string()))?;
        match event {
            SidecarEvent::HelloAccepted {
                protocol_version: PROTOCOL_VERSION,
                providers_digest,
            } => {
                *self.providers_digest.lock().await = providers_digest;
            }
            _ => {
                return Err(SupervisorError::Protocol(
                    "sidecar rejected protocol hello".into(),
                ))
            }
        }

        let (mut writer, mut reader) = websocket.split();
        let (commands, mut outbound) = mpsc::channel::<Message>(512);
        *self.commands.lock().await = Some(commands);
        let event_bus = self.events.clone();
        let weak_supervisor = Arc::downgrade(self);
        tokio::spawn(async move {
            while let Some(message) = outbound.recv().await {
                if writer.send(message).await.is_err() {
                    break;
                }
            }
            let _ = writer.close().await;
        });
        tokio::spawn(async move {
            while let Some(Ok(message)) = reader.next().await {
                if let Ok(text) = message.to_text() {
                    if let Ok(event) = serde_json::from_str::<SidecarEvent>(text) {
                        let _ = event_bus.send(event);
                    }
                }
            }
            if let Some(supervisor) = weak_supervisor.upgrade() {
                let child = {
                    let mut child = supervisor.child.lock().await;
                    if supervisor.generation.load(Ordering::Acquire) != generation {
                        // A newer launch owns the sidecar now.
                        return;
                    }
                    supervisor.commands.lock().await.take();
                    child.take()
                };
                if let Some(mut child) = child {
                    let _ = child.terminate(TERMINATE_GRACE).await;
                }
                let _ = event_bus.send(SidecarEvent::Error {
                    code: "sidecar_disconnected".into(),
                    message: "Inference process disconnected; the desktop remains available".into(),
                    recoverable: true,
                });
            }
        });
        Ok(())
    }

    async fn wait_for_configured_websocket(
        &self,
        url: &str,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        SupervisorError,
    > {
        let deadline = tokio::time::Instant::now() + self.config.startup_timeout;
        loop {
            match connect_async(url).await {
                Ok((websocket, _)) => return Ok(websocket),
                Err(_) if tokio::time::Instant::now() < deadline => {
                    sleep(Duration::from_millis(100)).await;
                }
                Err(_) => return Err(SupervisorError::StartupTimeout),
            }
        }
    }

    async fn wait_for_websocket_or_exit(
        &self,
        url: &str,
        diagnostics: Arc<Mutex<Vec<u8>>>,
        stderr_task: Option<tokio::task::JoinHandle<()>>,
    ) -> Result<String, SupervisorError> {
        let deadline = tokio::time::Instant::now() + self.config.startup_timeout;
        loop {
            match connect_async(url).await {
                Ok((websocket, _)) => {
                    drop(websocket);
                    return Ok(url.to_string());
                }
                Err(_) => {}
            }
            let exited = {
                let mut child = self.child.lock().await;
                match child.as_mut() {
                    Some(child) => child.try_wait().map_err(SupervisorError::ProcessStatus)?,
                    None => None,
                }
            };
            if let Some(status) = exited {
                self.child.lock().await.take();
                if let Some(task) = stderr_task {
                    let _ = task.await;
                }
                let output = diagnostics.lock().await;
                let diagnostics = String::from_utf8_lossy(&output).trim().to_string();
                return Err(SupervisorError::StartupExit {
                    status: status.to_string(),
                    diagnostics: if diagnostics.is_empty() {
                        "no diagnostics were produced".into()
                    } else {
                        diagnostics
                    },
                });
            }
            if tokio::time::Instant::now() >= deadline {
                if let Some(mut child) = self.child.lock().await.take() {
                    let _ = child.terminate(TERMINATE_GRACE).await;
                }
                return Err(SupervisorError::StartupTimeout);
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn send_command(&self, command: SidecarCommand) -> Result<(), SupervisorError> {
        let message = Message::Text(
            serde_json::to_string(&command)
                .map_err(|error| SupervisorError::Protocol(error.to_string()))?
                .into(),
        );
        self.commands
            .lock()
            .await
            .as_ref()
            .ok_or(SupervisorError::ChannelClosed)?
            .send(message)
            .await
            .map_err(|_| SupervisorError::ChannelClosed)
    }

    pub async fn send_audio(&self, packet: Vec<u8>) -> Result<(), SupervisorError> {
        self.commands
            .lock()
            .await
            .as_ref()
            .ok_or(SupervisorError::ChannelClosed)?
            .send(Message::Binary(packet.into()))
            .await
            .map_err(|_| SupervisorError::ChannelClosed)
    }

    /// Ask the sidecar to exit, give it [`SHUTDOWN_EXIT_GRACE`] to do so and
    /// then terminate whatever remains of its process tree.
    pub async fn shutdown(&self) {
        let _ = self.send_command(SidecarCommand::Shutdown).await;
        self.commands.lock().await.take();
        let child = self.child.lock().await.take();
        if let Some(mut child) = child {
            let _ = child.wait_timeout(SHUTDOWN_EXIT_GRACE).await;
            let _ = child.terminate(TERMINATE_GRACE).await;
        }
    }
}

/// Append one stderr chunk to `path`, creating parent directories and rotating
/// the current file to `<name>.1` (replacing any previous `.1`) once it would
/// exceed `max_bytes`. Chunks are written verbatim; the sidecar itself never
/// logs credentials.
pub fn append_log_chunk(path: &Path, chunk: &[u8], max_bytes: u64) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let current_len = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if current_len > 0 && current_len + chunk.len() as u64 > max_bytes {
        let rotated = rotated_log_path(path);
        let _ = std::fs::remove_file(&rotated);
        std::fs::rename(path, &rotated)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(chunk)
}

fn rotated_log_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "sidecar.log".into());
    name.push(".1");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioFrameHeader;
    use serde_json::json;

    #[test]
    fn stderr_log_appends_creates_parents_and_rotates() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("logs/nested/sidecar.log");
        append_log_chunk(&path, b"first line\n", 32).unwrap();
        append_log_chunk(&path, b"second line\n", 32).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "first line\nsecond line\n"
        );
        // 23 bytes so far; the next chunk would exceed 32 and rotates first.
        append_log_chunk(&path, b"third line\n", 32).unwrap();
        let rotated = directory.path().join("logs/nested/sidecar.log.1");
        assert_eq!(
            std::fs::read_to_string(&rotated).unwrap(),
            "first line\nsecond line\n"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "third line\n");
        // A second rotation replaces the previous `.1` rather than stacking.
        append_log_chunk(&path, &[b'x'; 30], 32).unwrap();
        assert_eq!(std::fs::read_to_string(&rotated).unwrap(), "third line\n");
        assert_eq!(std::fs::read(&path).unwrap().len(), 30);
        // A chunk larger than the limit on an empty file is still written.
        std::fs::remove_file(&path).unwrap();
        append_log_chunk(&path, &[b'y'; 40], 32).unwrap();
        assert_eq!(std::fs::read(&path).unwrap().len(), 40);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn startup_reports_child_exit_without_waiting_for_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("broken-sidecar");
        std::fs::write(
            &executable,
            "#!/bin/sh\necho 'dyld: incompatible embedded library signature' >&2\nexit 86\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();

        let supervisor = InferenceSupervisor::new(SidecarLaunchConfig {
            project_root: directory.path().to_path_buf(),
            conda_environment: "unused".into(),
            executable: Some(executable),
            configured_url: None,
            startup_timeout: Duration::from_secs(5),
            log_path: Some(directory.path().join("logs/sidecar.log")),
        });
        let started = std::time::Instant::now();
        let error = supervisor.ensure_started().await.unwrap_err();

        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(
            matches!(
                &error,
                SupervisorError::StartupExit { status, diagnostics }
                    if status.contains("86")
                        && diagnostics.contains("incompatible embedded library signature")
            ),
            "unexpected startup error: {error:?}"
        );
        let logged = std::fs::read_to_string(directory.path().join("logs/sidecar.log")).unwrap();
        assert!(logged.contains("incompatible embedded library signature"));
    }

    #[tokio::test]
    #[ignore = "requires loopback sockets and the echolingo-spike1 Conda environment"]
    async fn rust_supervises_python_sidecar_end_to_end() {
        let executable = std::env::var_os("ECHOLINGO_SIDECAR_EXECUTABLE").map(PathBuf::from);
        let project_root = if executable.is_some() {
            PathBuf::from("/path/intentionally/absent-from-packaged-sidecar-test")
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
        };
        let supervisor = InferenceSupervisor::new(SidecarLaunchConfig {
            project_root,
            conda_environment: "echolingo-spike1".into(),
            executable: executable.clone(),
            configured_url: None,
            startup_timeout: Duration::from_secs(if executable.is_some() { 120 } else { 20 }),
            log_path: None,
        });
        supervisor.ensure_started().await.unwrap();
        let mut events = supervisor.subscribe();
        let session_id = Uuid::new_v4();
        let payload = json!({
            "session_id": session_id,
            "source_language": "en",
            "target_language": "zh",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "audio_profile": "raw",
            "inference_mode": "auto",
            "asr_provider": "mock",
            "translation_provider": "mock",
            "alignment_enabled": true,
            "alignment_provider": "mock",
            "privacy": {"audio_upload_allowed": false, "transcript_upload_allowed": false}
        });
        supervisor
            .send_command(SidecarCommand::PlanSession(payload.clone()))
            .await
            .unwrap();
        let plan = tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            plan,
            SidecarEvent::RoutePlan { session_id: planned_id, services_to_start, .. }
                if planned_id == session_id && services_to_start.is_empty()
        ));
        supervisor
            .send_command(SidecarCommand::StartSession(payload))
            .await
            .unwrap();
        let ready = tokio::time::timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(ready, SidecarEvent::Ready { session_id: ready_id, .. } if ready_id == session_id)
        );

        let header = AudioFrameHeader {
            flags: 0,
            sequence: 10,
            capture_monotonic_ns: 100,
            sample_rate_hz: 16_000,
            channels: 1,
            frame_count: 160,
        };
        let mut packet = header.encode().to_vec();
        for _ in 0..160 {
            packet.extend_from_slice(&0.25_f32.to_le_bytes());
        }
        supervisor.send_audio(packet).await.unwrap();
        let transcript = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let SidecarEvent::Transcript(value) = events.recv().await.unwrap() {
                    break value;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(transcript["provider"], "mock");
        supervisor
            .send_command(SidecarCommand::FinishSession { session_id })
            .await
            .unwrap();
        let alignment = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let SidecarEvent::AlignmentUpdate(value) = events.recv().await.unwrap() {
                    break value;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(alignment["session_id"], session_id.to_string());
        assert_eq!(alignment["timestamp_quality"], "forced");
        supervisor.shutdown().await;
    }
}
