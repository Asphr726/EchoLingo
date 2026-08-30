use crate::{Hello, SidecarCommand, SidecarEvent, PROTOCOL_VERSION};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SidecarLaunchConfig {
    pub project_root: PathBuf,
    pub conda_environment: String,
    pub configured_url: Option<String>,
    pub startup_timeout: Duration,
}

impl SidecarLaunchConfig {
    pub fn development(project_root: PathBuf) -> Self {
        Self {
            project_root,
            conda_environment: "echolingo-spike1".into(),
            configured_url: std::env::var("ECHOLINGO_SIDECAR_URL").ok(),
            startup_timeout: Duration::from_secs(15),
        }
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
    child: Mutex<Option<Child>>,
    events: broadcast::Sender<SidecarEvent>,
    secret_environment: Mutex<HashMap<String, String>>,
}

impl InferenceSupervisor {
    pub fn new(config: SidecarLaunchConfig) -> Arc<Self> {
        let (events, _) = broadcast::channel(512);
        Arc::new(Self {
            config,
            commands: Mutex::new(None),
            child: Mutex::new(None),
            events,
            secret_environment: Mutex::new(HashMap::new()),
        })
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
            let mut command = Command::new("conda");
            command
                .current_dir(&self.config.project_root)
                .env("ECHOLINGO_IPC_TOKEN", &token)
                .args([
                    "run",
                    "--no-capture-output",
                    "-n",
                    &self.config.conda_environment,
                    "python",
                    "-m",
                    "echolingo.service",
                    "--port",
                    &port.to_string(),
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            for (name, value) in self.secret_environment.lock().await.iter() {
                command.env(name, value);
            }
            *self.child.lock().await = Some(command.spawn().map_err(SupervisorError::Launch)?);
            format!("ws://127.0.0.1:{port}")
        };

        let deadline = tokio::time::Instant::now() + self.config.startup_timeout;
        let mut websocket = loop {
            match connect_async(&url).await {
                Ok((websocket, _)) => break websocket,
                Err(error) if tokio::time::Instant::now() < deadline => {
                    let _ = error;
                    sleep(Duration::from_millis(100)).await;
                }
                Err(_) => return Err(SupervisorError::StartupTimeout),
            }
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
        if !matches!(
            event,
            SidecarEvent::HelloAccepted {
                protocol_version: PROTOCOL_VERSION
            }
        ) {
            return Err(SupervisorError::Protocol(
                "sidecar rejected protocol hello".into(),
            ));
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
                supervisor.commands.lock().await.take();
                if let Some(mut child) = supervisor.child.lock().await.take() {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
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

    pub async fn shutdown(&self) {
        let _ = self.send_command(SidecarCommand::Shutdown).await;
        self.commands.lock().await.take();
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioFrameHeader;
    use serde_json::json;

    #[tokio::test]
    #[ignore = "requires loopback sockets and the echolingo-spike1 Conda environment"]
    async fn rust_supervises_python_sidecar_end_to_end() {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let supervisor = InferenceSupervisor::new(SidecarLaunchConfig {
            project_root,
            conda_environment: "echolingo-spike1".into(),
            configured_url: None,
            startup_timeout: Duration::from_secs(20),
        });
        supervisor.ensure_started().await.unwrap();
        let mut events = supervisor.subscribe();
        let session_id = Uuid::new_v4();
        supervisor
            .send_command(SidecarCommand::StartSession(json!({
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
            })))
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
