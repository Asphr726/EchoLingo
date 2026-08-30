//! Versioned Rust/Python inference and UI event protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::convert::TryInto;
use uuid::Uuid;

mod supervisor;
pub use supervisor::{InferenceSupervisor, SidecarLaunchConfig, SupervisorError};

pub const PROTOCOL_VERSION: u16 = 1;
pub const AUDIO_MAGIC: [u8; 4] = *b"ELAF";
pub const AUDIO_HEADER_BYTES: usize = 32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub protocol_version: u16,
    pub authentication_token: String,
    pub build: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum SidecarCommand {
    Hello(Hello),
    PlanSession(Value),
    StartSession(Value),
    Pause { session_id: Uuid, epoch: u32 },
    Resume { session_id: Uuid, epoch: u32 },
    FinishSession { session_id: Uuid },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum SidecarEvent {
    HelloAccepted {
        protocol_version: u16,
    },
    Ready {
        session_id: Uuid,
        route: Value,
    },
    RoutePlan {
        session_id: Uuid,
        route: Value,
        services_to_start: Vec<String>,
    },
    Transcript(Value),
    Translation(Value),
    Metrics(Value),
    SegmentCommitted(Value),
    AlignmentUpdate(Value),
    BackendHealth(Value),
    SessionFinished {
        session_id: Uuid,
    },
    Error {
        code: String,
        message: String,
        recoverable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiEventKind {
    SessionState,
    TranscriptRevision,
    TranslationRevision,
    SegmentCommitted,
    Metrics,
    RouteDecision,
    BackendHealth,
    AudioDeviceChange,
    SettingsChanged,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiEventEnvelope {
    pub schema_version: u16,
    pub sequence: u64,
    pub session_id: Option<Uuid>,
    pub kind: UiEventKind,
    pub emitted_at_unix_ms: i64,
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFrameHeader {
    pub flags: u16,
    pub sequence: u64,
    pub capture_monotonic_ns: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_count: u16,
}

impl AudioFrameHeader {
    pub fn encode(self) -> [u8; AUDIO_HEADER_BYTES] {
        let mut bytes = [0_u8; AUDIO_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&AUDIO_MAGIC);
        bytes[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.capture_monotonic_ns.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.sample_rate_hz.to_le_bytes());
        bytes[28..30].copy_from_slice(&self.channels.to_le_bytes());
        bytes[30..32].copy_from_slice(&self.frame_count.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AudioPacketError> {
        if bytes.len() < AUDIO_HEADER_BYTES {
            return Err(AudioPacketError::HeaderTooShort);
        }
        if bytes[0..4] != AUDIO_MAGIC {
            return Err(AudioPacketError::InvalidMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != PROTOCOL_VERSION {
            return Err(AudioPacketError::UnsupportedVersion(version));
        }
        Ok(Self {
            flags: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            sequence: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            capture_monotonic_ns: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            sample_rate_hz: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            channels: u16::from_le_bytes(bytes[28..30].try_into().unwrap()),
            frame_count: u16::from_le_bytes(bytes[30..32].try_into().unwrap()),
        })
    }

    pub fn validate_pcm_bytes(self, pcm_bytes: usize) -> Result<(), AudioPacketError> {
        let expected = self.frame_count as usize * self.channels as usize * size_of::<f32>();
        if pcm_bytes != expected {
            return Err(AudioPacketError::PcmLength {
                expected,
                actual: pcm_bytes,
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AudioPacketError {
    #[error("audio packet header is too short")]
    HeaderTooShort,
    #[error("audio packet magic is invalid")]
    InvalidMagic,
    #[error("unsupported audio packet version {0}")]
    UnsupportedVersion(u16),
    #[error("audio packet PCM length mismatch: expected {expected}, got {actual}")]
    PcmLength { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_header_round_trips_and_validates_pcm() {
        let header = AudioFrameHeader {
            flags: 1,
            sequence: 42,
            capture_monotonic_ns: 99,
            sample_rate_hz: 48_000,
            channels: 2,
            frame_count: 480,
        };
        assert_eq!(AudioFrameHeader::decode(&header.encode()).unwrap(), header);
        assert_eq!(header.validate_pcm_bytes(480 * 2 * 4), Ok(()));
        assert!(matches!(
            header.validate_pcm_bytes(8),
            Err(AudioPacketError::PcmLength { .. })
        ));
    }

    #[test]
    fn command_and_event_shapes_are_stable() {
        assert_eq!(
            serde_json::to_value(SidecarCommand::Shutdown).unwrap()["type"],
            "shutdown"
        );
        let event = UiEventEnvelope {
            schema_version: 1,
            sequence: 1,
            session_id: None,
            kind: UiEventKind::SessionState,
            emitted_at_unix_ms: 0,
            payload: serde_json::json!({"phase": "IDLE"}),
        };
        assert!(serde_json::to_string(&event)
            .unwrap()
            .contains("session_state"));
    }
}
