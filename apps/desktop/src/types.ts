export type SessionPhase =
  | "IDLE"
  | "STARTING"
  | "LISTENING"
  | "PAUSED"
  | "STOPPING"
  | "COMPLETED";

export type AudioSourceKind =
  | "microphone"
  | "system_audio"
  | "system_audio_and_microphone";

export type InferenceMode = "auto" | "local" | "cloud";
export type CaptionDisplay = "both" | "original" | "translation";

export interface PrivacyPolicy {
  audio_upload_allowed: boolean;
  transcript_upload_allowed: boolean;
}

export interface StartSessionRequest {
  expected_state_revision: number;
  source_language: string;
  target_language: string;
  audio_source: AudioSourceKind;
  audio_device_id: string | null;
  audio_profile: string;
  inference_mode: InferenceMode;
  asr_provider: string;
  translation_provider: string;
  privacy: PrivacyPolicy;
}

export interface RouteStatus {
  asr_provider: string;
  asr_model: string | null;
  asr_locality: string;
  asr_health: string;
  translation_provider: string;
  translation_model: string | null;
  translation_locality: string;
  translation_health: string;
  deployment: string;
  reason: string;
}

export interface LiveMetrics {
  input_rms_dbfs?: number | null;
  enhanced_rms_dbfs?: number | null;
  vad_probability?: number | null;
  speech_detected: boolean;
  frontend_latency_ms?: number | null;
  asr_first_partial_latency_ms?: number | null;
  asr_commit_latency_ms?: number | null;
  translation_latency_ms?: number | null;
  end_to_end_latency_ms?: number | null;
  cloud_roundtrip_latency_ms?: number | null;
  network_jitter_ms?: number | null;
  reconnect_count: number;
  buffered_audio_ms: number;
  dropped_audio_ms: number;
}

export interface SegmentSummary {
  id: string;
  ordinal: number;
  start_ms: number;
  end_ms: number;
  original: string;
  translation: string;
}

export interface SessionSnapshot {
  state_revision: number;
  phase: SessionPhase;
  session_id: string | null;
  started_at: string | null;
  ended_at: string | null;
  config: StartSessionRequest | null;
  route: RouteStatus | null;
  live: {
    original_committed: string;
    original_unstable: string;
    translation_committed: string;
    translation_editable: string;
    source_revision_id: number;
    translation_revision_id: number;
    translation_source_revision_id: number;
  };
  previous_segments: SegmentSummary[];
  metrics: LiveMetrics;
  recoverable_error: string | null;
  startup_status?: string | null;
}

export interface AudioDevice {
  id: string;
  name: string;
  kind: "microphone" | "system_audio";
  is_default: boolean;
  available: boolean;
  requires_picker: boolean;
}

export type PermissionState = "not_determined" | "denied" | "granted" | "unavailable";

export interface AudioPermissionStatus {
  microphone: PermissionState;
  system_audio: PermissionState;
}

export interface AudioTestResult {
  source: string;
  sample_rate_hz: number;
  channels: number;
  peak_rms_dbfs: number;
  frames_observed: number;
}

export interface CaptionPreferences {
  display: CaptionDisplay;
  recent_segments: number;
  font_size_px: number;
  opacity: number;
}

export interface CloudCredentialStatus {
  api_key_available: boolean;
  workspace_id_available: boolean;
  source: "macos_keychain" | "environment" | "none";
}

export interface CloudProbeResult {
  ok: boolean;
  region: string;
  audio_uploaded: boolean;
  code?: string;
  message?: string;
  asr: {
    status: string;
    model?: string;
    handshake_latency_ms?: number;
  };
  translation: {
    status: string;
    model?: string;
    latency_ms?: number;
  };
}

export type ModelInstallState = "not_downloaded" | "installing" | "ready" | "corrupt";

export interface ModelStatus {
  id: string;
  display_name: string;
  role: "asr" | "translation" | "alignment";
  size_bytes: number;
  state: ModelInstallState;
  path: string;
  revision: string;
}

export interface ModelProgress {
  model_id: string;
  bytes_completed: number;
  total_bytes: number;
  bytes_per_second: number | null;
  phase: string;
}

export interface SessionRecord {
  id: string;
  title: string;
  status: string;
  started_at: string;
  ended_at: string | null;
  source_language: string;
  target_language: string;
  audio_source: string;
  audio_profile: string;
  inference_mode: string;
  asr_backend: string;
  translation_backend: string;
  route_reason: string;
}

export interface SegmentRecord {
  id: string;
  start_ms: number;
  end_ms: number;
  source_text: string;
  translated_text: string;
  timestamp_quality: string;
}

export interface SessionDetail {
  session: SessionRecord;
  segments: SegmentRecord[];
}

export interface UiEventEnvelope {
  schema_version: number;
  sequence: number;
  session_id: string | null;
  kind:
    | "session_state"
    | "transcript_revision"
    | "translation_revision"
    | "segment_committed"
    | "metrics"
    | "route_decision"
    | "backend_health"
    | "audio_device_change"
    | "settings_changed"
    | "error";
  emitted_at_unix_ms: number;
  payload: unknown;
}

export const emptySnapshot: SessionSnapshot = {
  state_revision: 0,
  phase: "IDLE",
  session_id: null,
  started_at: null,
  ended_at: null,
  config: null,
  route: null,
  live: {
    original_committed: "",
    original_unstable: "",
    translation_committed: "",
    translation_editable: "",
    source_revision_id: 0,
    translation_revision_id: 0,
    translation_source_revision_id: 0,
  },
  previous_segments: [],
  metrics: {
    speech_detected: false,
    reconnect_count: 0,
    buffered_audio_ms: 0,
    dropped_audio_ms: 0,
  },
  recoverable_error: null,
  startup_status: null,
};

export const defaultCaptionPreferences: CaptionPreferences = {
  display: "both",
  recent_segments: 2,
  font_size_px: 30,
  opacity: 0.92,
};

export const defaultSessionDefaults: StartSessionRequest = {
  expected_state_revision: 0,
  source_language: "en",
  target_language: "zh",
  audio_source: "microphone",
  audio_device_id: null,
  audio_profile: "lecture",
  inference_mode: "auto",
  asr_provider: "auto",
  translation_provider: "auto",
  privacy: {
    audio_upload_allowed: false,
    transcript_upload_allowed: false,
  },
};
