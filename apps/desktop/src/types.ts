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
  /** Cloud recognizer the Auto route falls back to when the local model
   *  misses its latency target. A provider id from the catalog. */
  cloud_asr_preference: string;
  /** Cloud translator used by the Auto route, as above. */
  cloud_translation_preference: string;
  /** Per-lecture topic and terms, at most 2000 chars. */
  session_context: string;
  /** Standing terminology: one `term = translation` or `term` per line. */
  glossary: string;
  privacy: PrivacyPolicy;
}

export interface RouteStatus {
  asr_provider: string;
  asr_model: string | null;
  asr_display_name: string | null;
  asr_locality: string;
  asr_health: string;
  translation_provider: string;
  translation_model: string | null;
  translation_display_name: string | null;
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
  translation_queue_depth?: number;
  translation_backlog_ms?: number;
  translation_first_delta_ms?: number | null;
  translation_dropped_partials?: number;
  translation_cancelled_requests?: number;
  translation_errors?: number;
}

export type TranslationStatus = "pending" | "streaming" | "done" | "unavailable";

export interface SegmentSummary {
  id: string;
  ordinal: number;
  start_ms: number;
  end_ms: number;
  original: string;
  translation: string;
  translation_status?: TranslationStatus;
}

/** Three text tiers: committed rows (segments), open recognizer-committed
 *  text not yet closed into a row, and the revisable unstable tail. */
export interface LiveTranscript {
  original_committed: string;
  open_text: string;
  original_unstable: string;
  translation_committed: string;
  translation_editable: string;
  source_revision_id: number;
  translation_revision_id: number;
  translation_source_revision_id: number;
}

export interface SessionSnapshot {
  state_revision: number;
  phase: SessionPhase;
  session_id: string | null;
  started_at: string | null;
  ended_at: string | null;
  config: StartSessionRequest | null;
  route: RouteStatus | null;
  live: LiveTranscript;
  previous_segments: SegmentSummary[];
  metrics: LiveMetrics;
  recoverable_error: string | null;
  startup_status?: string | null;
  /** UI-only: local model services were pre-warmed and are ready. */
  models_ready?: boolean;
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

export interface RuntimePreferences {
  preload_local_models: boolean;
  /** Non-secret provider settings keyed by credential group id, then by
   *  setting key (for example `{ dashscope: { region: "beijing" } }`). */
  providers: Record<string, Record<string, string>>;
  assistant: AssistantPreferences;
}

/** AI assistant used for session notes and titles. */
export interface AssistantPreferences {
  /** Credential group id of an OpenAI-compatible chat provider, "" = off. */
  provider_group: string;
  /** Model name; "" = the preset default. */
  model: string;
  /** Explicit consent to send transcripts and attachments to the model. */
  transcript_upload_allowed: boolean;
  /** Name sessions with the assistant when they end. */
  auto_title: boolean;
}

export const defaultAssistantPreferences: AssistantPreferences = {
  provider_group: "",
  model: "",
  transcript_upload_allowed: false,
  auto_title: true,
};

/** Catalog entry for a chat provider the assistant can use. */
export interface AssistantPreset {
  group_id: string;
  display_name: string;
  default_model: string;
  models: string[];
}

export interface AssistantStatus {
  configured: boolean;
  consent: boolean;
  provider_group: string;
  model: string;
  display_name: string;
  key_available: boolean;
  auto_title: boolean;
}

export interface NoteAttachment {
  id: string;
  name: string;
  size_bytes: number;
  extension: string;
}

export interface NoteAttachmentReport {
  id?: string;
  name: string;
  pages?: number | null;
  chars?: number;
  truncated?: boolean;
  warning?: string | null;
}

export interface SessionNotes {
  session_id: string;
  markdown: string;
  language: string;
  provider: string;
  model: string;
  attachments: NoteAttachmentReport[];
  usage: { prompt_tokens?: number | null; completion_tokens?: number | null };
  prompt_version: string;
  source_chars: number;
  created_at: string;
  updated_at: string;
}

export type AssistantJobState = "running" | "completed" | "failed" | "cancelled";

export interface AssistantProgress {
  index?: number;
  total?: number;
  start_ms?: number;
  end_ms?: number;
}

/** Payload of the `assistant_update` UI event and of a running job. */
export interface AssistantJob {
  job_id: string;
  session_id: string | null;
  task: "notes" | "title" | "context" | "probe";
  state: AssistantJobState;
  stage?: string | null;
  progress?: AssistantProgress | null;
  /** Full markdown produced so far. */
  text?: string;
  result?: unknown;
  error?: { code: string; message: string } | null;
}

export interface SessionNotesState {
  notes: SessionNotes | null;
  job: AssistantJob | null;
}

export interface ContextImportResult {
  context: string;
  terms: string[];
  glossary: Array<{ source: string; target: string }>;
  warnings: string[];
}

export interface CaptionPreferences {
  display: CaptionDisplay;
  recent_segments: number;
  font_size_px: number;
  opacity: number;
}

// ---------------------------------------------------------------------------
// Provider catalog (mirrors configs/providers.json, exported by the Python
// registry which is the single source of truth for provider ids and flags).

export type ProviderKind = "asr" | "translation";
export type ProviderLocality = "local" | "cloud" | "mock";

export interface ProviderSpec {
  id: string;
  kind: ProviderKind;
  display_name: string;
  vendor: string;
  description: string;
  locality: ProviderLocality;
  /** Credential group id, or null for local and mock providers. */
  credential_group: string | null;
  local_service_id: string | null;
  /** Supported source languages; empty means no restriction. */
  languages: string[];
  audio_upload_required: boolean;
  transcript_upload_required: boolean;
  streaming_partials: boolean;
  auto_route_eligible: boolean;
  selectable: boolean;
}

export interface CredentialField {
  key: string;
  label: string;
  env_var: string;
  keychain_account: string;
  secret: boolean;
  required: boolean;
  min_len: number;
}

export interface ProviderSetting {
  key: string;
  label: string;
  env_var: string;
  kind: "select" | "text";
  default: string;
  /** `[value, label]` pairs for `select` settings; empty for `text`. */
  options: Array<[string, string]>;
  placeholder: string;
}

export interface CredentialGroup {
  id: string;
  display_name: string;
  vendor: string;
  docs_url: string;
  free_tier_note: string;
  fields: CredentialField[];
  settings: ProviderSetting[];
}

export interface ProviderCatalog {
  schema_version: number;
  asr: ProviderSpec[];
  translation: ProviderSpec[];
  credential_groups: CredentialGroup[];
  /** Chat presets for the AI assistant; missing in older catalogs. */
  assistant?: AssistantPreset[];
}

/** Result of `assistant_probe` (one fixed prompt, no transcript). */
export interface AssistantProbeResult {
  provider: string;
  model: string;
  latency_ms: number;
}

/** Payload of the `history_changed` UI event. */
export interface HistoryChange {
  session_id: string;
  reason: "title" | "notes" | "renamed" | "deleted";
}

export type CredentialSource = "keychain" | "environment" | "none";

export interface CredentialFieldStatus {
  key: string;
  available: boolean;
  source: CredentialSource;
}

/** Availability of one credential group. Values never leave the secure
 *  store; only presence and origin are reported. */
export interface CredentialGroupStatus {
  group_id: string;
  fields: CredentialFieldStatus[];
  /** Effective non-secret settings (runtime preferences over defaults). */
  settings: Record<string, string>;
}

export type CloudProbeRoleStatus =
  | "connected"
  | "failed"
  | "skipped"
  | "pending"
  | "not_tested";

export interface CloudProbeAsrResult {
  status: CloudProbeRoleStatus;
  provider?: string;
  model?: string | null;
  handshake_latency_ms?: number;
  region?: string;
  host?: string;
  workspace_scoped?: boolean;
}

export interface CloudProbeTranslationResult {
  status: CloudProbeRoleStatus;
  provider?: string;
  model?: string | null;
  latency_ms?: number;
}

/** Result of `probe_cloud`. Recognition probes complete the authenticated
 *  handshake only; `audio_uploaded` is always false. */
export interface CloudProbeResult {
  ok: boolean;
  audio_uploaded: boolean;
  asr: CloudProbeAsrResult;
  translation: CloudProbeTranslationResult;
  code?: string;
  message?: string;
  region?: string;
  host?: string;
  workspace_scoped?: boolean;
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
  /** `default` (generated), `user` (renamed) or `ai` (assistant title). */
  title_source?: "default" | "user" | "ai";
  /** The lecture context the session ran with. */
  context?: string;
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
    | "assistant_update"
    | "history_changed"
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
    open_text: "",
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
  cloud_asr_preference: "qwen_cloud",
  cloud_translation_preference: "qwen_cloud",
  session_context: "",
  glossary: "",
  privacy: {
    audio_upload_allowed: false,
    transcript_upload_allowed: false,
  },
};

// ---------------------------------------------------------------------------
// Click-to-consent. A control that needs a privacy flag stays clickable and
// asks through `requestConsent`; nothing is uploaded until it resolves true.

/** A party that would receive data, named the way the user knows it. */
export interface ConsentRecipient {
  /** Company, e.g. "Deepgram". */
  vendor: string;
  /** Provider as listed in the provider selects, e.g. "Deepgram streaming (cloud)". */
  providerLabel: string;
}

/** Session upload flags (`PrivacyPolicy`), one row per missing flag. */
export interface SessionConsentRequest {
  kind: "session";
  audio?: ConsentRecipient;
  transcript?: ConsentRecipient;
  /** What this particular action sends, when it is narrower than a session. */
  note?: string;
}

/** What the AI assistant is about to be used for. */
export type AssistantPurpose = "notes" | "title" | "import";

/** The assistant's transcript consent (`AssistantPreferences.transcript_upload_allowed`). */
export interface AssistantConsentRequest {
  kind: "assistant";
  /** Provider display name, e.g. "Qwen (Alibaba Model Studio)". */
  vendor: string;
  /** Model name; "" when the shell reports none. */
  model: string;
  purpose: AssistantPurpose;
}

/** Consent cannot help: the assistant has no provider or no key yet. */
export interface SetupMissingRequest {
  kind: "setup-missing";
  need: "provider" | "key";
  vendor?: string;
  purpose: AssistantPurpose;
}

export type ConsentRequest = SessionConsentRequest | AssistantConsentRequest | SetupMissingRequest;

/** How a consent request was answered. "declined" is the dialog's quiet
 *  button, after which an action may go on without the upload (slide import
 *  extracts on this Mac); "cancelled" is Esc, Close, the backdrop or a newer
 *  request, after which nothing goes on. */
export type ConsentAnswer = "granted" | "declined" | "cancelled";

/** The rows of a session request the user left checked. */
export interface ConsentSelection {
  audio: boolean;
  transcript: boolean;
}
