import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AudioDevice,
  AudioPermissionStatus,
  AudioTestResult,
  CaptionPreferences,
  CloudCredentialStatus,
  CloudProbeResult,
  ModelProgress,
  RuntimePreferences,
  ModelStatus,
  SessionDetail,
  SessionRecord,
  SessionSnapshot,
  StartSessionRequest,
  UiEventEnvelope,
} from "../types";
import { defaultCaptionPreferences, defaultSessionDefaults, emptySnapshot } from "../types";

const isTauri = () => "__TAURI_INTERNALS__" in window;

/** `?preview=live` in a plain browser replays a scripted bilingual session so
 *  the live layout can be designed and reviewed without model runtimes. */
const previewMode = () =>
  !isTauri() && new URLSearchParams(window.location.search).get("preview") === "live";

export async function command<T>(name: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    return browserFallback<T>(name, args);
  }
  return invoke<T>(name, args);
}

function browserFallback<T>(name: string, args?: Record<string, unknown>): T {
  const values: Record<string, unknown> = {
    get_app_snapshot: previewMode()
      ? {
          ...emptySnapshot,
          phase: "LISTENING",
          session_id: "preview-session",
          started_at: new Date().toISOString(),
          config: { ...defaultSessionDefaults },
          route: {
            asr_provider: "qwen_local",
            asr_model: "qwen3-asr-0.6b",
            asr_locality: "local",
            asr_health: "connected",
            translation_provider: "hymt_local",
            translation_model: "tencent/Hy-MT2-1.8B",
            translation_locality: "local",
            translation_health: "connected",
            deployment: "local",
            reason: "Preview replay of a recorded session",
          },
        }
      : emptySnapshot,
    onboarding_status: previewMode(),
    list_audio_devices: [
      {
        id: "preview-microphone",
        name: "Default microphone",
        kind: "microphone",
        is_default: true,
        available: true,
        requires_picker: false,
      },
      {
        id: "macos-screen-capture-kit",
        name: "System Audio…",
        kind: "system_audio",
        is_default: false,
        available: true,
        requires_picker: true,
      },
    ] satisfies AudioDevice[],
    get_caption_preferences: defaultCaptionPreferences,
    get_runtime_preferences: { preload_local_models: true } satisfies RuntimePreferences,
    get_session_defaults: defaultSessionDefaults,
    history_search: previewMode() ? [previewSessionRecord()] : [],
    history_open: previewMode()
      ? ({
          session: previewSessionRecord(),
          segments: previewScript.map(([source, target], index) => ({
            id: `preview-history-${index}`,
            start_ms: index * 6_500,
            end_ms: index * 6_500 + 6_000,
            source_text: source,
            translated_text: index === 3 ? "" : target,
            timestamp_quality: "forced",
          })),
        } satisfies SessionDetail)
      : undefined,
    credential_status: {
      api_key_available: false,
      workspace_id_available: false,
      source: "none",
    } satisfies CloudCredentialStatus,
    list_models: [
      {
        id: "qwen3-asr-0.6b",
        display_name: "Qwen3-ASR 0.6B",
        role: "asr",
        size_bytes: 1876091704,
        state: "not_downloaded",
        path: "/Applications/EchoLingo/models/qwen3-asr-0.6b",
        revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
      },
    ] satisfies ModelStatus[],
    audio_permission_status: {
      microphone: "not_determined",
      system_audio: "not_determined",
    } satisfies AudioPermissionStatus,
  };
  if (name === "update_session_defaults") {
    return args?.defaults as T;
  }
  if (name === "update_caption_preferences" || name === "update_runtime_preferences") {
    return args?.preferences as T;
  }
  if (name in values) return values[name] as T;
  throw new Error(`${name} requires the Tauri desktop runtime`);
}

export async function subscribeUiEvents(
  handler: (event: UiEventEnvelope) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return previewMode() ? startPreviewFeed(handler) : () => undefined;
  return listen<UiEventEnvelope>("echolingo://ui-event", ({ payload }) => handler(payload));
}

// Sentences recorded from a real local replay (Qwen3-ASR 0.6B + Hy-MT2 1.8B).
const previewScript: Array<[string, string]> = [
  ["Look at that line and think about what's happening in the orthogonal direction to it.", "看看那条线，思考一下在垂直方向上会发生什么。"],
  ["It's gonna wrap around the six corners of the cube, the hexagon of the cube away from white.", "它会围绕立方体的六个角，也就是从白色区域向外延伸的六边形区域。"],
  ["And you can see if you look at those, then it actually traces along that hex.", "如果你观察那些点，就会发现它们实际上沿着那个六边形路径移动。"],
  ["Okay, so starting at the lower left corner here, it's red.", "好的，那么从这里的左下角开始，是红色的。"],
  ["It transitions up through some oranges, through a yellow, through to green.", "它向上过渡为一些橙色，再经过黄色，最终变为绿色。"],
  ["If you see this color organization this way, but if you have to pick.", "如果你这样理解这种颜色排列方式，但当你必须做出选择时……"],
];

function previewSessionRecord(): SessionRecord {
  return {
    id: "preview-session",
    title: "en → zh lecture",
    status: "completed",
    started_at: new Date(Date.now() - 3_600_000).toISOString(),
    ended_at: new Date().toISOString(),
    source_language: "en",
    target_language: "zh",
    audio_source: "microphone",
    audio_profile: "lecture",
    inference_mode: "auto",
    asr_backend: "qwen_local",
    translation_backend: "hymt_local",
    route_reason: "preview",
  };
}

let previewFeedToken = 0;

function startPreviewFeed(handler: (event: UiEventEnvelope) => void): UnlistenFn {
  // React StrictMode mounts effects twice in development; only the newest
  // feed stays alive.
  const token = ++previewFeedToken;
  let sequence = 0;
  const isCancelled = () => token !== previewFeedToken;
  const emit = (kind: UiEventEnvelope["kind"], payload: unknown) =>
    handler({
      schema_version: 1,
      sequence: ++sequence,
      session_id: "preview-session",
      kind,
      emitted_at_unix_ms: Date.now(),
      payload,
    });
  const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
  void (async () => {
    let revision = 0;
    let startMs = 0;
    const metricsTimer = window.setInterval(
      () =>
        emit("metrics", {
          input_rms_dbfs: -34 + Math.random() * 8,
          vad_probability: 0.6 + Math.random() * 0.4,
          speech_detected: true,
          asr_first_partial_latency_ms: 1180,
          asr_commit_latency_ms: 4300,
          translation_latency_ms: 720,
          translation_first_delta_ms: 230,
          end_to_end_latency_ms: 5020,
          translation_queue_depth: 0,
          translation_backlog_ms: 0,
        }),
      500,
    );
    for (let round = 0; !isCancelled(); round += 1) {
      for (const [source, target] of previewScript) {
        if (isCancelled()) break;
        const words = source.split(" ");
        let open = "";
        for (let index = 1; index <= words.length && !isCancelled(); index += 1) {
          const shown = words.slice(0, index);
          const openCount = Math.max(0, index - 4);
          open = shown.slice(0, openCount).join(" ");
          revision += 1;
          emit("transcript_revision", {
            kind: "partial",
            revision_id: revision,
            text: shown.join(" "),
            stable_text: open,
            unstable_text: shown.slice(openCount).join(" "),
          });
          if (index % 3 === 0) {
            emit("translation_revision", {
              kind: "partial",
              revision_id: index,
              source_revision_id: revision,
              source_committed: false,
              text: target.slice(0, Math.round((target.length * index) / words.length)),
              editable_text: target.slice(0, Math.round((target.length * index) / words.length)),
            });
          }
          await sleep(180);
        }
        if (isCancelled()) break;
        revision += 1;
        const endMs = startMs + words.length * 420;
        emit("transcript_revision", {
          kind: "stable",
          revision_id: revision,
          event_id: `preview-${round}-${revision}`,
          text: source,
          committed_text: source,
          start_ms: startMs,
          end_ms: endMs,
        });
        emit("segment_committed", {
          id: `preview-${round}-${revision}`,
          ordinal: revision,
          start_ms: startMs,
          end_ms: endMs,
          original: source,
          translation: "",
          translation_status: "pending",
        });
        const stableRevision = revision;
        await sleep(350);
        for (let cut = 4; cut < target.length && !isCancelled(); cut += 4) {
          emit("segment_committed", {
            id: `preview-${round}-${stableRevision}`,
            ordinal: stableRevision,
            start_ms: startMs,
            end_ms: endMs,
            original: source,
            translation: target.slice(0, cut),
            translation_status: "streaming",
          });
          await sleep(60);
        }
        emit("segment_committed", {
          id: `preview-${round}-${stableRevision}`,
          ordinal: stableRevision,
          start_ms: startMs,
          end_ms: endMs,
          original: source,
          translation: target,
          translation_status: "done",
        });
        emit("translation_revision", {
          kind: "final",
          revision_id: 1,
          source_revision_id: stableRevision,
          source_committed: true,
          text: target,
          committed_text: target,
        });
        startMs = endMs + 600;
        await sleep(700);
      }
    }
    window.clearInterval(metricsTimer);
  })();
  return () => {
    if (token === previewFeedToken) previewFeedToken += 1;
  };
}

export async function subscribeModelProgress(
  handler: (event: ModelProgress) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return () => undefined;
  return listen<ModelProgress>("echolingo://model-progress", ({ payload }) => handler(payload));
}

export const api = {
  snapshot: () => command<SessionSnapshot>("get_app_snapshot"),
  onboardingStatus: () => command<boolean>("onboarding_status"),
  completeOnboarding: () => command<boolean>("complete_onboarding"),
  audioDevices: () => command<AudioDevice[]>("list_audio_devices"),
  audioPermissionStatus: () =>
    command<AudioPermissionStatus>("audio_permission_status"),
  requestAudioPermission: (kind: "microphone" | "system_audio") =>
    command<string>("request_audio_permission", { kind }),
  testAudioInput: (source: StartSessionRequest["audio_source"], deviceId: string | null) =>
    command<AudioTestResult>("test_audio_input", { source, deviceId }),
  start: (request: StartSessionRequest) =>
    command<SessionSnapshot>("start_session", { request }),
  pause: (revision: number) =>
    command<SessionSnapshot>("pause_session", { expectedStateRevision: revision }),
  resume: (revision: number) =>
    command<SessionSnapshot>("resume_session", { expectedStateRevision: revision }),
  stop: (revision: number) =>
    command<SessionSnapshot>("stop_session", { expectedStateRevision: revision }),
  captionPreferences: () => command<CaptionPreferences>("get_caption_preferences"),
  runtimePreferences: () => command<RuntimePreferences>("get_runtime_preferences"),
  updateRuntimePreferences: (preferences: RuntimePreferences) =>
    command<RuntimePreferences>("update_runtime_preferences", { preferences }),
  sessionDefaults: () => command<StartSessionRequest>("get_session_defaults"),
  updateSessionDefaults: (defaults: StartSessionRequest) =>
    command<StartSessionRequest>("update_session_defaults", { defaults }),
  updateCaptionPreferences: (preferences: CaptionPreferences) =>
    command<CaptionPreferences>("update_caption_preferences", { preferences }),
  credentialStatus: () => command<CloudCredentialStatus>("credential_status"),
  setCloudCredentials: (apiKey: string, workspaceId: string) =>
    command<CloudCredentialStatus>("set_cloud_credentials", {
      input: { api_key: apiKey, workspace_id: workspaceId },
    }),
  clearCloudCredentials: () =>
    command<CloudCredentialStatus>("clear_cloud_credentials"),
  probeQwenCloud: (includeTranslation: boolean) =>
    command<CloudProbeResult>("probe_qwen_cloud", { includeTranslation }),
  models: () => command<ModelStatus[]>("list_models"),
  installModel: (modelId: string) =>
    command<ModelStatus>("install_model", { modelId }),
  verifyModel: (modelId: string) =>
    command<ModelStatus>("verify_model", { modelId }),
  deleteModel: (modelId: string) =>
    command<ModelStatus>("delete_model", { modelId }),
  showCaption: () => command<void>("show_caption_window"),
  hideCaption: () => command<void>("hide_caption_window"),
  historySearch: (query = "") =>
    command<SessionRecord[]>("history_search", { query, limit: 100 }),
  historyOpen: (sessionId: string) =>
    command<SessionDetail>("history_open", { sessionId }),
  historyRename: (sessionId: string, title: string) =>
    command<void>("history_rename", { sessionId, title }),
  historyDelete: (sessionId: string) =>
    command<void>("history_delete", { sessionId }),
  historyExport: (sessionId: string, format: string) =>
    command<string>("history_export", { sessionId, format }),
  historyExportToPath: (sessionId: string, format: string, path: string) =>
    command<void>("history_export_to_path", { sessionId, format, path }),
};
