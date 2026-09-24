import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AssistantJob,
  AssistantProbeResult,
  AssistantStatus,
  AudioDevice,
  AudioPermissionStatus,
  AudioTestResult,
  CaptionPreferences,
  CloudProbeResult,
  ConsentRequest,
  ContextImportResult,
  CredentialGroupStatus,
  GpuAccelerationStatus,
  ModelProgress,
  NoteAttachment,
  ProviderCatalog,
  RuntimePreferences,
  SegmentSummary,
  ModelStatus,
  SessionDetail,
  SessionNotes,
  SessionNotesState,
  SessionRecord,
  SessionSnapshot,
  StartSessionRequest,
  UiEventEnvelope,
} from "../types";
import { defaultAssistantPreferences, defaultCaptionPreferences, defaultSessionDefaults, emptySnapshot } from "../types";
import { platform } from "./platform";
import { PreviewAssistant, previewAttachments, previewContextImport, previewNotesMode } from "./previewNotes";
// The committed catalog doubles as the browser-preview fixture. In the
// desktop runtime the shell serves the same document via `list_providers`.
import providerCatalogJson from "../../../../configs/providers.json";

const previewCatalog = providerCatalogJson as unknown as ProviderCatalog;

const isTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

/** `?preview=live` in a plain browser replays a scripted bilingual session so
 *  the live layout can be designed and reviewed without model runtimes. */
const previewMode = () =>
  typeof window !== "undefined" &&
  !isTauri() &&
  new URLSearchParams(window.location.search).get("preview") === "live";

/** `&phase=idle` keeps the preview before Start (session setup, lecture
 *  context) instead of replaying a running session. */
const previewIdle = () =>
  previewMode() && new URLSearchParams(window.location.search).get("phase") === "idle";

/** A query parameter of the browser preview; null outside it. */
const previewParam = (key: string) =>
  previewMode() ? new URLSearchParams(window.location.search).get(key) : null;

/** `&backlog=N` starts the live preview with N committed rows (at most the
 *  200-row window), so scrolling back through a long lecture can be tried at once. */
function previewBacklog(): SegmentSummary[] {
  const count = Math.min(200, Math.max(0, Math.trunc(Number(previewParam("backlog") ?? 0)) || 0));
  return Array.from({ length: count }, (_, index) => {
    const [original, translation] = previewScript[index % previewScript.length];
    return {
      id: `preview-backlog-${index}`,
      ordinal: index - count,
      start_ms: index * 4200,
      end_ms: index * 4200 + 3800,
      original,
      translation,
      translation_status: "done",
    };
  });
}

/** `&route=cloud` starts the preview on a cloud route (Deepgram + DeepL)
 *  with both uploads still off, to review the consent gate on Start. */
const previewCloudRoute = () => previewParam("route") === "cloud";

/** `&assistant=consent`: the assistant is set up but not yet allowed to
 *  receive transcripts. Granting it in the preview sticks until reload. */
let previewAssistantConsent: boolean | null = null;
const previewAssistantConsentValue = (fallback: boolean) => {
  if (previewAssistantConsent === null && previewParam("assistant") === "consent") previewAssistantConsent = false;
  return previewAssistantConsent ?? fallback;
};

/** `&store=unavailable`: the OS secure store cannot be reached, as when no
 *  Secret Service provider runs on Linux. */
const previewStoreUnavailable = () => previewParam("store") === "unavailable";

/** Devices, permissions and folders of the platform the preview shows
 *  (`&platform=windows|linux`, otherwise the browser's own). */
function previewPlatformFixtures() {
  const microphone: AudioDevice = {
    id: "preview-microphone",
    name: "Default microphone",
    kind: "microphone",
    is_default: true,
    available: true,
    requires_picker: false,
  };
  switch (platform()) {
    case "windows":
      return {
        devices: [
          microphone,
          { id: "windows-wasapi-loopback", name: "System audio (default output)", kind: "system_audio", is_default: false, available: true, requires_picker: false },
        ] satisfies AudioDevice[],
        permissions: { microphone: "granted", system_audio: "granted" } satisfies AudioPermissionStatus,
        dataDirectory: "C:\\Users\\you\\AppData\\Local\\app.echolingo.desktop",
        separator: "\\",
      };
    case "linux":
      return {
        devices: [
          microphone,
          { id: "system-audio-unavailable", name: "System audio", kind: "system_audio", is_default: false, available: false, requires_picker: false },
        ] satisfies AudioDevice[],
        permissions: { microphone: "granted", system_audio: "unavailable" } satisfies AudioPermissionStatus,
        dataDirectory: "~/.local/share/app.echolingo.desktop",
        separator: "/",
      };
    default:
      return {
        devices: [
          microphone,
          { id: "macos-screen-capture-kit", name: "System Audio…", kind: "system_audio", is_default: false, available: true, requires_picker: true },
        ] satisfies AudioDevice[],
        permissions: { microphone: "not_determined", system_audio: "not_determined" } satisfies AudioPermissionStatus,
        dataDirectory: "~/Library/Application Support/app.echolingo.desktop",
        separator: "/",
      };
  }
}

// GPU acceleration pack of the preview. `&gpu=eligible|ineligible|none|
// installed|update|fallback` shows the Windows and Linux card; without it the
// platform has none, as on macOS.
const previewRtx = { name: "NVIDIA GeForce RTX 3060 Laptop GPU", compute_capability: "8.6", driver_version: "581.15" };
const previewGpuUnsupported: GpuAccelerationStatus = {
  supported_platform: false,
  gpu: null,
  eligible: false,
  ineligible_reason: null,
  pack_state: "not_installed",
  pack_version: null,
  download_bytes: null,
  installed_bytes: null,
  cuda_available: null,
  device_name: null,
  enabled: true,
  active: false,
  fallback_reason: null,
};
const previewGpuEligible: GpuAccelerationStatus = {
  ...previewGpuUnsupported,
  supported_platform: true,
  gpu: previewRtx,
  eligible: true,
  download_bytes: 2_791_728_742,
};
const previewGpuInstalled: GpuAccelerationStatus = {
  ...previewGpuEligible,
  pack_state: "ready",
  pack_version: "0.2.0",
  installed_bytes: 5_798_205_849,
  cuda_available: true,
  device_name: previewRtx.name,
  active: true,
};
let previewGpuState: GpuAccelerationStatus | null = null;

function previewGpu(): GpuAccelerationStatus {
  if (previewGpuState) return previewGpuState;
  switch (previewParam("gpu")) {
    case "eligible":
      return (previewGpuState = previewGpuEligible);
    case "ineligible":
      return (previewGpuState = {
        ...previewGpuUnsupported,
        supported_platform: true,
        gpu: { name: "NVIDIA GeForce GTX 1060 6GB", compute_capability: "6.1", driver_version: "581.15" },
        ineligible_reason: "GeForce GTX 1060 6GB has compute capability 6.1; the pack needs 7.5 (GeForce RTX 20 or GTX 16 series) or newer.",
      });
    case "none":
      return (previewGpuState = { ...previewGpuUnsupported, supported_platform: true, ineligible_reason: "No NVIDIA GPU detected." });
    case "installed":
      return (previewGpuState = previewGpuInstalled);
    case "update":
      return (previewGpuState = { ...previewGpuEligible, pack_state: "update_required", pack_version: "0.1.9", installed_bytes: 5_798_205_849 });
    case "fallback":
      return (previewGpuState = {
        ...previewGpuInstalled,
        active: false,
        fallback_reason: "Qwen3-ASR did not start on the GPU (CUDA error: out of memory).",
      });
    default:
      return previewGpuUnsupported;
  }
}

/** Walks the pack download through its phases on the model progress channel. */
async function previewInstallGpuPack(): Promise<GpuAccelerationStatus> {
  const total = previewGpu().download_bytes ?? 2_791_728_742;
  const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
  const emit = (phase: string, bytes: number) =>
    emitPreviewProgress({ model_id: "gpu-pack", bytes_completed: bytes, total_bytes: total, bytes_per_second: 48_000_000, phase });
  previewGpuState = { ...previewGpu(), pack_state: "installing" };
  emit("starting", 0);
  for (let step = 1; step <= 20; step += 1) {
    await sleep(160);
    emit("downloading", Math.round((total * step) / 20));
  }
  for (const phase of ["verifying", "extracting", "testing"]) {
    await sleep(700);
    emit(phase, total);
  }
  previewGpuState = { ...previewGpuInstalled, enabled: previewGpu().enabled, active: false };
  emit("ready", total);
  return previewGpuState;
}

/** `?consent=session|assistant|setup` opens the consent dialog on load. */
export function previewConsentRequest(): ConsentRequest | null {
  switch (previewParam("consent")) {
    case "session":
      return {
        kind: "session",
        audio: { vendor: "Deepgram", providerLabel: "Deepgram streaming (cloud)" },
        transcript: { vendor: "DeepL", providerLabel: "DeepL (cloud)" },
      };
    case "assistant":
      return { kind: "assistant", vendor: "Qwen (Alibaba Model Studio)", model: "qwen-plus", purpose: "notes" };
    case "setup":
      return { kind: "setup-missing", need: "key", vendor: "Qwen (Alibaba Model Studio)", purpose: "notes" };
    default:
      return null;
  }
}

/** `&history=long` pads History with older sessions to review scrolling. */
function previewLongHistory(query: string): SessionRecord[] {
  if (previewParam("history") !== "long") return [];
  const topics = [
    "Fourier transforms and sampling", "Image pyramids and blending", "Edge detection", "Feature matching with SIFT",
    "RANSAC and homographies", "Panorama stitching", "Camera models", "Stereo and epipolar geometry",
    "Structure from motion", "Optical flow", "Neural radiance fields", "Diffusion models for images",
    "Texture synthesis", "Seam carving", "Morphing faces", "Light fields",
    "Colour constancy", "HDR imaging", "Deblurring", "Self-supervised learning",
  ];
  const needle = query.trim().toLowerCase();
  return topics
    .map((topic, index): SessionRecord => ({
      id: `preview-long-${index}`,
      title: `CS180 lecture ${topics.length - index}: ${topic}`,
      title_source: index % 3 === 0 ? "ai" : "user",
      context: "",
      status: "completed",
      started_at: new Date(Date.now() - (4 + index) * 86_400_000).toISOString(),
      ended_at: new Date(Date.now() - (4 + index) * 86_400_000 + 4_800_000).toISOString(),
      source_language: "en",
      target_language: "zh",
      audio_source: "microphone",
      audio_profile: "lecture",
      inference_mode: "auto",
      asr_backend: "qwen_local",
      translation_backend: "hymt_local",
      route_reason: "preview",
    }))
    .filter((session) => !needle || session.title.toLowerCase().includes(needle));
}

/** Files the attachment sheet starts with in `?notes=sheet` previews. */
export function previewInitialAttachments(): NoteAttachment[] {
  return previewMode() && previewNotesMode() === "sheet" ? previewAttachments.slice(0, 2) : [];
}

/** `?notes=…` previews open History on the AI notes tab. */
export function previewOpensNotesTab(): boolean {
  return previewMode() && new URLSearchParams(window.location.search).has("notes");
}

/** True when the preview asks the AI notes panel to open its attachment sheet. */
export function previewOpensAttachmentSheet(): boolean {
  return previewMode() && previewNotesMode() === "sheet";
}

let previewAssistantInstance: PreviewAssistant | null = null;
const previewAssistant = () =>
  (previewAssistantInstance ??= new PreviewAssistant((kind, payload, sessionId) => emitPreview(kind, payload, sessionId)));

export async function command<T>(name: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    return browserFallback<T>(name, args);
  }
  return invoke<T>(name, args);
}

function browserFallback<T>(name: string, args?: Record<string, unknown>): T {
  const fixtures = previewPlatformFixtures();
  const values: Record<string, unknown> = {
    get_app_snapshot: previewMode() && !previewIdle()
      ? {
          ...emptySnapshot,
          phase: "LISTENING",
          session_id: "preview-session",
          previous_segments: previewBacklog(),
          started_at: new Date().toISOString(),
          config: { ...defaultSessionDefaults },
          route: {
            asr_provider: "qwen_local",
            asr_model: "qwen3-asr-0.6b",
            asr_display_name: "Qwen3-ASR (local)",
            asr_locality: "local",
            asr_health: "connected",
            translation_provider: "hymt_local",
            translation_model: "tencent/Hy-MT2-1.8B",
            translation_display_name: "Hy-MT2 (local)",
            translation_locality: "local",
            translation_health: "connected",
            deployment: "local",
            reason: "qwen3-asr-0.6b passed local ASR calibration; tencent/Hy-MT2-1.8B passed local translation calibration",
          },
        }
      : emptySnapshot,
    onboarding_status: previewMode(),
    list_audio_devices: fixtures.devices,
    get_caption_preferences: defaultCaptionPreferences,
    get_runtime_preferences: {
      preload_local_models: true,
      providers: {},
      assistant: previewMode() && previewNotesMode() !== "unconfigured"
        ? { provider_group: "dashscope", model: "", transcript_upload_allowed: previewAssistantConsentValue(true), auto_title: true }
        : defaultAssistantPreferences,
    } satisfies RuntimePreferences,
    get_session_defaults: previewMode()
      ? {
          ...defaultSessionDefaults,
          session_context: "CS180 Lecture 12: colour spaces and texture perception\nBéla Julesz · Anne Treisman\nsaccade = 眼跳\npre-attentive vision = 前注意视觉",
          glossary: "# Course-wide terms\nconvolution = 卷积\nGaussian pyramid = 高斯金字塔\nAlyosha Efros",
          ...(previewCloudRoute() ? { asr_provider: "deepgram", translation_provider: "deepl" } : {}),
        }
      : defaultSessionDefaults,
    list_providers: previewCatalog,
    credential_status: previewCatalog.credential_groups.map((group) =>
      previewGroupStatus(group.id, previewMode() && group.id === "dashscope" ? "keychain" : "none"),
    ) satisfies CredentialGroupStatus[],
    logs_directory: [fixtures.dataDirectory, "logs"].join(fixtures.separator),
    list_models: [
      {
        id: "qwen3-asr-0.6b",
        display_name: "Qwen3-ASR 0.6B",
        role: "asr",
        size_bytes: 1876091704,
        state: "not_downloaded",
        path: [fixtures.dataDirectory, "models", "qwen3-asr-0.6b"].join(fixtures.separator),
        revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
      },
    ] satisfies ModelStatus[],
    audio_permission_status: fixtures.permissions,
    gpu_acceleration_status: previewGpu(),
  };
  if (name === "update_session_defaults") {
    return args?.defaults as T;
  }
  if (name === "update_runtime_preferences" && previewMode()) {
    const next = args?.preferences as RuntimePreferences | undefined;
    if (next?.assistant) previewAssistantConsent = next.assistant.transcript_upload_allowed;
  }
  if (name === "update_caption_preferences" || name === "update_runtime_preferences") {
    return args?.preferences as T;
  }
  if (name === "set_credentials") {
    const typed = (args?.fields ?? {}) as Record<string, string>;
    return previewGroupStatus(String(args?.groupId), "keychain", Object.keys(typed)) as T;
  }
  if (name === "clear_credentials") {
    return previewGroupStatus(String(args?.groupId), "none") as T;
  }
  if (name === "update_provider_settings") {
    return {
      preload_local_models: true,
      providers: { [String(args?.groupId)]: (args?.settings ?? {}) as Record<string, string> },
      assistant: defaultAssistantPreferences,
    } satisfies RuntimePreferences as T;
  }
  if (previewMode()) {
    const assistant = previewAssistant();
    switch (name) {
      case "history_search": {
        const query = String(args?.query ?? "");
        return [...assistant.search(query), ...previewLongHistory(query)] as T;
      }
      case "history_open": {
        const sessionId = String(args?.sessionId);
        const padded = previewLongHistory("").find((session) => session.id === sessionId);
        const detail = assistant.open(sessionId);
        return (padded ? { ...detail, session: padded } : detail) as T;
      }
      case "history_rename":
      case "history_delete":
        return undefined as T;
      case "assistant_status": {
        const status = assistant.status();
        return { ...status, consent: status.configured && previewAssistantConsentValue(status.consent) } as T;
      }
      case "assistant_probe":
        return assistant.probe() as T;
      case "get_session_notes":
        return assistant.sessionNotes(String(args?.sessionId)) as T;
      case "pick_note_attachments":
        return assistant.pick() as T;
      case "create_session_notes":
        return assistant.create(String(args?.sessionId)) as T;
      case "cancel_session_notes":
        assistant.cancel(String(args?.jobId));
        return undefined as T;
      case "generate_session_title":
        return assistant.title(String(args?.sessionId)) as T;
      case "import_context_files":
        return previewContextImport as T;
      case "install_gpu_pack":
        return previewInstallGpuPack() as T;
      case "remove_gpu_pack":
        previewGpuState = { ...previewGpuEligible, enabled: previewGpu().enabled };
        return previewGpuState as T;
      case "set_gpu_acceleration": {
        const enabled = Boolean(args?.enabled);
        const current = previewGpu();
        previewGpuState = { ...current, enabled, active: enabled && current.pack_state === "ready" && !current.fallback_reason };
        return previewGpuState as T;
      }
    }
  }
  if (name === "history_search") return [] as T;
  if (name === "probe_cloud" && previewMode()) {
    return previewProbe(
      (args?.asrProvider as string | null | undefined) ?? null,
      (args?.translationProvider as string | null | undefined) ?? null,
    ) as T;
  }
  if (name in values) return values[name] as T;
  throw new Error(`${name} requires the Tauri desktop runtime`);
}

/** Credential availability for the browser preview. `only` limits the
 *  available fields (a save of a subset); otherwise every field of the
 *  group follows `source`. Settings report the catalog defaults. */
function previewGroupStatus(
  groupId: string,
  source: CredentialGroupStatus["fields"][number]["source"],
  only?: string[],
): CredentialGroupStatus {
  const group = previewCatalog.credential_groups.find((entry) => entry.id === groupId);
  const storeUnavailable = previewStoreUnavailable();
  const effectiveSource = storeUnavailable && source === "keychain" ? "none" : source;
  return {
    group_id: groupId,
    fields: (group?.fields ?? []).map((field) => {
      const available = effectiveSource !== "none" && (only === undefined || only.includes(field.key));
      return { key: field.key, available, source: available ? effectiveSource : "none" };
    }),
    settings: Object.fromEntries((group?.settings ?? []).map((setting) => [setting.key, setting.default])),
    store_available: !storeUnavailable,
    store_error: storeUnavailable ? "The name org.freedesktop.secrets was not provided by any .service files" : null,
  };
}

/** A successful probe for the preview; no network is touched. */
function previewProbe(asrProvider: string | null, translationProvider: string | null): CloudProbeResult {
  const dashscope = asrProvider === "qwen_cloud" || translationProvider === "qwen_cloud";
  return {
    ok: true,
    audio_uploaded: false,
    asr: asrProvider
      ? {
          status: "connected",
          provider: asrProvider,
          model: asrProvider === "qwen_cloud" ? "qwen3-asr-flash-realtime" : null,
          handshake_latency_ms: 412,
          ...(asrProvider === "qwen_cloud"
            ? { region: "singapore", host: "dashscope-intl.aliyuncs.com", workspace_scoped: true }
            : {}),
        }
      : { status: "skipped" },
    translation: translationProvider
      ? {
          status: "connected",
          provider: translationProvider,
          model: translationProvider === "qwen_cloud" ? "qwen-mt-turbo" : null,
          latency_ms: 288,
        }
      : { status: "skipped" },
    ...(dashscope
      ? { region: "singapore", host: "dashscope-intl.aliyuncs.com", workspace_scoped: true }
      : {}),
  };
}

export async function subscribeUiEvents(
  handler: (event: UiEventEnvelope) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return previewMode() ? subscribePreview(handler) : () => undefined;
  return listen<UiEventEnvelope>("echolingo://ui-event", ({ payload }) => handler(payload));
}

// Preview events fan out to every subscriber (the app shell and History),
// like the shell's event channel does. The scripted live feed runs while at
// least one subscriber is listening.
const previewHandlers = new Set<(event: UiEventEnvelope) => void>();
let previewSequence = 0;

function emitPreview(kind: UiEventEnvelope["kind"], payload: unknown, sessionId: string | null) {
  const event: UiEventEnvelope = {
    schema_version: 1,
    sequence: ++previewSequence,
    session_id: sessionId,
    kind,
    emitted_at_unix_ms: Date.now(),
    payload,
  };
  for (const handler of [...previewHandlers]) handler(event);
}

function subscribePreview(handler: (event: UiEventEnvelope) => void): UnlistenFn {
  previewHandlers.add(handler);
  if (previewHandlers.size === 1 && !previewIdle()) startPreviewFeed();
  return () => {
    previewHandlers.delete(handler);
    if (previewHandlers.size === 0) previewFeedToken += 1;
  };
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

let previewFeedToken = 0;

function startPreviewFeed(): void {
  // React StrictMode mounts effects twice in development; only the newest
  // feed stays alive.
  const token = ++previewFeedToken;
  const isCancelled = () => token !== previewFeedToken;
  const emit = (kind: UiEventEnvelope["kind"], payload: unknown) => emitPreview(kind, payload, "preview-session");
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
}

const previewProgressHandlers = new Set<(event: ModelProgress) => void>();

function emitPreviewProgress(progress: ModelProgress) {
  for (const handler of [...previewProgressHandlers]) handler(progress);
}

export async function subscribeModelProgress(
  handler: (event: ModelProgress) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) {
    if (!previewMode()) return () => undefined;
    previewProgressHandlers.add(handler);
    return () => void previewProgressHandlers.delete(handler);
  }
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
  /** Provider catalog served by the shell (`configs/providers.json`). */
  listProviders: () => command<ProviderCatalog>("list_providers"),
  /** Availability of every credential group; values never leave the secure store. */
  credentialStatus: () => command<CredentialGroupStatus[]>("credential_status"),
  /** Store only the fields the user typed; untouched fields keep their value. */
  setCredentials: (groupId: string, fields: Record<string, string>) =>
    command<CredentialGroupStatus>("set_credentials", { groupId, fields }),
  clearCredentials: (groupId: string) =>
    command<CredentialGroupStatus>("clear_credentials", { groupId }),
  updateProviderSettings: (groupId: string, settings: Record<string, string>) =>
    command<RuntimePreferences>("update_provider_settings", { groupId, settings }),
  /** Recognition probes complete the handshake only and upload no audio;
   *  a translation probe sends one fixed sentence, so callers pass it only
   *  when transcript upload is allowed. */
  probeCloud: (asrProvider: string | null, translationProvider: string | null) =>
    command<CloudProbeResult>("probe_cloud", { asrProvider, translationProvider }),
  logsDirectory: () => command<string>("logs_directory"),
  models: () => command<ModelStatus[]>("list_models"),
  installModel: (modelId: string) =>
    command<ModelStatus>("install_model", { modelId }),
  verifyModel: (modelId: string) =>
    command<ModelStatus>("verify_model", { modelId }),
  deleteModel: (modelId: string) =>
    command<ModelStatus>("delete_model", { modelId }),

  // NVIDIA GPU acceleration pack (Windows and Linux). Everything but the
  // status is refused while a session runs.
  gpuAccelerationStatus: () => command<GpuAccelerationStatus>("gpu_acceleration_status"),
  /** Downloads, verifies, unpacks and self-tests the pack; progress arrives
   *  on the model progress channel as `model_id: "gpu-pack"`. */
  installGpuPack: () => command<GpuAccelerationStatus>("install_gpu_pack"),
  removeGpuPack: () => command<GpuAccelerationStatus>("remove_gpu_pack"),
  setGpuAcceleration: (enabled: boolean) =>
    command<GpuAccelerationStatus>("set_gpu_acceleration", { enabled }),
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

  // AI assistant. Transcripts and attachments leave the
  // device only through these commands, gated in Rust and Python by the
  // assistant's consent flag.
  assistantStatus: () => command<AssistantStatus>("assistant_status"),
  /** Sends one fixed prompt, never a transcript. */
  assistantProbe: () => command<AssistantProbeResult>("assistant_probe"),
  /** Saved notes plus any running job (with its text so far). */
  sessionNotes: async (sessionId: string) =>
    normalizeNotesState(await command<unknown>("get_session_notes", { sessionId })),
  /** Native picker in the shell; paths never reach the webview. */
  pickNoteAttachments: () => command<NoteAttachment[]>("pick_note_attachments"),
  createSessionNotes: (sessionId: string, attachmentIds: string[]) =>
    command<{ job_id: string }>("create_session_notes", { sessionId, attachmentIds }),
  cancelSessionNotes: (jobId: string) => command<void>("cancel_session_notes", { jobId }),
  generateSessionTitle: (sessionId: string) =>
    command<{ title: string; applied?: boolean }>("generate_session_title", { sessionId }),
  /** Opens the native picker itself; `useLlm` only when the assistant is
   *  configured and consented. */
  importContextFiles: (useLlm: boolean) =>
    command<ContextImportResult>("import_context_files", { useLlm }),
};

/** `get_session_notes` returns `{notes, job}`; tolerate a bare
 *  `SessionNotes | null` from an older shell. */
export function normalizeNotesState(value: unknown): SessionNotesState {
  if (!value || typeof value !== "object") return { notes: null, job: null };
  const record = value as Record<string, unknown>;
  if ("markdown" in record) return { notes: record as unknown as SessionNotes, job: null };
  return {
    notes: (record.notes as SessionNotes | null | undefined) ?? null,
    job: (record.job as AssistantJob | null | undefined) ?? null,
  };
}
