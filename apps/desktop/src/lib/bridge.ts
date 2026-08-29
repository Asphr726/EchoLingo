import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AudioDevice,
  CaptionPreferences,
  SessionDetail,
  SessionRecord,
  SessionSnapshot,
  StartSessionRequest,
  UiEventEnvelope,
} from "../types";
import { defaultCaptionPreferences, defaultSessionDefaults, emptySnapshot } from "../types";

const isTauri = () => "__TAURI_INTERNALS__" in window;

export async function command<T>(name: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    return browserFallback<T>(name, args);
  }
  return invoke<T>(name, args);
}

function browserFallback<T>(name: string, args?: Record<string, unknown>): T {
  const values: Record<string, unknown> = {
    get_app_snapshot: emptySnapshot,
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
    get_session_defaults: defaultSessionDefaults,
    history_search: [],
  };
  if (name === "update_session_defaults") {
    return args?.defaults as T;
  }
  if (name === "update_caption_preferences") {
    return args?.preferences as T;
  }
  if (name in values) return values[name] as T;
  throw new Error(`${name} requires the Tauri desktop runtime`);
}

export async function subscribeUiEvents(
  handler: (event: UiEventEnvelope) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return () => undefined;
  return listen<UiEventEnvelope>("echolingo://ui-event", ({ payload }) => handler(payload));
}

export const api = {
  snapshot: () => command<SessionSnapshot>("get_app_snapshot"),
  audioDevices: () => command<AudioDevice[]>("list_audio_devices"),
  start: (request: StartSessionRequest) =>
    command<SessionSnapshot>("start_session", { request }),
  pause: (revision: number) =>
    command<SessionSnapshot>("pause_session", { expectedStateRevision: revision }),
  resume: (revision: number) =>
    command<SessionSnapshot>("resume_session", { expectedStateRevision: revision }),
  stop: (revision: number) =>
    command<SessionSnapshot>("stop_session", { expectedStateRevision: revision }),
  captionPreferences: () => command<CaptionPreferences>("get_caption_preferences"),
  sessionDefaults: () => command<StartSessionRequest>("get_session_defaults"),
  updateSessionDefaults: (defaults: StartSessionRequest) =>
    command<StartSessionRequest>("update_session_defaults", { defaults }),
  updateCaptionPreferences: (preferences: CaptionPreferences) =>
    command<CaptionPreferences>("update_caption_preferences", { preferences }),
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
};
