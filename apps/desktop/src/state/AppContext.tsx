import {
  createContext,
  type Dispatch,
  type PropsWithChildren,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api, subscribeUiEvents } from "../lib/bridge";
import type {
  AudioDevice,
  CaptionPreferences,
  SessionSnapshot,
  StartSessionRequest,
  UiEventEnvelope,
} from "../types";
import { defaultCaptionPreferences, emptySnapshot } from "../types";

const smokeBackend = import.meta.env.VITE_ECHOLINGO_SMOKE_BACKEND === "mock";

const defaultDraft: StartSessionRequest = {
  expected_state_revision: 0,
  source_language: "en",
  target_language: "zh",
  audio_source: "microphone",
  audio_device_id: null,
  audio_profile: "lecture",
  inference_mode: "auto",
  asr_provider: smokeBackend ? "mock" : "auto",
  translation_provider: smokeBackend ? "mock" : "auto",
  privacy: {
    audio_upload_allowed: false,
    transcript_upload_allowed: false,
  },
};

interface AppContextValue {
  snapshot: SessionSnapshot;
  draft: StartSessionRequest;
  setDraft: Dispatch<SetStateAction<StartSessionRequest>>;
  devices: AudioDevice[];
  caption: CaptionPreferences;
  loading: boolean;
  actionPending: boolean;
  error: string | null;
  clearError: () => void;
  start: () => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  updateCaption: (next: CaptionPreferences) => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function applyUiEvent(snapshot: SessionSnapshot, event: UiEventEnvelope): SessionSnapshot {
  const payload = event.payload as Record<string, unknown>;
  if (event.kind === "session_state") return event.payload as SessionSnapshot;
  if (event.kind === "metrics") {
    return { ...snapshot, metrics: { ...snapshot.metrics, ...payload } };
  }
  if (event.kind === "transcript_revision") {
    const kind = String(payload.kind ?? "partial");
    const text = String(payload.text ?? "");
    const committed = String(payload.committed_text ?? "");
    const revision = Number(payload.revision_id ?? snapshot.live.source_revision_id);
    const live = { ...snapshot.live, source_revision_id: revision };
    if (kind === "partial") {
      live.original_unstable = String(payload.unstable_text ?? text);
    } else {
      live.original_committed = committed || text;
      live.original_unstable = "";
    }
    return { ...snapshot, live };
  }
  if (event.kind === "translation_revision") {
    const live = {
      ...snapshot.live,
      translation_revision_id: Number(
        payload.revision_id ?? snapshot.live.translation_revision_id,
      ),
      translation_committed:
        String(payload.committed_text ?? "") || snapshot.live.translation_committed,
      translation_editable: String(payload.editable_text ?? payload.text ?? ""),
    };
    return { ...snapshot, live };
  }
  if (event.kind === "error") {
    return {
      ...snapshot,
      recoverable_error: String(payload.message ?? "Unexpected desktop error"),
    };
  }
  return snapshot;
}

export function AppProvider({ children }: PropsWithChildren) {
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(emptySnapshot);
  const [draft, setDraft] = useState<StartSessionRequest>(defaultDraft);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [caption, setCaption] = useState<CaptionPreferences>(defaultCaptionPreferences);
  const [loading, setLoading] = useState(true);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let unlisten: () => void = () => undefined;
    Promise.all([api.snapshot(), api.audioDevices(), api.captionPreferences()])
      .then(([nextSnapshot, nextDevices, nextCaption]) => {
        if (!active) return;
        setSnapshot(nextSnapshot);
        setDraft((current) => ({
          ...current,
          expected_state_revision: nextSnapshot.state_revision,
          audio_device_id:
            current.audio_device_id ??
            nextDevices.find((device) => device.kind === "microphone" && device.is_default)?.id ??
            null,
        }));
        setDevices(nextDevices);
        setCaption(nextCaption);
      })
      .catch((failure) => active && setError(message(failure)))
      .finally(() => active && setLoading(false));
    subscribeUiEvents((event) => {
      setSnapshot((current) => applyUiEvent(current, event));
      if (event.kind === "settings_changed") {
        setCaption(event.payload as CaptionPreferences);
      }
    })
      .then((dispose) => {
        if (active) unlisten = dispose;
        else dispose();
      })
      .catch((failure) => active && setError(message(failure)));
    return () => {
      active = false;
      unlisten();
    };
  }, []);

  const run = useCallback(async (action: () => Promise<SessionSnapshot>) => {
    setActionPending(true);
    setError(null);
    try {
      setSnapshot(await action());
    } catch (failure) {
      setError(message(failure));
    } finally {
      setActionPending(false);
    }
  }, []);

  const start = useCallback(
    () =>
      run(() =>
        api.start({
          ...draft,
          expected_state_revision: snapshot.state_revision,
          audio_device_id: draft.audio_source === "microphone" ? draft.audio_device_id : null,
        }),
      ),
    [draft, run, snapshot.state_revision],
  );
  const pause = useCallback(
    () => run(() => api.pause(snapshot.state_revision)),
    [run, snapshot.state_revision],
  );
  const resume = useCallback(
    () => run(() => api.resume(snapshot.state_revision)),
    [run, snapshot.state_revision],
  );
  const stop = useCallback(
    () => run(() => api.stop(snapshot.state_revision)),
    [run, snapshot.state_revision],
  );
  const updateCaption = useCallback(async (next: CaptionPreferences) => {
    setCaption(next);
    try {
      setCaption(await api.updateCaptionPreferences(next));
    } catch (failure) {
      setError(message(failure));
    }
  }, []);

  const value = useMemo<AppContextValue>(
    () => ({
      snapshot,
      draft,
      setDraft,
      devices,
      caption,
      loading,
      actionPending,
      error,
      clearError: () => setError(null),
      start,
      pause,
      resume,
      stop,
      updateCaption,
    }),
    [
      actionPending,
      caption,
      devices,
      draft,
      error,
      loading,
      pause,
      resume,
      snapshot,
      start,
      stop,
      updateCaption,
    ],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}
