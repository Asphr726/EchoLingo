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
import {
  defaultCaptionPreferences,
  defaultSessionDefaults,
  emptySnapshot,
} from "../types";

const smokeBackend = import.meta.env.VITE_ECHOLINGO_SMOKE_BACKEND === "mock";

const defaultDraft: StartSessionRequest = {
  ...defaultSessionDefaults,
  asr_provider: smokeBackend ? "mock" : defaultSessionDefaults.asr_provider,
  translation_provider: smokeBackend ? "mock" : defaultSessionDefaults.translation_provider,
};

interface AppContextValue {
  snapshot: SessionSnapshot;
  draft: StartSessionRequest;
  setDraft: Dispatch<SetStateAction<StartSessionRequest>>;
  devices: AudioDevice[];
  caption: CaptionPreferences;
  loading: boolean;
  actionPending: boolean;
  onboardingComplete: boolean;
  error: string | null;
  clearError: () => void;
  completeOnboarding: () => Promise<void>;
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
  if (event.kind === "backend_health") {
    const service = String(payload.service ?? "inference backend");
    const state = String(payload.state ?? "starting");
    const label = service === "qwen_asr" ? "Qwen3-ASR" : service === "hymt" ? "Hy-MT2" : service;
    const startupStatus = state === "connected"
      ? `${label} is ready.`
      : service === "qwen_asr"
        ? "Loading Qwen3-ASR on this device. The first start can take 1–2 minutes."
        : `Loading ${label} on this device…`;
    return { ...snapshot, startup_status: startupStatus };
  }
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
      if (live.translation_source_revision_id !== revision) {
        live.translation_editable = "";
      }
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
      translation_source_revision_id: Number(
        payload.source_revision_id ?? snapshot.live.translation_source_revision_id,
      ),
      translation_committed:
        String(payload.committed_text ?? "") || snapshot.live.translation_committed,
      translation_editable: String(payload.editable_text ?? payload.text ?? ""),
    };
    return { ...snapshot, live };
  }
  if (event.kind === "segment_committed") {
    const segment = event.payload as SessionSnapshot["previous_segments"][number];
    const index = snapshot.previous_segments.findIndex(
      (current) => current.ordinal === segment.ordinal,
    );
    const previous_segments = [...snapshot.previous_segments];
    if (index >= 0) previous_segments[index] = segment;
    else previous_segments.push(segment);
    return { ...snapshot, previous_segments: previous_segments.slice(-200) };
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
  const [onboardingComplete, setOnboardingComplete] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let unlisten: () => void = () => undefined;
    Promise.all([
      api.snapshot(),
      api.onboardingStatus(),
      api.audioDevices(),
      api.captionPreferences(),
      api.sessionDefaults(),
    ])
      .then(([nextSnapshot, nextOnboarding, nextDevices, nextCaption, nextDefaults]) => {
        if (!active) return;
        setSnapshot(nextSnapshot);
        setOnboardingComplete(nextOnboarding);
        setDraft({
          ...nextDefaults,
          expected_state_revision: nextSnapshot.state_revision,
          asr_provider: smokeBackend ? "mock" : nextDefaults.asr_provider,
          translation_provider: smokeBackend ? "mock" : nextDefaults.translation_provider,
          audio_device_id:
            nextDevices.find(
              (device) => device.id === nextDefaults.audio_device_id && device.available,
            )?.id ??
            nextDevices.find((device) => device.kind === "microphone" && device.is_default)?.id ??
            null,
        });
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

  useEffect(() => {
    if (loading || smokeBackend) return;
    const timer = window.setTimeout(() => {
      void api
        .updateSessionDefaults({
          ...draft,
          expected_state_revision: snapshot.state_revision,
        })
        .catch((failure) => setError(message(failure)));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [draft, loading, snapshot.state_revision]);

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
  const completeOnboarding = useCallback(async () => {
    setActionPending(true);
    setError(null);
    try {
      setOnboardingComplete(await api.completeOnboarding());
    } catch (failure) {
      setError(message(failure));
    } finally {
      setActionPending(false);
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
      onboardingComplete,
      error,
      clearError: () => setError(null),
      completeOnboarding,
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
      onboardingComplete,
      completeOnboarding,
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
