import {
  createContext,
  type Dispatch,
  type PropsWithChildren,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { restoreAudioSelection } from "../lib/audio";
import { api, subscribeUiEvents } from "../lib/bridge";
import { requestNavigation } from "../lib/navigation";
import { announceAssistantChanged, setupNeed } from "../lib/notes";
import {
  effectiveProviderId,
  needsAudioUpload,
  needsTranscriptUpload,
  providerLabel,
  providerVendor,
} from "../lib/providers";
import type {
  AssistantPurpose,
  AssistantStatus,
  AudioDevice,
  CaptionPreferences,
  ConsentAnswer,
  ConsentRecipient,
  ConsentRequest,
  ConsentSelection,
  LiveMetrics,
  PrivacyPolicy,
  ProviderCatalog,
  ProviderKind,
  SessionConsentRequest,
  SessionSnapshot,
  StartSessionRequest,
  UiEventEnvelope,
} from "../types";
import {
  defaultAssistantPreferences,
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
  /** Provider catalog from `list_providers`; null until it has loaded. */
  catalog: ProviderCatalog | null;
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
  /** Opens the consent dialog; resolves true once the user granted
   *  everything the request needs (and it has been saved). */
  requestConsent: (request: ConsentRequest) => Promise<boolean>;
  /** Like `requestConsent`, but tells a decline ("Not now") from a cancel,
   *  for actions that can go on without the upload. */
  askConsent: (request: ConsentRequest) => Promise<ConsentAnswer>;
}

const AppContext = createContext<AppContextValue | null>(null);

interface ConsentContextValue {
  /** The request the dialog shows; null when it is closed. */
  request: ConsentRequest | null;
  requestConsent: (request: ConsentRequest) => Promise<boolean>;
  askConsent: (request: ConsentRequest) => Promise<ConsentAnswer>;
  /** Grants what the user selected (throws when saving fails). */
  confirm: (selection: ConsentSelection) => Promise<void>;
  /** Closes without granting: the quiet button declines, anything else
   *  (Esc, Close, the backdrop) cancels. */
  dismiss: (answer: Exclude<ConsentAnswer, "granted">) => void;
}

// Outside the provider (server-rendered tests) nothing can be granted.
const ConsentContext = createContext<ConsentContextValue>({
  request: null,
  requestConsent: async () => false,
  askConsent: async () => "cancelled",
  confirm: async () => undefined,
  dismiss: () => undefined,
});

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Events from another session (e.g. alignment of a finished session) never
 *  touch the current one. Snapshot replacement is always accepted. */
function belongsToCurrentSession(snapshot: SessionSnapshot, event: UiEventEnvelope): boolean {
  if (event.kind === "session_state" || !event.session_id || !snapshot.session_id) return true;
  return event.session_id === snapshot.session_id;
}

export function applyUiEvent(snapshot: SessionSnapshot, event: UiEventEnvelope): SessionSnapshot {
  const payload = event.payload as Record<string, unknown>;
  if (event.kind === "session_state") return event.payload as SessionSnapshot;
  if (!belongsToCurrentSession(snapshot, event)) return snapshot;
  if (event.kind === "backend_health") {
    const service = String(payload.service ?? "inference backend");
    const state = String(payload.state ?? "starting");
    if (service === "warmup") {
      // Background pre-warm of the sidecar and local models at launch.
      if (state === "starting") {
        return { ...snapshot, models_ready: false, startup_status: "Preparing local models in the background…" };
      }
      if (state === "connected") {
        return { ...snapshot, models_ready: true, startup_status: "Local models are ready. Start is immediate." };
      }
      return {
        ...snapshot,
        models_ready: false,
        startup_status: `Local models will load when you press Start (${String(payload.message ?? "preload unavailable")}).`,
      };
    }
    const label = service === "qwen_asr" ? "Qwen3-ASR" : service === "hymt" ? "Hy-MT2" : service;
    const startupStatus = state === "connected"
      ? `${label} is ready.`
      : service === "qwen_asr"
        ? "Loading Qwen3-ASR on this device. The first load can take 1–2 minutes."
        : `Loading ${label} on this device…`;
    return { ...snapshot, startup_status: startupStatus };
  }
  if (event.kind === "metrics") {
    return { ...snapshot, metrics: { ...snapshot.metrics, ...payload } };
  }
  if (event.kind === "transcript_revision") {
    const kind = String(payload.kind ?? "partial");
    // Alignment refines stored timings only; errors are diagnostics.
    if (kind === "alignment_update" || kind === "error") return snapshot;
    const text = String(payload.text ?? "");
    const committed = String(payload.committed_text ?? "");
    const revision = Number(payload.revision_id ?? snapshot.live.source_revision_id);
    const live = { ...snapshot.live, source_revision_id: revision };
    if (kind === "partial") {
      live.original_unstable = String(payload.unstable_text ?? text);
      live.open_text = String(payload.stable_text ?? "");
    } else {
      if (committed) live.original_committed = committed;
      live.open_text = "";
      live.original_unstable = "";
      live.translation_editable = "";
    }
    return { ...snapshot, live };
  }
  if (event.kind === "translation_revision") {
    const kind = String(payload.kind ?? "partial");
    const sourceCommitted = Boolean(payload.source_committed);
    const sourceRevision = Number(
      payload.source_revision_id ?? snapshot.live.translation_source_revision_id,
    );
    const text = String(payload.text ?? "");
    const editable = String(payload.editable_text ?? "") || text;
    const live = {
      ...snapshot.live,
      translation_revision_id: Number(
        payload.revision_id ?? snapshot.live.translation_revision_id,
      ),
      translation_source_revision_id: sourceRevision,
    };
    if (sourceCommitted) {
      // Row translations arrive through segment_committed; the live tail is
      // never touched by a committed span.
      if (kind === "final" && payload.committed_text) {
        live.translation_committed = String(payload.committed_text);
      }
      return { ...snapshot, live };
    }
    if (kind === "error") return snapshot;
    const hasLiveSource = Boolean(snapshot.live.original_unstable || snapshot.live.open_text);
    live.translation_editable = hasLiveSource ? editable : "";
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

/** Session defaults persisted by an older shell may predate the cloud
 *  preference fields; fill them from the built-in defaults so the provider
 *  selects stay controlled. */
export function withProviderDefaults(defaults: Partial<StartSessionRequest>): StartSessionRequest {
  return {
    ...defaultSessionDefaults,
    ...defaults,
    cloud_asr_preference: defaults.cloud_asr_preference || defaultSessionDefaults.cloud_asr_preference,
    cloud_translation_preference:
      defaults.cloud_translation_preference || defaultSessionDefaults.cloud_translation_preference,
    session_context: defaults.session_context ?? "",
    glossary: defaults.glossary ?? "",
  };
}

/** Keep live text that arrived between a command's emit and its reply. */
export function mergeCommandSnapshot(current: SessionSnapshot, next: SessionSnapshot): SessionSnapshot {
  if (
    current.session_id === next.session_id &&
    current.state_revision >= next.state_revision &&
    current.phase === next.phase
  ) {
    return current;
  }
  return next;
}

// ---------------------------------------------------------------------------
// Consent requests, kept pure so every case is unit tested.

function recipient(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  id: string,
): ConsentRecipient {
  return { vendor: providerVendor(catalog, kind, id), providerLabel: providerLabel(catalog, kind, id) };
}

/** What Start needs the user to allow for this draft: one row per upload
 *  the effective providers need and the privacy flags do not yet allow.
 *  Null when the session can start as configured. */
export function sessionConsentRequest(
  catalog: ProviderCatalog | null | undefined,
  draft: StartSessionRequest,
): SessionConsentRequest | null {
  const asrId = effectiveProviderId(draft, "asr");
  const translationId = effectiveProviderId(draft, "translation");
  const audio =
    asrId && needsAudioUpload(catalog, draft) && !draft.privacy.audio_upload_allowed
      ? recipient(catalog, "asr", asrId)
      : undefined;
  const transcript =
    translationId && needsTranscriptUpload(catalog, draft) && !draft.privacy.transcript_upload_allowed
      ? recipient(catalog, "translation", translationId)
      : undefined;
  if (!audio && !transcript) return null;
  return { kind: "session", ...(audio ? { audio } : {}), ...(transcript ? { transcript } : {}) };
}

/** "uploads audio to Deepgram and sends transcript text to DeepL". */
export function sessionConsentSummary(request: SessionConsentRequest): string {
  const parts = [
    request.audio ? `uploads audio to ${request.audio.vendor}` : null,
    request.transcript ? `sends transcript text to ${request.transcript.vendor}` : null,
  ].filter(Boolean);
  return parts.join(" and ");
}

/** What using the assistant for `purpose` needs first: its consent, or the
 *  setup consent cannot replace. Null when it is ready, or while the status
 *  is still loading (the shell enforces the gate either way). */
export function assistantConsentRequest(
  status: AssistantStatus | null,
  purpose: AssistantPurpose,
): ConsentRequest | null {
  const need = setupNeed(status);
  if (!status || !need) return null;
  const vendor = status.display_name || undefined;
  if (need === "consent") {
    return { kind: "assistant", vendor: vendor ?? "the AI assistant", model: status.model, purpose };
  }
  return { kind: "setup-missing", need, ...(vendor ? { vendor } : {}), purpose };
}

/** Accessible name for a control that asks first. */
export function gatedLabel(action: string, request: ConsentRequest): string {
  switch (request.kind) {
    case "session":
      return `${action} (asks to allow cloud upload first)`;
    case "assistant":
      return `${action} (asks to send text to ${request.vendor} first)`;
    default:
      return request.need === "key"
        ? `${action} (needs a key for ${request.vendor ?? "the AI assistant"})`
        : `${action} (set up the AI assistant first)`;
  }
}

/** Privacy flags a confirmed session request turns on. */
export function privacyGrant(request: SessionConsentRequest, selection: ConsentSelection): Partial<PrivacyPolicy> {
  return {
    ...(request.audio && selection.audio ? { audio_upload_allowed: true } : {}),
    ...(request.transcript && selection.transcript ? { transcript_upload_allowed: true } : {}),
  };
}

/** True when the selection covers every row the request asked for. */
export function consentSatisfied(request: SessionConsentRequest, selection: ConsentSelection): boolean {
  return (!request.audio || selection.audio) && (!request.transcript || selection.transcript);
}

const MetricsContext = createContext<LiveMetrics>(emptySnapshot.metrics);

export function AppProvider({ children }: PropsWithChildren) {
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(emptySnapshot);
  // Metrics tick ~10 times per second; keeping them out of the snapshot
  // means transcript rows do not re-render on every level-meter update.
  const [metrics, setMetrics] = useState<LiveMetrics>(emptySnapshot.metrics);
  const [draft, setDraft] = useState<StartSessionRequest>(defaultDraft);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [caption, setCaption] = useState<CaptionPreferences>(defaultCaptionPreferences);
  const [catalog, setCatalog] = useState<ProviderCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionPending, setActionPending] = useState(false);
  const [onboardingComplete, setOnboardingComplete] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [consent, setConsent] = useState<{ request: ConsentRequest; resolve: (answer: ConsentAnswer) => void } | null>(null);
  const consentRef = useRef(consent);
  // Start reads the latest draft and revision: a consent grant made just
  // before it (in the same click) is not rendered yet.
  const latest = useRef({ draft, revision: snapshot.state_revision });
  latest.current = { draft, revision: snapshot.state_revision };

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
        setMetrics(nextSnapshot.metrics);
        setOnboardingComplete(nextOnboarding);
        setDraft({
          ...withProviderDefaults(nextDefaults),
          expected_state_revision: nextSnapshot.state_revision,
          asr_provider: smokeBackend ? "mock" : nextDefaults.asr_provider,
          translation_provider: smokeBackend ? "mock" : nextDefaults.translation_provider,
          ...restoreAudioSelection(nextDefaults, nextDevices),
        });
        setDevices(nextDevices);
        setCaption(nextCaption);
      })
      .catch((failure) => active && setError(message(failure)))
      .finally(() => active && setLoading(false));
    // The catalog is static for the life of the shell; a failure leaves the
    // provider selects on Auto plus the persisted value rather than blocking
    // the app.
    api
      .listProviders()
      .then((nextCatalog) => active && setCatalog(nextCatalog))
      .catch(() => undefined);
    subscribeUiEvents((event) => {
      if (event.kind === "metrics") {
        setMetrics((current) => ({ ...current, ...(event.payload as Partial<LiveMetrics>) }));
        return;
      }
      if (event.kind === "session_state") {
        setMetrics((event.payload as SessionSnapshot).metrics);
      }
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
      const next = await action();
      setMetrics(next.metrics);
      setSnapshot((current) => mergeCommandSnapshot(current, next));
    } catch (failure) {
      setError(message(failure));
    } finally {
      setActionPending(false);
    }
  }, []);

  const start = useCallback(
    () =>
      run(() => {
        const { draft: current, revision } = latest.current;
        return api.start({
          ...current,
          expected_state_revision: revision,
          audio_device_id: current.audio_source === "microphone" ? current.audio_device_id : null,
        });
      }),
    [run],
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

  const settleConsent = useCallback((answer: ConsentAnswer) => {
    const pending = consentRef.current;
    if (!pending) return;
    consentRef.current = null;
    setConsent(null);
    pending.resolve(answer);
  }, []);

  const askConsent = useCallback(
    (request: ConsentRequest) =>
      new Promise<ConsentAnswer>((resolve) => {
        // A newer request replaces one still open; the older one is cancelled.
        consentRef.current?.resolve("cancelled");
        const next = { request, resolve };
        consentRef.current = next;
        setConsent(next);
      }),
    [],
  );

  const requestConsent = useCallback(
    async (request: ConsentRequest) => (await askConsent(request)) === "granted",
    [askConsent],
  );

  const confirmConsent = useCallback(
    async (selection: ConsentSelection) => {
      const request = consentRef.current?.request;
      if (!request) return;
      if (request.kind === "session") {
        const grant = privacyGrant(request, selection);
        const current = latest.current.draft;
        // Saved by the session-defaults autosave like any other draft edit.
        latest.current = { ...latest.current, draft: { ...current, privacy: { ...current.privacy, ...grant } } };
        setDraft((draftNow) => ({ ...draftNow, privacy: { ...draftNow.privacy, ...grant } }));
        settleConsent(consentSatisfied(request, selection) ? "granted" : "declined");
        return;
      }
      if (request.kind === "assistant") {
        const preferences = await api.runtimePreferences();
        await api.updateRuntimePreferences({
          ...preferences,
          assistant: { ...defaultAssistantPreferences, ...preferences.assistant, transcript_upload_allowed: true },
        });
        announceAssistantChanged();
        settleConsent("granted");
        return;
      }
      // Setup happens in Settings; the action that asked does not go on.
      requestNavigation({ view: "settings", section: request.need === "key" ? "cloud-providers" : "ai-assistant" });
      settleConsent("cancelled");
    },
    [settleConsent],
  );

  const dismissConsent = useCallback(
    (answer: Exclude<ConsentAnswer, "granted">) => settleConsent(answer),
    [settleConsent],
  );

  const consentValue = useMemo<ConsentContextValue>(
    () => ({
      request: consent?.request ?? null,
      requestConsent,
      askConsent,
      confirm: confirmConsent,
      dismiss: dismissConsent,
    }),
    [askConsent, confirmConsent, consent, dismissConsent, requestConsent],
  );

  const value = useMemo<AppContextValue>(
    () => ({
      snapshot,
      draft,
      setDraft,
      devices,
      caption,
      catalog,
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
      requestConsent,
      askConsent,
    }),
    [
      actionPending,
      caption,
      catalog,
      devices,
      draft,
      error,
      loading,
      onboardingComplete,
      completeOnboarding,
      pause,
      requestConsent,
      askConsent,
      resume,
      snapshot,
      start,
      stop,
      updateCaption,
    ],
  );
  return (
    <AppContext.Provider value={value}>
      <ConsentContext.Provider value={consentValue}>
        <MetricsContext.Provider value={metrics}>{children}</MetricsContext.Provider>
      </ConsentContext.Provider>
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}

export function useLiveMetrics(): LiveMetrics {
  return useContext(MetricsContext);
}

/** The consent dialog's state, and `requestConsent` for components that are
 *  also rendered outside the provider (it then always answers false). */
export function useConsent(): ConsentContextValue {
  return useContext(ConsentContext);
}
