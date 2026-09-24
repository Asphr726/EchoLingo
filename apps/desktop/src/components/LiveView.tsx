import {
  ArrowDown,
  ClosedCaptioning,
  CloudArrowUp,
  Lock,
  LockKey,
  Microphone,
  Pause,
  Play,
  Stop,
  Waveform,
} from "@phosphor-icons/react";
import { memo, type ReactNode, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { systemAudioAvailable } from "../lib/audio";
import { api } from "../lib/bridge";
import { anchorCorrection, type FeedAnchor, followChangeAfterScroll, pickAnchor, scrollsBack } from "../lib/feedFollow";
import {
  effectiveProviderId,
  findProvider,
  languageName,
  needsAudioUpload as catalogNeedsAudioUpload,
  needsTranscriptUpload as catalogNeedsTranscriptUpload,
  providerVendor,
  supportsLanguage,
} from "../lib/providers";
import { sessionConsentRequest, sessionConsentSummary, useApp, useLiveMetrics } from "../state/AppContext";
import type { SegmentSummary, StartSessionRequest } from "../types";
import { LectureContextPanel } from "./LectureContextPanel";
import { ProviderSelect } from "./ProviderSelect";

const languages = [
  ["en", "English"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
];

export function LiveView() {
  const {
    actionPending,
    catalog,
    devices,
    draft,
    loading,
    pause,
    requestConsent,
    resume,
    setDraft,
    snapshot,
    start,
    stop,
  } = useApp();
  const consentNoteId = useId();
  const locked = !["IDLE", "COMPLETED"].includes(snapshot.phase);
  const microphones = devices.filter((device) => device.kind === "microphone");
  const systemAudioOffered = systemAudioAvailable(devices);
  const needsAudioUpload = catalogNeedsAudioUpload(catalog, draft);
  const needsTranscriptUpload = catalogNeedsTranscriptUpload(catalog, draft);
  const selectedAsr = findProvider(catalog, "asr", effectiveProviderId(draft, "asr") ?? "auto");
  const languageUnsupported =
    selectedAsr && !supportsLanguage(selectedAsr, draft.source_language)
      ? `${selectedAsr.display_name} does not support ${languageName(draft.source_language)} as the source language.`
      : null;
  // Uploads the chosen route needs and the privacy flags do not allow yet.
  // Start stays clickable and asks for them in the consent dialog.
  const consentNeeded = sessionConsentRequest(catalog, draft);

  const startSession = async () => {
    const request = sessionConsentRequest(catalog, draft);
    // The grant is applied to the draft `start` reads before it resolves.
    if (request && !(await requestConsent(request))) return;
    await start();
  };

  const update = <K extends keyof StartSessionRequest>(
    key: K,
    value: StartSessionRequest[K],
  ) => setDraft((current) => ({ ...current, [key]: value }));

  return (
    <div className="live-layout">
      <section className="live-main">
        {!locked && <div className="session-config" aria-label="Session configuration">
          <Field label="Source language">
            <select
              value={draft.source_language}
              disabled={locked}
              onChange={(event) => update("source_language", event.target.value)}
            >
              {languages.map(([code, label]) => (
                <option key={code} value={code} disabled={code === draft.target_language}>
                  {label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Target language">
            <select
              value={draft.target_language}
              disabled={locked}
              onChange={(event) => update("target_language", event.target.value)}
            >
              {languages.map(([code, label]) => (
                <option key={code} value={code} disabled={code === draft.source_language}>
                  {label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Audio source">
            <select
              value={draft.audio_source}
              disabled={locked}
              onChange={(event) => {
                const source = event.target.value as StartSessionRequest["audio_source"];
                setDraft((current) => ({
                  ...current,
                  audio_source: source,
                  audio_device_id:
                    source === "microphone"
                      ? microphones.find((device) => device.is_default)?.id ??
                        microphones[0]?.id ??
                        null
                      : null,
                }));
              }}
            >
              <option value="microphone">Microphone</option>
              {/* Not offered where the shell cannot capture it (Linux for now). */}
              {(systemAudioOffered || draft.audio_source === "system_audio") && (
                <option value="system_audio" disabled={!systemAudioOffered}>System audio</option>
              )}
            </select>
          </Field>
          <Field label="Audio profile">
            <select
              value={draft.audio_profile}
              disabled={locked}
              onChange={(event) => update("audio_profile", event.target.value)}
            >
              <option value="lecture">Lecture / far-field</option>
              <option value="conversation">Conversation</option>
              <option value="raw">Raw diagnostics</option>
            </select>
          </Field>
          {draft.audio_source === "microphone" && (
            <Field label="Input device" wide>
              <select
                value={draft.audio_device_id ?? ""}
                disabled={locked || microphones.length === 0}
                onChange={(event) => update("audio_device_id", event.target.value || null)}
              >
                {microphones.length === 0 ? (
                  <option value="">No microphone detected</option>
                ) : (
                  microphones.map((device) => (
                    <option key={device.id} value={device.id}>
                      {device.name}{device.is_default ? " — Default" : ""}
                    </option>
                  ))
                )}
              </select>
            </Field>
          )}
          <Field label="Inference mode">
            <select
              value={draft.inference_mode}
              disabled={locked}
              onChange={(event) =>
                update("inference_mode", event.target.value as StartSessionRequest["inference_mode"])
              }
            >
              <option value="auto">Auto</option>
              <option value="local">Local</option>
              <option value="cloud">Cloud</option>
            </select>
          </Field>
        </div>}

        {locked && <LiveSessionHeader />}

        <LectureContextPanel locked={locked} />

        <div className="session-actions">
          <div className="primary-actions">
            {["IDLE", "COMPLETED"].includes(snapshot.phase) && (
              <button
                className="button button--primary"
                type="button"
                disabled={loading || actionPending || microphones.length === 0 && draft.audio_source === "microphone"}
                aria-describedby={consentNeeded ? consentNoteId : undefined}
                onClick={() => void startSession()}
              >
                <Play size={17} weight="fill" aria-hidden="true" />
                Start
                {consentNeeded && (
                  <>
                    <Lock className="gate-lock" size={13} weight="bold" aria-hidden="true" />
                    <span className="visually-hidden"> (asks to allow cloud upload first)</span>
                  </>
                )}
              </button>
            )}
            {snapshot.phase === "LISTENING" && (
              <button className="button" type="button" disabled={actionPending} onClick={() => void pause()}>
                <Pause size={17} weight="fill" aria-hidden="true" />
                Pause
              </button>
            )}
            {snapshot.phase === "PAUSED" && (
              <button className="button button--primary" type="button" disabled={actionPending} onClick={() => void resume()}>
                <Play size={17} weight="fill" aria-hidden="true" />
                Resume
              </button>
            )}
            {["STARTING", "LISTENING", "PAUSED"].includes(snapshot.phase) && (
              <button className="button button--danger" type="button" disabled={actionPending} onClick={() => void stop()}>
                <Stop size={17} weight="fill" aria-hidden="true" />
                Stop
              </button>
            )}
          </div>
          <button className="button button--quiet" type="button" onClick={() => void api.showCaption()}>
            <ClosedCaptioning size={18} weight="regular" aria-hidden="true" />
            Floating caption
          </button>
        </div>
        {actionPending && <div className="action-progress" aria-label="Session action in progress" />}
        {snapshot.phase === "STARTING" && (
          <p className="startup-status" role="status">
            {snapshot.startup_status ?? "Starting the inference service and checking local models…"}
          </p>
        )}
        {["IDLE", "COMPLETED"].includes(snapshot.phase) && snapshot.startup_status && (
          <p className={`startup-status ${snapshot.models_ready ? "startup-status--ready" : ""}`} role="status">
            {snapshot.startup_status}
          </p>
        )}
        {consentNeeded && !locked && (
          <p className="consent-notice" id={consentNoteId}>
            <Lock size={13} weight="bold" aria-hidden="true" />
            <span>
              This route {sessionConsentSummary(consentNeeded)}. Start asks for your permission first.{" "}
              <button className="text-button text-button--inline" type="button" onClick={() => void requestConsent(consentNeeded)}>
                Review now
              </button>
            </span>
          </p>
        )}
        {!locked && languageUnsupported && (
          <p className="field-error" role="alert">{languageUnsupported}</p>
        )}

        <TranscriptStage />
      </section>

      <aside className="live-inspector">
        <RoutePanel />
        <AudioPanel />
        <LatencyPanel />
        <PrivacyDisclosure needsAudio={needsAudioUpload} needsTranscript={needsTranscriptUpload} />
      </aside>
    </div>
  );
}

function LiveSessionHeader() {
  const { snapshot } = useApp();
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const started = snapshot.started_at ? new Date(snapshot.started_at).valueOf() : now;
  const duration = Math.max(0, now - started);
  return (
    <header className="live-session-header">
      <div>
        <span className="section-kicker">Live session</span>
        <h2>{snapshot.phase === "PAUSED" ? "Lecture paused" : snapshot.phase === "STARTING" ? "Preparing inference" : "Listening to the lecture"}</h2>
      </div>
      <time>{formatDuration(duration)}</time>
    </header>
  );
}

function Field({
  children,
  label,
  wide = false,
}: {
  children: ReactNode;
  label: string;
  wide?: boolean;
}) {
  return (
    <label className={`field ${wide ? "field--wide" : ""}`}>
      <span>{label}</span>
      {children}
    </label>
  );
}

const languageLabel = (code: string | undefined) =>
  languages.find(([value]) => value === code)?.[1] ?? code ?? "";

export function segmentTranslationLabel(segment: SegmentSummary): string | null {
  if (segment.translation) return null;
  switch (segment.translation_status) {
    case "unavailable":
      return "Translation unavailable";
    case "streaming":
      return "Translating…";
    default:
      return "Translating…";
  }
}

const SegmentRow = memo(function SegmentRow({
  segment,
  sourceLanguage,
  targetLanguage,
}: {
  segment: SegmentSummary;
  sourceLanguage?: string;
  targetLanguage?: string;
}) {
  const placeholder = segmentTranslationLabel(segment);
  const streaming = segment.translation_status === "streaming";
  return (
    <article className="bilingual-row" data-status={segment.translation_status ?? "pending"} data-segment-id={segment.id}>
      <time>{formatTime(segment.start_ms)}</time>
      <p className="row-original" lang={sourceLanguage}>{segment.original}</p>
      {placeholder ? (
        <p className={`row-translation row-translation--placeholder ${segment.translation_status === "unavailable" ? "row-translation--unavailable" : ""}`} lang={targetLanguage}>
          {placeholder}
        </p>
      ) : (
        <p className={`row-translation ${streaming ? "row-translation--streaming" : ""}`} lang={targetLanguage}>
          {segment.translation}
        </p>
      )}
    </article>
  );
});

function LiveRow({
  openText,
  unstableText,
  provisionalTranslation,
  sourceLanguage,
  targetLanguage,
}: {
  openText: string;
  unstableText: string;
  provisionalTranslation: string;
  sourceLanguage?: string;
  targetLanguage?: string;
}) {
  const joiner = sourceLanguage === "zh" || sourceLanguage === "ja" ? "" : " ";
  return (
    <article className="bilingual-row bilingual-row--live" aria-label="Live, not yet committed">
      <span className="live-marker">Live</span>
      <p className="row-original" lang={sourceLanguage}>
        {openText && <span className="text-open">{openText}</span>}
        {openText && unstableText ? joiner : ""}
        {unstableText && <span className="text-unstable">{unstableText}</span>}
      </p>
      <p className="row-translation row-translation--provisional" lang={targetLanguage}>
        {provisionalTranslation || (openText || unstableText ? "…" : "")}
      </p>
    </article>
  );
}

function SpeechIndicator() {
  const metrics = useLiveMetrics();
  return (
    <div className={`speech-indicator ${metrics.speech_detected ? "speech-indicator--active" : ""}`}>
      <Waveform size={17} weight="bold" aria-hidden="true" />
      {metrics.speech_detected ? "Speech detected" : "Waiting for speech"}
    </div>
  );
}

const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

/** The reader's place in the feed, from the rows' on-screen boxes. */
function readAnchor(feed: HTMLElement): FeedAnchor | null {
  function* rows() {
    for (const row of feed.querySelectorAll<HTMLElement>("[data-segment-id]")) {
      const box = row.getBoundingClientRect();
      yield { id: row.dataset.segmentId ?? "", top: box.top, bottom: box.bottom };
    }
  }
  return pickAnchor(rows(), feed.getBoundingClientRect().top);
}

function TranscriptStage() {
  const { draft, snapshot } = useApp();
  const feedRef = useRef<HTMLDivElement>(null);
  // The layout effect reads the ref, so a scroll back the reader starts in
  // this frame wins over the next pin to the bottom; the state copy only
  // drives the "Back to live" button.
  const followingRef = useRef(true);
  const [following, setFollowingState] = useState(true);
  const anchorRef = useRef<FeedAnchor | null>(null);
  // Where the feed's own last scroll left it, to tell the reader's scrolls
  // from ours, and the position the previous scroll event saw.
  const ownScrollTopRef = useRef<number | null>(null);
  const lastScrollTopRef = useRef(0);
  const touchYRef = useRef<number | null>(null);
  const openText = snapshot.live.open_text.trim();
  const unstableText = snapshot.live.original_unstable.trim();
  const provisionalTranslation = openText || unstableText ? snapshot.live.translation_editable.trim() : "";
  const segments = snapshot.previous_segments;
  const hasCopy = segments.length > 0 || openText || unstableText;
  // Before a session has a config, the heading shows the languages Start will use.
  const sourceLanguage = snapshot.config?.source_language ?? draft.source_language;
  const targetLanguage = snapshot.config?.target_language ?? draft.target_language;

  const setFollowing = (value: boolean) => {
    followingRef.current = value;
    setFollowingState(value);
    if (value) anchorRef.current = null;
  };

  // The anchor is taken from the scroll event that follows, once the
  // browser has applied the reader's scroll.
  const scrollBack = () => {
    if (followingRef.current) setFollowing(false);
  };

  const scrollToLive = () => {
    const feed = feedRef.current;
    if (!feed) return;
    setFollowing(true);
    feed.scrollTo({ top: feed.scrollHeight, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  };

  // Every new session starts at its newest line.
  useEffect(() => {
    setFollowing(true);
  }, [snapshot.session_id]);

  useLayoutEffect(() => {
    const feed = feedRef.current;
    if (!feed) return;
    if (followingRef.current) {
      feed.scrollTop = feed.scrollHeight;
    } else {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const row = feed.querySelector<HTMLElement>(`[data-segment-id="${CSS.escape(anchor.id)}"]`);
      if (!row) {
        // The row dropped off the window; the next one becomes the place.
        anchorRef.current = readAnchor(feed);
        return;
      }
      const correction = anchorCorrection(anchor, row.getBoundingClientRect().top, feed.getBoundingClientRect().top);
      if (Math.abs(correction) < 0.5) return;
      feed.scrollTop += correction;
    }
    ownScrollTopRef.current = feed.scrollTop;
  }, [segments, openText, unstableText, provisionalTranslation]);

  return (
    <section className="transcript-stage" aria-label="Live transcript">
      <div className="transcript-heading">
        <div>
          <span className="section-kicker">Live subtitles</span>
          <h2>
            {languageLabel(sourceLanguage)}
            <span className="language-arrow" aria-hidden="true">→</span>
            {languageLabel(targetLanguage)}
          </h2>
        </div>
        <SpeechIndicator />
      </div>
      {!hasCopy ? (
        <div className="transcript-empty">
          <Microphone size={24} weight="regular" aria-hidden="true" />
          <strong>{snapshot.phase === "LISTENING" ? "Listening to the room" : "Ready for a lecture"}</strong>
          <span>
            {snapshot.phase === "LISTENING"
              ? "Far-field audio is flowing. The first words will appear here as they are recognized."
              : "Choose your audio source and start a session. No audio leaves this device without permission."}
          </span>
        </div>
      ) : (
        <div className="live-feed-shell">
          <div
            className="live-transcript-feed"
            ref={feedRef}
            tabIndex={0}
            onWheel={(event) => {
              if (event.deltaY < 0 && !event.ctrlKey && event.currentTarget.scrollTop > 0) scrollBack();
            }}
            onKeyDown={(event) => {
              if (scrollsBack(event.key) && event.currentTarget.scrollTop > 0) scrollBack();
            }}
            onTouchStart={(event) => {
              touchYRef.current = event.touches[0]?.clientY ?? null;
            }}
            onTouchMove={(event) => {
              const y = event.touches[0]?.clientY;
              // A finger moving down drags older lines into view.
              if (touchYRef.current !== null && y !== undefined && y > touchYRef.current + 2) scrollBack();
            }}
            onScroll={(event) => {
              const feed = event.currentTarget;
              const own = ownScrollTopRef.current !== null && Math.abs(feed.scrollTop - ownScrollTopRef.current) < 1;
              ownScrollTopRef.current = null;
              if (!own) {
                const change = followChangeAfterScroll(followingRef.current, lastScrollTopRef.current, feed);
                if (change) setFollowing(change === "follow");
                if (!followingRef.current) anchorRef.current = readAnchor(feed);
              }
              lastScrollTopRef.current = feed.scrollTop;
            }}
          >
            <div className="bilingual-row bilingual-row--header" aria-hidden="true">
              <span />
              <span>Original · {languageLabel(sourceLanguage)}</span>
              <span>Translation · {languageLabel(targetLanguage)}</span>
            </div>
            {segments.map((segment) => (
              <SegmentRow
                key={segment.id}
                segment={segment}
                sourceLanguage={sourceLanguage}
                targetLanguage={targetLanguage}
              />
            ))}
            {(openText || unstableText) && (
              <LiveRow
                openText={openText}
                unstableText={unstableText}
                provisionalTranslation={provisionalTranslation}
                sourceLanguage={sourceLanguage}
                targetLanguage={targetLanguage}
              />
            )}
          </div>
          {!following && (
            <button className="back-to-live" type="button" onClick={scrollToLive}>
              <ArrowDown size={14} weight="bold" aria-hidden="true" />
              Back to live
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function RoutePanel() {
  const { catalog, draft, setDraft, snapshot } = useApp();
  const locked = !["IDLE", "COMPLETED"].includes(snapshot.phase);
  const route = snapshot.route;
  return (
    <section className="inspector-section">
      <span className="section-kicker">Inference</span>
      <div className="backend-field">
        <label htmlFor="asr-provider">ASR provider</label>
        <ProviderSelect
          id="asr-provider"
          kind="asr"
          catalog={catalog}
          sourceLanguage={draft.source_language}
          value={draft.asr_provider}
          disabled={locked}
          onChange={(value) => setDraft((current) => ({ ...current, asr_provider: value }))}
        />
      </div>
      <div className="backend-field">
        <label htmlFor="translation-provider">Translation provider</label>
        <ProviderSelect
          id="translation-provider"
          kind="translation"
          catalog={catalog}
          sourceLanguage={draft.source_language}
          value={draft.translation_provider}
          disabled={locked}
          onChange={(value) => setDraft((current) => ({ ...current, translation_provider: value }))}
        />
      </div>
      <dl className="route-summary">
        <div><dt>ASR</dt><dd>{route?.asr_model ?? route?.asr_display_name ?? "Selected at start"}</dd></div>
        <div><dt>Translation</dt><dd>{route?.translation_model ?? route?.translation_display_name ?? "Selected at start"}</dd></div>
      </dl>
      <p className="route-reason">
        {route?.reason ?? "Auto will use capability calibration, model availability, privacy, and network state."}
      </p>
    </section>
  );
}

function AudioPanel() {
  const metrics = useLiveMetrics();
  const rms = metrics.input_rms_dbfs;
  const level = rms == null ? 0 : Math.min(1, Math.max(0, (rms + 60) / 60));
  return (
    <section className="inspector-section">
      <div className="inspector-title-row">
        <span className="section-kicker">Audio / VAD</span>
        <span>{rms == null ? "—" : `${rms.toFixed(1)} dBFS`}</span>
      </div>
      <div className="level-track" aria-label="Input RMS level">
        <span style={{ transform: `scaleX(${level})` }} />
      </div>
      <dl className="metric-grid">
        <div><dt>VAD probability</dt><dd>{percent(metrics.vad_probability)}</dd></div>
        <div><dt>Speech</dt><dd>{metrics.speech_detected ? "Detected" : "No"}</dd></div>
        <div><dt>Buffered</dt><dd>{formatMs(metrics.buffered_audio_ms)}</dd></div>
        <div><dt>Dropped</dt><dd>{formatMs(metrics.dropped_audio_ms)}</dd></div>
      </dl>
    </section>
  );
}

function LatencyPanel() {
  const metrics = useLiveMetrics();
  const backlog = metrics.translation_queue_depth ?? 0;
  return (
    <section className="inspector-section">
      <span className="section-kicker">Latency</span>
      <dl className="latency-list">
        <div><dt>First partial</dt><dd>{formatMs(metrics.asr_first_partial_latency_ms)}</dd></div>
        <div><dt>Sentence commit</dt><dd>{formatMs(metrics.asr_commit_latency_ms)}</dd></div>
        <div><dt>Translation</dt><dd>{formatMs(metrics.translation_latency_ms)}</dd></div>
        <div><dt>First delta</dt><dd>{formatMs(metrics.translation_first_delta_ms)}</dd></div>
        <div><dt>End to end</dt><dd>{formatMs(metrics.end_to_end_latency_ms)}</dd></div>
        <div>
          <dt>Translation backlog</dt>
          <dd>{backlog === 0 ? "clear" : `${backlog} · ${formatMs(metrics.translation_backlog_ms)}`}</dd>
        </div>
        {(metrics.translation_errors ?? 0) > 0 && (
          <div><dt>Translation errors</dt><dd>{metrics.translation_errors}</dd></div>
        )}
        {metrics.cloud_roundtrip_latency_ms != null && (
          <div><dt>Cloud roundtrip</dt><dd>{formatMs(metrics.cloud_roundtrip_latency_ms)}</dd></div>
        )}
        {metrics.network_jitter_ms != null && (
          <div><dt>Network jitter</dt><dd>{formatMs(metrics.network_jitter_ms)}</dd></div>
        )}
      </dl>
    </section>
  );
}

function PrivacyDisclosure({ needsAudio, needsTranscript }: { needsAudio: boolean; needsTranscript: boolean }) {
  const { catalog, draft, setDraft } = useApp();
  const asrVendor = providerVendor(catalog, "asr", effectiveProviderId(draft, "asr") ?? "");
  const translationVendor = providerVendor(catalog, "translation", effectiveProviderId(draft, "translation") ?? "");
  const audioCopy = needsAudio
    ? `Audio is uploaded to ${asrVendor}.`
    : "Audio stays on this device.";
  const transcriptCopy = needsTranscript
    ? `Transcript text is sent to ${translationVendor}.`
    : "Transcript text stays on this device.";
  return (
    <section className="privacy-disclosure">
      <div className="privacy-heading">
        {needsAudio ? <CloudArrowUp size={18} weight="regular" /> : <LockKey size={18} weight="regular" />}
        <strong>{needsAudio || needsTranscript ? "Cloud data boundary" : "Local data boundary"}</strong>
      </div>
      <p>
        {needsAudio || needsTranscript
          ? `${audioCopy} ${transcriptCopy}`
          : "Audio and transcript stay on this device unless Auto is allowed to choose cloud."}
      </p>
      {(needsAudio || draft.inference_mode === "auto") && (
        <label className="check-row">
          <input
            type="checkbox"
            checked={draft.privacy.audio_upload_allowed}
            onChange={(event) => setDraft((current) => ({
              ...current,
              privacy: { ...current.privacy, audio_upload_allowed: event.target.checked },
            }))}
          />
          <span>Allow audio upload</span>
        </label>
      )}
      {(needsTranscript || draft.inference_mode === "auto") && (
        <label className="check-row">
          <input
            type="checkbox"
            checked={draft.privacy.transcript_upload_allowed}
            onChange={(event) => setDraft((current) => ({
              ...current,
              privacy: { ...current.privacy, transcript_upload_allowed: event.target.checked },
            }))}
          />
          <span>Allow transcript upload</span>
        </label>
      )}
    </section>
  );
}

function formatTime(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

function formatDuration(ms: number) {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds % 3600 / 60);
  const remainder = seconds % 60;
  return `${hours > 0 ? `${hours.toString().padStart(2, "0")}:` : ""}${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

function formatMs(value?: number | null) {
  return value == null ? "—" : `${Math.round(value)} ms`;
}

function percent(value?: number | null) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}
