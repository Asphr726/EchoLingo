import {
  ArrowDown,
  ClosedCaptioning,
  CloudArrowUp,
  LockKey,
  Microphone,
  Pause,
  Play,
  Stop,
  Waveform,
} from "@phosphor-icons/react";
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../lib/bridge";
import { useApp } from "../state/AppContext";
import type { StartSessionRequest } from "../types";

const languages = [
  ["en", "English"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
];

export function LiveView() {
  const {
    actionPending,
    devices,
    draft,
    loading,
    pause,
    resume,
    setDraft,
    snapshot,
    start,
    stop,
  } = useApp();
  const locked = !["IDLE", "COMPLETED"].includes(snapshot.phase);
  const microphones = devices.filter((device) => device.kind === "microphone");
  const needsAudioUpload =
    draft.asr_provider === "qwen_cloud" ||
    (draft.inference_mode === "cloud" && draft.asr_provider === "auto");
  const needsTranscriptUpload =
    draft.translation_provider === "qwen_cloud" ||
    (draft.inference_mode === "cloud" && draft.translation_provider === "auto");
  const privacyBlocked =
    (needsAudioUpload && !draft.privacy.audio_upload_allowed) ||
    (needsTranscriptUpload && !draft.privacy.transcript_upload_allowed);

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
              <option value="system_audio">System audio</option>
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

        <div className="session-actions">
          <div className="primary-actions">
            {["IDLE", "COMPLETED"].includes(snapshot.phase) && (
              <button
                className="button button--primary"
                type="button"
                disabled={loading || actionPending || privacyBlocked || microphones.length === 0 && draft.audio_source === "microphone"}
                onClick={() => void start()}
              >
                <Play size={17} weight="fill" aria-hidden="true" />
                Start
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
        {privacyBlocked && (
          <p className="field-error" role="alert">
            Enable the required upload permission before starting this cloud route.
          </p>
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

function TranscriptStage() {
  const { snapshot } = useApp();
  const feedRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const pendingOriginal = snapshot.live.original_unstable.trim();
  const pendingTranslation =
    snapshot.live.translation_source_revision_id === snapshot.live.source_revision_id
      ? snapshot.live.translation_editable.trim()
      : "";
  const segments = snapshot.previous_segments;
  const hasCopy = segments.length > 0 || pendingOriginal || pendingTranslation;
  const lastSegment = segments.at(-1);
  const contentToken = `${segments.length}:${lastSegment?.translation ?? ""}:${pendingOriginal}:${pendingTranslation}`;

  const scrollToLive = () => {
    const feed = feedRef.current;
    if (!feed) return;
    feed.scrollTo({ top: feed.scrollHeight, behavior: "smooth" });
    setFollowing(true);
  };

  useLayoutEffect(() => {
    const feed = feedRef.current;
    if (feed && following) feed.scrollTop = feed.scrollHeight;
  }, [contentToken, following]);

  return (
    <section className="transcript-stage" aria-label="Live transcript">
      <div className="transcript-heading">
        <div>
          <span className="section-kicker">Current</span>
          <h2>Live subtitles</h2>
        </div>
        <div className={`speech-indicator ${snapshot.metrics.speech_detected ? "speech-indicator--active" : ""}`}>
          <Waveform size={17} weight="bold" aria-hidden="true" />
          {snapshot.metrics.speech_detected ? "Speech detected" : "Waiting for speech"}
        </div>
      </div>
      {!hasCopy ? (
        <div className="transcript-empty">
          <Microphone size={24} weight="regular" aria-hidden="true" />
          <strong>{snapshot.phase === "LISTENING" ? "Listening to the room" : "Ready for a lecture"}</strong>
          <span>
            {snapshot.phase === "LISTENING"
              ? "Far-field audio is flowing. The first stable words will appear here."
              : "Choose your audio source and start a session. No audio leaves this device without permission."}
          </span>
        </div>
      ) : (
        <div className="live-feed-shell">
          <div
            className="live-transcript-feed"
            ref={feedRef}
            aria-live="polite"
            onScroll={(event) => {
              const feed = event.currentTarget;
              setFollowing(feed.scrollHeight - feed.scrollTop - feed.clientHeight < 48);
            }}
          >
            {segments.map((segment) => (
              <article className="live-segment" key={segment.id}>
                <time>{formatTime(segment.start_ms)}</time>
                <div className="live-segment-copy">
                  <p lang={snapshot.config?.source_language}>{segment.original}</p>
                  {segment.translation ? (
                    <p className="segment-translation" lang={snapshot.config?.target_language}>
                      {segment.translation}
                    </p>
                  ) : (
                    <span className="translation-pending">Translating…</span>
                  )}
                </div>
              </article>
            ))}
            {(pendingOriginal || pendingTranslation) && (
              <article className="live-segment live-segment--editable">
                <span className="live-marker">Live</span>
                <div className="live-segment-copy">
                  {pendingOriginal && (
                    <p lang={snapshot.config?.source_language}>{pendingOriginal}</p>
                  )}
                  {pendingTranslation && (
                    <p className="segment-translation" lang={snapshot.config?.target_language}>
                      {pendingTranslation}
                    </p>
                  )}
                </div>
              </article>
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
  const { draft, setDraft, snapshot } = useApp();
  const locked = !["IDLE", "COMPLETED"].includes(snapshot.phase);
  const route = snapshot.route;
  return (
    <section className="inspector-section">
      <span className="section-kicker">Inference</span>
      <div className="backend-field">
        <label htmlFor="asr-provider">ASR provider</label>
        <select
          id="asr-provider"
          value={draft.asr_provider}
          disabled={locked}
          onChange={(event) => setDraft((current) => ({ ...current, asr_provider: event.target.value }))}
        >
          <option value="auto">Auto select</option>
          <option value="qwen_local">Qwen3-ASR local</option>
          <option value="qwen_cloud">Qwen realtime cloud</option>
          <option value="simulstreaming">Whisper fallback</option>
          <option value="mock">Mock diagnostics</option>
        </select>
      </div>
      <div className="backend-field">
        <label htmlFor="translation-provider">Translation provider</label>
        <select
          id="translation-provider"
          value={draft.translation_provider}
          disabled={locked}
          onChange={(event) => setDraft((current) => ({ ...current, translation_provider: event.target.value }))}
        >
          <option value="auto">Auto select</option>
          <option value="hymt_local">Hy-MT2 local</option>
          <option value="qwen_cloud">Qwen-MT cloud</option>
          <option value="mock">Mock diagnostics</option>
        </select>
      </div>
      <dl className="route-summary">
        <div><dt>ASR</dt><dd>{route?.asr_model ?? "Selected at start"}</dd></div>
        <div><dt>Translation</dt><dd>{route?.translation_model ?? "Selected at start"}</dd></div>
      </dl>
      <p className="route-reason">
        {route?.reason ?? "Auto will use capability calibration, model availability, privacy, and network state."}
      </p>
    </section>
  );
}

function AudioPanel() {
  const { snapshot } = useApp();
  const rms = snapshot.metrics.input_rms_dbfs;
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
        <div><dt>VAD probability</dt><dd>{percent(snapshot.metrics.vad_probability)}</dd></div>
        <div><dt>Speech</dt><dd>{snapshot.metrics.speech_detected ? "Detected" : "No"}</dd></div>
        <div><dt>Buffered</dt><dd>{formatMs(snapshot.metrics.buffered_audio_ms)}</dd></div>
        <div><dt>Dropped</dt><dd>{formatMs(snapshot.metrics.dropped_audio_ms)}</dd></div>
      </dl>
    </section>
  );
}

function LatencyPanel() {
  const { snapshot } = useApp();
  const metrics = snapshot.metrics;
  return (
    <section className="inspector-section">
      <span className="section-kicker">Latency</span>
      <dl className="latency-list">
        <div><dt>First partial</dt><dd>{formatMs(metrics.asr_first_partial_latency_ms)}</dd></div>
        <div><dt>ASR stable</dt><dd>{formatMs(metrics.asr_commit_latency_ms)}</dd></div>
        <div><dt>Translation</dt><dd>{formatMs(metrics.translation_latency_ms)}</dd></div>
        <div><dt>End to end</dt><dd>{formatMs(metrics.end_to_end_latency_ms)}</dd></div>
        <div><dt>Cloud roundtrip</dt><dd>{formatMs(metrics.cloud_roundtrip_latency_ms)}</dd></div>
        <div><dt>Network jitter</dt><dd>{formatMs(metrics.network_jitter_ms)}</dd></div>
      </dl>
    </section>
  );
}

function PrivacyDisclosure({ needsAudio, needsTranscript }: { needsAudio: boolean; needsTranscript: boolean }) {
  const { draft, setDraft } = useApp();
  return (
    <section className="privacy-disclosure">
      <div className="privacy-heading">
        {needsAudio ? <CloudArrowUp size={18} weight="regular" /> : <LockKey size={18} weight="regular" />}
        <strong>{needsAudio || needsTranscript ? "Cloud data boundary" : "Local data boundary"}</strong>
      </div>
      <p>
        {needsAudio
          ? "Processed speech audio is sent to Alibaba Cloud Qwen ASR."
          : needsTranscript
            ? "Audio stays on this device. Source transcript text is sent to Qwen-MT."
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
