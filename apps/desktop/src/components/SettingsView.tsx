import { CheckCircle, ClosedCaptioning, CloudArrowUp, Key, LockKey, SlidersHorizontal, Warning } from "@phosphor-icons/react";
import { type ReactNode, useEffect, useState } from "react";
import { api, subscribeModelProgress } from "../lib/bridge";
import { useApp } from "../state/AppContext";
import type { CaptionDisplay, CloudCredentialStatus, CloudProbeResult, InferenceMode, ModelProgress, ModelStatus, RuntimePreferences, StartSessionRequest } from "../types";

const sections = [
  "General",
  "Audio",
  "Languages",
  "Inference",
  "Models",
  "Privacy",
  "Translation",
  "Appearance",
  "Advanced",
] as const;
type Section = (typeof sections)[number];

export function SettingsView() {
  const [section, setSection] = useState<Section>("General");
  const [credentialStatus, setCredentialStatus] = useState<CloudCredentialStatus | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [workspaceId, setWorkspaceId] = useState("");
  const [credentialPending, setCredentialPending] = useState(false);
  const [credentialError, setCredentialError] = useState<string | null>(null);
  const [cloudProbe, setCloudProbe] = useState<CloudProbeResult | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [modelProgress, setModelProgress] = useState<Record<string, ModelProgress>>({});
  const [modelPending, setModelPending] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const { caption, draft, setDraft, snapshot, updateCaption } = useApp();
  const [runtime, setRuntime] = useState<RuntimePreferences>({ preload_local_models: true });
  useEffect(() => {
    let active = true;
    api.runtimePreferences().then((value) => active && setRuntime(value)).catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);
  const updateRuntime = async (next: RuntimePreferences) => {
    setRuntime(next);
    try {
      setRuntime(await api.updateRuntimePreferences(next));
    } catch {
      setRuntime(runtime);
    }
  };
  const update = <K extends keyof StartSessionRequest>(key: K, value: StartSessionRequest[K]) =>
    setDraft((current) => ({ ...current, [key]: value }));

  useEffect(() => {
    void api.credentialStatus().then(setCredentialStatus).catch((error) => {
      setCredentialError(String(error));
    });
  }, []);

  useEffect(() => {
    let active = true;
    let dispose: () => void = () => undefined;
    void api.models().then((value) => active && setModels(value)).catch((error) => {
      if (active) setModelError(String(error));
    });
    void subscribeModelProgress((progress) => {
      setModelProgress((current) => ({ ...current, [progress.model_id]: progress }));
    }).then((unlisten) => {
      if (active) dispose = unlisten;
      else unlisten();
    });
    return () => {
      active = false;
      dispose();
    };
  }, []);

  const updateModel = async (modelId: string, action: "install" | "verify" | "delete") => {
    setModelPending(modelId);
    setModelError(null);
    try {
      const status = action === "install" ? await api.installModel(modelId) : action === "verify" ? await api.verifyModel(modelId) : await api.deleteModel(modelId);
      setModels((current) => current.map((model) => model.id === modelId ? status : model));
      if (action !== "install") {
        setModelProgress((current) => {
          const next = { ...current };
          delete next[modelId];
          return next;
        });
      }
    } catch (error) {
      setModelError(String(error));
      setModels(await api.models().catch(() => models));
    } finally {
      setModelPending(null);
    }
  };

  const saveCredentials = async () => {
    setCredentialPending(true);
    setCredentialError(null);
    try {
      setCredentialStatus(await api.setCloudCredentials(apiKey, workspaceId));
      setCloudProbe(null);
      setApiKey("");
      setWorkspaceId("");
    } catch (error) {
      setCredentialError(String(error));
    } finally {
      setCredentialPending(false);
    }
  };

  const testCloud = async () => {
    setCredentialPending(true);
    setCredentialError(null);
    setCloudProbe(null);
    try {
      setCloudProbe(
        await api.probeQwenCloud(draft.privacy.transcript_upload_allowed),
      );
    } catch (error) {
      setCredentialError(String(error));
    } finally {
      setCredentialPending(false);
    }
  };

  const clearCredentials = async () => {
    setCredentialPending(true);
    setCredentialError(null);
    try {
      setCredentialStatus(await api.clearCloudCredentials());
      setCloudProbe(null);
    } catch (error) {
      setCredentialError(String(error));
    } finally {
      setCredentialPending(false);
    }
  };

  return (
    <div className="settings-layout">
      <nav className="settings-nav" aria-label="Settings sections">
        {sections.map((item) => (
          <button
            className={section === item ? "settings-link settings-link--active" : "settings-link"}
            key={item}
            type="button"
            onClick={() => setSection(item)}
          >
            {item}
          </button>
        ))}
      </nav>
      <section className="settings-content">
        <header className="settings-header">
          <span className="section-kicker">Preferences</span>
          <h2>{section}</h2>
        </header>

        {section === "General" && (
          <div className="settings-groups">
            <SettingsGroup title="Floating caption" description="Keep bilingual text above lecture slides and note-taking apps.">
              <div className="settings-inline">
                <button className="button" type="button" onClick={() => void api.showCaption()}>
                  <ClosedCaptioning size={18} weight="regular" aria-hidden="true" />
                  Open caption
                </button>
              </div>
            </SettingsGroup>
          </div>
        )}

        {section === "Audio" && (
          <div className="settings-groups">
            <SettingsGroup title="Far-field frontend" description="Audio capture, enhancement, AGC, VAD annotation, resampling, and buffering always run locally.">
              <Control label="Default profile">
                <select value={draft.audio_profile} onChange={(event) => update("audio_profile", event.target.value)}>
                  <option value="lecture">Lecture / far-field</option>
                  <option value="conversation">Conversation</option>
                  <option value="raw">Raw diagnostics</option>
                </select>
              </Control>
              <div className="settings-note">
                <SlidersHorizontal size={19} weight="regular" aria-hidden="true" />
                <p>Lecture mode treats VAD as an annotation. Quiet distant speech is not hard-dropped before ASR.</p>
              </div>
            </SettingsGroup>
          </div>
        )}

        {section === "Languages" && (
          <div className="settings-groups">
            <SettingsGroup title="Session defaults" description="Core V1 regression languages are English, Chinese, Japanese, and Korean.">
              <Control label="Source language">
                <LanguageSelect value={draft.source_language} onChange={(value) => update("source_language", value)} exclude={draft.target_language} />
              </Control>
              <Control label="Target language">
                <LanguageSelect value={draft.target_language} onChange={(value) => update("target_language", value)} exclude={draft.source_language} />
              </Control>
            </SettingsGroup>
          </div>
        )}

        {section === "Inference" && (
          <div className="settings-groups">
            <SettingsGroup title="Runtime mode" description="ASR and translation route independently, so Hybrid sessions remain possible.">
              <div className="segmented-control" role="radiogroup" aria-label="Inference mode">
                {(["auto", "local", "cloud"] as InferenceMode[]).map((mode) => (
                  <button key={mode} className={draft.inference_mode === mode ? "segmented-control--active" : ""} type="button" role="radio" aria-checked={draft.inference_mode === mode} onClick={() => update("inference_mode", mode)}>
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </button>
                ))}
              </div>
              <div className="auto-decision">
                <span>Selected ASR</span>
                <strong>{snapshot.route?.asr_model ?? "Evaluated when the session starts"}</strong>
                <p>{snapshot.route?.reason ?? "Auto compares calibration, memory, local model availability, network, credentials, and privacy policy."}</p>
              </div>
            </SettingsGroup>
            <SettingsGroup title="Startup" description="Local models take about a minute to load. Preloading keeps them resident so the first Start responds immediately.">
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={runtime.preload_local_models}
                  onChange={(event) => void updateRuntime({ ...runtime, preload_local_models: event.target.checked })}
                />
                <span>Preload local models when EchoLingo launches</span>
              </label>
              <p className="settings-helper">{snapshot.startup_status ?? "Models load in the background after launch and stay loaded until you quit."}</p>
            </SettingsGroup>
          </div>
        )}

        {section === "Models" && (
          <div className="settings-groups">
            <SettingsGroup title="Provider preferences" description="Unavailable local models fall back through RuntimeRouter; provider APIs never leak into the pipeline.">
              <Control label="ASR provider">
                <select value={draft.asr_provider} onChange={(event) => update("asr_provider", event.target.value)}>
                  <option value="auto">Auto select</option>
                  <option value="qwen_local">Qwen3-ASR local</option>
                  <option value="qwen_cloud">Qwen realtime cloud</option>
                  <option value="simulstreaming">SimulStreaming / Whisper</option>
                </select>
              </Control>
              <Control label="Translation provider">
                <select value={draft.translation_provider} onChange={(event) => update("translation_provider", event.target.value)}>
                  <option value="auto">Auto select</option>
                  <option value="hymt_local">Hy-MT2 local</option>
                  <option value="qwen_cloud">Qwen-MT cloud</option>
                </select>
              </Control>
              <p className="settings-helper">Auto uses an installed model only after the matching runtime calibration meets the configured latency target.</p>
            </SettingsGroup>
            <SettingsGroup title="Local model manager" description="Downloads are pinned, checked, and activated atomically in EchoLingo application storage.">
              <div className="model-list">
                {models.length === 0 && <p className="settings-helper">Loading the local model catalog…</p>}
                {models.map((model) => {
                  const progress = modelProgress[model.id];
                  const total = progress?.total_bytes || model.size_bytes;
                  const completed = progress?.bytes_completed ?? 0;
                  const percentage = total > 0 ? Math.min(100, Math.round(completed / total * 100)) : 0;
                  const busy = modelPending === model.id;
                  return (
                    <article className="model-row" key={model.id}>
                      <div className="model-row-copy">
                        <span>{model.role}</span>
                        <strong>{model.display_name}</strong>
                        <small>{formatBytes(model.size_bytes)} · {model.state.replace("_", " ")}</small>
                      </div>
                      <div className="model-row-actions">
                        {(busy || model.state === "installing") && (
                          <div className="model-progress" aria-label={`${model.display_name} ${percentage}%`}>
                            <span style={{ width: `${percentage}%` }} />
                            <small>{progress?.phase ?? "preparing"} · {percentage}%</small>
                          </div>
                        )}
                        {model.state === "ready" ? (
                          <>
                            <button className="button" type="button" disabled={busy} onClick={() => void updateModel(model.id, "verify")}>Verify</button>
                            <button className="button" type="button" disabled={busy} onClick={() => void updateModel(model.id, "delete")}>Delete</button>
                          </>
                        ) : (
                          <button className="button button--primary" type="button" disabled={busy} onClick={() => void updateModel(model.id, "install")}>{model.state === "corrupt" ? "Retry" : "Download"}</button>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
              {modelError && <p className="settings-error" role="alert">{modelError}</p>}
            </SettingsGroup>
            <SettingsGroup title="Cloud credentials" description="Saved in macOS Keychain. Secret values never enter EchoLingo settings, history, events, or logs.">
              <div className="credential-status" role="status">
                {credentialStatus?.api_key_available && credentialStatus.workspace_id_available ? (
                  <CheckCircle size={20} weight="fill" aria-hidden="true" />
                ) : (
                  <Warning size={20} weight="fill" aria-hidden="true" />
                )}
                <div>
                  <strong>{credentialStatus?.api_key_available && credentialStatus.workspace_id_available ? "Qwen Cloud configured" : "Qwen Cloud not configured"}</strong>
                  <small>{credentialStatus?.source === "macos_keychain" ? "Credentials are stored in macOS Keychain." : credentialStatus?.source === "environment" ? "Using development environment variables." : "Cloud sessions stay unavailable until both values are saved."}</small>
                </div>
              </div>
              <Control label="DashScope API key">
                <input type="password" autoComplete="off" value={apiKey} placeholder={credentialStatus?.api_key_available ? "Saved — enter to replace" : "Enter API key"} onChange={(event) => setApiKey(event.target.value)} />
              </Control>
              <Control label="DashScope workspace ID">
                <input type="password" autoComplete="off" value={workspaceId} placeholder={credentialStatus?.workspace_id_available ? "Saved — enter to replace" : "Enter workspace ID"} onChange={(event) => setWorkspaceId(event.target.value)} />
              </Control>
              <p className="settings-helper">
                Use an international Model Studio API key and workspace ID from the same Singapore region. The connection test opens an authenticated ASR WebSocket but uploads no audio.
              </p>
              {credentialError && <p className="settings-error" role="alert">{credentialError}</p>}
              {cloudProbe && (
                <div className={cloudProbe.ok ? "cloud-probe cloud-probe--ok" : "cloud-probe cloud-probe--error"} role="status">
                  <strong>{cloudProbe.ok ? "Qwen Cloud connection works" : "Qwen Cloud needs attention"}</strong>
                  <span>
                    ASR: {cloudProbe.asr.status}
                    {cloudProbe.asr.handshake_latency_ms != null ? ` · ${cloudProbe.asr.handshake_latency_ms.toFixed(0)} ms handshake` : ""}
                    {` · Translation: ${cloudProbe.translation.status}`}
                  </span>
                  {cloudProbe.message && <small>{cloudProbe.message}</small>}
                  <small>Audio uploaded: no{draft.privacy.transcript_upload_allowed ? " · A fixed test sentence was sent to Qwen-MT." : " · Enable transcript upload to include Qwen-MT in this test."}</small>
                </div>
              )}
              <div className="settings-inline">
                <button className="button button--primary" type="button" disabled={credentialPending || apiKey.trim().length < 8 || workspaceId.trim().length < 3} onClick={() => void saveCredentials()}>
                  <Key size={18} weight="regular" aria-hidden="true" />
                  {credentialPending ? "Saving…" : "Save to Keychain"}
                </button>
                <button className="button" type="button" disabled={credentialPending || credentialStatus?.source !== "macos_keychain"} onClick={() => void clearCredentials()}>
                  Remove
                </button>
                <button className="button" type="button" disabled={credentialPending || !credentialStatus?.api_key_available || !credentialStatus.workspace_id_available} onClick={() => void testCloud()}>
                  {credentialPending ? "Working…" : "Test connection"}
                </button>
              </div>
            </SettingsGroup>
          </div>
        )}

        {section === "Privacy" && (
          <div className="settings-groups">
            <SettingsGroup title="Explicit upload consent" description="EchoLingo never silently uploads audio or transcript text.">
              <PrivacyToggle
                icon={<CloudArrowUp size={20} weight="regular" aria-hidden="true" />}
                title="Audio upload"
                description="Required only for Cloud ASR. Processed ASR-bound audio is sent to the selected provider."
                checked={draft.privacy.audio_upload_allowed}
                onChange={(checked) => setDraft((current) => ({ ...current, privacy: { ...current.privacy, audio_upload_allowed: checked } }))}
              />
              <PrivacyToggle
                icon={<LockKey size={20} weight="regular" aria-hidden="true" />}
                title="Transcript upload"
                description="Required only for Cloud Translation. It sends source text, never microphone audio."
                checked={draft.privacy.transcript_upload_allowed}
                onChange={(checked) => setDraft((current) => ({ ...current, privacy: { ...current.privacy, transcript_upload_allowed: checked } }))}
              />
            </SettingsGroup>
          </div>
        )}

        {section === "Translation" && (
          <div className="settings-groups">
            <SettingsGroup title="Context and terminology" description="Streaming translation preserves an editable window and commits stable target text progressively.">
              <Control label="Domain hint">
                <input type="text" placeholder="e.g. computer science lecture" disabled title="Domain settings backend persistence is scheduled for the next increment" />
              </Control>
              <Control label="Terminology / glossary">
                <textarea rows={5} placeholder="One source = target term per line" disabled title="Glossary editor backend persistence is scheduled for the next increment" />
              </Control>
              <p className="settings-helper">The backend interfaces already accept domain, context, glossary, and editable-window data. Desktop persistence for these fields is next.</p>
            </SettingsGroup>
          </div>
        )}

        {section === "Appearance" && (
          <div className="settings-groups">
            <SettingsGroup title="Floating caption" description="Adjust caption density and contrast for long lectures without changing transcript data.">
              <Control label="Display">
                <select value={caption.display} onChange={(event) => void updateCaption({ ...caption, display: event.target.value as CaptionDisplay })}>
                  <option value="both">Original and translation</option>
                  <option value="original">Original only</option>
                  <option value="translation">Translation only</option>
                </select>
              </Control>
              <Control label={`Font size — ${caption.font_size_px} px`}>
                <input type="range" min="18" max="72" step="1" value={caption.font_size_px} onChange={(event) => void updateCaption({ ...caption, font_size_px: Number(event.target.value) })} />
              </Control>
              <Control label={`Opacity — ${Math.round(caption.opacity * 100)}%`}>
                <input type="range" min="0.35" max="1" step="0.05" value={caption.opacity} onChange={(event) => void updateCaption({ ...caption, opacity: Number(event.target.value) })} />
              </Control>
              <Control label="Recent segments">
                <select value={caption.recent_segments} onChange={(event) => void updateCaption({ ...caption, recent_segments: Number(event.target.value) })}>
                  <option value="1">1 segment</option><option value="2">2 segments</option><option value="3">3 segments</option>
                </select>
              </Control>
            </SettingsGroup>
          </div>
        )}

        {section === "Advanced" && (
          <div className="settings-groups">
            <SettingsGroup title="Diagnostics" description="Metrics remain visible for lecture tuning and release-gate evidence.">
              <dl className="advanced-status">
                <div><dt>Protocol</dt><dd>Canonical UI events v1</dd></div>
                <div><dt>Forced alignment</dt><dd>Background after Stop · local model</dd></div>
                <div><dt>Physical far-field test</dt><dd>Pending hardware validation</dd></div>
                <div><dt>Cloud benchmark</dt><dd>Pending credentials</dd></div>
              </dl>
            </SettingsGroup>
          </div>
        )}
      </section>
    </div>
  );
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function SettingsGroup({ children, description, title }: { children: ReactNode; description: string; title: string }) {
  return (
    <section className="settings-group">
      <header><h3>{title}</h3><p>{description}</p></header>
      <div className="settings-group-body">{children}</div>
    </section>
  );
}

function Control({ children, label }: { children: ReactNode; label: string }) {
  return <label className="settings-control"><span>{label}</span>{children}</label>;
}

function LanguageSelect({ exclude, onChange, value }: { exclude: string; onChange: (value: string) => void; value: string }) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      <option value="en" disabled={exclude === "en"}>English</option>
      <option value="zh" disabled={exclude === "zh"}>Chinese</option>
      <option value="ja" disabled={exclude === "ja"}>Japanese</option>
      <option value="ko" disabled={exclude === "ko"}>Korean</option>
    </select>
  );
}

function PrivacyToggle({ checked, description, icon, onChange, title }: { checked: boolean; description: string; icon: ReactNode; onChange: (checked: boolean) => void; title: string }) {
  return (
    <label className="privacy-toggle">
      <span className="privacy-toggle-icon">{icon}</span>
      <span><strong>{title}</strong><small>{description}</small></span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}
