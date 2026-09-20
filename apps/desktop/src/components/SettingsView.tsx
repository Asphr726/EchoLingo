import { ArrowSquareOut, CheckCircle, ClosedCaptioning, CloudArrowUp, Key, LockKey, PlugsConnected, SlidersHorizontal, Warning } from "@phosphor-icons/react";
import { type ReactNode, useEffect, useState } from "react";
import { api, subscribeModelProgress } from "../lib/bridge";
import { providersForGroup } from "../lib/providers";
import { useApp } from "../state/AppContext";
import type {
  CaptionDisplay,
  CloudProbeAsrResult,
  CloudProbeResult,
  CloudProbeTranslationResult,
  CredentialGroup,
  CredentialGroupStatus,
  CredentialSource,
  InferenceMode,
  ModelProgress,
  ModelStatus,
  ProviderSpec,
  RuntimePreferences,
  StartSessionRequest,
} from "../types";
import { ProviderSelect } from "./ProviderSelect";

const sections = [
  "General",
  "Audio",
  "Languages",
  "Inference",
  "Models",
  "Cloud providers",
  "Privacy",
  "Translation",
  "Appearance",
  "Advanced",
] as const;
type Section = (typeof sections)[number];

const sectionSlug = (section: Section) => section.toLowerCase().replace(/\s+/g, "-");

/** `?section=cloud-providers` opens a section directly (browser preview and
 *  screenshots); the in-app navigation never changes the URL. */
function initialSection(): Section {
  if (typeof window === "undefined") return "General";
  const requested = new URLSearchParams(window.location.search).get("section");
  return sections.find((section) => sectionSlug(section) === requested) ?? "General";
}

export function SettingsView() {
  const [section, setSection] = useState<Section>(initialSection);
  // Availability per credential group; secret values are never held here.
  const [credentialStatus, setCredentialStatus] = useState<Record<string, CredentialGroupStatus>>({});
  const [credentialError, setCredentialError] = useState<string | null>(null);
  const [logsDirectory, setLogsDirectory] = useState<string | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [modelProgress, setModelProgress] = useState<Record<string, ModelProgress>>({});
  const [modelPending, setModelPending] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const { caption, catalog, draft, setDraft, snapshot, updateCaption } = useApp();
  const sessionActive = !["IDLE", "COMPLETED"].includes(snapshot.phase);
  const [runtime, setRuntime] = useState<RuntimePreferences>({ preload_local_models: true, providers: {} });
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
    let active = true;
    void api
      .credentialStatus()
      .then((groups) => {
        if (!active) return;
        setCredentialStatus(Object.fromEntries(groups.map((group) => [group.group_id, group])));
      })
      .catch((error) => active && setCredentialError(String(error)));
    void api.logsDirectory().then((path) => active && setLogsDirectory(path)).catch(() => undefined);
    return () => {
      active = false;
    };
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
                <strong>{snapshot.route?.asr_model ?? snapshot.route?.asr_display_name ?? "Evaluated when the session starts"}</strong>
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
                <ProviderSelect kind="asr" catalog={catalog} sourceLanguage={draft.source_language} value={draft.asr_provider} onChange={(value) => update("asr_provider", value)} />
              </Control>
              <Control label="Translation provider">
                <ProviderSelect kind="translation" catalog={catalog} sourceLanguage={draft.source_language} value={draft.translation_provider} onChange={(value) => update("translation_provider", value)} />
              </Control>
              <p className="settings-helper">Auto uses an installed model only after the matching runtime calibration meets the configured latency target. Cloud entries need their credentials saved under Cloud providers.</p>
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
          </div>
        )}

        {section === "Cloud providers" && (
          <div className="settings-groups">
            <SettingsGroup title="Auto route" description="Auto keeps the local Qwen3-ASR and Hy-MT2 route as long as it meets the latency target; these are the cloud providers it hands over to.">
              <Control label="Preferred cloud recognizer">
                <ProviderSelect kind="asr" cloudOnly catalog={catalog} sourceLanguage={draft.source_language} value={draft.cloud_asr_preference} onChange={(value) => update("cloud_asr_preference", value)} />
              </Control>
              <Control label="Preferred cloud translator">
                <ProviderSelect kind="translation" cloudOnly catalog={catalog} sourceLanguage={draft.source_language} value={draft.cloud_translation_preference} onChange={(value) => update("cloud_translation_preference", value)} />
              </Control>
              <p className="settings-helper">Used when the route is Auto and the local model misses its latency target. Audio or transcript upload must be allowed under Privacy before Auto may hand over.</p>
            </SettingsGroup>
            <SettingsGroup title="Credentials" description="Saved in the OS secure store. Secret values never enter EchoLingo settings, history, events, or logs; the inference service receives them only for the duration of a session.">
              {!catalog && <p className="settings-helper">Loading the provider catalog…</p>}
              {credentialError && <p className="settings-error" role="alert">{credentialError}</p>}
              <div className="provider-cards">
                {catalog?.credential_groups.map((group) => (
                  <CredentialGroupCard
                    key={group.id}
                    group={group}
                    providers={providersForGroup(catalog, group.id)}
                    status={credentialStatus[group.id] ?? null}
                    transcriptUploadAllowed={draft.privacy.transcript_upload_allowed}
                    sessionActive={sessionActive}
                    onStatus={(next) => setCredentialStatus((current) => ({ ...current, [group.id]: next }))}
                    onRuntime={setRuntime}
                  />
                ))}
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
            <SettingsGroup title="Logs" description="Shell and inference-service logs for support requests. Credentials are never written to them.">
              <Control label="Logs directory">
                <code className="logs-path" tabIndex={0}>{logsDirectory ?? "Resolving…"}</code>
              </Control>
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

const sourceLabels: Record<CredentialSource, string> = {
  keychain: "Stored in the secure store",
  environment: "Using development environment variables",
  none: "Not configured",
};

/** One credential group (vendor account) with its secret fields, non-secret
 *  settings and a connection test. Typed values live only in component
 *  state until Save; the shell reports availability, never the values. */
function CredentialGroupCard({
  group,
  onRuntime,
  onStatus,
  providers,
  sessionActive,
  status,
  transcriptUploadAllowed,
}: {
  group: CredentialGroup;
  onRuntime: (runtime: RuntimePreferences) => void;
  onStatus: (status: CredentialGroupStatus) => void;
  providers: { asr?: ProviderSpec; translation?: ProviderSpec };
  sessionActive: boolean;
  status: CredentialGroupStatus | null;
  transcriptUploadAllowed: boolean;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [settingsDraft, setSettingsDraft] = useState<Record<string, string>>({});
  const [pending, setPending] = useState<"save" | "clear" | "test" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<CloudProbeResult | null>(null);

  const fieldStatus = (key: string) => status?.fields.find((field) => field.key === key);
  const available = (key: string) => fieldStatus(key)?.available ?? false;
  const configured = group.fields.filter((field) => field.required).every((field) => available(field.key));
  const anyAvailable = group.fields.some((field) => available(field.key));
  const source: CredentialSource = status?.fields.some((field) => field.available && field.source === "keychain")
    ? "keychain"
    : status?.fields.some((field) => field.available && field.source === "environment")
      ? "environment"
      : "none";
  const storedSetting = (key: string) =>
    status?.settings[key] ?? group.settings.find((setting) => setting.key === key)?.default ?? "";
  const effectiveSetting = (key: string) => settingsDraft[key] ?? storedSetting(key);
  const settingsChanged = group.settings.some((setting) => effectiveSetting(setting.key) !== storedSetting(setting.key));

  const typed: Record<string, string> = {};
  for (const field of group.fields) {
    const value = (values[field.key] ?? "").trim();
    if (value.length > 0) typed[field.key] = value;
  }
  const typedCount = Object.keys(typed).length;
  const tooShort = group.fields.find((field) => typed[field.key] !== undefined && typed[field.key].length < field.min_len);
  const missingRequired = group.fields.find((field) => field.required && !available(field.key) && typed[field.key] === undefined);
  const canSave = !pending && !tooShort && (typedCount > 0 ? !missingRequired : settingsChanged);

  const asrId = providers.asr?.id ?? null;
  const translationId = transcriptUploadAllowed ? providers.translation?.id ?? null : null;
  const canTest = !pending && !sessionActive && configured && Boolean(asrId || translationId);
  const testHint = sessionActive
    ? "Stop the current session before testing."
    : !configured
      ? "Save the required credentials first."
      : !asrId && !translationId
        ? "Enable transcript upload under Privacy to test translation."
        : undefined;

  const save = async () => {
    setPending("save");
    setError(null);
    try {
      let next = status;
      if (typedCount > 0) {
        next = await api.setCredentials(group.id, typed);
        setValues({});
        setProbe(null);
      }
      if (settingsChanged) {
        const settings = Object.fromEntries(group.settings.map((setting) => [setting.key, effectiveSetting(setting.key)]));
        const runtime = await api.updateProviderSettings(group.id, settings);
        onRuntime(runtime);
        next = {
          group_id: group.id,
          fields: next?.fields ?? group.fields.map((field) => ({ key: field.key, available: false, source: "none" as const })),
          settings: { ...(next?.settings ?? {}), ...settings, ...(runtime.providers[group.id] ?? {}) },
        };
        setSettingsDraft({});
        setProbe(null);
      }
      if (next) onStatus(next);
    } catch (failure) {
      setError(String(failure));
    } finally {
      setPending(null);
    }
  };

  const clear = async () => {
    setPending("clear");
    setError(null);
    try {
      onStatus(await api.clearCredentials(group.id));
      setValues({});
      setProbe(null);
    } catch (failure) {
      setError(String(failure));
    } finally {
      setPending(null);
    }
  };

  const test = async () => {
    setPending("test");
    setError(null);
    setProbe(null);
    try {
      setProbe(await api.probeCloud(asrId, translationId));
    } catch (failure) {
      setError(String(failure));
    } finally {
      setPending(null);
    }
  };

  return (
    <article className="provider-card" aria-label={group.display_name}>
      <header className="provider-card-header">
        <div className="provider-card-copy">
          <strong>{group.display_name}</strong>
          <small>
            {group.vendor}
            {group.free_tier_note ? ` · ${group.free_tier_note}` : ""}
            {group.docs_url && (
              <>
                {" "}
                <a href={group.docs_url} target="_blank" rel="noreferrer">
                  Get a key
                  <ArrowSquareOut size={11} weight="bold" aria-hidden="true" />
                </a>
              </>
            )}
          </small>
          <span className="provider-card-roles">
            {providers.asr && <span>Recognition · {providers.asr.display_name}</span>}
            {providers.translation && <span>Translation · {providers.translation.display_name}</span>}
          </span>
        </div>
        <span className={anyAvailable ? "provider-badge provider-badge--ok" : configured ? "provider-badge provider-badge--neutral" : "provider-badge"} role="status">
          {anyAvailable ? <CheckCircle size={15} weight="fill" aria-hidden="true" /> : configured ? <Key size={15} weight="regular" aria-hidden="true" /> : <Warning size={15} weight="fill" aria-hidden="true" />}
          {anyAvailable ? sourceLabels[source] : !status ? "Checking…" : configured ? "Key optional" : "Not configured"}
        </span>
      </header>
      <div className="provider-card-grid">
        {group.fields.map((field) => (
          <Control key={field.key} label={field.label}>
            <input
              type={field.secret ? "password" : "text"}
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              value={values[field.key] ?? ""}
              placeholder={available(field.key) ? "Saved — enter to replace" : `Enter ${field.label}`}
              onChange={(event) => setValues((current) => ({ ...current, [field.key]: event.target.value }))}
            />
          </Control>
        ))}
        {group.settings.map((setting) => (
          <Control key={setting.key} label={setting.label}>
            {setting.kind === "select" ? (
              <select value={effectiveSetting(setting.key)} onChange={(event) => setSettingsDraft((current) => ({ ...current, [setting.key]: event.target.value }))}>
                {setting.options.map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={effectiveSetting(setting.key)}
                placeholder={setting.placeholder}
                onChange={(event) => setSettingsDraft((current) => ({ ...current, [setting.key]: event.target.value }))}
              />
            )}
          </Control>
        ))}
      </div>
      <p className="settings-helper">
        {providers.translation
          ? `Test sends one fixed English sentence to ${group.vendor} when transcript upload is enabled; recognition tests never upload audio.`
          : "Test completes the authenticated handshake only and never uploads audio."}
      </p>
      {tooShort && <p className="settings-error" role="alert">{tooShort.label} must be at least {tooShort.min_len} characters.</p>}
      {error && <p className="settings-error" role="alert">{error}</p>}
      {probe && <ProbeResult group={group} probe={probe} translationTested={Boolean(translationId)} translationAvailable={Boolean(providers.translation)} />}
      <div className="settings-inline">
        <button className="button button--primary" type="button" disabled={!canSave} onClick={() => void save()}>
          <Key size={18} weight="regular" aria-hidden="true" />
          {pending === "save" ? "Saving…" : "Save"}
        </button>
        <button className="button" type="button" disabled={Boolean(pending) || !anyAvailable || source !== "keychain"} onClick={() => void clear()}>
          {pending === "clear" ? "Clearing…" : "Clear"}
        </button>
        <button className="button" type="button" disabled={!canTest} title={testHint} onClick={() => void test()}>
          <PlugsConnected size={18} weight="regular" aria-hidden="true" />
          {pending === "test" ? "Testing…" : "Test"}
        </button>
      </div>
    </article>
  );
}

function ProbeResult({
  group,
  probe,
  translationAvailable,
  translationTested,
}: {
  group: CredentialGroup;
  probe: CloudProbeResult;
  translationAvailable: boolean;
  translationTested: boolean;
}) {
  const showRegion = group.id === "dashscope" && (probe.region || probe.host || probe.workspace_scoped != null);
  return (
    <div className={probe.ok ? "cloud-probe cloud-probe--ok" : "cloud-probe cloud-probe--error"} role="status">
      <strong>{probe.ok ? `${group.display_name} connection works` : `${group.display_name} needs attention`}</strong>
      {probe.asr.status !== "skipped" && <span>Recognition: {roleSummary(probe.asr, "handshake")}</span>}
      {probe.translation.status !== "skipped" && <span>Translation: {roleSummary(probe.translation)}</span>}
      {showRegion && (
        <span>
          Endpoint: {[probe.region, probe.host, probe.workspace_scoped == null ? null : probe.workspace_scoped ? "workspace-scoped" : "default workspace"]
            .filter(Boolean)
            .join(" · ")}
        </span>
      )}
      {probe.message && <small>{probe.message}{probe.code ? ` (${probe.code})` : ""}</small>}
      <small>
        Audio uploaded: {probe.audio_uploaded ? "yes" : "no"}
        {translationTested
          ? ` · One fixed sentence was sent to ${group.vendor}.`
          : translationAvailable
            ? " · Enable transcript upload to include translation in this test."
            : ""}
      </small>
    </div>
  );
}

function roleSummary(result: CloudProbeAsrResult | CloudProbeTranslationResult, latencySuffix = ""): string {
  const latency =
    "handshake_latency_ms" in result
      ? result.handshake_latency_ms
      : "latency_ms" in result
        ? result.latency_ms
        : undefined;
  const parts = [result.status.replace("_", " ")];
  if (result.status === "connected") {
    if (result.model) parts.push(result.model);
    if (latency != null) parts.push(`${latency.toFixed(0)} ms${latencySuffix ? ` ${latencySuffix}` : ""}`);
  }
  return parts.join(" · ");
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
