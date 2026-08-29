import { ClosedCaptioning, CloudArrowUp, LockKey, SlidersHorizontal } from "@phosphor-icons/react";
import { type ReactNode, useState } from "react";
import { api } from "../lib/bridge";
import { useApp } from "../state/AppContext";
import type { CaptionDisplay, InferenceMode, StartSessionRequest } from "../types";

const sections = [
  "General",
  "Audio",
  "Languages",
  "Inference",
  "Models",
  "Privacy",
  "Translation",
  "Advanced",
] as const;
type Section = (typeof sections)[number];

export function SettingsView() {
  const [section, setSection] = useState<Section>("General");
  const { caption, draft, setDraft, snapshot, updateCaption } = useApp();
  const update = <K extends keyof StartSessionRequest>(key: K, value: StartSessionRequest[K]) =>
    setDraft((current) => ({ ...current, [key]: value }));

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
                  <option value="1">1 segment</option>
                  <option value="2">2 segments</option>
                  <option value="3">3 segments</option>
                </select>
              </Control>
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
              <p className="settings-helper">Model downloads and calibration controls will activate when their runtime packages are available.</p>
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
