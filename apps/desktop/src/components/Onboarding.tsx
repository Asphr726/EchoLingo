import {
  ArrowLeft,
  ArrowRight,
  Check,
  CloudCheck,
  Headphones,
  LockKey,
  Microphone,
  Play,
  ShieldCheck,
  Waveform,
} from "@phosphor-icons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { systemAudioAvailable, systemAudioDevice } from "../lib/audio";
import { api } from "../lib/bridge";
import { type Platform, platform } from "../lib/platform";
import { useApp } from "../state/AppContext";
import type { AudioDevice, AudioPermissionStatus, AudioTestResult, InferenceMode, StartSessionRequest } from "../types";

const steps = ["Welcome", "Languages", "Audio", "Inference", "Permissions", "Audio test", "Ready"];
const languages = [["en", "English"], ["zh", "Chinese"], ["ja", "Japanese"], ["ko", "Korean"]];

export function Onboarding() {
  const { actionPending, completeOnboarding, devices, draft, error, setDraft } = useApp();
  const [step, setStep] = useState(0);
  const [permissions, setPermissions] = useState<AudioPermissionStatus | null>(null);
  const [testResult, setTestResult] = useState<AudioTestResult | null>(null);
  const [testPending, setTestPending] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const microphones = devices.filter((device) => device.kind === "microphone");
  const selectedDevice = microphones.find((device) => device.id === draft.audio_device_id);
  const systemAudio = systemAudioDevice(devices);
  const systemAudioOffered = systemAudioAvailable(devices);
  // Windows loopback records the default output device; no picker follows.
  const systemAudioPicker = systemAudio?.requires_picker ?? true;
  const update = <K extends keyof StartSessionRequest>(key: K, value: StartSessionRequest[K]) =>
    setDraft((current) => ({ ...current, [key]: value }));

  useEffect(() => {
    if (step === 4) {
      void api.audioPermissionStatus().then(setPermissions).catch((failure) => setLocalError(String(failure)));
    }
  }, [step]);

  const permissionKind = draft.audio_source === "system_audio" ? "system_audio" : "microphone";
  const selectedPermission = permissions?.[permissionKind];
  const permissionText = permissionCopy(platform(), permissionKind, systemAudio);
  const canContinue = useMemo(() => {
    if (step === 1) return draft.source_language !== draft.target_language;
    if (step === 2) return draft.audio_source !== "microphone" || Boolean(draft.audio_device_id);
    if (step === 4) return selectedPermission !== "denied";
    return true;
  }, [draft, selectedPermission, step]);

  const requestPermission = async () => {
    setLocalError(null);
    try {
      await api.requestAudioPermission(permissionKind);
      setPermissions(await api.audioPermissionStatus());
    } catch (failure) {
      setLocalError(String(failure));
    }
  };

  const runAudioTest = async () => {
    setTestPending(true);
    setTestResult(null);
    setLocalError(null);
    try {
      setTestResult(await api.testAudioInput(draft.audio_source, draft.audio_device_id));
    } catch (failure) {
      setLocalError(String(failure));
    } finally {
      setTestPending(false);
    }
  };

  return (
    <main className="onboarding-shell">
      <aside className="onboarding-progress" aria-label="Setup progress">
        <div className="onboarding-brand"><span>EL</span><strong>EchoLingo</strong></div>
        <ol>
          {steps.map((label, index) => (
            <li className={index === step ? "onboarding-step onboarding-step--active" : index < step ? "onboarding-step onboarding-step--done" : "onboarding-step"} key={label}>
              <span>{index < step ? <Check size={12} weight="bold" /> : index + 1}</span>{label}
            </li>
          ))}
        </ol>
        <p>Far-field first · Audio stays local unless you explicitly allow upload.</p>
      </aside>
      <section className="onboarding-stage">
        <div className="onboarding-copy">
          {step === 0 && <Intro />}
          {step === 1 && (
            <Step title="Choose your languages" description="These become the default for new lecture sessions.">
              <div className="onboarding-grid">
                <Field label="Source language"><select value={draft.source_language} onChange={(event) => update("source_language", event.target.value)}>{languages.map(([code, label]) => <option key={code} value={code} disabled={code === draft.target_language}>{label}</option>)}</select></Field>
                <Field label="Target language"><select value={draft.target_language} onChange={(event) => update("target_language", event.target.value)}>{languages.map(([code, label]) => <option key={code} value={code} disabled={code === draft.source_language}>{label}</option>)}</select></Field>
              </div>
            </Step>
          )}
          {step === 2 && (
            <Step title="Choose an audio source" description="Lecture mode keeps distant speech and treats VAD as an annotation, never a hard gate.">
              <div className="choice-grid">
                <Choice active={draft.audio_source === "microphone"} icon={<Microphone size={23} />} title="Microphone" detail="Room, lecture hall, or external array microphone" onClick={() => update("audio_source", "microphone")} />
                <Choice active={draft.audio_source === "system_audio"} disabled={!systemAudioOffered} icon={<Headphones size={23} />} title="System audio" detail={!systemAudioOffered ? "Not available on this platform yet" : systemAudioPicker ? "Browser, local video, or meeting playback" : "What the default output device plays"} onClick={() => update("audio_source", "system_audio")} />
              </div>
              {draft.audio_source === "microphone" && <Field label="Input device"><select value={draft.audio_device_id ?? ""} onChange={(event) => update("audio_device_id", event.target.value || null)}>{microphones.length === 0 ? <option value="">No microphone detected</option> : microphones.map((device) => <option key={device.id} value={device.id}>{device.name}{device.is_default ? " — Default" : ""}</option>)}</select></Field>}
              <Field label="Audio profile"><select value={draft.audio_profile} onChange={(event) => update("audio_profile", event.target.value)}><option value="lecture">Lecture / far-field</option><option value="conversation">Near-field conversation</option><option value="raw">Raw diagnostics</option></select></Field>
            </Step>
          )}
          {step === 3 && (
            <Step title="Select inference mode" description="Auto chooses ASR and translation independently from calibration, privacy, network, credentials, and installed models.">
              <div className="choice-grid choice-grid--three">{(["auto", "local", "cloud"] as InferenceMode[]).map((mode) => <Choice key={mode} active={draft.inference_mode === mode} icon={mode === "auto" ? <Waveform size={23} /> : mode === "local" ? <ShieldCheck size={23} /> : <CloudCheck size={23} />} title={mode[0].toUpperCase() + mode.slice(1)} detail={mode === "auto" ? "Recommended per-device route" : mode === "local" ? "No inference data upload" : "Best quality without a local accelerator"} onClick={() => update("inference_mode", mode)} />)}</div>
              <div className="privacy-summary"><LockKey size={19} /><p>Auto never grants upload permission. You can review audio and transcript upload separately in Settings.</p></div>
            </Step>
          )}
          {step === 4 && (
            <Step title="Grant audio permission" description={permissionText.description}>
              <div className="permission-card"><div><strong>{permissionKind === "microphone" ? "Microphone" : "System audio"}</strong><span>Status: {selectedPermission?.replace("_", " ") ?? "checking"}</span></div><button className="button button--primary" type="button" disabled={selectedPermission === "granted" || selectedPermission === "unavailable"} onClick={() => void requestPermission()}>{selectedPermission === "granted" ? <><Check size={17} />Granted</> : selectedPermission === "unavailable" ? "Not available" : "Request permission"}</button></div>
              {selectedPermission === "denied" && <p className="field-error">{permissionText.denied}</p>}
            </Step>
          )}
          {step === 5 && (
            <Step title="Test your audio" description={draft.audio_source === "microphone" ? `Speak from your expected lecture position using ${selectedDevice?.name ?? "the selected microphone"}.` : systemAudioPicker ? "Choose a display or application, play audio, then wait for the meter result." : "Play audio on this computer, then wait for the meter result."}>
              <button className="audio-test" type="button" disabled={testPending} onClick={() => void runAudioTest()}><Play size={22} weight="fill" /><span><strong>{testPending ? "Listening for 3 seconds…" : "Run audio test"}</strong><small>No audio is uploaded or retained.</small></span></button>
              {testResult && <div className="test-result"><Check size={19} weight="bold" /><div><strong>Audio received at {testResult.peak_rms_dbfs.toFixed(1)} dBFS</strong><span>{testResult.sample_rate_hz / 1000} kHz · {testResult.channels} channel{testResult.channels === 1 ? "" : "s"}</span></div></div>}
            </Step>
          )}
          {step === 6 && <Ready draft={draft} testResult={testResult} />}
          {(localError || error) && <p className="onboarding-error" role="alert">{localError ?? error}</p>}
        </div>
        <footer className="onboarding-actions">
          <button className="button" type="button" disabled={step === 0 || actionPending} onClick={() => setStep((value) => value - 1)}><ArrowLeft size={17} />Back</button>
          {step < steps.length - 1 ? <button className="button button--primary" type="button" disabled={!canContinue || actionPending} onClick={() => setStep((value) => value + 1)}>Continue<ArrowRight size={17} /></button> : <button className="button button--primary" type="button" disabled={actionPending} onClick={() => void completeOnboarding()}><Check size={17} weight="bold" />Open EchoLingo</button>}
        </footer>
      </section>
    </main>
  );
}

export interface PermissionCopy {
  description: string;
  /** Shown when the OS reports the permission as denied. */
  denied: string;
}

/** What the permission step says on each platform. macOS asks through its
 *  own prompts; Windows only has the privacy switches in Settings; Linux
 *  has no prompt at all. */
export function permissionCopy(current: Platform, kind: "microphone" | "system_audio", systemAudio?: AudioDevice): PermissionCopy {
  if (kind === "system_audio") {
    if (systemAudio && !systemAudio.available) {
      return {
        description: "System audio capture is not available on this platform yet. Choose Microphone instead.",
        denied: "System audio capture is not available on this platform yet.",
      };
    }
    if (systemAudio && !systemAudio.requires_picker) {
      return {
        description: "EchoLingo captures what the default output device plays, so no display or app needs to be chosen. Use the audio test to check that sound arrives.",
        denied: "System audio could not be captured. Check the default output device in your sound settings, then return here.",
      };
    }
  }
  switch (current) {
    case "windows":
      return {
        description:
          "Windows does not show a prompt. EchoLingo can listen as long as Settings → Privacy & security → Microphone allows desktop apps to use the microphone.",
        denied:
          "Microphone access is blocked. Open Settings → Privacy & security → Microphone, turn on microphone access and “Let desktop apps access your microphone”, then return here.",
      };
    case "linux":
      return {
        description:
          "Linux does not ask for microphone permission. If the audio test stays silent, check the input device and its level in your system sound settings.",
        denied: "The microphone could not be opened. Check the input device in your system sound settings, then return here.",
      };
    default:
      return kind === "microphone"
        ? {
            description: "macOS requires microphone permission before EchoLingo can listen.",
            denied: "Permission was denied. Open macOS System Settings → Privacy & Security, enable EchoLingo, then return here.",
          }
        : {
            description: "macOS requires screen/audio capture permission. You still choose a display or app for each capture.",
            denied: "Permission was denied. Open macOS System Settings → Privacy & Security, enable EchoLingo, then return here.",
          };
  }
}

function Intro() { return <div className="onboarding-intro"><div className="intro-symbol"><Waveform size={38} weight="light" /></div><span className="section-kicker">Welcome</span><h1>Understand the lecture,<br />without losing the moment.</h1><p>EchoLingo turns room or system audio into stable original text and rolling translation. Setup takes about two minutes.</p></div>; }
function Step({ children, description, title }: { children: ReactNode; description: string; title: string }) { return <div className="onboarding-section"><span className="section-kicker">First-run setup</span><h1>{title}</h1><p>{description}</p><div className="onboarding-fields">{children}</div></div>; }
function Field({ children, label }: { children: ReactNode; label: string }) { return <label className="field"><span>{label}</span>{children}</label>; }
function Choice({ active, detail, disabled = false, icon, onClick, title }: { active: boolean; detail: string; disabled?: boolean; icon: ReactNode; onClick: () => void; title: string }) { return <button className={active ? "setup-choice setup-choice--active" : "setup-choice"} type="button" disabled={disabled} onClick={onClick}>{icon}<strong>{title}</strong><span>{detail}</span>{active && <Check className="choice-check" size={15} weight="bold" />}</button>; }
function Ready({ draft, testResult }: { draft: StartSessionRequest; testResult: AudioTestResult | null }) { return <div className="onboarding-section"><span className="section-kicker">Ready</span><h1>EchoLingo is ready.</h1><p>Review the defaults below. You can change every item before a session.</p><dl className="ready-summary"><div><dt>Languages</dt><dd>{draft.source_language.toUpperCase()} → {draft.target_language.toUpperCase()}</dd></div><div><dt>Audio</dt><dd>{draft.audio_source.replace("_", " ")} · {draft.audio_profile}</dd></div><div><dt>Inference</dt><dd>{draft.inference_mode}</dd></div><div><dt>Audio test</dt><dd>{testResult ? `${testResult.peak_rms_dbfs.toFixed(1)} dBFS` : "Skipped"}</dd></div></dl></div>; }
