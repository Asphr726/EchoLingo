import { Warning } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/bridge";
import { GPU_STATUS_POLL_MS, gpuCardView, gpuInstallQuiet, gpuProgress, gpuProgressOutcome } from "../lib/gpu";
import type { GpuAccelerationStatus, ModelProgress } from "../types";

type GpuAction = "install" | "remove" | "toggle";

/** GPU pack status plus its actions. The status reloads when a download
 *  reports ready or failed and when a session starts or stops, since the
 *  runtime may fall back to the CPU while one starts. A shell without the
 *  commands leaves the status null, so nothing is shown. */
export function useGpuAcceleration(progress: ModelProgress | undefined, sessionActive: boolean) {
  const [status, setStatus] = useState<GpuAccelerationStatus | null>(null);
  const [pending, setPending] = useState<GpuAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastProgressAt = useRef(0);
  // The last event from before this card mounted is history, not news.
  const seenProgress = useRef(progress);

  const reload = useCallback(async () => {
    try {
      setStatus(await api.gpuAccelerationStatus());
    } catch {
      // Older shells have no GPU pack; the card stays hidden.
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload, sessionActive]);

  // Every event is a new object, so a failure right after another one
  // still counts.
  useEffect(() => {
    if (!progress || progress === seenProgress.current) return;
    seenProgress.current = progress;
    lastProgressAt.current = Date.now();
    const outcome = gpuProgressOutcome(progress);
    if (outcome.error) {
      setError(outcome.error);
      // The install command may still be settling; the buttons come back now.
      setPending((current) => (current === "install" ? null : current));
    }
    if (outcome.reload) void reload();
  }, [progress, reload]);

  // An install this window did not start (or one whose events stopped) must
  // not leave the card on "Installing…": ask for the status while it lasts.
  const shellInstalling = status?.pack_state === "installing";
  useEffect(() => {
    if (!shellInstalling) return;
    const timer = window.setInterval(() => {
      if (gpuInstallQuiet("installing", lastProgressAt.current, Date.now())) void reload();
    }, GPU_STATUS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [reload, shellInstalling]);

  const run = async (action: GpuAction, call: () => Promise<GpuAccelerationStatus>) => {
    setPending(action);
    setError(null);
    try {
      setStatus(await call());
    } catch (failure) {
      setError(String(failure));
      await reload();
    } finally {
      setPending(null);
    }
  };

  return {
    status,
    pending,
    error,
    install: () => run("install", api.installGpuPack),
    remove: () => run("remove", api.removeGpuPack),
    setEnabled: (enabled: boolean) => run("toggle", () => api.setGpuAcceleration(enabled)),
  };
}

const actionLabels = { download: "Download", update: "Update", retry: "Retry" } as const;

/** Settings → Models: the NVIDIA GPU acceleration pack. Renders nothing
 *  where GPU acceleration does not exist (macOS). */
export function GpuAccelerationCard({
  error,
  onInstall,
  onRemove,
  onToggle,
  pending,
  progress,
  sessionActive,
  status,
}: {
  status: GpuAccelerationStatus;
  progress?: ModelProgress;
  pending: GpuAction | null;
  error: string | null;
  sessionActive: boolean;
  onInstall: () => void;
  onRemove: () => void;
  onToggle: (enabled: boolean) => void;
}) {
  const installing = pending === "install" || status.pack_state === "installing";
  const view = gpuCardView(status, { installing, sessionActive });
  if (!view) return null;
  const locked = Boolean(view.lockedReason) || pending !== null || installing;
  const bar = gpuProgress(progress);
  return (
    <>
      <div className="model-list">
        <article className="model-row gpu-row">
          <div className="model-row-copy">
            <span>NVIDIA GPU</span>
            <strong>{view.title}</strong>
            <small>{view.detail}</small>
          </div>
          <div className="model-row-actions">
            {installing && (
              <div className="model-progress" aria-label={`GPU acceleration pack ${bar.percentage}%`}>
                <span style={{ width: `${bar.percentage}%` }} />
                <small>{bar.label}</small>
              </div>
            )}
            {view.action && (
              <button className="button button--primary" type="button" disabled={locked} title={view.lockedReason ?? undefined} onClick={onInstall}>
                {installing ? "Installing…" : actionLabels[view.action]}
              </button>
            )}
            {view.canRemove && (
              <button className="button" type="button" disabled={locked} title={view.lockedReason ?? undefined} onClick={onRemove}>
                {pending === "remove" ? "Removing…" : "Remove"}
              </button>
            )}
          </div>
        </article>
      </div>
      {view.showToggle && (
        <label className="check-row check-row--settings">
          <input
            type="checkbox"
            checked={status.enabled}
            disabled={locked}
            onChange={(event) => onToggle(event.target.checked)}
          />
          <span>Use GPU acceleration</span>
        </label>
      )}
      {view.usage && <p className="settings-helper">{view.usage}</p>}
      {view.notice && (
        <div className="settings-note settings-note--warning" role="status">
          <Warning size={19} weight="regular" aria-hidden="true" />
          <p>
            <strong>{view.notice.title}</strong> {view.notice.detail}
          </p>
        </div>
      )}
      {view.lockedReason && <p className="settings-helper">{view.lockedReason}</p>}
      {error && <p className="settings-error" role="alert">{error}</p>}
    </>
  );
}
