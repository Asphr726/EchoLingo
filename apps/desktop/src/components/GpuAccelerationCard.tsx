import { Warning } from "@phosphor-icons/react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/bridge";
import { gpuCardView, gpuProgress } from "../lib/gpu";
import type { GpuAccelerationStatus, ModelProgress } from "../types";

type GpuAction = "install" | "remove" | "toggle";

/** GPU pack status plus its actions. The status reloads when a download
 *  reports ready and when a session starts or stops, since the runtime may
 *  fall back to the CPU while one starts. A shell without the commands
 *  leaves the status null, so nothing is shown. */
export function useGpuAcceleration(progress: ModelProgress | undefined, sessionActive: boolean) {
  const [status, setStatus] = useState<GpuAccelerationStatus | null>(null);
  const [pending, setPending] = useState<GpuAction | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const phase = progress?.phase;
  useEffect(() => {
    if (phase === "ready") void reload();
  }, [phase, reload]);

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
  // A finished earlier download must not show as the start of a new one.
  const bar = gpuProgress(progress?.phase === "ready" ? undefined : progress);
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
