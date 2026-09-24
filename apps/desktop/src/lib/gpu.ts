import type { GpuAccelerationStatus, ModelProgress } from "../types";

/** `model_id` of the GPU pack on the model progress channel. */
export const GPU_PACK_PROGRESS_ID = "gpu-pack";

export type GpuPackAction = "download" | "update" | "retry";

/** What the Settings → Models GPU card shows for one status. */
export interface GpuCardView {
  /** The GPU name, or what is missing. */
  title: string;
  /** One line under the title. */
  detail: string;
  /** The primary button, or null when there is nothing to download. */
  action: GpuPackAction | null;
  canRemove: boolean;
  /** The pack is installed, so the "Use GPU acceleration" switch applies. */
  showToggle: boolean;
  /** How the switch currently affects recognition and translation. */
  usage: string | null;
  /** A problem worth a warning: CPU fallback, failed self-test, old driver. */
  notice: { title: string; detail: string } | null;
  /** Why every control is disabled, or null when they can be used. */
  lockedReason: string | null;
}

export function formatGigabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

/** A shell message as a sentence of its own. */
function sentence(text: string): string {
  const trimmed = text.trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

const actionForState: Record<GpuAccelerationStatus["pack_state"], GpuPackAction | null> = {
  not_installed: "download",
  installing: "download",
  ready: null,
  update_required: "update",
  corrupt: "retry",
};

/** The card for a status; null where GPU acceleration does not exist
 *  (macOS). `installing` is true while this window runs the download. */
export function gpuCardView(
  status: GpuAccelerationStatus,
  { installing = false, sessionActive = false }: { installing?: boolean; sessionActive?: boolean } = {},
): GpuCardView | null {
  if (!status.supported_platform) return null;
  const installed = status.pack_state !== "not_installed" && status.pack_state !== "installing";
  const busy = installing || status.pack_state === "installing";
  const size = status.download_bytes ? `${formatGigabytes(status.download_bytes)} download` : "about 2–3 GB download";
  const gpu = status.gpu;
  const title = gpu?.name ?? "No NVIDIA GPU detected";

  let detail: string;
  if (status.pack_state === "ready") {
    detail = [
      status.pack_version ? `Pack ${status.pack_version}` : "Installed",
      status.installed_bytes ? formatGigabytes(status.installed_bytes) : null,
      status.device_name ? `CUDA on ${status.device_name}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
  } else if (status.pack_state === "update_required") {
    detail = `The installed pack${status.pack_version ? ` (${status.pack_version})` : ""} was made for another version of EchoLingo.`;
  } else if (status.pack_state === "corrupt") {
    detail = "The installed pack is incomplete or damaged.";
  } else if (!status.eligible) {
    const reason = status.ineligible_reason?.trim();
    const repeatsTitle = reason?.replace(/\.$/, "").toLowerCase() === title.toLowerCase();
    detail = reason && !repeatsTitle ? reason : "GPU acceleration needs an NVIDIA GeForce RTX 20 or GTX 16 series card or newer.";
  } else {
    detail = [gpu ? `Driver ${gpu.driver_version}` : null, gpu ? `compute capability ${gpu.compute_capability}` : null, size]
      .filter(Boolean)
      .join(" · ");
  }

  // Downloading again cannot help a GPU the pack does not support, so an
  // ineligible GPU gets no Download, Update or Retry; the notice says why.
  // An installed pack can still be removed.
  const action = status.eligible || busy ? actionForState[status.pack_state] : null;

  // At most one warning; while one shows, the usage line stays empty so the
  // card never says both "in use" and "not usable".
  let notice: GpuCardView["notice"] = null;
  if (installed && !status.eligible) {
    notice = {
      title: "This GPU cannot use the pack",
      detail: status.ineligible_reason ?? "GPU acceleration needs an NVIDIA GeForce RTX 20 or GTX 16 series card or newer.",
    };
  } else if (status.fallback_reason) {
    notice = {
      title: "Running on the CPU for now",
      detail: `The GPU runtime failed to start, so recognition and translation went back to the CPU. ${sentence(status.fallback_reason)} To try the GPU again, turn Use GPU acceleration off and on, or remove the pack and download it again.`,
    };
  } else if (status.pack_state === "ready" && status.cuda_available === false) {
    notice = {
      title: "CUDA is not usable",
      detail: "The pack's self-test found no usable CUDA device, so speech recognition stays on the CPU. Update the NVIDIA driver, then remove the pack and download it again.",
    };
  }

  let usage: string | null = null;
  if (status.pack_state === "ready" && !notice) {
    // Translation runs on the pack's Vulkan runtime, which a shell may
    // report separately from CUDA.
    const translationOnCpu = status.llama_gpu_ok === false;
    usage = !status.enabled
      ? "Off: recognition and translation run on the CPU."
      : translationOnCpu
        ? `${status.active ? "In use" : "Used the next time the local models load"}: recognition runs on the GPU; translation stays on the CPU because the pack's Vulkan runtime did not pass its check.`
        : status.active
          ? "In use: recognition and translation run on the GPU."
          : "Used the next time the local models load.";
  } else if (!installed && !busy && !status.eligible) {
    usage = "Recognition and translation run on the CPU. If they fall behind the lecture, a cloud recognizer under Cloud providers keeps up on any computer.";
  }

  return {
    title,
    detail,
    action,
    canRemove: installed,
    showToggle: status.pack_state === "ready",
    usage,
    notice,
    lockedReason: sessionActive ? "Stop the current session to change GPU acceleration." : null,
  };
}

const phaseLabels: Record<string, string> = {
  starting: "Starting",
  downloading: "Downloading",
  verifying: "Verifying",
  extracting: "Extracting",
  testing: "Testing",
};

/** How long an install may go without a progress event before the card
 *  asks the shell for the status itself. */
export const GPU_STATUS_POLL_MS = 4000;

/** What a new progress event asks of the card: the last phases reload the
 *  status, and a failure is shown with the shell's message. */
export function gpuProgressOutcome(progress: ModelProgress): { reload: boolean; error: string | null } {
  if (progress.phase === "ready") return { reload: true, error: null };
  if (progress.phase === "failed") {
    return { reload: true, error: progress.message?.trim() || "The GPU acceleration pack could not be installed." };
  }
  return { reload: false, error: null };
}

/** True while the shell reports an install whose progress has not reached
 *  this card for a while: it mounted mid-install, or the events stopped. */
export function gpuInstallQuiet(
  packState: GpuAccelerationStatus["pack_state"] | undefined,
  lastProgressAt: number,
  now: number,
): boolean {
  return packState === "installing" && now - lastProgressAt >= GPU_STATUS_POLL_MS;
}

/** Progress bar width and caption for the pack download. Only the download
 *  itself reports bytes; the later phases show their name alone. The last
 *  event of an earlier attempt (`ready` or `failed`) reads as a new start. */
export function gpuProgress(progress: ModelProgress | undefined): { percentage: number; label: string } {
  if (progress && (progress.phase === "ready" || progress.phase === "failed")) progress = undefined;
  const phase = progress?.phase ?? "starting";
  const label = phaseLabels[phase] ?? phase;
  if (!progress || progress.total_bytes <= 0) return { percentage: 0, label };
  const percentage = Math.min(100, Math.round((progress.bytes_completed / progress.total_bytes) * 100));
  if (phase === "downloading") {
    return {
      percentage,
      label: `${label} · ${formatGigabytes(progress.bytes_completed)} of ${formatGigabytes(progress.total_bytes)}`,
    };
  }
  return { percentage, label: `${label} · ${percentage}%` };
}
