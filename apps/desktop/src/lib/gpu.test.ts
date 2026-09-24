import { describe, expect, it } from "vitest";
import type { GpuAccelerationStatus } from "../types";
import { GPU_STATUS_POLL_MS, gpuCardView, gpuInstallQuiet, gpuProgress, gpuProgressOutcome } from "./gpu";

const unsupported: GpuAccelerationStatus = {
  supported_platform: false,
  gpu: null,
  eligible: false,
  ineligible_reason: null,
  pack_state: "not_installed",
  pack_version: null,
  download_bytes: null,
  installed_bytes: null,
  cuda_available: null,
  device_name: null,
  enabled: true,
  active: false,
  fallback_reason: null,
};
const rtx = { name: "NVIDIA GeForce RTX 3060", compute_capability: "8.6", driver_version: "581.15" };
const eligible: GpuAccelerationStatus = {
  ...unsupported,
  supported_platform: true,
  gpu: rtx,
  eligible: true,
  download_bytes: 2_684_354_560,
};
const installed: GpuAccelerationStatus = {
  ...eligible,
  pack_state: "ready",
  pack_version: "0.2.0",
  installed_bytes: 5_368_709_120,
  cuda_available: true,
  device_name: rtx.name,
  active: true,
};

describe("GPU acceleration card", () => {
  it("is hidden where the platform has no GPU pack", () => {
    expect(gpuCardView(unsupported)).toBeNull();
  });

  it("offers the download on an eligible GPU with its size", () => {
    const view = gpuCardView(eligible)!;
    expect(view.title).toBe("NVIDIA GeForce RTX 3060");
    expect(view.detail).toBe("Driver 581.15 · compute capability 8.6 · 2.5 GB download");
    expect(view.action).toBe("download");
    expect(view.canRemove).toBe(false);
    expect(view.showToggle).toBe(false);
    expect(view.lockedReason).toBeNull();
  });

  it("estimates the size before the manifest is known", () => {
    expect(gpuCardView({ ...eligible, download_bytes: null })!.detail).toContain("about 2–3 GB download");
  });

  it("explains why an ineligible GPU cannot be used and offers nothing", () => {
    const view = gpuCardView({
      ...unsupported,
      supported_platform: true,
      gpu: { name: "NVIDIA GeForce GTX 1060", compute_capability: "6.1", driver_version: "581.15" },
      ineligible_reason: "Compute capability 6.1 is older than 7.5.",
    })!;
    expect(view.title).toBe("NVIDIA GeForce GTX 1060");
    expect(view.detail).toBe("Compute capability 6.1 is older than 7.5.");
    expect(view.action).toBeNull();
    expect(view.usage).toMatch(/cloud recognizer/);
    const none = gpuCardView({ ...unsupported, supported_platform: true, ineligible_reason: "No NVIDIA GPU detected." })!;
    expect(none.title).toBe("No NVIDIA GPU detected");
    expect(none.detail).toMatch(/RTX 20 or GTX 16 series/);
    expect(none.action).toBeNull();
  });

  it("shows the installed pack with its switch and whether it is in use", () => {
    const view = gpuCardView(installed)!;
    expect(view.detail).toBe("Pack 0.2.0 · 5.0 GB · CUDA on NVIDIA GeForce RTX 3060");
    expect(view.action).toBeNull();
    expect(view.canRemove).toBe(true);
    expect(view.showToggle).toBe(true);
    expect(view.usage).toMatch(/^In use/);
    expect(gpuCardView({ ...installed, active: false })!.usage).toMatch(/next time the local models load/);
    expect(gpuCardView({ ...installed, enabled: false, active: false })!.usage).toMatch(/^Off/);
    expect(view.notice).toBeNull();
  });

  it("asks for an update or a retry of a pack that cannot be used", () => {
    const update = gpuCardView({ ...eligible, pack_state: "update_required", pack_version: "0.1.9" })!;
    expect(update.action).toBe("update");
    expect(update.detail).toContain("(0.1.9) was made for another version");
    expect(update.canRemove).toBe(true);
    const corrupt = gpuCardView({ ...eligible, pack_state: "corrupt" })!;
    expect(corrupt.action).toBe("retry");
    expect(corrupt.canRemove).toBe(true);
  });

  it("offers no Update or Retry to a GPU the pack does not support, only its reason", () => {
    const reason = "Driver 551.86 is older than 580.65";
    for (const pack_state of ["update_required", "corrupt"] as const) {
      const view = gpuCardView({ ...eligible, eligible: false, ineligible_reason: reason, pack_state })!;
      expect(view.action).toBeNull();
      expect(view.canRemove).toBe(true);
      expect(view.notice).toEqual({ title: "This GPU cannot use the pack", detail: reason });
    }
    const ready = gpuCardView({ ...installed, eligible: false, ineligible_reason: reason })!;
    expect(ready.notice?.detail).toBe(reason);
    expect(ready.usage).toBeNull();
  });

  it("explains a fallback to the CPU and how to try the GPU again", () => {
    const fallback = gpuCardView({ ...installed, active: false, fallback_reason: "CUDA error: out of memory" })!;
    expect(fallback.notice?.title).toBe("Running on the CPU for now");
    expect(fallback.notice?.detail).toContain("CUDA error: out of memory. To try the GPU again, turn Use GPU acceleration off and on");
    expect(fallback.notice?.detail).not.toMatch(/restart/i);
    expect(fallback.usage).toBeNull();
  });

  it("says only that CUDA is unusable when the self-test found no device", () => {
    const view = gpuCardView({ ...installed, active: false, cuda_available: false, device_name: null })!;
    expect(view.notice?.title).toBe("CUDA is not usable");
    expect(view.usage).toBeNull();
    expect(view.detail).not.toContain("CUDA on");
  });

  it("tells when translation stays on the CPU although recognition uses the GPU", () => {
    expect(gpuCardView({ ...installed, llama_gpu_ok: false })!.usage).toMatch(/^In use: recognition runs on the GPU; translation stays on the CPU/);
    expect(gpuCardView({ ...installed, llama_gpu_ok: true })!.usage).toBe("In use: recognition and translation run on the GPU.");
    expect(gpuCardView({ ...installed, llama_gpu_ok: null })!.usage).toBe("In use: recognition and translation run on the GPU.");
  });

  it("locks every control while a session runs", () => {
    expect(gpuCardView(installed, { sessionActive: true })!.lockedReason).toMatch(/Stop the current session/);
  });
});

describe("GPU pack progress", () => {
  it("shows bytes while downloading and the phase afterwards", () => {
    expect(gpuProgress(undefined)).toEqual({ percentage: 0, label: "Starting" });
    const total = 2_684_354_560;
    expect(
      gpuProgress({ model_id: "gpu-pack", phase: "downloading", bytes_completed: total / 4, total_bytes: total, bytes_per_second: null }),
    ).toEqual({ percentage: 25, label: "Downloading · 0.6 GB of 2.5 GB" });
    expect(
      gpuProgress({ model_id: "gpu-pack", phase: "extracting", bytes_completed: total, total_bytes: total, bytes_per_second: null }),
    ).toEqual({ percentage: 100, label: "Extracting · 100%" });
    expect(gpuProgress({ model_id: "gpu-pack", phase: "testing", bytes_completed: 0, total_bytes: 0, bytes_per_second: null }).label).toBe(
      "Testing",
    );
  });

  it("starts over after an earlier attempt ended", () => {
    for (const phase of ["ready", "failed"]) {
      expect(gpuProgress({ model_id: "gpu-pack", phase, bytes_completed: 5, total_bytes: 10, bytes_per_second: null })).toEqual({
        percentage: 0,
        label: "Starting",
      });
    }
  });

  it("reloads the status when an install ends and shows why it failed", () => {
    const event = (phase: string, message?: string | null) => ({
      model_id: "gpu-pack",
      phase,
      bytes_completed: 0,
      total_bytes: 0,
      bytes_per_second: null,
      message,
    });
    expect(gpuProgressOutcome(event("downloading"))).toEqual({ reload: false, error: null });
    expect(gpuProgressOutcome(event("ready"))).toEqual({ reload: true, error: null });
    expect(gpuProgressOutcome(event("failed", "Part 2 failed its SHA-256 check"))).toEqual({
      reload: true,
      error: "Part 2 failed its SHA-256 check",
    });
    expect(gpuProgressOutcome(event("failed", null)).error).toBe("The GPU acceleration pack could not be installed.");
  });

  it("polls an install only after its events have gone quiet", () => {
    const now = 1_000_000;
    expect(gpuInstallQuiet("installing", 0, now)).toBe(true);
    expect(gpuInstallQuiet("installing", now - GPU_STATUS_POLL_MS + 1, now)).toBe(false);
    expect(gpuInstallQuiet("installing", now - GPU_STATUS_POLL_MS, now)).toBe(true);
    expect(gpuInstallQuiet("ready", 0, now)).toBe(false);
    expect(gpuInstallQuiet(undefined, 0, now)).toBe(false);
  });
});
