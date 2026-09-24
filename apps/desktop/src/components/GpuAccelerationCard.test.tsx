import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { GpuAccelerationStatus, ModelProgress } from "../types";
import { GpuAccelerationCard } from "./GpuAccelerationCard";

const eligible: GpuAccelerationStatus = {
  supported_platform: true,
  gpu: { name: "NVIDIA GeForce RTX 3060", compute_capability: "8.6", driver_version: "581.15" },
  eligible: true,
  ineligible_reason: null,
  pack_state: "not_installed",
  pack_version: null,
  download_bytes: 2_684_354_560,
  installed_bytes: null,
  cuda_available: null,
  device_name: null,
  enabled: true,
  active: false,
  fallback_reason: null,
};
const installed: GpuAccelerationStatus = {
  ...eligible,
  pack_state: "ready",
  pack_version: "0.2.0",
  installed_bytes: 5_368_709_120,
  cuda_available: true,
  device_name: "NVIDIA GeForce RTX 3060",
  active: true,
};

function render(
  status: GpuAccelerationStatus,
  {
    error = null,
    pending = null,
    progress,
    sessionActive = false,
  }: { error?: string | null; pending?: "install" | "remove" | "toggle" | null; progress?: ModelProgress; sessionActive?: boolean } = {},
) {
  return renderToStaticMarkup(
    <GpuAccelerationCard
      status={status}
      progress={progress}
      pending={pending}
      error={error}
      sessionActive={sessionActive}
      onInstall={() => undefined}
      onRemove={() => undefined}
      onToggle={() => undefined}
    />,
  ).replace(/<!-- -->/g, "");
}

describe("GpuAccelerationCard", () => {
  it("renders nothing on macOS", () => {
    expect(render({ ...eligible, supported_platform: false })).toBe("");
  });

  it("offers the download on an eligible GPU", () => {
    const html = render(eligible);
    expect(html).toContain("<strong>NVIDIA GeForce RTX 3060</strong>");
    expect(html).toContain("2.5 GB download");
    expect(html).toContain('<button class="button button--primary" type="button">Download</button>');
    expect(html).not.toContain("Use GPU acceleration");
  });

  it("shows progress while the pack installs", () => {
    const total = 2_684_354_560;
    const html = render(
      { ...eligible, pack_state: "installing" },
      { progress: { model_id: "gpu-pack", phase: "downloading", bytes_completed: total / 2, total_bytes: total, bytes_per_second: null } },
    );
    expect(html).toContain('aria-label="GPU acceleration pack 50%"');
    expect(html).toContain("Downloading · 1.3 GB of 2.5 GB");
    expect(html).toContain("disabled");
    expect(html).toContain("Installing…");
  });

  it("shows the switch and Remove once installed", () => {
    const html = render(installed);
    expect(html).toContain("CUDA on NVIDIA GeForce RTX 3060");
    expect(html).toContain('<input type="checkbox" checked=""/><span>Use GPU acceleration</span>');
    expect(html).toContain(">Remove</button>");
    expect(html).toContain("In use: recognition and translation run on the GPU.");
  });

  it("disables everything during a session and says why", () => {
    const html = render(installed, { sessionActive: true });
    expect(html).toContain('<input type="checkbox" disabled="" checked=""/>');
    expect(html).toContain("Stop the current session to change GPU acceleration.");
  });

  it("warns after a fallback to the CPU", () => {
    const html = render({ ...installed, active: false, fallback_reason: "CUDA error: out of memory" });
    expect(html).toContain('class="settings-note settings-note--warning" role="status"');
    expect(html).toContain("CUDA error: out of memory");
  });

  it("shows a failed install and offers the download again", () => {
    const html = render(eligible, {
      error: "Part 2 of 2 failed its SHA-256 check; download it again.",
      progress: { model_id: "gpu-pack", phase: "failed", bytes_completed: 0, total_bytes: 0, bytes_per_second: null, message: "x" },
    });
    expect(html).toContain('<p class="settings-error" role="alert">Part 2 of 2 failed its SHA-256 check; download it again.</p>');
    expect(html).toContain('<button class="button button--primary" type="button">Download</button>');
    expect(html).not.toContain("model-progress");
  });

  it("asks for an update of an outdated pack", () => {
    const html = render({ ...installed, pack_state: "update_required", pack_version: "0.1.9" });
    expect(html).toContain(">Update</button>");
    expect(html).toContain(">Remove</button>");
    expect(html).not.toContain("Use GPU acceleration");
  });
});
