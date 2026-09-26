import { describe, expect, it } from "vitest";
import type { UpdateStatus } from "../types";
import {
  formatCheckedAt,
  formatReleaseDate,
  isSessionActive,
  SESSION_BLOCK_REASON,
  updatePillLabel,
  updateProgress,
  updateView,
  versionLabel,
} from "./update";

const status: UpdateStatus = {
  current_version: "0.3.1",
  state: "available",
  available: { version: "0.3.2", notes: null, date: null, release_url: "https://github.com/Asphr726/EchoLingo/releases/tag/v0.3.2" },
  last_checked_at: null,
  error: null,
  in_place_supported: true,
  unsupported_reason: null,
  check_at_launch: true,
  progress: null,
  install_blocked_reason: null,
};

describe("update helpers", () => {
  it("chooses the action and why it is disabled", () => {
    expect(updateView(status, { sessionActive: false })).toEqual({ checking: false, busy: false, action: "install", blockedReason: null });
    expect(updateView(status, { sessionActive: true }).blockedReason).toBe(SESSION_BLOCK_REASON);
    expect(updateView({ ...status, install_blocked_reason: "An AI assistant task is running; update when it has finished." }, { sessionActive: false }).blockedReason).toBe(
      "An AI assistant task is running; update when it has finished.",
    );
    // The release page needs no idle app.
    expect(updateView({ ...status, in_place_supported: false }, { sessionActive: true })).toMatchObject({ action: "download", blockedReason: null });
    expect(updateView({ ...status, state: "downloading" }, { sessionActive: false })).toMatchObject({ busy: true, action: null });
    expect(updateView({ ...status, available: null, state: "up_to_date" }, { sessionActive: false }).action).toBeNull();
  });

  it("counts sessions that are not idle or completed", () => {
    expect(isSessionActive("IDLE")).toBe(false);
    expect(isSessionActive("COMPLETED")).toBe(false);
    for (const phase of ["STARTING", "LISTENING", "PAUSED", "STOPPING"] as const) expect(isSessionActive(phase)).toBe(true);
  });

  it("describes when the last check ran", () => {
    const now = Date.parse("2026-09-24T12:00:00Z");
    expect(formatCheckedAt(null, now)).toBe("Not checked yet");
    expect(formatCheckedAt("not a date", now)).toBe("Not checked yet");
    expect(formatCheckedAt("2026-09-24T11:59:40Z", now)).toBe("Last checked just now");
    expect(formatCheckedAt("2026-09-24T11:59:00Z", now)).toBe("Last checked 1 minute ago");
    expect(formatCheckedAt("2026-09-24T09:00:00Z", now)).toBe("Last checked 3 hours ago");
    expect(formatCheckedAt("2026-09-20T12:00:00Z", now)).toBe("Last checked on Sep 20, 2026");
    expect(formatReleaseDate("2026-09-22T12:00:00Z")).toBe("September 22, 2026");
    expect(formatReleaseDate(null)).toBeNull();
  });

  it("reports download progress in tens for screen readers", () => {
    const downloading: UpdateStatus = { ...status, state: "downloading", progress: { downloaded_bytes: 3_460_300, total_bytes: 10_485_760 } };
    expect(updateProgress(downloading)).toEqual({
      percentage: 33,
      label: "Downloading · 3.3 MB of 10.0 MB",
      announcement: "Downloading the update: 30%.",
    });
    const unknownLength: UpdateStatus = { ...downloading, progress: { downloaded_bytes: 1_048_576, total_bytes: null } };
    expect(updateProgress(unknownLength)).toMatchObject({ percentage: 0, label: "Downloading · 1.0 MB" });
    expect(updateProgress({ ...status, state: "installing" }).percentage).toBe(100);
  });

  it("labels the pill", () => {
    expect(versionLabel("v0.3.2")).toBe("v0.3.2");
    expect(updatePillLabel(status, false)).toBe("Update available · v0.3.2");
    expect(updatePillLabel(status, true)).toBeNull();
    expect(updatePillLabel({ ...status, state: "installing" }, false)).toBe("Updating to v0.3.2…");
    expect(updatePillLabel(null, false)).toBeNull();
  });
});
