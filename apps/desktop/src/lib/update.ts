import type { SessionPhase, UpdateStatus } from "../types";

/** Element id of Settings → General → Updates; the top-bar pill focuses it. */
export const UPDATES_ANCHOR = "settings-updates";

/** How often the status is asked for while something the shell reports
 *  (an alignment, an assistant job) holds the install back; those end
 *  without an update event. */
export const UPDATE_STATUS_POLL_MS = 5000;

/** The shell refuses to install during a session; the UI says so before. */
export const SESSION_BLOCK_REASON = "Stop the current session before updating EchoLingo.";

export function isSessionActive(phase: SessionPhase): boolean {
  return phase !== "IDLE" && phase !== "COMPLETED";
}

/** What the Updates group offers for an available update. */
export type UpdateAction = "install" | "download";

export interface UpdateView {
  checking: boolean;
  /** Downloading or installing: the progress replaces the action. */
  busy: boolean;
  /** Install in place, the release-page download, or nothing. */
  action: UpdateAction | null;
  /** Why Install is disabled, or null when it can be used. */
  blockedReason: string | null;
}

export function updateView(status: UpdateStatus, { sessionActive }: { sessionActive: boolean }): UpdateView {
  const busy = status.state === "downloading" || status.state === "installing";
  const action: UpdateAction | null = !status.available || busy ? null : status.in_place_supported ? "install" : "download";
  const blockedReason = action === "install" ? (sessionActive ? SESSION_BLOCK_REASON : status.install_blocked_reason) : null;
  return { checking: status.state === "checking", busy, action, blockedReason };
}

/** "Last checked 5 minutes ago", from the shell's RFC 3339 time. */
export function formatCheckedAt(checkedAt: string | null, now: number = Date.now()): string {
  if (!checkedAt) return "Not checked yet";
  const at = Date.parse(checkedAt);
  if (!Number.isFinite(at)) return "Not checked yet";
  const minutes = Math.floor(Math.max(0, now - at) / 60_000);
  if (minutes < 1) return "Last checked just now";
  if (minutes < 60) return `Last checked ${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Last checked ${hours} hour${hours === 1 ? "" : "s"} ago`;
  return `Last checked on ${new Date(at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}`;
}

/** A release date for people: "September 22, 2026". */
export function formatReleaseDate(date: string | null): string | null {
  if (!date) return null;
  const at = Date.parse(date);
  if (!Number.isFinite(at)) return null;
  return new Date(at).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

function formatMegabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Bar width and caption of the download. Without a known length the bar
 *  stays empty and the caption counts what has arrived. */
export function updateProgress(status: UpdateStatus): { percentage: number; label: string; announcement: string } {
  if (status.state === "installing") {
    return {
      percentage: 100,
      label: "Installing",
      announcement: "Installing the update. EchoLingo restarts when it is done.",
    };
  }
  const downloaded = status.progress?.downloaded_bytes ?? 0;
  const total = status.progress?.total_bytes ?? null;
  if (!total || total <= 0) {
    return { percentage: 0, label: `Downloading · ${formatMegabytes(downloaded)}`, announcement: "Downloading the update." };
  }
  const percentage = Math.min(100, Math.round((downloaded / total) * 100));
  return {
    percentage,
    label: `Downloading · ${formatMegabytes(downloaded)} of ${formatMegabytes(total)}`,
    // Tens only, so a screen reader is not flooded by every event.
    announcement: `Downloading the update: ${Math.floor(percentage / 10) * 10}%.`,
  };
}

export function versionLabel(version: string): string {
  return `v${version.trim().replace(/^v/i, "")}`;
}

/** Text of the top-bar pill, or null when it is hidden: no update, or a
 *  session runs. */
export function updatePillLabel(status: UpdateStatus | null, sessionActive: boolean): string | null {
  if (!status?.available || sessionActive) return null;
  const version = versionLabel(status.available.version);
  return status.state === "downloading" || status.state === "installing" ? `Updating to ${version}…` : `Update available · ${version}`;
}
