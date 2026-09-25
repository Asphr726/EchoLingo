import { ArrowCircleUp, ArrowSquareOut, ArrowsClockwise, DownloadSimple, ShieldCheck } from "@phosphor-icons/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, subscribeUpdateStatus } from "../lib/bridge";
import { parseNotesDocument } from "../lib/markdown";
import { type Platform, platform as currentPlatform } from "../lib/platform";
import {
  formatCheckedAt,
  formatReleaseDate,
  UPDATE_STATUS_POLL_MS,
  updatePillLabel,
  updateProgress,
  updateView,
  versionLabel,
} from "../lib/update";
import type { SessionPhase, UpdateStatus } from "../types";
import { NotesMarkdown } from "./NotesMarkdown";

type UpdateAction = "check" | "install";

/** The shell's update status, kept current by its events, plus the manual
 *  actions. `status` stays null where the shell has no updater (a plain
 *  browser outside the preview); `unavailable` then turns true. */
export function useUpdates(phase: SessionPhase) {
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [pending, setPending] = useState<UpdateAction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      setStatus(await api.getUpdateStatus());
      setUnavailable(false);
    } catch {
      setUnavailable(true);
    }
  }, []);

  useEffect(() => {
    let active = true;
    let dispose: () => void = () => undefined;
    void subscribeUpdateStatus((next) => {
      if (active) setStatus(next);
    }).then((unlisten) => {
      if (active) dispose = unlisten;
      else unlisten();
    });
    return () => {
      active = false;
      dispose();
    };
  }, []);

  // A session starting or stopping changes whether Install is allowed.
  useEffect(() => {
    void reload();
  }, [reload, phase]);

  // What the shell reports as holding the install back (an alignment, an
  // assistant job) ends without an update event.
  const heldBack = Boolean(status?.available && status.install_blocked_reason && status.state !== "downloading" && status.state !== "installing");
  useEffect(() => {
    if (!heldBack) return;
    const timer = window.setInterval(() => void reload(), UPDATE_STATUS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [heldBack, reload]);

  const run = async (action: UpdateAction, call: () => Promise<UpdateStatus>) => {
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
    unavailable,
    pending,
    error,
    check: () => run("check", api.checkForUpdate),
    install: () => run("install", api.installUpdate),
  };
}

/** Settings → General → Updates: the running version, the manual check,
 *  the launch-check preference and, when a newer version exists, its notes
 *  and the install (or, where EchoLingo cannot replace itself, the release
 *  page). */
export function UpdatesCard({
  checkAtLaunch,
  error,
  now,
  onCheck,
  onInstall,
  onToggleCheckAtLaunch,
  pending,
  platform = currentPlatform(),
  sessionActive,
  status,
}: {
  status: UpdateStatus;
  sessionActive: boolean;
  /** The `check_updates_at_launch` runtime preference. */
  checkAtLaunch: boolean;
  pending: UpdateAction | null;
  /** A refused or failed command. */
  error: string | null;
  platform?: Platform;
  /** For "Last checked …"; the current time by default. */
  now?: number;
  onCheck: () => void;
  onInstall: () => void;
  onToggleCheckAtLaunch: (enabled: boolean) => void;
}) {
  const view = updateView(status, { sessionActive });
  const available = status.available;
  const checked = formatCheckedAt(status.last_checked_at, now);
  const summary = view.checking
    ? "Checking for updates…"
    : status.state === "up_to_date"
      ? `Up to date · ${checked}`
      : checked;
  const notes = useMemo(() => (available?.notes ? parseNotesDocument(available.notes).blocks : []), [available?.notes]);
  const released = formatReleaseDate(available?.date ?? null);
  const progress = view.busy ? updateProgress(status) : null;
  const installDisabled = pending !== null || Boolean(view.blockedReason);
  const shownError = error ?? status.error;
  // Next to the install when there is one, else under the check.
  const errorLine = shownError ? <p className="settings-error" role="alert">{shownError}</p> : null;

  return (
    <>
      <div className="update-summary">
        <div className="update-summary-copy">
          <strong>EchoLingo {status.current_version}</strong>
          <small role="status">{summary}</small>
        </div>
        <button className="button" type="button" disabled={view.checking || view.busy || pending !== null} onClick={onCheck}>
          <ArrowsClockwise size={16} weight="regular" aria-hidden="true" />
          {view.checking || pending === "check" ? "Checking…" : "Check for updates"}
        </button>
      </div>
      <label className="check-row check-row--settings">
        <input type="checkbox" checked={checkAtLaunch} onChange={(event) => onToggleCheckAtLaunch(event.target.checked)} />
        <span>Check for updates when EchoLingo starts</span>
      </label>

      {available && (
        <article className="update-card" aria-label={`EchoLingo ${available.version}`}>
          <header className="update-card-header">
            <div className="update-card-copy">
              <span>
                <ArrowCircleUp size={13} weight="fill" aria-hidden="true" />
                New version
              </span>
              <strong>EchoLingo {versionLabel(available.version)}</strong>
              <small>{released ? `Released ${released}` : "Release date not given"}</small>
            </div>
            {view.action === "install" && (
              <button
                className="button button--primary"
                type="button"
                disabled={installDisabled}
                title={view.blockedReason ?? undefined}
                aria-describedby={view.blockedReason ? "update-blocked-reason" : undefined}
                onClick={onInstall}
              >
                <DownloadSimple size={16} weight="regular" aria-hidden="true" />
                {pending === "install" ? "Starting download…" : "Download and install"}
              </button>
            )}
            {view.action === "download" && (
              <a className="button button--primary" href={available.release_url} target="_blank" rel="noreferrer">
                Download
                <ArrowSquareOut size={14} weight="bold" aria-hidden="true" />
              </a>
            )}
          </header>
          {progress && (
            <div className="update-progress">
              <div
                className="model-progress"
                role="progressbar"
                aria-label="Update download"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress.percentage}
              >
                <span style={{ width: `${progress.percentage}%` }} />
                <small>{progress.label}</small>
              </div>
              <p className="visually-hidden" aria-live="polite">{progress.announcement}</p>
            </div>
          )}
          {view.blockedReason && (
            <p className="settings-helper" id="update-blocked-reason">{view.blockedReason}</p>
          )}
          {view.action === "download" && status.unsupported_reason && (
            <p className="settings-helper">{status.unsupported_reason}</p>
          )}
          {errorLine}
          {notes.length > 0 && (
            <section className="update-notes" aria-label={`What's new in EchoLingo ${available.version}`} tabIndex={0}>
              <NotesMarkdown blocks={notes} />
            </section>
          )}
          {platform === "mac" && (
            <div className="settings-note">
              <ShieldCheck size={19} weight="regular" aria-hidden="true" />
              <p>
                After updating, macOS may ask again for microphone and screen &amp; system audio recording permission, and whether
                EchoLingo may use its saved keys in the Keychain. The beta is ad-hoc signed, so macOS sees every new version as a
                new app.
              </p>
            </div>
          )}
        </article>
      )}
      {!available && errorLine}
    </>
  );
}

/** Top-bar pill: a newer version exists. Hidden during a session; opens
 *  Settings → General → Updates. */
export function UpdatePill({
  onOpen,
  sessionActive,
  status,
}: {
  status: UpdateStatus | null;
  sessionActive: boolean;
  onOpen: () => void;
}) {
  const label = updatePillLabel(status, sessionActive);
  if (!label) return null;
  return (
    <button className="update-pill" type="button" onClick={onOpen} title="Open the update in Settings">
      <ArrowCircleUp size={14} weight="fill" aria-hidden="true" />
      {label}
    </button>
  );
}
