import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Platform } from "../lib/platform";
import type { UpdateStatus } from "../types";
import { UpdatePill, UpdatesCard } from "./UpdatesCard";

const now = Date.parse("2026-09-24T12:00:00Z");

const idle: UpdateStatus = {
  current_version: "0.3.0",
  state: "idle",
  available: null,
  last_checked_at: null,
  error: null,
  in_place_supported: true,
  unsupported_reason: null,
  check_at_launch: true,
  progress: null,
  install_blocked_reason: null,
};
const available: UpdateStatus = {
  ...idle,
  state: "available",
  last_checked_at: "2026-09-24T11:48:00Z",
  available: {
    version: "0.3.1",
    notes: "## Fixes\n\n- The caption window keeps its **size**.\n\n<script>alert(1)</script>",
    date: "2026-09-22T12:00:00Z",
    release_url: "https://github.com/Asphr726/EchoLingo/releases/tag/v0.3.1",
  },
};

function render(
  status: UpdateStatus,
  {
    checkAtLaunch = true,
    error = null,
    pending = null,
    platform = "mac",
    sessionActive = false,
  }: { checkAtLaunch?: boolean; error?: string | null; pending?: "check" | "install" | null; platform?: Platform; sessionActive?: boolean } = {},
) {
  return renderToStaticMarkup(
    <UpdatesCard
      status={status}
      sessionActive={sessionActive}
      checkAtLaunch={checkAtLaunch}
      pending={pending}
      error={error}
      platform={platform}
      now={now}
      onCheck={() => undefined}
      onInstall={() => undefined}
      onToggleCheckAtLaunch={() => undefined}
    />,
  ).replace(/<!-- -->/g, "");
}

describe("UpdatesCard", () => {
  it("shows the version, the manual check and the launch preference before any check", () => {
    const html = render(idle, { checkAtLaunch: false });
    expect(html).toContain("<strong>EchoLingo 0.3.0</strong>");
    expect(html).toContain('<small role="status">Not checked yet</small>');
    expect(html).toMatch(/<button class="button" type="button">.*Check for updates<\/button>/);
    expect(html).toContain('<input type="checkbox"/><span>Check for updates when EchoLingo starts</span>');
    expect(html).not.toContain("update-card");
  });

  it("says when the running version is the latest", () => {
    const html = render({ ...idle, state: "up_to_date", last_checked_at: "2026-09-24T11:55:00Z" });
    expect(html).toContain("Up to date · Last checked 5 minutes ago");
    expect(html).toContain('<input type="checkbox" checked=""/>');
    expect(html).not.toContain("Download and install");
  });

  it("offers an available update with its notes and the macOS permission note", () => {
    const html = render(available);
    expect(html).toContain('aria-label="EchoLingo 0.3.1"');
    expect(html).toContain("<strong>EchoLingo v0.3.1</strong>");
    expect(html).toContain("Released September 22, 2026");
    expect(html).toMatch(/<button class="button button--primary" type="button">.*Download and install<\/button>/);
    // Release notes go through the notes renderer: Markdown, never raw HTML.
    expect(html).toContain('class="update-notes"');
    expect(html).toContain("<strong>size</strong>");
    expect(html).not.toContain("<script>");
    expect(html).toContain("macOS may ask again for microphone and screen &amp; system audio recording permission");
    expect(html).toContain("Keychain");
    expect(html).toContain("Last checked 12 minutes ago");
  });

  it("leaves the macOS note out elsewhere", () => {
    expect(render(available, { platform: "windows" })).not.toContain("Keychain");
  });

  it("shows the download progress with an announcement instead of the button", () => {
    const html = render({
      ...available,
      state: "downloading",
      progress: { downloaded_bytes: 26_214_400, total_bytes: 52_428_800 },
      install_blocked_reason: "An update is already being installed.",
    });
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-valuenow="50"');
    expect(html).toContain('style="width:50%"');
    expect(html).toContain("Downloading · 25.0 MB of 50.0 MB");
    expect(html).toContain('<p class="visually-hidden" aria-live="polite">Downloading the update: 50%.</p>');
    expect(html).not.toContain("Download and install");
    expect(html).not.toContain("already being installed");
    // No second check while downloading.
    expect(html).toMatch(/<button class="button" type="button" disabled="">.*Check for updates<\/button>/);
  });

  it("says EchoLingo restarts while installing", () => {
    const html = render({ ...available, state: "installing" });
    expect(html).toContain("Installing the update. EchoLingo restarts when it is done.");
    expect(html).toContain('aria-valuenow="100"');
  });

  it("shows a failed download and offers the install again", () => {
    const html = render({
      ...available,
      state: "error",
      error: "The update could not be downloaded: connection reset",
    });
    expect(html).toContain('<p class="settings-error" role="alert">The update could not be downloaded: connection reset</p>');
    expect(html).toMatch(/<button class="button button--primary" type="button">.*Download and install<\/button>/);
  });

  it("prefers a refused command's message over the status error", () => {
    const html = render({ ...idle, state: "error", error: "Could not check for updates: offline" }, { error: "Stop the current session before updating EchoLingo." });
    expect(html).toContain('role="alert">Stop the current session before updating EchoLingo.</p>');
    expect(html).not.toContain("offline");
  });

  it("links to the release page where EchoLingo cannot replace itself", () => {
    const html = render(
      { ...available, in_place_supported: false, unsupported_reason: "On Linux, install the new .deb package from the release page." },
      { platform: "linux" },
    );
    expect(html).toContain('<a class="button button--primary" href="https://github.com/Asphr726/EchoLingo/releases/tag/v0.3.1" target="_blank" rel="noreferrer">Download');
    expect(html).toContain("On Linux, install the new .deb package from the release page.");
    expect(html).not.toContain("Download and install");
  });

  it("disables the install during a session and says why", () => {
    const html = render(available, { sessionActive: true });
    expect(html).toMatch(
      /<button class="button button--primary" type="button" disabled="" title="Stop the current session before updating EchoLingo." aria-describedby="update-blocked-reason">/,
    );
    expect(html).toContain('<p class="settings-helper" id="update-blocked-reason">Stop the current session before updating EchoLingo.</p>');
  });

  it("disables the install while the shell reports other work", () => {
    const reason = "Word timings of the last session are still being aligned; update in a minute.";
    const html = render({ ...available, install_blocked_reason: reason });
    expect(html).toContain(`disabled="" title="${reason}"`);
    expect(html).toContain(`id="update-blocked-reason">${reason}</p>`);
  });

  it("says a check is running", () => {
    const html = render({ ...idle, state: "checking" });
    expect(html).toContain('<small role="status">Checking for updates…</small>');
    expect(html).toMatch(/disabled="">.*Checking…<\/button>/);
  });
});

describe("UpdatePill", () => {
  const pill = (status: UpdateStatus | null, sessionActive = false) =>
    renderToStaticMarkup(<UpdatePill status={status} sessionActive={sessionActive} onOpen={() => undefined} />).replace(/<!-- -->/g, "");

  it("names the new version", () => {
    const html = pill(available);
    expect(html).toMatch(/^<button class="update-pill" type="button" title="Open the update in Settings">/);
    expect(html).toContain("Update available · v0.3.1</button>");
  });

  it("is hidden while recording and without an update", () => {
    expect(pill(available, true)).toBe("");
    expect(pill({ ...idle, state: "up_to_date" })).toBe("");
    expect(pill(null)).toBe("");
  });

  it("follows a running install", () => {
    expect(pill({ ...available, state: "downloading" })).toContain("Updating to v0.3.1…");
  });
});
