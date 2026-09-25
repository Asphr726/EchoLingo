import { describe, expect, it } from "vitest";
import type { AssistantStatus, CredentialGroup, SegmentRecord, SessionNotes } from "../types";
import {
  assistantPresets,
  credentialNeed,
  errorGuidance,
  formatBytes,
  formatTokens,
  groupHasKey,
  notesView,
  progressFraction,
  segmentAtOrAfter,
  setupNeed,
  stageLabel,
  titleNotice,
  visibleMarkdown,
} from "./notes";

const ready: AssistantStatus = {
  configured: true,
  consent: true,
  provider_group: "dashscope",
  model: "qwen-plus",
  display_name: "Qwen (Alibaba Model Studio)",
  key_available: true,
  auto_title: true,
};

const notes = { markdown: "# Title\n\nBody", session_id: "s" } as SessionNotes;
const running = { job_id: "j", session_id: "s", task: "notes" as const, state: "running" as const, text: "# Ti" };

describe("notes panel state", () => {
  it("names what setup is missing, in the order a user fixes it", () => {
    expect(setupNeed(null)).toBeNull();
    expect(setupNeed({ ...ready, provider_group: "", configured: false })).toBe("provider");
    expect(setupNeed({ ...ready, key_available: false })).toBe("key");
    expect(setupNeed({ ...ready, consent: false })).toBe("consent");
    expect(setupNeed({ ...ready, configured: false })).toBe("provider");
    // A saved group the catalog no longer offers: the shell reports neither
    // a provider nor a key, and the fix is choosing a provider.
    expect(
      setupNeed({ ...ready, configured: false, key_available: false, display_name: "", model: "" }),
    ).toBe("provider");
    expect(setupNeed(ready)).toBeNull();
  });

  it("prefers a running job, then saved notes, then setup or the empty state", () => {
    expect(notesView(ready, { notes, job: running }, false)).toBe("generating");
    expect(notesView({ ...ready, consent: false }, { notes, job: null }, false)).toBe("saved");
    expect(notesView(ready, { notes: null, job: null }, true)).toBe("loading");
    expect(notesView({ ...ready, key_available: false }, { notes: null, job: null }, false)).toBe("setup");
    expect(notesView(ready, { notes: null, job: null }, false)).toBe("empty");
    const failed = { ...running, state: "failed" as const, text: "" };
    expect(notesView(ready, { notes: null, job: failed }, false)).toBe("empty");
    expect(notesView(ready, { notes, job: failed }, false)).toBe("saved");
  });

  it("shows running text, else saved notes, else a completed job until it is saved", () => {
    expect(visibleMarkdown({ notes, job: running })).toBe("# Ti");
    expect(visibleMarkdown({ notes, job: null })).toBe(notes.markdown);
    expect(visibleMarkdown({ notes: null, job: { ...running, state: "completed", text: "done" } })).toBe("done");
    expect(visibleMarkdown({ notes: null, job: { ...running, state: "cancelled" } })).toBeNull();
  });
});

describe("stage copy", () => {
  it("describes sections with their time range", () => {
    expect(stageLabel({ stage: "section", progress: { index: 3, total: 6, start_ms: 860_000, end_ms: 1_725_000 } })).toBe(
      "Writing section 3 of 6 · 14:20–28:45",
    );
    expect(stageLabel({ stage: "section", progress: { index: 1, total: 1 } })).toBe("Writing notes");
    expect(stageLabel({ stage: "extracting", progress: null })).toBe("Reading course materials…");
    expect(stageLabel({ stage: null, progress: null })).toBe("Starting…");
  });

  it("reports determinate progress only for multi-section runs", () => {
    expect(progressFraction({ stage: "section", progress: { index: 2, total: 4 } })).toBeCloseTo(0.375);
    expect(progressFraction({ stage: "section", progress: { index: 1, total: 1 } })).toBeNull();
    expect(progressFraction({ stage: "reading", progress: null })).toBeNull();
  });
});

describe("error guidance", () => {
  it("points each actionable code at the setting that fixes it", () => {
    expect(errorGuidance("authentication_failed")).toMatchObject({ section: "cloud-providers" });
    expect(errorGuidance("privacy_policy_denied")).toMatchObject({ section: "ai-assistant" });
    expect(errorGuidance("context_too_long").text).toContain("larger context");
    expect(errorGuidance("rate_limited").section).toBeUndefined();
    expect(errorGuidance("something_new").text).toBe("The notes could not be created.");
  });
});

describe("title notice", () => {
  const titleJob = { job_id: "t", session_id: "s", task: "title" as const, state: "running" as const };
  const failed = {
    ...titleJob,
    state: "failed" as const,
    error: { code: "empty_response", message: "DeepSeek spent its reply budget on reasoning and returned no title." },
  };

  it("shows why a title job of the open session failed", () => {
    expect(titleNotice(failed, "s")).toBe("DeepSeek spent its reply budget on reasoning and returned no title.");
    expect(titleNotice({ ...failed, error: null }, "s")).toBe("The AI title could not be created.");
  });

  it("clears when a title job starts, succeeds or is cancelled", () => {
    expect(titleNotice(titleJob, "s")).toBeNull();
    expect(titleNotice({ ...titleJob, state: "completed" }, "s")).toBeNull();
    expect(titleNotice({ ...titleJob, state: "cancelled" }, "s")).toBeNull();
  });

  it("ignores other sessions and other tasks", () => {
    expect(titleNotice(failed, "other")).toBeUndefined();
    expect(titleNotice(failed, null)).toBeUndefined();
    expect(titleNotice({ ...failed, session_id: null }, "s")).toBeUndefined();
    expect(titleNotice({ ...failed, task: "notes" }, "s")).toBeUndefined();
  });
});

describe("time chip target", () => {
  const rows = [0, 250_000, 520_400, 900_000].map(
    (start, index) => ({ id: String(index), start_ms: start, end_ms: start + 5000 }) as SegmentRecord,
  );

  it("jumps to the first row starting at or after the section, within a second", () => {
    expect(segmentAtOrAfter(rows, 520_000)?.id).toBe("2");
    expect(segmentAtOrAfter(rows, 521_000)?.id).toBe("2");
    expect(segmentAtOrAfter(rows, 600_000)?.id).toBe("3");
    expect(segmentAtOrAfter(rows, 5_000_000)?.id).toBe("3");
    expect(segmentAtOrAfter([], 0)).toBeNull();
  });
});

describe("settings helpers", () => {
  const group = {
    id: "dashscope",
    fields: [
      { key: "api_key", required: true },
      { key: "workspace", required: false },
    ],
  } as CredentialGroup;

  it("treats a group as keyed when its required fields are available", () => {
    const status = (available: boolean) => ({
      group_id: "dashscope",
      fields: [{ key: "api_key", available, source: available ? ("keychain" as const) : ("none" as const) }],
      settings: {},
    });
    expect(groupHasKey(group, status(true))).toBe(true);
    expect(groupHasKey(group, status(false))).toBe(false);
    expect(groupHasKey(undefined, status(true))).toBe(false);
  });

  it("mirrors the shell for a custom endpoint: base URL required, key optional", () => {
    const custom = {
      id: "custom_openai",
      fields: [{ key: "api_key", required: false }],
      settings: [{ key: "base_url" }, { key: "chat_model" }],
    } as CredentialGroup;
    const status = (keyAvailable: boolean, baseUrl: string) => ({
      group_id: "custom_openai",
      fields: [{ key: "api_key", available: keyAvailable, source: keyAvailable ? ("keychain" as const) : ("none" as const) }],
      settings: { base_url: baseUrl, chat_model: "" },
    });
    expect(credentialNeed(custom, status(false, "http://127.0.0.1:11434/v1"))).toBeNull();
    expect(groupHasKey(custom, status(false, "http://127.0.0.1:11434/v1"))).toBe(true);
    expect(credentialNeed(custom, status(true, "  "))).toBe("endpoint");
    expect(credentialNeed(group, { group_id: "dashscope", fields: [], settings: {} })).toBe("key");
  });

  it("reads assistant presets from catalogs that may predate them", () => {
    expect(assistantPresets(null)).toEqual([]);
    expect(assistantPresets({ schema_version: 1, asr: [], translation: [], credential_groups: [] })).toEqual([]);
  });

  it("formats sizes and token counts", () => {
    expect(formatBytes(38_400)).toBe("38 KB");
    expect(formatBytes(4_812_300)).toBe("4.6 MB");
    expect(formatTokens({ prompt_tokens: 21_406, completion_tokens: 2_874 })).toBe("21,406 tokens in, 2,874 out");
    expect(formatTokens({})).toBeNull();
  });
});
