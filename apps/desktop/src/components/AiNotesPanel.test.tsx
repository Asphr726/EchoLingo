import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { AssistantJob, AssistantStatus, SegmentRecord, SessionNotes, SessionNotesState, SessionRecord } from "../types";
import { AiNotesPanel } from "./AiNotesPanel";

const session = {
  id: "s1",
  title: "en → zh lecture",
  title_source: "default",
  source_language: "en",
  target_language: "zh",
} as SessionRecord;

const segments = [
  { id: "r1", start_ms: 0, end_ms: 5000, source_text: "Look at that line and think about it." },
  { id: "r2", start_ms: 860_000, end_ms: 866_000, source_text: "Julesz asked which textures pop out." },
] as SegmentRecord[];

const ready: AssistantStatus = {
  configured: true,
  consent: true,
  provider_group: "dashscope",
  model: "qwen-plus",
  display_name: "Qwen (Alibaba Model Studio)",
  key_available: true,
  auto_title: true,
};

const notes: SessionNotes = {
  session_id: "s1",
  markdown: "# Texture perception\n\nSummary.\n\n## Pop-out (14:20–28:45)\n\n- Julesz (?)",
  language: "zh",
  provider: "dashscope",
  model: "qwen-plus",
  attachments: [{ id: "a", name: "slides.pdf", pages: 12, chars: 4000, truncated: false, warning: null }],
  usage: { prompt_tokens: 21_406, completion_tokens: 2_874 },
  prompt_version: "notes.v1",
  source_chars: 5000,
  created_at: "2026-09-22T10:00:00Z",
  updated_at: "2026-09-22T10:00:00Z",
};

const running: AssistantJob = {
  job_id: "j1",
  session_id: "s1",
  task: "notes",
  state: "running",
  stage: "section",
  progress: { index: 3, total: 6, start_ms: 860_000, end_ms: 1_725_000 },
  text: "# Texture perception\n\n## Pop-out (14:20–28:45)\n\nPartial",
};

function render(state: SessionNotesState, assistant: AssistantStatus | null = ready) {
  return renderMarkup(state, assistant).replace(/<!-- -->/g, "");
}

function renderMarkup(state: SessionNotesState, assistant: AssistantStatus | null) {
  return renderToStaticMarkup(
    <AiNotesPanel
      session={session}
      segments={segments}
      assistant={assistant}
      state={state}
      stateLoading={false}
      onJob={() => undefined}
      onJump={() => undefined}
      focusSection={null}
      onFocusHandled={() => undefined}
    />,
  );
}

describe("AiNotesPanel states", () => {
  it("asks for setup with the specific missing piece", () => {
    expect(render({ notes: null, job: null }, { ...ready, key_available: false })).toContain("Add a key for Qwen (Alibaba Model Studio)");
    expect(render({ notes: null, job: null }, { ...ready, consent: false })).toContain("Open AI assistant settings");
    expect(render({ notes: null, job: null }, { ...ready, provider_group: "", configured: false })).toContain("Set up the AI assistant");
  });

  it("offers to create notes and says what will be sent", () => {
    const html = render({ notes: null, job: null });
    expect(html).toContain("Turn this session into study notes");
    expect(html).toContain("~14 words will be sent to Qwen (Alibaba Model Studio) · qwen-plus");
    expect(html).toContain("Create notes");
    expect(html).toContain("Add course materials?");
  });

  it("streams with a stage line, a progress rule and Cancel", () => {
    const html = render({ notes: null, job: running });
    expect(html).toContain("Writing section 3 of 6 · 14:20–28:45");
    expect(html).toContain('role="progressbar"');
    expect(html).toContain(">Cancel</button>");
    expect(html).toContain("Partial");
    expect(html).not.toContain("Regenerate");
  });

  it("shows saved notes with provenance, contents and actions", () => {
    const html = render({ notes, job: null });
    expect(html).toContain('<h3 class="notes-title" lang="zh">Texture perception</h3>');
    expect(html).toContain("21,406 tokens in, 2,874 out");
    expect(html).toContain("slides.pdf");
    expect(html).toContain("Contents");
    expect(html).toContain("Export .md");
    expect(html).toContain('class="uncertain"');
  });

  it("keeps previous notes visible when a new run fails", () => {
    const failed: AssistantJob = { ...running, state: "failed", error: { code: "authentication_failed", message: "HTTP 401" } };
    const html = render({ notes, job: failed });
    expect(html).toContain("Your previous notes are unchanged.");
    expect(html).toContain("Check the key in Settings → Cloud providers.");
    expect(html).toContain("Open Cloud providers");
    expect(html).toContain("Summary.");
    expect(html).toContain("Try again");
  });

  it("does not offer a retry that the missing setup would refuse", () => {
    const failed: AssistantJob = { ...running, state: "failed", error: { code: "privacy_policy_denied", message: "consent" } };
    const html = render({ notes: null, job: failed }, { ...ready, consent: false });
    expect(html).toContain("Allow transcripts to go to Qwen (Alibaba Model Studio)");
    expect(html).toContain("Enable consent in Settings → AI assistant.");
    expect(html).not.toContain("Try again");
  });

  it("tolerates malformed stored provenance", () => {
    const damaged = { ...notes, attachments: {} as unknown as SessionNotes["attachments"], usage: null as unknown as SessionNotes["usage"] };
    expect(() => render({ notes: damaged, job: null })).not.toThrow();
  });
});
