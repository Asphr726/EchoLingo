import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { titleNotice } from "../lib/notes";
import type { AssistantJob } from "../types";
import { TitleFailure } from "./HistoryView";

const reasoning = "DeepSeek spent its reply budget on reasoning and returned no title.";

const job = (patch: Partial<AssistantJob>): AssistantJob => ({
  job_id: "t1",
  session_id: "s1",
  task: "title",
  state: "running",
  ...patch,
});

/** The notice after a run of `assistant_update` events, as History applies them. */
function noticeAfter(updates: AssistantJob[], openSession = "s1") {
  let notice: string | null = null;
  for (const update of updates) {
    const next = titleNotice(update, openSession);
    if (next !== undefined) notice = next;
  }
  return renderToStaticMarkup(<TitleFailure message={notice} />);
}

describe("History title failures", () => {
  it("shows why an automatic title failed as an inline error", () => {
    const html = noticeAfter([job({}), job({ state: "failed", error: { code: "empty_response", message: reasoning } })]);
    expect(html).toBe(`<p class="inline-error title-failure" role="alert">${reasoning}</p>`);
  });

  it("keeps the notice through notes updates and clears it on the next title run", () => {
    const failed = job({ state: "failed", error: { code: "empty_response", message: reasoning } });
    const notes = job({ job_id: "n1", task: "notes", state: "completed" });
    expect(noticeAfter([failed, notes])).toContain(reasoning);
    expect(noticeAfter([failed, job({ job_id: "t2" })])).toBe("");
    expect(noticeAfter([failed, job({ job_id: "t2" }), job({ job_id: "t2", state: "completed" })])).toBe("");
  });

  it("ignores title jobs of other sessions", () => {
    const elsewhere = job({ session_id: "s2", state: "failed", error: { code: "empty_response", message: reasoning } });
    expect(noticeAfter([elsewhere])).toBe("");
  });
});
