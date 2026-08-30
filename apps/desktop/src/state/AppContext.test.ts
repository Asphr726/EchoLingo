import { describe, expect, it } from "vitest";
import { emptySnapshot, type UiEventEnvelope } from "../types";
import { applyUiEvent } from "./AppContext";

function event(kind: UiEventEnvelope["kind"], payload: unknown): UiEventEnvelope {
  return {
    schema_version: 1,
    sequence: 1,
    session_id: null,
    kind,
    emitted_at_unix_ms: 0,
    payload,
  };
}

describe("canonical desktop event projection", () => {
  it("projects metrics without changing the lifecycle revision", () => {
    const current = { ...emptySnapshot, state_revision: 7 };
    const next = applyUiEvent(
      current,
      event("metrics", { input_rms_dbfs: -34.2, vad_probability: 0.72 }),
    );
    expect(next.state_revision).toBe(7);
    expect(next.metrics.input_rms_dbfs).toBe(-34.2);
    expect(next.metrics.vad_probability).toBe(0.72);
  });

  it("keeps partial and committed transcript regions separate", () => {
    const partial = applyUiEvent(
      emptySnapshot,
      event("transcript_revision", {
        kind: "partial",
        revision_id: 3,
        unstable_text: "real time",
      }),
    );
    const stable = applyUiEvent(
      partial,
      event("transcript_revision", {
        kind: "stable",
        revision_id: 4,
        text: "real-time interpretation",
      }),
    );
    expect(partial.live.original_unstable).toBe("real time");
    expect(stable.live.original_committed).toBe("real-time interpretation");
    expect(stable.live.original_unstable).toBe("");
  });

  it("projects local model startup progress from backend health events", () => {
    const loading = applyUiEvent(
      { ...emptySnapshot, phase: "STARTING" },
      event("backend_health", { service: "qwen_asr", state: "starting" }),
    );
    const ready = applyUiEvent(
      loading,
      event("backend_health", { service: "qwen_asr", state: "connected" }),
    );
    expect(loading.startup_status).toContain("1–2 minutes");
    expect(ready.startup_status).toBe("Qwen3-ASR is ready.");
  });
});
