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

  it("keeps open, unstable and committed transcript tiers separate", () => {
    const partial = applyUiEvent(
      emptySnapshot,
      event("transcript_revision", {
        kind: "partial",
        revision_id: 3,
        stable_text: "real time",
        unstable_text: "interpre",
      }),
    );
    const stable = applyUiEvent(
      partial,
      event("transcript_revision", {
        kind: "stable",
        revision_id: 4,
        text: "real-time interpretation.",
        committed_text: "real-time interpretation.",
      }),
    );
    expect(partial.live.open_text).toBe("real time");
    expect(partial.live.original_unstable).toBe("interpre");
    expect(stable.live.original_committed).toBe("real-time interpretation.");
    expect(stable.live.open_text).toBe("");
    expect(stable.live.original_unstable).toBe("");
    // Alignment updates refine stored timings only.
    const aligned = applyUiEvent(
      partial,
      event("transcript_revision", { kind: "alignment_update", revision_id: 2, text: "old" }),
    );
    expect(aligned).toBe(partial);
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

  it("upserts a canonical bilingual segment without duplicating it", () => {
    const source = applyUiEvent(
      emptySnapshot,
      event("segment_committed", {
        id: "segment-4",
        ordinal: 4,
        start_ms: 1000,
        end_ms: 2400,
        original: "Good morning.",
        translation: "",
      }),
    );
    const translated = applyUiEvent(
      source,
      event("segment_committed", {
        id: "segment-4",
        ordinal: 4,
        start_ms: 1000,
        end_ms: 2400,
        original: "Good morning.",
        translation: "早上好。",
      }),
    );

    expect(translated.previous_segments).toHaveLength(1);
    expect(translated.previous_segments[0].translation).toBe("早上好。");
  });

  it("routes provisional translations to the live tail and committed spans to rows", () => {
    const withSource = applyUiEvent(
      emptySnapshot,
      event("transcript_revision", { kind: "partial", revision_id: 4, unstable_text: "new source" }),
    );
    const translated = applyUiEvent(
      withSource,
      event("translation_revision", {
        kind: "partial",
        revision_id: 9,
        source_revision_id: 4,
        source_committed: false,
        editable_text: "临时译文",
      }),
    );
    expect(translated.live.translation_source_revision_id).toBe(4);
    expect(translated.live.translation_editable).toBe("临时译文");

    // A newer partial keeps the provisional translation until the unit closes.
    const newerSource = applyUiEvent(
      translated,
      event("transcript_revision", { kind: "partial", revision_id: 5, unstable_text: "new source words" }),
    );
    expect(newerSource.live.translation_editable).toBe("临时译文");

    // Streaming deltas of a committed span never reach the live tail.
    const committedDelta = applyUiEvent(
      newerSource,
      event("translation_revision", {
        kind: "partial",
        revision_id: 2,
        source_revision_id: 3,
        source_committed: true,
        text: "已提交",
        editable_text: "",
      }),
    );
    expect(committedDelta.live.translation_editable).toBe("临时译文");

    const stable = applyUiEvent(
      committedDelta,
      event("transcript_revision", {
        kind: "stable",
        revision_id: 6,
        text: "new source committed.",
        committed_text: "new source committed.",
      }),
    );
    expect(stable.live.translation_editable).toBe("");

    // A late provisional event with no live source text is dropped.
    const late = applyUiEvent(
      stable,
      event("translation_revision", {
        kind: "stable",
        revision_id: 10,
        source_revision_id: 5,
        source_committed: false,
        editable_text: "迟到",
      }),
    );
    expect(late.live.translation_editable).toBe("");

    // A committed final carries the cumulative committed target.
    const finalCommit = applyUiEvent(
      late,
      event("translation_revision", {
        kind: "final",
        revision_id: 3,
        source_revision_id: 6,
        source_committed: true,
        text: "新的源已提交。",
        committed_text: "新的源已提交。",
      }),
    );
    expect(finalCommit.live.translation_committed).toBe("新的源已提交。");
    // Errors never change live text.
    const errored = applyUiEvent(
      finalCommit,
      event("translation_revision", { kind: "error", source_revision_id: 7, source_committed: false, text: "timeout" }),
    );
    expect(errored).toBe(finalCommit);
  });

  it("ignores live events from another session", () => {
    const current = { ...emptySnapshot, session_id: "session-a" };
    const foreign: UiEventEnvelope = {
      ...event("transcript_revision", { kind: "partial", revision_id: 1, unstable_text: "other" }),
      session_id: "session-b",
    };
    expect(applyUiEvent(current, foreign)).toBe(current);
    const own: UiEventEnvelope = { ...foreign, session_id: "session-a" };
    expect(applyUiEvent(current, own).live.original_unstable).toBe("other");
  });
});
