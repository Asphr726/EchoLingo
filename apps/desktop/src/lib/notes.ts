/** State and copy for the AI notes panel, kept free of React
 *  so every state is unit tested. */

import type {
  AssistantJob,
  AssistantPreset,
  AssistantStatus,
  CredentialGroup,
  CredentialGroupStatus,
  ProviderCatalog,
  SegmentRecord,
  SessionNotesState,
} from "../types";
import { formatRange } from "./markdown";

export type NotesView = "loading" | "setup" | "empty" | "generating" | "saved";

/** What is missing before notes can be created, in the order a user fixes it. */
export type SetupNeed = "provider" | "key" | "consent";

export function setupNeed(status: AssistantStatus | null): SetupNeed | null {
  if (!status) return null;
  // `configured` is false when the saved group is not (or no longer) an
  // assistant preset; the shell then also reports no key, but the fix is to
  // choose a provider, not to add a key.
  if (!status.provider_group || !status.configured) return "provider";
  if (!status.key_available) return "key";
  if (!status.consent) return "consent";
  return null;
}

/** Window event sent after the assistant's preferences change outside the
 *  view showing them (for example consent granted from a dialog). Views that
 *  cache `assistant_status` or the runtime preferences refetch on it. */
export const ASSISTANT_CHANGED_EVENT = "echolingo:assistant-changed";

export function announceAssistantChanged(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(ASSISTANT_CHANGED_EVENT));
}

export function onAssistantChanged(handler: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(ASSISTANT_CHANGED_EVENT, handler);
  return () => window.removeEventListener(ASSISTANT_CHANGED_EVENT, handler);
}

/** Markdown the panel shows: the running job's text, else saved notes, else
 *  a just-completed job's text until the saved copy has been fetched. */
export function visibleMarkdown(state: SessionNotesState): string | null {
  const { job, notes } = state;
  if (job?.state === "running") return job.text ?? "";
  if (notes?.markdown.trim()) return notes.markdown;
  if (job?.state === "completed" && job.text?.trim()) return job.text;
  return null;
}

export function notesView(status: AssistantStatus | null, state: SessionNotesState, loading: boolean): NotesView {
  if (state.job?.state === "running") return "generating";
  if (visibleMarkdown(state) !== null) return "saved";
  if (loading || !status) return "loading";
  return setupNeed(status) ? "setup" : "empty";
}

/** "Writing section 3 of 6 · 14:20–28:45". */
export function stageLabel(job: Pick<AssistantJob, "stage" | "progress">): string {
  const progress = job.progress ?? {};
  const range =
    progress.start_ms != null && progress.end_ms != null && progress.end_ms >= progress.start_ms
      ? formatRange(progress.start_ms, progress.end_ms)
      : null;
  switch (job.stage) {
    case "extracting":
      return "Reading course materials…";
    case "reading":
      return "Reading the transcript…";
    case "section": {
      const total = progress.total ?? 0;
      const index = Math.min(Math.max(progress.index ?? 1, 1), Math.max(total, 1));
      const head = total > 1 ? `Writing section ${index} of ${total}` : "Writing notes";
      return range ? `${head} · ${range}` : head;
    }
    case "writing":
      return range ? `Writing notes · ${range}` : "Writing notes…";
    case "finishing":
      return "Adding the title and key terms…";
    default:
      return "Starting…";
  }
}

/** Fraction done for a determinate progress rule, or null. */
export function progressFraction(job: Pick<AssistantJob, "stage" | "progress">): number | null {
  const { index, total } = job.progress ?? {};
  if (job.stage === "finishing") return 0.96;
  if (job.stage !== "section" || !total || total < 2 || index == null) return null;
  return Math.min(1, Math.max(0, (index - 0.5) / total));
}

export interface ErrorGuidance {
  text: string;
  /** Settings section slug that fixes it. */
  section?: "ai-assistant" | "cloud-providers";
  actionLabel?: string;
}

export function errorGuidance(code: string | null | undefined): ErrorGuidance {
  switch (code) {
    case "authentication_failed":
      return {
        text: "The provider rejected the API key. Check the key in Settings → Cloud providers.",
        section: "cloud-providers",
        actionLabel: "Open Cloud providers",
      };
    case "privacy_policy_denied":
      return {
        text: "Sending transcripts to the AI assistant is turned off. Enable consent in Settings → AI assistant.",
        section: "ai-assistant",
        actionLabel: "Open AI assistant settings",
      };
    case "not_configured":
    case "missing_credentials":
      return {
        text: "The AI assistant is not set up yet. Choose a provider and model in Settings → AI assistant.",
        section: "ai-assistant",
        actionLabel: "Open AI assistant settings",
      };
    case "rate_limited":
      return { text: "The provider’s rate limit or quota was reached. Wait a minute, then try again." };
    case "context_too_long":
      return {
        text: "This session is too long for the selected model. Try a model with a larger context, or attach fewer files.",
        section: "ai-assistant",
        actionLabel: "Choose another model",
      };
    case "content_filtered":
      return { text: "The provider declined this transcript under its content policy." };
    case "network_error":
      return { text: "EchoLingo could not reach the provider. Check the network connection, then try again." };
    case "provider_timeout":
      return { text: "The provider stopped responding. Try again in a moment." };
    case "attachment_error":
    case "attachment_unreadable":
      return { text: "An attached file could not be read. Remove it or try a different export of the file." };
    default:
      return { text: "The notes could not be created." };
  }
}

/** The transcript row a section time chip jumps to: the first row that
 *  starts at or after `startMs` (1 s tolerance for mm:ss rounding), else the
 *  last row. */
export function segmentAtOrAfter(segments: SegmentRecord[], startMs: number): SegmentRecord | null {
  if (segments.length === 0) return null;
  return segments.find((segment) => segment.start_ms >= startMs - 1000) ?? segments[segments.length - 1];
}

// ---------------------------------------------------------------------------
// Settings helpers

export function assistantPresets(catalog: ProviderCatalog | null): AssistantPreset[] {
  return Array.isArray(catalog?.assistant) ? catalog.assistant : [];
}

/** What the group still needs before the assistant can call it, mirroring
 *  the shell's `missing_credential_labels`: every required field available,
 *  and a non-empty endpoint for groups that have a `base_url` setting (a
 *  custom endpoint's key is optional). */
export function credentialNeed(
  group: CredentialGroup | undefined,
  status: CredentialGroupStatus | undefined,
): "key" | "endpoint" | null {
  if (!group || !status) return "key";
  const available = (key: string) => status.fields.some((field) => field.key === key && field.available);
  if (group.fields.some((field) => field.required && !available(field.key))) return "key";
  const hasEndpoint = (group.settings ?? []).some((setting) => setting.key === "base_url");
  if (hasEndpoint && !(status.settings?.base_url ?? "").trim()) return "endpoint";
  return null;
}

/** True when the group has everything the assistant needs to call it. */
export function groupHasKey(
  group: CredentialGroup | undefined,
  status: CredentialGroupStatus | undefined,
): boolean {
  return credentialNeed(group, status) === null;
}

// ---------------------------------------------------------------------------
// Formatting

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatTokens(usage: { prompt_tokens?: number | null; completion_tokens?: number | null }): string | null {
  const input = usage.prompt_tokens;
  const output = usage.completion_tokens;
  if (input == null && output == null) return null;
  const format = (value: number) => value.toLocaleString("en-US");
  if (input != null && output != null) return `${format(input)} tokens in, ${format(output)} out`;
  return input != null ? `${format(input)} tokens in` : `${format(output ?? 0)} tokens out`;
}

export function safeFilename(value: string): string {
  return value.trim().replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/^-+|-+$/g, "") || "echolingo-session";
}
