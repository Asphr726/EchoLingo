import {
  ArrowClockwise,
  CaretDown,
  Check,
  CloudArrowUp,
  Copy,
  DownloadSimple,
  Paperclip,
  Warning,
  WarningCircle,
} from "@phosphor-icons/react";
import { save } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, previewInitialAttachments, previewOpensAttachmentSheet } from "../lib/bridge";
import { estimateWords, formatRange, parseNotesDocument } from "../lib/markdown";
import { requestNavigation } from "../lib/navigation";
import {
  errorGuidance,
  formatTokens,
  notesView,
  progressFraction,
  safeFilename,
  setupNeed,
  stageLabel,
  visibleMarkdown,
} from "../lib/notes";
import type {
  AssistantJob,
  AssistantStatus,
  NoteAttachment,
  NoteAttachmentReport,
  SegmentRecord,
  SessionNotes,
  SessionNotesState,
  SessionRecord,
} from "../types";
import { AttachmentSheet } from "./AttachmentSheet";
import { NotesMarkdown, sectionAnchorId } from "./NotesMarkdown";

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export function scrollToSection(index: number) {
  const heading = document.getElementById(sectionAnchorId(index));
  if (!heading) return;
  heading.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  heading.setAttribute("tabindex", "-1");
  heading.focus({ preventScroll: true });
}

function openSettings(section: string) {
  requestNavigation({ view: "settings", section });
}

export function AiNotesPanel({
  assistant,
  focusSection,
  onFocusHandled,
  onJob,
  onJump,
  segments,
  session,
  state,
  stateLoading,
}: {
  session: SessionRecord;
  segments: SegmentRecord[];
  /** `assistant_status`; null while loading. */
  assistant: AssistantStatus | null;
  state: SessionNotesState;
  stateLoading: boolean;
  /** A notes job was started from this panel. */
  onJob: (job: AssistantJob) => void;
  /** A section time chip was pressed. */
  onJump: (startMs: number, sectionIndex: number | null) => void;
  /** Section to bring into view (returning from the transcript). */
  focusSection: number | null;
  onFocusHandled: () => void;
}) {
  const view = notesView(assistant, state, stateLoading);
  const markdown = visibleMarkdown(state) ?? "";
  const doc = useMemo(() => parseNotesDocument(markdown), [markdown]);
  const [sheetOpen, setSheetOpen] = useState(previewOpensAttachmentSheet);
  const [pending, setPending] = useState<"create" | "cancel" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeSection, setActiveSection] = useState<number | null>(null);
  const copyTimer = useRef<number | undefined>(undefined);
  const job = state.job;
  const running = job?.state === "running";

  useEffect(() => () => window.clearTimeout(copyTimer.current), []);
  useEffect(() => {
    setActionError(null);
    setActiveSection(null);
    if (!previewOpensAttachmentSheet()) setSheetOpen(false);
  }, [session.id]);

  // Scroll spy for the contents rail.
  const sectionCount = doc.sections.length;
  useEffect(() => {
    if (sectionCount === 0 || typeof IntersectionObserver === "undefined") return;
    const headings = Array.from(document.querySelectorAll<HTMLElement>(".notes-body [data-section-index]"));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (visible.length === 0) return;
        const top = visible.reduce((first, entry) => (entry.boundingClientRect.top < first.boundingClientRect.top ? entry : first));
        setActiveSection(Number((top.target as HTMLElement).dataset.sectionIndex));
      },
      { rootMargin: "0px 0px -65% 0px" },
    );
    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  }, [sectionCount, session.id, view]);

  useEffect(() => {
    if (focusSection == null || (view !== "saved" && view !== "generating")) return;
    const frame = window.requestAnimationFrame(() => {
      scrollToSection(focusSection);
      onFocusHandled();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [focusSection, onFocusHandled, view]);

  const create = async (attachments: NoteAttachment[]) => {
    setSheetOpen(false);
    setPending("create");
    setActionError(null);
    try {
      const { job_id } = await api.createSessionNotes(
        session.id,
        attachments.map((file) => file.id),
      );
      onJob({
        job_id,
        session_id: session.id,
        task: "notes",
        state: "running",
        stage: attachments.length > 0 ? "extracting" : "reading",
        progress: null,
        text: "",
        error: null,
      });
    } catch (failure) {
      setActionError(String(failure));
    } finally {
      setPending(null);
    }
  };

  const cancel = async () => {
    if (!job) return;
    setPending("cancel");
    try {
      await api.cancelSessionNotes(job.job_id);
    } catch (failure) {
      setActionError(String(failure));
    } finally {
      setPending(null);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(markdown);
      setCopied(true);
      window.clearTimeout(copyTimer.current);
      copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
    } catch (failure) {
      setActionError(`Copy failed: ${String(failure)}`);
    }
  };

  const exportMarkdown = async () => {
    setActionError(null);
    try {
      const path = await save({
        defaultPath: `${safeFilename(doc.title ?? session.title)} notes.md`,
        filters: [{ name: "Markdown", extensions: ["md"] }],
      });
      if (!path) return;
      try {
        await api.historyExportToPath(session.id, "notes", path);
      } catch (failure) {
        // Shells without a notes-only format export the transcript
        // Markdown, which embeds the notes at the top.
        if (!/unknown variant/i.test(String(failure))) throw failure;
        await api.historyExportToPath(session.id, "markdown", path);
      }
    } catch (failure) {
      setActionError(String(failure));
    }
  };

  const failure = job?.state === "failed" ? job : null;
  const hasNotes = view === "saved" || view === "generating";
  // Retrying cannot succeed until the setup the shell checks is complete.
  const canRetry = setupNeed(assistant) === null && segments.length > 0;
  const errorBlock = failure ? (
    <NotesError job={failure} hadNotes={Boolean(state.notes)} onRetry={canRetry ? () => setSheetOpen(true) : undefined} />
  ) : job?.state === "cancelled" && !running ? (
    <p className="notes-quiet-notice" role="status">
      Notes generation was cancelled.{state.notes ? " Your previous notes are unchanged." : ""}
    </p>
  ) : null;

  return (
    <div className="ai-notes">
      {view === "loading" && (
        <div className="notes-sheet notes-sheet--placeholder" aria-busy="true" aria-label="Loading AI notes">
          <span />
          <span />
          <span />
        </div>
      )}

      {view === "setup" && assistant && (
        <div className="notes-sheet notes-cover">
          <SetupCover status={assistant} />
          {errorBlock}
          {actionError && <p className="inline-error" role="alert">{actionError}</p>}
        </div>
      )}

      {view === "empty" && assistant && (
        <div className="notes-sheet notes-cover">
          {errorBlock}
          <h3>Turn this session into study notes</h3>
          <p>
            The assistant restructures the transcript into sections, fixes likely recognition errors and keeps the
            details: definitions, examples, numbers and announcements.
          </p>
          {segments.length === 0 ? (
            <p className="notes-send-line">This session has no transcript to work from.</p>
          ) : (
            <p className="notes-send-line">
              <CloudArrowUp size={16} weight="regular" aria-hidden="true" />
              <span>
                ~{approximate(estimateWords(segments.map((segment) => segment.source_text).join("\n")))} words will be
                sent to {assistant.display_name || "the assistant"}
                {assistant.model ? ` · ${assistant.model}` : ""}
              </span>
            </p>
          )}
          <div className="notes-cover-actions">
            <button
              className="button button--primary"
              type="button"
              disabled={segments.length === 0 || pending === "create"}
              onClick={() => setSheetOpen(true)}
            >
              {pending === "create" ? "Starting…" : "Create notes"}
            </button>
          </div>
          {actionError && <p className="inline-error" role="alert">{actionError}</p>}
        </div>
      )}

      {hasNotes && (
        <div className={`ai-notes-layout ${sectionCount > 0 ? "" : "ai-notes-layout--single"}`}>
          {sectionCount > 0 && (
            <>
              <nav className="notes-rail" aria-label="Contents">
                <p className="notes-rail-label">Contents</p>
                <SectionList sections={doc.sections} active={activeSection} />
              </nav>
              <details className="notes-contents-disclosure">
                <summary>
                  <CaretDown size={12} weight="bold" aria-hidden="true" />
                  Contents
                  <span>{sectionCount} {sectionCount === 1 ? "section" : "sections"}</span>
                </summary>
                <SectionList sections={doc.sections} active={activeSection} />
              </details>
            </>
          )}
          <article className={`notes-sheet ${running ? "notes-sheet--running" : ""}`} aria-busy={running || undefined}>
            {running && <ProgressRule fraction={progressFraction(job)} />}
            <header className="notes-sheet-header">
              <div className="notes-sheet-titlebar">
                <h3 className="notes-title" lang={session.target_language}>
                  {doc.title ?? (running ? "Writing notes…" : session.title)}
                </h3>
                <div className="notes-actions">
                  {running ? (
                    <button className="button" type="button" disabled={pending === "cancel"} onClick={() => void cancel()}>
                      {pending === "cancel" ? "Cancelling…" : "Cancel"}
                    </button>
                  ) : (
                    <>
                      <button className="button button--small" type="button" onClick={() => void copy()}>
                        {copied ? <Check size={14} weight="bold" aria-hidden="true" /> : <Copy size={14} weight="regular" aria-hidden="true" />}
                        {copied ? "Copied" : "Copy"}
                      </button>
                      <button className="button button--small" type="button" onClick={() => void exportMarkdown()}>
                        <DownloadSimple size={14} weight="regular" aria-hidden="true" />
                        Export .md
                      </button>
                      <button
                        className="button button--small"
                        type="button"
                        disabled={Boolean(setupNeed(assistant)) || pending === "create"}
                        title={setupNeed(assistant) ? "Set up the AI assistant in Settings first" : "Write new notes; the current ones are replaced when the new ones are saved"}
                        onClick={() => setSheetOpen(true)}
                      >
                        <ArrowClockwise size={14} weight="regular" aria-hidden="true" />
                        Regenerate
                      </button>
                    </>
                  )}
                </div>
              </div>
              {running && job ? (
                <p className="notes-stage" role="status" aria-live="polite">
                  {stageLabel(job)}
                </p>
              ) : (
                state.notes && <Provenance notes={state.notes} />
              )}
            </header>
            {errorBlock}
            {actionError && <p className="inline-error" role="alert">{actionError}</p>}
            {doc.blocks.length === 0 && running ? (
              <p className="notes-waiting">The first section appears here as soon as the model starts writing.</p>
            ) : (
              <NotesMarkdown blocks={doc.blocks} lang={session.target_language} onJump={segments.length > 0 ? onJump : undefined} />
            )}
          </article>
        </div>
      )}

      <AttachmentSheet
        // Files picked for one lecture are not offered for the next one.
        key={session.id}
        open={sheetOpen}
        initial={previewInitialAttachments()}
        onClose={() => setSheetOpen(false)}
        onCreate={(files) => void create(files)}
      />
    </div>
  );
}

function SectionList({
  active,
  sections,
}: {
  sections: ReturnType<typeof parseNotesDocument>["sections"];
  active: number | null;
}) {
  return (
    <ol className="notes-section-list">
      {sections.map((section) => (
        <li key={section.index}>
          <button
            className={active === section.index ? "notes-section-link notes-section-link--active" : "notes-section-link"}
            type="button"
            aria-current={active === section.index ? "location" : undefined}
            onClick={(event) => {
              scrollToSection(section.index);
              const details = event.currentTarget.closest("details");
              if (details) details.open = false;
            }}
          >
            <span className="notes-section-title">{section.title}</span>
            {section.startMs != null && section.endMs != null && (
              <span className="notes-section-time">{formatRange(section.startMs, section.endMs)}</span>
            )}
          </button>
        </li>
      ))}
    </ol>
  );
}

function ProgressRule({ fraction }: { fraction: number | null }) {
  return (
    <div
      className={`notes-progress ${fraction == null ? "notes-progress--indeterminate" : ""}`}
      role="progressbar"
      aria-label="Writing notes"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={fraction == null ? undefined : Math.round(fraction * 100)}
    >
      <span style={fraction == null ? undefined : { width: `${Math.round(fraction * 100)}%` }} />
    </div>
  );
}

function Provenance({ notes }: { notes: SessionNotes }) {
  const tokens = formatTokens(notes.usage && typeof notes.usage === "object" ? notes.usage : {});
  return (
    <div className="notes-provenance">
      <span className="provenance-model" title={notes.provider ? `Provider: ${notes.provider}` : undefined}>
        {notes.model || notes.provider}
      </span>
      <time dateTime={notes.updated_at}>{formatDate(notes.updated_at)}</time>
      {(Array.isArray(notes.attachments) ? notes.attachments : []).map((attachment, index) => (
        <AttachmentChip key={attachment.id ?? `${attachment.name}-${index}`} attachment={attachment} />
      ))}
      {tokens && <span className="provenance-tokens">{tokens}</span>}
    </div>
  );
}

function AttachmentChip({ attachment }: { attachment: NoteAttachmentReport }) {
  const details = [
    attachment.pages ? `${attachment.pages} ${attachment.pages === 1 ? "page" : "pages"}` : null,
    attachment.chars ? `${attachment.chars.toLocaleString("en-US")} characters read` : null,
    attachment.truncated ? "truncated to fit" : null,
    attachment.warning ?? null,
  ].filter(Boolean);
  const warn = Boolean(attachment.warning || attachment.truncated);
  return (
    <span className={`attachment-chip ${warn ? "attachment-chip--warn" : ""}`} title={details.join(" · ") || undefined}>
      {warn ? <Warning size={11} weight="bold" aria-hidden="true" /> : <Paperclip size={11} weight="bold" aria-hidden="true" />}
      <span>{attachment.name}</span>
      {warn && <span className="visually-hidden"> ({details.join(", ")})</span>}
    </span>
  );
}

function SetupCover({ status }: { status: AssistantStatus }) {
  const need = setupNeed(status);
  const name = status.display_name || "the provider";
  if (need === "key") {
    return (
      <>
        <h3>Add a key for {name}</h3>
        <p>The AI assistant uses the {name} key saved under Cloud providers. No key is saved yet.</p>
        <div className="notes-cover-actions">
          <button className="button button--primary" type="button" onClick={() => openSettings("cloud-providers")}>
            Open Cloud providers
          </button>
        </div>
      </>
    );
  }
  if (need === "consent") {
    return (
      <>
        <h3>Allow transcripts to go to {name}</h3>
        <p>
          Creating notes sends this session’s transcript, and the text of any files you attach, to {name}. Turn this on
          in Settings → AI assistant.
        </p>
        <div className="notes-cover-actions">
          <button className="button button--primary" type="button" onClick={() => openSettings("ai-assistant")}>
            Open AI assistant settings
          </button>
        </div>
      </>
    );
  }
  return (
    <>
      <h3>Set up the AI assistant</h3>
      <p>
        Study notes are written by a language model you choose, with your own API key. Pick a provider and model in
        Settings; nothing is sent until you allow it there.
      </p>
      <div className="notes-cover-actions">
        <button className="button button--primary" type="button" onClick={() => openSettings("ai-assistant")}>
          Open AI assistant settings
        </button>
      </div>
    </>
  );
}

function NotesError({ hadNotes, job, onRetry }: { job: AssistantJob; hadNotes: boolean; onRetry?: () => void }) {
  const guidance = errorGuidance(job.error?.code);
  return (
    <div className="notes-error" role="alert">
      <WarningCircle size={18} weight="regular" aria-hidden="true" />
      <div>
        <strong>{hadNotes ? "New notes were not created. Your previous notes are unchanged." : "Notes were not created."}</strong>
        <p>{guidance.text}</p>
        {job.error?.message && <small>{job.error.message}</small>}
        <div className="notes-error-actions">
          {guidance.section && (
            <button className="button button--small" type="button" onClick={() => openSettings(guidance.section!)}>
              {guidance.actionLabel}
            </button>
          )}
          {onRetry && (
            <button className="button button--small button--quiet" type="button" onClick={onRetry}>
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function approximate(words: number): string {
  const rounded = words >= 1000 ? Math.round(words / 100) * 100 : words >= 100 ? Math.round(words / 10) * 10 : words;
  return rounded.toLocaleString("en-US");
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}
