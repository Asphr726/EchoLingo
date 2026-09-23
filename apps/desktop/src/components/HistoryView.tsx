import {
  ArrowUUpLeft,
  DownloadSimple,
  FileText,
  MagnifyingGlass,
  PencilSimple,
  Sparkle,
  Trash,
  X,
} from "@phosphor-icons/react";
import { type KeyboardEvent, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { api, previewOpensNotesTab, subscribeUiEvents } from "../lib/bridge";
import { formatClock } from "../lib/markdown";
import { safeFilename, segmentAtOrAfter, setupNeed } from "../lib/notes";
import type {
  AssistantJob,
  AssistantStatus,
  HistoryChange,
  SessionDetail,
  SessionNotesState,
  SessionRecord,
} from "../types";
import { AiNotesPanel } from "./AiNotesPanel";

const formats = ["markdown", "txt", "json", "srt", "vtt"] as const;
type Tab = "transcript" | "notes";
const emptyNotes: SessionNotesState = { notes: null, job: null };

/** A time chip in the notes pointed at this transcript row. */
interface JumpTarget {
  segmentId: string;
  sectionIndex: number | null;
}

export function HistoryView() {
  const [query, setQuery] = useState("");
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [selected, setSelected] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [tab, setTab] = useState<Tab>("transcript");
  const [notesState, setNotesState] = useState<SessionNotesState>(emptyNotes);
  const [notesLoading, setNotesLoading] = useState(false);
  const [assistant, setAssistant] = useState<AssistantStatus | null>(null);
  const [titlePending, setTitlePending] = useState(false);
  const [jump, setJump] = useState<JumpTarget | null>(null);
  const [focusSection, setFocusSection] = useState<number | null>(null);
  const selectedId = useRef<string | null>(null);
  const queryRef = useRef("");
  queryRef.current = query;
  selectedId.current = selected?.session.id ?? null;

  const loadNotes = useCallback(async (sessionId: string) => {
    try {
      const next = await api.sessionNotes(sessionId);
      if (selectedId.current === sessionId) setNotesState(next);
      return next;
    } catch {
      // Older shells without assistant commands: notes stay unavailable.
      return emptyNotes;
    }
  }, []);

  const open = useCallback(
    async (session: SessionRecord) => {
      setError(null);
      try {
        const detail = await api.historyOpen(session.id);
        selectedId.current = detail.session.id;
        setSelected(detail);
        setTitle(detail.session.title);
        setRenaming(false);
        setConfirmDelete(false);
        setJump(null);
        setNotesState(emptyNotes);
        setNotesLoading(true);
        const notes = await loadNotes(detail.session.id);
        setNotesLoading(false);
        if (selectedId.current === detail.session.id) {
          setTab(notes.notes || notes.job?.state === "running" || previewOpensNotesTab() ? "notes" : "transcript");
        }
      } catch (failure) {
        setError(String(failure));
        setNotesLoading(false);
      }
    },
    [loadNotes],
  );

  const refresh = useCallback(
    async (search = queryRef.current) => {
      setLoading(true);
      setError(null);
      try {
        const found = await api.historySearch(search);
        setSessions(found);
        // Land on the most recent session instead of an empty detail pane.
        if (!search && found.length > 0 && selectedId.current === null) await open(found[0]);
      } catch (failure) {
        setError(String(failure));
      } finally {
        setLoading(false);
      }
    },
    [open],
  );

  /** Re-read the list after a background change and patch the open
   *  session's title without reloading its transcript. */
  const syncList = useCallback(async () => {
    try {
      const found = await api.historySearch(queryRef.current);
      setSessions(found);
      setSelected((current) => {
        if (!current) return current;
        const match = found.find((session) => session.id === current.session.id);
        if (!match) return current;
        return { ...current, session: { ...current.session, title: match.title, title_source: match.title_source } };
      });
    } catch {
      // The next explicit refresh reports errors.
    }
  }, []);

  useEffect(() => {
    void refresh("");
    void api.assistantStatus().then(setAssistant).catch(() =>
      setAssistant({
        configured: false,
        consent: false,
        provider_group: "",
        model: "",
        display_name: "",
        key_available: false,
        auto_title: false,
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let active = true;
    let dispose: () => void = () => undefined;
    void subscribeUiEvents((event) => {
      if (event.kind === "assistant_update") {
        const job = event.payload as AssistantJob;
        if (job.task !== "notes" || !job.session_id || job.session_id !== selectedId.current) return;
        setNotesState((current) => ({ ...current, job }));
        if (job.state === "completed") void loadNotes(job.session_id);
        return;
      }
      if (event.kind === "history_changed") {
        const change = event.payload as HistoryChange;
        if (change.reason === "deleted" && change.session_id === selectedId.current) {
          selectedId.current = null;
          setSelected(null);
          setNotesState(emptyNotes);
        }
        if (change.reason === "notes" && change.session_id === selectedId.current) {
          void loadNotes(change.session_id);
        }
        void syncList();
      }
    }).then((unlisten) => {
      if (active) dispose = unlisten;
      else unlisten();
    });
    return () => {
      active = false;
      dispose();
    };
  }, [loadNotes, syncList]);

  // Bring the row a time chip pointed at into view once the transcript is
  // visible.
  useLayoutEffect(() => {
    if (tab !== "transcript" || !jump) return;
    const row = document.getElementById(rowId(jump.segmentId));
    row?.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    row?.focus({ preventScroll: true });
  }, [jump, tab]);

  const rename = async () => {
    if (!selected || !title.trim()) return;
    try {
      await api.historyRename(selected.session.id, title.trim());
      setSelected({
        ...selected,
        session: { ...selected.session, title: title.trim(), title_source: "user" },
      });
      setRenaming(false);
      await refresh();
    } catch (failure) {
      setError(String(failure));
    }
  };

  const remove = async () => {
    if (!selected) return;
    try {
      await api.historyDelete(selected.session.id);
      selectedId.current = null;
      setSelected(null);
      setNotesState(emptyNotes);
      setConfirmDelete(false);
      await refresh();
    } catch (failure) {
      setError(String(failure));
    }
  };

  const generateTitle = async () => {
    if (!selected) return;
    const sessionId = selected.session.id;
    setTitlePending(true);
    setError(null);
    try {
      const result = await api.generateSessionTitle(sessionId);
      // `applied: false` means the store kept a title the user set meanwhile.
      setSelected((current) =>
        current && current.session.id === sessionId && result.applied !== false && current.session.title_source !== "user"
          ? { ...current, session: { ...current.session, title: result.title, title_source: "ai" } }
          : current,
      );
      await syncList();
    } catch (failure) {
      setError(String(failure));
    } finally {
      setTitlePending(false);
    }
  };

  const exportSession = async (format: (typeof formats)[number]) => {
    if (!selected) return;
    try {
      const suffix = extension(format);
      const path = await save({
        defaultPath: `${safeFilename(selected.session.title)}.${suffix}`,
        filters: [{ name: format === "markdown" ? "Markdown" : format.toUpperCase(), extensions: [suffix] }],
      });
      if (path) await api.historyExportToPath(selected.session.id, format, path);
    } catch (failure) {
      setError(String(failure));
    }
  };

  const jumpToTranscript = useCallback(
    (startMs: number, sectionIndex: number | null) => {
      const target = selected ? segmentAtOrAfter(selected.segments, startMs) : null;
      if (!target) return;
      setJump({ segmentId: target.id, sectionIndex });
      setTab("transcript");
    },
    [selected],
  );

  const backToNotes = () => {
    setFocusSection(jump?.sectionIndex ?? null);
    setJump(null);
    setTab("notes");
  };

  const onTabKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const next: Tab = tab === "transcript" ? "notes" : "transcript";
    setTab(next);
    document.getElementById(`history-tab-${next}`)?.focus();
  };

  const clearFocusSection = useCallback(() => setFocusSection(null), []);
  const titleNeed = setupNeed(assistant);
  const notesRunning = notesState.job?.state === "running";

  return (
    <div className="history-layout">
      <section className="history-list-pane">
        <form
          className="search-field"
          onSubmit={(event) => {
            event.preventDefault();
            void refresh();
          }}
        >
          <MagnifyingGlass size={17} weight="bold" aria-hidden="true" />
          <input
            aria-label="Search transcript history"
            placeholder="Search sessions or transcript"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {query && (
            <button type="button" aria-label="Clear search" onClick={() => { setQuery(""); void refresh(""); }}>
              <X size={15} weight="bold" aria-hidden="true" />
            </button>
          )}
        </form>
        <div className="history-list-heading">
          <span className="section-kicker">Sessions</span>
          <span>{sessions.length}</span>
        </div>
        {loading && sessions.length === 0 ? (
          <div className="history-skeleton" aria-label="Loading session history">
            <span /><span /><span />
          </div>
        ) : sessions.length === 0 ? (
          <div className="history-empty">
            <FileText size={25} weight="regular" aria-hidden="true" />
            <strong>{query ? "No matching sessions" : "No lectures saved yet"}</strong>
            <span>{query ? "Try a different phrase." : "Completed sessions will appear here automatically."}</span>
          </div>
        ) : (
          <div className="history-list">
            {sessions.map((session) => (
              <button
                className={selected?.session.id === session.id ? "history-row history-row--active" : "history-row"}
                key={session.id}
                type="button"
                onClick={() => void open(session)}
              >
                <div>
                  <strong>
                    {session.title_source === "ai" && (
                      <>
                        <Sparkle className="ai-title-mark" size={10} weight="fill" aria-hidden="true" />
                        <span className="visually-hidden">AI title: </span>
                      </>
                    )}
                    {session.title}
                  </strong>
                  <span>{session.source_language.toUpperCase()} → {session.target_language.toUpperCase()}</span>
                </div>
                <div>
                  <time>{formatDate(session.started_at)}</time>
                  <span>{session.status.replaceAll("_", " ")}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="history-detail-pane">
        {error && <p className="inline-error" role="alert">{error}</p>}
        {!selected ? (
          <div className="detail-empty">
            <span className="section-kicker">Session detail</span>
            <h2>Select a lecture</h2>
            <p>Open a saved session to review timestamps, original text, translation, AI notes and export options.</p>
          </div>
        ) : (
          <>
            <header className="detail-header">
              <div>
                <span className="section-kicker">{formatDate(selected.session.started_at)}</span>
                {renaming ? (
                  <div className="rename-row">
                    <input value={title} onChange={(event) => setTitle(event.target.value)} aria-label="Session title" />
                    <button className="button button--primary" type="button" onClick={() => void rename()}>Save</button>
                    <button className="button button--quiet" type="button" onClick={() => setRenaming(false)}>Cancel</button>
                  </div>
                ) : (
                  <h2 lang={selected.session.title_source === "ai" ? selected.session.target_language : undefined}>
                    {selected.session.title}
                  </h2>
                )}
                <p>{selected.session.source_language.toUpperCase()} → {selected.session.target_language.toUpperCase()} · {selected.session.audio_profile}</p>
              </div>
              <div className="detail-actions">
                {selected.session.title_source === "default" && (
                  <button
                    className="icon-button"
                    type="button"
                    aria-label={titlePending ? "Generating a title" : "Generate title"}
                    title={titleNeed ? "Set up the AI assistant in Settings to generate titles" : "Generate title"}
                    disabled={titlePending || Boolean(titleNeed) || selected.segments.length === 0}
                    aria-busy={titlePending || undefined}
                    onClick={() => void generateTitle()}
                  >
                    <Sparkle size={17} weight={titlePending ? "fill" : "regular"} aria-hidden="true" />
                  </button>
                )}
                <button className="icon-button" type="button" aria-label="Rename session" onClick={() => setRenaming(true)}>
                  <PencilSimple size={17} weight="regular" aria-hidden="true" />
                </button>
                <button className="icon-button icon-button--danger" type="button" aria-label="Delete session" onClick={() => setConfirmDelete(true)}>
                  <Trash size={17} weight="regular" aria-hidden="true" />
                </button>
              </div>
            </header>
            {confirmDelete && (
              <div className="delete-confirm" role="alert">
                <span>Delete this transcript{notesState.notes ? " and its AI notes" : ""} permanently?</span>
                <button className="button button--danger" type="button" onClick={() => void remove()}>Delete</button>
                <button className="button button--quiet" type="button" onClick={() => setConfirmDelete(false)}>Keep</button>
              </div>
            )}
            <div className="detail-tabs segmented-control segmented-control--two" role="tablist" aria-label="Session view" onKeyDown={onTabKey}>
              {(["transcript", "notes"] as const).map((id) => (
                <button
                  key={id}
                  id={`history-tab-${id}`}
                  className={tab === id ? "segmented-control--active" : ""}
                  type="button"
                  role="tab"
                  aria-selected={tab === id}
                  aria-controls={`history-panel-${id}`}
                  tabIndex={tab === id ? 0 : -1}
                  onClick={() => setTab(id)}
                >
                  {id === "transcript" ? "Transcript" : "AI notes"}
                  {id === "notes" && notesRunning && <span className="tab-activity" aria-label="(writing)" />}
                </button>
              ))}
            </div>

            <div id="history-panel-transcript" role="tabpanel" aria-labelledby="history-tab-transcript" hidden={tab !== "transcript"}>
              <div className="export-row" aria-label="Export transcript">
                <DownloadSimple size={17} weight="bold" aria-hidden="true" />
                <span>Export</span>
                {formats.map((format) => (
                  <button key={format} type="button" onClick={() => void exportSession(format)}>
                    {format === "markdown" ? "MD" : format.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="session-meta">
                <span>ASR <strong>{selected.session.asr_backend}</strong></span>
                <span>Translation <strong>{selected.session.translation_backend}</strong></span>
                <span>
                  Timing{" "}
                  <strong>
                    {selected.segments.some((segment) => segment.timestamp_quality === "forced")
                      ? "word-aligned"
                      : "segment timestamps"}
                  </strong>
                </span>
              </div>
              <div className="detail-segments">
                {selected.segments.length === 0 ? (
                  <p className="no-segments">This session ended before a stable segment was committed.</p>
                ) : (
                  <>
                    <div className="bilingual-row bilingual-row--header" aria-hidden="true">
                      <span />
                      <span>Original · {selected.session.source_language}</span>
                      <span>Translation · {selected.session.target_language}</span>
                    </div>
                    {selected.segments.map((segment) => (
                      <article
                        className={`bilingual-row bilingual-row--history ${jump?.segmentId === segment.id ? "bilingual-row--target" : ""}`}
                        id={rowId(segment.id)}
                        key={segment.id}
                        tabIndex={jump?.segmentId === segment.id ? -1 : undefined}
                      >
                        <time>{formatClock(segment.start_ms)}</time>
                        <p className="row-original" lang={selected.session.source_language}>{segment.source_text}</p>
                        <p
                          className={`row-translation ${segment.translated_text ? "" : "row-translation--placeholder"}`}
                          lang={selected.session.target_language}
                        >
                          {segment.translated_text || "—"}
                        </p>
                      </article>
                    ))}
                  </>
                )}
              </div>
              {jump && (
                <div className="back-to-notes-dock">
                  <button className="back-to-notes" type="button" onClick={backToNotes}>
                    <ArrowUUpLeft size={14} weight="bold" aria-hidden="true" />
                    Back to notes
                  </button>
                </div>
              )}
            </div>

            <div id="history-panel-notes" role="tabpanel" aria-labelledby="history-tab-notes" hidden={tab !== "notes"}>
              <AiNotesPanel
                session={selected.session}
                segments={selected.segments}
                assistant={assistant}
                state={notesState}
                stateLoading={notesLoading}
                onJob={(job) =>
                  setNotesState((current) =>
                    current.job?.job_id === job.job_id || selectedId.current !== job.session_id ? current : { ...current, job },
                  )
                }
                onJump={jumpToTranscript}
                focusSection={tab === "notes" ? focusSection : null}
                onFocusHandled={clearFocusSection}
              />
            </div>
          </>
        )}
      </section>
    </div>
  );
}

const rowId = (segmentId: string) => `transcript-row-${segmentId}`;

const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function extension(format: string) {
  return format === "markdown" ? "md" : format;
}
