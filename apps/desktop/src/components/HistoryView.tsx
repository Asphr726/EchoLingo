import {
  DownloadSimple,
  FileText,
  MagnifyingGlass,
  PencilSimple,
  Trash,
  X,
} from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { api } from "../lib/bridge";
import type { SessionDetail, SessionRecord } from "../types";

const formats = ["markdown", "txt", "json", "srt", "vtt"] as const;

export function HistoryView() {
  const [query, setQuery] = useState("");
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [selected, setSelected] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);

  const open = async (session: SessionRecord) => {
    setError(null);
    try {
      const detail = await api.historyOpen(session.id);
      setSelected(detail);
      setTitle(detail.session.title);
      setRenaming(false);
      setConfirmDelete(false);
    } catch (failure) {
      setError(String(failure));
    }
  };

  const refresh = async (search = query) => {
    setLoading(true);
    setError(null);
    try {
      const found = await api.historySearch(search);
      setSessions(found);
      // Land on the most recent session instead of an empty detail pane.
      if (!search && found.length > 0 && selected === null) await open(found[0]);
    } catch (failure) {
      setError(String(failure));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rename = async () => {
    if (!selected || !title.trim()) return;
    try {
      await api.historyRename(selected.session.id, title.trim());
      setSelected({
        ...selected,
        session: { ...selected.session, title: title.trim() },
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
      setSelected(null);
      setConfirmDelete(false);
      await refresh();
    } catch (failure) {
      setError(String(failure));
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
        {loading ? (
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
                  <strong>{session.title}</strong>
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
            <p>Open a saved session to review timestamps, original text, translation, and export options.</p>
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
                  <h2>{selected.session.title}</h2>
                )}
                <p>{selected.session.source_language.toUpperCase()} → {selected.session.target_language.toUpperCase()} · {selected.session.audio_profile}</p>
              </div>
              <div className="detail-actions">
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
                <span>Delete this transcript permanently?</span>
                <button className="button button--danger" type="button" onClick={() => void remove()}>Delete</button>
                <button className="button button--quiet" type="button" onClick={() => setConfirmDelete(false)}>Keep</button>
              </div>
            )}
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
                    <article className="bilingual-row bilingual-row--history" key={segment.id}>
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
          </>
        )}
      </section>
    </div>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatClock(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

function safeFilename(value: string) {
  return value.trim().replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/^-+|-+$/g, "") || "echolingo-session";
}

function extension(format: string) {
  return format === "markdown" ? "md" : format;
}
