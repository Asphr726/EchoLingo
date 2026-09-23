import { CaretRight, FileArrowUp, LockKey } from "@phosphor-icons/react";
import { useEffect, useId, useState } from "react";
import { api } from "../lib/bridge";
import { contextLineCount, isEmptyImport, mergeContext, SESSION_CONTEXT_LIMIT } from "../lib/context";
import { requestNavigation } from "../lib/navigation";
import { setupNeed } from "../lib/notes";
import { useApp } from "../state/AppContext";
import type { AssistantStatus } from "../types";

const PLACEHOLDER =
  "Topic, names and terms — one per line. Example: Béla Julesz · saccade · pre-attentive vision = 前注意视觉";

/** Per-lecture topic and terms (docs/adr/0006), sent with the next Start as
 *  `session_context`. Read-only while a session runs. */
export function LectureContextPanel({ locked }: { locked: boolean }) {
  const { draft, setDraft, snapshot } = useApp();
  const value = locked ? snapshot.config?.session_context ?? draft.session_context : draft.session_context;
  const [open, setOpen] = useState(!locked);
  const [assistant, setAssistant] = useState<AssistantStatus | null>(null);
  const [importing, setImporting] = useState(false);
  const [notice, setNotice] = useState<{ added: number; skipped: number; warnings: string[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bodyId = useId();
  const countId = useId();

  useEffect(() => {
    let active = true;
    void api
      .assistantStatus()
      .then((status) => active && setAssistant(status))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (locked) setOpen(false);
  }, [locked]);

  const useLlm = assistant !== null && setupNeed(assistant) === null;
  const lines = contextLineCount(value);

  const importSlides = async () => {
    setImporting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.importContextFiles(useLlm);
      // The shell answers a cancelled picker with an empty result.
      if (isEmptyImport(result)) return;
      // The native picker is modal, so the draft has not changed meanwhile;
      // the updater still merges into the latest text.
      const preview = mergeContext(draft.session_context, result);
      setDraft((current) => {
        const next = mergeContext(current.session_context, result);
        return next.text === current.session_context ? current : { ...current, session_context: next.text };
      });
      setNotice({ added: preview.added, skipped: preview.skipped, warnings: result.warnings ?? [] });
    } catch (failure) {
      setError(String(failure));
    } finally {
      setImporting(false);
    }
  };

  return (
    <section className={`lecture-context ${open ? "lecture-context--open" : ""}`} aria-label="Lecture context">
      <button
        className="lecture-context-toggle"
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((current) => !current)}
      >
        <CaretRight size={12} weight="bold" aria-hidden="true" className="lecture-context-caret" />
        <span className="lecture-context-title">Lecture context</span>
        <span className="lecture-context-summary">
          {lines > 0 ? `${lines} ${lines === 1 ? "line" : "lines"}` : locked ? "None" : "Helps with names and terms"}
          {locked && lines > 0 ? " · in use" : ""}
        </span>
      </button>
      <div className="lecture-context-body" id={bodyId} hidden={!open}>
        <textarea
          aria-label="Lecture context"
          aria-describedby={countId}
          rows={4}
          maxLength={SESSION_CONTEXT_LIMIT}
          spellCheck={false}
          placeholder={PLACEHOLDER}
          value={value}
          readOnly={locked}
          disabled={locked}
          onChange={(event) => {
            // The shell rejects NUL in session text; pasted PDF text can carry it.
            const text = event.target.value.replace(/\0/g, "").slice(0, SESSION_CONTEXT_LIMIT);
            setDraft((current) => ({ ...current, session_context: text }));
          }}
        />
        <div className="lecture-context-bar">
          <button className="button button--small" type="button" disabled={locked || importing} onClick={() => void importSlides()}>
            <FileArrowUp size={15} weight="regular" aria-hidden="true" />
            {importing ? "Reading files…" : "Import from slides…"}
          </button>
          <span className="lecture-context-hint">
            {useLlm
              ? `Slide text is sent to ${assistant?.display_name || "the AI assistant"} to pick out terms and translations.`
              : "Text is extracted on this Mac."}
          </span>
          <span id={countId} className={value.length > SESSION_CONTEXT_LIMIT * 0.9 ? "char-count char-count--near" : "char-count"}>
            {value.length.toLocaleString("en-US")} / {SESSION_CONTEXT_LIMIT.toLocaleString("en-US")}
          </span>
        </div>
        {notice && (
          <div className="lecture-context-notice" role="status">
            {notice.added > 0
              ? `Added ${notice.added} ${notice.added === 1 ? "line" : "lines"} from the files.`
              : notice.warnings.length === 0
                ? "Nothing new to add."
                : null}
            {notice.skipped > 0 && ` ${notice.skipped} more did not fit in ${SESSION_CONTEXT_LIMIT.toLocaleString("en-US")} characters.`}
            {notice.warnings.length > 0 && (
              <ul>
                {notice.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        {error && <p className="field-error" role="alert">{error}</p>}
        <p className="lecture-context-privacy">
          <LockKey size={13} weight="regular" aria-hidden="true" />
          <span>
            Context is sent to cloud recognition/translation only when those uploads are enabled. Standing terms for every
            lecture live in the{" "}
            <button className="text-button text-button--inline" type="button" onClick={() => requestNavigation({ view: "settings", section: "translation" })}>
              glossary
            </button>
            .
          </span>
        </p>
      </div>
    </section>
  );
}
