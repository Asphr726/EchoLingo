import { File, FileCsv, FileDoc, FileMd, FilePdf, FilePpt, FileTxt, Plus, X } from "@phosphor-icons/react";
import { type ComponentType, useEffect, useRef, useState } from "react";
import { api } from "../lib/bridge";
import { formatBytes } from "../lib/notes";
import type { NoteAttachment } from "../types";

export const MAX_ATTACHMENTS = 5;

const icons: Record<string, ComponentType<{ size?: number; weight?: "regular"; "aria-hidden"?: boolean | "true" }>> = {
  pdf: FilePdf,
  pptx: FilePpt,
  docx: FileDoc,
  csv: FileCsv,
  txt: FileTxt,
  md: FileMd,
  markdown: FileMd,
};

/** "Add course materials?" — asked when the user presses Create notes. A
 *  native modal dialog: the page behind is inert, focus stays inside and
 *  Esc closes it. File paths stay in the shell; only ids reach the page. */
export function AttachmentSheet({
  initial = [],
  onClose,
  onCreate,
  open,
}: {
  open: boolean;
  initial?: NoteAttachment[];
  onClose: () => void;
  onCreate: (attachments: NoteAttachment[]) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const addRef = useRef<HTMLButtonElement>(null);
  const [files, setFiles] = useState<NoteAttachment[]>(initial);
  const [picking, setPicking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      setNotice(null);
      dialog.showModal();
      (files.length > 0 ? primaryRef.current : addRef.current)?.focus();
    } else if (!open && dialog.open) {
      dialog.close();
    }
    // Focus is chosen once per opening.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const addFiles = async () => {
    setPicking(true);
    setNotice(null);
    try {
      const picked = await api.pickNoteAttachments();
      // Picking is disabled while a pick runs, so `files` is current here.
      const known = new Set(files.map((file) => file.id));
      const fresh = picked.filter((file) => !known.has(file.id));
      setFiles([...files, ...fresh].slice(0, MAX_ATTACHMENTS));
      if (files.length + fresh.length > MAX_ATTACHMENTS) {
        setNotice(`Up to ${MAX_ATTACHMENTS} files can be attached; the rest were left out.`);
      }
    } catch (failure) {
      setNotice(String(failure));
    } finally {
      setPicking(false);
    }
  };

  const count = files.length;
  return (
    <dialog
      ref={dialogRef}
      className="attachment-sheet"
      aria-labelledby="attachment-sheet-title"
      aria-describedby="attachment-sheet-body"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      // Keeps `open` in sync when the dialog closes natively (for example a
      // second Esc, which browsers may not let a page cancel).
      onClose={() => onClose()}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="attachment-sheet-inner">
        <header className="attachment-sheet-header">
          <h3 id="attachment-sheet-title">Add course materials?</h3>
          <button className="icon-button" type="button" aria-label="Close" onClick={onClose}>
            <X size={16} weight="bold" aria-hidden="true" />
          </button>
        </header>
        <p id="attachment-sheet-body" className="attachment-sheet-body">
          Slides or handouts help fix names, terms and formulas. Text is extracted on this computer; only the text is sent.
        </p>
        {count > 0 && (
          <ul className="attachment-list" aria-label="Files to attach">
            {files.map((file) => {
              const Icon = icons[file.extension.toLowerCase()] ?? File;
              return (
                <li key={file.id}>
                  <Icon size={20} weight="regular" aria-hidden="true" />
                  <span className="attachment-name" title={file.name}>{file.name}</span>
                  <span className="attachment-meta">
                    {file.extension.toUpperCase()} · {formatBytes(file.size_bytes)}
                  </span>
                  <button
                    className="icon-button"
                    type="button"
                    aria-label={`Remove ${file.name}`}
                    onClick={() => setFiles((current) => current.filter((entry) => entry.id !== file.id))}
                  >
                    <X size={14} weight="bold" aria-hidden="true" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <div className="attachment-add-row">
          <button
            ref={addRef}
            className="button"
            type="button"
            disabled={picking || count >= MAX_ATTACHMENTS}
            onClick={() => void addFiles()}
          >
            <Plus size={15} weight="bold" aria-hidden="true" />
            {picking ? "Choosing…" : "Add files…"}
          </button>
          <span className="attachment-hint">PDF, PowerPoint, Word, Markdown, LaTeX or text · up to {MAX_ATTACHMENTS} files, 25{"\u00a0"}MB each</span>
        </div>
        {notice && <p className="attachment-notice" role="status">{notice}</p>}
        <footer className="attachment-sheet-footer">
          <button className="button button--quiet" type="button" onClick={() => onCreate([])}>
            Create without files
          </button>
          <button
            ref={primaryRef}
            className="button button--primary"
            type="button"
            disabled={count === 0}
            onClick={() => onCreate(files)}
          >
            {count === 0 ? "Create with files" : `Create with ${count} ${count === 1 ? "file" : "files"}`}
          </button>
        </footer>
      </div>
    </dialog>
  );
}
