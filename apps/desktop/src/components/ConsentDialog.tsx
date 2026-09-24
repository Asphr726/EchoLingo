import {
  CloudArrowUp,
  FileText,
  GearSix,
  Key,
  LockKey,
  Microphone,
  Paperclip,
  TextAlignLeft,
  X,
} from "@phosphor-icons/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import type {
  AssistantConsentRequest,
  AssistantPurpose,
  ConsentAnswer,
  ConsentRequest,
  ConsentSelection,
  SessionConsentRequest,
  SetupMissingRequest,
} from "../types";

// Allow grants every upload the dialog lists; there is no partial grant, so
// the rows are statements rather than pre-ticked checkboxes.
const allSelected: ConsentSelection = { audio: true, transcript: true };

/** The one consent dialog of the main window. A native modal like the
 *  attachment sheet: the page behind is inert, Esc, Close and the backdrop
 *  cancel, and it opens on the cautious choice (the quiet button), so
 *  granting an upload always takes a deliberate click. Focus goes back to
 *  the control that asked. */
export function ConsentDialog({
  onConfirm,
  onDismiss,
  request,
}: {
  /** Open while non-null. */
  request: ConsentRequest | null;
  /** Grants the selection; a rejection is shown in the dialog. */
  onConfirm: (selection: ConsentSelection) => Promise<void>;
  /** "declined" from the quiet button, "cancelled" from Esc, Close or the
   *  backdrop. */
  onDismiss: (answer: Exclude<ConsentAnswer, "granted">) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const quietRef = useRef<HTMLButtonElement>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  // The control that asked; WebKit does not always restore it on close.
  const returnFocus = useRef<HTMLElement | null>(null);
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (request) {
      setPending(false);
      setFailure(null);
      if (!dialog.open) {
        const active = document.activeElement;
        returnFocus.current = active instanceof HTMLElement && active !== document.body ? active : null;
        dialog.showModal();
      }
      // Opening Settings is harmless; granting an upload is not the default.
      (request.kind === "setup-missing" ? primaryRef : quietRef).current?.focus();
    } else if (dialog.open) {
      dialog.close();
      const target = returnFocus.current;
      returnFocus.current = null;
      // A trigger that went away (a deep link switched views) is skipped;
      // the view that opened moves focus itself.
      if (target?.isConnected && !dialog.contains(document.activeElement)) target.focus({ preventScroll: true });
    }
  }, [request]);

  const dismiss = (answer: Exclude<ConsentAnswer, "granted"> = "cancelled") => {
    if (!pending) onDismiss(answer);
  };

  const confirm = async () => {
    setPending(true);
    setFailure(null);
    try {
      await onConfirm(allSelected);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
      setPending(false);
    }
  };

  const copy = request ? consentCopy(request) : null;

  return (
    <dialog
      ref={dialogRef}
      className="attachment-sheet consent-sheet"
      aria-labelledby="consent-title"
      aria-describedby="consent-body"
      onCancel={(event) => {
        event.preventDefault();
        dismiss();
      }}
      // A native close (a second Esc) must still settle the request. The
      // event is queued, so a dialog already reopened for a newer request
      // ignores it.
      onClose={(event) => {
        if (request && !event.currentTarget.open) onDismiss("cancelled");
      }}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === event.currentTarget) dismiss();
      }}
    >
      {request && copy && (
        <div className="attachment-sheet-inner">
          <header className="attachment-sheet-header">
            <div className="consent-heading">
              <span className="consent-mark" aria-hidden="true">
                {request.kind === "setup-missing" ? <GearSix size={18} weight="regular" /> : <LockKey size={18} weight="regular" />}
              </span>
              <h3 id="consent-title">{copy.title}</h3>
            </div>
            <button className="icon-button" type="button" aria-label="Close" onClick={() => dismiss("cancelled")}>
              <X size={16} weight="bold" aria-hidden="true" />
            </button>
          </header>
          <p id="consent-body" className="attachment-sheet-body">{copy.body}</p>

          {request.kind === "session" && (
            <SessionRows request={request} />
          )}
          {request.kind === "assistant" && <AssistantFlow request={request} />}
          {request.kind === "session" && request.note && <p className="consent-note">{request.note}</p>}

          {copy.footnote && <p className="consent-footnote">{copy.footnote}</p>}
          {failure && <p className="attachment-notice" role="alert">{failure}</p>}

          <footer className="attachment-sheet-footer">
            <button ref={quietRef} className="button button--quiet" type="button" disabled={pending} onClick={() => dismiss("declined")}>
              {copy.secondary ?? "Not now"}
            </button>
            <button
              ref={primaryRef}
              className="button button--primary"
              type="button"
              disabled={pending}
              aria-busy={pending || undefined}
              onClick={() => void confirm()}
            >
              {request.kind === "setup-missing" && (request.need === "key" ? <Key size={15} weight="regular" aria-hidden="true" /> : <GearSix size={15} weight="regular" aria-hidden="true" />)}
              {pending ? "Saving…" : copy.primary}
            </button>
          </footer>
        </div>
      )}
    </dialog>
  );
}

function SessionRows({ request }: { request: SessionConsentRequest }) {
  const rows: Array<{ key: keyof ConsentSelection; icon: ReactNode; text: string; detail: string }> = [];
  if (request.audio) {
    rows.push({
      key: "audio",
      icon: <Microphone size={18} weight="regular" aria-hidden="true" />,
      text: `Audio from this session is uploaded to ${request.audio.vendor} for recognition.`,
      detail: `${request.audio.providerLabel} · processed microphone or system audio, while the session runs`,
    });
  }
  if (request.transcript) {
    rows.push({
      key: "transcript",
      icon: <TextAlignLeft size={18} weight="regular" aria-hidden="true" />,
      text: `Transcript text is sent to ${request.transcript.vendor} for translation.`,
      detail: `${request.transcript.providerLabel} · recognized text and lecture context, never audio`,
    });
  }
  return (
    <ul className="consent-list" aria-label="What Allow permits">
      {rows.map((row) => (
        <li key={row.key} className="consent-row">
          <span className="consent-row-icon">{row.icon}</span>
          <span className="consent-row-copy">
            <strong>{row.text}</strong>
            <small>{row.detail}</small>
          </span>
        </li>
      ))}
    </ul>
  );
}

/** What goes to the assistant and what never does. */
function AssistantFlow({ request }: { request: AssistantConsentRequest }) {
  const transcript = <FileText size={13} weight="regular" aria-hidden="true" />;
  const files = <Paperclip size={13} weight="regular" aria-hidden="true" />;
  const sent: Array<[ReactNode, string]> =
    request.purpose === "import"
      ? [[files, "Text extracted from the slides you pick"], [transcript, "Transcripts, when you create notes or titles"]]
      : [[transcript, "This session’s transcript and lecture context"], [files, "Text extracted from files you attach"]];
  return (
    <dl className="consent-flow">
      <div>
        <dt>
          <CloudArrowUp size={15} weight="regular" aria-hidden="true" />
          Sent to {request.vendor}
        </dt>
        {sent.map(([icon, item]) => (
          <dd key={item}>
            {icon}
            {item}
          </dd>
        ))}
      </div>
      <div>
        <dt>
          <LockKey size={15} weight="regular" aria-hidden="true" />
          Stays on this computer
        </dt>
        <dd>
          <Microphone size={13} weight="regular" aria-hidden="true" />
          Audio, always
        </dd>
      </div>
    </dl>
  );
}

export interface ConsentCopy {
  title: string;
  body: string;
  primary: string;
  /** The quiet button; "Not now" unless declining still does something. */
  secondary?: string;
  footnote?: string;
}

const purposeAction: Record<AssistantPurpose, string> = {
  notes: "Creating notes",
  title: "Generating a title",
  import: "Picking terms from slides",
};

/** Title, body and button copy for each kind of request. */
export function consentCopy(request: ConsentRequest): ConsentCopy {
  switch (request.kind) {
    case "session":
      return {
        title: "Allow cloud processing?",
        body: request.note
          ? "Testing needs the same permission a cloud session uses. Nothing leaves this computer until you allow it."
          : "The providers chosen for this session run in the cloud. Nothing leaves this computer until you allow it.",
        primary: "Allow and continue",
        footnote: "Saved as your default. Turn it off anytime in Settings → Privacy.",
      };
    case "assistant":
      return assistantCopy(request);
    default:
      return setupCopy(request);
  }
}

function assistantCopy(request: AssistantConsentRequest): ConsentCopy {
  const target = request.model ? `${request.vendor} · ${request.model}` : request.vendor;
  if (request.purpose === "import") {
    return {
      title: `Send slide text to ${target}?`,
      body: `${request.vendor} reads the text extracted from the slides you pick and suggests terms and translations for the lecture context. Allowing this also lets it write notes and titles from your transcripts. Nothing else is sent, and audio never leaves this computer.`,
      primary: "Allow and import",
      secondary: "Extract on this computer",
      footnote: "To import without sending anything, choose Extract on this computer. You can turn this permission off anytime in Settings → AI assistant.",
    };
  }
  return {
    title: `Send this session to ${target}?`,
    body: `To write notes and titles, the transcript and the text of files you attach go to ${request.vendor}. Nothing else is sent, and audio never leaves this computer.`,
    primary: "Allow",
    footnote: "Applies to notes and titles from now on. Turn it off anytime in Settings → AI assistant.",
  };
}

function setupCopy(request: SetupMissingRequest): ConsentCopy {
  const action = purposeAction[request.purpose];
  if (request.need === "key") {
    const vendor = request.vendor ?? "the AI assistant";
    return {
      title: `Add a key for ${vendor}`,
      body: `${action} uses the ${vendor} key saved under Cloud providers. No key is saved yet, so nothing can be sent.`,
      primary: "Open Cloud providers",
    };
  }
  return {
    title: "Set up the AI assistant",
    body: `${action} needs a chat model you choose, with your own API key. Pick a provider and model in Settings → AI assistant; nothing is sent until you allow it.`,
    primary: "Open AI assistant settings",
  };
}
