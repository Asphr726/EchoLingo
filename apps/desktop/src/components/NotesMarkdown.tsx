import "katex/dist/katex.min.css";
import { api } from "../lib/bridge";
import { Check, Copy } from "@phosphor-icons/react";
import { createContext, memo, type ReactNode, useContext, useEffect, useRef, useState } from "react";
import type { JSX } from "react";
import Markdown, { type Components, type ExtraProps, type Options } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { formatClock, formatRange, isSectionHeading, rehypeSectionRanges, rehypeUncertain } from "../lib/markdown";

/** Renders AI notes safely: no raw HTML (react-markdown shows it as text),
 *  links never navigate the app window, only registered local figures load, and KaTeX never
 *  throws. Each top-level block is its own memoized tree, so while notes
 *  stream in only the changed tail re-renders. */

export interface NotesActions {
  /** A section time chip was pressed. */
  onJump?: (startMs: number, sectionIndex: number | null) => void;
}

const AssetSessionContext = createContext<string | undefined>(undefined);

const NotesActionsContext = createContext<NotesActions>({});
/** Index of the `##` section a block heads, for anchors. */
const SectionContext = createContext<number | null>(null);

export const sectionAnchorId = (index: number) => `notes-section-${index}`;

const remarkPlugins: Options["remarkPlugins"] = [remarkGfm, [remarkMath, { singleDollarTextMath: true }]];
const rehypePlugins: Options["rehypePlugins"] = [
  [rehypeKatex, { throwOnError: false, strict: "ignore", trust: false, maxSize: 20, maxExpand: 500 }],
  rehypeSectionRanges,
  rehypeUncertain,
];

type HeadingProps = JSX.IntrinsicElements["h1"] & ExtraProps;

function numeric(value: unknown): number | null {
  const number = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(number) ? number : null;
}

function makeHeading(level: number) {
  // The notes title is an h3 under the session title (h2); `##` sections
  // are h4 and deeper levels follow.
  const Tag = `h${Math.min(6, level + 2)}` as "h3" | "h4" | "h5" | "h6";
  return function NotesHeading({ children, node }: HeadingProps) {
    const section = useContext(SectionContext);
    const { onJump } = useContext(NotesActionsContext);
    const startMs = numeric(node?.properties?.dataStartMs);
    const endMs = numeric(node?.properties?.dataEndMs);
    const isSection = level === 2 && section != null;
    return (
      <Tag
        className={`notes-heading notes-heading--${level}`}
        id={isSection ? sectionAnchorId(section) : undefined}
        data-section-index={isSection ? section : undefined}
      >
        <span className="notes-heading-text">{children}</span>
        {startMs != null && endMs != null && (
          <TimeChip startMs={startMs} endMs={endMs} onJump={onJump ? () => onJump(startMs, isSection ? section : null) : undefined} />
        )}
      </Tag>
    );
  };
}

export function TimeChip({ endMs, onJump, startMs }: { startMs: number; endMs: number; onJump?: () => void }) {
  const label = formatRange(startMs, endMs);
  if (!onJump) return <span className="time-chip time-chip--static">{label}</span>;
  return (
    <button
      className="time-chip"
      type="button"
      title="Show this part of the transcript"
      aria-label={`${label}, show the transcript from ${formatClock(startMs)}`}
      onClick={onJump}
    >
      {label}
    </button>
  );
}

function textOf(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node === "object" && "props" in node) {
    return textOf((node.props as { children?: ReactNode }).children);
  }
  return "";
}

/** Links are shown, never followed: the text, the address when it differs,
 *  and a copy button. */
function NotesLink({ children, href }: JSX.IntrinsicElements["a"] & ExtraProps) {
  const url = typeof href === "string" ? href.trim() : "";
  const external = /^(https?:|mailto:)/i.test(url);
  const text = textOf(children).trim();
  const showAddress = external && text !== url && text !== url.replace(/^mailto:/i, "");
  return (
    <span className="notes-link">
      <span className="notes-link-text">{children}</span>
      {showAddress && <span className="notes-link-url"> ({url})</span>}
      {external && <CopyLinkButton url={url} />}
    </span>
  );
}

function CopyLinkButton({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <button
      className="notes-link-copy"
      type="button"
      aria-label={copied ? "Link address copied" : "Copy link address"}
      title={copied ? "Copied" : "Copy link address"}
      onClick={() => {
        void navigator.clipboard?.writeText(url).then(() => {
          setCopied(true);
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => setCopied(false), 1600);
        });
      }}
    >
      {copied ? <Check size={11} weight="bold" aria-hidden="true" /> : <Copy size={11} weight="regular" aria-hidden="true" />}
    </button>
  );
}

const components: Components = {
  h1: makeHeading(1),
  h2: makeHeading(2),
  h3: makeHeading(3),
  h4: makeHeading(4),
  h5: makeHeading(5),
  h6: makeHeading(6),
  a: NotesLink,
  // Only Rust-owned figures can resolve to image bytes.
  img: ({ alt, src }) => <NoteFigure alt={alt} src={src} />,
  table: ({ children }) => (
    <div className="notes-table-scroll" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
};

const NotesBlock = memo(function NotesBlock({ section, source }: { source: string; section: number | null }) {
  return (
    <SectionContext.Provider value={section}>
      <Markdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components} urlTransform={(url) => /^echolingo-asset:[0-9a-f-]{36}$/.test(url) ? url : /^(https?:|mailto:)/i.test(url) ? url : ""}>
        {source}
      </Markdown>
    </SectionContext.Provider>
  );
});

export function NotesMarkdown({
  blocks,
  sessionId,
  lang,
  onJump,
}: {
  /** Top-level blocks from `parseNotesDocument`. */
  blocks: string[];
  sessionId?: string;
  lang?: string;
  onJump?: NotesActions["onJump"];
}) {
  const actions = useStableActions(onJump);
  let section = -1;
  return (
    <AssetSessionContext.Provider value={sessionId}>
    <NotesActionsContext.Provider value={actions}>
      <div className="notes-body" lang={lang}>
        {blocks.map((block, index) => {
          const heading = isSectionHeading(block);
          if (heading) section += 1;
          return <NotesBlock key={index} source={block} section={heading ? section : null} />;
        })}
      </div>
    </NotesActionsContext.Provider>
    </AssetSessionContext.Provider>
  );
}

/** A context value that never changes identity, so memoized blocks do not
 *  re-render when the parent passes a new callback. */
function useStableActions(onJump: NotesActions["onJump"]): NotesActions {
  const latest = useRef(onJump);
  latest.current = onJump;
  const [actions] = useState<NotesActions>(() => ({
    onJump: (startMs, sectionIndex) => latest.current?.(startMs, sectionIndex),
  }));
  return onJump ? actions : NO_ACTIONS;
}

const NO_ACTIONS: NotesActions = {};


function NoteFigure({ src, alt }: { src?: string | Blob; alt?: string }) {
  const session = useContext(AssetSessionContext);
  const id = typeof src === "string" && /^echolingo-asset:[0-9a-f-]{36}$/.test(src) ? src.slice(16) : null;
  const [data, setData] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    let live = true;
    setData(null);
    if (session && id) void api.readNoteAsset(session, id).then((value) => {
      if (live && value.startsWith("data:image/jpeg;base64,")) setData(value);
    }).catch(() => {});
    return () => { live = false; };
  }, [session, id]);
  useEffect(() => {
    if (expanded) dialog.current?.showModal();
    else dialog.current?.close();
  }, [expanded]);
  if (!data) return <span className="notes-image">[Image{alt ? `: ${alt}` : ""}]</span>;
  return <span className="note-figure">
    <button type="button" className="note-figure-button" aria-label={`Enlarge ${alt || "figure"}`} onClick={() => setExpanded(true)}><img src={data} alt={alt || "Course figure"} loading="lazy" /></button>
    <dialog ref={dialog} className="note-figure-dialog" onCancel={() => setExpanded(false)} onClose={() => setExpanded(false)}>
      <button type="button" onClick={() => setExpanded(false)}>Close</button><img src={data} alt={alt || "Course figure"} />
    </dialog>
  </span>;
}
