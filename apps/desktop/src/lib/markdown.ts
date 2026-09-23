/** Pure helpers for rendering AI notes. No React here so the
 *  delimiter conversion, block splitting and heading parsing stay unit
 *  testable in the node test environment. */

// ---------------------------------------------------------------------------
// Time ranges in section headings: "## Texture segmentation (14:20–28:45)".

export interface SectionRange {
  title: string;
  startMs: number | null;
  endMs: number | null;
}

const CLOCK = String.raw`(\d{1,3}(?::\d{1,2}){1,2})`;
// Trailing "(mm:ss–mm:ss)"; also "-", "—", "~", "至", full-width
// parentheses and square brackets, which models emit in CJK output.
const TRAILING_RANGE = new RegExp(
  String.raw`\s*[(（\[【]\s*${CLOCK}\s*(?:[-‐‑–—~〜～]|至|to)\s*${CLOCK}\s*[)）\]】]\s*$`,
);

/** `mm:ss` (minutes may exceed 59) or `h:mm:ss` to milliseconds. */
export function parseClock(value: string): number | null {
  const parts = value.split(":").map((part) => Number(part));
  if (parts.some((part) => !Number.isInteger(part) || part < 0)) return null;
  if (parts.length === 2) {
    const [minutes, seconds] = parts;
    return seconds < 60 ? (minutes * 60 + seconds) * 1000 : null;
  }
  if (parts.length === 3) {
    const [hours, minutes, seconds] = parts;
    return minutes < 60 && seconds < 60 ? ((hours * 60 + minutes) * 60 + seconds) * 1000 : null;
  }
  return null;
}

/** Splits a trailing time range off `text`. Returns null when there is no
 *  well-formed range (the heading then renders unchanged). */
export function stripRangeSuffix(
  text: string,
): { text: string; startMs: number; endMs: number } | null {
  const match = TRAILING_RANGE.exec(text);
  if (!match) return null;
  const startMs = parseClock(match[1]);
  const endMs = parseClock(match[2]);
  if (startMs == null || endMs == null || endMs < startMs) return null;
  return { text: text.slice(0, match.index), startMs, endMs };
}

export function parseSectionRange(heading: string): SectionRange {
  const stripped = stripRangeSuffix(heading);
  if (!stripped) return { title: heading.trim(), startMs: null, endMs: null };
  return { title: stripped.text.trim(), startMs: stripped.startMs, endMs: stripped.endMs };
}

/** Transcript-style clock: `mm:ss`, minutes not wrapped at 60, matching the
 *  timestamps in the Transcript tab and in model headings. */
export function formatClock(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

export function formatRange(startMs: number, endMs: number): string {
  return `${formatClock(startMs)}–${formatClock(endMs)}`;
}

// ---------------------------------------------------------------------------
// Math delimiters. remark-math understands `$…$` and `$$…$$`; models also
// write `\(…\)` and `\[…\]`. A display block only renders as display math
// when its `$$` fences stand on their own lines.

const FENCE_OPEN = /^(\s{0,3})(`{3,}|~{3,})/;

interface Region {
  code: boolean;
  text: string;
}

/** Splits markdown into code (fenced blocks, inline code spans) and text
 *  regions so delimiter rewriting never touches code. */
function codeRegions(markdown: string): Region[] {
  const regions: Region[] = [];
  const push = (code: boolean, text: string) => {
    if (!text) return;
    const last = regions.at(-1);
    if (last && last.code === code) last.text += text;
    else regions.push({ code, text });
  };
  const lines = markdown.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  let fence: string | null = null;
  let prose = "";
  const flushProse = () => {
    for (const region of inlineCodeRegions(prose)) push(region.code, region.text);
    prose = "";
  };
  for (const line of lines) {
    if (fence) {
      push(true, line);
      const trimmed = line.trim();
      if (trimmed.startsWith(fence) && trimmed.replace(/[`~]/g, "") === "" && trimmed.length >= fence.length) {
        fence = null;
      }
      continue;
    }
    const open = FENCE_OPEN.exec(line);
    if (open) {
      flushProse();
      fence = open[2];
      push(true, line);
      continue;
    }
    prose += line;
  }
  flushProse();
  return regions;
}

function inlineCodeRegions(text: string): Region[] {
  const regions: Region[] = [];
  let index = 0;
  let plainStart = 0;
  while (index < text.length) {
    if (text[index] === "\\" && text[index + 1] === "`") {
      index += 2;
      continue;
    }
    if (text[index] !== "`") {
      index += 1;
      continue;
    }
    let run = 0;
    while (text[index + run] === "`") run += 1;
    const closer = findBacktickRun(text, index + run, run);
    if (closer < 0) {
      index += run;
      continue;
    }
    if (index > plainStart) regions.push({ code: false, text: text.slice(plainStart, index) });
    regions.push({ code: true, text: text.slice(index, closer + run) });
    index = closer + run;
    plainStart = index;
  }
  if (plainStart < text.length) regions.push({ code: false, text: text.slice(plainStart) });
  return regions;
}

function findBacktickRun(text: string, from: number, length: number): number {
  let index = from;
  while (index < text.length) {
    if (text[index] !== "`") {
      index += 1;
      continue;
    }
    let run = 0;
    while (text[index + run] === "`") run += 1;
    if (run === length) return index;
    index += run;
  }
  return -1;
}

/** Leading indentation and block markers (`>`, a list bullet) of the line
 *  that contains `offset`, when only they precede it; otherwise null. */
function linePrefixBefore(text: string, offset: number): string | null {
  const lineStart = text.lastIndexOf("\n", offset - 1) + 1;
  const before = text.slice(lineStart, offset);
  return /^[ \t>]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?$/.test(before) ? before : null;
}

function restOfLineIsBlank(text: string, offset: number): boolean {
  const lineEnd = text.indexOf("\n", offset);
  return text.slice(offset, lineEnd < 0 ? undefined : lineEnd).trim() === "";
}

/** Indentation that keeps continuation lines inside the list item or quote
 *  that `prefix` opened. */
function continuationIndent(prefix: string): string {
  return prefix.replace(/[-*+]|\d{1,9}[.)]/g, (marker) => " ".repeat(marker.length));
}

/** `$$` fence lines around `body`; the caller emits `prefix` itself. */
function displayBody(prefix: string, body: string): string {
  const indent = continuationIndent(prefix);
  const lines = body.trim().split("\n").map((line) => line.trim());
  return `$$\n${lines.map((line) => `${indent}${line}`).join("\n")}\n${indent}$$`;
}

function rewriteMath(text: string): string {
  // \[ … \] → $$ … $$, as a display block when it stands alone on its line.
  let out = text.replace(
    /(^|[^\\])\\\[((?:(?!\n[ \t>]*\n)[\s\S])+?)\\\]/g,
    (whole: string, lead: string, body: string, offset: number, source: string) => {
      const start = offset + lead.length;
      const prefix = linePrefixBefore(source, start);
      if (prefix !== null && restOfLineIsBlank(source, offset + whole.length)) {
        return `${lead}${displayBody(prefix, body)}`;
      }
      return `${lead}$$${body.trim()}$$`;
    },
  );
  // \( … \) → $…$
  out = out.replace(/(^|[^\\])\\\(((?:(?!\n[ \t>]*\n)[\s\S])+?)\\\)/g, (_whole: string, lead: string, body: string) => `${lead}$${body.trim()}$`);
  // A one-line "$$ … $$" on its own line → display block.
  out = out.replace(
    /^([ \t>]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?)\$\$([^$\n]+?)\$\$[ \t]*$/gm,
    (_whole: string, prefix: string, body: string) => `${prefix}${displayBody(prefix, body)}`,
  );
  return out;
}

const MASK = "\u0000";

/** Converts `\( \)` to `$ $` and `\[ \]` to `$$ $$` outside code, and turns
 *  one-line `$$…$$` paragraphs into display blocks. */
export function normalizeMath(markdown: string): string {
  if (!markdown.includes("\\(") && !markdown.includes("\\[") && !markdown.includes("$$")) return markdown;
  // Code is masked (not skipped) so line starts stay visible to the
  // display-block rule; a trailing newline stays outside the mask.
  const stash: string[] = [];
  const masked = codeRegions(markdown.replaceAll(MASK, ""))
    .map((region) => {
      if (!region.code) return region.text;
      const trailing = region.text.endsWith("\n") ? "\n" : "";
      stash.push(region.text.slice(0, region.text.length - trailing.length));
      return `${MASK}${stash.length - 1}${MASK}${trailing}`;
    })
    .join("");
  return rewriteMath(masked).replace(/\u0000(\d+)\u0000/g, (_whole, index: string) => stash[Number(index)] ?? "");
}

// ---------------------------------------------------------------------------
// Top-level block splitting. Each block renders as its own memoized
// markdown tree, so a streaming update re-renders only the tail.

const LIST_ITEM = /^\s{0,3}(?:[-*+]|\d{1,9}[.)])(?:\s|$)/;
const ATX_HEADING = /^#{1,6}(?:\s|$)/;
const MATH_FENCE = /^[ \t>]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?\$\$/;

export function splitBlocks(markdown: string): string[] {
  const blocks: string[] = [];
  let current: string[] = [];
  let fence: string | null = null;
  let sawBlank = false;
  let isList = false;

  const flush = () => {
    while (current.length && current[current.length - 1].trim() === "") current.pop();
    if (current.length) blocks.push(current.join("\n"));
    current = [];
    isList = false;
    sawBlank = false;
  };

  for (const line of markdown.replace(/\r\n?/g, "\n").split("\n")) {
    if (fence) {
      current.push(line);
      const trimmed = line.trim();
      const closes =
        fence === "$$"
          ? trimmed === "$$" || (trimmed.endsWith("$$") && !trimmed.startsWith("$$"))
          : trimmed.startsWith(fence) && trimmed.replace(/[`~]/g, "") === "";
      if (closes) fence = null;
      continue;
    }
    if (line.trim() === "") {
      if (current.length) {
        current.push(line);
        sawBlank = true;
      }
      continue;
    }
    if (sawBlank) {
      const continues = isList && (/^\s/.test(line) || LIST_ITEM.test(line));
      if (!continues) flush();
      sawBlank = false;
    }
    if (ATX_HEADING.test(line)) {
      flush();
      blocks.push(line);
      continue;
    }
    if (current.length === 0) isList = LIST_ITEM.test(line);
    current.push(line);
    const open = FENCE_OPEN.exec(line);
    if (open) {
      fence = open[2];
    } else if (MATH_FENCE.test(line)) {
      const after = line.slice(line.indexOf("$$") + 2);
      if (!after.includes("$$")) fence = "$$";
    }
  }
  flush();
  return blocks;
}

// ---------------------------------------------------------------------------
// Outline

export interface NotesSection extends SectionRange {
  index: number;
}

/** Splits a leading `# Title` off the notes. */
export function splitTitle(markdown: string): { title: string | null; body: string } {
  const match = /^\s*#\s+(.+?)\s*#*\s*(?:\n|$)/.exec(markdown);
  if (!match) return { title: null, body: markdown };
  return { title: match[1].trim(), body: markdown.slice(match[0].length) };
}

const SECTION_HEADING = /^##\s+(.+?)\s*#*\s*$/;

/** True for a block that is a `##` section heading; the renderer and the
 *  outline number sections with this same test. */
export function isSectionHeading(block: string): boolean {
  return SECTION_HEADING.test(block);
}

/** `##` headings among already split top-level blocks. */
export function sectionsFromBlocks(blocks: string[]): NotesSection[] {
  const sections: NotesSection[] = [];
  for (const block of blocks) {
    const match = SECTION_HEADING.exec(block);
    if (!match) continue;
    const text = match[1].replace(/\*\*|__|`/g, "");
    sections.push({ ...parseSectionRange(text), index: sections.length });
  }
  return sections;
}

/** `##` headings in document order, outside code. */
export function outlineSections(markdown: string): NotesSection[] {
  return sectionsFromBlocks(splitBlocks(markdown));
}

export interface NotesDocument {
  /** Text of the leading `# Title`, if any. */
  title: string | null;
  /** Top-level blocks after the title, math delimiters normalized. */
  blocks: string[];
  sections: NotesSection[];
}

/** Everything the notes view needs from one pass over the markdown. */
export function parseNotesDocument(markdown: string): NotesDocument {
  const { title, body } = splitTitle(normalizeMath(markdown.replace(/\r\n?/g, "\n")));
  const blocks = splitBlocks(body);
  return { title, blocks, sections: sectionsFromBlocks(blocks) };
}

// ---------------------------------------------------------------------------
// Word estimate shown before sending a transcript.

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/gu;

/** Space-delimited words plus one per CJK character. */
export function estimateWords(text: string): number {
  const cjk = text.match(CJK)?.length ?? 0;
  const rest = text.replace(CJK, " ").trim();
  const words = rest ? rest.split(/\s+/).filter((word) => /[\p{L}\p{N}]/u.test(word)).length : 0;
  return cjk + words;
}

// ---------------------------------------------------------------------------
// hast plugins (loosely typed so no transitive @types package is imported).

interface HastNode {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

const SKIP_TAGS = new Set(["code", "pre", "math", "svg", "script", "style"]);

function classList(node: HastNode): string[] {
  const value = node.properties?.className;
  return Array.isArray(value) ? value.map(String) : typeof value === "string" ? value.split(/\s+/) : [];
}

/** Adds `data-start-ms`/`data-end-ms` to headings that end in a time range
 *  and removes the range text from the heading. */
export function rehypeSectionRanges() {
  return (tree: HastNode) => {
    const visit = (node: HastNode) => {
      if (node.type === "element" && /^h[1-6]$/.test(node.tagName ?? "")) {
        const last = node.children?.at(-1);
        if (last?.type === "text" && typeof last.value === "string") {
          const stripped = stripRangeSuffix(last.value);
          if (stripped) {
            last.value = stripped.text.replace(/\s+$/, "");
            node.properties = { ...(node.properties ?? {}), dataStartMs: stripped.startMs, dataEndMs: stripped.endMs };
          }
        }
        return;
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

const UNCERTAIN_SOURCE = String.raw`\s?[(（]\s?[?？]\s?[)）]`;
const HAS_UNCERTAIN = new RegExp(UNCERTAIN_SOURCE);
const LATIN_TOKEN = /[\p{Script=Latin}\p{Script=Greek}\p{N}][\p{Script=Latin}\p{Script=Greek}\p{N}'’.\-]*$/u;
export const UNCERTAIN_TITLE = "Uncertain recognition: check against the recording";

/** Wraps `(?)` marks (and the Latin-script word before them) so uncertain
 *  corrections are visible. Code and math are left untouched. */
export function rehypeUncertain() {
  return (tree: HastNode) => {
    const visit = (node: HastNode) => {
      if (node.type === "element") {
        if (SKIP_TAGS.has(node.tagName ?? "") || classList(node).some((name) => name.startsWith("katex") || name.startsWith("math"))) {
          return;
        }
      }
      if (!node.children) return;
      const next: HastNode[] = [];
      for (const child of node.children) {
        if (child.type === "text" && typeof child.value === "string" && HAS_UNCERTAIN.test(child.value)) {
          next.push(...splitUncertain(child.value));
        } else {
          visit(child);
          next.push(child);
        }
      }
      node.children = next;
    };
    visit(tree);
  };
}

function splitUncertain(value: string): HastNode[] {
  const nodes: HastNode[] = [];
  let cursor = 0;
  for (const match of value.matchAll(new RegExp(UNCERTAIN_SOURCE, "g"))) {
    const start = match.index ?? 0;
    let before = value.slice(cursor, start);
    const token = LATIN_TOKEN.exec(before)?.[0] ?? "";
    before = before.slice(0, before.length - token.length);
    if (before) nodes.push({ type: "text", value: before });
    const mark: HastNode = {
      type: "element",
      tagName: "span",
      properties: { className: ["uncertain-mark"], ariaHidden: "true" },
      children: [{ type: "text", value: "?" }],
    };
    nodes.push({
      type: "element",
      tagName: "span",
      properties: { className: token ? ["uncertain"] : ["uncertain", "uncertain--mark-only"], title: UNCERTAIN_TITLE },
      children: [
        ...(token ? [{ type: "text", value: token }] : []),
        mark,
        { type: "element", tagName: "span", properties: { className: ["visually-hidden"] }, children: [{ type: "text", value: " (uncertain)" }] },
      ],
    });
    cursor = start + match[0].length;
  }
  if (cursor < value.length) nodes.push({ type: "text", value: value.slice(cursor) });
  return nodes;
}
