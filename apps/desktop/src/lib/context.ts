import type { ContextImportResult } from "../types";

/** Limits from docs/adr/0006. */
export const SESSION_CONTEXT_LIMIT = 2000;
export const GLOSSARY_LIMIT = 4000;

const normalized = (line: string) => line.trim().replace(/\s+/g, " ").toLowerCase();

/** True for the result of a cancelled picker (nothing read, nothing to say). */
export function isEmptyImport(result: ContextImportResult): boolean {
  return (
    !(result.context ?? "").trim() &&
    (result.terms ?? []).length === 0 &&
    (result.glossary ?? []).length === 0 &&
    (result.warnings ?? []).length === 0
  );
}

/** Lines an import adds: its context text, then any terms and glossary
 *  pairs the context does not already mention. */
export function importedLines(result: ContextImportResult): string[] {
  const lines = (result.context ?? "")
    .replace(/\0/g, "")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.trim() !== "");
  const mentioned = new Set(lines.map((line) => normalized(line.split("=")[0])));
  for (const pair of result.glossary ?? []) {
    const source = pair.source.replace(/\0/g, "").trim();
    const target = pair.target.replace(/\0/g, "").trim();
    if (!source || mentioned.has(normalized(source))) continue;
    lines.push(target ? `${source} = ${target}` : source);
    mentioned.add(normalized(source));
  }
  for (const term of result.terms ?? []) {
    const trimmed = term.replace(/\0/g, "").trim();
    if (!trimmed || mentioned.has(normalized(trimmed))) continue;
    lines.push(trimmed);
    mentioned.add(normalized(trimmed));
  }
  return lines;
}

/** Appends imported context to what the user already typed, skipping lines
 *  already present and dropping whole lines that would pass `limit`. */
export function mergeContext(
  current: string,
  result: ContextImportResult,
  limit = SESSION_CONTEXT_LIMIT,
): { text: string; added: number; skipped: number } {
  const existing = new Set(current.split("\n").map(normalized).filter(Boolean));
  let text = current.replace(/\s+$/, "");
  let added = 0;
  let skipped = 0;
  for (const line of importedLines(result)) {
    if (existing.has(normalized(line))) continue;
    const candidate = text ? `${text}\n${line}` : line;
    if (candidate.length > limit) {
      skipped += 1;
      continue;
    }
    text = candidate;
    existing.add(normalized(line));
    added += 1;
  }
  return { text, added, skipped };
}

/** Non-empty, non-comment lines, for the collapsed summary. */
export function contextLineCount(text: string): number {
  return text.split("\n").filter((line) => line.trim() && !line.trim().startsWith("#")).length;
}
