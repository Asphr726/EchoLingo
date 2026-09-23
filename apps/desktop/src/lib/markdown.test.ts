import { describe, expect, it } from "vitest";
import {
  estimateWords,
  formatClock,
  normalizeMath,
  outlineSections,
  parseClock,
  parseNotesDocument,
  parseSectionRange,
  splitBlocks,
  splitTitle,
} from "./markdown";

describe("normalizeMath", () => {
  it("converts inline \\( \\) to single dollars", () => {
    expect(normalizeMath("Energy \\( E = mc^2 \\) holds.")).toBe("Energy $E = mc^2$ holds.");
  });

  it("turns a standalone \\[ \\] into a fenced display block", () => {
    expect(normalizeMath("Therefore:\n\n\\[\n  a^2 + b^2 = c^2\n\\]\n\nDone.")).toBe(
      "Therefore:\n\n$$\na^2 + b^2 = c^2\n$$\n\nDone.",
    );
    expect(normalizeMath("\\[ x = 1 \\]")).toBe("$$\nx = 1\n$$");
  });

  it("keeps an inline \\[ \\] inside a sentence inline", () => {
    expect(normalizeMath("so \\[x+1\\] is odd")).toBe("so $$x+1$$ is odd");
  });

  it("keeps display math inside a list item indented", () => {
    expect(normalizeMath("- Area:\n  \\[\\pi r^2\\]\n- Next")).toBe("- Area:\n  $$\n  \\pi r^2\n  $$\n- Next");
    expect(normalizeMath("1. $$V = \\max(R,G,B)$$")).toBe("1. $$\n   V = \\max(R,G,B)\n   $$");
  });

  it("splits a one-line $$ … $$ paragraph into display fences", () => {
    expect(normalizeMath("$$ RT = a + b n $$")).toBe("$$\nRT = a + b n\n$$");
    expect(normalizeMath("> $$x$$")).toBe("> $$\n> x\n> $$");
  });

  it("leaves fenced and inline code untouched", () => {
    const fenced = "```latex\n\\( not math \\)\n\\[ also not \\]\n```\n\nText \\(x\\).";
    expect(normalizeMath(fenced)).toBe("```latex\n\\( not math \\)\n\\[ also not \\]\n```\n\nText $x$.");
    expect(normalizeMath("Use `\\(x\\)` or \\(y\\).")).toBe("Use `\\(x\\)` or $y$.");
    expect(normalizeMath("Double ``a `\\(b\\)` c`` then \\(d\\)")).toBe("Double ``a `\\(b\\)` c`` then $d$");
  });

  it("does not treat an escaped backslash as a delimiter or cross paragraphs", () => {
    expect(normalizeMath("a \\\\(literal) b")).toBe("a \\\\(literal) b");
    expect(normalizeMath("stray \\( open\n\nnext paragraph \\) close")).toBe(
      "stray \\( open\n\nnext paragraph \\) close",
    );
  });

  it("is the identity for markdown without math delimiters", () => {
    const text = "# Title\n\nPlain $5 price and `code`.";
    expect(normalizeMath(text)).toBe(text);
  });
});

describe("parseSectionRange", () => {
  it("splits a trailing mm:ss range with en dash, hyphen and em dash", () => {
    expect(parseSectionRange("Texture segmentation (14:20–28:45)")).toEqual({
      title: "Texture segmentation",
      startMs: 860_000,
      endMs: 1_725_000,
    });
    expect(parseSectionRange("Intro (00:00-08:40)")).toMatchObject({ startMs: 0, endMs: 520_000 });
    expect(parseSectionRange("Intro (00:00 — 08:40)")).toMatchObject({ title: "Intro", endMs: 520_000 });
  });

  it("accepts minutes beyond 59 and h:mm:ss", () => {
    expect(parseSectionRange("Wrap-up (58:10–64:05)")).toMatchObject({ startMs: 3_490_000, endMs: 3_845_000 });
    expect(parseSectionRange("Late (1:02:03–1:10:00)")).toMatchObject({ startMs: 3_723_000, endMs: 4_200_000 });
  });

  it("accepts full-width parentheses and brackets from CJK output", () => {
    expect(parseSectionRange("前注意视觉（21:15–34:50）")).toEqual({
      title: "前注意视觉",
      startMs: 1_275_000,
      endMs: 2_090_000,
    });
    expect(parseSectionRange("HSV [08:40–21:15]")).toMatchObject({ title: "HSV", startMs: 520_000 });
  });

  it("leaves headings without a well-formed trailing range unchanged", () => {
    expect(parseSectionRange("Key terms")).toEqual({ title: "Key terms", startMs: null, endMs: null });
    expect(parseSectionRange("Range (12:00–10:00)")).toMatchObject({ startMs: null });
    expect(parseSectionRange("Bad (12:75–13:00)")).toMatchObject({ startMs: null });
    expect(parseSectionRange("(01:00–02:00) leading is not a suffix")).toMatchObject({ startMs: null });
  });

  it("formats clocks without wrapping minutes", () => {
    expect(parseClock("64:05")).toBe(3_845_000);
    expect(formatClock(3_845_000)).toBe("64:05");
    expect(formatClock(0)).toBe("00:00");
  });
});

describe("splitBlocks", () => {
  it("splits on blank lines and isolates headings", () => {
    expect(splitBlocks("# Title\nSummary line\n\n## One (00:00–01:00)\nBody\n\nMore")).toEqual([
      "# Title",
      "Summary line",
      "## One (00:00–01:00)",
      "Body",
      "More",
    ]);
  });

  it("keeps fenced code and display math with blank lines in one block", () => {
    const code = "```\na\n\nb\n```";
    const math = "$$\nx\n\ny\n$$";
    expect(splitBlocks(`${code}\n\n${math}\n\nTail`)).toEqual([code, math, "Tail"]);
  });

  it("keeps loose lists and indented continuations together", () => {
    const list = "- one\n\n- two\n\n  continued paragraph\n\n1. three";
    expect(splitBlocks(`${list}\n\nAfter`)).toEqual([list, "After"]);
  });

  it("keeps a table in one block and tolerates CRLF", () => {
    expect(splitBlocks("| a | b |\r\n| - | - |\r\n| 1 | 2 |\r\n\r\nNext")).toEqual([
      "| a | b |\n| - | - |\n| 1 | 2 |",
      "Next",
    ]);
  });

  it("returns stable leading blocks while the tail streams", () => {
    const partial = splitBlocks("# T\n\nFirst paragraph.\n\n## Sec (00:00–01:00)\n\n- item one\n- item tw");
    const later = splitBlocks("# T\n\nFirst paragraph.\n\n## Sec (00:00–01:00)\n\n- item one\n- item two\n- item three");
    expect(later.slice(0, 3)).toEqual(partial.slice(0, 3));
    expect(later).toHaveLength(partial.length);
  });

  it("does not open a math fence on a one-line $$ … $$", () => {
    expect(splitBlocks("$$x$$\n\nnext")).toEqual(["$$x$$", "next"]);
  });
});

describe("outline", () => {
  it("extracts the title and ## sections with ranges", () => {
    const markdown = "# Colour spaces\n\nIntro.\n\n## RGB cube (00:00–08:40)\n\ntext\n\n### Detail\n\n```\n## not a heading\n```\n\n## Key **terms**";
    expect(splitTitle(markdown).title).toBe("Colour spaces");
    expect(outlineSections(splitTitle(markdown).body)).toEqual([
      { title: "RGB cube", startMs: 0, endMs: 520_000, index: 0 },
      { title: "Key terms", startMs: null, endMs: null, index: 1 },
    ]);
  });

  it("returns no title when the notes do not start with # ", () => {
    expect(splitTitle("## Only sections")).toEqual({ title: null, body: "## Only sections" });
  });
});

describe("parseNotesDocument", () => {
  it("splits the title, normalizes math and numbers sections like the renderer", () => {
    const doc = parseNotesDocument("# Notes\r\n\r\nIntro \\(x\\).\r\n\r\n## A (00:00–01:00)\r\n\r\nText\r\n\r\n## B\r\n");
    expect(doc.title).toBe("Notes");
    expect(doc.blocks).toEqual(["Intro $x$.", "## A (00:00–01:00)", "Text", "## B"]);
    expect(doc.sections.map((section) => [section.index, section.title, section.startMs])).toEqual([
      [0, "A", 0],
      [1, "B", null],
    ]);
  });

  it("handles notes without a title or sections", () => {
    expect(parseNotesDocument("")).toEqual({ title: null, blocks: [], sections: [] });
    expect(parseNotesDocument("Just text").blocks).toEqual(["Just text"]);
  });
});

describe("estimateWords", () => {
  it("counts space-delimited words and CJK characters", () => {
    expect(estimateWords("Look at that line, okay?")).toBe(5);
    expect(estimateWords("前注意视觉 texture")).toBe(6);
    expect(estimateWords("  — ")).toBe(0);
  });
});
