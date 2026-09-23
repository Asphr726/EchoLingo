import { describe, expect, it } from "vitest";
import { contextLineCount, importedLines, isEmptyImport, mergeContext } from "./context";

const result = {
  context: "Texture perception and pre-attentive vision\nBéla Julesz\n",
  terms: ["saccade", "Béla Julesz", "texton"],
  glossary: [
    { source: "pre-attentive vision", target: "前注意视觉" },
    { source: "texton", target: "纹理基元" },
  ],
  warnings: [],
};

describe("lecture context import", () => {
  it("adds glossary pairs and terms the context does not mention yet", () => {
    expect(importedLines(result)).toEqual([
      "Texture perception and pre-attentive vision",
      "Béla Julesz",
      "pre-attentive vision = 前注意视觉",
      "texton = 纹理基元",
      "saccade",
    ]);
  });

  it("appends to typed context without duplicating lines", () => {
    const merged = mergeContext("CS180 lecture 12\nbéla  julesz", result);
    expect(merged.text).toBe(
      "CS180 lecture 12\nbéla  julesz\nTexture perception and pre-attentive vision\npre-attentive vision = 前注意视觉\ntexton = 纹理基元\nsaccade",
    );
    expect(merged.added).toBe(4);
    expect(merged.skipped).toBe(0);
  });

  it("drops whole lines that would pass the limit", () => {
    const merged = mergeContext("", result, 60);
    expect(merged.text).toBe("Texture perception and pre-attentive vision\nBéla Julesz");
    expect(merged.text.length).toBeLessThanOrEqual(60);
    expect(merged.skipped).toBe(3);
  });

  it("recognizes a cancelled picker and strips NUL from imported text", () => {
    expect(isEmptyImport({ context: "", terms: [], glossary: [], warnings: [] })).toBe(true);
    expect(isEmptyImport({ context: "", terms: [], glossary: [], warnings: ["Slide 3 has no text"] })).toBe(false);
    expect(importedLines({ context: "Topic\0 A", terms: ["te\0xton"], glossary: [], warnings: [] })).toEqual([
      "Topic A",
      "texton",
    ]);
  });

  it("counts lines that are not comments", () => {
    expect(contextLineCount("# comment\nTopic\n\nterm = 译名\n")).toBe(2);
  });
});
