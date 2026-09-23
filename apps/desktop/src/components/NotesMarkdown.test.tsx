import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { parseNotesDocument } from "../lib/markdown";
import { NotesMarkdown } from "./NotesMarkdown";

function render(markdown: string, onJump?: () => void) {
  const doc = parseNotesDocument(markdown);
  return renderToStaticMarkup(<NotesMarkdown blocks={doc.blocks} onJump={onJump} />);
}

describe("NotesMarkdown", () => {
  it("renders section headings with a time chip and an anchor", () => {
    const html = render("# T\n\n## Texture (14:20–28:45)\n\nBody", () => undefined);
    expect(html).toContain('id="notes-section-0"');
    expect(html).toMatch(/<h4[^>]*class="notes-heading notes-heading--2"/);
    expect(html).toContain('<button class="time-chip"');
    expect(html).toContain("14:20–28:45</button>");
    expect(html).not.toContain("(14:20–28:45)");
  });

  it("shows a static chip when there is no transcript to jump to", () => {
    expect(render("## A (00:00–01:00)")).toContain('<span class="time-chip time-chip--static">00:00–01:00</span>');
  });

  it("typesets inline and display math, including \\( \\) and \\[ \\] delimiters", () => {
    const html = render("Energy \\(E = mc^2\\).\n\n\\[\na^2 + b^2 = c^2\n\\]");
    expect(html).toContain('class="katex"');
    expect(html).toContain('class="katex-display"');
    expect(html).not.toContain("\\(");
  });

  it("does not throw on invalid LaTeX", () => {
    expect(() => render("Broken $\\frac{1}{$ math")).not.toThrow();
  });

  it("neutralizes links and images and shows raw HTML as text", () => {
    const html = render(
      "See [the course](https://example.edu/cs180) or https://example.edu.\n\n![diagram](https://tracker.example/pixel.png)\n\n<script>alert(1)</script>",
    );
    expect(html).not.toMatch(/<a\s/);
    expect(html).not.toMatch(/<img\s/);
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("(https://example.edu/cs180)");
    expect(html).toContain('aria-label="Copy link address"');
    expect(html).toContain("[Image: diagram]");
  });

  it("marks (?) as uncertain outside code", () => {
    const html = render("The speaker said Julesz (?) and 1990 年 (?).\n\n`keep (?) here`");
    expect(html).toContain('<span class="uncertain" title="Uncertain recognition: check against the recording">Julesz');
    expect(html).toContain("uncertain uncertain--mark-only");
    expect(html).toContain("<code>keep (?) here</code>");
  });

  it("renders GFM tables inside a scroll container", () => {
    const html = render("| a | b |\n| - | - |\n| 1 | 2 |");
    expect(html).toContain('<div class="notes-table-scroll" tabindex="0"><table>');
  });
});
