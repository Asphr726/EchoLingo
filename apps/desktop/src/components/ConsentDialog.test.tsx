import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ConsentRequest } from "../types";
import { ConsentDialog, consentCopy } from "./ConsentDialog";

function render(request: ConsentRequest | null) {
  return renderToStaticMarkup(
    <ConsentDialog request={request} onConfirm={async () => undefined} onDismiss={() => undefined} />,
  ).replace(/<!-- -->/g, "");
}

const checkboxes = (html: string) => html.match(/<input type="checkbox"[^>]*>/g) ?? [];

describe("ConsentDialog", () => {
  it("renders nothing inside the dialog while closed", () => {
    const html = render(null);
    expect(html).toContain('<dialog class="attachment-sheet consent-sheet"');
    expect(html).not.toContain("consent-title\">");
    expect(checkboxes(html)).toHaveLength(0);
  });

  it("lists one row per session upload, naming the vendor, with no checkboxes", () => {
    const html = render({
      kind: "session",
      audio: { vendor: "Deepgram", providerLabel: "Deepgram streaming (cloud)" },
      transcript: { vendor: "DeepL", providerLabel: "DeepL (cloud)" },
    });
    expect(html).toContain("Allow cloud processing?");
    expect(html).toContain("Audio from this session is uploaded to Deepgram for recognition.");
    expect(html).toContain("Transcript text is sent to DeepL for translation.");
    expect(html).toContain("Deepgram streaming (cloud)");
    expect(html).toContain("Saved as your default. Turn it off anytime in Settings → Privacy.");
    expect(html.match(/class="consent-row"/g)).toHaveLength(2);
    expect(checkboxes(html)).toHaveLength(0);
    expect(html).toContain(">Allow and continue</button>");
    expect(html).toContain(">Not now</button>");
  });

  it("asks only for what is missing", () => {
    const html = render({ kind: "session", transcript: { vendor: "DeepL", providerLabel: "DeepL (cloud)" } });
    expect(html.match(/class="consent-row"/g)).toHaveLength(1);
    expect(html).not.toContain("Audio from this session");
  });

  it("explains a connection test in its own words", () => {
    const request: ConsentRequest = {
      kind: "session",
      transcript: { vendor: "DeepL", providerLabel: "DeepL (cloud)" },
      note: "This test sends one fixed English sentence to DeepL, no lecture text and no audio.",
    };
    expect(consentCopy(request).body).toMatch(/^Testing needs the same permission/);
    expect(render(request)).toContain("one fixed English sentence to DeepL");
  });

  it("names the assistant vendor and model and keeps audio local", () => {
    const html = render({ kind: "assistant", vendor: "Qwen (Alibaba Model Studio)", model: "qwen-plus", purpose: "notes" });
    expect(html).toContain("Send this session to Qwen (Alibaba Model Studio) · qwen-plus?");
    expect(html).toContain("the transcript and the text of files you attach go to Qwen (Alibaba Model Studio)");
    expect(html).toContain("audio never leaves this computer");
    expect(html).toContain("Stays on this computer");
    expect(html).toContain(">Allow</button>");
    expect(checkboxes(html)).toHaveLength(0);
  });

  it("offers slide import on this computer instead of Not now", () => {
    const html = render({ kind: "assistant", vendor: "OpenAI", model: "", purpose: "import" });
    expect(html).toContain("Send slide text to OpenAI?");
    expect(html).toContain(">Extract on this computer</button>");
    expect(html).toContain(">Allow and import</button>");
    expect(html).not.toContain(">Not now</button>");
    expect(html).toContain("To import without sending anything, choose Extract on this computer.");
  });

  it("points setup that consent cannot fix to the right settings", () => {
    const key = render({ kind: "setup-missing", need: "key", vendor: "OpenAI", purpose: "title" });
    expect(key).toContain("Add a key for OpenAI");
    expect(key).toContain("Generating a title uses the OpenAI key");
    expect(key).toContain("Open Cloud providers");
    const provider = render({ kind: "setup-missing", need: "provider", purpose: "notes" });
    expect(provider).toContain("Set up the AI assistant");
    expect(provider).toContain("Open AI assistant settings");
    expect(provider).not.toContain(">Allow</button>");
  });
});
