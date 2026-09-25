import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearPendingSection,
  NAVIGATE_EVENT,
  type NavigationTarget,
  onNavigate,
  peekPendingAnchor,
  peekPendingSection,
  requestNavigation,
} from "./navigation";

describe("in-app navigation", () => {
  beforeEach(() => {
    // The views listen on `window`; an EventTarget is all they use.
    vi.stubGlobal("window", new EventTarget());
    clearPendingSection();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("reports an unhandled request and still remembers the section", () => {
    expect(requestNavigation({ view: "settings", section: "ai-assistant" })).toBe(false);
    expect(peekPendingSection()).toBe("ai-assistant");
  });

  it("delivers the target to every listener and marks it handled", () => {
    const main: NavigationTarget[] = [];
    const settings: NavigationTarget[] = [];
    const stopMain = onNavigate((target) => main.push(target));
    const stopSettings = onNavigate((target) => settings.push(target));
    expect(requestNavigation({ view: "settings", section: "cloud-providers" })).toBe(true);
    expect(main).toEqual([{ view: "settings", section: "cloud-providers" }]);
    expect(settings).toEqual(main);
    stopMain();
    stopSettings();
    expect(requestNavigation({ view: "history" })).toBe(false);
    expect(main).toHaveLength(1);
  });

  it("only remembers a section for Settings", () => {
    requestNavigation({ view: "settings", section: "translation" });
    requestNavigation({ view: "history", section: "translation" });
    expect(peekPendingSection()).toBeNull();
  });

  it("remembers the element to focus with the section", () => {
    requestNavigation({ view: "settings", section: "general", anchor: "settings-updates" });
    expect(peekPendingSection()).toBe("general");
    expect(peekPendingAnchor()).toBe("settings-updates");
    requestNavigation({ view: "settings", section: "privacy" });
    expect(peekPendingAnchor()).toBeNull();
    requestNavigation({ view: "settings", section: "general", anchor: "settings-updates" });
    clearPendingSection();
    expect(peekPendingSection()).toBeNull();
    expect(peekPendingAnchor()).toBeNull();
  });

  it("ignores malformed requests", () => {
    const seen: NavigationTarget[] = [];
    const stop = onNavigate((target) => seen.push(target));
    const handled = !window.dispatchEvent(new CustomEvent(NAVIGATE_EVENT, { detail: { view: "nowhere" }, cancelable: true }));
    expect(handled).toBe(false);
    expect(seen).toHaveLength(0);
    stop();
  });
});
