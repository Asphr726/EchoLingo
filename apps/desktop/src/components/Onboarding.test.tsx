import { describe, expect, it } from "vitest";
import type { AudioDevice } from "../types";
import { permissionCopy } from "./Onboarding";

const systemAudio: AudioDevice = {
  id: "macos-screen-capture-kit",
  name: "System Audio…",
  kind: "system_audio",
  is_default: false,
  available: true,
  requires_picker: true,
};

describe("permission step copy", () => {
  it("keeps the macOS wording", () => {
    expect(permissionCopy("mac", "microphone").description).toBe("macOS requires microphone permission before EchoLingo can listen.");
    expect(permissionCopy("mac", "system_audio", systemAudio).description).toBe(
      "macOS requires screen/audio capture permission. You still choose a display or app for each capture.",
    );
    expect(permissionCopy("mac", "microphone").denied).toContain("macOS System Settings → Privacy & Security");
  });

  it("points Windows users to the microphone privacy switches", () => {
    const copy = permissionCopy("windows", "microphone");
    expect(copy.description).toContain("Settings → Privacy & security → Microphone");
    expect(copy.description).toContain("desktop apps");
    expect(copy.denied).toContain("Let desktop apps access your microphone");
  });

  it("says Windows loopback captures the default output device", () => {
    const loopback = { ...systemAudio, id: "windows-wasapi-loopback", requires_picker: false };
    expect(permissionCopy("windows", "system_audio", loopback).description).toMatch(/captures what the default output device plays/);
  });

  it("tells Linux users there is no prompt and where to look instead", () => {
    const copy = permissionCopy("linux", "microphone");
    expect(copy.description).toMatch(/^Linux does not ask for microphone permission/);
    expect(copy.description).toContain("system sound settings");
    const unavailable = { ...systemAudio, id: "system-audio-unavailable", available: false, requires_picker: false };
    expect(permissionCopy("linux", "system_audio", unavailable).description).toMatch(/not available on this platform yet/);
  });
});
