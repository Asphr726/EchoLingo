import { describe, expect, it } from "vitest";
import type { AudioDevice } from "../types";
import { restoreAudioSelection, systemAudioAvailable, systemAudioDevice } from "./audio";

const microphone: AudioDevice = {
  id: "mic-default",
  name: "Built-in microphone",
  kind: "microphone",
  is_default: true,
  available: true,
  requires_picker: false,
};
const usbMicrophone: AudioDevice = { ...microphone, id: "mic-usb", name: "USB microphone", is_default: false };
const macSystemAudio: AudioDevice = {
  id: "macos-screen-capture-kit",
  name: "System Audio…",
  kind: "system_audio",
  is_default: false,
  available: true,
  requires_picker: true,
};
const windowsLoopback: AudioDevice = {
  ...macSystemAudio,
  id: "windows-wasapi-loopback",
  name: "System audio (default output)",
  requires_picker: false,
};
const linuxSystemAudio: AudioDevice = {
  ...macSystemAudio,
  id: "system-audio-unavailable",
  name: "System audio",
  available: false,
  requires_picker: false,
};

describe("system audio availability", () => {
  it("follows the shell's system-audio entry", () => {
    expect(systemAudioDevice([microphone, windowsLoopback])?.id).toBe("windows-wasapi-loopback");
    expect(systemAudioAvailable([microphone, macSystemAudio])).toBe(true);
    expect(systemAudioAvailable([microphone, windowsLoopback])).toBe(true);
    expect(systemAudioAvailable([microphone, linuxSystemAudio])).toBe(false);
    expect(systemAudioAvailable([microphone])).toBe(false);
  });

  it("stays offered until the device list has loaded", () => {
    expect(systemAudioAvailable([])).toBe(true);
  });
});

describe("restoring the saved audio source", () => {
  it("keeps a saved microphone that is still connected", () => {
    expect(restoreAudioSelection({ audio_source: "microphone", audio_device_id: "mic-usb" }, [microphone, usbMicrophone])).toEqual({
      audio_source: "microphone",
      audio_device_id: "mic-usb",
    });
  });

  it("falls back to the default microphone when the saved one is gone", () => {
    expect(restoreAudioSelection({ audio_source: "microphone", audio_device_id: "mic-usb" }, [microphone])).toEqual({
      audio_source: "microphone",
      audio_device_id: "mic-default",
    });
  });

  it("keeps system audio where it can be captured", () => {
    expect(
      restoreAudioSelection({ audio_source: "system_audio", audio_device_id: null }, [microphone, windowsLoopback]).audio_source,
    ).toBe("system_audio");
  });

  it("switches a saved system-audio source to the default microphone where it cannot be captured", () => {
    expect(restoreAudioSelection({ audio_source: "system_audio", audio_device_id: null }, [microphone, linuxSystemAudio])).toEqual({
      audio_source: "microphone",
      audio_device_id: "mic-default",
    });
    expect(
      restoreAudioSelection({ audio_source: "system_audio_and_microphone", audio_device_id: "mic-usb" }, [usbMicrophone, linuxSystemAudio]),
    ).toEqual({ audio_source: "microphone", audio_device_id: null });
  });
});
