import type { AudioDevice, StartSessionRequest } from "../types";

/** The shell's system-audio entry: the capture picker on macOS, loopback of
 *  the default output device on Windows, an unavailable entry elsewhere. */
export function systemAudioDevice(devices: AudioDevice[]): AudioDevice | undefined {
  return devices.find((device) => device.kind === "system_audio");
}

/** Whether System audio can be offered. Nothing is known before the device
 *  list has loaded, so it stays offered until then. */
export function systemAudioAvailable(devices: AudioDevice[]): boolean {
  if (devices.length === 0) return true;
  return systemAudioDevice(devices)?.available ?? false;
}

type AudioSelection = Pick<StartSessionRequest, "audio_source" | "audio_device_id">;

/** The saved audio source and device, checked against this computer's
 *  devices: a microphone that is gone becomes the default one, and a
 *  system-audio source this computer cannot capture falls back to the
 *  default microphone. */
export function restoreAudioSelection(saved: AudioSelection, devices: AudioDevice[]): AudioSelection {
  const defaultMicrophone = devices.find((device) => device.kind === "microphone" && device.is_default)?.id ?? null;
  if (saved.audio_source !== "microphone" && !systemAudioAvailable(devices)) {
    return { audio_source: "microphone", audio_device_id: defaultMicrophone };
  }
  return {
    audio_source: saved.audio_source,
    audio_device_id:
      devices.find((device) => device.id === saved.audio_device_id && device.available)?.id ?? defaultMicrophone,
  };
}
