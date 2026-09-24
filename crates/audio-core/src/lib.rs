//! Native audio capture with a stable, provider-independent multichannel boundary.
//!
//! Capture never performs inference or uploads data. It only produces local PCM
//! frames and lifecycle events for the Rust app core.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use thiserror::Error;
use tokio::sync::mpsc;

pub const INTERNAL_SAMPLE_RATE_HZ: u32 = 48_000;
pub const CAPTURE_QUEUE_FRAMES: usize = 64;
/// System-audio device ids (see [`AudioDevice::system_audio`]).
pub const MACOS_SYSTEM_AUDIO_DEVICE_ID: &str = "macos-screen-capture-kit";
pub const WINDOWS_LOOPBACK_DEVICE_ID: &str = "windows-wasapi-loopback";
pub const UNAVAILABLE_SYSTEM_AUDIO_DEVICE_ID: &str = "system-audio-unavailable";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSourceKind {
    Microphone,
    SystemAudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionKind {
    Microphone,
    SystemAudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionState {
    NotDetermined,
    Denied,
    Granted,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPermissionStatus {
    pub microphone: PermissionState,
    pub system_audio: PermissionState,
}

#[cfg(target_os = "macos")]
fn permission_from_native(value: i32) -> PermissionState {
    match value {
        0 => PermissionState::NotDetermined,
        1 => PermissionState::Denied,
        2 => PermissionState::Granted,
        _ => PermissionState::Unavailable,
    }
}

#[cfg(target_os = "macos")]
pub fn audio_permission_status() -> AudioPermissionStatus {
    extern "C" {
        fn el_macos_audio_permission_status(kind: i32) -> i32;
    }
    AudioPermissionStatus {
        microphone: permission_from_native(unsafe { el_macos_audio_permission_status(0) }),
        system_audio: permission_from_native(unsafe { el_macos_audio_permission_status(1) }),
    }
}

/// Windows records the default output through WASAPI loopback without a
/// separate consent; other platforms without a native implementation have
/// no system audio at all.
#[cfg(not(target_os = "macos"))]
fn system_audio_permission() -> PermissionState {
    if cfg!(target_os = "windows") {
        PermissionState::Granted
    } else {
        PermissionState::Unavailable
    }
}

/// Off macOS the OS grants desktop apps microphone access without an
/// in-app prompt.
#[cfg(not(target_os = "macos"))]
pub fn audio_permission_status() -> AudioPermissionStatus {
    AudioPermissionStatus {
        microphone: PermissionState::Granted,
        system_audio: system_audio_permission(),
    }
}

#[cfg(target_os = "macos")]
pub fn request_audio_permission(kind: PermissionKind) -> PermissionState {
    extern "C" {
        fn el_macos_request_audio_permission(kind: i32) -> i32;
    }
    let kind = match kind {
        PermissionKind::Microphone => 0,
        PermissionKind::SystemAudio => 1,
    };
    permission_from_native(unsafe { el_macos_request_audio_permission(kind) })
}

#[cfg(not(target_os = "macos"))]
pub fn request_audio_permission(kind: PermissionKind) -> PermissionState {
    match kind {
        PermissionKind::Microphone => PermissionState::Granted,
        PermissionKind::SystemAudio => system_audio_permission(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub kind: AudioSourceKind,
    pub is_default: bool,
    pub available: bool,
    pub requires_picker: bool,
}

impl AudioDevice {
    /// The system-audio entry of the device list: the ScreenCaptureKit picker
    /// on macOS, loopback of the default output on Windows, and an
    /// unavailable placeholder elsewhere.
    #[cfg(target_os = "macos")]
    pub fn system_audio() -> Self {
        Self {
            id: MACOS_SYSTEM_AUDIO_DEVICE_ID.into(),
            name: "System Audio…".into(),
            kind: AudioSourceKind::SystemAudio,
            is_default: false,
            available: true,
            requires_picker: true,
        }
    }

    #[cfg(target_os = "windows")]
    pub fn system_audio() -> Self {
        Self {
            id: WINDOWS_LOOPBACK_DEVICE_ID.into(),
            name: "System audio (default output)".into(),
            kind: AudioSourceKind::SystemAudio,
            is_default: false,
            available: cpal::default_host().default_output_device().is_some(),
            requires_picker: false,
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    pub fn system_audio() -> Self {
        Self {
            id: UNAVAILABLE_SYSTEM_AUDIO_DEVICE_ID.into(),
            name: "System audio".into(),
            kind: AudioSourceKind::SystemAudio,
            is_default: false,
            available: false,
            requires_picker: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub capture_monotonic_ns: u64,
    pub overflow: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl AudioFrame {
    pub fn frame_count(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    pub fn rms_dbfs(&self) -> f32 {
        if self.samples.is_empty() {
            return -120.0;
        }
        let mean_square = self
            .samples
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / self.samples.len() as f64;
        (20.0 * mean_square.sqrt().max(1.0e-6).log10()) as f32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum AudioSourceEvent {
    PickerPresented,
    SourceSelected,
    Started,
    Paused,
    Resumed,
    Stopped,
    PickerCancelled,
    Overflow,
    DeviceRemoved { message: String },
    Error { message: String, recoverable: bool },
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to enumerate input devices: {0}")]
    Enumerate(String),
    #[error("audio input device was not found: {0}")]
    DeviceNotFound(String),
    #[error("audio device does not expose a supported PCM input format")]
    UnsupportedFormat,
    #[error("failed to build audio stream: {0}")]
    Build(String),
    #[error("failed to control audio stream: {0}")]
    Control(String),
    #[error("system audio capture is not supported on this platform")]
    SystemAudioUnsupported,
    #[error("failed to initialize system audio capture")]
    SystemAudioInitialization,
}

pub fn list_audio_devices() -> Result<Vec<AudioDevice>, AudioError> {
    let host = cpal::default_host();
    let default_id = host
        .default_input_device()
        .and_then(|device| device.id().ok())
        .map(|id| id.to_string());
    let devices = host
        .input_devices()
        .map_err(|error| AudioError::Enumerate(error.to_string()))?;
    let mut result = Vec::new();
    for device in devices {
        let Ok(id) = device.id() else { continue };
        let id = id.to_string();
        if cfg!(target_os = "linux")
            && default_id.as_deref() != Some(id.as_str())
            && !is_listed_alsa_input(&id)
        {
            continue;
        }
        let name = device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "Unknown microphone".into());
        result.push(AudioDevice {
            is_default: default_id.as_deref() == Some(id.as_str()),
            id,
            name,
            kind: AudioSourceKind::Microphone,
            available: true,
            requires_picker: false,
        });
    }
    result.sort_by_key(|device| (!device.is_default, device.name.clone()));
    result.push(AudioDevice::system_audio());
    Ok(result)
}

/// ALSA lists every plugin and hardware alias of every card; offer the ones
/// a user would pick (the sound server and each card's default).
fn is_listed_alsa_input(device_id: &str) -> bool {
    let pcm = device_id
        .split_once(':')
        .map_or(device_id, |(_, pcm)| pcm);
    matches!(pcm, "default" | "pipewire" | "pulse") || pcm.starts_with("sysdefault:CARD=")
}

pub struct AudioCaptureSession {
    frames: Option<mpsc::Receiver<AudioFrame>>,
    events: Option<mpsc::Receiver<AudioSourceEvent>>,
    control: CaptureControl,
    format: CaptureFormat,
    /// Set once the source is gone (device removed, stream error, or the
    /// default output of a loopback capture changed).
    invalidated: Arc<AtomicBool>,
}

impl AudioCaptureSession {
    /// The PCM format the frames of this capture arrive in.
    pub fn format(&self) -> CaptureFormat {
        self.format
    }

    /// The source failed and cannot resume; start a new capture instead.
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    pub fn take_frames(&mut self) -> Option<mpsc::Receiver<AudioFrame>> {
        self.frames.take()
    }

    pub fn take_events(&mut self) -> Option<mpsc::Receiver<AudioSourceEvent>> {
        self.events.take()
    }

    pub fn pause(&self) -> Result<(), AudioError> {
        self.control.pause()
    }

    pub fn resume(&self) -> Result<(), AudioError> {
        self.control.resume()
    }

    /// Stop capturing. Best effort: a stream whose device already failed
    /// cannot be paused any more, which never keeps a session from stopping.
    pub fn stop(&mut self) -> Result<(), AudioError> {
        self.control.stop()
    }
}

impl Drop for AudioCaptureSession {
    fn drop(&mut self) {
        let _ = self.control.stop();
    }
}

enum CaptureControl {
    Cpal {
        stream: cpal::Stream,
        /// A silent output stream that keeps a loopback endpoint running.
        keep_alive: Option<cpal::Stream>,
        events: mpsc::Sender<AudioSourceEvent>,
        stopped: Arc<AtomicBool>,
        invalidated: Arc<AtomicBool>,
    },
    #[cfg(target_os = "macos")]
    System(macos::SystemCapture),
}

impl CaptureControl {
    fn pause(&self) -> Result<(), AudioError> {
        match self {
            Self::Cpal {
                stream,
                events,
                invalidated,
                ..
            } => {
                // A failed stream delivers nothing any more; pausing it only
                // errors.
                if !invalidated.load(Ordering::Acquire) {
                    stream
                        .pause()
                        .map_err(|error| AudioError::Control(error.to_string()))?;
                }
                let _ = events.try_send(AudioSourceEvent::Paused);
                Ok(())
            }
            #[cfg(target_os = "macos")]
            Self::System(capture) => capture.pause(),
        }
    }

    fn resume(&self) -> Result<(), AudioError> {
        match self {
            Self::Cpal {
                stream,
                events,
                invalidated,
                ..
            } => {
                if invalidated.load(Ordering::Acquire) {
                    return Err(AudioError::Control(
                        "the audio source is no longer available; start it again".into(),
                    ));
                }
                stream
                    .play()
                    .map_err(|error| AudioError::Control(error.to_string()))?;
                let _ = events.try_send(AudioSourceEvent::Resumed);
                Ok(())
            }
            #[cfg(target_os = "macos")]
            Self::System(capture) => capture.resume(),
        }
    }

    fn stop(&mut self) -> Result<(), AudioError> {
        match self {
            Self::Cpal {
                stream,
                keep_alive,
                events,
                stopped,
                invalidated,
            } => {
                if !stopped.swap(true, Ordering::SeqCst) {
                    if let Some(keep_alive) = keep_alive {
                        let _ = keep_alive.pause();
                    }
                    // After a device error cpal refuses to pause the stream
                    // forever; the stream is released on drop either way.
                    if let Err(error) = stream.pause() {
                        if !invalidated.load(Ordering::Acquire) {
                            eprintln!("audio stream did not pause on stop: {error}");
                        }
                    }
                    let _ = events.try_send(AudioSourceEvent::Stopped);
                }
                Ok(())
            }
            #[cfg(target_os = "macos")]
            Self::System(capture) => capture.stop(),
        }
    }
}

pub fn start_microphone(device_id: Option<&str>) -> Result<AudioCaptureSession, AudioError> {
    let (device, supported) = select_microphone_config(device_id)?;
    start_cpal_capture(&device, supported, None)
}

/// Capture `device` in its `supported` format. `keep_alive` is a stream that
/// has to live (and stop) with the capture.
fn start_cpal_capture(
    device: &cpal::Device,
    supported: cpal::SupportedStreamConfig,
    keep_alive: Option<cpal::Stream>,
) -> Result<AudioCaptureSession, AudioError> {
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let format = CaptureFormat {
        sample_rate_hz: config.sample_rate,
        channels: config.channels,
    };
    let (frame_tx, frame_rx) = mpsc::channel(CAPTURE_QUEUE_FRAMES);
    let (event_tx, event_rx) = mpsc::channel(32);
    let invalidated = Arc::new(AtomicBool::new(false));
    let sink = StreamSink {
        frames: frame_tx,
        events: event_tx.clone(),
        dropped: Arc::new(AtomicBool::new(false)),
        invalidated: invalidated.clone(),
    };
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(device, &config, sink)?,
        SampleFormat::F64 => build_stream::<f64>(device, &config, sink)?,
        SampleFormat::I8 => build_stream::<i8>(device, &config, sink)?,
        SampleFormat::I16 => build_stream::<i16>(device, &config, sink)?,
        SampleFormat::I24 => build_stream::<cpal::I24>(device, &config, sink)?,
        SampleFormat::I32 => build_stream::<i32>(device, &config, sink)?,
        SampleFormat::I64 => build_stream::<i64>(device, &config, sink)?,
        SampleFormat::U8 => build_stream::<u8>(device, &config, sink)?,
        SampleFormat::U16 => build_stream::<u16>(device, &config, sink)?,
        SampleFormat::U24 => build_stream::<cpal::U24>(device, &config, sink)?,
        SampleFormat::U32 => build_stream::<u32>(device, &config, sink)?,
        SampleFormat::U64 => build_stream::<u64>(device, &config, sink)?,
        _ => return Err(AudioError::UnsupportedFormat),
    };
    stream
        .play()
        .map_err(|error| AudioError::Control(error.to_string()))?;
    let _ = event_tx.try_send(AudioSourceEvent::Started);
    Ok(AudioCaptureSession {
        frames: Some(frame_rx),
        events: Some(event_rx),
        control: CaptureControl::Cpal {
            stream,
            keep_alive,
            events: event_tx,
            stopped: Arc::new(AtomicBool::new(false)),
            invalidated: invalidated.clone(),
        },
        format,
        invalidated,
    })
}

pub fn preferred_capture_format(
    source: AudioSourceKind,
    device_id: Option<&str>,
) -> Result<CaptureFormat, AudioError> {
    match source {
        AudioSourceKind::Microphone => {
            let (_, config) = select_microphone_config(device_id)?;
            Ok(CaptureFormat {
                sample_rate_hz: config.sample_rate(),
                channels: config.channels(),
            })
        }
        AudioSourceKind::SystemAudio => system_audio_format(),
    }
}

#[cfg(target_os = "macos")]
fn system_audio_format() -> Result<CaptureFormat, AudioError> {
    Ok(CaptureFormat {
        sample_rate_hz: INTERNAL_SAMPLE_RATE_HZ,
        channels: 2,
    })
}

/// Loopback delivers the default output's mix format.
#[cfg(target_os = "windows")]
fn system_audio_format() -> Result<CaptureFormat, AudioError> {
    let (_, config) = windows_loopback::select_config()?;
    Ok(CaptureFormat {
        sample_rate_hz: config.sample_rate(),
        channels: config.channels(),
    })
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn system_audio_format() -> Result<CaptureFormat, AudioError> {
    Err(AudioError::SystemAudioUnsupported)
}

fn select_microphone_config(
    device_id: Option<&str>,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig), AudioError> {
    let host = cpal::default_host();
    let device = match device_id {
        Some(expected) => host
            .input_devices()
            .map_err(|error| AudioError::Enumerate(error.to_string()))?
            .find(|device| {
                device
                    .id()
                    .map(|id| id.to_string() == expected)
                    .unwrap_or(false)
            })
            .ok_or_else(|| AudioError::DeviceNotFound(expected.into()))?,
        None => host
            .default_input_device()
            .ok_or_else(|| AudioError::DeviceNotFound("default".into()))?,
    };
    let configurations = device
        .supported_input_configs()
        .map_err(|error| AudioError::Build(error.to_string()))?
        .collect::<Vec<_>>();
    let supported = configurations
        .iter()
        .filter_map(|range| range.clone().try_with_sample_rate(INTERNAL_SAMPLE_RATE_HZ))
        .min_by_key(|config| {
            let mono_penalty = if config.channels() == 1 { 0 } else { 1 };
            let format_penalty = if config.sample_format() == SampleFormat::F32 {
                0
            } else {
                1
            };
            (mono_penalty, format_penalty, config.channels())
        });
    if let Some(supported) = supported {
        return Ok((device, supported));
    }
    let fallback = device
        .default_input_config()
        .map_err(|_| AudioError::UnsupportedFormat)?;
    Ok((device, fallback))
}

/// Where a cpal input stream delivers its frames and lifecycle events.
struct StreamSink {
    frames: mpsc::Sender<AudioFrame>,
    events: mpsc::Sender<AudioSourceEvent>,
    dropped: Arc<AtomicBool>,
    invalidated: Arc<AtomicBool>,
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sink: StreamSink,
) -> Result<cpal::Stream, AudioError>
where
    T: SizedSample + Sample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let output_channels = config.channels;
    let output_sample_rate = config.sample_rate;
    let StreamSink {
        frames: frame_tx,
        events: event_tx,
        dropped,
        invalidated,
    } = sink;
    let error_events = event_tx.clone();
    let error_dropped = dropped.clone();
    device
        .build_input_stream::<T, _, _>(
            config,
            move |input, _| {
                if input.len() < channels {
                    return;
                }
                let samples = input.iter().copied().map(f32::from_sample).collect();
                let overflow = dropped.swap(false, Ordering::Relaxed);
                let audio = AudioFrame {
                    samples,
                    sample_rate_hz: output_sample_rate,
                    channels: output_channels,
                    capture_monotonic_ns: monotonic_ns(),
                    overflow,
                };
                if frame_tx.try_send(audio).is_err() {
                    dropped.store(true, Ordering::Relaxed);
                    let _ = event_tx.try_send(AudioSourceEvent::Overflow);
                }
            },
            move |error| match error {
                // A glitch, not a lost device: flag the gap and keep going.
                cpal::StreamError::BufferUnderrun => {
                    error_dropped.store(true, Ordering::Relaxed);
                    let _ = error_events.try_send(AudioSourceEvent::Overflow);
                }
                error => {
                    if !invalidated.swap(true, Ordering::AcqRel) {
                        let _ = error_events.try_send(AudioSourceEvent::DeviceRemoved {
                            message: error.to_string(),
                        });
                    }
                }
            },
            None,
        )
        .map_err(|error| AudioError::Build(error.to_string()))
}

fn monotonic_ns() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(target_os = "macos")]
pub fn start_system_audio() -> Result<AudioCaptureSession, AudioError> {
    macos::start()
}

#[cfg(target_os = "windows")]
pub fn start_system_audio() -> Result<AudioCaptureSession, AudioError> {
    windows_loopback::start()
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn start_system_audio() -> Result<AudioCaptureSession, AudioError> {
    Err(AudioError::SystemAudioUnsupported)
}

/// WASAPI loopback of the default output device: cpal opens an input stream
/// on a render endpoint in loopback mode (`AUDCLNT_STREAMFLAGS_LOOPBACK`).
#[cfg(target_os = "windows")]
mod windows_loopback {
    use super::*;

    pub(super) fn select_config(
    ) -> Result<(cpal::Device, cpal::SupportedStreamConfig), AudioError> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| AudioError::DeviceNotFound("default output".into()))?;
        // Shared-mode loopback has to use the endpoint's mix format.
        let config = device
            .default_output_config()
            .map_err(|error| AudioError::Build(error.to_string()))?;
        Ok((device, config))
    }

    pub(super) fn start() -> Result<AudioCaptureSession, AudioError> {
        let (device, config) = select_config()?;
        // A loopback stream only receives packets while the endpoint renders;
        // silence keeps it running when nothing else plays, so the pipeline
        // sees continuous (silent) audio instead of a stalled source.
        let keep_alive = match silent_output(&device, &config) {
            Ok(stream) => Some(stream),
            Err(error) => {
                eprintln!(
                    "system audio keep-alive stream unavailable ({error}); \
                     loopback delivers audio only while something plays"
                );
                None
            }
        };
        let session = start_cpal_capture(&device, config, keep_alive)?;
        if let (Ok(id), CaptureControl::Cpal {
            events, stopped, ..
        }) = (device.id(), &session.control)
        {
            watch_default_output(
                id.to_string(),
                events.clone(),
                stopped.clone(),
                session.invalidated.clone(),
            );
        }
        Ok(session)
    }

    /// Loopback stays bound to the endpoint it was opened on. When the user
    /// switches the default output (say, to headphones) the capture would go
    /// silent without an error, so report it as a removed device: the
    /// session pauses for recovery and Resume opens the new default.
    fn watch_default_output(
        device_id: String,
        events: mpsc::Sender<AudioSourceEvent>,
        stopped: Arc<AtomicBool>,
        invalidated: Arc<AtomicBool>,
    ) {
        const POLL: std::time::Duration = std::time::Duration::from_millis(1500);
        const STEP: std::time::Duration = std::time::Duration::from_millis(100);
        let watcher = std::thread::Builder::new()
            .name("echolingo-loopback-watch".into())
            .spawn(move || loop {
                let mut waited = std::time::Duration::ZERO;
                while waited < POLL {
                    if stopped.load(Ordering::Acquire) || invalidated.load(Ordering::Acquire) {
                        return;
                    }
                    std::thread::sleep(STEP);
                    waited += STEP;
                }
                let current = cpal::default_host()
                    .default_output_device()
                    .and_then(|device| device.id().ok())
                    .map(|id| id.to_string());
                if current.as_deref() != Some(device_id.as_str()) {
                    if !invalidated.swap(true, Ordering::AcqRel) {
                        let _ = events.try_send(AudioSourceEvent::DeviceRemoved {
                            message: "The default output device changed; resume to capture \
                                      the new one"
                                .into(),
                        });
                    }
                    return;
                }
            });
        if let Err(error) = watcher {
            eprintln!("cannot watch the default output device: {error}");
        }
    }

    fn silent_output(
        device: &cpal::Device,
        supported: &cpal::SupportedStreamConfig,
    ) -> Result<cpal::Stream, AudioError> {
        let config = supported.config();
        let stream = match supported.sample_format() {
            SampleFormat::F32 => silent::<f32>(device, &config)?,
            SampleFormat::F64 => silent::<f64>(device, &config)?,
            SampleFormat::I16 => silent::<i16>(device, &config)?,
            SampleFormat::I24 => silent::<cpal::I24>(device, &config)?,
            SampleFormat::I32 => silent::<i32>(device, &config)?,
            SampleFormat::U8 => silent::<u8>(device, &config)?,
            SampleFormat::U16 => silent::<u16>(device, &config)?,
            _ => return Err(AudioError::UnsupportedFormat),
        };
        stream
            .play()
            .map_err(|error| AudioError::Control(error.to_string()))?;
        Ok(stream)
    }

    fn silent<T: SizedSample + Send + 'static>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
    ) -> Result<cpal::Stream, AudioError> {
        device
            .build_output_stream::<T, _, _>(
                config,
                |output: &mut [T], _| output.fill(T::EQUILIBRIUM),
                |_| {},
                None,
            )
            .map_err(|error| AudioError::Build(error.to_string()))
    }
}

/// Apply public AppKit `NSWindow.alphaValue` without enabling Tauri's private
/// transparent-window API. `ns_view` must come from an AppKit raw window handle.
#[cfg(target_os = "macos")]
pub unsafe fn set_macos_window_opacity(ns_view: *mut std::ffi::c_void, opacity: f64) {
    extern "C" {
        fn el_macos_set_window_opacity(ns_view: *mut std::ffi::c_void, opacity: f64);
    }
    el_macos_set_window_opacity(ns_view, opacity);
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::slice;

    type AudioCallback = unsafe extern "C" fn(*const f32, usize, u32, u32, u64, *mut c_void);
    type StateCallback = unsafe extern "C" fn(i32, *const c_char, *mut c_void);

    extern "C" {
        fn el_macos_system_audio_create(
            audio_callback: AudioCallback,
            state_callback: StateCallback,
            context: *mut c_void,
        ) -> *mut c_void;
        fn el_macos_system_audio_present(handle: *mut c_void);
        fn el_macos_system_audio_pause(handle: *mut c_void);
        fn el_macos_system_audio_resume(handle: *mut c_void);
        fn el_macos_system_audio_stop(handle: *mut c_void);
        fn el_macos_system_audio_destroy(handle: *mut c_void);
    }

    struct CallbackContext {
        frames: mpsc::Sender<AudioFrame>,
        events: mpsc::Sender<AudioSourceEvent>,
        paused: AtomicBool,
        dropped: AtomicBool,
        invalidated: Arc<AtomicBool>,
    }

    pub(super) struct SystemCapture {
        handle: *mut c_void,
        context: *mut CallbackContext,
        events: mpsc::Sender<AudioSourceEvent>,
        stopped: AtomicBool,
    }

    // The Objective-C capture object owns its queues. Rust only invokes its
    // synchronized control functions, so moving this opaque handle is safe.
    unsafe impl Send for SystemCapture {}

    impl SystemCapture {
        pub(super) fn pause(&self) -> Result<(), AudioError> {
            unsafe { (*self.context).paused.store(true, Ordering::Release) };
            unsafe { el_macos_system_audio_pause(self.handle) };
            let _ = self.events.try_send(AudioSourceEvent::Paused);
            Ok(())
        }

        pub(super) fn resume(&self) -> Result<(), AudioError> {
            unsafe { el_macos_system_audio_resume(self.handle) };
            unsafe { (*self.context).paused.store(false, Ordering::Release) };
            let _ = self.events.try_send(AudioSourceEvent::Resumed);
            Ok(())
        }

        pub(super) fn stop(&mut self) -> Result<(), AudioError> {
            if self.stopped.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            unsafe {
                el_macos_system_audio_stop(self.handle);
                el_macos_system_audio_destroy(self.handle);
                drop(Box::from_raw(self.context));
            }
            self.handle = std::ptr::null_mut();
            self.context = std::ptr::null_mut();
            Ok(())
        }
    }

    unsafe extern "C" fn audio_callback(
        samples: *const f32,
        frames: usize,
        sample_rate: u32,
        channels: u32,
        capture_ns: u64,
        context: *mut c_void,
    ) {
        if samples.is_null() || context.is_null() || frames == 0 || channels == 0 {
            return;
        }
        let context = &*(context.cast::<CallbackContext>());
        if context.paused.load(Ordering::Acquire) {
            return;
        }
        let frame = AudioFrame {
            samples: slice::from_raw_parts(samples, frames * channels as usize).to_vec(),
            sample_rate_hz: sample_rate,
            channels: channels.try_into().unwrap_or(u16::MAX),
            capture_monotonic_ns: capture_ns,
            overflow: context.dropped.swap(false, Ordering::Relaxed),
        };
        if context.frames.try_send(frame).is_err() {
            context.dropped.store(true, Ordering::Relaxed);
            let _ = context.events.try_send(AudioSourceEvent::Overflow);
        }
    }

    unsafe extern "C" fn state_callback(code: i32, message: *const c_char, context: *mut c_void) {
        if context.is_null() {
            return;
        }
        let context = &*(context.cast::<CallbackContext>());
        let message = if message.is_null() {
            String::new()
        } else {
            CStr::from_ptr(message).to_string_lossy().into_owned()
        };
        let event = match code {
            1 => AudioSourceEvent::PickerPresented,
            2 => AudioSourceEvent::SourceSelected,
            3 => AudioSourceEvent::Started,
            4 => AudioSourceEvent::Stopped,
            5 => AudioSourceEvent::PickerCancelled,
            -2 => {
                context.invalidated.store(true, Ordering::Release);
                AudioSourceEvent::DeviceRemoved { message }
            }
            _ => AudioSourceEvent::Error {
                message,
                recoverable: true,
            },
        };
        let _ = context.events.try_send(event);
    }

    pub(super) fn start() -> Result<AudioCaptureSession, AudioError> {
        let format = system_audio_format()?;
        let (frame_tx, frame_rx) = mpsc::channel(CAPTURE_QUEUE_FRAMES);
        let (event_tx, event_rx) = mpsc::channel(32);
        let invalidated = Arc::new(AtomicBool::new(false));
        let context = Box::into_raw(Box::new(CallbackContext {
            frames: frame_tx,
            events: event_tx.clone(),
            paused: AtomicBool::new(false),
            dropped: AtomicBool::new(false),
            invalidated: invalidated.clone(),
        }));
        let handle = unsafe {
            el_macos_system_audio_create(audio_callback, state_callback, context.cast::<c_void>())
        };
        if handle.is_null() {
            unsafe { drop(Box::from_raw(context)) };
            return Err(AudioError::SystemAudioInitialization);
        }
        unsafe { el_macos_system_audio_present(handle) };
        Ok(AudioCaptureSession {
            frames: Some(frame_rx),
            events: Some(event_rx),
            control: CaptureControl::System(SystemCapture {
                handle,
                context,
                events: event_tx,
                stopped: AtomicBool::new(false),
            }),
            format,
            invalidated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_reports_count_and_rms() {
        let frame = AudioFrame {
            samples: vec![0.5, -0.5, 0.5, -0.5],
            sample_rate_hz: 48_000,
            channels: 1,
            capture_monotonic_ns: 1,
            overflow: false,
        };
        assert_eq!(frame.frame_count(), 4);
        assert!((frame.rms_dbfs() + 6.0206).abs() < 0.01);
    }

    #[test]
    fn system_audio_descriptor_matches_the_platform() {
        let device = AudioDevice::system_audio();
        assert_eq!(device.kind, AudioSourceKind::SystemAudio);
        assert!(!device.is_default);
        if cfg!(target_os = "macos") {
            assert_eq!(device.id, MACOS_SYSTEM_AUDIO_DEVICE_ID);
            assert!(device.requires_picker && device.available);
        } else if cfg!(target_os = "windows") {
            assert_eq!(device.id, WINDOWS_LOOPBACK_DEVICE_ID);
            assert_eq!(device.name, "System audio (default output)");
            assert!(!device.requires_picker);
        } else {
            assert_eq!(device.id, UNAVAILABLE_SYSTEM_AUDIO_DEVICE_ID);
            assert!(!device.available && !device.requires_picker);
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn permissions_off_macos_need_no_prompt() {
        let status = audio_permission_status();
        assert_eq!(status.microphone, PermissionState::Granted);
        let system = if cfg!(target_os = "windows") {
            PermissionState::Granted
        } else {
            PermissionState::Unavailable
        };
        assert_eq!(status.system_audio, system);
        assert_eq!(
            request_audio_permission(PermissionKind::SystemAudio),
            system
        );
    }

    #[test]
    fn linux_device_list_keeps_servers_and_card_defaults() {
        for id in [
            "alsa:default",
            "alsa:pipewire",
            "alsa:pulse",
            "alsa:sysdefault:CARD=PCH",
        ] {
            assert!(is_listed_alsa_input(id), "{id}");
        }
        for id in [
            "alsa:hw:CARD=PCH,DEV=0",
            "alsa:plughw:CARD=PCH,DEV=0",
            "alsa:dsnoop:CARD=PCH,DEV=0",
            "alsa:surround51:CARD=PCH,DEV=0",
            "alsa:null",
        ] {
            assert!(!is_listed_alsa_input(id), "{id}");
        }
    }
}
