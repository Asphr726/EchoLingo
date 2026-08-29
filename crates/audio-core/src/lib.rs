//! Native audio capture with a stable, provider-independent 48 kHz mono boundary.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSourceKind {
    Microphone,
    SystemAudio,
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
    pub fn system_audio() -> Self {
        Self {
            id: "macos-screen-capture-kit".into(),
            name: "System Audio…".into(),
            kind: AudioSourceKind::SystemAudio,
            is_default: false,
            available: cfg!(target_os = "macos"),
            requires_picker: true,
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
    #[error("audio device does not support 48 kHz PCM input")]
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

pub struct AudioCaptureSession {
    frames: Option<mpsc::Receiver<AudioFrame>>,
    events: Option<mpsc::Receiver<AudioSourceEvent>>,
    control: CaptureControl,
}

impl AudioCaptureSession {
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
    Microphone {
        stream: cpal::Stream,
        events: mpsc::Sender<AudioSourceEvent>,
        stopped: AtomicBool,
    },
    #[cfg(target_os = "macos")]
    System(macos::SystemCapture),
}

impl CaptureControl {
    fn pause(&self) -> Result<(), AudioError> {
        match self {
            Self::Microphone { stream, events, .. } => {
                stream
                    .pause()
                    .map_err(|error| AudioError::Control(error.to_string()))?;
                let _ = events.try_send(AudioSourceEvent::Paused);
                Ok(())
            }
            #[cfg(target_os = "macos")]
            Self::System(capture) => capture.pause(),
        }
    }

    fn resume(&self) -> Result<(), AudioError> {
        match self {
            Self::Microphone { stream, events, .. } => {
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
            Self::Microphone {
                stream,
                events,
                stopped,
            } => {
                if !stopped.swap(true, Ordering::SeqCst) {
                    stream
                        .pause()
                        .map_err(|error| AudioError::Control(error.to_string()))?;
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
    let supported = device
        .supported_input_configs()
        .map_err(|error| AudioError::Build(error.to_string()))?
        .filter_map(|range| range.try_with_sample_rate(INTERNAL_SAMPLE_RATE_HZ))
        .min_by_key(|config| {
            let mono_penalty = if config.channels() == 1 { 0 } else { 1 };
            let format_penalty = if config.sample_format() == SampleFormat::F32 {
                0
            } else {
                1
            };
            (mono_penalty, format_penalty, config.channels())
        })
        .ok_or(AudioError::UnsupportedFormat)?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let (frame_tx, frame_rx) = mpsc::channel(CAPTURE_QUEUE_FRAMES);
    let (event_tx, event_rx) = mpsc::channel(32);
    let dropped = Arc::new(AtomicBool::new(false));
    let stream = match sample_format {
        SampleFormat::F32 => {
            build_stream::<f32>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::F64 => {
            build_stream::<f64>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::I8 => {
            build_stream::<i8>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::I16 => {
            build_stream::<i16>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::I32 => {
            build_stream::<i32>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::I64 => {
            build_stream::<i64>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::U8 => {
            build_stream::<u8>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::U16 => {
            build_stream::<u16>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::U32 => {
            build_stream::<u32>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        SampleFormat::U64 => {
            build_stream::<u64>(&device, &config, frame_tx, event_tx.clone(), dropped)?
        }
        _ => return Err(AudioError::UnsupportedFormat),
    };
    stream
        .play()
        .map_err(|error| AudioError::Control(error.to_string()))?;
    let _ = event_tx.try_send(AudioSourceEvent::Started);
    Ok(AudioCaptureSession {
        frames: Some(frame_rx),
        events: Some(event_rx),
        control: CaptureControl::Microphone {
            stream,
            events: event_tx,
            stopped: AtomicBool::new(false),
        },
    })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    frame_tx: mpsc::Sender<AudioFrame>,
    event_tx: mpsc::Sender<AudioSourceEvent>,
    dropped: Arc<AtomicBool>,
) -> Result<cpal::Stream, AudioError>
where
    T: SizedSample + Sample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let error_events = event_tx.clone();
    device
        .build_input_stream::<T, _, _>(
            config,
            move |input, _| {
                if input.len() < channels {
                    return;
                }
                let mut mono = Vec::with_capacity(input.len() / channels);
                for frame in input.chunks_exact(channels) {
                    let sum = frame.iter().copied().map(f32::from_sample).sum::<f32>();
                    mono.push(sum / channels as f32);
                }
                let overflow = dropped.swap(false, Ordering::Relaxed);
                let audio = AudioFrame {
                    samples: mono,
                    sample_rate_hz: INTERNAL_SAMPLE_RATE_HZ,
                    channels: 1,
                    capture_monotonic_ns: monotonic_ns(),
                    overflow,
                };
                if frame_tx.try_send(audio).is_err() {
                    dropped.store(true, Ordering::Relaxed);
                    let _ = event_tx.try_send(AudioSourceEvent::Overflow);
                }
            },
            move |error| {
                let _ = error_events.try_send(AudioSourceEvent::DeviceRemoved {
                    message: error.to_string(),
                });
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

#[cfg(not(target_os = "macos"))]
pub fn start_system_audio() -> Result<AudioCaptureSession, AudioError> {
    Err(AudioError::SystemAudioUnsupported)
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

    type AudioCallback = unsafe extern "C" fn(*const f32, usize, u32, u64, *mut c_void);
    type StateCallback = unsafe extern "C" fn(i32, *const c_char, *mut c_void);

    extern "C" {
        fn el_macos_system_audio_create(
            audio_callback: AudioCallback,
            state_callback: StateCallback,
            context: *mut c_void,
        ) -> *mut c_void;
        fn el_macos_system_audio_present(handle: *mut c_void);
        fn el_macos_system_audio_stop(handle: *mut c_void);
        fn el_macos_system_audio_destroy(handle: *mut c_void);
    }

    struct CallbackContext {
        frames: mpsc::Sender<AudioFrame>,
        events: mpsc::Sender<AudioSourceEvent>,
        paused: AtomicBool,
        dropped: AtomicBool,
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
            let _ = self.events.try_send(AudioSourceEvent::Paused);
            Ok(())
        }

        pub(super) fn resume(&self) -> Result<(), AudioError> {
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
        capture_ns: u64,
        context: *mut c_void,
    ) {
        if samples.is_null() || context.is_null() || frames == 0 {
            return;
        }
        let context = &*(context.cast::<CallbackContext>());
        if context.paused.load(Ordering::Acquire) {
            return;
        }
        let frame = AudioFrame {
            samples: slice::from_raw_parts(samples, frames).to_vec(),
            sample_rate_hz: sample_rate,
            channels: 1,
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
            -2 => AudioSourceEvent::DeviceRemoved { message },
            _ => AudioSourceEvent::Error {
                message,
                recoverable: true,
            },
        };
        let _ = context.events.try_send(event);
    }

    pub(super) fn start() -> Result<AudioCaptureSession, AudioError> {
        let (frame_tx, frame_rx) = mpsc::channel(CAPTURE_QUEUE_FRAMES);
        let (event_tx, event_rx) = mpsc::channel(32);
        let context = Box::into_raw(Box::new(CallbackContext {
            frames: frame_tx,
            events: event_tx.clone(),
            paused: AtomicBool::new(false),
            dropped: AtomicBool::new(false),
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
    fn system_audio_descriptor_is_explicit_about_picker() {
        let device = AudioDevice::system_audio();
        assert_eq!(device.kind, AudioSourceKind::SystemAudio);
        assert!(device.requires_picker);
    }
}
