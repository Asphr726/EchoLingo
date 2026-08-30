#import <AppKit/AppKit.h>
#import <AudioToolbox/AudioToolbox.h>
#import <AVFAudio/AVFAudio.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <mach/mach_time.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <atomic>

extern "C" {
typedef void (*ELAudioCallback)(const float *samples, size_t frames,
                                uint32_t sample_rate, uint32_t channels,
                                uint64_t capture_ns, void *context);
typedef void (*ELStateCallback)(int32_t code, const char *message,
                                void *context);
}

namespace {
std::atomic_bool screenPermissionRequested{false};

uint64_t monotonicNanoseconds() {
  static mach_timebase_info_data_t info = [] {
    mach_timebase_info_data_t value{};
    mach_timebase_info(&value);
    return value;
  }();
  const uint64_t ticks = mach_continuous_time();
  return ticks * info.numer / info.denom;
}
} // namespace

extern "C" int32_t el_macos_audio_permission_status(int32_t kind) {
  @autoreleasepool {
    if (kind == 0) {
      switch (AVAudioApplication.sharedInstance.recordPermission) {
      case AVAudioApplicationRecordPermissionGranted:
        return 2;
      case AVAudioApplicationRecordPermissionDenied:
        return 1;
      default:
        return 0;
      }
    }
    if (kind == 1) {
      if (CGPreflightScreenCaptureAccess()) {
        return 2;
      }
      return screenPermissionRequested.load() ? 1 : 0;
    }
    return 3;
  }
}

extern "C" int32_t el_macos_request_audio_permission(int32_t kind) {
  @autoreleasepool {
    if (kind == 0) {
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      __block BOOL granted = NO;
      dispatch_async(dispatch_get_main_queue(), ^{
        [AVAudioApplication
            requestRecordPermissionWithCompletionHandler:^(BOOL allowed) {
              granted = allowed;
              dispatch_semaphore_signal(semaphore);
            }];
      });
      if (dispatch_semaphore_wait(
              semaphore,
              dispatch_time(DISPATCH_TIME_NOW,
                            static_cast<int64_t>(120 * NSEC_PER_SEC))) != 0) {
        return 0;
      }
      return granted ? 2 : 1;
    }
    if (kind == 1) {
      screenPermissionRequested.store(true);
      return CGRequestScreenCaptureAccess() ? 2 : 1;
    }
    return 3;
  }
}

@interface ELMacSystemAudioCapture
    : NSObject <SCStreamOutput, SCStreamDelegate,
                SCContentSharingPickerObserver>
@property(nonatomic, assign) ELAudioCallback audioCallback;
@property(nonatomic, assign) ELStateCallback stateCallback;
@property(nonatomic, assign) void *callbackContext;
@property(nonatomic, strong) SCStream *stream;
@property(nonatomic, strong) dispatch_queue_t sampleQueue;
@property(nonatomic, assign) BOOL stopping;
- (instancetype)initWithAudioCallback:(ELAudioCallback)audioCallback
                         stateCallback:(ELStateCallback)stateCallback
                               context:(void *)context;
- (void)presentPicker;
- (void)setPausedSynchronously:(BOOL)paused;
- (void)stopSynchronously;
@end

@implementation ELMacSystemAudioCapture

- (instancetype)initWithAudioCallback:(ELAudioCallback)audioCallback
                         stateCallback:(ELStateCallback)stateCallback
                               context:(void *)context {
  self = [super init];
  if (self) {
    _audioCallback = audioCallback;
    _stateCallback = stateCallback;
    _callbackContext = context;
    _sampleQueue = dispatch_queue_create("app.echolingo.system-audio",
                                         DISPATCH_QUEUE_SERIAL);
  }
  return self;
}

- (void)reportCode:(int32_t)code message:(NSString *)message {
  if (self.stateCallback != nullptr) {
    self.stateCallback(code, message.UTF8String ?: "", self.callbackContext);
  }
}

- (void)presentPicker {
  dispatch_async(dispatch_get_main_queue(), ^{
    SCContentSharingPicker *picker = SCContentSharingPicker.sharedPicker;
    SCContentSharingPickerConfiguration *configuration =
        [[SCContentSharingPickerConfiguration alloc] init];
    configuration.allowedPickerModes =
        SCContentSharingPickerModeSingleDisplay |
        SCContentSharingPickerModeSingleApplication;
    NSString *bundleID = NSBundle.mainBundle.bundleIdentifier;
    if (bundleID.length > 0) {
      configuration.excludedBundleIDs = @[ bundleID ];
    }
    picker.defaultConfiguration = configuration;
    [picker addObserver:self];
    picker.active = YES;
    [picker presentPickerUsingContentStyle:SCShareableContentStyleDisplay];
    [self reportCode:1 message:@"Choose a display or application to capture audio"];
  });
}

- (void)contentSharingPicker:(SCContentSharingPicker *)picker
         didUpdateWithFilter:(SCContentFilter *)filter
                   forStream:(SCStream *)stream {
  (void)picker;
  (void)stream;
  [self startWithFilter:filter];
}

- (void)contentSharingPicker:(SCContentSharingPicker *)picker
          didCancelForStream:(SCStream *)stream {
  (void)picker;
  (void)stream;
  [self reportCode:5 message:@"System audio selection was cancelled"];
}

- (void)contentSharingPickerStartDidFailWithError:(NSError *)error {
  [self reportCode:-1 message:error.localizedDescription];
}

- (void)startWithFilter:(SCContentFilter *)filter {
  if (self.stopping) {
    return;
  }
  SCStreamConfiguration *configuration =
      [[SCStreamConfiguration alloc] init];
  configuration.width = 2;
  configuration.height = 2;
  configuration.minimumFrameInterval = CMTimeMake(1, 1);
  configuration.queueDepth = 3;
  configuration.capturesAudio = YES;
  configuration.excludesCurrentProcessAudio = YES;
  configuration.sampleRate = 48000;
  configuration.channelCount = 2;

  SCStream *newStream = [[SCStream alloc] initWithFilter:filter
                                           configuration:configuration
                                                delegate:self];
  NSError *outputError = nil;
  if (![newStream addStreamOutput:self
                             type:SCStreamOutputTypeAudio
               sampleHandlerQueue:self.sampleQueue
                            error:&outputError]) {
    [self reportCode:-1 message:outputError.localizedDescription];
    return;
  }
  self.stream = newStream;
  [self reportCode:2 message:@"System audio source selected"];
  [newStream startCaptureWithCompletionHandler:^(NSError *error) {
    if (error != nil) {
      [self reportCode:-1 message:error.localizedDescription];
    } else {
      [self reportCode:3 message:@"System audio capture started"];
    }
  }];
}

- (void)stream:(SCStream *)stream
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
                  ofType:(SCStreamOutputType)type {
  (void)stream;
  if (type != SCStreamOutputTypeAudio || self.stopping ||
      !CMSampleBufferDataIsReady(sampleBuffer) ||
      self.audioCallback == nullptr) {
    return;
  }

  CMAudioFormatDescriptionRef format =
      (CMAudioFormatDescriptionRef)CMSampleBufferGetFormatDescription(sampleBuffer);
  const AudioStreamBasicDescription *asbd =
      CMAudioFormatDescriptionGetStreamBasicDescription(format);
  if (asbd == nullptr || asbd->mFormatID != kAudioFormatLinearPCM ||
      (asbd->mFormatFlags & kAudioFormatFlagIsFloat) == 0 ||
      asbd->mBitsPerChannel != 32 || asbd->mChannelsPerFrame == 0) {
    [self reportCode:-1 message:@"Unsupported ScreenCaptureKit audio format"];
    return;
  }

  const UInt32 channels = asbd->mChannelsPerFrame;
  const size_t listSize = offsetof(AudioBufferList, mBuffers) +
                          sizeof(AudioBuffer) * std::max<UInt32>(channels, 1);
  std::vector<uint8_t> listStorage(listSize);
  AudioBufferList *audioList =
      reinterpret_cast<AudioBufferList *>(listStorage.data());
  CMBlockBufferRef retainedBlock = nullptr;
  OSStatus status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
      sampleBuffer, nullptr, audioList, listSize, nullptr, nullptr,
      kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, &retainedBlock);
  if (status != noErr) {
    [self reportCode:-1 message:@"Failed to read ScreenCaptureKit audio buffer"];
    return;
  }

  const CMItemCount frameCount = CMSampleBufferGetNumSamples(sampleBuffer);
  std::vector<float> interleaved(static_cast<size_t>(frameCount) * channels,
                                 0.0f);
  const bool nonInterleaved =
      (asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
  if (nonInterleaved) {
    const UInt32 bufferCount = std::min(audioList->mNumberBuffers, channels);
    for (UInt32 channel = 0; channel < bufferCount; ++channel) {
      const float *source =
          static_cast<const float *>(audioList->mBuffers[channel].mData);
      if (source == nullptr) {
        continue;
      }
      for (CMItemCount frame = 0; frame < frameCount; ++frame) {
        interleaved[static_cast<size_t>(frame) * channels + channel] =
            source[frame];
      }
    }
  } else if (audioList->mNumberBuffers > 0 &&
             audioList->mBuffers[0].mData != nullptr) {
    const float *source =
        static_cast<const float *>(audioList->mBuffers[0].mData);
    std::copy_n(source, interleaved.size(), interleaved.data());
  }

  self.audioCallback(interleaved.data(), static_cast<size_t>(frameCount),
                     static_cast<uint32_t>(asbd->mSampleRate),
                     channels, monotonicNanoseconds(), self.callbackContext);
  if (retainedBlock != nullptr) {
    CFRelease(retainedBlock);
  }
}

- (void)setPausedSynchronously:(BOOL)paused {
  SCStream *activeStream = self.stream;
  if (activeStream == nil || self.stopping) {
    return;
  }
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  void (^completion)(NSError *) = ^(NSError *error) {
    if (error != nil) {
      [self reportCode:-1 message:error.localizedDescription];
    }
    dispatch_semaphore_signal(semaphore);
  };
  if (paused) {
    [activeStream stopCaptureWithCompletionHandler:completion];
  } else {
    [activeStream startCaptureWithCompletionHandler:completion];
  }
  dispatch_semaphore_wait(
      semaphore,
      dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(5 * NSEC_PER_SEC)));
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
  (void)stream;
  if (!self.stopping) {
    [self reportCode:-2 message:error.localizedDescription];
  }
}

- (void)stopSynchronously {
  self.stopping = YES;
  SCStream *activeStream = self.stream;
  if (activeStream != nil) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    [activeStream stopCaptureWithCompletionHandler:^(NSError *error) {
      if (error != nil) {
        [self reportCode:-1 message:error.localizedDescription];
      }
      dispatch_semaphore_signal(semaphore);
    }];
    dispatch_semaphore_wait(
        semaphore,
        dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(2 * NSEC_PER_SEC)));
    self.stream = nil;
  }
  dispatch_sync(self.sampleQueue, ^{});
  void (^cleanup)(void) = ^{
    SCContentSharingPicker *picker = SCContentSharingPicker.sharedPicker;
    [picker removeObserver:self];
    picker.active = NO;
  };
  if (NSThread.isMainThread) {
    cleanup();
  } else {
    dispatch_sync(dispatch_get_main_queue(), cleanup);
  }
  [self reportCode:4 message:@"System audio capture stopped"];
}

@end

extern "C" void *el_macos_system_audio_create(ELAudioCallback audioCallback,
                                               ELStateCallback stateCallback,
                                               void *context) {
  @autoreleasepool {
    ELMacSystemAudioCapture *capture =
        [[ELMacSystemAudioCapture alloc] initWithAudioCallback:audioCallback
                                                stateCallback:stateCallback
                                                      context:context];
    return (__bridge_retained void *)capture;
  }
}

extern "C" void el_macos_system_audio_present(void *handle) {
  ELMacSystemAudioCapture *capture =
      (__bridge ELMacSystemAudioCapture *)handle;
  [capture presentPicker];
}

extern "C" void el_macos_system_audio_stop(void *handle) {
  ELMacSystemAudioCapture *capture =
      (__bridge ELMacSystemAudioCapture *)handle;
  [capture stopSynchronously];
}

extern "C" void el_macos_system_audio_pause(void *handle) {
  ELMacSystemAudioCapture *capture =
      (__bridge ELMacSystemAudioCapture *)handle;
  [capture setPausedSynchronously:YES];
}

extern "C" void el_macos_system_audio_resume(void *handle) {
  ELMacSystemAudioCapture *capture =
      (__bridge ELMacSystemAudioCapture *)handle;
  [capture setPausedSynchronously:NO];
}

extern "C" void el_macos_system_audio_destroy(void *handle) {
  if (handle != nullptr) {
    CFBridgingRelease(handle);
  }
}

extern "C" void el_macos_set_window_opacity(void *nsView, double opacity) {
  if (nsView == nullptr) {
    return;
  }
  NSView *view = (__bridge NSView *)nsView;
  dispatch_async(dispatch_get_main_queue(), ^{
    view.window.alphaValue = std::clamp(opacity, 0.35, 1.0);
  });
}
