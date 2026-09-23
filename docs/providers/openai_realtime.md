# OpenAI Realtime transcription (`openai_realtime`)

Cloud ASR adapter for the OpenAI Realtime API in transcription mode
(`wss://api.openai.com/v1/realtime?intent=transcription`). Module:
`src/echolingo/backends/asr/openai_realtime.py`, class
`OpenAiRealtimeAsrBackend`. Credential group `openai` (shared with the
`openai_chat` translation adapter).

## Getting a key

1. Create an API key at <https://platform.openai.com/api-keys>. The key's
   project must have access to the Realtime API and to the transcription model
   you select.
2. In EchoLingo open **Settings → Cloud providers → OpenAI**, paste the key
   and choose **Save** (it is stored in the macOS Keychain). The desktop shell injects it into
   the sidecar as `OPENAI_API_KEY`; the adapter never persists or logs it.
3. Choose **Test** (see below).
4. Enable **Settings → Privacy → Audio upload** and select *OpenAI realtime
   transcription (cloud)* as the ASR provider. Translation routes
   independently and may stay local (Hybrid).

For development only, `OPENAI_API_KEY` in the environment is used when the
keychain has no key.

## Pricing and free tier

There is no free tier. Realtime transcription is billed per minute of audio
uploaded (see OpenAI's pricing page for the current `gpt-4o-transcribe` and
`gpt-4o-mini-transcribe` rates). EchoLingo uploads audio continuously while a
session runs, independent of VAD, so a 60-minute lecture is billed as roughly
60 audio minutes plus any replay after a reconnect (at most
`replay_overlap_ms`, 500 ms by default, per reconnect).

## Models and settings

| Setting | Env var (settings card) | Default | Notes |
| --- | --- | --- | --- |
| model | `ECHOLINGO_OPENAI_REALTIME_MODEL` | `gpt-4o-transcribe` | `gpt-4o-mini-transcribe` is cheaper and slightly less accurate; `whisper-1` works but streams no deltas (text arrives only when a segment completes). |
| `vad_threshold` | config `asr.openai_realtime` | `0.5` | Server VAD sensitivity (0–1). Lower values catch quieter far-field speech. |
| `prefix_padding_ms` | config | `300` | Audio kept before detected speech. |
| `silence_duration_ms` | config | `800` | Silence that closes a segment. |
| `noise_reduction` | config | `far_field` | `far_field`, `near_field` or empty (disabled). |
| `send_batch_ms` | config | `100` | Upload batch size. |

Session languages map to ISO 639-1 codes sent in
`session.audio.input.transcription.language`:

| EchoLingo | Sent to OpenAI |
| --- | --- |
| `en` | `en` |
| `zh` | `zh` |
| `ja` | `ja` |
| `ko` | `ko` |
| `auto` | key omitted; the model detects the language |

All four product languages are supported by both `gpt-4o` transcription
models. Any other value falls back to detection (the key is omitted) rather
than being refused.

## What leaves the machine

- After **Audio upload** consent and only inside a running session: PCM16 mono
  audio resampled locally to 24 kHz, base64 encoded in
  `input_audio_buffer.append` frames. Capture, enhancement, AGC, local VAD
  annotation and the replay ring stay on the machine.
- The `session.update` message: model, language, VAD and noise-reduction
  settings, plus the lecture context (topic and hint terms from the context
  and the glossary, at most 1000 characters) as the transcription `prompt`.
  No transcript history or translation is sent.
- Nothing is stored by EchoLingo on OpenAI's side; audio retention on the
  provider is governed by the OpenAI API data-usage policy for the account.
- The connection test sends no audio and no session configuration.

## Transcript tiers

OpenAI's server VAD commits speech segments on its own. Each segment becomes a
conversation item whose transcript streams as append-only
`conversation.item.input_audio_transcription.delta` events and closes with
`…completed`. Deltas are never revised, so the adapter forwards the
accumulated item text as *confirmed* text: sentence units are released as
STABLE rows as soon as a sentence boundary appears, and the `completed`
transcript closes the utterance with a FINAL row. There is no unstable
(PARTIAL) tier for this provider. Deltas (and completions) of a later segment
that arrive while an earlier one is still streaming are buffered and released
in segment order once the earlier one closes, so rows never interleave. Text
already forwarded for a segment that later reports `…failed` stays (committed
text never rolls back); the failure surfaces as a recoverable ERROR event.

`input_audio_buffer.speech_stopped` (`audio_end_ms`) marks the speech end used
for commit latency and for the replay anchor; the adapter converts the
provider's per-connection clock into ring time so reconnects keep the metrics
correct.

## Reconnect and finish

- A dropped socket is reconnected with the shared retry policy
  (`network.reconnect_budget_s`); `session.update` is re-sent and the last
  `replay_overlap_ms` before the last speech end is replayed from the local
  ring. Committed text never rolls back; per-connection item state is dropped.
- Stop sends `input_audio_buffer.commit`. The finish resolves on the
  `completed` event of the committed remainder, on the
  `input_audio_buffer_commit_empty` error (server VAD already committed
  everything and no segment is still transcribing), on the `failed` event of
  the last open segment, or on a server close. If a segment is still being
  transcribed when Stop arrives, the adapter waits for its `completed` event
  (bounded by `finish_timeout_s`, 30 s).
- No application-level keepalive is needed; WebSocket ping/pong keeps idle
  connections alive.

## Errors

| Surface | Code / message | Meaning |
| --- | --- | --- |
| handshake HTTP 401 | `authentication_failed` — "OpenAI rejected the API key (HTTP 401). Verify the key is active and the project has Realtime API access." | Invalid, revoked or wrong-project key. |
| handshake HTTP 403 | `authentication_failed` — "OpenAI refused the Realtime transcription session (HTTP 403) …" | The project lacks Realtime API access or the model is not enabled. |
| handshake HTTP 429 | `rate_limited` — "OpenAI rate limit or quota exceeded (HTTP 429) …" | Usage limit or billing quota reached; recoverable. |
| handshake timeout / other | `network_error` — "OpenAI Realtime connection timed out …" / "could not be reached (…)" | Network or proxy problem; the message names only the exception type. |
| event `error` | provider `error.code` (or `error.type`) | Recoverable when the code/type contains `rate_limit` or `server_error`; otherwise terminal for the session. |
| event `…transcription.failed` | provider `error.code` | One segment could not be transcribed; the session continues (recoverable). |
| finish | `provider_timeout` / `network_error` | No finish acknowledgement within `finish_timeout_s`, or the socket died while finishing. |

Error text never contains the key, the URL or raw handshake bodies. Provider
messages that quote a (masked) `sk-…` key are redacted before they reach the
UI or logs.

## What "Test" does

The probe opens the authenticated WebSocket (`Authorization: Bearer`,
`OpenAI-Beta: realtime=v1`), measures the handshake latency and closes the
socket. It uploads no audio and sends no `session.update`; the result reports
`model` and `host` (`api.openai.com`). A successful test is a configuration
check, not a latency benchmark: EN/ZH/JA/KO latency and accuracy evidence
(required for Auto-route eligibility) is still pending until measured with a
funded account.

## Region notes

OpenAI serves the Realtime API from a single global endpoint; there is no
region setting. `HTTPS_PROXY` / `NO_PROXY` are honoured by `websockets`.
