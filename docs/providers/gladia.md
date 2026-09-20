# Gladia live (cloud ASR)

Provider id `gladia`. Adapter: `src/echolingo/backends/asr/gladia.py`
(`GladiaAsrBackend`, built on `CloudStreamingAsrBase`). Registry entry and
credential group: `gladia` in `src/echolingo/backends/registry.py`.

## Getting a key

1. Sign in at <https://app.gladia.io/> and create an API key.
2. In EchoLingo, open Settings → Providers → Gladia and paste the key. The
   desktop shell stores it in the OS secure store and injects it into the
   sidecar as `GLADIA_API_KEY`; the environment variable is a development
   fallback only.
3. Select "Gladia live (cloud)" as the recognition provider and enable
   "Allow audio upload" (`privacy.audio_upload_allowed`). The adapter refuses
   to start without that flag, before any network request.

Free plan: a monthly allowance of real-time minutes (check the dashboard for
the current quota); live minutes beyond the allowance require a paid plan.

## What leaves the machine

- The session configuration (encoding, sample rate, model, language) via
  `POST https://api.gladia.io/v2/live` with the `x-gladia-key` header.
- Raw 16 kHz PCM16 audio as binary WebSocket frames, only after
  `start_session` with `audio_upload_allowed=True`. Gladia processes the audio
  on its own servers; no transcript or glossary text is uploaded by this
  adapter. Nothing else (session context, custom vocabulary) is sent.
- Capture, enhancement, AGC, VAD, resampling and the local audio ring stay on
  the machine. Server VAD events are annotations only; local audio is never
  dropped.

The REST init returns a single-use `wss://api.gladia.io/v2/live?token=…` URL.
The adapter connects to it **without** any auth header, keeps it private on
the instance (`endpoint` reports the REST endpoint instead) and never echoes
it, the key or a response body in error messages or logs.

## Protocol (Gladia real-time v2)

| Step | Adapter |
| --- | --- |
| Session init | `POST /v2/live` body: `encoding=wav/pcm`, `sample_rate=16000`, `bit_depth=16`, `channels=1`, `model` (`asr.gladia.model`, default `solaria-1`), `language_config` (omitted for `auto`), `messages_config` (partials, finals, speech events, errors, lifecycle events; acks and processing events off). |
| Audio | Binary PCM16 frames, `asr.gladia.send_batch_ms` (100 ms) batches. No resampling (Gladia accepts 16 kHz). |
| `transcript` `is_final=false` | PARTIAL: the utterance text so far is the unstable tail. |
| `transcript` `is_final=true` | STABLE sentence units + FINAL; `utterance.end` becomes the speech-end anchor. |
| `speech_start` / `speech_end` | Ignored / speech-end anchor (`data.time`). |
| `error` | ERROR event with the provider code and message; rate-limit/timeout codes are marked recoverable. |
| Stop | `{"type": "stop_recording"}`; Gladia flushes the remaining finals, then `post_final_transcript` (and `end_session`) resolves `finish_session`. A 30 s finish timeout yields a recoverable `provider_timeout` ERROR instead of blocking Stop. |
| Keepalive | None: the continuous audio stream keeps the socket alive. |

Timeline: Gladia reports utterance times relative to the audio it received on
the current connection. The adapter records the source time of the first chunk
uploaded on each connection and shifts `speech_end`/`utterance.end` back into
source time, so replay anchors and commit latency stay consistent across
reconnects.

## Languages

| EchoLingo | Gladia `language_config.languages` |
| --- | --- |
| `en` | `["en"]` |
| `zh` | `["zh"]` |
| `ja` | `["ja"]` |
| `ko` | `["ko"]` |
| `auto` | omitted (Gladia auto-detects the spoken language) |

`code_switching` is off so one language is locked per session. Any other
source language raises `ConfigurationError` before the REST init.

## Reconnect

Gladia session URLs are single-use, so every reconnect performs a fresh REST
init and then replays the local audio ring from `replay_overlap_ms` before the
last speech end. Committed text never rolls back; the reconciler merges
replayed finals by overlap. An HTTP 401/402/403 during reconnect stops the
retry loop immediately (`AuthenticationError`); other failures retry within
`network.reconnect_budget_s`.

## Errors

| Condition | Exception / code | Message (never contains the key or a URL) |
| --- | --- | --- |
| No key configured | `AuthenticationError` / `authentication_failed` | "Gladia requires an API key (create one in the Gladia dashboard)." |
| HTTP 401 | `AuthenticationError` | "Gladia rejected the API key (HTTP 401). Verify the key in the Gladia dashboard (app.gladia.io) and that it has not been revoked." |
| HTTP 402 / 403 | `AuthenticationError` | "Gladia refused the live session (HTTP 40x). The plan or quota does not allow real-time transcription; check the plan and the remaining live minutes in the Gladia dashboard." |
| HTTP 429 | `RateLimitError` / `rate_limited` | "Gladia rate limit exceeded (HTTP 429): too many concurrent live sessions or requests. Retry shortly." |
| HTTP 400 / 422 | `BackendError` / `backend_error` | "Gladia rejected the session configuration (HTTP 40x). Check the model and language settings." |
| HTTP 5xx | `BackendUnavailableError` / `backend_unavailable` | "Gladia is temporarily unavailable (HTTP 5xx). Retry later." |
| Timeout | `ConnectionError` / `network_error` | "Gladia connection timed out. Check network access and try again." |
| Other transport error | `ConnectionError` / `network_error` | "Gladia could not be reached (<ExceptionName>). Check the network and any proxy settings." |
| `{"type": "error"}` frame | ERROR event with the provider code | provider message, surfaced as a transcript ERROR row |
| Audio upload not allowed | `PolicyDeniedError` / `privacy_policy_denied` | raised before any request |

## What "Test" does

The Settings "Test" button (`probe_connection`) performs the REST session init
only: it validates the key, plan/quota and network reachability, reports the
REST round-trip as `handshake_latency_ms` together with `model` and `host`
(`api.gladia.io`), and returns. No WebSocket is opened and no audio is sent.
Gladia bills live transcription on audio duration, so a probe that sends no
audio should not consume live minutes; the unused session expires on Gladia's
side.

## Region and network notes

Gladia serves a single global endpoint (`api.gladia.io`); there is no region
selector. `HTTPS_PROXY` / `NO_PROXY` are honoured for both the REST init
(`httpx`) and the WebSocket (`websockets`).

## Status

Unit tests: `tests/test_gladia_backend.py` (mock transport and fake socket,
no live network). Inclusion in the Auto route (`auto_route_eligible`) is
pending measured EN/ZH/JA/KO latency and accuracy evidence in
`docs/benchmark.md`.
