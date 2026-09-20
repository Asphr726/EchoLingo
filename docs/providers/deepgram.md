# Deepgram streaming ASR

Provider id `deepgram` · adapter `src/echolingo/backends/asr/deepgram.py`
(`DeepgramAsrBackend`, built on `CloudStreamingAsrBase`) · tests
`tests/test_deepgram_backend.py` (fake WebSocket, no network).

Status: unit tests PASS (mock transport); live validation
`PENDING CREDENTIALS`; not eligible for the Auto route until EN/ZH/JA/KO
latency and accuracy evidence is recorded in `docs/benchmark.md`.

## Getting a key

1. Sign up at the [Deepgram console](https://console.deepgram.com/). New
   accounts receive roughly $200 of free credit without a card; streaming
   usage is billed per audio minute after that.
2. Create a project API key (**API Keys → Create a New API Key**). The key
   needs the default *Member* scope; nothing else is required for streaming.
3. In EchoLingo open **Settings → Models → Cloud credentials → Deepgram**,
   paste the key and choose **Save to Keychain**. The desktop shell injects it
   into the sidecar as `DEEPGRAM_API_KEY`; the adapter never logs, persists
   or places it in a URL. Setting `DEEPGRAM_API_KEY` in the process
   environment is a development fallback only.
4. Optional: pick the model in the same card (setting
   `ECHOLINGO_DEEPGRAM_MODEL`, default `nova-3`).

## What leaves the machine

- The authenticated WebSocket handshake (`Authorization: Token <key>` header,
  query parameters below).
- After **Start** with **Settings → Privacy → Audio upload** enabled: the
  session's 16 kHz mono PCM16 audio as raw binary frames, in 100 ms batches
  (`send_batch_ms`). Capture, enhancement, AGC, VAD and resampling stay local;
  VAD never drops audio before upload.
- `KeepAlive` and `CloseStream` control frames.

Nothing else: no transcript history, no context text, no diagnostics. Without
the privacy flag `start_session` raises `PolicyDeniedError` before a socket
is opened. Deepgram's data-retention terms apply to the audio it receives;
check the console's project settings if you need retention disabled.

Region: Deepgram serves `api.deepgram.com` from US infrastructure; there is
no EU/CN endpoint selector in the streaming API. `HTTPS_PROXY`/`NO_PROXY`
from the environment are honoured by the `websockets` client.

## What "Test connection" does

`probe_connection` opens the same `wss://api.deepgram.com/v1/listen?...`
URL the session would use and closes it as soon as the handshake succeeds. No
audio, no session messages, no `KeepAlive`. The result reports the handshake
latency, `host`, the model that would be used (`model`), the configured one
(`requested_model`) and the Deepgram `language` value. Because Deepgram
validates the query string during the handshake, an unsupported
model/language pair fails the test with an HTTP 400 rather than mid-session.

## Stream parameters

`wss://api.deepgram.com/v1/listen` with:

| Parameter | Value | Source |
| --- | --- | --- |
| `model` | `nova-3` (see fallback table) | `asr.deepgram.model` / `ECHOLINGO_DEEPGRAM_MODEL` |
| `language` | `en`, `zh`, `ja`, `ko`, `multi` or omitted | session language |
| `encoding`, `sample_rate`, `channels` | `linear16`, `16000`, `1` | fixed |
| `interim_results`, `punctuate`, `vad_events` | `true` | fixed |
| `smart_format` | `true`/`false` | `asr.deepgram.smart_format` (default `true`) |
| `endpointing` | ms, or `false` when `<= 0` | `asr.deepgram.endpointing_ms` (default 300) |
| `utterance_end_ms` | ms, omitted when `<= 0` | `asr.deepgram.utterance_end_ms` (default 1000) |

`keepalive_interval_s` (default 5.0; `0` disables) sends `{"type":"KeepAlive"}`
so Deepgram does not close an idle socket (it drops connections that receive
no audio for about 10 s). Reconnects reuse `network.replay_overlap_ms`,
`network.audio_ring_buffer_ms` and `network.reconnect_budget_s`.

## Language table

| Session language | `language` sent | Model used with default `nova-3` |
| --- | --- | --- |
| `en` | `en` | `nova-3` |
| `ja` | `ja` | `nova-3` |
| `zh` | `zh` | `nova-2` (fallback, see below) |
| `ko` | `ko` | `nova-2` (fallback, see below) |
| `auto` | `multi` on a Nova-3 model; omitted otherwise | `nova-3` |

`MODEL_LANGUAGE_FALLBACK` in the adapter maps `(model family, language)` to
the model actually put on the wire. It currently contains
`("nova-3", "zh") → "nova-2"` and `("nova-3", "ko") → "nova-2"`.

Uncertainty: Deepgram's per-model language matrix has been expanding through
2025–2026. Nova-2 lists Mandarin (`zh`) and Korean (`ko`) explicitly, and
Nova-3 documents English plus a multilingual (`multi`) set that includes
Japanese; whether Nova-3 accepts `zh`/`ko` as *monolingual* streaming
languages could not be verified offline when this adapter was written. A
wrong guess would fail every ZH/KO session at connect time (HTTP 400), so the
adapter prefers the model that is known to work. Once the Deepgram console
(or a live **Test connection** with `ECHOLINGO_DEEPGRAM_MODEL=nova-3` and a
ZH/KO session language) confirms support, delete the two entries; the table
is the only place to edit and `tests/test_deepgram_backend.py` pins the
behaviour. `auto` on a non-Nova-3 model means Deepgram's default (English):
the streaming API has no language detection outside `multi`.

## Message mapping

| Deepgram message | EchoLingo handling |
| --- | --- |
| `Results`, empty transcript | ignored (unless `speech_final` closes an open utterance) |
| `Results`, `is_final=false` | unstable tail → `PARTIAL` |
| `Results`, `is_final=true`, `speech_final=false` | confirmed chunk, tail cleared → `STABLE` units at sentence boundaries |
| `Results`, `is_final=true`, `speech_final=true` | confirmed chunk + utterance close → `STABLE` + `FINAL`; `start+duration` anchors the speech end |
| `UtteranceEnd` | closes an utterance that never saw `speech_final` (`FINAL`); otherwise only records the speech end for replay/latency |
| `SpeechStarted`, `Warning` | ignored (a warning's description is logged; it carries no credentials) |
| `Metadata` | end-of-stream acknowledgement after `CloseStream`; ignored before |
| `Error` | `ERROR` event with the provider code; recoverable when it hints at rate limiting or a timeout |

Committed text never rolls back: per-chunk finals are merged by overlap, so a
reconnect that replays the last `replay_overlap_ms` of audio does not
duplicate text. Deepgram's timestamps are relative to the audio on the
current socket; the adapter re-bases them to session time after a reconnect
so `commit_latency_ms` and the ring cutoff stay correct.

Stop sends `{"type":"CloseStream"}` and waits (up to `finish_timeout_s`) for
the final `Results` + `Metadata`. If the stream ends with text still open, a
`FINAL` is emitted before the event stream closes.

## Error table

| Symptom | Meaning | Action |
| --- | --- | --- |
| `HTTP 401` — Deepgram rejected the API key | key invalid, revoked or mistyped | create a new key in the console and save it again |
| `HTTP 402` — insufficient credit | free credit exhausted / no payment method | add credit in the console |
| `HTTP 403` — refused | key scope or project restriction | check the key's project and scope |
| `HTTP 400` — stream parameters rejected | model/language pair not supported | change the model or the fallback table |
| `HTTP 429` — rate limit | too many concurrent streams | retry shortly; the session reconnects automatically |
| `timed out` / `could not be reached` | network, proxy or DNS | check connectivity; no audio was uploaded by the probe |
| `Cloud ASR connection unavailable` (in-session) | reconnect budget exhausted | recoverable; the session can be restarted |

Messages name the provider and the HTTP status only; they never echo the URL,
query string or key.
