# AssemblyAI Universal-Streaming (ASR)

| | |
|---|---|
| Provider id | `assemblyai` |
| Adapter | `src/echolingo/backends/asr/assemblyai.py` (`AssemblyAiAsrBackend`) |
| Model | `universal-streaming` (AssemblyAI Streaming API v3) |
| Endpoint | `wss://streaming.assemblyai.com/v3/ws` |
| Credential | `ASSEMBLYAI_API_KEY` (OS secure store in the desktop app; the environment variable is a development fallback only) |
| Privacy flag | `privacy.audio_upload_allowed` — the adapter refuses to start without it |
| Languages | English only (`en`); `auto` is treated as English |
| Auto route | not eligible (no EN/ZH/JA/KO benchmark evidence yet; English-only) |

## Getting a key

1. Sign up at <https://www.assemblyai.com/dashboard> and copy the API key
   shown on the dashboard home page.
2. In EchoLingo open **Settings → Cloud providers → AssemblyAI**, paste the
   key and choose **Save**. It is stored in the system secure store (macOS
   Keychain, Windows Credential Manager, or Secret Service on Linux).
3. Choose **Test**.

Free accounts include starter credit. Streaming is billed per session hour
(connection time, not speech time), so a session that stays open during a long
break still costs money. Pause the session instead of leaving it connected.

## What leaves the machine

* **Audio.** The full 16 kHz mono PCM16 stream of the session is uploaded to
  AssemblyAI as it is captured. Capture, enhancement, AGC, VAD, resampling and
  the audio ring stay local; VAD annotations are not sent and never gate the
  upload.
* **Hint terms.** The terms from the lecture context and the glossary travel
  as `keyterms_prompt` (at most 100 terms of at most 50 characters, 2000
  UTF-8 bytes in total). The topic line itself is not sent.
* **Nothing else.** No transcript, translation or session history is uploaded
  by this adapter.
* **Test** performs the authenticated WebSocket handshake only and
  sends no audio and no messages. It reports `model`, `host` and the
  handshake latency.

Sessions are processed in the United States; the streaming API exposes no
region selection. `HTTPS_PROXY` / `WSS_PROXY` / `NO_PROXY` from the environment
are honoured by the `websockets` client.

## Languages

| EchoLingo language | AssemblyAI | Behaviour |
|---|---|---|
| `en` | English | supported |
| `auto` | English | accepted; no language identification, English is assumed |
| `zh` | — | `start_session` raises `BackendUnavailableError`; config validation rejects it first |
| `ja` | — | same |
| `ko` | — | same |

The Auto route skips AssemblyAI for non-English sessions
(`ProviderSpec.supports_language`).

## Settings (`[asr.assemblyai]`)

| Setting | Default | Wire parameter |
|---|---|---|
| `format_turns` | `true` | `format_turns` — the final turn is repeated with punctuation and casing |
| `end_of_turn_confidence_threshold` | `0.7` | `end_of_turn_confidence_threshold` (0..1) |
| `min_end_of_turn_silence_when_confident_ms` | `160` | `min_end_of_turn_silence_when_confident` |
| `max_turn_silence_ms` | `2400` | `max_turn_silence` |
| `send_batch_ms` | `100` | size of each binary audio frame (50–1000 ms) |

Lower `end_of_turn_confidence_threshold` and `max_turn_silence_ms` close turns
sooner (lower commit latency, shorter rows); raise them for lecture audio
where the speaker pauses mid-sentence.

## Protocol mapping

| AssemblyAI message | EchoLingo event |
|---|---|
| `Begin` | ignored (session id kept as `provider_session_id`) |
| `Turn`, `end_of_turn=false` | PARTIAL: `transcript` (cumulative for the turn) plus any trailing `word_is_final=false` words |
| `Turn`, `end_of_turn=true`, unformatted while `format_turns=true` | PARTIAL: wait for the formatted repeat |
| `Turn`, `end_of_turn=true`, `turn_is_formatted=true` (or `format_turns=false`) | STABLE row(s) for the sentence units of `transcript`, then FINAL; speech end = last word `end` |
| repeated `Turn` for a closed `turn_order` | ignored |
| `Termination` | resolves `finish_session` |
| any message with `error` | ERROR event; recoverable when the text mentions a rate limit |

* Audio is sent as raw binary PCM16 frames. Batches shorter than 50 ms (only
  the flush at Stop) are padded with silence; batches longer than 1000 ms are
  split.
* Stop sends `ForceEndpoint` (closes the open turn) and then `Terminate`, and
  waits up to `finish_timeout_s` for `Termination`.
* No keepalive message is required.
* On reconnect the adapter replays the local audio ring from
  `replay_overlap_ms` before the last speech end, restarts turn numbering and
  shifts the provider's word timestamps (which restart at zero per socket) to
  source time so `commit_latency_ms` and the replay anchor stay correct.
  Committed text never rolls back; the reconciler merges overlap.

## Errors

| Symptom | Code | Meaning / action |
|---|---|---|
| `AssemblyAI rejected the API key (HTTP 401)` | `authentication_failed` | Wrong or revoked key. Reconnects stop immediately. |
| `AssemblyAI refused the streaming session (HTTP 403)` | `authentication_failed` | Key valid but the account lacks streaming access. |
| `AssemblyAI reports insufficient funds (HTTP 402)` | `backend_unavailable` | Add credit in the dashboard. |
| `AssemblyAI rate limit exceeded (HTTP 429)` | `rate_limited` | Too many concurrent sessions; retry later. |
| `AssemblyAI connection timed out` / `could not be reached` | `network_error` | Check network and proxy settings. |
| `AssemblyAI requires an API key …` | `authentication_failed` | No key in the secure store or `ASSEMBLYAI_API_KEY`. |
| `Cloud ASR requires explicit audio upload consent` | `privacy_policy_denied` | Enable **Settings → Privacy → Audio upload**. |
| `AssemblyAI Universal-Streaming transcribes English only …` | `backend_unavailable` | Choose English or another provider. |
| ERROR event `AssemblyAI: …` during a session | `provider_error` | Provider-side error text; recoverable only for rate limits. |
| `Cloud ASR finish timed out` | `provider_timeout` | `Termination` did not arrive; the transcript so far is kept. |

Error messages never include the key or the request URL.

## Testing

`tests/test_assemblyai_backend.py` uses a fake WebSocket (no network): URL
and header construction, binary PCM16 framing, every message kind, the
`ForceEndpoint`/`Terminate` handshake, probe without audio, the privacy and
credential gates, 401/402/429/timeout mapping without key leakage, reconnect
with ring replay and timestamp offsets, the language table and the registry
factory.

```
NUMBA_CACHE_DIR=/tmp/echolingo-numba-cache conda run -n echolingo-spike1 \
  pytest -q -p no:cacheprovider tests/test_assemblyai_backend.py
```

Live latency/accuracy measurements against real audio are still pending;
they need a funded account.
