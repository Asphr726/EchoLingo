# OpenAI-compatible chat translation

One adapter (`src/echolingo/backends/translation/openai_chat.py`) serves every
translation provider that speaks the OpenAI chat completions API. Pick a preset
as the translation provider on the Live screen or in **Settings → Models →
Provider preferences** (or `[translation] provider = "<preset>"`), store its
key, enable **Settings → Privacy → Transcript upload**, and the route is
live. ASR keeps routing independently: local Qwen3-ASR with a cloud chat
translator is a supported Hybrid combination.

| Preset id | Service | Default model | Key env (dev fallback) | Base URL |
|---|---|---|---|---|
| `openai_chat` | OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `deepseek_chat` | DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| `gemini_chat` | Google Gemini (AI Studio) | `gemini-2.0-flash` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `groq_chat` | Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `openrouter_chat` | OpenRouter | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `siliconflow_chat` | SiliconFlow | `Qwen/Qwen2.5-7B-Instruct` | `SILICONFLOW_API_KEY` | `https://api.siliconflow.cn/v1` |
| `custom_chat` | Any OpenAI-compatible server | (required, empty) | `CUSTOM_OPENAI_API_KEY` (optional) | (required, empty) |

Defaults live in `CHAT_PRESETS` in `src/echolingo/backends/registry.py`; the
exported `configs/providers.json` is what the desktop Settings window reads.
None of these presets is in the Auto route yet: `auto_route_eligible` needs
measured EN/ZH/JA/KO latency and accuracy evidence.

## Getting a key

Product credentials go into the OS secure store from the provider's card in
**Settings → Cloud providers → Save**; the desktop shell injects them into
the sidecar as environment variables. Setting the env var yourself is a
development fallback only. Never commit a key.

| Service | Where | Free tier |
|---|---|---|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | None; pay as you go per token. |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) → API keys | None; very low token prices, prepaid balance. |
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | AI Studio keys include a free tier with per-minute and per-day request limits; free-tier prompts may be used by Google to improve its products, so treat it as non-confidential. Not available in every country. |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | Free tier with per-minute request and token limits; enough for one lecture at a time. |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | Model ids ending in `:free` cost nothing but are rate limited and may be routed to providers that log prompts; paid models are billed per token. |
| SiliconFlow | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/) → API keys | New accounts receive a free quota; several small open models (Qwen2.5-7B among them) are free of charge. Servers are in mainland China. |

Free tiers change; check the vendor page before relying on one for a lecture.

## Choosing the model

Every keyed preset exposes a **Chat translation model** setting; the custom
endpoint exposes **Base URL** and **Model**. Each setting is also readable from
the environment for headless runs:

| Preset | Model env var | Base URL override |
|---|---|---|
| `openai_chat` | `ECHOLINGO_OPENAI_CHAT_MODEL` | `ECHOLINGO_OPENAI_BASE_URL` |
| `deepseek_chat` | `ECHOLINGO_DEEPSEEK_CHAT_MODEL` | `ECHOLINGO_DEEPSEEK_BASE_URL` |
| `gemini_chat` | `ECHOLINGO_GEMINI_CHAT_MODEL` | `ECHOLINGO_GEMINI_BASE_URL` |
| `groq_chat` | `ECHOLINGO_GROQ_CHAT_MODEL` | `ECHOLINGO_GROQ_BASE_URL` |
| `openrouter_chat` | `ECHOLINGO_OPENROUTER_CHAT_MODEL` | `ECHOLINGO_OPENROUTER_BASE_URL` |
| `siliconflow_chat` | `ECHOLINGO_SILICONFLOW_CHAT_MODEL` | `ECHOLINGO_SILICONFLOW_BASE_URL` |
| `custom_chat` | `ECHOLINGO_CUSTOM_OPENAI_CHAT_MODEL` | `ECHOLINGO_CUSTOM_OPENAI_BASE_URL` |

Precedence is environment variable, then the `[translation.<preset>]` TOML
section, then the preset default. The same TOML section also takes
`timeout_s` (30), `temperature` (0.1), `max_output_tokens` (400),
`background_spans` (2) and `api_key_env` (to read the key from a differently
named variable).

Small, fast instruction-tuned models are the right choice: captions are
retranslated on every stable revision, so first-delta latency matters more
than the last point of quality. The preset defaults are starting points, not
measurements: EN/ZH/JA/KO latency and accuracy for these presets have not
been measured with real keys yet.

## Custom endpoint

`custom_chat` targets any server that implements `POST
<base_url>/chat/completions` with SSE streaming. The key is optional: a local
server runs without one, and the adapter then sends no `Authorization` header.
Loopback hosts (`127.0.0.1`, `localhost`, `::1`) bypass `HTTPS_PROXY`.

| Server | Base URL | Model value |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | the pulled tag, e.g. `qwen2.5:7b` |
| LM Studio (local server tab) | `http://127.0.0.1:1234/v1` | the loaded model id shown in the server tab |
| vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct`) | `http://127.0.0.1:8000/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| llama.cpp `llama-server` | `http://127.0.0.1:8080/v1` | any (the server ignores it) |
| A hosted OpenAI-compatible gateway | its documented `/v1` URL | its model id, with the gateway key |

```toml
[translation]
provider = "custom_chat"

[translation.custom_chat]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5:7b"
```

The custom preset never sends `stream_options`, so servers that reject unknown
fields still stream; token usage is then reported only for non-streaming
window retranslations. The privacy flag is still required for `custom_chat`
because the registry cannot tell a LAN GPU box from a hosted gateway; the
built-in local Hy-MT route (`hymt_local`) remains the way to translate with
nothing leaving the process boundary.

## What leaves the machine

- The text of the span being translated, a glossary when one is configured,
  the lecture topic as background (at most 400 characters), and up to
  `background_spans` (default 2, at most 600 characters) of the *previous
  source-language* transcript as context. Earlier translations are
  never sent: chat models copy target text from the prompt back as the answer.
- Nothing is sent until **Transcript upload** is enabled. Audio never reaches
  a chat provider under any setting.
- OpenRouter requests additionally carry `HTTP-Referer:
  https://github.com/echolingo` and `X-Title: EchoLingo`, OpenRouter's
  attribution headers; no other preset sends them.
- Keys travel only in the `Authorization: Bearer` header, never in the URL,
  logs or error messages.

## What "Test" does

**Test** on the provider's card in **Settings → Cloud providers** builds the adapter from the stored credentials,
temporarily grants transcript consent and translates the fixed sentence
"Welcome to the lecture." from English to Chinese with a non-streaming
request. It uploads no session history and no audio. A successful test proves
the key, base URL and model id; it is not a latency benchmark.

## Prompt

The system message says: professional simultaneous interpreter for lecture
transcripts; translate from `<source language>` into `<target language>`;
output only the translation with no explanations, quotes or preface; keep
numbers, names, code identifiers and technical terms accurate. A `Glossary:`
block (`source => target`) and an `Earlier context (do not translate):` block
follow when they exist, and provisional (uncommitted) spans add "The text may
be an unfinished sentence; translate what is present without completing it."
The user message is the source text alone. `max_tokens` is
`min(max_output_tokens, max(24, 3 × source length + 16))`.

Answers are cleaned of a leading "Translation:" / "译文：" label, "Here is the
translation:" preambles and wrapping quotes. A stream that repeats the same
fragment four times is cut and committed with `finish_reason = "repetition"`;
one that exceeds the per-request timeout is committed with what has arrived
and `finish_reason = "timeout"`.

## Errors

| Symptom | Code | Meaning |
|---|---|---|
| "Cloud translation requires explicit transcript upload consent" | `privacy_policy_denied` | Enable **Settings → Privacy → Transcript upload**. |
| "<Provider> requires an API key." | `authentication_failed` | No key in the keychain or the provider's env var. |
| "Set the base URL for the custom endpoint" | configuration error | `custom_chat` selected with an empty base URL. |
| "<Provider> rejected the API key (HTTP 401/403)" | `authentication_failed` | Key invalid, revoked, wrong project, or the model is not enabled for the key (Gemini also answers 403 from unsupported countries). |
| "<Provider> rate limit exceeded (HTTP 429)" | `rate_limited` | Free-tier per-minute limit or an exhausted balance/quota (DeepSeek and SiliconFlow answer 429 when prepaid credit runs out). A committed span is retried once; a provisional span waits for the next revision. |
| "<Provider> reported HTTP 404 for model '…'" | `model_not_found` | Wrong model id for the preset, or Ollama has not pulled the tag. |
| "<Provider> rejected the request (HTTP 400): … stream_options" | `bad_request` | The server does not accept `stream_options`; select `custom_chat` for it, which omits the field. |
| HTTP 400 mentioning context length | `context_overflow` | A committed span is retried once without background context. |
| Other HTTP 4xx/5xx | `http_<status>` | Provider-side failure; a committed span is retried once, then reported as a translation error event. |
| `finish_reason = "timeout"` on a FINAL event | (not an error) | The provider was slower than `timeout_s` (30 s committed / 8 s provisional); the partial translation was committed. |
| Connection or TLS errors | `ConnectError` / `ReadTimeout` (httpx) | Check `HTTPS_PROXY` / `NO_PROXY` and that the base URL ends in `/v1` (or the vendor's documented prefix). |

## Regions and data handling

- OpenAI, Groq and OpenRouter are US-hosted; DeepSeek and `api.siliconflow.cn`
  are mainland-China-hosted; Gemini's endpoint is global but the free tier is
  not offered in every country (EEA, UK and Switzerland need a billed project
  at the time of writing).
- All presets honour `HTTPS_PROXY` / `NO_PROXY` from the environment through
  `httpx`, except loopback custom endpoints.
- Each provider's own retention and training policies apply to the uploaded
  transcript text; check the vendor's data-use page before translating
  confidential material, and prefer the local Hy-MT route when in doubt.
