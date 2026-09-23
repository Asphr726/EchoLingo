# Cloud providers: setup and connection tests

EchoLingo is local-first. Cloud recognition or translation is opt-in per
provider and per privacy flag: cloud ASR needs **Settings → Privacy → Audio
upload**, cloud translation needs **Transcript upload**. Nothing is uploaded
without those switches, and the connection tests below never upload
microphone audio.

Credentials are stored in the macOS Keychain (service
`app.echolingo.desktop`) and reach the local inference service only as
environment variables for the lifetime of a session. On the command line the same
variables are development fallbacks. Secret values never appear in settings
files, history, events or logs.

| Group | Providers | Free tier | Page |
| --- | --- | --- | --- |
| Qwen Cloud (Alibaba Model Studio) | Qwen realtime ASR, Qwen-MT | Beijing region only, 90 days per model | [qwen_cloud](providers/qwen_cloud.md) |
| OpenAI | Realtime transcription, chat translation | none (pay as you go) | [openai_realtime](providers/openai_realtime.md), [chat models](providers/openai_chat.md) |
| Deepgram | Streaming ASR (nova-3) | ~$200 starter credit | [deepgram](providers/deepgram.md) |
| AssemblyAI | Universal-Streaming ASR (English) | starter credit | [assemblyai](providers/assemblyai.md) |
| Gladia | Live ASR v2 | monthly live minutes | [gladia](providers/gladia.md) |
| DeepSeek, Gemini, Groq, OpenRouter, SiliconFlow, custom endpoint | Chat translation (OpenAI-compatible) | Gemini / Groq / OpenRouter / SiliconFlow free tiers | [openai_chat](providers/openai_chat.md) |
| DeepL | Translation | API Free: 500k characters/month | [deepl](providers/deepl.md) |
| Google Cloud Translation | Translation v2 | 500k characters/month | [google_translate](providers/google_translate.md) |
| Azure AI Translator | Translation v3 | F0: 2M characters/month | [azure_translator](providers/azure_translator.md) |

## Using the Settings → Cloud providers cards

1. Open the card for the vendor, paste the key (and any non-secret field such
   as the DashScope workspace ID, which is optional) and choose **Save**.
2. Set the card's options when they exist: the Qwen Cloud **Region**
   (Singapore or Beijing), model names, the DeepL plan, the Azure region.
3. Choose **Test**. Recognition tests perform the authenticated handshake
   only. Translation tests send one fixed English sentence ("Welcome to the
   lecture.") and need Transcript upload: if it is off, a translation-only
   card asks for it before testing, and a card with both roles tests
   recognition first and offers **Include translation in the test…**.
4. Pick the provider for a session on the Live screen or in **Settings →
   Models → Provider preferences**, or leave the route on **Auto** and set a **Preferred
   cloud recognizer / translator**: Auto uses it when the local model misses
   its latency target.

A test is a configuration check, not a latency benchmark. EchoLingo does not
yet publish EN/ZH/JA/KO latency or accuracy figures for the cloud providers.

## Interpreting failures

| Code | Meaning | What to do |
| --- | --- | --- |
| `authentication_failed` (HTTP 401/403) | key rejected, wrong region, model not enabled | The message names the host that was tried. For Qwen Cloud check the Region setting matches the console where the key was created; clear the workspace ID to use the key's default workspace. |
| `rate_limited` (HTTP 429/456) | quota or rate limit | Wait, or switch the preferred cloud provider. |
| `network_error` / `provider_timeout` | host unreachable | Check connectivity and proxy settings; `HTTPS_PROXY`/`NO_PROXY` are honoured by the inference service (for the app, set them with `launchctl setenv` and reopen it). Mainland China networks usually need a proxy for OpenAI, Deepgram, AssemblyAI, Gladia, DeepL, Google and Azure; DashScope Beijing, DeepSeek and SiliconFlow are reachable directly. |
| `privacy_policy_denied` | consent switch off | Enable Audio upload / Transcript upload in Settings → Privacy. |
| `unknown_provider` | stale preference | Re-select the provider on the Live screen or in **Settings → Models**. |

Diagnostics from the inference service are appended to `logs/sidecar.log`
inside the app data directory (shown under **Settings → Advanced**); values
of credentials are never written there.

## AI assistant: session notes and titles

History can turn a recorded session into structured study notes and names
each session automatically.

1. Add a key for one chat provider under **Settings → Cloud providers**:
   Qwen (Alibaba Model Studio — the Beijing region has free quota), OpenAI,
   DeepSeek, Google Gemini, Groq, OpenRouter, SiliconFlow, or a custom
   OpenAI-compatible endpoint (Ollama, LM Studio, vLLM).
2. In **Settings → AI assistant** choose that provider and, optionally, a
   model (the default is shown as the placeholder; long lectures need a model
   with a large context window, e.g. `qwen-plus`, `gpt-4o-mini`,
   `gemini-2.0-flash`).
3. Turn on **Send transcripts and attached files to this model for notes and
   titles**, or allow it in the dialog the first time you create notes.
   Nothing is sent before this consent is given; it covers the selected
   provider only, and switching providers turns it off. Audio never leaves
   the Mac for notes or titles. **Test** sends one fixed prompt and no
   transcript.
4. With **Name sessions automatically when they end** on, a session with at
   least 30 words gets an AI title in its target language right after Stop.
   Titles you rename yourself are never replaced.

**Creating notes.** Open a session in History → **AI notes** → **Create
notes**. EchoLingo asks whether to add course materials: PDF, PowerPoint
(.pptx), Word (.docx), Markdown, LaTeX, CSV or plain text, up to 5 files of
25 MB each. Text is extracted on this Mac (scanned PDFs without a text layer
are reported and skipped); only the extracted text is sent, so every provider
accepts it. Notes are written in the session's translation language, stream
in as they are generated, render Markdown and LaTeX, and are saved with the
session: reopening it shows them without calling the model again. **Export
.md**, **Copy** and **Regenerate** are on the notes toolbar; each section
heading carries a time range that jumps to that part of the transcript.

Sessions longer than about 20 minutes are written in 12–15 minute parts that
continue one document, followed by the title and overview.

| Error | Message in the notes panel | What to do |
| --- | --- | --- |
| `privacy_policy_denied` | Sending transcripts to the AI assistant is turned off. | enable consent in Settings → AI assistant |
| `not_configured` | The AI assistant is not set up yet. | choose a provider and save its key |
| `authentication_failed` | The provider rejected the API key. | check the key (and the Qwen region) in Cloud providers |
| `rate_limited` | The provider’s rate limit or quota was reached. | wait, or pick another provider |
| `context_too_long` | This session is too long for the selected model. | choose a larger-context model, or attach fewer files |
| `network_error` | EchoLingo could not reach the provider. | check the network connection and proxy |

## Lecture context (recognition and translation)

The **Lecture context** panel on the Live screen takes the topic, names and
terms of the next session, one per line (`term = translation` adds a glossary
pair); **Import from slides…** fills it from course materials. **Settings →
Translation → Glossary** keeps terms for every session. The local Qwen
recognizer receives the context in its prompt; cloud recognizers receive it,
or its terms, as their prompt, hint or keyterm parameter; translation
receives the topic as background and the glossary pairs where the provider
supports them (each provider page lists exactly
what it receives). Context reaches cloud
providers only under the same upload switches as audio and transcripts.

## Command-line sessions

```bash
export DEEPGRAM_API_KEY=...            # or DASHSCOPE_API_KEY, OPENAI_API_KEY, ...
export DEEPL_API_KEY=...
export ECHOLINGO_QWEN_REGION=beijing   # DashScope only; default singapore
conda run -n echolingo-spike1 python -m echolingo --config configs/lecture.toml listen \
  --asr deepgram --translation deepl --language en --target-language zh \
  --allow-audio-upload --allow-transcript-upload
```

`--asr` / `--translation` accept every id printed by
`python -m echolingo.backends.registry`. Cloud providers need the matching
upload permission, either from `--allow-audio-upload` /
`--allow-transcript-upload` or from the config's `privacy` table.
