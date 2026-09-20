# Cloud providers: setup and connection tests

EchoLingo is local-first. Cloud recognition or translation is opt-in per
provider and per privacy flag: cloud ASR needs **Settings → Privacy → Audio
upload**, cloud translation needs **Transcript upload**. Nothing is uploaded
without those switches, and the connection tests below never upload
microphone audio.

Credentials are stored in the macOS Keychain (service
`app.echolingo.desktop`) and reach the inference sidecar only as environment
variables for the lifetime of a session. On the command line the same
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

## Using the Settings → Cloud cards

1. Open the card for the vendor, paste the key (and any non-secret field such
   as the DashScope workspace ID, which is optional) and choose **Save**.
2. Set the card's options when they exist: the Qwen Cloud **Region**
   (Singapore or Beijing), model names, the DeepL plan, the Azure region.
3. Choose **Test**. Recognition tests perform the authenticated handshake
   only. Translation tests send one fixed English sentence ("Welcome to the
   lecture.") and run only when Transcript upload is enabled.
4. Pick the provider for a session on the Live screen or in **Settings →
   Session defaults**, or leave the route on **Auto** and set a **Preferred
   cloud recognizer / translator**: Auto uses it when the local model misses
   its latency target.

A test is a configuration check, not a latency benchmark. The EN/ZH/JA/KO
cloud benchmarks in `docs/benchmark.md` stay `PENDING CREDENTIALS` until they
run with real accounts and fixtures.

## Interpreting failures

| Code | Meaning | What to do |
| --- | --- | --- |
| `authentication_failed` (HTTP 401/403) | key rejected, wrong region, model not enabled | The message names the host that was tried. For Qwen Cloud check the Region setting matches the console where the key was created; clear the workspace ID to use the key's default workspace. |
| `rate_limited` (HTTP 429/456) | quota or rate limit | Wait, or switch the preferred cloud provider. |
| `network_error` / `provider_timeout` | host unreachable | Check connectivity and proxy settings; `HTTPS_PROXY`/`NO_PROXY` are honoured by the sidecar. Mainland China networks usually need a proxy for OpenAI, Deepgram, AssemblyAI, Gladia, DeepL, Google and Azure; DashScope Beijing, DeepSeek and SiliconFlow are reachable directly. |
| `privacy_policy_denied` | consent switch off | Enable Audio upload / Transcript upload in Settings → Privacy. |
| `unknown_provider` | stale preference | Re-select the provider; the catalog is `configs/providers.json`. |

Sidecar diagnostics are appended to `logs/sidecar.log` inside the app data
directory (shown under **Settings → Advanced**); values of credentials are
never written there.

## Command-line sessions

```bash
export DEEPGRAM_API_KEY=...            # or DASHSCOPE_API_KEY, OPENAI_API_KEY, ...
export ECHOLINGO_QWEN_REGION=beijing   # DashScope only; default singapore
conda run -n echolingo-spike1 python -m echolingo --config configs/lecture.toml run \
  --asr deepgram --translation deepl --language en --target-language zh
```

`--asr` / `--translation` accept every id printed by
`python -m echolingo.backends.registry`; the config's `privacy` table must
allow the corresponding upload.
