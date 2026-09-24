# DeepL (translation)

Provider id `deepl`, credential group `deepl`, adapter
`src/echolingo/backends/translation/deepl.py` (`DeepLTranslation`, a
`RestTranslationBase` subclass over the DeepL API v2 `POST /v2/translate`
endpoint). Translation only; DeepL has no streaming ASR.

## Getting a key

1. Sign up for **DeepL API Free** or **DeepL API Pro** at
   <https://www.deepl.com/pro-api> (the Free plan needs a card for
   verification but is not charged).
2. Copy the **Authentication Key for DeepL API** from
   <https://www.deepl.com/your-account/keys>.
3. In EchoLingo open **Settings → Cloud providers → DeepL**, paste the key, pick the
   **Plan** (API Free / API Pro) and choose **Save**. The key is stored in the
   system secure store (macOS Keychain, Windows Credential Manager, or Secret
   Service on Linux) and reaches the sidecar only as the `DEEPL_API_KEY`
   environment variable for the lifetime of a session. On the command line
   `DEEPL_API_KEY` is the development fallback.

| Plan | Quota | Host | Key |
| --- | --- | --- | --- |
| API Free | 500,000 characters per month, then HTTP 456 until the next billing period | `api-free.deepl.com` | ends with `:fx` |
| API Pro | pay per character (plus subscription) | `api.deepl.com` | no suffix |

### Plan setting and the `:fx` rule

The plan is the `tier` setting (`free` by default), forwarded by the desktop
shell as `ECHOLINGO_DEEPL_TIER=free|pro` and also settable in the config as
`[translation.deepl] tier = "pro"`. The value is case-insensitive; anything
other than `free`/`pro` is ignored by the config loader and rejected by the
adapter.

DeepL API Free keys always end with `:fx`, and the Pro host rejects them with
HTTP 403 ("Wrong endpoint"). EchoLingo therefore sends any key ending in `:fx`
to `api-free.deepl.com` **regardless of the plan setting**; the setting only
matters for keys without the suffix. A Pro key with the plan left on
API Free fails with HTTP 403: switch the plan to API Pro.

## What leaves the machine

Only when **Settings → Privacy → Transcript upload** is on; the adapter
refuses to start otherwise (`privacy_policy_denied`). Per request, over HTTPS
to the host above:

- the source text of one **stable** transcript unit (`text`);
- the lecture topic (at most 200 characters) and the source text of the
  previous two stable units as DeepL `context`, joined by newlines, capped at
  600 characters. Target-language text is never sent,
  so nothing translated can be echoed back. DeepL documents `context` as
  not translated and, at the time of writing, not counted toward billing;
  check the DeepL API reference for the current terms;
- `source_lang`, `target_lang`, `model_type`, `preserve_formatting: true`,
  `split_sentences: "nonewlines"`;
- the key in the `Authorization: DeepL-Auth-Key …` header (never in the URL,
  logs or error messages) and `User-Agent: EchoLingo`.

No audio, no session history beyond the two previous source units, no
glossary terms, no target text.

### Only stable sentences are sent

The registry marks DeepL `streaming_partials=False`. The translation
scheduler then translates only stable units and finals and never the
provisional live tail, because a request/response translator gains nothing
from re-translating an unfinished sentence and each attempt would bill
characters. This saves API Free quota; the trade-off is that the target
caption updates once per committed sentence rather than word by word. Each
request yields exactly one FINAL translation event.

### Language codes

| EchoLingo | `source_lang` | `target_lang` |
| --- | --- | --- |
| `en` | `EN` | `EN-US` (`en-GB` → `EN-GB`) |
| `zh` | `ZH` | `ZH-HANS` (`zh-TW` / `zh-Hant` → `ZH-HANT`) |
| `ja` | `JA` | `JA` |
| `ko` | `KO` | `KO` |
| `auto` | omitted (DeepL detects; the result is kept as `last_detected_source_language`) | — |

Other codes are upper-cased and passed through (`de` → `DE`); DeepL answers
HTTP 400 "not supported" for pairs it does not offer.

### Model type

`[translation.deepl] model_type` (default `latency_optimized`) is sent as
DeepL `model_type`; `quality_optimized` and `prefer_quality_optimized` are
the other documented values (`quality_optimized` is only available for some
language pairs, `prefer_quality_optimized` falls back). The event's model
name is `deepl-<model_type>`; when DeepL reports `model_type_used`, the event
carries the model that actually ran (`deepl-quality_optimized`).

### Glossary

DeepL only applies glossaries that were created beforehand through its
glossary endpoint and referenced by `glossary_id`; there is no per-request
term list. EchoLingo's glossary terms are therefore **ignored** by this
provider (the sidecar logs one warning with the number of terms, never the
terms themselves). Use the local Hy-MT route or Qwen-MT when term enforcement
matters.

## What "Test" does

**Settings → Cloud providers → DeepL → Test** builds the adapter from the saved key and
plan, forces transcript consent for the probe only, and translates the fixed
sentence "Welcome to the lecture." from English to Chinese. It sends nothing
from the microphone or the session history. A success reports the model name
(`deepl-<model_type>`) and the round-trip latency of that one request; the
probe is a configuration check, not a latency benchmark (EN/ZH/JA/KO figures
have not been measured yet). The same host rule
applies: a `:fx` key is probed against `api-free.deepl.com`.

## Errors

| HTTP | EchoLingo code | Meaning | What to do |
| --- | --- | --- | --- |
| 401 / 403 | `authentication_failed` | key rejected, or Pro key on the Free host / Free key on the Pro host | Message: "DeepL rejected the authentication key (HTTP 403). Free keys end with ':fx' and must use the API Free plan." followed by DeepL's own reason when it sends one. Re-copy the key; set the plan to match the key. |
| 456 | `rate_limited` (recoverable) | monthly character quota exhausted | Wait for the next billing period, upgrade, or switch the preferred cloud translator. |
| 429 / 529 | `rate_limited` (recoverable) | too many requests | The scheduler retries the stable unit; sustained bursts mean the lecture is producing more sentences than the plan allows. |
| 400 | `bad_request` | invalid parameter; "unsupported language" when DeepL says a language is not supported | Pick a supported language pair. Other 400s are retried once without context. |
| 413 | `request_too_large` | request body over DeepL's limit | Retried once without context. |
| 404 | `not_found` | endpoint missing on the selected host | Check for a proxy rewriting the URL. |
| 5xx | `http_<status>` | DeepL outage | Retry later; the local Hy-MT route keeps working. |
| — | `provider_timeout` | no answer within `timeout_s` (20 s default) or the request budget | Check connectivity. |
| — | `network_error` (recoverable) | DNS/TLS/connection failure | Mainland China networks usually need a proxy; `HTTPS_PROXY`/`NO_PROXY` are honoured. |
| — | `privacy_policy_denied` | Transcript upload is off | Enable it in Settings → Privacy. |
| — | `invalid_response` / `empty_response` | non-JSON body or no `translations` entry | Usually a proxy or captive portal; retry. |

Error messages never contain the key; if DeepL ever echoed it, the adapter
replaces it with `[redacted]`.

## Validation

- Unit tests: `NUMBA_CACHE_DIR=/tmp/echolingo-numba-cache conda run -n
  echolingo-spike1 pytest -q tests/test_deepl_backend.py` (mock transport,
  no network) — PASS.
- Live probe and EN/ZH/JA/KO latency/accuracy: pending credentials; DeepL
  is not `auto_route_eligible` until they have been measured.
