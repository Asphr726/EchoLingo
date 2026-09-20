# Google Cloud Translation (translation)

Provider id `google_translate`, credential group `google_translate`, adapter
`src/echolingo/backends/translation/google_translate.py` (`GoogleTranslateV2`,
a `RestTranslationBase` subclass over the Cloud Translation **Basic (v2)**
endpoint `POST https://translation.googleapis.com/language/translate/v2`).
Translation only; the model name reported in events is `nmt`.

## Getting a key

1. In the Google Cloud console pick (or create) a project with billing
   enabled, then enable the **Cloud Translation API** at
   <https://console.cloud.google.com/apis/library/translate.googleapis.com>.
2. Open **APIs & Services → Credentials → Create credentials → API key**.
   Under **API restrictions** restrict the key to **Cloud Translation API**
   only; leave application restrictions unset (the sidecar has no HTTP
   referrer and no fixed IP). Keys look like `AIza…` (39 characters).
3. In EchoLingo open **Settings → Cloud → Google Cloud Translation**, paste
   the key and choose **Save**. The key is stored in the macOS Keychain and
   reaches the sidecar only as the `GOOGLE_TRANSLATE_API_KEY` environment
   variable for the lifetime of a session. On the command line
   `GOOGLE_TRANSLATE_API_KEY` is the development fallback.

| Tier | Quota | Notes |
| --- | --- | --- |
| Free tier | 500,000 characters per month | Counted per project across v2 and v3; requires a billing account on the project even if nothing is charged. |
| Paid | per million characters beyond the free tier | Set a budget alert in the Cloud console; EchoLingo cannot see the running total. |

There is no region setting: the v2 endpoint is global. Cloud Translation
Advanced (v3, glossaries, adaptive MT) needs a service account rather than an
API key and is not used by this adapter.

## What leaves the machine

Only when **Settings → Privacy → Transcript upload** is on; the adapter
refuses to start otherwise (`privacy_policy_denied`). Per request, over HTTPS
to `translation.googleapis.com`:

- the source text of one **stable** transcript unit (`q`);
- `target`, `format: "text"` and, unless the source language is `auto`,
  `source`;
- the key in the `X-goog-api-key` header (never as the `?key=` query
  parameter, so it cannot land in proxy logs, `httpx` exception text or
  crash reports) and `User-Agent: EchoLingo`.

No audio, no previous units or session history, no target text, no glossary
terms. The request body is exactly
`{"q": ["…"], "target": "zh-CN", "source": "en", "format": "text"}`.

### Only stable sentences are sent

The registry marks Google `streaming_partials=False`. The translation
scheduler then translates only stable units and finals and never the
provisional live tail: a request/response translator gains nothing from
re-translating an unfinished sentence and each attempt would bill
characters. This keeps the free tier for committed text; the trade-off is
that the target caption updates once per committed sentence rather than word
by word. Each request yields exactly one FINAL translation event.

### Language codes

| EchoLingo | v2 `source` / `target` |
| --- | --- |
| `en` | `en` (`en-US`, `en-GB` → `en`) |
| `zh` | `zh-CN` (`zh-Hans`, `zh-SG` → `zh-CN`; `zh-TW`, `zh-HK`, `zh-Hant` → `zh-TW`) |
| `ja` | `ja` |
| `ko` | `ko` |
| `auto` | `source` omitted (Google detects; the result is kept as `last_detected_source_language`) |

Other codes are lower-cased to their base language and passed through
(`DE` → `de`); Google answers HTTP 400 "Invalid Value" for codes it does not
support. A request without an explicit target language is rejected locally
(`bad_request`) before anything is sent.

### HTML entities

Even with `format: "text"`, v2 sometimes returns HTML-escaped characters
(`&#39;`, `&amp;`, `&quot;`). The adapter passes `translatedText` through
`html.unescape` before the caption sees it.

### Glossary

Cloud Translation Basic has no glossary or per-request term list (glossaries
are a v3 feature). EchoLingo's glossary terms are therefore **ignored** by
this provider; the sidecar logs one warning with the number of terms, never
the terms themselves. Use the local Hy-MT route or Qwen-MT when term
enforcement matters.

## What "Test" does

**Settings → Cloud → Google Cloud Translation → Test** builds the adapter
from the saved key, forces transcript consent for the probe only, and
translates the fixed sentence "Welcome to the lecture." from English to
Chinese. It sends nothing from the microphone or the session history. A
success reports the model name (`nmt`) and the round-trip latency of that one
request; the probe is a configuration check, not a latency benchmark
(EN/ZH/JA/KO figures stay `PENDING CREDENTIALS` in `docs/benchmark.md`).

## Errors

| HTTP | EchoLingo code | Meaning | What to do |
| --- | --- | --- | --- |
| 400 with "API key not valid" | `authentication_failed` | the key is malformed, deleted or belongs to another project (Google answers 400, not 401, for this) | Message: "Google rejected the API key as not valid (HTTP 400). Re-copy the key from the Cloud console." Re-copy the key; check it was not truncated. |
| 403 | `authentication_failed` | key rejected, the Cloud Translation API is not enabled for the project, the key's API restriction excludes it, or billing is disabled | Message: "Google rejected the API key or the Cloud Translation API is not enabled for the project (HTTP 403)." followed by Google's own reason when it sends one. Enable the API, relax the key restriction to include Cloud Translation, or attach billing. |
| 401 | `authentication_failed` | credentials missing or invalid | Re-copy the key. |
| 429 | `rate_limited` (recoverable) | per-minute character/request quota or the monthly free tier exhausted (`RESOURCE_EXHAUSTED`) | The scheduler retries the stable unit; sustained 429s mean the quota in the Cloud console is too low for the lecture's pace, or the month's free characters are used up. |
| 400 (other) | `bad_request` | invalid parameter, usually an unsupported language code | Pick a supported language pair. |
| 413 | `request_too_large` | request body over Google's limit | Rare with single sentences; a proxy may be rewriting the request. |
| other 4xx / 5xx | `http_<status>` | endpoint missing (404, usually a proxy) or a Google outage | Retry later; the local Hy-MT route keeps working. |
| — | `provider_timeout` | no answer within `timeout_s` (20 s default) or the request budget | Check connectivity. |
| — | `network_error` (recoverable) | DNS/TLS/connection failure | Mainland China networks need a proxy for `translation.googleapis.com`; `HTTPS_PROXY`/`NO_PROXY` are honoured. |
| — | `privacy_policy_denied` | Transcript upload is off | Enable it in Settings → Privacy. |
| — | `invalid_response` / `empty_response` | non-JSON body or no `data.translations` entry | Usually a proxy or captive portal; retry. |

Error messages never contain the key or the request URL. When Google's own
reason is quoted, the configured key and anything shaped like a Google API
key (`AIza…`) is replaced with `[redacted]`.

## Validation

- Unit tests: `NUMBA_CACHE_DIR=/tmp/echolingo-numba-cache conda run -n
  echolingo-spike1 pytest -q tests/test_google_translate_backend.py` (mock
  transport, no network) — PASS (25 tests).
- Live probe and EN/ZH/JA/KO latency/accuracy: `PENDING CREDENTIALS`; Google
  Cloud Translation is not `auto_route_eligible` until `docs/benchmark.md`
  records them.
