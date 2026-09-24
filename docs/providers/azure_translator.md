# Azure AI Translator (translation)

Provider id `azure_translator`, credential group `azure_translator`, adapter
`src/echolingo/backends/translation/azure_translator.py` (`AzureTranslator`, a
`RestTranslationBase` subclass over the Azure AI Translator **Text API v3.0**
`POST /translate` endpoint). Translation only; the Azure Speech service is a
different resource and is not used.

## Getting a key

1. Sign in to the [Azure portal](https://portal.azure.com/) and choose
   **Create a resource → AI + Machine Learning → Translator** (search for
   "Translator"; do not pick "Speech" or the multi-service "Azure AI
   services" resource unless you already use one).
2. Pick a subscription, a resource group, a **Region** (for example
   `eastasia`, `japaneast`, `westeurope`, `eastus`) and a name. Choose
   **Global** as the region only if you do not need data residency; a
   global resource does not need the region setting below.
3. Pick the **Pricing tier**. **F0 (Free)** translates **2,000,000
   characters per month** at no cost (one F0 Translator resource per
   subscription); **S1** is pay per million characters. Create the resource.
4. Open the resource, then **Resource Management → Keys and Endpoint**. Copy
   **KEY 1** (KEY 2 is a spare for rotation) and note the **Location/Region**
   value.
5. In EchoLingo open **Settings → Cloud providers → Azure AI Translator**, paste the
   key, type the **Resource region** exactly as the portal shows it (lower
   case, no spaces: `eastasia`) and choose **Save**. The key is stored in the
   system secure store (macOS Keychain, Windows Credential Manager, or Secret
   Service on Linux) and reaches the sidecar only as the `AZURE_TRANSLATOR_KEY`
   environment variable for the lifetime of a session; the region is
   forwarded as `ECHOLINGO_AZURE_TRANSLATOR_REGION`. On the command line both
   variables are the development fallback:

   ```sh
   AZURE_TRANSLATOR_KEY=… ECHOLINGO_AZURE_TRANSLATOR_REGION=eastasia \
     conda run -n echolingo-spike1 python -m echolingo --config configs/lecture.toml doctor --json
   ```

| Tier | Quota | Note |
| --- | --- | --- |
| F0 (Free) | 2,000,000 characters per month, then HTTP 403 (Azure 403001) until the next month | one per subscription; no card charge |
| S1 (Standard) | pay per million characters | no monthly cap |

### Region setting

Regional Translator resources require the `Ocp-Apim-Subscription-Region`
header on every request; without it (or with the wrong value) Azure answers
HTTP 401 even though the key is correct. The region is the
`[translation.azure_translator] region` setting (default empty), also
settable as `ECHOLINGO_AZURE_TRANSLATOR_REGION`; the adapter lower-cases it
and removes spaces (`East Asia` → `eastasia`). Leave it **empty** for a
resource created with the Global region: the header is then omitted.

### Endpoint

The default endpoint is the global text-translation endpoint
`https://api.cognitive.microsofttranslator.com`, which Azure routes to the
nearest datacenter regardless of the resource's region. `[translation.azure_translator]
endpoint` overrides it for sovereign clouds or custom-domain resources;
the value must be the directory that contains `/translate` (trailing slashes
are stripped):

| Deployment | `endpoint` |
| --- | --- |
| Global (default) | `https://api.cognitive.microsofttranslator.com` |
| Custom domain / private endpoint | `https://<resource-name>.cognitiveservices.azure.com/translator/text/v3.0` |
| Azure Government | `https://api.cognitive.microsofttranslator.us` |
| Azure China (21Vianet) | `https://api.translator.azure.cn` |

## What leaves the machine

Only when **Settings → Privacy → Transcript upload** is on; the adapter
refuses to start otherwise (`privacy_policy_denied`). Per request, over HTTPS
to the endpoint above:

- `POST /translate?api-version=3.0&to=<target>&from=<source>&textType=plain`
  (`from` is omitted when the source language is `auto`, so Azure detects
  it; the detected code is kept locally as `last_detected_source_language`);
- the body `[{"Text": "<source text of one stable transcript unit>"}]`;
- the headers `Ocp-Apim-Subscription-Key` (the key, never in the URL, logs
  or error messages), `Ocp-Apim-Subscription-Region` (only when a region is
  set), `Content-Type: application/json`, `User-Agent: EchoLingo` and a
  random `X-ClientTraceId` (a UUID generated per request and kept as
  `last_trace_id`; Azure support asks for it when investigating a request).

No audio, no session history, no previous or target-side text, no glossary
terms.

### Only stable sentences are sent

The registry marks Azure `streaming_partials=False`. The translation
scheduler then translates only stable units and finals and never the
provisional live tail, because a request/response translator gains nothing
from re-translating an unfinished sentence and each attempt would count
characters against the F0 quota. The target caption updates once per
committed sentence rather than word by word; each request yields exactly one
FINAL translation event with model name `translator-v3`.

### Language codes

| EchoLingo | Azure `from` / `to` |
| --- | --- |
| `en` (`en-US`, `en-GB`) | `en` |
| `zh` (`zh-CN`, `zh-SG`, `zh-Hans`) | `zh-Hans` |
| `zh-TW` / `zh-HK` / `zh-Hant` | `zh-Hant` |
| `ja` | `ja` |
| `ko` | `ko` |
| `auto` (source only) | omitted (Azure detects; the result is kept locally) |

Other codes are lower-cased and passed through (`fr-CA` → `fr-ca`); Azure
answers HTTP 400 with error 400019/400023/400035/400036 for languages or
pairs it does not offer.

### Glossary

Azure's dynamic dictionary needs `<mstrans:dictionary>` markup with
`textType=html`; this adapter sends plain text, so EchoLingo's glossary terms
are **ignored** by this provider (the sidecar logs one warning with the
number of terms, never the terms themselves). Use the local Hy-MT route or
Qwen-MT when term enforcement matters.

## What "Test" does

**Settings → Cloud providers → Azure AI Translator → Test** builds the adapter from the
saved key, region and endpoint, forces transcript consent for the probe only,
and translates the fixed sentence "Welcome to the lecture." from English to
Chinese (`zh-Hans`). It sends nothing from the microphone or the session
history. A success reports the model name (`translator-v3`) and the
round-trip latency of that one request; the probe is a configuration check,
not a latency benchmark (EN/ZH/JA/KO figures have not been measured yet).

## Errors

Azure puts a numeric code in the body (`{"error": {"code": 401000,
"message": …}}`); EchoLingo repeats it as `Azure <code>` in the message,
followed by Azure's own text when there is one.

| HTTP | Azure code | EchoLingo code | Meaning | What to do |
| --- | --- | --- | --- | --- |
| 401 | 401000 | `authentication_failed` | key rejected, or a regional resource without (or with the wrong) region | Message: "Azure Translator rejected the key (HTTP 401, Azure 401000). Check the key and set the resource region (e.g. eastasia) for regional resources." Re-copy KEY 1; set the region to the portal's Location value, or clear it for a Global resource. |
| 401 | 401015 | `authentication_failed` | the key belongs to a Speech resource | Create a Translator resource and use its key. |
| 403 | 403001 | `authentication_failed` | F0 monthly quota (2M characters) exhausted | Wait for the next month, move the resource to S1, or switch the preferred cloud translator. |
| 403 | 403000 | `authentication_failed` | operation not allowed for this resource/key | Check the resource kind (Translator, not Speech) and that the key belongs to it. |
| 429 | 429000–429002 | `rate_limited` (recoverable) | request or character-per-hour limits exceeded | The scheduler retries the stable unit; sustained bursts mean the lecture produces more text than the tier allows. |
| 400 | 400019 / 400023 / 400035 / 400036 (and 400003, 400006) | `bad_request` | unsupported language or language pair | Pick a supported pair; the code number is in the message. |
| 400 | 400050 / 400077 | `request_too_large` | text or request over Azure's size limit | The unit is skipped; long stable units are rare. |
| 400 | other | `bad_request` | invalid parameter or body | Message carries the Azure code and text. |
| 404 | — | `not_found` | endpoint path wrong | A custom endpoint must end with the directory that contains `/translate` (see the endpoint table). |
| 408 | 408001 / 408002 | `provider_timeout` | Azure timed out server-side (custom system warming up) | Retry later. |
| 5xx | 500000 / 503000 | `http_<status>` | Azure outage | Retry later; the local Hy-MT route keeps working. |
| — | — | `provider_timeout` | no answer within `timeout_s` (20 s default) or the request's wall-clock budget | Check connectivity. |
| — | — | `network_error` (recoverable) | DNS/TLS/connection failure | Mainland China networks usually need a proxy for the global endpoint; `HTTPS_PROXY`/`NO_PROXY` are honoured. |
| — | — | `privacy_policy_denied` | Transcript upload is off | Enable it in Settings → Privacy. |
| — | — | `invalid_response` / `empty_response` | non-JSON body or no `translations` entry | Usually a proxy or captive portal; retry. |

Error messages never contain the key; if Azure ever echoed it, the adapter
replaces it with `[redacted]`. Messages name the endpoint host, never the
full URL.

## Validation

- Unit tests: `NUMBA_CACHE_DIR=/tmp/echolingo-numba-cache conda run -n
  echolingo-spike1 pytest -q tests/test_azure_translator_backend.py` (mock
  transport, no network) — PASS.
- Live probe and EN/ZH/JA/KO latency/accuracy: pending credentials; Azure
  is not `auto_route_eligible` until they have been measured.
