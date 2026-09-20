# Qwen Cloud setup and connection test

EchoLingo uses Alibaba Cloud Model Studio's international Singapore endpoint.
The API key and workspace ID must come from the same Singapore workspace.

1. Create or select a Singapore Model Studio workspace.
2. Create an API key using the official [API key guide](https://www.alibabacloud.com/help/en/model-studio/get-api-key).
3. Copy the workspace ID using the official [workspace guide](https://www.alibabacloud.com/help/en/model-studio/obtain-the-app-id-and-workspace-id).
4. In EchoLingo, open **Settings → Models → Cloud credentials**, enter both
   values, and choose **Save to Keychain**.
5. Choose **Test connection**. This performs an authenticated Qwen Realtime
   ASR WebSocket handshake but uploads no audio.
6. To test Qwen-MT in the same check, first enable **Settings → Privacy →
   Transcript upload**. The probe sends only a fixed built-in English sentence,
   not microphone audio or session history.

For an actual cloud session, select Qwen realtime cloud for ASR and/or Qwen-MT
cloud for Translation. Cloud ASR also requires explicit **Audio upload**
permission. Cloud translation requires **Transcript upload** permission. Hybrid
routes remain supported, so either provider can stay local independently.

## Interpreting failures

- `HTTP 401`: the API key is invalid, inactive, or belongs to another region.
- `HTTP 403`: the key and workspace may not belong to the same Singapore
  workspace, or realtime ASR access is not enabled for that workspace.
- timeout/network error: verify internet access and retry; no automatic local
  transcript or audio upload occurs during the connection probe.

The connection test is a configuration check, not a latency benchmark. The
EN/ZH/JA/KO p50/p95 cloud benchmark remains `PENDING_CREDENTIALS` until it is
run with an authorized account and real prerecorded fixtures.


## Region and free quota

Both cloud adapters support the Singapore (`ap-southeast-1`) and Beijing
(`cn-beijing`) Model Studio regions; the default is Singapore. Alibaba Cloud
Model Studio grants new-user free quota only in the Beijing region, so a
mainland account that wants to use it must set `ECHOLINGO_QWEN_REGION=beijing`
(the value is applied to both the ASR and translation route and to the
Settings connection probe) and use an API key and workspace ID created in that
region. Free quota is per model, expires after 90 days, and, when the
"stop when exhausted" switch is on, the service answers HTTP 403
`AllocationQuota.FreeTierOnly` once it is used up; EchoLingo reports that as a
terminal authentication error for the session.
