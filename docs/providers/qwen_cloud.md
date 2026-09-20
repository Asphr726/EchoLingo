# Qwen Cloud (Alibaba Cloud Model Studio / DashScope)

Providers: `qwen_cloud` for realtime recognition (`qwen3-asr-flash-realtime`)
and `qwen_cloud` for translation (`qwen-mt-flash` streaming, `qwen-mt-plus`
window retranslation). Credential group `dashscope`.

## Keys, regions and free quota

Model Studio has two independent regions with separate consoles, keys and
quotas:

| Region setting | Console | Classic host (key only) | Workspace host |
| --- | --- | --- | --- |
| `beijing` | bailian.console.aliyun.com | `dashscope.aliyuncs.com` | `<workspace>.cn-beijing.maas.aliyuncs.com` |
| `singapore` | modelstudio.console.alibabacloud.com | `dashscope-intl.aliyuncs.com` | `<workspace>.ap-southeast-1.maas.aliyuncs.com` |

- New accounts receive free quota per model for 90 days in the **Beijing**
  region only. Singapore has no free quota. When the "stop when exhausted"
  switch is on, an exhausted quota answers HTTP 403
  `AllocationQuota.FreeTierOnly`.
- A key is valid in one region only. The most common failure is a Beijing key
  with the region left on Singapore: HTTP 401 from `dashscope-intl`. Set
  **Region** on the Qwen Cloud card (or `ECHOLINGO_QWEN_REGION=beijing`).
- The **workspace ID is optional**. Leave it empty to use the key's default
  workspace through the classic host; fill it to pin a specific workspace.

## Setup

1. Create a key in the console of the region you want to use.
2. Settings → Cloud → Qwen Cloud: paste the key, optionally the workspace ID,
   choose the Region, **Save**, then **Test**. The result shows the region and
   host that were used and whether the call was workspace-scoped.
3. Enable **Audio upload** for cloud recognition and/or **Transcript upload**
   for Qwen-MT.

## What leaves the machine

Cloud recognition uploads the enhanced 16 kHz mono stream after the local
audio front end; nothing else. Qwen-MT receives the source sentence, glossary
terms and up to ten previous source/target pairs as translation memory.

## Errors

| Message | Cause |
| --- | --- |
| HTTP 401 "…sent to the Singapore region…" | key from the other region, inactive key |
| HTTP 403 workspace | key does not belong to that workspace, or the model is not enabled for it |
| HTTP 403 `AllocationQuota.FreeTierOnly` | free quota exhausted |
| timeout to `dashscope.aliyuncs.com` | network; Beijing is reachable from mainland China without a proxy |
