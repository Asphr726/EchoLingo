from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from ...errors import AuthenticationError, PolicyDeniedError, RateLimitError
from ...models import (
    BackendDescriptor,
    BackendLocality,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranslationKind,
    TranslationRequest,
)


_REGION_HOSTS = {
    "singapore": "ap-southeast-1.maas.aliyuncs.com",
    "beijing": "cn-beijing.maas.aliyuncs.com",
}

_LANGUAGE_NAMES = {
    "auto": "auto",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


class CloudQwenMtBackend:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        workspace_id: str | None = None,
        region: str = "singapore",
        interactive_model: str = "qwen-mt-flash",
        quality_model: str = "qwen-mt-plus",
        transcript_upload_allowed: bool = False,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if region not in _REGION_HOSTS:
            raise ValueError("Qwen-MT region must be singapore or beijing")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.workspace_id = workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID")
        self.region = region
        self.interactive_model = interactive_model
        self.quality_model = quality_model
        self.transcript_upload_allowed = transcript_upload_allowed
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self.glossary: tuple[GlossaryTerm, ...] = ()
        self.descriptor = BackendDescriptor(
            "qwen_cloud",
            interactive_model,
            BackendLocality.CLOUD,
            transcript_upload_required=True,
        )

    @property
    def base_url(self) -> str:
        host = _REGION_HOSTS[self.region]
        return f"https://{self.workspace_id}.{host}/compatible-mode/v1"

    def _authorize(self) -> None:
        if not self.transcript_upload_allowed:
            raise PolicyDeniedError(
                "Cloud translation requires explicit transcript upload consent"
            )
        if not self.api_key or not self.workspace_id:
            raise AuthenticationError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID are required"
            )

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    def build_payload(
        self, request: TranslationRequest, *, model: str, stream: bool
    ) -> dict:
        terms = request.terms or self.glossary
        memory = [
            {"source": item.source, "target": item.target}
            for item in request.translation_memory
        ]
        memory.extend(
            {"source": item.source, "target": item.target}
            for item in request.context
            if item.target
        )
        options: dict[str, object] = {
            "source_lang": _LANGUAGE_NAMES.get(request.source_lang, request.source_lang),
            "target_lang": _LANGUAGE_NAMES.get(request.target_lang, request.target_lang),
        }
        if terms:
            options["terms"] = [
                {"source": item.source, "target": item.target} for item in terms
            ]
        if memory:
            options["tm_list"] = memory[-10:]
        if request.domain:
            options["domains"] = request.domain
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.source_text}],
            "translation_options": options,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _raise_status(response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise AuthenticationError("Cloud Qwen-MT authentication failed")
        if response.status_code == 429:
            raise RateLimitError("Cloud Qwen-MT rate limit exceeded")
        response.raise_for_status()

    def _event(
        self,
        request: TranslationRequest,
        kind: TranslationKind,
        text: str,
        revision: int,
        model: str,
        started_ns: int,
        first_delta_ms: float | None = None,
        usage: dict | None = None,
    ) -> CanonicalTranslationEvent:
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=revision,
            source_revision_id=request.source_revision_id,
            kind=kind,
            text=text,
            provider="qwen_cloud",
            model=model,
            locality=BackendLocality.CLOUD,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text=text if kind == TranslationKind.PARTIAL else "",
            committed_text=text if kind == TranslationKind.FINAL else "",
            first_delta_latency_ms=first_delta_ms,
            total_latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
        )

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            self._authorize()
            started = time.monotonic_ns()
            text = ""
            revision = 0
            first_delta = None
            usage: dict | None = None
            payload = self.build_payload(
                request, model=self.interactive_model, stream=True
            )
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                self._raise_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    usage = chunk.get("usage") or usage
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    content = str(choices[0].get("delta", {}).get("content") or "")
                    if not content:
                        continue
                    if self.interactive_model in {"qwen-mt-plus", "qwen-mt-turbo"}:
                        text = content
                    else:
                        text += content
                    revision += 1
                    if first_delta is None:
                        first_delta = (time.monotonic_ns() - started) / 1_000_000.0
                    yield self._event(
                        request,
                        TranslationKind.PARTIAL,
                        text,
                        revision,
                        self.interactive_model,
                        started,
                        first_delta,
                        usage,
                    )
            revision += 1
            yield self._event(
                request,
                TranslationKind.FINAL,
                text,
                revision,
                self.interactive_model,
                started,
                first_delta,
                usage,
            )

        return generate()

    async def retranslate_window(
        self, request: TranslationRequest
    ) -> CanonicalTranslationEvent:
        self._authorize()
        started = time.monotonic_ns()
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self.build_payload(request, model=self.quality_model, stream=False),
        )
        self._raise_status(response)
        value = response.json()
        text = str(value["choices"][0]["message"]["content"])
        return self._event(
            request,
            TranslationKind.FINAL,
            text,
            1,
            self.quality_model,
            started,
            usage=value.get("usage"),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
