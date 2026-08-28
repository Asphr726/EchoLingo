from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from ...models import (
    BackendDescriptor,
    BackendLocality,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranslationKind,
    TranslationRequest,
)


class LocalHyMtBackend:
    """Hy-MT adapter for an isolated localhost OpenAI-compatible model server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8010/v1",
        model: str = "tencent/Hy-MT2-1.8B",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("ECHOLINGO_LOCAL_MT_API_KEY", "local")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=60.0)
        self.glossary: tuple[GlossaryTerm, ...] = ()
        self.descriptor = BackendDescriptor(
            "hymt_local", model, BackendLocality.LOCAL, ("en", "zh", "ja", "ko")
        )

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    def build_prompt(self, request: TranslationRequest) -> str:
        terms = request.terms or self.glossary
        lines = [
            f"Translate from {request.source_lang} to {request.target_lang}.",
            "Return only the translation.",
        ]
        if request.domain:
            lines.append(f"Domain: {request.domain}")
        if terms:
            lines.append(
                "Terminology: "
                + "; ".join(f"{item.source} => {item.target}" for item in terms)
            )
        if request.context:
            lines.append("Previous context:")
            lines.extend(
                f"{item.source} => {item.target}" for item in request.context[-5:]
            )
        lines.extend(["Source:", request.source_text])
        return "\n".join(lines)

    def _payload(self, request: TranslationRequest, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": self.build_prompt(request)}],
            "stream": stream,
        }

    def _event(
        self,
        request: TranslationRequest,
        kind: TranslationKind,
        text: str,
        revision: int,
        started: int,
    ) -> CanonicalTranslationEvent:
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=revision,
            source_revision_id=request.source_revision_id,
            kind=kind,
            text=text,
            provider="hymt_local",
            model=self.model,
            locality=BackendLocality.LOCAL,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text=text if kind == TranslationKind.PARTIAL else "",
            committed_text=text if kind == TranslationKind.FINAL else "",
            total_latency_ms=(time.monotonic_ns() - started) / 1_000_000.0,
        )

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            started = time.monotonic_ns()
            text = ""
            revision = 0
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._payload(request, True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    value = json.loads(raw)
                    choices = value.get("choices") or []
                    if not choices:
                        continue
                    delta = str(choices[0].get("delta", {}).get("content") or "")
                    if not delta:
                        continue
                    text += delta
                    revision += 1
                    yield self._event(
                        request, TranslationKind.PARTIAL, text, revision, started
                    )
            yield self._event(
                request, TranslationKind.FINAL, text, revision + 1, started
            )

        return generate()

    async def retranslate_window(
        self, request: TranslationRequest
    ) -> CanonicalTranslationEvent:
        started = time.monotonic_ns()
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._payload(request, False),
        )
        response.raise_for_status()
        text = str(response.json()["choices"][0]["message"]["content"])
        return self._event(request, TranslationKind.FINAL, text, 1, started)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

