from __future__ import annotations

import asyncio
import json
import logging
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
from ...translation.policy import TranslationRequestError

logger = logging.getLogger(__name__)

# Re-exported for existing imports; the helpers live in ``_text``.
from ._text import LANGUAGE_NAMES, language_name, repeated_tail, strip_instruction_echo  # noqa: E402

__all__ = [
    "LANGUAGE_NAMES",
    "LocalHyMtBackend",
    "language_name",
    "repeated_tail",
    "strip_instruction_echo",
]


class LocalHyMtBackend:
    """Hy-MT adapter for an isolated localhost OpenAI-compatible model server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8010/v1",
        model: str = "tencent/Hy-MT2-1.8B",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        partial_timeout_s: float = 8.0,
        commit_timeout_s: float = 20.0,
        max_output_tokens: int = 400,
        temperature: float = 0.1,
        top_p: float = 0.6,
        top_k: int = 20,
        repeat_penalty: float = 1.05,
        background_spans: int = 2,
        background_max_chars: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("ECHOLINGO_LOCAL_MT_API_KEY", "local")
        self._owns_client = client is None
        # Local llama-server: never route through an HTTPS_PROXY from the environment.
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(commit_timeout_s, connect=5.0), trust_env=False
        )
        self.partial_timeout_s = partial_timeout_s
        self.commit_timeout_s = commit_timeout_s
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.background_spans = background_spans
        self.background_max_chars = background_max_chars
        self.glossary: tuple[GlossaryTerm, ...] = ()
        self.descriptor = BackendDescriptor(
            "hymt_local", model, BackendLocality.LOCAL, ("en", "zh", "ja", "ko")
        )
        self._runtime_checked = False

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------

    def _background(self, request: TranslationRequest) -> str:
        """Previous source sentences only: target text in the prompt gets copied."""
        spans: list[str] = []
        total = 0
        for item in reversed(request.context[-self.background_spans :]):
            source = item.source.strip()
            if not source:
                continue
            if total + len(source) > self.background_max_chars:
                break
            spans.append(source)
            total += len(source)
        return "\n".join(reversed(spans))

    def build_prompt(self, request: TranslationRequest) -> str:
        chinese_prompt = request.target_lang.lower().startswith("zh")
        target = language_name(request.target_lang, chinese_prompt=chinese_prompt)
        terms = request.terms or self.glossary
        background = self._background(request)
        source = request.source_text.strip()
        parts: list[str] = []
        if terms:
            if chinese_prompt:
                parts.append(
                    "参考下面的翻译：\n"
                    + "\n".join(f"{item.source} 翻译成 {item.target}" for item in terms)
                )
            else:
                parts.append(
                    "Reference the following translations:\n"
                    + "\n".join(f"{item.source} translates to {item.target}" for item in terms)
                )
        if background:
            if chinese_prompt:
                parts.append(
                    f"【背景信息】\n{background}\n\n"
                    f"请结合背景信息将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释。\n\n"
                    f"【待翻译文本】\n{source}"
                )
            else:
                parts.append(
                    f"[Background Information]\n{background}\n\n"
                    f"Please translate the following text into {target}, taking the provided "
                    "background information into consideration. Note that you should only output "
                    "the translated result without any additional explanation.\n\n"
                    f"[Source Text]\n{source}"
                )
        elif chinese_prompt:
            parts.append(f"将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source}")
        else:
            parts.append(
                f"Translate the following text into {target}. Note that you should only output "
                f"the translated result without any additional explanation:\n\n{source}"
            )
        return "\n\n".join(parts)

    def _max_tokens(self, request: TranslationRequest) -> int:
        chars = len(request.source_text)
        if request.source_committed:
            return min(self.max_output_tokens, max(24, 3 * chars + 16))
        return min(320, max(24, 2 * chars + 16))

    def _payload(self, request: TranslationRequest, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.build_prompt(request)}],
            "stream": stream,
            "max_tokens": self._max_tokens(request),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": 64,
            "cache_prompt": True,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _timeout_s(self, request: TranslationRequest) -> float:
        default = self.commit_timeout_s if request.source_committed else self.partial_timeout_s
        if request.timeout_s is not None:
            return max(0.5, min(default, request.timeout_s))
        return default

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _event(
        self,
        request: TranslationRequest,
        kind: TranslationKind,
        text: str,
        revision: int,
        started: int,
        *,
        first_delta_ms: float | None = None,
        truncated: bool = False,
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
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
            first_delta_latency_ms=first_delta_ms,
            total_latency_ms=(time.monotonic_ns() - started) / 1_000_000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source_committed=request.source_committed,
            truncated=truncated,
            finish_reason=finish_reason,
        )

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            started = time.monotonic_ns()
            text = ""
            revision = 0
            first_delta_ms: float | None = None
            finish_reason: str | None = None
            truncated = False
            prompt_tokens = completion_tokens = None
            source_chars = max(1, len(request.source_text))
            limit_chars = 2.5 * source_chars + 32
            timeout_s = self._timeout_s(request)
            try:
                async with asyncio.timeout(timeout_s):
                    async with self.client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=self._payload(request, True),
                    ) as response:
                        if response.status_code >= 400:
                            body = (await response.aread()).decode("utf-8", "replace")[:300]
                            code = "context_overflow" if "context" in body else f"http_{response.status_code}"
                            raise TranslationRequestError(
                                code,
                                f"local translation server returned {response.status_code}: {body}",
                                retry_without_context=response.status_code < 500 and bool(request.context),
                            )
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw or raw == "[DONE]":
                                continue
                            value = json.loads(raw)
                            usage = value.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                            choices = value.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])
                            delta = str((choice.get("delta") or {}).get("content") or "")
                            if not delta:
                                continue
                            if first_delta_ms is None:
                                first_delta_ms = (time.monotonic_ns() - started) / 1_000_000.0
                            text = strip_instruction_echo(text + delta)
                            unit = repeated_tail(text)
                            if unit is not None:
                                text = text[: len(text.rstrip()) - len(unit) * 3].rstrip()
                                finish_reason = "repetition"
                                truncated = True
                                break
                            if len(text) > limit_chars:
                                finish_reason = "length"
                                truncated = True
                                break
                            revision += 1
                            yield self._event(
                                request,
                                TranslationKind.PARTIAL,
                                text,
                                revision,
                                started,
                                first_delta_ms=first_delta_ms,
                            )
            except TimeoutError:
                finish_reason = "timeout"
                truncated = True
                logger.warning(
                    "local translation timed out after %.1f s (%d chars, committed=%s)",
                    timeout_s,
                    source_chars,
                    request.source_committed,
                )
            if finish_reason == "length":
                truncated = True
            if truncated and finish_reason in {"repetition", "length"}:
                logger.warning(
                    "local translation output truncated (%s) for %d source chars",
                    finish_reason,
                    source_chars,
                )
            yield self._event(
                request,
                TranslationKind.FINAL,
                text.strip(),
                revision + 1,
                started,
                first_delta_ms=first_delta_ms,
                truncated=truncated,
                finish_reason=finish_reason or "stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
            timeout=self._timeout_s(request),
        )
        response.raise_for_status()
        body = response.json()
        text = strip_instruction_echo(str(body["choices"][0]["message"]["content"]))
        usage = body.get("usage") or {}
        return self._event(
            request,
            TranslationKind.FINAL,
            text.strip(),
            1,
            started,
            finish_reason=str(body["choices"][0].get("finish_reason") or "stop"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def describe_runtime(self) -> dict:
        """Confirm the server applies the model's chat template (logged once)."""
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        info: dict = {}
        try:
            props = await self.client.get(
                f"{root}/props", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5.0
            )
            props.raise_for_status()
            data = props.json()
            template = str(data.get("chat_template") or "")
            info["chat_template_ok"] = "hy_User" in template
            settings = data.get("default_generation_settings") or {}
            info["n_ctx"] = settings.get("n_ctx")
            if not info["chat_template_ok"]:
                logger.warning("local translation server is not using the Hy-MT chat template")
        except Exception as error:  # pragma: no cover - diagnostics only
            logger.info("local translation runtime probe unavailable: %s", error)
        self._runtime_checked = True
        return info

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
