"""Qwen-MT over the DashScope OpenAI-compatible endpoint."""

from __future__ import annotations

import os

import httpx

from ...errors import AuthenticationError, RateLimitError
from ...models import TranslationRequest
from ...translation.policy import TranslationRequestError
from .. import dashscope
from .chat_base import OpenAiCompatibleChatTranslation

_LANGUAGE_NAMES = {
    "auto": "auto",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


class CloudQwenMtBackend(OpenAiCompatibleChatTranslation):
    provider_id = "qwen_cloud"
    display_name = "Qwen-MT"

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
        if region not in dashscope.REGIONS:
            raise ValueError("Qwen-MT region must be singapore or beijing")
        self.workspace_id = (
            (workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip() or None
        )
        self.region = region
        self.interactive_model = interactive_model
        self.quality_model = quality_model
        super().__init__(
            base_url=dashscope.resolve_endpoint(region, self.workspace_id).compatible_base_url,
            model=interactive_model,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            transcript_upload_allowed=transcript_upload_allowed,
            timeout_s=timeout_s,
            client=client,
        )

    @property
    def dashscope_endpoint(self) -> dashscope.DashScopeEndpoint:
        return dashscope.resolve_endpoint(self.region, self.workspace_id)

    def describe_endpoint(self) -> dict[str, object]:
        return self.dashscope_endpoint.describe()

    def missing_credentials_message(self) -> str:
        return "DASHSCOPE_API_KEY is required"

    def authorize(self) -> None:
        if not self.transcript_upload_allowed:
            from ...errors import PolicyDeniedError

            raise PolicyDeniedError("Cloud translation requires explicit transcript upload consent")
        if not self.api_key:
            raise AuthenticationError(self.missing_credentials_message())

    def model_for(self, request: TranslationRequest) -> str:
        return self.interactive_model

    def window_model_for(self, request: TranslationRequest) -> str:
        return self.quality_model

    def accumulate(self, text: str, delta: str) -> str:
        # qwen-mt-plus/turbo stream the whole translation on every delta.
        if self.interactive_model in {"qwen-mt-plus", "qwen-mt-turbo"}:
            return delta
        return text + delta

    def guard(self, text: str, request: TranslationRequest) -> tuple[str, str | None]:
        # Qwen-MT is a dedicated translation model; no repetition guard needed.
        return text, None

    def build_payload(self, request: TranslationRequest, *, model: str, stream: bool) -> dict:
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

    def _raise_status(self, response: httpx.Response) -> None:
        endpoint = self.dashscope_endpoint
        if response.status_code == 401:
            raise AuthenticationError(dashscope.unauthorized_message("Qwen-MT", endpoint))
        if response.status_code == 403:
            raise AuthenticationError(dashscope.forbidden_message("Qwen-MT", endpoint))
        if response.status_code == 429:
            raise RateLimitError("Cloud Qwen-MT rate limit exceeded")
        if response.status_code >= 400:
            raise TranslationRequestError(
                f"http_{response.status_code}",
                f"Qwen-MT returned HTTP {response.status_code}",
            )

    async def raise_status(self, response: httpx.Response, request: TranslationRequest) -> None:
        self._raise_status(response)
