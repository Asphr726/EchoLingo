"""AI assistant for History: structured notes, session titles and context terms.

The assistant runs inside the inference sidecar and talks to a user-chosen
OpenAI-compatible chat endpoint (docs/adr/0006). Transcripts and attachments
leave the machine only when the request carries explicit consent
(``consent.transcript_upload_allowed``); keys come from the sidecar
environment that the desktop shell fills from the OS secure store.
"""

from __future__ import annotations

__all__ = ["PROMPT_VERSION"]

from .prompts import PROMPT_VERSION
