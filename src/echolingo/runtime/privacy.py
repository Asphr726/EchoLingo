from __future__ import annotations

from dataclasses import dataclass

from ..errors import PolicyDeniedError


@dataclass(slots=True, frozen=True)
class PrivacyPolicy:
    audio_upload_allowed: bool = False
    transcript_upload_allowed: bool = False

    def require_audio_upload(self) -> None:
        if not self.audio_upload_allowed:
            raise PolicyDeniedError("Cloud ASR requires explicit audio upload consent")

    def require_transcript_upload(self) -> None:
        if not self.transcript_upload_allowed:
            raise PolicyDeniedError(
                "Cloud translation requires explicit transcript upload consent"
            )

