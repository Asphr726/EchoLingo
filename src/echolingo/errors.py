from __future__ import annotations


class EchoLingoError(Exception):
    """Base error for errors safe to surface at an application boundary."""


class ConfigurationError(EchoLingoError):
    pass


class BackendError(EchoLingoError):
    code = "backend_error"
    recoverable = False

    @property
    def error_code(self) -> str:
        """Stable machine-readable code for UI/event payloads."""
        return self.code


class BackendUnavailableError(BackendError):
    code = "backend_unavailable"
    recoverable = True


class AlignmentUnavailableError(BackendUnavailableError):
    code = "alignment_unavailable"


class AuthenticationError(BackendError):
    code = "authentication_failed"


class RateLimitError(BackendError):
    code = "rate_limited"
    recoverable = True


class NetworkError(BackendError):
    code = "network_error"
    recoverable = True


class ProviderTimeoutError(NetworkError):
    code = "provider_timeout"


class PolicyDeniedError(BackendError):
    code = "privacy_policy_denied"
