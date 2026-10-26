"""Stable, sanitized error types. Messages must never contain customer content or secrets."""

from __future__ import annotations


class AgAgentError(Exception):
    code = "error"
    exit_code = 1

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class ConfigError(AgAgentError):
    code = "config_invalid"
    exit_code = 2


class AccessDenied(AgAgentError):
    code = "access_denied"
    exit_code = 2


class NotFound(AgAgentError):
    code = "not_found"
    exit_code = 2


class InputError(AgAgentError):
    code = "input_invalid"
    exit_code = 3


class ExtractionError(AgAgentError):
    code = "extraction_error"
    exit_code = 3


class ProviderError(AgAgentError):
    code = "provider_error"

    def __init__(self, message: str, code: str | None = None, *, transient: bool = False,
                 completion_unknown: bool = False, request_id: str | None = None) -> None:
        super().__init__(message, code)
        self.transient = transient
        self.completion_unknown = completion_unknown
        self.request_id = request_id


class BudgetExhausted(AgAgentError):
    code = "budget_exhausted"


class AuditFailure(AgAgentError):
    code = "audit_write_failed"
    exit_code = 4


class RunCancelled(AgAgentError):
    code = "cancelled"
    exit_code = 130
