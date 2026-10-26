"""Stable, sanitized error types. Messages must never contain customer content or secrets."""

from __future__ import annotations


class CadmusError(Exception):
    code = "error"
    exit_code = 1

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class ConfigError(CadmusError):
    code = "config_invalid"
    exit_code = 2


class AccessDenied(CadmusError):
    code = "access_denied"
    exit_code = 2


class NotFound(CadmusError):
    code = "not_found"
    exit_code = 2


class InputError(CadmusError):
    code = "input_invalid"
    exit_code = 3


class ExtractionError(CadmusError):
    code = "extraction_error"
    exit_code = 3


class ProviderError(CadmusError):
    code = "provider_error"

    def __init__(self, message: str, code: str | None = None, *, transient: bool = False,
                 completion_unknown: bool = False, request_id: str | None = None) -> None:
        super().__init__(message, code)
        self.transient = transient
        self.completion_unknown = completion_unknown
        self.request_id = request_id


class BudgetExhausted(CadmusError):
    code = "budget_exhausted"


class AuditFailure(CadmusError):
    code = "audit_write_failed"
    exit_code = 4


class RunCancelled(CadmusError):
    code = "cancelled"
    exit_code = 130
