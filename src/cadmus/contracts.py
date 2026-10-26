"""Typed, versioned contracts shared by every module."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class InputMode(StrEnum):
    JSON_RECORDS = "json_records"
    DIRECT_FILES = "direct_files"


class TaskKind(StrEnum):
    SINGLE_LABEL = "single_label"
    MULTILABEL = "multilabel"


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    text: str
    page: int | None = None


class MediaRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    sha256: str
    mime_type: str
    width: int
    height: int
    page: int | None = None


class EvidenceRecord(BaseModel):
    """Normalized, model-visible evidence. Never contains ground-truth labels."""

    model_config = ConfigDict(frozen=True)
    schema_version: int = SCHEMA_VERSION
    document_id: str
    family_id: str
    input_mode: InputMode
    source_hash: str
    pipeline_id: str
    text: list[TextBlock] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    media: list[MediaRef] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def canonical(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    def evidence_hash(self) -> str:
        return sha256_text(self.canonical())

    def has_evidence(self) -> bool:
        return bool(self.media) or bool(self.fields) or any(b.text.strip() for b in self.text)


class PredictionStatus(StrEnum):
    SUCCESS = "success"
    ABSTAINED = "abstained"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_ERROR = "provider_error"
    EXTRACTION_ERROR = "extraction_error"
    NO_EVIDENCE = "no_evidence"
    CANCELLED = "cancelled"


CACHEABLE_STATUSES = {PredictionStatus.SUCCESS, PredictionStatus.ABSTAINED}
INFRA_STATUSES = {PredictionStatus.PROVIDER_ERROR, PredictionStatus.CANCELLED}


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class Prediction(BaseModel):
    document_id: str
    status: PredictionStatus
    labels: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None  # uncalibrated model-reported metadata only
    bundle_id: str
    evidence_hash: str | None = None
    repeat_index: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int | None = None
    error_code: str | None = None
    cached: bool = False


class FewShotExample(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    family_id: str
    evidence_hash: str
    labels: list[str]


class PromptBundle(BaseModel):
    bundle_id: str
    compatibility_key: str
    guidance: str
    examples: list[FewShotExample] = Field(default_factory=list)
    parent_id: str | None = None
    origin: Literal["baseline", "candidate"] = "candidate"
    created_at: str = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def compute_id(compatibility_key: str, guidance: str, examples: list[FewShotExample]) -> str:
        body = {"c": compatibility_key, "g": guidance, "e": [e.model_dump() for e in examples]}
        return "b-" + sha256_text(canonical_json(body))[:20]

    @classmethod
    def create(cls, *, compatibility_key: str, guidance: str, examples: list[FewShotExample] | None = None,
               parent_id: str | None = None, origin: Literal["baseline", "candidate"] = "candidate",
               provenance: dict[str, Any] | None = None) -> PromptBundle:
        ex = list(examples or [])
        return cls(bundle_id=cls.compute_id(compatibility_key, guidance, ex), compatibility_key=compatibility_key,
                   guidance=guidance, examples=ex, parent_id=parent_id, origin=origin,
                   provenance=provenance or {})


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


class EvaluationReport(BaseModel):
    bundle_id: str
    split: str
    expected: int
    completed: int
    status_counts: dict[str, int]
    coverage: float
    invalid_rate: float
    infra_failure_rate: float
    accuracy: float | None = None
    answered_accuracy: float | None = None
    exact_match: float | None = None
    macro_f1: float
    micro_f1: float
    per_class: dict[str, ClassMetrics]
    input_tokens: int = 0
    output_tokens: int = 0
    unknown_usage: int = 0
    cost_usd: float | None = None
    latency_ms_p50: int | None = None
    latency_ms_p95: int | None = None

    def metric(self, name: str) -> float:
        value = {"macro_f1": self.macro_f1, "micro_f1": self.micro_f1,
                 "accuracy": self.accuracy if self.accuracy is not None else self.exact_match}[name]
        return float(value or 0.0)


class GateResult(BaseModel):
    name: str
    passed: bool
    observed: float | int | str | None = None
    threshold: float | int | str | None = None
    reason_code: str
    detail: str | None = None


DecisionOutcome = Literal[
    "promoted", "rejected", "insufficient_evidence", "budget_exhausted", "stale_incumbent", "failed", "cancelled"
]


class PromotionDecision(BaseModel):
    run_id: str
    outcome: DecisionOutcome
    reason_codes: list[str]
    incumbent_bundle_id: str | None = None
    finalist_bundle_id: str | None = None
    gates: list[GateResult] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    policy_version: str = ""
    created_at: str = Field(default_factory=utc_now)
