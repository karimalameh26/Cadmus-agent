"""Validated TOML configuration. Credentials are referenced by environment variable name only."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ag_agent.contracts import InputMode, TaskKind, canonical_json, sha256_text
from ag_agent.errors import ConfigError

PIPELINE_VERSION = "1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    kind: TaskKind = TaskKind.SINGLE_LABEL
    labels: list[str] = Field(min_length=1)
    critical_labels: list[str] = Field(default_factory=list)
    allow_abstain: bool = True
    instructions: str = ""
    baseline_guidance: str = "Classify the document using the evidence provided."

    @model_validator(mode="after")
    def _check(self) -> TaskConfig:
        if len(set(self.labels)) != len(self.labels) or any(not lbl.strip() for lbl in self.labels):
            raise ValueError("task.labels must be unique, non-empty strings")
        unknown = set(self.critical_labels) - set(self.labels)
        if unknown:
            raise ValueError(f"task.critical_labels not in taxonomy: {sorted(unknown)}")
        return self

    def identity(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind.value, "labels": self.labels,
                "allow_abstain": self.allow_abstain, "instructions": sha256_text(self.instructions)}


class JsonInputConfig(_Strict):
    shape: Literal["array", "keyed", "jsonl"]
    records_path: str | None = None
    id_field: str | None = None
    family_field: str | None = None
    text_fields: list[str] = Field(default_factory=list)
    description_fields: list[str] = Field(default_factory=list)
    metadata_fields: list[str] = Field(default_factory=list)
    feature_fields: list[str] = Field(default_factory=list)
    label_field: str | None = None
    expose_id: bool = False
    max_field_chars: int = Field(default=20_000, gt=0)

    @model_validator(mode="after")
    def _check(self) -> JsonInputConfig:
        if self.shape in ("array", "jsonl") and not self.id_field:
            raise ValueError("input.json.id_field is required for array/jsonl shapes")
        if self.shape == "keyed" and self.id_field:
            raise ValueError("input.json.id_field must be omitted for keyed shape (the key is the id)")
        visible = self.text_fields + self.description_fields + self.metadata_fields + self.feature_fields
        if not visible:
            raise ValueError("input.json must map at least one model-visible field")
        if len(set(visible)) != len(visible):
            raise ValueError("input.json visible fields must not be mapped twice")
        if self.label_field and self.label_field in visible:
            raise ValueError("input.json.label_field must not be a model-visible field")
        if self.label_field and self.label_field in (self.id_field, self.family_field):
            raise ValueError("input.json.label_field must not be the id or family field")
        return self


class FilesInputConfig(_Strict):
    family_from: Literal["stem", "document"] = "stem"
    expose_filename: bool = False


class InputConfig(_Strict):
    mode: InputMode
    roots: list[Path] = Field(default_factory=list)
    json_records: JsonInputConfig | None = Field(default=None, alias="json")
    files: FilesInputConfig = Field(default_factory=FilesInputConfig)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _check(self) -> InputConfig:
        if self.mode == InputMode.JSON_RECORDS and self.json_records is None:
            raise ValueError("input.json is required when input.mode = 'json_records'")
        return self


class DocumentConfig(_Strict):
    max_file_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_pages: int = Field(default=10, gt=0)
    oversize_policy: Literal["truncate", "reject"] = "truncate"
    max_text_chars: int = Field(default=20_000, gt=0)
    image_max_side: int = Field(default=1536, ge=64)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    jpeg_quality: int = Field(default=90, ge=50, le=100)
    pdf_min_text_chars: int = Field(default=20, ge=0)
    pdf_scan_strategy: Literal["render", "fail"] = "render"
    pdf_render_scale: float = Field(default=1.5, gt=0, le=4)
    max_request_chars: int = Field(default=200_000, gt=0)


class PricingConfig(_Strict):
    input_per_1k: float | None = Field(default=None, ge=0)
    output_per_1k: float | None = Field(default=None, ge=0)
    image_token_estimate: int = Field(default=800, ge=0)

    @property
    def known(self) -> bool:
        return self.input_per_1k is not None and self.output_per_1k is not None


class ProviderConfig(_Strict):
    kind: Literal["fake", "openai_compatible"] = "fake"
    model: str = "fake-classifier-v1"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    auth_header: Literal["bearer", "api-key"] = "bearer"
    structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    temperature: float | None = 0.0
    max_output_tokens: int = Field(default=400, gt=0)
    image_detail: Literal["auto", "low", "high"] = "auto"
    supports_images: bool = True
    timeout_s: float = Field(default=60, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_s: float = Field(default=1.0, ge=0)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @model_validator(mode="after")
    def _check(self) -> ProviderConfig:
        if self.kind == "openai_compatible":
            if not self.base_url:
                raise ValueError("provider.base_url is required for openai_compatible")
            local = self.base_url.startswith(("http://localhost", "http://127.0.0.1"))
            if not (self.base_url.startswith("https://") or local):
                raise ValueError("provider.base_url must use https (plain http only for localhost)")
        return self

    def identity(self) -> dict[str, object]:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens, "image_detail": self.image_detail,
                "structured_output": self.structured_output}


class SplitRatios(_Strict):
    development: float = Field(default=0.5, gt=0)
    selection: float = Field(default=0.25, gt=0)
    gate: float = Field(default=0.25, gt=0)

    @model_validator(mode="after")
    def _check(self) -> SplitRatios:
        if abs(self.development + self.selection + self.gate - 1.0) > 1e-6:
            raise ValueError("optimizer.splits must sum to 1.0")
        return self


class OptimizerConfig(_Strict):
    proposer: Literal["model", "heuristic"] = "model"
    provider: ProviderConfig | None = None
    candidates_per_round: int = Field(default=4, ge=1, le=16)
    max_rounds: int = Field(default=5, ge=1, le=50)
    patience: int = Field(default=2, ge=1)
    min_dev_gain: float = Field(default=0.0, ge=0)
    shortlist_size: int = Field(default=2, ge=1, le=8)
    max_examples: int = Field(default=4, ge=0, le=16)
    max_guidance_chars: int = Field(default=4000, gt=0)
    max_failures_shown: int = Field(default=12, ge=1)
    max_requests: int = Field(default=2000, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_elapsed_s: float | None = Field(default=None, gt=0)
    repeats: int = Field(default=1, ge=1, le=10)
    seed: int = 7
    splits: SplitRatios = Field(default_factory=SplitRatios)


class PolicyConfig(_Strict):
    version: str = "1"
    approved_for_live: bool = False
    primary_metric: Literal["macro_f1", "micro_f1", "accuracy"] = "macro_f1"
    min_gain: float = Field(default=0.02, ge=0)
    confidence: float = Field(default=0.9, gt=0.5, lt=1)
    bootstrap_resamples: int = Field(default=1000, ge=100)
    min_gate_groups: int = Field(default=10, ge=1)
    min_class_support: int = Field(default=2, ge=1)
    critical_tolerance: float = Field(default=0.0, ge=0)
    min_coverage: float = Field(default=0.9, ge=0, le=1)
    max_invalid_rate: float = Field(default=0.05, ge=0, le=1)
    max_infra_failure_rate: float = Field(default=0.0, ge=0, le=1)
    absolute_floor: float = Field(default=0.5, ge=0, le=1)
    max_cost_ratio: float | None = Field(default=None, gt=0)


class MonitorConfig(_Strict):
    window_hours: float = Field(default=168, gt=0)
    min_samples: int = Field(default=30, ge=1)
    max_drop: float = Field(default=0.05, ge=0)
    max_invalid_rate: float = Field(default=0.1, ge=0, le=1)
    cooldown_hours: float = Field(default=24, ge=0)
    auto_rollback: bool = True


class LoggingConfig(_Strict):
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    jsonl_max_bytes: int = Field(default=10 * 1024 * 1024, gt=1024)
    jsonl_backups: int = Field(default=5, ge=0)
    retention_days: int | None = Field(default=None, gt=0)


class AppConfig(_Strict):
    client_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    data_dir: Path = Path(".ag-agent")
    reviewers: list[str] = Field(default_factory=list)
    task: TaskConfig
    input: InputConfig
    documents: DocumentConfig = Field(default_factory=DocumentConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("reviewers")
    @classmethod
    def _reviewers(cls, v: list[str]) -> list[str]:
        reserved = {"optimizer", "system", "model", "classifier"}
        if reserved & {r.lower() for r in v}:
            raise ValueError(f"reviewers must not include reserved identities {sorted(reserved)}")
        return v

    @model_validator(mode="after")
    def _check(self) -> AppConfig:
        if self.optimizer.max_cost_usd is not None and not self.provider.pricing.known \
                and self.provider.kind != "fake":
            raise ValueError("optimizer.max_cost_usd requires provider.pricing input/output rates")
        return self

    @property
    def proposer_provider(self) -> ProviderConfig:
        return self.optimizer.provider or self.provider

    def config_hash(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json", by_alias=True)))[:16]

    def pipeline_id(self) -> str:
        body = {"v": PIPELINE_VERSION, "mode": self.input.mode.value,
                "json": self.input.json_records.model_dump(mode="json") if self.input.json_records else None,
                "files": self.input.files.model_dump(mode="json"), "documents": self.documents.model_dump(mode="json")}
        return "p-" + sha256_text(canonical_json(body))[:16]

    def compatibility_key(self) -> str:
        body = {"client": self.client_id, "task": self.task.identity(), "mode": self.input.mode.value,
                "pipeline": self.pipeline_id(), "provider": self.provider.identity(), "schema": 1}
        return "k-" + sha256_text(canonical_json(body))[:16]

    def resolved_data_dir(self) -> Path:
        return self.data_dir.resolve()


def initialize_config(path: Path, *, labels: list[str], model: str, client_id: str,
                      reviewer: str, mode: str = "json_records",
                      base_url: str = "https://api.openai.com/v1",
                      api_key_env: str = "OPENAI_API_KEY") -> Path:
    raw: dict[str, object] = {
        "client_id": client_id,
        "reviewers": [reviewer],
        "task": {"name": "classification", "labels": labels},
        "input": {"mode": mode},
        "provider": {"kind": "openai_compatible", "model": model,
                     "base_url": base_url, "api_key_env": api_key_env},
    }
    input_settings: dict[str, object] = {"mode": mode}
    if mode == "json_records":
        input_settings["json"] = {"shape": "array", "id_field": "id", "text_fields": ["text"],
                                  "label_field": "ground_truth"}
    elif mode == "direct_files":
        input_settings["roots"] = ["documents"]
    raw["input"] = input_settings
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid project settings: {exc}") from None
    if not model.strip() or not reviewer.strip() or not api_key_env.strip():
        raise ConfigError("Model, reviewer and credential variable name must not be empty")
    settings = cfg.model_dump(mode="json", by_alias=True, exclude_none=True,
                             include={"client_id", "data_dir", "reviewers", "task", "input", "provider", "policy"})
    settings["optimizer"] = {"proposer": "model", "max_requests": 2000, "max_elapsed_s": 1800}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(tomli_w.dumps(settings))
    except FileExistsError:
        raise ConfigError("Configuration already exists; choose another path", "config_exists") from None
    except OSError:
        raise ConfigError("Cannot write configuration file", "config_write_failed") from None
    return path


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p.name}", "config_not_found")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Configuration is not valid TOML: {exc}", "config_parse_error") from None
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"Invalid configuration: {details}") from None
    base = p.resolve().parent
    cfg.data_dir = cfg.data_dir if cfg.data_dir.is_absolute() else (base / cfg.data_dir)
    cfg.input.roots = [r if r.is_absolute() else (base / r) for r in cfg.input.roots]
    return cfg
