from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ag_agent.config import AppConfig
from ag_agent.demo import make_records
from ag_agent.providers import FakeModel
from ag_agent.runtime import Framework


def base_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "client_id": "acme",
        "data_dir": str(tmp_path / "state"),
        "reviewers": ["rev-1", "rev-2"],
        "task": {"name": "doc-type", "labels": ["invoice", "receipt", "contract"], "critical_labels": ["contract"],
                 "baseline_guidance": "Classify the document by type."},
        "input": {"mode": "json_records", "json": {
            "shape": "array", "id_field": "image_name", "text_fields": ["ocr_text"],
            "description_fields": ["image_description"], "metadata_fields": ["metadata"], "label_field": "label"}},
        "provider": {"kind": "fake", "backoff_base_s": 0},
        "optimizer": {"candidates_per_round": 4, "max_rounds": 5, "patience": 2, "max_examples": 3},
        "policy": {"bootstrap_resamples": 300, "min_coverage": 0.8},
        "monitor": {"min_samples": 10, "cooldown_hours": 0},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def write_records(path: Path, batch: int = 1, per_class: int = 30) -> Path:
    path.write_text(json.dumps(make_records(batch, per_class)), encoding="utf-8")
    return path


@pytest.fixture
def make_fw(tmp_path: Path):
    created: list[Framework] = []

    def factory(model: FakeModel | None = None, **overrides: Any) -> Framework:
        cfg = AppConfig.model_validate(base_config(tmp_path, **overrides))
        fw = Framework(cfg, model=model or FakeModel(), sleep=lambda s: None)
        created.append(fw)
        return fw

    yield factory
    for fw in created:
        fw.close()


@pytest.fixture
def records(tmp_path: Path) -> Path:
    return write_records(tmp_path / "records.json")
