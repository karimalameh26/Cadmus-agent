"""Synthetic, credential-free demo data. Scripted fake-provider improvement proves orchestration only."""

from __future__ import annotations

import json
import random
from pathlib import Path

CUES = {
    "invoice": ["invoice", "remittance", "payable", "billing"],
    "receipt": ["receipt", "cashier", "change", "purchase"],
    "contract": ["agreement", "hereby", "termination", "parties"],
}
NOISE = ["total", "date", "company", "page", "reference", "amount", "customer", "address", "number", "document",
         "signature", "office", "account", "period"]
DESCRIPTIONS = ["scanned page with a logo and a table", "photo of a printed page", "single page with stamps",
                "two-column printed layout"]

DEMO_CONFIG = """# Demo configuration. Thresholds are demonstrations only, not recommendations for client data.
client_id = "demo"
data_dir = ".cadmus"
reviewers = ["reviewer-1", "reviewer-2"]

[task]
name = "doc-type"
kind = "single_label"
labels = ["invoice", "receipt", "contract"]
critical_labels = ["contract"]
baseline_guidance = "Classify the document by its business document type."

[input]
mode = "json_records"

[input.json]
shape = "array"
id_field = "image_name"
text_fields = ["ocr_text"]
description_fields = ["image_description"]
metadata_fields = ["metadata"]
label_field = "label"

[provider]
kind = "fake"

[optimizer]
candidates_per_round = 4
max_rounds = 5
patience = 2
max_examples = 3
seed = 7

[policy]
version = "demo-1"
primary_metric = "macro_f1"
min_gain = 0.02
confidence = 0.9
bootstrap_resamples = 500
min_gate_groups = 10
min_class_support = 2
absolute_floor = 0.5
min_coverage = 0.8

[monitor]
min_samples = 10
cooldown_hours = 0
"""


def make_records(batch: int = 1, per_class: int = 30, seed: int = 7) -> list[dict[str, object]]:
    rng = random.Random(seed * 100 + batch)
    records: list[dict[str, object]] = []
    labels = list(CUES)
    order = [lbl for lbl in labels for _ in range(per_class)]
    rng.shuffle(order)
    for i, label in enumerate(order, start=1):
        words = rng.sample(CUES[label], 2) + rng.sample(NOISE, 6)
        rng.shuffle(words)
        records.append({
            "image_name": f"scan_{batch}{i:04d}.png",
            "ocr_text": " ".join(words).capitalize() + ".",
            "image_description": rng.choice(DESCRIPTIONS),
            "metadata": {"source": rng.choice(["scanner-a", "scanner-b", "mobile"]), "pages": rng.randint(1, 3)},
            "label": label,
        })
    return records


def write_demo(directory: Path, batch: int = 1) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    cfg = directory / "cadmus.toml"
    if not cfg.exists():
        cfg.write_text(DEMO_CONFIG, encoding="utf-8")
        written["config"] = str(cfg)
    data = directory / f"records_batch{batch}.json"
    data.write_text(json.dumps(make_records(batch), indent=2), encoding="utf-8")
    written["records"] = str(data)
    return written
