from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from ag_agent.config import AppConfig
from ag_agent.contracts import InputMode
from ag_agent.inputs import JsonRecordAdapter
from ag_agent.prompting import render_evidence
from tests.conftest import base_config


def adapter(tmp_path: Path, **json_overrides) -> JsonRecordAdapter:
    raw = base_config(tmp_path)
    raw["input"]["json"].update(json_overrides)
    return JsonRecordAdapter(AppConfig.model_validate(raw))


def test_array_mapping_routes_label_away_from_evidence(tmp_path: Path) -> None:
    src = tmp_path / "r.json"
    src.write_text(json.dumps([{"image_name": "a.png", "ocr_text": "Invoice number 7", "image_description": "logo",
                                "metadata": {"pages": 2}, "label": "invoice", "secret_path": "C:/x/a.png"}]))
    [item] = list(adapter(tmp_path).load([src]))
    assert item.ok and item.source_labels == ["invoice"]
    ev = item.evidence
    assert ev.input_mode == InputMode.JSON_RECORDS
    rendered = render_evidence(ev)
    assert "invoice number 7" in rendered.lower()
    assert "supplied description" in rendered and "not direct inspection" in rendered
    assert "label" not in json.dumps(ev.fields) and "C:/x" not in ev.canonical()
    assert "a.png" not in rendered  # names are identifiers unless explicitly exposed


def test_keyed_and_jsonl_shapes(tmp_path: Path) -> None:
    keyed = tmp_path / "k.json"
    keyed.write_text(json.dumps({"scan1.png": {"ocr_text": "hello world"}, "scan2.png": "not-an-object"}))
    items = list(adapter(tmp_path, shape="keyed", id_field=None).load([keyed]))
    assert [i.ok for i in items] == [True, False] and items[1].error_code == "record_not_object"

    jl = tmp_path / "r.jsonl"
    jl.write_text('{"image_name": "x.png", "ocr_text": "abc def"}\n{bad json\n\n{"ocr_text": "no id"}\n')
    items = list(adapter(tmp_path, shape="jsonl").load([jl]))
    assert [i.error_code for i in items] == [None, "malformed_json", "missing_id"]


def test_duplicates_empty_and_type_errors_are_rejected_not_dropped(tmp_path: Path) -> None:
    src = tmp_path / "r.json"
    src.write_text(json.dumps([
        {"image_name": "d.png", "ocr_text": "one"}, {"image_name": "d.png", "ocr_text": "two"},
        {"image_name": "e.png"}, {"image_name": "f.png", "ocr_text": 42}, {"image_name": "g.png", "ocr_text": "ok",
                                                                              "label": 5}]))
    codes = [i.error_code for i in adapter(tmp_path).load([src])]
    assert codes == ["duplicate_id", "duplicate_id", "no_evidence", "field_type_invalid", "label_type_invalid"]


def test_json_mode_never_opens_original_files_or_network(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "r.json"
    src.write_text(json.dumps([{"image_name": str(tmp_path / "real.png"), "ocr_text": "Receipt cashier"}]))
    (tmp_path / "real.png").write_bytes(b"not read")
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("file read")))
    monkeypatch.setattr("ag_agent.inputs.extract_file", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    [item] = list(adapter(tmp_path).load([src]))
    assert item.ok and not item.evidence.media


def test_oversize_field_policy(tmp_path: Path) -> None:
    src = tmp_path / "r.json"
    src.write_text(json.dumps([{"image_name": "a.png", "ocr_text": "x" * 50}]))
    [item] = list(adapter(tmp_path, max_field_chars=10).load([src]))
    assert item.ok and item.evidence.provenance["truncated_fields"] == ["ocr_text"]
    raw = base_config(tmp_path, documents={"oversize_policy": "reject"})
    raw["input"]["json"]["max_field_chars"] = 10
    [item] = list(JsonRecordAdapter(AppConfig.model_validate(raw)).load([src]))
    assert item.error_code == "field_too_large"


@pytest.mark.parametrize("change", [
    {"label_field": "ocr_text"},
    {"text_fields": [], "description_fields": [], "metadata_fields": []},
    {"shape": "keyed"},
])
def test_ambiguous_mappings_rejected(tmp_path: Path, change: dict) -> None:
    raw = base_config(tmp_path)
    raw["input"]["json"].update(change)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_live_provider_requires_https(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(base_config(tmp_path, provider={"kind": "openai_compatible",
                                                                 "base_url": "http://example.com/v1"}))
