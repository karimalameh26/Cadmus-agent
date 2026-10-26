"""Input adapters: explicitly selected `json_records` or `direct_files` mode into canonical evidence.

JSON mode never opens original documents, follows embedded paths, or performs network access.
Ground-truth label fields are routed to the label channel and never become model-visible evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from cadmus.config import AppConfig, JsonInputConfig
from cadmus.contracts import EvidenceRecord, InputMode, TextBlock, canonical_json, sha256_bytes, sha256_text
from cadmus.documents import SUPPORTED_EXTENSIONS, extract_file
from cadmus.errors import ConfigError, ExtractionError, InputError
from cadmus.storage import ArtifactStore


@dataclass
class InputItem:
    ref: str
    evidence: EvidenceRecord | None = None
    source_labels: list[str] | None = None
    error_code: str | None = None
    detail: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    name: str | None = None  # source name for explicit user output only; never logged
    @property
    def ok(self) -> bool:
        return self.evidence is not None


class InputAdapter(Protocol):
    mode: InputMode

    def load(self, sources: Sequence[Path]) -> Iterator[InputItem]: ...


def document_id_for(client_id: str, name: str) -> str:
    return "doc-" + sha256_text(f"{client_id}|{name}")[:16]


def family_id_for(client_id: str, key: str) -> str:
    return "fam-" + sha256_text(f"{client_id}|{key.lower()}")[:16]


def _stem(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).stem or name


def _get(record: dict[str, Any], dotted: str) -> Any:
    cur: Any = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


class JsonRecordAdapter:
    mode = InputMode.JSON_RECORDS

    def __init__(self, cfg: AppConfig) -> None:
        if cfg.input.json_records is None:
            raise ConfigError("input.json mapping is required for json_records mode")
        self.cfg = cfg
        self.mapping: JsonInputConfig = cfg.input.json_records
        self.pipeline_id = cfg.pipeline_id()

    def _raw_records(self, path: Path) -> Iterator[tuple[str, Any] | InputItem]:
        m = self.mapping
        if m.shape == "jsonl":
            with path.open("r", encoding="utf-8-sig") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        yield InputItem(ref=f"line:{lineno}", error_code="malformed_json", detail="invalid JSON line")
                        continue
                    yield self._id_of(rec, f"line:{lineno}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise InputError("JSON source could not be parsed", "malformed_json") from None
        if m.records_path:
            data = _get(data, m.records_path) if isinstance(data, dict) else None
        if m.shape == "array":
            if not isinstance(data, list):
                raise InputError("Configured JSON shape 'array' does not match the source", "shape_mismatch")
            for idx, rec in enumerate(data):
                yield self._id_of(rec, f"index:{idx}")
        else:
            if not isinstance(data, dict):
                raise InputError("Configured JSON shape 'keyed' does not match the source", "shape_mismatch")
            for key, rec in data.items():
                if not isinstance(rec, dict):
                    yield InputItem(ref=f"key:{sha256_text(key)[:12]}", error_code="record_not_object")
                    continue
                yield (str(key), rec)

    def _id_of(self, rec: Any, ref: str) -> tuple[str, Any] | InputItem:
        if not isinstance(rec, dict):
            return InputItem(ref=ref, error_code="record_not_object")
        assert self.mapping.id_field
        value = _get(rec, self.mapping.id_field)
        if not isinstance(value, (str, int)) or str(value).strip() == "":
            return InputItem(ref=ref, error_code="missing_id")
        return (str(value), rec)

    def _text_value(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return "\n".join(value)
        return None

    def _build(self, name: str, rec: dict[str, Any]) -> InputItem:
        m, cid = self.mapping, self.cfg.client_id
        doc_id = document_id_for(cid, name)
        policy = self.cfg.documents.oversize_policy
        text: list[TextBlock] = []
        fields: dict[str, Any] = {}
        truncated: list[str] = []
        missing: list[str] = []

        def bounded(field_name: str, value: str) -> str | InputItem:
            if len(value) <= m.max_field_chars:
                return value
            if policy == "reject":
                return InputItem(ref=doc_id, error_code="field_too_large", detail=f"field '{field_name}'")
            truncated.append(field_name)
            return value[: m.max_field_chars]

        for kind, names in (("text", m.text_fields), ("description", m.description_fields)):
            for fname in names:
                raw = _get(rec, fname)
                if raw is None:
                    missing.append(fname)
                    continue
                value = self._text_value(raw)
                if value is None:
                    return InputItem(ref=doc_id, error_code="field_type_invalid", detail=f"field '{fname}'")
                checked = bounded(fname, value)
                if isinstance(checked, InputItem):
                    return checked
                if checked.strip():
                    text.append(TextBlock(source=f"{kind}:{fname}", text=checked))
        for prefix, names in (("metadata", m.metadata_fields), ("features", m.feature_fields)):
            for fname in names:
                raw = _get(rec, fname)
                if raw is None:
                    missing.append(fname)
                    continue
                if len(canonical_json(raw)) > m.max_field_chars:
                    return InputItem(ref=doc_id, error_code="field_too_large", detail=f"field '{fname}'")
                fields[f"{prefix}.{fname}"] = raw
        if m.expose_id:
            fields["document_name"] = name

        labels: list[str] | None = None
        if m.label_field:
            raw_label = _get(rec, m.label_field)
            if isinstance(raw_label, str) and raw_label.strip():
                labels = [raw_label.strip()]
            elif isinstance(raw_label, list) and all(isinstance(x, str) for x in raw_label):
                labels = [x.strip() for x in raw_label]
            elif raw_label is not None:
                return InputItem(ref=doc_id, error_code="label_type_invalid")

        family_raw = _get(rec, m.family_field) if m.family_field else None
        family = family_id_for(cid, str(family_raw) if family_raw not in (None, "") else _stem(name))
        source = {k: v for k, v in rec.items() if k != (m.label_field or "").split(".")[0]}
        ev = EvidenceRecord(
            document_id=doc_id, family_id=family, input_mode=InputMode.JSON_RECORDS,
            source_hash=sha256_text(canonical_json(source)), pipeline_id=self.pipeline_id,
            text=text, fields=fields,
            provenance={"origin": "json_record", "truncated": bool(truncated), "truncated_fields": truncated,
                        "missing_fields": missing,
                        "note": "descriptions are supplied text, not direct inspection of the original"},
        )
        if not ev.has_evidence():
            return InputItem(ref=doc_id, error_code="no_evidence", source_labels=labels)
        return InputItem(ref=doc_id, evidence=ev, source_labels=labels, provenance=ev.provenance)

    def load(self, sources: Sequence[Path]) -> Iterator[InputItem]:
        raw: list[tuple[str, Any] | InputItem] = []
        for src in sources:
            if not src.is_file():
                raise InputError("JSON source file not found", "source_not_found")
            raw.extend(self._raw_records(src))
        counts = Counter(r[0] for r in raw if isinstance(r, tuple))
        for r in raw:
            if isinstance(r, InputItem):
                yield r
            elif counts[r[0]] > 1:
                yield InputItem(ref=document_id_for(self.cfg.client_id, r[0]), error_code="duplicate_id", name=r[0])
            else:
                item = self._build(*r)
                item.name = r[0]
                yield item


class DirectFileAdapter:
    mode = InputMode.DIRECT_FILES

    def __init__(self, cfg: AppConfig, artifacts: ArtifactStore) -> None:
        if not cfg.input.roots:
            raise ConfigError("input.roots must list allowed directories for direct_files mode", "roots_required")
        self.cfg = cfg
        self.artifacts = artifacts
        self.roots = [r.resolve() for r in cfg.input.roots]
        self.pipeline_id = cfg.pipeline_id()

    def _root_for(self, path: Path) -> Path | None:
        resolved = path.resolve()
        for root in self.roots:
            if resolved.is_relative_to(root):
                return root
        return None

    def discover(self, sources: Sequence[Path]) -> Iterator[tuple[Path, str] | InputItem]:
        for src in sources:
            root = self._root_for(src)
            if root is None:
                yield InputItem(ref=f"src:{sha256_text(str(src))[:12]}", error_code="outside_input_roots")
                continue
            if not src.exists():
                yield InputItem(ref=f"src:{sha256_text(str(src))[:12]}", error_code="source_not_found")
                continue
            files = sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else [src]
            for f in files:
                froot = self._root_for(f)
                name = f.resolve().relative_to(froot).as_posix() if froot else ""
                if froot is None:
                    yield InputItem(ref=f"src:{sha256_text(str(f))[:12]}", error_code="outside_input_roots")
                elif f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    yield InputItem(ref=document_id_for(self.cfg.client_id, name), error_code="unsupported_format",
                                    name=name)
                else:
                    yield (f, name)

    def load(self, sources: Sequence[Path]) -> Iterator[InputItem]:
        cid = self.cfg.client_id
        for found in self.discover(sources):
            if isinstance(found, InputItem):
                yield found
                continue
            path, name = found
            doc_id = document_id_for(cid, name)
            try:
                extracted, data = extract_file(path, self.cfg.documents, self.artifacts)
            except ExtractionError as exc:
                yield InputItem(ref=doc_id, error_code=exc.code, detail=exc.message, name=name)
                continue
            fields: dict[str, Any] = {}
            if self.cfg.input.files.expose_filename:
                fields["file_name"] = PurePosixPath(name).name
            family_key = _stem(name) if self.cfg.input.files.family_from == "stem" else name
            ev = EvidenceRecord(
                document_id=doc_id, family_id=family_id_for(cid, family_key), input_mode=InputMode.DIRECT_FILES,
                source_hash=sha256_bytes(data), pipeline_id=self.pipeline_id, text=extracted.text, fields=fields,
                media=extracted.media, provenance=extracted.provenance,
            )
            if not ev.has_evidence():
                yield InputItem(ref=doc_id, error_code="no_evidence", provenance=ev.provenance, name=name)
            else:
                yield InputItem(ref=doc_id, evidence=ev, provenance=ev.provenance, name=name)


def make_input_adapter(cfg: AppConfig, artifacts: ArtifactStore) -> InputAdapter:
    if cfg.input.mode == InputMode.JSON_RECORDS:
        return JsonRecordAdapter(cfg)
    return DirectFileAdapter(cfg, artifacts)
