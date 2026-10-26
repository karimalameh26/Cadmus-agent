"""Structured run events: durable SQLite audit trail, resumable JSONL export, console and SDK subscribers.

Events carry identifiers, counts and reason codes only. Customer content, prompts and secrets are never fields.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ag_agent.contracts import SCHEMA_VERSION, utc_now
from ag_agent.errors import AuditFailure
from ag_agent.storage import Store

logging.getLogger("ag_agent").addHandler(logging.NullHandler())

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

ALLOWED_FIELDS = frozenset({
    "client_id", "operation_id", "parent_operation_id", "attempt", "document_id", "family_id", "candidate_id",
    "bundle_id", "snapshot_id", "input_mode", "duration_ms", "completed", "total", "failed", "accepted",
    "rejected", "skipped", "model_hash", "config_hash", "cache_hit", "input_tokens", "output_tokens",
    "estimated_cost_usd", "actual_cost_usd", "budget_remaining", "provider_request_id", "error_code",
    "retry_delay_ms", "artifact", "reason_codes", "round", "split", "counts", "metrics", "gate", "passed",
    "observed", "threshold", "outcome", "pages_total", "pages_included", "pages_omitted", "truncated",
    "transforms", "evidence_hash", "label_id", "reviewer", "label_status", "purpose", "repeat_index",
    "diff_lines", "score", "rank", "compat_key", "from_bundle", "to_bundle", "request_chars", "media_count",
    "status_detail", "pid", "sequence_from", "elapsed_s", "lower_bound", "upper_bound", "examples",
})

Subscriber = Callable[[dict[str, Any]], None]


class JsonlExporter:
    """Append-only JSONL export with size rotation; recovers lag from the database by sequence number."""

    def __init__(self, path: Path, max_bytes: int, backups: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self._marker = path.with_suffix(".seq")
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def last_sequence(self) -> int:
        try:
            return int(self._marker.read_text(encoding="utf-8").strip() or 0)
        except (FileNotFoundError, ValueError):
            return 0

    def _rotate(self) -> None:
        if self.backups == 0:
            self.path.unlink(missing_ok=True)
            return
        for i in range(self.backups - 1, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{i}")
            if src.exists():
                src.replace(self.path.with_name(f"{self.path.name}.{i + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            if event["sequence"] <= self.last_sequence():
                return
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                self._rotate()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            self._marker.write_text(str(event["sequence"]), encoding="utf-8")

    def sync_from(self, events: list[dict[str, Any]]) -> int:
        written = 0
        for event in events:
            if event["sequence"] > self.last_sequence():
                self.append(event)
                written += 1
        return written


class EventEmitter:
    def __init__(self, store: Store, run_id: str, client_id: str, exporter: JsonlExporter | None,
                 subscribers: list[Subscriber] | None = None) -> None:
        self.store = store
        self.run_id = run_id
        self.client_id = client_id
        self.exporter = exporter
        self.subscribers = list(subscribers or [])
        self._lock = threading.Lock()
        row = store.one("SELECT MAX(sequence) AS s FROM events WHERE run_id = ?", (run_id,))
        self._seq = int(row["s"]) if row and row["s"] is not None else 0
        self._export_failed = False
        self._failed_subscribers: set[int] = set()
        if exporter is not None:
            self._recover_export()

    def _recover_export(self) -> None:
        assert self.exporter is not None
        if self.exporter.last_sequence() < self._seq:
            self.exporter.sync_from(load_events(self.store, self.run_id))

    def emit(self, event_type: str, stage: str, status: str, message: str, level: str = "INFO",
             **fields: Any) -> dict[str, Any]:
        unknown = set(fields) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"event fields not allowlisted: {sorted(unknown)}")
        with self._lock:
            self._seq += 1
            event: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "event_id": "evt-" + uuid.uuid4().hex[:16],
                "timestamp": utc_now(), "sequence": self._seq, "level": level, "event_type": event_type,
                "run_id": self.run_id, "stage": stage, "status": status, "message": message,
                "client_id": self.client_id,
            }
            event.update({k: v for k, v in fields.items() if v is not None})
            try:
                with self.store.tx() as conn:
                    conn.execute(
                        "INSERT INTO events (run_id, sequence, event_id, timestamp, level, event_type, stage, "
                        "status, payload) VALUES (?,?,?,?,?,?,?,?,?)",
                        (self.run_id, self._seq, event["event_id"], event["timestamp"], level, event_type, stage,
                         status, json.dumps(event, ensure_ascii=False, sort_keys=True)))
                    conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (event["timestamp"], self.run_id))
            except sqlite3.Error as exc:
                self._seq -= 1
                sys.stderr.write(f"ag-agent: durable audit write failed ({type(exc).__name__}); stopping\n")
                raise AuditFailure("Durable event persistence failed; work stopped and champion unchanged") from None
            if self.exporter is not None:
                try:
                    self.exporter.append(event)
                except OSError as exc:
                    if not self._export_failed:
                        sys.stderr.write(f"ag-agent: JSONL export failed ({type(exc).__name__}); events remain in "
                                         "the database and will be re-exported on recovery\n")
                    self._export_failed = True
        for sub in self.subscribers:
            try:
                sub(event)
            except Exception as exc:  # a broken subscriber must never affect run state
                if id(sub) not in self._failed_subscribers:
                    self._failed_subscribers.add(id(sub))
                    sys.stderr.write(f"ag-agent: event subscriber failed ({type(exc).__name__}); ignoring it\n")
        return event


def load_events(store: Store, run_id: str, after: int = 0) -> list[dict[str, Any]]:
    rows = store.all("SELECT payload FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                     (run_id, after))
    return [json.loads(r["payload"]) for r in rows]


ITEM_EVENT_PREFIXES = ("document.", "provider.", "prediction.", "request.", "label.")


class ConsoleSubscriber:
    """Human-readable progress on stderr. Per-item events are shown only in verbose mode."""

    def __init__(self, stream: TextIO | None = None, level: str = "INFO", verbose: bool = False,
                 color: bool | None = None) -> None:
        self.stream = stream or sys.stderr
        self.level = LEVELS[level]
        self.verbose = verbose
        isatty = getattr(self.stream, "isatty", lambda: False)()
        self.color = isatty if color is None else color
        self.tty = isatty

    def __call__(self, event: dict[str, Any]) -> None:
        lvl = LEVELS.get(event["level"], 20)
        if lvl < self.level:
            return
        if not self.verbose and event["event_type"].startswith(ITEM_EVENT_PREFIXES) and lvl < 30:
            return
        ts = event["timestamp"][11:19]
        parts = [f"{ts}", f"{event['stage']:<22}", event["message"]]
        if "completed" in event and "total" in event:
            parts.append(f"[{event['completed']}/{event['total']}]")
        for key in ("document_id", "candidate_id", "bundle_id", "outcome", "error_code", "score", "passed",
                    "observed", "threshold", "reason_codes", "budget_remaining", "duration_ms"):
            if key in event:
                parts.append(f"{key}={event[key]}")
        line = "  ".join(str(p) for p in parts)
        if self.color and lvl >= 30:
            line = ("\x1b[33m" if lvl == 30 else "\x1b[31m") + line + "\x1b[0m"
        self.stream.write(line + "\n")
        self.stream.flush()


def safe_error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    return "unexpected_" + type(exc).__name__.lower()
