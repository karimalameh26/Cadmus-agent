"""SQLite state store with migrations plus a content-addressed, client-scoped artifact store."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cadmus.contracts import EvidenceRecord, sha256_bytes

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
    params TEXT NOT NULL, config_hash TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    summary TEXT
);
CREATE TABLE events (
    run_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_id TEXT NOT NULL UNIQUE, timestamp TEXT NOT NULL,
    level TEXT NOT NULL, event_type TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
CREATE TABLE evidence (
    client_id TEXT NOT NULL, input_mode TEXT NOT NULL, document_id TEXT NOT NULL, family_id TEXT NOT NULL,
    evidence_hash TEXT NOT NULL, pipeline_id TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (client_id, input_mode, document_id, evidence_hash)
);
CREATE TABLE labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT NOT NULL, task TEXT NOT NULL,
    document_id TEXT NOT NULL, family_id TEXT NOT NULL, labels TEXT NOT NULL, status TEXT NOT NULL,
    source TEXT NOT NULL, submitted_by TEXT NOT NULL, verifier TEXT, reason TEXT,
    created_at TEXT NOT NULL, decided_at TEXT, superseded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX labels_doc ON labels (client_id, task, document_id);
CREATE TABLE snapshots (
    snapshot_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, task TEXT NOT NULL, input_mode TEXT NOT NULL,
    pipeline_id TEXT NOT NULL, seed INTEGER NOT NULL, created_at TEXT NOT NULL, summary TEXT NOT NULL
);
CREATE TABLE snapshot_items (
    snapshot_id TEXT NOT NULL, document_id TEXT NOT NULL, family_id TEXT NOT NULL, evidence_hash TEXT NOT NULL,
    labels TEXT NOT NULL, split TEXT NOT NULL, PRIMARY KEY (snapshot_id, document_id)
);
CREATE TABLE gate_exposures (
    client_id TEXT NOT NULL, task TEXT NOT NULL, input_mode TEXT NOT NULL, family_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL, run_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (client_id, task, input_mode, family_id)
);
CREATE TABLE predictions (
    cache_key TEXT PRIMARY KEY, client_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE run_predictions (
    run_id TEXT NOT NULL, purpose TEXT NOT NULL, bundle_id TEXT NOT NULL, document_id TEXT NOT NULL,
    repeat_index INTEGER NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, purpose, bundle_id, document_id, repeat_index)
);
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, cache_key TEXT NOT NULL, attempt INTEGER NOT NULL,
    status TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, input_tokens INTEGER, output_tokens INTEGER,
    reserved_cost REAL, actual_cost REAL, error_code TEXT, provider_request_id TEXT
);
CREATE TABLE bundles (
    bundle_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, compat_key TEXT NOT NULL, payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE champions (
    client_id TEXT NOT NULL, compat_key TEXT NOT NULL, bundle_id TEXT NOT NULL, version INTEGER NOT NULL,
    updated_at TEXT NOT NULL, PRIMARY KEY (client_id, compat_key)
);
CREATE TABLE champion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT NOT NULL, compat_key TEXT NOT NULL,
    from_bundle TEXT, to_bundle TEXT NOT NULL, action TEXT NOT NULL, run_id TEXT, reason TEXT,
    reference_score REAL, created_at TEXT NOT NULL
);
CREATE TABLE decisions (
    run_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, compat_key TEXT NOT NULL, outcome TEXT NOT NULL,
    payload TEXT NOT NULL, created_at TEXT NOT NULL
);
"""),
]


class Store:
    """Single-connection SQLite store; one serialized writer per process."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
            applied = {r[0] for r in self._conn.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in filter(str.strip, sql.split(";")):
                        self._conn.execute(statement)
                    self._conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise

    @property
    def schema_version(self) -> int:
        row = self.one("SELECT MAX(version) AS v FROM schema_migrations")
        return int(row["v"]) if row and row["v"] is not None else 0

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ArtifactStore:
    """Content-addressed bytes under data_dir/artifacts/<client_id>/; never shared across clients."""

    def __init__(self, data_dir: Path, client_id: str) -> None:
        self.root = data_dir / "artifacts" / client_id
        self.runs_root = data_dir / "runs" / client_id

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise ValueError("invalid artifact digest")
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self._path(digest)
        if not path.exists():
            atomic_write(path, data)
        return digest

    def get(self, digest: str) -> bytes:
        data = self._path(digest).read_bytes()
        if sha256_bytes(data) != digest:
            raise ValueError("artifact integrity check failed")
        return data

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def put_evidence(self, evidence: EvidenceRecord) -> str:
        digest = self.put(evidence.canonical().encode("utf-8"))
        assert digest == evidence.evidence_hash()
        return digest

    def get_evidence(self, digest: str) -> EvidenceRecord:
        return EvidenceRecord.model_validate_json(self.get(digest))

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_run_json(self, run_id: str, name: str, obj: Any) -> Path:
        path = self.run_dir(run_id) / name
        atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"))
        return path

    def write_run_text(self, run_id: str, name: str, text: str) -> Path:
        path = self.run_dir(run_id) / name
        atomic_write(path, text.encode("utf-8"))
        return path
