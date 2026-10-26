"""Trusted labels, evidence persistence and immutable grouped split snapshots."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cadmus.config import AppConfig
from cadmus.contracts import EvidenceRecord, canonical_json, sha256_text, utc_now
from cadmus.errors import AccessDenied, InputError, NotFound
from cadmus.observability import EventEmitter
from cadmus.storage import ArtifactStore, Store
from cadmus.tasks import ClassificationTask

SPLITS = ("development", "selection", "gate")


@dataclass(frozen=True)
class SnapshotItem:
    document_id: str
    family_id: str
    evidence_hash: str
    labels: list[str]
    split: str


class DatasetService:
    def __init__(self, cfg: AppConfig, task: ClassificationTask, store: Store, artifacts: ArtifactStore) -> None:
        self.cfg = cfg
        self.task = task
        self.store = store
        self.artifacts = artifacts

    # --- evidence -------------------------------------------------------------------------------------------
    def save_evidence(self, ev: EvidenceRecord) -> str:
        digest = self.artifacts.put_evidence(ev)
        with self.store.tx() as conn:
            conn.execute("INSERT OR IGNORE INTO evidence (client_id, input_mode, document_id, family_id, "
                         "evidence_hash, pipeline_id, created_at) VALUES (?,?,?,?,?,?,?)",
                         (self.cfg.client_id, ev.input_mode.value, ev.document_id, ev.family_id, digest,
                          ev.pipeline_id, utc_now()))
        return digest

    def load_evidence(self, evidence_hash: str) -> EvidenceRecord:
        try:
            return self.artifacts.get_evidence(evidence_hash)
        except FileNotFoundError:
            raise NotFound("Evidence artifact is not retained; exact replay unavailable", "evidence_missing") from None

    # --- labels ---------------------------------------------------------------------------------------------
    def _check_reviewer(self, reviewer: str) -> None:
        if reviewer not in self.cfg.reviewers:
            raise AccessDenied("Identity is not a configured reviewer", "reviewer_not_authorized")

    def add_label(self, document_id: str, family_id: str, labels: list[str], *, source: str, submitted_by: str,
                  verified_by: str | None = None) -> int:
        canon = self.task.validate_labels(labels)
        if verified_by is not None:
            self._check_reviewer(verified_by)
        now = utc_now()
        with self.store.tx() as conn:
            cur = conn.execute(
                "INSERT INTO labels (client_id, task, document_id, family_id, labels, status, source, submitted_by, "
                "verifier, created_at, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.cfg.client_id, self.cfg.task.name, document_id, family_id, json.dumps(canon),
                 "verified" if verified_by else "pending", source, submitted_by, verified_by, now,
                 now if verified_by else None))
            return int(cur.lastrowid or 0)

    def _label_row(self, label_id: int) -> Any:
        row = self.store.one("SELECT * FROM labels WHERE label_id=? AND client_id=? AND task=?",
                             (label_id, self.cfg.client_id, self.cfg.task.name))
        if row is None:
            raise NotFound("Label not found for this client/task", "label_not_found")
        return row

    def verify(self, label_id: int, reviewer: str) -> dict[str, Any]:
        self._check_reviewer(reviewer)
        row = self._label_row(label_id)
        if row["status"] != "pending":
            raise InputError(f"Label is already {row['status']}", "label_not_pending")
        if row["submitted_by"] == reviewer and row["source"] == "correction":
            raise AccessDenied("A reviewer cannot verify their own correction", "self_verification")
        with self.store.tx() as conn:
            conn.execute("UPDATE labels SET superseded=1 WHERE client_id=? AND task=? AND document_id=? "
                         "AND status='verified' AND superseded=0", (self.cfg.client_id, self.cfg.task.name,
                                                                    row["document_id"]))
            conn.execute("UPDATE labels SET status='verified', verifier=?, decided_at=? WHERE label_id=?",
                         (reviewer, utc_now(), label_id))
        return {"label_id": label_id, "status": "verified", "document_id": row["document_id"]}

    def reject(self, label_id: int, reviewer: str, reason: str) -> dict[str, Any]:
        self._check_reviewer(reviewer)
        row = self._label_row(label_id)
        if row["status"] != "pending":
            raise InputError(f"Label is already {row['status']}", "label_not_pending")
        self.store.execute("UPDATE labels SET status='rejected', verifier=?, reason=?, decided_at=? WHERE label_id=?",
                           (reviewer, reason[:500], utc_now(), label_id))
        return {"label_id": label_id, "status": "rejected", "document_id": row["document_id"]}

    def list_labels(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM labels WHERE client_id=? AND task=?"
        params: list[Any] = [self.cfg.client_id, self.cfg.task.name]
        if status:
            sql += " AND status=?"
            params.append(status)
        return [dict(r) | {"labels": json.loads(r["labels"])} for r in self.store.all(sql + " ORDER BY label_id",
                                                                                         tuple(params))]

    def active_labels(self) -> tuple[dict[str, list[str]], dict[str, str], list[str]]:
        """Return (document -> labels, document -> family, conflicting document ids)."""
        rows = self.store.all("SELECT document_id, family_id, labels FROM labels WHERE client_id=? AND task=? "
                              "AND status='verified' AND superseded=0 ORDER BY label_id",
                              (self.cfg.client_id, self.cfg.task.name))
        seen: dict[str, set[str]] = defaultdict(set)
        families: dict[str, str] = {}
        for r in rows:
            seen[r["document_id"]].add(r["labels"])
            families[r["document_id"]] = r["family_id"]
        labels = {d: json.loads(next(iter(v))) for d, v in seen.items() if len(v) == 1}
        conflicts = sorted(d for d, v in seen.items() if len(v) > 1)
        return labels, families, conflicts

    # --- snapshots ------------------------------------------------------------------------------------------
    def exposed_families(self, exclude_run: str | None = None) -> set[str]:
        rows = self.store.all("SELECT family_id, run_id FROM gate_exposures WHERE client_id=? AND task=? AND "
                              "input_mode=?", (self.cfg.client_id, self.cfg.task.name, self.cfg.input.mode.value))
        return {r["family_id"] for r in rows if r["run_id"] != exclude_run}

    def create_snapshot(self, emitter: EventEmitter, seed: int | None = None) -> dict[str, Any]:
        seed = self.cfg.optimizer.seed if seed is None else seed
        mode, pipeline = self.cfg.input.mode, self.cfg.pipeline_id()
        stage = "snapshot"
        emitter.emit("snapshot.started", stage, "started", "Building immutable dataset snapshot",
                     input_mode=mode.value)
        ev_rows = self.store.all("SELECT document_id, family_id, evidence_hash FROM evidence WHERE client_id=? AND "
                                 "input_mode=? AND pipeline_id=? ORDER BY created_at, rowid",
                                 (self.cfg.client_id, mode.value, pipeline))
        latest: dict[str, tuple[str, str]] = {}
        for r in ev_rows:
            latest[r["document_id"]] = (r["family_id"], r["evidence_hash"])
        labels, _, conflicts = self.active_labels()
        for doc in conflicts:
            emitter.emit("label.conflict_excluded", stage, "skipped", "Conflicting verified labels excluded",
                         "WARNING", document_id=doc, reason_codes=["conflicting_verified_labels"])
        docs = sorted(d for d in latest if d in labels)
        if not docs:
            raise InputError("No documents have both current-pipeline evidence and verified labels", "empty_snapshot")
        exposed = self.exposed_families()
        family_of = self._merge_duplicate_content(docs, latest, emitter, stage)
        families: dict[str, list[str]] = defaultdict(list)
        for d in docs:
            families[family_of[d]].append(d)
        originals: dict[str, set[str]] = defaultdict(set)
        for d in docs:
            originals[family_of[d]].add(latest[d][0])
        exposed = {f for f, members in originals.items() if members & exposed or f in exposed}
        strata: dict[str, list[str]] = defaultdict(list)
        for fam, members in families.items():
            fam_labels = sorted({lbl for d in members for lbl in labels[d]}, key=self.task.labels.index)
            strata[fam_labels[0] if fam_labels else "__empty__"].append(fam)
        ratios = self.cfg.optimizer.splits
        assignment: dict[str, str] = {}
        for _, fams in sorted(strata.items()):
            fresh = sorted((f for f in fams if f not in exposed), key=lambda f: sha256_text(f"{seed}|{f}"))
            retired = sorted((f for f in fams if f in exposed), key=lambda f: sha256_text(f"{seed}|{f}"))
            for i, fam in enumerate(fresh):
                pos = (i + 0.5) / len(fresh)
                assignment[fam] = ("development" if pos < ratios.development else
                                   "selection" if pos < ratios.development + ratios.selection else "gate")
            dev_share = ratios.development / (ratios.development + ratios.selection)
            for i, fam in enumerate(retired):
                pos = (i + 0.5) / len(retired)
                assignment[fam] = "development" if pos < dev_share else "selection"
        items = [SnapshotItem(d, family_of[d], latest[d][1], labels[d], assignment[family_of[d]]) for d in docs]
        summary = self._summarize(items)
        summary["retired_gate_families"] = sum(1 for f in families if f in exposed)
        summary["excluded_conflicts"] = len(conflicts)
        summary["policy"] = {"seed": seed, "ratios": ratios.model_dump(), "grouping": "family_id",
                             "stratify": "first_label"}
        body = {"items": [i.__dict__ for i in items], "seed": seed, "pipeline": pipeline, "mode": mode.value}
        snapshot_id = "snap-" + sha256_text(canonical_json(body))[:16]
        with self.store.tx() as conn:
            exists = conn.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if not exists:
                conn.execute("INSERT INTO snapshots (snapshot_id, client_id, task, input_mode, pipeline_id, seed, "
                             "created_at, summary) VALUES (?,?,?,?,?,?,?,?)",
                             (snapshot_id, self.cfg.client_id, self.cfg.task.name, mode.value, pipeline, seed,
                              utc_now(), json.dumps(summary)))
                conn.executemany("INSERT INTO snapshot_items (snapshot_id, document_id, family_id, evidence_hash, "
                                 "labels, split) VALUES (?,?,?,?,?,?)",
                                 [(snapshot_id, i.document_id, i.family_id, i.evidence_hash, json.dumps(i.labels),
                                   i.split) for i in items])
        for split in SPLITS:
            missing = [lbl for lbl, n in summary["class_support"][split].items() if n == 0]
            if missing:
                emitter.emit("snapshot.support_warning", stage, "warning", "Labels without support in split",
                             "WARNING", split=split, reason_codes=[f"no_support:{m}" for m in missing])
        emitter.emit("snapshot.completed", stage, "completed", "Snapshot frozen", snapshot_id=snapshot_id,
                     counts=summary["counts"], input_mode=mode.value)
        return {"snapshot_id": snapshot_id, **summary}

    def _merge_duplicate_content(self, docs: list[str], latest: dict[str, tuple[str, str]], emitter: EventEmitter,
                                 stage: str) -> dict[str, str]:
        """Union families whose model-visible content is identical (exact duplicates under different names)."""
        parent = {latest[d][0]: latest[d][0] for d in docs}

        def find(f: str) -> str:
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f

        by_content: dict[str, str] = {}
        for d in docs:
            ev = self.load_evidence(latest[d][1])
            visible = {k: v for k, v in ev.fields.items() if k not in ("document_name", "file_name")}
            key = sha256_text(canonical_json({"t": [b.model_dump() for b in ev.text], "f": visible,
                                              "m": [m.sha256 for m in ev.media]}))
            fam = latest[d][0]
            if key in by_content:
                a, b = find(by_content[key]), find(fam)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                    emitter.emit("snapshot.duplicate_grouped", stage, "completed",
                                 "Identical content grouped into one family", "WARNING", document_id=d,
                                 family_id=min(a, b), reason_codes=["exact_duplicate_content"])
            else:
                by_content[key] = fam
        return {d: find(latest[d][0]) for d in docs}

    def _summarize(self, items: list[SnapshotItem]) -> dict[str, Any]:
        counts = {s: {"documents": 0, "families": 0} for s in SPLITS}
        fams: dict[str, set[str]] = {s: set() for s in SPLITS}
        support = {s: dict.fromkeys(self.task.labels, 0) for s in SPLITS}
        for i in items:
            counts[i.split]["documents"] += 1
            fams[i.split].add(i.family_id)
            for lbl in i.labels:
                support[i.split][lbl] += 1
        for s in SPLITS:
            counts[s]["families"] = len(fams[s])
        return {"counts": counts, "class_support": support}

    def snapshot_meta(self, snapshot_id: str) -> dict[str, Any]:
        row = self.store.one("SELECT * FROM snapshots WHERE snapshot_id=? AND client_id=? AND task=?",
                             (snapshot_id, self.cfg.client_id, self.cfg.task.name))
        if row is None:
            raise NotFound("Snapshot not found for this client/task", "snapshot_not_found")
        if row["input_mode"] != self.cfg.input.mode.value or row["pipeline_id"] != self.cfg.pipeline_id():
            raise InputError("Snapshot was built for a different input mode or evidence pipeline",
                             "snapshot_incompatible")
        return dict(row) | {"summary": json.loads(row["summary"])}

    def snapshot_items(self, snapshot_id: str, split: str | None = None) -> list[SnapshotItem]:
        self.snapshot_meta(snapshot_id)
        sql = "SELECT * FROM snapshot_items WHERE snapshot_id=?"
        params: tuple[Any, ...] = (snapshot_id,)
        if split:
            sql += " AND split=?"
            params = (snapshot_id, split)
        return [SnapshotItem(r["document_id"], r["family_id"], r["evidence_hash"], json.loads(r["labels"]), r["split"])
                for r in self.store.all(sql + " ORDER BY document_id", params)]

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.store.all("SELECT snapshot_id, input_mode, created_at, summary FROM snapshots WHERE client_id=? "
                              "AND task=? ORDER BY created_at", (self.cfg.client_id, self.cfg.task.name))
        return [dict(r) | {"summary": json.loads(r["summary"])} for r in rows]

    # --- optimizer exposure ledger (any role excludes a family from future final gates) -----------------------
    def record_exposure(self, snapshot_id: str, run_id: str, families: set[str], role: str) -> None:
        mode = self.cfg.input.mode.value
        with self.store.tx() as conn:
            conn.executemany("INSERT OR IGNORE INTO gate_exposures (client_id, task, input_mode, family_id, "
                             "snapshot_id, run_id, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                             [(self.cfg.client_id, self.cfg.task.name, mode, f, snapshot_id, run_id, role, utc_now())
                              for f in sorted(families)])

    def consume_gate(self, snapshot_id: str, run_id: str, families: set[str]) -> None:
        """Mark gate families consumed before any evaluation dispatch. Idempotent for the same run (resume)."""
        mode = self.cfg.input.mode.value
        with self.store.tx() as conn:
            rows = conn.execute(
                f"SELECT family_id, run_id FROM gate_exposures WHERE client_id=? AND task=? AND input_mode=? "  # noqa: S608
                f"AND family_id IN ({','.join('?' * len(families))})",
                (self.cfg.client_id, self.cfg.task.name, mode, *sorted(families))).fetchall()
            foreign = [r["family_id"] for r in rows if r["run_id"] != run_id]
            if foreign:
                raise InputError("Final-gate families were already exposed to a previous run", "gate_already_exposed")
            now = utc_now()
            conn.executemany("INSERT OR IGNORE INTO gate_exposures (client_id, task, input_mode, family_id, "
                             "snapshot_id, run_id, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                             [(self.cfg.client_id, self.cfg.task.name, mode, f, snapshot_id, run_id, "gate", now)
                              for f in sorted(families)])
