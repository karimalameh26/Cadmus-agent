"""SDK facade. All business behavior lives here; the CLI only parses arguments and prints results."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ag_agent.config import AppConfig
from ag_agent.contracts import (
    INFRA_STATUSES,
    PredictionStatus,
    PromptBundle,
    utc_now,
)
from ag_agent.datasets import DatasetService
from ag_agent.errors import AgAgentError, ConfigError, InputError, NotFound, ProviderError, RunCancelled
from ag_agent.evaluation import build_report
from ag_agent.inference import Budget, Classifier
from ag_agent.inputs import InputItem, document_id_for, family_id_for, make_input_adapter
from ag_agent.observability import EventEmitter, JsonlExporter, Subscriber, load_events, safe_error_code
from ag_agent.optimization import HeuristicProposer, ModelProposer, OptimizationCycle, Proposer, baseline_bundle
from ag_agent.policy import baseline_gates
from ag_agent.prompting import render_evidence
from ag_agent.providers import ModelAdapter, ModelRequest, ModelResponse, TextPart, make_model
from ag_agent.registry import Registry, diff_bundles
from ag_agent.storage import ArtifactStore, Store
from ag_agent.tasks import ClassificationTask, compute_metrics

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


class Framework:
    def __init__(self, cfg: AppConfig, *, model: ModelAdapter | None = None,
                 proposer_model: ModelAdapter | None = None, proposer: Proposer | None = None,
                 subscribers: Sequence[Subscriber] = (), sleep: Callable[[float], None] | None = None) -> None:
        self.cfg = cfg
        self.task = ClassificationTask(cfg.task)
        self.data_dir = cfg.data_dir
        self.store = Store(self.data_dir / "state.sqlite3")
        self.artifacts = ArtifactStore(self.data_dir, cfg.client_id)
        self.datasets = DatasetService(cfg, self.task, self.store, self.artifacts)
        self.registry = Registry(self.store, cfg.client_id)
        self._model = model
        self._proposer_model = proposer_model
        self._proposer = proposer
        self.subscribers = list(subscribers)
        self._sleep = sleep

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Framework:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- components -----------------------------------------------------------------------------------------
    @property
    def model(self) -> ModelAdapter:
        if self._model is None:
            self._model = make_model(self.cfg.provider, multilabel=self.task.multilabel)
        return self._model

    @property
    def compat_key(self) -> str:
        return self.cfg.compatibility_key()

    def _exporter(self, run_id: str) -> JsonlExporter:
        return JsonlExporter(self.data_dir / "logs" / self.cfg.client_id / f"{run_id}.jsonl",
                             self.cfg.logging.jsonl_max_bytes, self.cfg.logging.jsonl_backups)

    def _run_row(self, run_id: str) -> Any:
        row = self.store.one("SELECT * FROM runs WHERE run_id=? AND client_id=?", (run_id, self.cfg.client_id))
        if row is None:
            raise NotFound("Run not found for this client", "run_not_found")
        return row

    def _execute(self, kind: str, params: dict[str, Any], body: Callable[[str, EventEmitter], dict[str, Any]],
                 run_id: str | None = None) -> dict[str, Any]:
        now = utc_now()
        if run_id is None:
            run_id = f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            self.store.execute("INSERT INTO runs (run_id, client_id, kind, status, params, config_hash, created_at, "
                               "updated_at) VALUES (?,?,?,?,?,?,?,?)",
                               (run_id, self.cfg.client_id, kind, "running", json.dumps(params),
                                self.cfg.config_hash(), now, now))
        else:
            self.store.execute("UPDATE runs SET status='running', updated_at=? WHERE run_id=?", (now, run_id))
        emitter = EventEmitter(self.store, run_id, self.cfg.client_id, self._exporter(run_id), self.subscribers)
        emitter.emit("run.started", "run", "started", f"{kind} run started", config_hash=self.cfg.config_hash(),
                     input_mode=self.cfg.input.mode.value, pid=os.getpid())
        t0 = time.monotonic()
        status: str = "failed"
        summary: dict[str, Any] = {}
        error: BaseException | None = None
        try:
            summary = body(run_id, emitter)
            status = "completed"
        except (KeyboardInterrupt, RunCancelled):
            status, error = "cancelled", RunCancelled("Run cancelled by user")
        except AgAgentError as exc:
            error = exc
        except Exception as exc:  # unexpected defect: record safely, then re-raise for a traceback
            error = exc
        finally:
            duration = int((time.monotonic() - t0) * 1000)
            if error is not None:
                message = error.message if isinstance(error, AgAgentError) else "Unexpected error"
                summary = {"error": {"code": safe_error_code(error), "message": message}}
            self.store.execute("UPDATE runs SET status=?, updated_at=?, summary=? WHERE run_id=?",
                               (status, utc_now(), json.dumps(summary, default=str), run_id))
            with contextlib.suppress(AgAgentError):  # run status is already persisted
                emitter.emit("run.completed" if status == "completed" else f"run.{status}", "run", status,
                             f"{kind} run {status}", "INFO" if status == "completed" else "ERROR",
                             duration_ms=duration, error_code=safe_error_code(error) if error else None)
        if error is not None:
            raise error
        return {"run_id": run_id, "status": status, **summary}

    def _load(self, sources: Sequence[Path]) -> Iterator[InputItem]:
        return make_input_adapter(self.cfg, self.artifacts).load([Path(s) for s in sources])

    # --- configuration & inputs -----------------------------------------------------------------------------
    def validate_config(self) -> dict[str, Any]:
        warnings = []
        live = self.cfg.provider.kind != "fake" or self.cfg.proposer_provider.kind != "fake"
        if live and not self.cfg.policy.approved_for_live:
            warnings.append("policy.approved_for_live is false: optimize will refuse to run with a live provider")
        if not self.cfg.task.critical_labels:
            warnings.append("task.critical_labels is empty: no per-class non-regression gate will apply")
        if not self.cfg.reviewers:
            warnings.append("reviewers is empty: labels cannot be verified")
        if self.cfg.provider.kind == "fake":
            warnings.append("provider.kind is 'fake': results demonstrate orchestration, not real accuracy")
        return {"valid": True, "client_id": self.cfg.client_id, "input_mode": self.cfg.input.mode.value,
                "config_hash": self.cfg.config_hash(), "pipeline_id": self.cfg.pipeline_id(),
                "compatibility_key": self.compat_key, "provider": self.cfg.provider.kind, "warnings": warnings}

    def validate_inputs(self, sources: Sequence[Path]) -> dict[str, Any]:
        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            accepted, rejected = 0, []
            for item in self._load(sources):
                if item.ok:
                    accepted += 1
                    em.emit("document.validated", "input_validation", "accepted", "Input accepted",
                            document_id=item.ref, truncated=item.provenance.get("truncated"))
                else:
                    rejected.append({"ref": item.ref, "error_code": item.error_code})
                    em.emit("document.rejected", "input_validation", "rejected", "Input rejected", "WARNING",
                            document_id=item.ref, error_code=item.error_code)
            em.emit("inputs.summary", "input_validation", "completed", "Input validation finished",
                    accepted=accepted, rejected=len(rejected))
            return {"accepted": accepted, "rejected": rejected}
        return self._execute("inputs.validate", {"sources": [str(s) for s in sources]}, body)

    def preview(self, sources: Sequence[Path], limit: int = 3) -> list[dict[str, Any]]:
        """Explicit local preview of exactly what the model would see. Not logged."""
        out: list[dict[str, Any]] = []
        for item in self._load(sources):
            if len(out) >= limit:
                break
            if item.ok:
                assert item.evidence is not None
                out.append({"name": item.name, "document_id": item.ref, "model_visible_evidence":
                            render_evidence(item.evidence), "images": len(item.evidence.media),
                            "provenance": item.evidence.provenance})
            else:
                out.append({"name": item.name, "document_id": item.ref, "rejected": item.error_code})
        return out

    # --- datasets -------------------------------------------------------------------------------------------
    def _label_exists(self, document_id: str, labels: list[str]) -> bool:
        row = self.store.one("SELECT 1 FROM labels WHERE client_id=? AND task=? AND document_id=? AND labels=? AND "
                             "status IN ('verified','pending') AND superseded=0",
                             (self.cfg.client_id, self.cfg.task.name, document_id,
                              json.dumps(self.task.validate_labels(labels))))
        return row is not None

    def import_dataset(self, sources: Sequence[Path], verified_by: str | None = None) -> dict[str, Any]:
        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            accepted = rejected = labeled = 0
            for item in self._load(sources):
                if not item.ok:
                    rejected += 1
                    em.emit("document.rejected", "import", "rejected", "Input rejected", "WARNING",
                            document_id=item.ref, error_code=item.error_code)
                    continue
                assert item.evidence is not None
                digest = self.datasets.save_evidence(item.evidence)
                accepted += 1
                em.emit("document.imported", "import", "completed", "Evidence stored", document_id=item.ref,
                        family_id=item.evidence.family_id, evidence_hash=digest[:16],
                        pages_included=item.provenance.get("pages_included"),
                        pages_omitted=item.provenance.get("pages_omitted"), truncated=item.provenance.get("truncated"),
                        transforms=item.provenance.get("transforms"))
                if item.source_labels:
                    try:
                        if self._label_exists(item.ref, item.source_labels):
                            continue
                        label_id = self.datasets.add_label(item.ref, item.evidence.family_id, item.source_labels,
                                                           source="import", submitted_by=verified_by or "import",
                                                           verified_by=verified_by)
                        labeled += 1
                        em.emit("label.recorded", "label_verification", "verified" if verified_by else "pending",
                                "Label recorded", document_id=item.ref, label_id=label_id, reviewer=verified_by,
                                label_status="verified" if verified_by else "pending")
                    except InputError as exc:
                        em.emit("label.rejected", "label_verification", "rejected", "Source label invalid",
                                "WARNING", document_id=item.ref, error_code=exc.code)
            em.emit("import.summary", "import", "completed", "Import finished", accepted=accepted, rejected=rejected,
                    counts={"labels_recorded": labeled})
            return {"accepted": accepted, "rejected": rejected, "labels_recorded": labeled}
        return self._execute("dataset.import", {"sources": [str(s) for s in sources]}, body)

    def _family_for(self, document_id: str, name: str | None) -> str:
        row = self.store.one("SELECT family_id FROM evidence WHERE client_id=? AND document_id=? LIMIT 1",
                             (self.cfg.client_id, document_id))
        if row:
            return str(row["family_id"])
        if name is None:
            raise NotFound("Document has no stored evidence; reference it by source name", "document_unknown")
        return family_id_for(self.cfg.client_id, Path(name.replace("\\", "/")).stem)

    def _resolve_doc(self, ref: str) -> tuple[str, str | None]:
        return (ref, None) if ref.startswith("doc-") else (document_id_for(self.cfg.client_id, ref), ref)

    def import_labels(self, path: Path, verified_by: str | None = None) -> dict[str, Any]:
        """Import labels from a JSON array of {"name" | "document_id", "labels": [...]} objects."""
        try:
            rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            raise InputError("Label file could not be read as JSON", "malformed_json") from None
        if not isinstance(rows, list):
            raise InputError("Label file must be a JSON array", "shape_mismatch")

        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            recorded, rejected = 0, 0
            for i, row in enumerate(rows):
                ref = row.get("document_id") or row.get("name") if isinstance(row, dict) else None
                labels = row.get("labels") if isinstance(row, dict) else None
                if isinstance(labels, str):
                    labels = [labels]
                try:
                    if not isinstance(ref, str) or not isinstance(labels, list):
                        raise InputError("Row needs name/document_id and labels", "label_row_invalid")
                    doc, name = self._resolve_doc(ref)
                    if self._label_exists(doc, labels):
                        continue
                    label_id = self.datasets.add_label(doc, self._family_for(doc, name), labels, source="import",
                                                       submitted_by=verified_by or "import", verified_by=verified_by)
                    recorded += 1
                    em.emit("label.recorded", "label_verification", "verified" if verified_by else "pending",
                            "Label recorded", document_id=doc, label_id=label_id, reviewer=verified_by)
                except AgAgentError as exc:
                    rejected += 1
                    em.emit("label.rejected", "label_verification", "rejected", f"Label row {i} rejected", "WARNING",
                            error_code=exc.code)
            return {"labels_recorded": recorded, "rejected": rejected}
        return self._execute("labels.import", {"path": str(path)}, body)

    def snapshot(self, seed: int | None = None) -> dict[str, Any]:
        return self._execute("dataset.snapshot", {"seed": seed},
                             lambda run_id, em: self.datasets.create_snapshot(em, seed))

    def inspect_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        meta = self.datasets.snapshot_meta(snapshot_id)
        return {"snapshot_id": snapshot_id, "input_mode": meta["input_mode"], "created_at": meta["created_at"],
                **meta["summary"]}

    # --- corrections ----------------------------------------------------------------------------------------
    def add_correction(self, ref: str, labels: list[str], submitted_by: str) -> dict[str, Any]:
        doc, name = self._resolve_doc(ref)

        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            label_id = self.datasets.add_label(doc, self._family_for(doc, name), labels, source="correction",
                                               submitted_by=submitted_by)
            em.emit("label.submitted", "label_verification", "pending", "Correction pending verification",
                    document_id=doc, label_id=label_id, label_status="pending")
            return {"label_id": label_id, "document_id": doc, "status": "pending"}
        return self._execute("corrections.add", {"document_id": doc}, body)

    def verify_correction(self, label_id: int, reviewer: str) -> dict[str, Any]:
        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            out = self.datasets.verify(label_id, reviewer)
            em.emit("label.verified", "label_verification", "verified", "Label verified", label_id=label_id,
                    reviewer=reviewer, document_id=out["document_id"], label_status="verified")
            return out
        return self._execute("corrections.verify", {"label_id": label_id}, body)

    def reject_correction(self, label_id: int, reviewer: str, reason: str) -> dict[str, Any]:
        def body(run_id: str, em: EventEmitter) -> dict[str, Any]:
            out = self.datasets.reject(label_id, reviewer, reason)
            em.emit("label.rejected", "label_verification", "rejected", "Label rejected", label_id=label_id,
                    reviewer=reviewer, document_id=out["document_id"], label_status="rejected")
            return out
        return self._execute("corrections.reject", {"label_id": label_id}, body)

    def list_corrections(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.datasets.list_labels(status)

    # --- classification & evaluation ------------------------------------------------------------------------
    def _bundle_for_inference(self, bundle_id: str | None) -> PromptBundle:
        if bundle_id:
            bundle = self.registry.get(bundle_id)
            if bundle.compatibility_key != self.compat_key:
                raise InputError("Bundle is not compatible with the current configuration", "bundle_incompatible")
            return bundle
        champion = self.registry.champion(self.compat_key)
        if champion is not None:
            return champion
        bundle = baseline_bundle(self.cfg)
        self.registry.save(bundle)
        return bundle

    def _classifier(self, em: EventEmitter, budget: Budget) -> Classifier:
        return Classifier(self.cfg, self.task, self.model, self.store, self.artifacts, em, budget, sleep=self._sleep)

    def classify(self, sources: Sequence[Path], bundle_id: str | None = None,
                 run_id: str | None = None) -> dict[str, Any]:
        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            bundle = self._bundle_for_inference(bundle_id)
            is_champion = bundle.bundle_id == self.registry.champion_id(self.compat_key)
            em.emit("bundle.resolved", "request_preparation", "completed",
                    "Using champion bundle" if is_champion else "Using non-champion bundle",
                    "INFO" if is_champion else "WARNING", bundle_id=bundle.bundle_id, compat_key=self.compat_key,
                    reason_codes=[] if is_champion else ["no_champion_using_baseline" if not bundle_id
                                                         else "explicit_bundle"])
            evidences, names, rejected = [], {}, []
            for item in self._load(sources):
                if item.ok:
                    assert item.evidence is not None
                    self.datasets.save_evidence(item.evidence)
                    evidences.append(item.evidence)
                    names[item.ref] = item.name
                else:
                    rejected.append({"name": item.name, "document_id": item.ref, "error_code": item.error_code})
                    em.emit("document.rejected", "input_discovery", "rejected", "Input rejected", "WARNING",
                            document_id=item.ref, error_code=item.error_code)
            em.emit("inputs.summary", "input_discovery", "completed", "Inputs loaded", accepted=len(evidences),
                    rejected=len(rejected))
            budget = Budget(self.cfg.provider.pricing)
            preds = self._classifier(em, budget).run(rid, bundle, evidences, purpose="production", stage="inference")
            results = [{"name": names.get(p.document_id), **p.model_dump(mode="json", exclude={"evidence_hash"})}
                       for p in preds]
            path = self.artifacts.write_run_json(rid, "predictions.json", {"results": results, "rejected": rejected})
            counts: dict[str, int] = {}
            for p in preds:
                counts[p.status.value] = counts.get(p.status.value, 0) + 1
            return {"bundle_id": bundle.bundle_id, "counts": counts, "rejected": rejected, "results": results,
                    "results_file": str(path)}
        return self._execute("classify", {"sources": [str(s) for s in sources], "bundle_id": bundle_id}, body, run_id)

    def evaluate(self, snapshot_id: str, split: str = "development", bundle_id: str | None = None,
                 run_id: str | None = None) -> dict[str, Any]:
        if split not in ("development", "selection", "gate"):
            raise InputError("split must be development, selection or gate", "invalid_split")

        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            bundle = self._bundle_for_inference(bundle_id)
            items = self.datasets.snapshot_items(snapshot_id, split)
            if not items:
                raise InputError("Split is empty", "empty_split")
            if split == "gate":
                self.datasets.consume_gate(snapshot_id, rid, {i.family_id for i in items})
                em.emit("gate.consumed", "evaluation", "completed", "Manual gate evaluation consumes the gate batch",
                        "WARNING", snapshot_id=snapshot_id, total=len({i.family_id for i in items}))
            evs = [self.datasets.load_evidence(i.evidence_hash) for i in items]
            em.emit("evaluation.started", "evaluation", "started", "Evaluation started", snapshot_id=snapshot_id,
                    split=split, bundle_id=bundle.bundle_id, total=len(items))
            preds = self._classifier(em, Budget(self.cfg.provider.pricing)).run(
                rid, bundle, evs, purpose=f"evaluate:{split}", stage="evaluation")
            report = build_report(self.task, bundle.bundle_id, split, items, preds)
            by = {p.document_id: p for p in preds}
            records = [{"document_id": i.document_id, "truth": i.labels, "status": by[i.document_id].status.value,
                        "predicted": by[i.document_id].labels, "error_code": by[i.document_id].error_code,
                        "correct": by[i.document_id].status == PredictionStatus.SUCCESS
                        and set(by[i.document_id].labels) == set(i.labels)} for i in items]
            self.artifacts.write_run_json(rid, "evaluation.json", {"report": report.model_dump(mode="json"),
                                                                   "records": records})
            em.emit("evaluation.completed", "evaluation", "completed", "Evaluation aggregated", split=split,
                    total=report.expected, completed=report.completed,
                    metrics={"macro_f1": round(report.macro_f1, 4), "micro_f1": round(report.micro_f1, 4),
                             "coverage": round(report.coverage, 4)})
            return {"report": report.model_dump(mode="json"), "failures": [r for r in records if not r["correct"]]}
        return self._execute("evaluate", {"snapshot_id": snapshot_id, "split": split, "bundle_id": bundle_id}, body,
                             run_id)

    # --- optimization ---------------------------------------------------------------------------------------
    def _proposer_for(self, em: EventEmitter, budget: Budget, run_id: str) -> Proposer:
        if self._proposer is not None:
            return self._proposer
        pcfg = self.cfg.proposer_provider
        if self.cfg.optimizer.proposer == "heuristic":
            return HeuristicProposer()
        if pcfg.kind == "fake":
            em.emit("proposer.selected", "candidate_generation", "completed",
                    "Fake provider: using deterministic heuristic proposer", reason_codes=["fake_provider"])
            return HeuristicProposer()
        model = self._proposer_model or make_model(pcfg)

        def invoke(req: ModelRequest) -> ModelResponse:
            for attempt in range(1, pcfg.max_attempts + 1):
                res = budget.reserve(req.text_chars() // 4, req.max_output_tokens, pcfg.pricing)
                request_id = "req-" + uuid.uuid4().hex[:16]
                self.store.execute("INSERT INTO requests (request_id, run_id, cache_key, attempt, status, started_at, "
                                   "reserved_cost) VALUES (?,?,?,?,?,?,?)",
                                   (request_id, run_id, "proposer", attempt, "dispatched", utc_now(),
                                    res.estimated_cost))
                em.emit("provider.attempt.started", "candidate_generation", "started", "Proposer request dispatched",
                        attempt=attempt, purpose="proposer")
                t0 = time.monotonic()
                try:
                    resp = model.complete(req)
                except ProviderError as exc:
                    budget.settle(res, None, None, charge_if_unknown=exc.completion_unknown)
                    self.store.execute("UPDATE requests SET status=?, ended_at=?, error_code=? WHERE request_id=?",
                                       ("unknown" if exc.completion_unknown else "failed", utc_now(), exc.code,
                                        request_id))
                    em.emit("provider.attempt.failed", "candidate_generation", "failed", "Proposer request failed",
                            "WARNING", attempt=attempt, error_code=exc.code, purpose="proposer")
                    if exc.transient and attempt < pcfg.max_attempts:
                        (self._sleep or time.sleep)(pcfg.backoff_base_s * 2 ** (attempt - 1))
                        continue
                    return ModelResponse(text="")
                cost = budget.settle(res, resp.input_tokens, resp.output_tokens, charge_if_unknown=True)
                self.store.execute("UPDATE requests SET status='completed', ended_at=?, input_tokens=?, "
                                   "output_tokens=?, actual_cost=? WHERE request_id=?",
                                   (utc_now(), resp.input_tokens, resp.output_tokens, cost, request_id))
                em.emit("provider.attempt.completed", "candidate_generation", "completed", "Proposer responded",
                        attempt=attempt, purpose="proposer", input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens, duration_ms=int((time.monotonic() - t0) * 1000))
                return resp
            return ModelResponse(text="")
        return ModelProposer(invoke, self.cfg.optimizer.max_failures_shown)

    def optimize(self, snapshot_id: str, run_id: str | None = None) -> dict[str, Any]:
        live = self.cfg.provider.kind != "fake" or self.cfg.proposer_provider.kind != "fake"
        if live and not self.cfg.policy.approved_for_live:
            raise ConfigError("Automatic promotion with a live provider requires policy.approved_for_live = true "
                              "after setting thresholds, critical labels and budgets", "policy_not_approved_for_live")
        if run_id is not None:
            done = self.store.one("SELECT outcome FROM decisions WHERE run_id=?", (run_id,))
            if done is not None:
                raise InputError("Run already reached a decision; start a new cycle", "run_already_decided")

        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            opt = self.cfg.optimizer
            budget = Budget(self.cfg.provider.pricing, opt.max_requests, opt.max_cost_usd, opt.max_elapsed_s)
            cycle = OptimizationCycle(self.cfg, self.task, self._classifier(em, budget), self.datasets, self.registry,
                                      self.artifacts, em, self._proposer_for(em, budget, rid), rid)
            decision = cycle.run(snapshot_id)
            return {"decision": decision.model_dump(mode="json"),
                    "champion": self.registry.champion_id(self.compat_key),
                    "budget": {"requests": budget.requests, "spent_usd_estimate": round(budget.spent, 6),
                               "unknown_usage": budget.unknown_usage}}
        return self._execute("optimize", {"snapshot_id": snapshot_id}, body, run_id)

    def register_baseline(self, snapshot_id: str) -> dict[str, Any]:
        """Evaluate the configured baseline on development+selection data; it becomes champion only if absolute
        gates pass and no champion exists yet."""
        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            bundle = baseline_bundle(self.cfg)
            self.registry.save(bundle)
            items = [i for i in self.datasets.snapshot_items(snapshot_id) if i.split != "gate"]
            evs = [self.datasets.load_evidence(i.evidence_hash) for i in items]
            preds = self._classifier(em, Budget(self.cfg.provider.pricing)).run(
                rid, bundle, evs, purpose="baseline", stage="evaluation")
            report = build_report(self.task, bundle.bundle_id, "development+selection", items, preds)
            gates = baseline_gates(self.cfg.policy, report)
            for g in gates:
                em.emit("gate.evaluated", "promotion", "passed" if g.passed else "failed", f"Baseline gate {g.name}",
                        "INFO" if g.passed else "WARNING", gate=g.name, passed=g.passed, observed=g.observed,
                        threshold=g.threshold)
            promoted = False
            if all(g.passed for g in gates):
                promoted = self.registry.promote(self.compat_key, None, bundle, run_id=rid, action="baseline",
                                                 reason="baseline_gates_passed",
                                                 reference_score=report.metric(self.cfg.policy.primary_metric))
            em.emit("promotion.transaction", "promotion", "completed" if promoted else "skipped",
                    "Baseline registered as champion" if promoted else "Baseline not registered as champion",
                    bundle_id=bundle.bundle_id, outcome="promoted" if promoted else "not_promoted")
            return {"bundle_id": bundle.bundle_id, "promoted": promoted, "gates": [g.model_dump() for g in gates],
                    "report": report.model_dump(mode="json")}
        return self._execute("prompts.baseline", {"snapshot_id": snapshot_id}, body)

    # --- prompts --------------------------------------------------------------------------------------------
    def list_prompts(self) -> list[dict[str, Any]]:
        return self.registry.list_bundles(self.compat_key)

    def show_prompt(self, bundle_id: str | None = None) -> dict[str, Any]:
        bundle = self.registry.get(bundle_id) if bundle_id else self._bundle_for_inference(None)
        return bundle.model_dump(mode="json") | {
            "champion": bundle.bundle_id == self.registry.champion_id(self.compat_key),
            "history": self.registry.history(self.compat_key)}

    def diff_prompts(self, a: str, b: str) -> str:
        return diff_bundles(self.registry.get(a), self.registry.get(b))

    def rollback(self, target: str | None = None, reason: str = "manual") -> dict[str, Any]:
        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            current = self.registry.champion_id(self.compat_key)
            bundle = self.registry.rollback_target(self.compat_key, target)
            ok = self.registry.promote(self.compat_key, current, bundle, run_id=rid, action="rollback", reason=reason,
                                       reference_score=self.registry.reference_score(self.compat_key, bundle.bundle_id))
            em.emit("rollback.transaction", "rollback", "completed" if ok else "failed", "Rollback applied" if ok
                    else "Champion changed concurrently; rollback not applied", from_bundle=current,
                    to_bundle=bundle.bundle_id, compat_key=self.compat_key)
            return {"rolled_back": ok, "from_bundle": current, "to_bundle": bundle.bundle_id}
        return self._execute("prompts.rollback", {"target": target}, body)

    # --- monitoring -----------------------------------------------------------------------------------------
    def monitor_evaluate(self, apply: bool | None = None) -> dict[str, Any]:
        mcfg = self.cfg.monitor
        apply = mcfg.auto_rollback if apply is None else apply

        def body(rid: str, em: EventEmitter) -> dict[str, Any]:
            champ = self.registry.champion_id(self.compat_key)
            if champ is None:
                em.emit("monitor.skipped", "monitoring", "skipped", "No champion", reason_codes=["no_champion"])
                return {"status": "no_champion"}
            cutoff = (datetime.now(UTC) - timedelta(hours=mcfg.window_hours)).isoformat(
                timespec="milliseconds").replace("+00:00", "Z")
            rows = self.store.all(
                "SELECT rp.document_id, rp.payload FROM run_predictions rp JOIN runs r ON r.run_id = rp.run_id "
                "WHERE r.client_id=? AND rp.purpose='production' AND rp.bundle_id=? AND rp.created_at >= ? "
                "ORDER BY rp.created_at", (self.cfg.client_id, champ, cutoff))
            latest = {r["document_id"]: json.loads(r["payload"]) for r in rows}
            labels, _, _ = self.datasets.active_labels()
            from ag_agent.contracts import Prediction
            preds = {d: Prediction.model_validate(p) for d, p in latest.items()}
            infra = [d for d, p in preds.items() if p.status in INFRA_STATUSES]
            eligible = [(labels[d], p) for d, p in preds.items() if d in labels and p.status not in INFRA_STATUSES]
            invalid = sum(p.status == PredictionStatus.INVALID_RESPONSE for _, p in eligible)
            result: dict[str, Any] = {"champion": champ, "window_hours": mcfg.window_hours,
                                      "production_predictions": len(preds), "labeled_samples": len(eligible),
                                      "excluded_infra_failures": len(infra), "triggers": [], "rolled_back": False}
            em.emit("monitor.window", "monitoring", "completed", "Monitoring window collected", bundle_id=champ,
                    total=len(preds), completed=len(eligible), skipped=len(infra))
            if len(eligible) < mcfg.min_samples:
                result["status"] = "insufficient_samples"
                em.emit("monitor.decision", "monitoring", "skipped", "Not enough verified samples",
                        observed=len(eligible), threshold=mcfg.min_samples, reason_codes=["insufficient_samples"])
                return result
            metric = self.cfg.policy.primary_metric
            m = compute_metrics(self.task, eligible)
            score = float(m.get(metric, m.get("accuracy", m.get("exact_match", 0.0))))
            invalid_rate = invalid / len(eligible)
            reference = self.registry.reference_score(self.compat_key, champ)
            result.update(score=score, reference_score=reference, invalid_rate=invalid_rate)
            if reference is not None and reference - score > mcfg.max_drop:
                result["triggers"].append("quality_regression")
            if invalid_rate > mcfg.max_invalid_rate:
                result["triggers"].append("schema_regression")
            last = self.registry.last_change(self.compat_key)
            cooling = bool(last and datetime.fromisoformat(last["created_at"].replace("Z", "+00:00")) >
                           datetime.now(UTC) - timedelta(hours=mcfg.cooldown_hours))
            em.emit("monitor.evaluated", "monitoring", "completed", "Monitoring metrics computed",
                    score=round(score, 4), observed=round(invalid_rate, 4),
                    reason_codes=result["triggers"] or ["no_regression"])
            if not result["triggers"]:
                result["status"] = "healthy"
                return result
            if cooling:
                result["status"] = "cooldown_active"
                em.emit("monitor.decision", "monitoring", "skipped", "Regression detected; rollback cooldown active",
                        "WARNING", reason_codes=["cooldown_active", *result["triggers"]])
                return result
            if not apply:
                result["status"] = "regression_detected"
                em.emit("monitor.decision", "monitoring", "skipped", "Regression detected; rollback not applied",
                        "WARNING", reason_codes=result["triggers"])
                return result
            try:
                target = self.registry.rollback_target(self.compat_key, None)
            except AgAgentError as exc:
                result["status"] = "regression_no_target"
                em.emit("monitor.decision", "monitoring", "failed", "No compatible rollback target", "WARNING",
                        error_code=exc.code)
                return result
            ok = self.registry.promote(self.compat_key, champ, target, run_id=rid, action="rollback",
                                       reason=",".join(result["triggers"]),
                                       reference_score=self.registry.reference_score(self.compat_key, target.bundle_id))
            result.update(status="rolled_back" if ok else "stale_incumbent", rolled_back=ok, to_bundle=target.bundle_id)
            em.emit("rollback.transaction", "rollback", "completed" if ok else "failed", "Automatic rollback",
                    "WARNING", from_bundle=champ, to_bundle=target.bundle_id, reason_codes=result["triggers"])
            return result
        return self._execute("monitor", {"apply": apply}, body)

    # --- runs -----------------------------------------------------------------------------------------------
    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.all("SELECT run_id, kind, status, created_at, updated_at FROM runs WHERE client_id=? "
                              "ORDER BY created_at DESC LIMIT ?", (self.cfg.client_id, limit))
        stale_before = (datetime.now(UTC) - timedelta(minutes=10)).isoformat(timespec="milliseconds")
        out = []
        for r in rows:
            d = dict(r)
            if d["status"] == "running" and d["updated_at"].replace("Z", "+00:00") < stale_before:
                d["status_detail"] = "no recent events; possibly interrupted (use 'runs resume')"
            out.append(d)
        return out

    def show_run(self, run_id: str) -> dict[str, Any]:
        row = self._run_row(run_id)
        events = load_events(self.store, run_id)
        stages: dict[str, dict[str, int]] = {}
        for e in events:
            s = stages.setdefault(e["stage"], {})
            s[e["status"]] = s.get(e["status"], 0) + 1
        decision = self.store.one("SELECT payload FROM decisions WHERE run_id=?", (run_id,))
        return {**{k: row[k] for k in ("run_id", "kind", "status", "created_at", "updated_at", "config_hash")},
                "params": json.loads(row["params"]), "summary": json.loads(row["summary"] or "{}"),
                "events": len(events), "stages": stages,
                "decision": json.loads(decision["payload"]) if decision else None}

    def run_logs(self, run_id: str, *, stage: str | None = None, level: str = "DEBUG", event_type: str | None = None,
                 document_id: str | None = None, candidate_id: str | None = None, follow: bool = False,
                 poll_s: float = 1.0) -> Iterator[dict[str, Any]]:
        from ag_agent.observability import LEVELS
        self._run_row(run_id)
        last = 0
        while True:
            for e in load_events(self.store, run_id, last):
                last = e["sequence"]
                if stage and e["stage"] != stage or LEVELS.get(e["level"], 20) < LEVELS[level]:
                    continue
                if event_type and not e["event_type"].startswith(event_type):
                    continue
                if document_id and e.get("document_id") != document_id:
                    continue
                if candidate_id and e.get("candidate_id") != candidate_id:
                    continue
                yield e
            if not follow or self._run_row(run_id)["status"] in TERMINAL_RUN_STATUSES:
                if follow:
                    for e in load_events(self.store, run_id, last):
                        yield e
                return
            time.sleep(poll_s)

    def run_report(self, run_id: str) -> dict[str, Any]:
        info = self.show_run(run_id)
        run_dir = self.artifacts.runs_root / run_id
        info["artifacts"] = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()) \
            if run_dir.exists() else []
        requests = self.store.one("SELECT COUNT(*) AS n, SUM(input_tokens) AS i, SUM(output_tokens) AS o, "
                                  "SUM(COALESCE(actual_cost, reserved_cost)) AS c, "
                                  "SUM(CASE WHEN status='unknown' THEN 1 ELSE 0 END) AS u, "
                                  "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS f FROM requests WHERE run_id=?",
                                  (run_id,))
        assert requests is not None
        info["requests"] = {"attempts": requests["n"], "input_tokens": requests["i"], "output_tokens": requests["o"],
                            "cost_usd_estimate": requests["c"], "unknown_completion": requests["u"],
                            "failed": requests["f"]}
        failures = [e for e in load_events(self.store, run_id) if e["level"] in ("WARNING", "ERROR")]
        info["warnings_and_errors"] = failures[-50:]
        return info

    def resume(self, run_id: str) -> dict[str, Any]:
        row = self._run_row(run_id)
        if row["status"] == "completed":
            raise InputError("Run already completed", "run_completed")
        if row["config_hash"] != self.cfg.config_hash():
            raise ConfigError("Configuration changed since the run started; start a new run",
                              "config_changed_requires_new_run")
        params = json.loads(row["params"])
        em = EventEmitter(self.store, run_id, self.cfg.client_id, self._exporter(run_id), self.subscribers)
        em.emit("run.resumed", "run", "started", "Resuming run; completed work is reused from checkpoints",
                status_detail=row["status"])
        stale = self.store.one("SELECT COUNT(*) AS n FROM requests WHERE run_id=? AND status='dispatched'", (run_id,))
        if stale and stale["n"]:
            self.store.execute("UPDATE requests SET status='unknown', error_code='interrupted' WHERE run_id=? AND "
                               "status='dispatched'", (run_id,))
            em.emit("request.reconciled", "run", "unknown", "In-flight requests marked unknown (possible charge)",
                    "WARNING", total=stale["n"])
        kind = row["kind"]
        if kind == "optimize":
            return self.optimize(params["snapshot_id"], run_id=run_id)
        if kind == "classify":
            return self.classify([Path(s) for s in params["sources"]], params.get("bundle_id"), run_id=run_id)
        if kind == "evaluate":
            return self.evaluate(params["snapshot_id"], params["split"], params.get("bundle_id"), run_id=run_id)
        raise InputError(f"Runs of kind '{kind}' are not resumable; rerun the command", "not_resumable")

    def prune(self) -> dict[str, Any]:
        """Apply logging.retention_days to JSONL exports and run artifacts. Database audit records are kept."""
        days = self.cfg.logging.retention_days
        if days is None:
            return {"pruned": 0, "detail": "logging.retention_days not set"}
        cutoff = time.time() - days * 86400
        pruned = 0
        for root in (self.data_dir / "logs" / self.cfg.client_id, self.artifacts.runs_root):
            if not root.exists():
                continue
            for p in root.iterdir():
                if p.stat().st_mtime < cutoff:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                    pruned += 1
        return {"pruned": pruned, "retention_days": days}

    # --- diagnostics ----------------------------------------------------------------------------------------
    def doctor(self, check_connectivity: bool = False) -> dict[str, Any]:
        checks: dict[str, Any] = {"config": "ok", "schema_version": self.store.schema_version}
        try:
            probe = self.data_dir / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks["storage_writable"] = True
        except OSError:
            checks["storage_writable"] = False
        deps = {}
        for mod in ("PIL", "pypdf", "pypdfium2"):
            try:
                __import__(mod)
                deps[mod] = True
            except ImportError:
                deps[mod] = False
        checks["optional_dependencies"] = deps
        if self.cfg.input.mode.value == "direct_files" and not all(deps.values()):
            checks["warning"] = "direct_files mode needs 'pip install ag-agent[files]' for images/PDFs"
        p = self.cfg.provider
        checks["provider"] = {"kind": p.kind, "model": p.model, "supports_images": p.supports_images}
        if p.kind != "fake":
            checks["provider"]["credential_present"] = bool(os.environ.get(p.api_key_env))
        checks["live_promotion_approved"] = self.cfg.policy.approved_for_live
        if check_connectivity:
            try:
                resp = self.model.complete(ModelRequest(system="Reply with the JSON object {\"ok\": true}.",
                                                        parts=[TextPart("connectivity check")], max_output_tokens=20))
                checks["connectivity"] = {"ok": True, "input_tokens": resp.input_tokens}
            except ProviderError as exc:
                checks["connectivity"] = {"ok": False, "error_code": exc.code}
        else:
            checks["connectivity"] = "not_run (use --check-connectivity)"
        checks["ok"] = bool(checks["storage_writable"])
        if self.cfg.input.mode.value == "direct_files":
            checks["ok"] = checks["ok"] and all(deps.values())
        if p.kind != "fake":
            checks["ok"] = checks["ok"] and checks["provider"]["credential_present"]
        if check_connectivity:
            checks["ok"] = checks["ok"] and checks["connectivity"]["ok"]
        return checks
