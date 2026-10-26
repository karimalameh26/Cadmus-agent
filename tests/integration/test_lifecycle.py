from __future__ import annotations

import json
from pathlib import Path

import pytest

from ag_agent.contracts import PromptBundle
from ag_agent.errors import AuditFailure, ProviderError
from ag_agent.observability import EventEmitter
from ag_agent.optimization import DevelopmentView, HeuristicProposer, Proposal
from ag_agent.providers import FakeModel
from tests.conftest import write_records

SECRET = "zzsecretpayloadzz"


def setup_snapshot(fw, records: Path) -> str:
    fw.import_dataset([records], verified_by="rev-1")
    return fw.snapshot()["snapshot_id"]


def test_full_lifecycle_promote_then_reject_on_fresh_data(make_fw, records: Path, tmp_path: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    first = fw.optimize(snap)
    assert first["decision"]["outcome"] == "promoted"
    champion = fw.registry.champion_id(fw.compat_key)
    assert champion == first["decision"]["finalist_bundle_id"]

    fw.import_dataset([write_records(tmp_path / "b2.json", batch=2)], verified_by="rev-1")
    snap2 = fw.snapshot()
    assert snap2["retired_gate_families"] == 90
    second = fw.optimize(snap2["snapshot_id"])
    assert second["decision"]["outcome"] in ("rejected", "promoted", "insufficient_evidence")
    if second["decision"]["outcome"] != "promoted":
        assert fw.registry.champion_id(fw.compat_key) == champion


def test_reusing_exposed_data_cannot_be_a_gate(make_fw, records: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    fw.optimize(snap)
    again = fw.optimize(snap)
    assert again["decision"]["outcome"] == "insufficient_evidence"
    assert again["decision"]["reason_codes"] == ["gate_already_exposed"]
    fresh = fw.snapshot(seed=99)
    assert fresh["counts"]["gate"]["documents"] == 0


class RecordingProposer:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.inner = HeuristicProposer()

    def propose(self, view: DevelopmentView, n: int, round_index: int) -> list[Proposal]:
        self.seen |= {i.document_id for i in view.items}
        return self.inner.propose(view, n, round_index)


def test_proposer_only_sees_development_data(make_fw, records: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    rec = RecordingProposer()
    fw._proposer = rec
    fw.optimize(snap)
    dev = {i.document_id for i in fw.datasets.snapshot_items(snap, "development")}
    assert rec.seen and rec.seen <= dev


class MemorizingProposer:
    """Overfit trap: rules keyed on exact development OCR strings are perfect on development, useless elsewhere."""

    def propose(self, view: DevelopmentView, n: int, round_index: int) -> list[Proposal]:
        rules = [f'- If the evidence mentions "{f.evidence.text[0].text.lower()}", classify as "{f.truth[0]}".'
                 for f in view.failures[:20]]
        return [Proposal(guidance=view.guidance + "\n" + "\n".join(rules))]


def test_development_overfit_is_not_promoted(make_fw, records: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    fw._proposer = MemorizingProposer()
    out = fw.optimize(snap)
    assert out["decision"]["outcome"] in ("rejected", "insufficient_evidence")
    assert fw.registry.champion_id(fw.compat_key) is None


class OutsideExampleProposer:
    def propose(self, view: DevelopmentView, n: int, round_index: int) -> list[Proposal]:
        return [Proposal(guidance="x", example_ids=["doc-not-in-dev"])]


def test_examples_outside_development_rejected(make_fw, records: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    fw._proposer = OutsideExampleProposer()
    out = fw.optimize(snap)
    events = list(fw.run_logs(out["run_id"], event_type="candidate.rejected"))
    assert events and events[0]["reason_codes"] == ["example_not_in_development"]
    assert out["decision"]["reason_codes"] == ["no_development_improvement"]


def test_budget_exhaustion_preserves_champion(make_fw, records: Path) -> None:
    fw = make_fw(optimizer={"max_requests": 60})
    snap = setup_snapshot(fw, records)
    out = fw.optimize(snap)
    assert out["decision"]["outcome"] == "budget_exhausted"
    assert out["decision"]["reason_codes"] == ["budget_requests"]
    assert fw.registry.champion_id(fw.compat_key) is None


def test_provider_outage_blocks_promotion_as_insufficient(make_fw, records: Path) -> None:
    err = ProviderError("down", "http_503", transient=True)
    model = FakeModel(fail_on={"cashier": err})
    fw = make_fw(model=model)
    snap = setup_snapshot(fw, records)
    out = fw.optimize(snap)
    assert out["decision"]["outcome"] in ("insufficient_evidence", "rejected")
    if out["decision"]["outcome"] == "insufficient_evidence":
        assert "incomplete_paired_comparison" in out["decision"]["reason_codes"]
    retries = list(fw.run_logs(out["run_id"], event_type="provider.attempt.failed"))
    assert any("retry_delay_ms" in e for e in retries)


class HijackingProposer(HeuristicProposer):
    def __init__(self, fw) -> None:
        self.fw = fw
        self.done = False

    def propose(self, view, n, round_index):
        if not self.done:
            self.done = True
            other = PromptBundle.create(compatibility_key=self.fw.compat_key, guidance="concurrent winner")
            assert self.fw.registry.promote(self.fw.compat_key, None, other, run_id="other", action="promote",
                                            reason="test", reference_score=None)
        return super().propose(view, n, round_index)


def test_concurrent_promotion_yields_stale_incumbent(make_fw, records: Path) -> None:
    fw = make_fw()
    snap = setup_snapshot(fw, records)
    fw._proposer = HijackingProposer(fw)
    out = fw.optimize(snap)
    assert out["decision"]["outcome"] == "stale_incumbent"
    assert fw.show_prompt()["guidance"] == "concurrent winner"


def test_interrupted_run_resumes_reusing_work(make_fw, records: Path) -> None:
    calls = {"n": 0}

    def hook(_req):
        calls["n"] += 1
        if calls["n"] == 120:
            raise KeyboardInterrupt

    model = FakeModel(hook=hook)
    fw = make_fw(model=model, provider={"kind": "fake", "max_concurrency": 1})
    snap = setup_snapshot(fw, records)
    with pytest.raises(Exception) as exc:
        fw.optimize(snap)
    assert getattr(exc.value, "code", "") == "cancelled"
    run_id = fw.list_runs()[0]["run_id"]
    assert fw.show_run(run_id)["status"] == "cancelled"
    before = calls["n"]
    resumed = fw.resume(run_id)
    assert resumed["decision"]["outcome"] == "promoted"
    fresh_calls = calls["n"] - before
    full = make_fw(model=FakeModel(), data_dir=str(Path(fw.cfg.data_dir).parent / "other"))
    full_snap = setup_snapshot(full, records)
    full.optimize(full_snap)
    assert fresh_calls < full.model.calls
    with pytest.raises(Exception):
        fw.resume(run_id)


def test_monitor_rolls_back_on_verified_regression_but_not_on_outage(make_fw, records: Path, tmp_path) -> None:
    fw = make_fw()
    good = PromptBundle.create(compatibility_key=fw.compat_key, guidance="\n".join(
        f'- If the evidence mentions "{w}", classify as "{lbl}".' for lbl, ws in
        {"invoice": ["invoice", "remittance", "payable", "billing"], "receipt": ["receipt", "cashier", "change",
         "purchase"], "contract": ["agreement", "hereby", "termination", "parties"]}.items() for w in ws))
    bad = PromptBundle.create(compatibility_key=fw.compat_key,
                              guidance='- If the evidence mentions "date", classify as "receipt".')
    assert fw.registry.promote(fw.compat_key, None, good, run_id="t", action="promote", reason="t", reference_score=1.0)
    assert fw.registry.promote(fw.compat_key, good.bundle_id, bad, run_id="t", action="promote", reason="t",
                               reference_score=1.0)
    fw.import_dataset([records], verified_by="rev-1")

    outage = make_fw(model=FakeModel(fail_on={"": ProviderError("down", "http_503")}))
    outage.classify([records])
    held = outage.monitor_evaluate()
    assert held["status"] == "insufficient_samples" and held["excluded_infra_failures"] == 90
    assert outage.registry.champion_id(outage.compat_key) == bad.bundle_id

    fw.classify([records])
    result = fw.monitor_evaluate()
    assert result["status"] == "rolled_back" and "quality_regression" in result["triggers"]
    assert fw.registry.champion_id(fw.compat_key) == good.bundle_id


def test_corrections_require_authorized_independent_reviewer(make_fw, records: Path) -> None:
    fw = make_fw()
    fw.import_dataset([records])
    assert fw.list_corrections("pending")
    item = fw.add_correction("scan_10001.png", ["contract"], submitted_by="rev-1")
    with pytest.raises(Exception) as e1:
        fw.verify_correction(item["label_id"], "intruder")
    assert e1.value.code == "reviewer_not_authorized"
    with pytest.raises(Exception) as e2:
        fw.verify_correction(item["label_id"], "rev-1")
    assert e2.value.code == "self_verification"
    assert fw.verify_correction(item["label_id"], "rev-2")["status"] == "verified"


def test_logs_are_ordered_complete_and_content_free(make_fw, tmp_path: Path) -> None:
    src = tmp_path / "secret.json"
    src.write_text(json.dumps([{"image_name": f"{SECRET}.png", "ocr_text": f"invoice {SECRET} payable",
                                "label": "invoice"}]))
    fw = make_fw()
    out = fw.classify([src])
    events = list(fw.run_logs(out["run_id"], level="DEBUG"))
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
    types = [e["event_type"] for e in events]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    assert "document.started" in types and "document.completed" in types
    jsonl = (fw.data_dir / "logs" / "acme" / f"{out['run_id']}.jsonl").read_text(encoding="utf-8")
    db_dump = json.dumps(events)
    assert SECRET not in jsonl and SECRET not in db_dump
    assert len(jsonl.strip().splitlines()) == len(events)


def test_jsonl_recovers_from_export_lag_and_subscriber_failure(make_fw, tmp_path: Path) -> None:
    fw = make_fw()

    def broken(_e):
        raise RuntimeError("ui crashed")

    fw.subscribers.append(broken)
    out = fw.validate_config()
    assert out["valid"]
    em = EventEmitter(fw.store, "run-x", "acme", fw._exporter("run-x"), [broken])
    fw.store.execute("INSERT INTO runs (run_id, client_id, kind, status, params, config_hash, created_at, updated_at) "
                     "VALUES ('run-x','acme','t','running','{}','h','t','t')")
    for _ in range(3):
        em.emit("x.y", "s", "ok", "m")
    path = fw.data_dir / "logs" / "acme" / "run-x.jsonl"
    path.unlink()
    path.with_suffix(".seq").unlink()
    EventEmitter(fw.store, "run-x", "acme", fw._exporter("run-x"))
    assert len(path.read_text().splitlines()) == 3
    with pytest.raises(ValueError):
        em.emit("x.y", "s", "ok", "m", ocr_text="leak")


def test_identical_content_under_different_names_shares_a_split(make_fw, tmp_path: Path) -> None:
    rows = json.loads(write_records(tmp_path / "r.json").read_text())
    dup = dict(rows[0], image_name="renamed_copy.png")
    (tmp_path / "r2.json").write_text(json.dumps(rows + [dup]))
    fw = make_fw()
    fw.import_dataset([tmp_path / "r2.json"], verified_by="rev-1")
    snap = fw.snapshot()["snapshot_id"]
    by_doc = {i.document_id: i for i in fw.datasets.snapshot_items(snap)}
    from ag_agent.inputs import document_id_for
    a, b = by_doc[document_id_for("acme", rows[0]["image_name"])], by_doc[document_id_for("acme", "renamed_copy.png")]
    assert a.family_id == b.family_id and a.split == b.split


def test_audit_write_failure_stops_work(make_fw) -> None:
    fw = make_fw()
    em = EventEmitter(fw.store, "run-y", "acme", None)
    fw.store.close()
    with pytest.raises(AuditFailure):
        em.emit("x", "s", "ok", "m")
