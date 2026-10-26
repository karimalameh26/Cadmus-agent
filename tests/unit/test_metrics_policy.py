from __future__ import annotations

import pytest

from cadmus.config import PolicyConfig, TaskConfig
from cadmus.contracts import Prediction, PredictionStatus, TaskKind
from cadmus.datasets import SnapshotItem
from cadmus.evaluation import build_report, paired_bootstrap
from cadmus.policy import decide
from cadmus.tasks import ClassificationTask, compute_metrics

S, A, INV = PredictionStatus.SUCCESS, PredictionStatus.ABSTAINED, PredictionStatus.INVALID_RESPONSE


def p(labels: list[str], status: PredictionStatus = S, doc: str = "d") -> Prediction:
    return Prediction(document_id=doc, status=status, labels=labels if status == S else [], bundle_id="b")


single = ClassificationTask(TaskConfig(name="t", labels=["a", "b", "c"]))
multi = ClassificationTask(TaskConfig(name="t", kind=TaskKind.MULTILABEL, labels=["a", "b", "c"]))


def test_single_label_exact_values_and_abstention_penalty() -> None:
    pairs = [(["a"], p(["a"])), (["a"], p(["b"])), (["b"], p(["b"])), (["c"], p([], A)), (["c"], p([], INV))]
    m = compute_metrics(single, pairs)
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["answered_accuracy"] == pytest.approx(2 / 3)
    assert m["per_class"]["a"].f1 == pytest.approx(2 / 3)  # P=1, R=0.5
    assert m["per_class"]["b"].f1 == pytest.approx(2 / 3)  # P=0.5, R=1
    assert m["per_class"]["c"].f1 == 0.0 and m["per_class"]["c"].support == 2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["micro_f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 3))


def test_multilabel_empty_set_vs_abstention() -> None:
    pairs = [([], p([])), ([], p([], A)), (["a", "b"], p(["a"])), (["c"], p(["c", "a"]))]
    m = compute_metrics(multi, pairs)
    assert m["exact_match"] == pytest.approx(1 / 4)
    assert m["per_class"]["a"].predicted == 2 and m["per_class"]["a"].support == 1


def test_zero_support_labels_excluded_from_macro_unless_predicted() -> None:
    assert compute_metrics(single, [(["a"], p(["a"]))])["macro_f1"] == 1.0
    assert compute_metrics(single, [(["a"], p(["b"]))])["macro_f1"] == 0.0


def test_parse_rejects_out_of_taxonomy_and_cardinality() -> None:
    assert single.parse('{"status":"success","labels":["z"]}').error_code == "label_not_in_taxonomy"
    assert single.parse('{"status":"success","labels":["a","b"]}').error_code == "label_cardinality"
    assert single.parse("not json").status == INV
    assert single.parse('```json\n{"status":"success","labels":["A"]}\n```').labels == ["a"]
    assert single.parse('{"status":"abstained","labels":["a"]}').error_code == "abstained_with_labels"


def items(n: int, label: str = "a") -> list[SnapshotItem]:
    return [SnapshotItem(f"d{i}", f"f{i}", "h", [label], "gate") for i in range(n)]


def test_missing_predictions_do_not_shrink_denominator() -> None:
    its = items(4)
    r = build_report(single, "b", "gate", its, [p(["a"], doc="d0"), p(["a"], doc="d1")])
    assert r.expected == 4 and r.accuracy == 0.5 and r.infra_failure_rate == 0.5


def test_bootstrap_is_seeded_and_paired() -> None:
    its = items(20)
    a = [p(["a"] if i < 10 else ["b"], doc=f"d{i}") for i in range(20)]
    b = [p(["a"], doc=f"d{i}") for i in range(20)]
    r1 = paired_bootstrap(single, "accuracy", its, a, b, resamples=200, seed=1, confidence=0.9)
    r2 = paired_bootstrap(single, "accuracy", its, a, b, resamples=200, seed=1, confidence=0.9)
    assert r1 == r2 and r1["difference"] == pytest.approx(0.5) and r1["lower_bound"] > 0


def _decision(task_cfg: TaskConfig, inc_preds, cand_preds, its, **policy):
    task = ClassificationTask(task_cfg)
    pol = PolicyConfig(bootstrap_resamples=200, **policy)
    inc = build_report(task, "inc", "gate", its, inc_preds)
    cand = build_report(task, "cand", "gate", its, cand_preds)
    comp = paired_bootstrap(task, pol.primary_metric, its, inc_preds, cand_preds, resamples=200, seed=3,
                            confidence=pol.confidence)
    return decide("run", pol, task_cfg, inc, cand, comp, len({i.family_id for i in its}), "inc", "cand")


def test_critical_class_regression_is_rejected_even_with_overall_gain() -> None:
    cfg = TaskConfig(name="t", labels=["a", "b", "c"], critical_labels=["c"])
    truth = ["a"] * 10 + ["b"] * 10 + ["c"] * 4
    its = [SnapshotItem(f"d{i}", f"f{i}", "h", [lbl], "gate") for i, lbl in enumerate(truth)]
    inc = [p(["c"] if it.labels == ["c"] else ["a"], doc=it.document_id) for it in its]
    cand = [p(it.labels if it.labels != ["c"] else (["c"] if i % 2 else ["a"]), doc=it.document_id)
            for i, it in enumerate(its)]
    d = _decision(cfg, inc, cand, its, min_gate_groups=10)
    assert d.outcome == "rejected" and "critical_class_regression" in d.reason_codes


def test_insufficient_support_yields_insufficient_evidence() -> None:
    cfg = TaskConfig(name="t", labels=["a", "b", "c"], critical_labels=["c"])
    its = items(6)
    d = _decision(cfg, [p([], A, doc=i.document_id) for i in its], [p(["a"], doc=i.document_id) for i in its], its,
                  min_gate_groups=10)
    assert d.outcome == "insufficient_evidence"
    assert {"insufficient_gate_groups", "insufficient_class_support"} <= set(d.reason_codes)
