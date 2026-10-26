"""Evaluation reports and seeded paired bootstrap comparison over document-family groups."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import median
from typing import Any

from cadmus.contracts import INFRA_STATUSES, EvaluationReport, Prediction, PredictionStatus
from cadmus.datasets import SnapshotItem
from cadmus.tasks import ClassificationTask, compute_metrics, primary_score


def build_report(task: ClassificationTask, bundle_id: str, split: str, items: Sequence[SnapshotItem],
                 preds: Sequence[Prediction]) -> EvaluationReport:
    by_doc: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        by_doc[p.document_id].append(p)
    pairs: list[tuple[list[str], Prediction]] = []
    for item in items:
        doc_preds = by_doc.get(item.document_id) or [
            Prediction(document_id=item.document_id, status=PredictionStatus.CANCELLED, bundle_id=bundle_id,
                       error_code="missing_prediction")]
        pairs += [(item.labels, p) for p in doc_preds]
    n = len(pairs)
    counts: dict[str, int] = defaultdict(int)
    for _, p in pairs:
        counts[p.status.value] += 1
    m = compute_metrics(task, pairs)
    latencies = sorted(p.latency_ms for _, p in pairs if p.latency_ms is not None and not p.cached)
    in_tok = sum(p.usage.input_tokens or 0 for _, p in pairs if not p.cached)
    out_tok = sum(p.usage.output_tokens or 0 for _, p in pairs if not p.cached)
    unknown = sum(1 for _, p in pairs if not p.cached and p.status not in (PredictionStatus.NO_EVIDENCE,)
                  and (p.usage.input_tokens is None or p.usage.output_tokens is None))
    costs = [p.usage.cost_usd for _, p in pairs if not p.cached and p.usage.cost_usd is not None]
    return EvaluationReport(
        bundle_id=bundle_id, split=split, expected=n,
        completed=sum(1 for _, p in pairs if p.status not in INFRA_STATUSES),
        status_counts=dict(counts),
        coverage=counts[PredictionStatus.SUCCESS.value] / n if n else 0.0,
        invalid_rate=counts[PredictionStatus.INVALID_RESPONSE.value] / n if n else 0.0,
        infra_failure_rate=sum(counts[s.value] for s in INFRA_STATUSES) / n if n else 0.0,
        accuracy=m.get("accuracy"), exact_match=m.get("exact_match"), answered_accuracy=m["answered_accuracy"],
        macro_f1=m["macro_f1"], micro_f1=m["micro_f1"], per_class=m["per_class"],
        input_tokens=in_tok, output_tokens=out_tok, unknown_usage=unknown,
        cost_usd=sum(costs) if costs else None,
        latency_ms_p50=int(median(latencies)) if latencies else None,
        latency_ms_p95=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
    )


def paired_bootstrap(task: ClassificationTask, metric: str, items: Sequence[SnapshotItem],
                     preds_a: Sequence[Prediction], preds_b: Sequence[Prediction], *, resamples: int, seed: int,
                     confidence: float) -> dict[str, Any]:
    """Difference B - A with a percentile interval, resampling whole families so repeats stay dependent."""
    a_by: dict[str, list[Prediction]] = defaultdict(list)
    b_by: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds_a:
        a_by[p.document_id].append(p)
    for p in preds_b:
        b_by[p.document_id].append(p)
    families: dict[str, list[SnapshotItem]] = defaultdict(list)
    for it in items:
        families[it.family_id].append(it)
    fam_ids = sorted(families)

    def pairs_for(fams: Sequence[str], by: dict[str, list[Prediction]]) -> list[tuple[list[str], Prediction]]:
        return [(it.labels, p) for f in fams for it in families[f] for p in by[it.document_id]]

    point_a = primary_score(task, metric, pairs_for(fam_ids, a_by))
    point_b = primary_score(task, metric, pairs_for(fam_ids, b_by))
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        sample = [fam_ids[rng.randrange(len(fam_ids))] for _ in fam_ids]
        diffs.append(primary_score(task, metric, pairs_for(sample, b_by)) -
                     primary_score(task, metric, pairs_for(sample, a_by)))
    diffs.sort()
    alpha = (1 - confidence) / 2
    lo = diffs[int(alpha * (len(diffs) - 1))]
    hi = diffs[int((1 - alpha) * (len(diffs) - 1))]
    return {"metric": metric, "incumbent": point_a, "candidate": point_b, "difference": point_b - point_a,
            "lower_bound": lo, "upper_bound": hi, "confidence": confidence, "resamples": resamples, "seed": seed,
            "groups": len(fam_ids), "method": "paired_family_bootstrap_percentile"}
