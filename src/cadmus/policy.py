"""Deterministic, versioned promotion gates. Every gate reports observed value, threshold and reason code."""

from __future__ import annotations

from typing import Any

from cadmus.config import PolicyConfig, TaskConfig
from cadmus.contracts import DecisionOutcome, EvaluationReport, GateResult, PromotionDecision

INSUFFICIENT_GATES = {"completeness", "gate_group_support", "critical_class_support"}


def evaluate_gates(policy: PolicyConfig, task: TaskConfig, incumbent: EvaluationReport, candidate: EvaluationReport,
                   comparison: dict[str, Any], gate_groups: int) -> list[GateResult]:
    gates: list[GateResult] = []

    def add(name: str, passed: bool, observed: Any, threshold: Any, reason: str, detail: str | None = None) -> None:
        gates.append(GateResult(name=name, passed=passed, observed=observed, threshold=threshold,
                                reason_code="ok" if passed else reason, detail=detail))

    infra = max(incumbent.infra_failure_rate, candidate.infra_failure_rate)
    add("completeness", infra <= policy.max_infra_failure_rate and incumbent.expected == candidate.expected,
        round(infra, 4), policy.max_infra_failure_rate, "incomplete_paired_comparison")
    add("gate_group_support", gate_groups >= policy.min_gate_groups, gate_groups, policy.min_gate_groups,
        "insufficient_gate_groups")
    for lbl in task.critical_labels:
        support = candidate.per_class[lbl].support
        add("critical_class_support", support >= policy.min_class_support, support, policy.min_class_support,
            "insufficient_class_support", lbl)
    score = candidate.metric(policy.primary_metric)
    add("absolute_floor", score >= policy.absolute_floor, round(score, 4), policy.absolute_floor, "below_quality_floor")
    diff = comparison["difference"]
    add("practical_gain", diff >= policy.min_gain, round(diff, 4), policy.min_gain, "gain_below_minimum")
    add("confidence_bound", comparison["lower_bound"] > 0, round(comparison["lower_bound"], 4), 0,
        "improvement_not_established", f"{int(policy.confidence * 100)}% interval lower bound")
    for lbl in task.critical_labels:
        before, after = incumbent.per_class[lbl].f1, candidate.per_class[lbl].f1
        add("critical_class_non_regression", after >= before - policy.critical_tolerance, round(after - before, 4),
            -policy.critical_tolerance, "critical_class_regression", lbl)
    add("coverage", candidate.coverage >= policy.min_coverage, round(candidate.coverage, 4), policy.min_coverage,
        "coverage_below_minimum")
    add("invalid_rate", candidate.invalid_rate <= policy.max_invalid_rate, round(candidate.invalid_rate, 4),
        policy.max_invalid_rate, "invalid_rate_too_high")
    if policy.max_cost_ratio is not None:
        base = incumbent.input_tokens + incumbent.output_tokens
        cand = candidate.input_tokens + candidate.output_tokens
        ratio = cand / base if base else 1.0
        add("resource_ratio", ratio <= policy.max_cost_ratio, round(ratio, 3), policy.max_cost_ratio,
            "resource_ratio_exceeded", "token ratio vs incumbent")
    return gates


def decide(run_id: str, policy: PolicyConfig, task: TaskConfig, incumbent: EvaluationReport,
           candidate: EvaluationReport, comparison: dict[str, Any], gate_groups: int,
           incumbent_id: str | None, finalist_id: str) -> PromotionDecision:
    gates = evaluate_gates(policy, task, incumbent, candidate, comparison, gate_groups)
    failed = [g for g in gates if not g.passed]
    outcome: DecisionOutcome
    if any(g.name in INSUFFICIENT_GATES for g in failed):
        outcome = "insufficient_evidence"
    elif failed:
        outcome = "rejected"
    else:
        outcome = "promoted"
    return PromotionDecision(run_id=run_id, outcome=outcome, reason_codes=sorted({g.reason_code for g in failed})
                             or ["all_gates_passed"], incumbent_bundle_id=incumbent_id, finalist_bundle_id=finalist_id,
                             gates=gates, comparison=comparison, policy_version=policy.version)


def baseline_gates(policy: PolicyConfig, report: EvaluationReport) -> list[GateResult]:
    """Absolute gates for registering a first champion when no incumbent exists."""
    score = report.metric(policy.primary_metric)
    return [
        GateResult(name="absolute_floor", passed=score >= policy.absolute_floor, observed=round(score, 4),
                   threshold=policy.absolute_floor, reason_code="ok" if score >= policy.absolute_floor
                   else "below_quality_floor"),
        GateResult(name="coverage", passed=report.coverage >= policy.min_coverage, observed=round(report.coverage, 4),
                   threshold=policy.min_coverage, reason_code="ok" if report.coverage >= policy.min_coverage
                   else "coverage_below_minimum"),
        GateResult(name="invalid_rate", passed=report.invalid_rate <= policy.max_invalid_rate,
                   observed=round(report.invalid_rate, 4), threshold=policy.max_invalid_rate,
                   reason_code="ok" if report.invalid_rate <= policy.max_invalid_rate else "invalid_rate_too_high"),
    ]
