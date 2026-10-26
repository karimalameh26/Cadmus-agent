"""Inference executor: budget reservation, ledgered attempts, retries, caching, resume reuse and progress events."""

from __future__ import annotations

import random
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from ag_agent.config import AppConfig, PricingConfig
from ag_agent.contracts import (
    CACHEABLE_STATUSES,
    EvidenceRecord,
    Prediction,
    PredictionStatus,
    PromptBundle,
    Usage,
    canonical_json,
    sha256_text,
    utc_now,
)
from ag_agent.errors import BudgetExhausted, NotFound, ProviderError, RunCancelled
from ag_agent.observability import EventEmitter
from ag_agent.prompting import compose_request
from ag_agent.providers import ModelAdapter, model_hash
from ag_agent.storage import ArtifactStore, Store
from ag_agent.tasks import ClassificationTask


@dataclass
class Reservation:
    estimated_cost: float
    pricing: PricingConfig


class Budget:
    """Conservative pre-dispatch reservation. Unknown usage is charged at the reserved estimate, never zero."""

    def __init__(self, pricing: PricingConfig, max_requests: int | None = None, max_cost_usd: float | None = None,
                 max_elapsed_s: float | None = None) -> None:
        self.pricing = pricing
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.max_elapsed_s = max_elapsed_s
        self.requests = 0
        self.reserved = 0.0
        self.spent = 0.0
        self.unknown_usage = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()

    @staticmethod
    def cost(pricing: PricingConfig, input_tokens: int, output_tokens: int) -> float:
        if pricing.input_per_1k is None or pricing.output_per_1k is None:
            return 0.0
        return input_tokens / 1000 * pricing.input_per_1k + output_tokens / 1000 * pricing.output_per_1k

    def reserve(self, est_in: int, est_out: int, pricing: PricingConfig | None = None) -> Reservation:
        pricing = pricing or self.pricing
        with self._lock:
            if self.max_elapsed_s is not None and time.monotonic() - self._start > self.max_elapsed_s:
                raise BudgetExhausted("Elapsed-time budget exhausted", "budget_time")
            if self.max_requests is not None and self.requests + 1 > self.max_requests:
                raise BudgetExhausted("Request budget exhausted", "budget_requests")
            est = self.cost(pricing, est_in, est_out)
            if self.max_cost_usd is not None and self.spent + self.reserved + est > self.max_cost_usd:
                raise BudgetExhausted("Cost budget exhausted", "budget_cost")
            self.requests += 1
            self.reserved += est
            return Reservation(est, pricing)

    def settle(self, res: Reservation, input_tokens: int | None, output_tokens: int | None,
               charge_if_unknown: bool) -> float | None:
        with self._lock:
            self.reserved -= res.estimated_cost
            if input_tokens is not None and output_tokens is not None:
                actual = self.cost(res.pricing, input_tokens, output_tokens)
                self.spent += actual
                return actual if res.pricing.known else None
            if charge_if_unknown:
                self.unknown_usage += 1
                self.spent += res.estimated_cost
            return None

    def remaining(self) -> dict[str, float | int | None]:
        with self._lock:
            left = None if self.max_cost_usd is None else round(self.max_cost_usd - self.spent - self.reserved, 6)
            return {
                "requests": None if self.max_requests is None else self.max_requests - self.requests,
                "cost_usd": left,
            }


class Classifier:
    def __init__(self, cfg: AppConfig, task: ClassificationTask, model: ModelAdapter, store: Store,
                 artifacts: ArtifactStore, emitter: EventEmitter, budget: Budget,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.cfg = cfg
        self.task = task
        self.model = model
        self.store = store
        self.artifacts = artifacts
        self.emitter = emitter
        self.budget = budget
        self.model_hash = model_hash(model)
        self._stop = threading.Event()
        self._sleep = sleep

    def cache_key(self, bundle: PromptBundle, evidence_hash: str, repeat_index: int) -> str:
        body = {"client": self.cfg.client_id, "compat": bundle.compatibility_key, "bundle": bundle.bundle_id,
                "evidence": evidence_hash, "repeat": repeat_index, "model": self.model_hash,
                "task": self.cfg.task.identity()}
        return "c-" + sha256_text(canonical_json(body))

    def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
        else:
            self._stop.wait(seconds)

    def _record(self, run_id: str, purpose: str, pred: Prediction) -> None:
        with self.store.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_predictions (run_id, purpose, bundle_id, document_id, repeat_index, "
                "status, payload, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, purpose, pred.bundle_id, pred.document_id, pred.repeat_index, pred.status.value,
                 pred.model_dump_json(), utc_now()))

    def _reuse(self, run_id: str, purpose: str, bundle: PromptBundle, ev: EvidenceRecord, repeat: int,
               key: str) -> Prediction | None:
        row = self.store.one(
            "SELECT payload FROM run_predictions WHERE run_id=? AND purpose=? AND bundle_id=? AND document_id=? "
            "AND repeat_index=? AND status != 'cancelled'", (run_id, purpose, bundle.bundle_id, ev.document_id, repeat))
        if row is not None:
            pred = Prediction.model_validate_json(row["payload"])
            if pred.evidence_hash == ev.evidence_hash():
                return pred.model_copy(update={"cached": True})
        row = self.store.one("SELECT payload FROM predictions WHERE cache_key=? AND client_id=?",
                             (key, self.cfg.client_id))
        if row is not None:
            return Prediction.model_validate_json(row["payload"]).model_copy(update={"cached": True})
        return None

    def _predict(self, run_id: str, purpose: str, stage: str, bundle: PromptBundle, ev: EvidenceRecord,
                 examples: list[EvidenceRecord], repeat: int) -> Prediction:
        op = "op-" + uuid.uuid4().hex[:12]
        ev_hash = ev.evidence_hash()
        key = self.cache_key(bundle, ev_hash, repeat)
        common: dict[str, Any] = {"operation_id": op, "document_id": ev.document_id, "bundle_id": bundle.bundle_id,
                                  "purpose": purpose, "repeat_index": repeat}
        self.emitter.emit("document.started", stage, "started", "Classification started", **common)

        def finish(pred: Prediction) -> Prediction:
            if pred.status in CACHEABLE_STATUSES and not pred.cached:
                with self.store.tx() as conn:
                    conn.execute("INSERT OR IGNORE INTO predictions (cache_key, client_id, payload, created_at) "
                                 "VALUES (?,?,?,?)", (key, self.cfg.client_id, pred.model_dump_json(), utc_now()))
            self._record(run_id, purpose, pred)
            level = "INFO" if pred.status in CACHEABLE_STATUSES else "WARNING"
            self.emitter.emit("document.completed", stage, pred.status.value, "Classification finished", level,
                              cache_hit=pred.cached, error_code=pred.error_code, duration_ms=pred.latency_ms, **common)
            return pred

        cached = self._reuse(run_id, purpose, bundle, ev, repeat, key)
        if cached is not None:
            return finish(cached)
        base: dict[str, Any] = {"document_id": ev.document_id, "bundle_id": bundle.bundle_id,
                                "evidence_hash": ev_hash, "repeat_index": repeat}
        if self._stop.is_set():
            return finish(Prediction(status=PredictionStatus.CANCELLED, error_code="cancelled", **base))
        if not ev.has_evidence():
            return finish(Prediction(status=PredictionStatus.NO_EVIDENCE, error_code="no_evidence", **base))
        media_count = len(ev.media) + sum(len(e.media) for e in examples)
        if media_count and not self.model.supports_images:
            return finish(Prediction(status=PredictionStatus.PROVIDER_ERROR, error_code="capability_mismatch", **base))
        request = compose_request(self.task, bundle, ev, examples, self.artifacts,
                                  self.cfg.provider.max_output_tokens, self.cfg.provider.temperature)
        chars = request.text_chars()
        self.emitter.emit("request.prepared", stage, "completed", "Request prepared", "DEBUG", request_chars=chars,
                          media_count=media_count, examples=len(examples), **common)
        if chars > self.cfg.documents.max_request_chars:
            return finish(Prediction(status=PredictionStatus.EXTRACTION_ERROR, error_code="request_too_large", **base))

        max_attempts = self.cfg.provider.max_attempts
        for attempt in range(1, max_attempts + 1):
            if self._stop.is_set():
                return finish(Prediction(status=PredictionStatus.CANCELLED, error_code="cancelled", **base))
            est_in = chars // 4 + request.image_count() * self.cfg.provider.pricing.image_token_estimate
            reservation = self.budget.reserve(est_in, self.cfg.provider.max_output_tokens)
            request_id = "req-" + uuid.uuid4().hex[:16]
            self.store.execute(
                "INSERT INTO requests (request_id, run_id, cache_key, attempt, status, started_at, reserved_cost) "
                "VALUES (?,?,?,?,?,?,?)", (request_id, run_id, key, attempt, "dispatched", utc_now(),
                                           reservation.estimated_cost))
            self.emitter.emit("provider.attempt.started", stage, "started", "Provider request dispatched",
                              attempt=attempt, estimated_cost_usd=reservation.estimated_cost or None, **common)
            t0 = time.monotonic()
            try:
                resp = self.model.complete(request)
            except ProviderError as exc:
                ms = int((time.monotonic() - t0) * 1000)
                self.budget.settle(reservation, None, None, charge_if_unknown=exc.completion_unknown)
                status = "unknown" if exc.completion_unknown else "failed"
                self.store.execute("UPDATE requests SET status=?, ended_at=?, error_code=?, provider_request_id=? "
                                   "WHERE request_id=?", (status, utc_now(), exc.code, exc.request_id, request_id))
                retry = exc.transient and attempt < max_attempts
                jitter = 0.5 + random.random()
                delay = self.cfg.provider.backoff_base_s * (2 ** (attempt - 1)) * jitter if retry else 0
                self.emitter.emit("provider.attempt.failed", stage, status, "Provider request failed", "WARNING",
                                  attempt=attempt, error_code=exc.code, duration_ms=ms,
                                  retry_delay_ms=int(delay * 1000) if retry else None,
                                  provider_request_id=exc.request_id, budget_remaining=self.budget.remaining(),
                                  **common)
                if retry:
                    self._wait(delay)
                    continue
                return finish(Prediction(status=PredictionStatus.PROVIDER_ERROR, error_code=exc.code, latency_ms=ms,
                                         **base))
            ms = int((time.monotonic() - t0) * 1000)
            cost = self.budget.settle(reservation, resp.input_tokens, resp.output_tokens, charge_if_unknown=True)
            self.store.execute(
                "UPDATE requests SET status='completed', ended_at=?, input_tokens=?, output_tokens=?, actual_cost=?, "
                "provider_request_id=? WHERE request_id=?",
                (utc_now(), resp.input_tokens, resp.output_tokens, cost, resp.provider_request_id, request_id))
            self.emitter.emit("provider.attempt.completed", stage, "completed", "Provider request completed",
                              attempt=attempt, duration_ms=ms, input_tokens=resp.input_tokens,
                              output_tokens=resp.output_tokens, actual_cost_usd=cost,
                              provider_request_id=resp.provider_request_id, **common)
            parsed = self.task.parse(resp.text)
            self.emitter.emit("prediction.validated", stage, parsed.status.value, "Response validated",
                              "DEBUG" if parsed.error_code is None else "WARNING", error_code=parsed.error_code,
                              **common)
            return finish(Prediction(
                status=parsed.status, labels=parsed.labels, evidence_refs=parsed.evidence_refs,
                explanation=parsed.explanation, confidence=parsed.confidence, latency_ms=ms,
                error_code=parsed.error_code,
                usage=Usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=cost), **base))
        raise AssertionError("unreachable")

    def run(self, run_id: str, bundle: PromptBundle, evidences: Sequence[EvidenceRecord], *, purpose: str,
            stage: str, repeat_index: int = 0) -> list[Prediction]:
        try:
            examples = [self.artifacts.get_evidence(e.evidence_hash) for e in bundle.examples]
        except FileNotFoundError:
            raise NotFound("Few-shot example evidence is no longer retained", "example_evidence_missing") from None
        total = len(evidences)
        self.emitter.emit("batch.started", stage, "started", f"Classifying {total} documents", total=total,
                          completed=0, bundle_id=bundle.bundle_id, purpose=purpose)
        results: dict[int, Prediction] = {}
        failure: BaseException | None = None
        step = max(1, total // 10)
        last_progress = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=self.cfg.provider.max_concurrency)
        try:
            futures: dict[Future[Prediction], int] = {
                pool.submit(self._predict, run_id, purpose, stage, bundle, ev, examples, repeat_index): i
                for i, ev in enumerate(evidences)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except BudgetExhausted as exc:
                    failure = failure or exc
                    self._stop.set()
                    continue
                done = len(results)
                if done % step == 0 or done == total or time.monotonic() - last_progress > 5:
                    last_progress = time.monotonic()
                    failed = sum(p.status not in CACHEABLE_STATUSES for p in results.values())
                    self.emitter.emit("batch.progress", stage, "running", "Classification progress",
                                      completed=done, total=total, failed=failed, bundle_id=bundle.bundle_id,
                                      budget_remaining=self.budget.remaining())
        except KeyboardInterrupt:
            self._stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            self.emitter.emit("batch.cancelled", stage, "cancelled", "Dispatch stopped by interrupt", "WARNING",
                              completed=len(results), total=total)
            raise RunCancelled("Run cancelled by user") from None
        finally:
            pool.shutdown(wait=True)
        if failure is not None:
            self.emitter.emit("batch.stopped", stage, "failed", "Budget exhausted; dispatch stopped", "WARNING",
                              error_code=getattr(failure, "code", None), completed=len(results), total=total,
                              budget_remaining=self.budget.remaining())
            raise failure
        self._stop.clear()
        preds = [results[i] for i in range(total)]
        counts: dict[str, int] = {}
        for p in preds:
            counts[p.status.value] = counts.get(p.status.value, 0) + 1
        self.emitter.emit("batch.completed", stage, "completed", "Classification batch finished", completed=total,
                          total=total, counts=counts, bundle_id=bundle.bundle_id)
        return preds
