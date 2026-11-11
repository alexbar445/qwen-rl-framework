"""Measured resource observations from the pinned vLLM scheduler."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import math
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class KVCacheSnapshot:
    """One immutable vLLM KV-cache observation and its freshness receipt."""

    usage_fraction: float | None
    num_running: int | None
    num_waiting: int | None
    num_preempted: int | None
    sampled_at: float | None
    age_seconds: float | None
    available: bool
    stale: bool
    error: str | None = None
    prefix_cache: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVCacheMetricsObserver:
    """Capture measured scheduler usage without estimating request memory."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_finished_request_metrics: int = 1024,
    ) -> None:
        if (isinstance(max_finished_request_metrics, bool)
                or not isinstance(max_finished_request_metrics, int)
                or max_finished_request_metrics <= 0):
            raise ValueError("max_finished_request_metrics must be a positive integer")
        self._clock = clock
        self._lock = threading.Lock()
        self._usage_fraction: float | None = None
        self._num_running: int | None = None
        self._num_waiting: int | None = None
        self._num_preempted: int | None = None
        self._preemption_valid = True
        self._sampled_at: float | None = None
        self._available = False
        self._error: str | None = "not_sampled"
        self._prefix_epoch = 0
        self._prefix_reset_count = 0
        self._prefix_request_count = 0
        self._prefix_queried_tokens = 0
        self._prefix_hit_tokens = 0
        self._prefix_totals_initialized = False
        self._prefix_totals_known = True
        self._prefix_sampled_at: float | None = None
        self._prefix_available = False
        self._prefix_error: str | None = "not_sampled"
        self._observed_prompt_tokens: int | None = None
        self._observed_generated_tokens: int | None = None
        self._iteration_totals_known = True
        self._max_finished_request_metrics = max_finished_request_metrics
        self._finished_request_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._finished_request_metrics_dropped = 0

    def record(self, scheduler_stats: Any, iteration_stats: Any = None) -> None:
        """Record one pinned-vLLM scheduler callback, rejecting malformed values."""
        now = self._clock()
        preempted_delta = getattr(iteration_stats, "num_preempted_reqs", None)
        preemption_error = (
            preempted_delta is not None and not _nonnegative_int(preempted_delta)
        )

        with self._lock:
            self._record_prefix_cache(scheduler_stats, iteration_stats, now)
            self._record_finished_request_metrics(iteration_stats)

            if preemption_error:
                self._preemption_valid = False
                self._num_preempted = None
            elif (preempted_delta is not None and self._preemption_valid
                  and self._num_preempted is not None):
                self._num_preempted += preempted_delta
            elif preempted_delta is not None and self._preemption_valid:
                self._num_preempted = preempted_delta

            # vLLM may issue a callback without a scheduler sample. Keep the
            # previous receipt and let its age make freshness explicit.
            if scheduler_stats is None:
                if preemption_error:
                    self._error = "invalid_preemption_count"
                return

            usage = getattr(scheduler_stats, "kv_cache_usage", None)
            running = getattr(scheduler_stats, "num_running_reqs", None)
            waiting = getattr(scheduler_stats, "num_waiting_reqs", None)
            if not _finite_fraction(usage):
                error = "invalid_usage_fraction"
            elif not _nonnegative_int(running) or not _nonnegative_int(waiting):
                error = "invalid_scheduler_counts"
            else:
                error = "invalid_preemption_count" if not self._preemption_valid else None

            self._sampled_at = now
            if error is not None:
                if error != "invalid_preemption_count":
                    self._usage_fraction = None
                    self._num_running = None
                    self._num_waiting = None
                    self._available = False
                else:
                    # KV occupancy remains independently valid even if the
                    # optional preemption counter is malformed.
                    self._usage_fraction = float(usage)
                    self._num_running = running
                    self._num_waiting = waiting
                    self._available = True
                self._error = error
                return

            self._usage_fraction = float(usage)
            self._num_running = running
            self._num_waiting = waiting
            self._available = True
            self._error = None

    def snapshot(self, *, max_age_seconds: float = 5.0) -> KVCacheSnapshot:
        """Return the latest observation, marking old or absent samples stale."""
        if (isinstance(max_age_seconds, bool)
                or not isinstance(max_age_seconds, (int, float))
                or not math.isfinite(float(max_age_seconds))
                or max_age_seconds <= 0):
            raise ValueError("max_age_seconds must be a finite positive number")

        with self._lock:
            now = self._clock()
            sampled_at = self._sampled_at
            error = self._error
            age = None if sampled_at is None else now - sampled_at
            if age is not None and (not math.isfinite(age) or age < 0):
                age = None
                error = "invalid_sample_clock"
                available = False
            else:
                available = self._available
            stale = age is None or age > max_age_seconds
            return KVCacheSnapshot(
                usage_fraction=self._usage_fraction,
                num_running=self._num_running,
                num_waiting=self._num_waiting,
                num_preempted=self._num_preempted,
                sampled_at=sampled_at,
                age_seconds=age,
                available=available,
                stale=stale,
                error=error,
                prefix_cache=self._prefix_cache_snapshot(now, max_age_seconds),
            )

    def take_finished_request_metrics(self, request_id: str) -> dict[str, Any] | None:
        """Consume metrics for one request ID from vLLM FinishedRequestStats.

        The queue is bounded because finished outputs may outpace the caller.
        Values are copied on removal and only fields validated as finite and
        nonnegative are returned; absent timings are omitted rather than
        represented as zero.
        """
        if not isinstance(request_id, str) or not request_id:
            return None
        with self._lock:
            metrics = self._finished_request_metrics.pop(request_id, None)
            return None if metrics is None else dict(metrics)

    def _record_prefix_cache(
        self, scheduler_stats: Any, iteration_stats: Any, now: float
    ) -> None:
        if iteration_stats is not None:
            self._record_iteration_token_totals(iteration_stats)

        if scheduler_stats is None:
            return

        self._prefix_sampled_at = now
        stats = getattr(scheduler_stats, "prefix_cache_stats", None)
        if stats is None:
            self._invalidate_prefix_totals("missing_prefix_cache_stats")
            return

        reset = getattr(stats, "reset", None)
        if not isinstance(reset, bool):
            self._invalidate_prefix_totals("invalid_prefix_cache_reset")
            return
        if reset:
            self._prefix_epoch += 1
            self._prefix_reset_count += 1
            self._prefix_request_count = 0
            self._prefix_queried_tokens = 0
            self._prefix_hit_tokens = 0
            self._prefix_totals_initialized = False
            self._prefix_totals_known = True

        requests = getattr(stats, "requests", None)
        queries = getattr(stats, "queries", None)
        hits = getattr(stats, "hits", None)
        if (not _nonnegative_int(requests)
                or not _nonnegative_int(queries)
                or not _nonnegative_int(hits)
                or hits > queries):
            self._invalidate_prefix_totals("invalid_prefix_cache_counts")
            return

        if not self._prefix_totals_known:
            self._prefix_available = False
            self._prefix_error = "prefix_cache_delta_gap"
            return

        self._prefix_request_count += requests
        self._prefix_queried_tokens += queries
        self._prefix_hit_tokens += hits
        self._prefix_totals_initialized = True
        self._prefix_available = True
        self._prefix_error = None

    def _invalidate_prefix_totals(self, error: str) -> None:
        self._prefix_totals_known = False
        self._prefix_available = False
        self._prefix_error = error

    def _record_iteration_token_totals(self, iteration_stats: Any) -> None:
        prompt_tokens = getattr(iteration_stats, "num_prompt_tokens", None)
        generated_tokens = getattr(iteration_stats, "num_generation_tokens", None)
        if (not _nonnegative_int(prompt_tokens)
                or not _nonnegative_int(generated_tokens)):
            self._iteration_totals_known = False
            self._observed_prompt_tokens = None
            self._observed_generated_tokens = None
            return
        if not self._iteration_totals_known:
            return
        self._observed_prompt_tokens = (
            prompt_tokens if self._observed_prompt_tokens is None
            else self._observed_prompt_tokens + prompt_tokens
        )
        self._observed_generated_tokens = (
            generated_tokens if self._observed_generated_tokens is None
            else self._observed_generated_tokens + generated_tokens
        )

    def _record_finished_request_metrics(self, iteration_stats: Any) -> None:
        finished_requests = getattr(iteration_stats, "finished_requests", None)
        if not isinstance(finished_requests, (list, tuple)):
            return
        for finished in finished_requests:
            request_id = getattr(finished, "request_id", None)
            if not isinstance(request_id, str) or not request_id:
                continue
            metrics: dict[str, Any] = {"source": "vllm_finished_request_stats"}
            for source_name, output_name in (
                ("queued_time", "queue_seconds"),
                ("prefill_time", "prefill_seconds"),
                ("decode_time", "decode_seconds"),
            ):
                value = getattr(finished, source_name, None)
                if _finite_nonnegative(value):
                    metrics[output_name] = float(value)
            cached_tokens = getattr(finished, "num_cached_tokens", None)
            if _nonnegative_int(cached_tokens):
                metrics["num_cached_tokens"] = cached_tokens
            if len(metrics) == 1:
                continue
            self._finished_request_metrics.pop(request_id, None)
            self._finished_request_metrics[request_id] = metrics
            while len(self._finished_request_metrics) > self._max_finished_request_metrics:
                self._finished_request_metrics.popitem(last=False)
                self._finished_request_metrics_dropped += 1

    def _prefix_cache_snapshot(
        self, now: float, max_age_seconds: float
    ) -> dict[str, Any]:
        sampled_at = self._prefix_sampled_at
        age = None if sampled_at is None else now - sampled_at
        error = self._prefix_error
        available = self._prefix_available
        if age is not None and (not math.isfinite(age) or age < 0):
            age = None
            available = False
            error = "invalid_prefix_sample_clock"
        stale = age is None or age > max_age_seconds
        known = self._prefix_totals_known and self._prefix_totals_initialized
        queried_tokens = self._prefix_queried_tokens if known else None
        hit_tokens = self._prefix_hit_tokens if known else None
        request_count = self._prefix_request_count if known else None
        hit_fraction = (
            hit_tokens / queried_tokens
            if queried_tokens is not None and queried_tokens > 0
            and hit_tokens is not None else None
        )
        return {
            "scope": "since_prefix_cache_reset",
            "request_count": request_count,
            "queried_tokens": queried_tokens,
            "hit_tokens": hit_tokens,
            "hit_fraction": hit_fraction,
            "epoch": self._prefix_epoch,
            "reset_count": self._prefix_reset_count,
            "sampled_at": sampled_at,
            "age_seconds": age,
            "available": available,
            "stale": stale,
            "error": error,
            "observed_prompt_tokens": (
                self._observed_prompt_tokens if self._iteration_totals_known else None
            ),
            "observed_generated_tokens": (
                self._observed_generated_tokens if self._iteration_totals_known else None
            ),
            "finished_request_metrics_pending": len(self._finished_request_metrics),
            "finished_request_metrics_dropped": self._finished_request_metrics_dropped,
        }

    def stat_logger_factory(self) -> Callable[..., Any]:
        """Build a factory matching vLLM 0.29's public StatLoggerBase API.

        Keep the optional Worker dependency lazy so the observer and its CPU
        tests can be imported without importing CUDA or the vLLM runtime.
        """
        try:
            from vllm.v1.metrics.loggers import StatLoggerBase
        except ImportError as exc:  # pragma: no cover - exercised in Worker image
            raise RuntimeError("vLLM StatLoggerBase is required for KV telemetry") from exc

        observer = self

        class _KVCacheStatLogger(StatLoggerBase):
            def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
                self.engine_index = engine_index

            def record(
                self,
                scheduler_stats: Any,
                iteration_stats: Any,
                mm_cache_stats: Any = None,
                engine_idx: int = 0,
            ) -> None:
                observer.record(scheduler_stats, iteration_stats)

            def log_engine_initialized(self) -> None:
                return None

            def log(self) -> None:
                return None

            def record_sleep_state(self, is_awake: int, level: int) -> None:
                return None

        def factory(vllm_config: Any, engine_index: int = 0) -> Any:
            return _KVCacheStatLogger(vllm_config, engine_index)

        return factory


def _finite_fraction(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= value <= 1.0
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )
