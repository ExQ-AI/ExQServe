"""Low-cardinality Prometheus metrics for protocol-neutral serving."""

from __future__ import annotations

import logging
from collections.abc import Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from exqserve.core.engine_stats import RuntimeEngineState, RuntimeEngineStats
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage

_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
)
_PREFILL_RATE_BUCKETS = (
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
    20000.0,
)
_DECODE_RATE_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected"})
_RENDERER_KINDS = frozenset({"chat", "raw_text", "count_tokens"})
_ENGINE_NUMERIC_FIELDS = (
    ("active_jobs", "exqserve_engine_active_jobs", "Current ExLlamaV3 active jobs."),
    ("pending_jobs", "exqserve_engine_pending_jobs", "Current ExLlamaV3 pending jobs."),
    ("max_batch_size", "exqserve_engine_max_batch_size", "Current ExLlamaV3 maximum batch size."),
    ("kv_pages_total", "exqserve_engine_kv_pages_total", "Total ExLlamaV3 KV pages."),
    ("kv_pages_referenced", "exqserve_engine_kv_pages_referenced", "Currently referenced ExLlamaV3 KV pages."),
    ("kv_pages_unreferenced", "exqserve_engine_kv_pages_unreferenced", "Currently unreferenced/allocatable ExLlamaV3 KV pages."),
    (
        "kv_pages_evicted_since_generator_start",
        "exqserve_engine_kv_pages_evicted_since_generator_start",
        "KV pages repurposed since the current Generator started.",
    ),
    (
        "kv_cached_pages_evicted_since_generator_start",
        "exqserve_engine_kv_cached_pages_evicted_since_generator_start",
        "Reusable cached KV pages lost since the current Generator started.",
    ),
    (
        "kv_recurrent_checkpoints_stranded_since_generator_start",
        "exqserve_engine_kv_recurrent_checkpoints_stranded_since_generator_start",
        "Recurrent checkpoints stranded by live KV eviction since the current Generator started.",
    ),
    (
        "kv_pages_allocated_since_generator_start",
        "exqserve_engine_kv_pages_allocated_since_generator_start",
        "KV pages claimed by jobs since the current Generator started.",
    ),
    (
        "kv_cached_pages_reused_since_generator_start",
        "exqserve_engine_kv_cached_pages_reused_since_generator_start",
        "Cached KV pages reused since the current Generator started.",
    ),
    (
        "kv_pages_restored_from_cpu_tier_since_generator_start",
        "exqserve_engine_kv_pages_restored_from_cpu_tier_since_generator_start",
        "KV pages restored from the CPU tier since the current Generator started.",
    ),
    (
        "kv_cached_kv_only_pages_since_generator_start",
        "exqserve_engine_kv_cached_kv_only_pages_since_generator_start",
        "Cached KV-only pages without a resumable recurrent checkpoint since the current Generator started.",
    ),
    ("cpu_kv_cached_pages", "exqserve_engine_cpu_kv_cached_pages", "Current pages retained in the CPU KV tier."),
    ("cpu_kv_cache_max_pages", "exqserve_engine_cpu_kv_cache_max_pages", "Maximum page slots in the CPU KV tier."),
    (
        "cpu_kv_cache_pushes_since_generator_start",
        "exqserve_engine_cpu_kv_cache_pushes_since_generator_start",
        "KV pages copied into the CPU tier since the current Generator started.",
    ),
    (
        "cpu_kv_cache_restores_since_generator_start",
        "exqserve_engine_cpu_kv_cache_restores_since_generator_start",
        "KV pages restored from the CPU tier since the current Generator started.",
    ),
    (
        "cpu_kv_cache_evictions_since_generator_start",
        "exqserve_engine_cpu_kv_cache_evictions_since_generator_start",
        "CPU-tier KV entries evicted since the current Generator started.",
    ),
    (
        "cpu_kv_cache_dedup_hits_since_generator_start",
        "exqserve_engine_cpu_kv_cache_dedup_hits_since_generator_start",
        "Duplicate CPU-tier pushes avoided since the current Generator started.",
    ),
    ("recurrent_cache_entries", "exqserve_engine_recurrent_cache_entries", "Current recurrent checkpoint entries."),
    ("recurrent_cache_bytes", "exqserve_engine_recurrent_cache_bytes", "Current recurrent checkpoint bytes in host memory."),
    (
        "recurrent_cache_evictions_since_generator_start",
        "exqserve_engine_recurrent_cache_evictions_since_generator_start",
        "Recurrent checkpoints evicted since the current Generator started.",
    ),
    (
        "recurrent_cache_pruned_since_generator_start",
        "exqserve_engine_recurrent_cache_pruned_since_generator_start",
        "Stranded recurrent checkpoints pruned since the current Generator started.",
    ),
    (
        "vision_cache_budget_bytes",
        "exqserve_engine_vision_cache_budget_bytes",
        "Configured retained-tensor byte budget for the Vision embedding cache.",
    ),
    (
        "vision_cache_retained_entries",
        "exqserve_engine_vision_cache_retained_entries",
        "Current Vision embedding cache entries.",
    ),
    (
        "vision_cache_retained_tensor_bytes",
        "exqserve_engine_vision_cache_retained_tensor_bytes",
        "Current retained Vision embedding tensor bytes.",
    ),
    ("vision_cache_queries", "exqserve_engine_vision_cache_queries", "Cumulative unique Vision media lookups."),
    ("vision_cache_hits", "exqserve_engine_vision_cache_hits", "Cumulative Vision embedding cache hits."),
    ("vision_cache_misses", "exqserve_engine_vision_cache_misses", "Cumulative Vision embedding cache misses."),
    ("vision_cache_evictions", "exqserve_engine_vision_cache_evictions", "Cumulative Vision embedding cache evictions."),
    (
        "vision_cache_admission_skipped",
        "exqserve_engine_vision_cache_admission_skipped",
        "Cumulative Vision embeddings skipped from persistent admission.",
    ),
    (
        "vision_cache_over_budget_requests",
        "exqserve_engine_vision_cache_over_budget_requests",
        "Cumulative multimodal requests whose unique Vision working set exceeded the cache budget.",
    ),
    (
        "vision_cache_incomplete_prefix_retention_requests",
        "exqserve_engine_vision_cache_incomplete_prefix_retention_requests",
        "Cumulative multimodal requests without full persistent prefix identity retention.",
    ),
    (
        "vision_cache_last_request_unique_media_count",
        "exqserve_engine_vision_cache_last_request_unique_media_count",
        "Unique media count in the last committed multimodal request.",
    ),
    (
        "vision_cache_last_request_unique_media_bytes",
        "exqserve_engine_vision_cache_last_request_unique_media_bytes",
        "Retained-tensor bytes required by unique media in the last committed multimodal request.",
    ),
    (
        "vision_cache_last_request_retained_media_bytes",
        "exqserve_engine_vision_cache_last_request_retained_media_bytes",
        "Bytes of last-request Vision media retained after commit.",
    ),
    (
        "vision_cache_last_request_protected_prefix_entries",
        "exqserve_engine_vision_cache_last_request_protected_prefix_entries",
        "Prefix-priority media entries protected for the last committed multimodal request.",
    ),
    (
        "vision_cache_last_request_protected_prefix_bytes",
        "exqserve_engine_vision_cache_last_request_protected_prefix_bytes",
        "Prefix-priority Vision embedding bytes protected for the last committed multimodal request.",
    ),
    (
        "vision_cache_last_request_first_unretained_media_ordinal",
        "exqserve_engine_vision_cache_last_request_first_unretained_media_ordinal",
        "One-based ordinal of the first media item outside persistent prefix retention.",
    ),
)


class MetricsRegistry:
    """Own a private Prometheus registry so app/test instances never collide."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self._requests = Counter(
            "exqserve_requests",
            "Serving requests by bounded terminal status.",
            ("status",),
            registry=self.registry,
        )
        self._active = Gauge(
            "exqserve_active_requests",
            "Currently accepted serving requests.",
            registry=self.registry,
        )
        self._request_latency = Histogram(
            "exqserve_request_latency_seconds",
            "Accepted request latency through terminal outcome.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._ttfe = Histogram(
            "exqserve_time_to_first_semantic_event_seconds",
            "Time from submit start to first client-meaningful semantic event.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._tool_start = Histogram(
            "exqserve_time_to_tool_call_start_seconds",
            "Time from submit start to first tool-call start.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._backend_queue = Histogram(
            "exqserve_backend_queue_seconds",
            "Measured backend queue duration.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._backend_prefill = Histogram(
            "exqserve_backend_prefill_seconds",
            "Measured backend prefill duration.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._backend_generation = Histogram(
            "exqserve_backend_generation_seconds",
            "Measured backend generation duration.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._input_tokens = Counter(
            "exqserve_input_tokens",
            "Measured input tokens.",
            registry=self.registry,
        )
        self._cached_input_tokens = Counter(
            "exqserve_cached_input_tokens",
            "Measured cached input tokens.",
            registry=self.registry,
        )
        self._output_tokens = Counter(
            "exqserve_output_tokens",
            "Measured output tokens.",
            registry=self.registry,
        )
        self._cached_ratio = Histogram(
            "exqserve_cached_input_ratio",
            "Per-request measured cached input ratio.",
            buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
            registry=self.registry,
        )
        self._prefill_rate = Histogram(
            "exqserve_prefill_tokens_per_second",
            "Measured newly-prefilled tokens per backend prefill second.",
            buckets=_PREFILL_RATE_BUCKETS,
            registry=self.registry,
        )
        self._decode_rate = Histogram(
            "exqserve_decode_tokens_per_second",
            "Measured output tokens per backend generation second.",
            buckets=_DECODE_RATE_BUCKETS,
            registry=self.registry,
        )
        self._capture_failures = Counter(
            "exqserve_capture_failures",
            "Capture sink failures isolated from serving responses.",
            registry=self.registry,
        )
        self._recovery_attempts = Counter(
            "exqserve_recovery_attempts",
            "Started bounded inference recovery attempts by kind and failure cause.",
            ("kind", "cause"),
            registry=self.registry,
        )
        self._recovery_success = Counter(
            "exqserve_recovery_success",
            "Successful bounded inference recoveries by kind and original failure cause.",
            ("kind", "cause"),
            registry=self.registry,
        )
        self._recovery_exhausted = Counter(
            "exqserve_recovery_exhausted",
            "Bounded inference recoveries that exhausted their extra-attempt budget.",
            ("kind", "cause"),
            registry=self.registry,
        )
        self._recovery_skipped = Counter(
            "exqserve_recovery_skipped",
            "Inference recovery opportunities skipped for a bounded policy reason.",
            ("reason",),
            registry=self.registry,
        )
        self._unclassified_terminal = Counter(
            "exqserve_unclassified_terminal",
            "Serving terminal failures without a typed FailureCause.",
            ("surface",),
            registry=self.registry,
        )
        self._engine_state = Gauge(
            "exqserve_engine_state",
            "Current runtime Generator lifecycle state.",
            ("state",),
            registry=self.registry,
        )
        self._engine_numeric = {
            field: Gauge(metric, help_text, registry=self.registry)
            for field, metric, help_text in _ENGINE_NUMERIC_FIELDS
        }
        self._engine_stats_provider: Callable[[], RuntimeEngineStats] | None = None
        self._renderer_waiting = Gauge(
            "exqserve_renderer_waiting",
            "Prompt-preprocessing operations currently waiting for a renderer lane.",
            registry=self.registry,
        )
        self._renderer_in_flight = Gauge(
            "exqserve_renderer_in_flight",
            "Prompt-preprocessing worker operations currently executing.",
            registry=self.registry,
        )
        self._renderer_queue = Histogram(
            "exqserve_renderer_queue_seconds",
            "Time spent waiting for a renderer lane.",
            registry=self.registry,
        )
        self._renderer_execution = Histogram(
            "exqserve_renderer_execution_seconds",
            "Worker execution time after a renderer lane is acquired.",
            registry=self.registry,
        )
        self._renderer_requests = Counter(
            "exqserve_renderer_requests",
            "Prompt-preprocessing operations by bounded kind.",
            ("kind",),
            registry=self.registry,
        )

    def bind_engine_stats_provider(self, provider: Callable[[], RuntimeEngineStats]) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._engine_stats_provider = provider

    def _refresh_engine_stats(self) -> None:
        provider = self._engine_stats_provider
        try:
            stats = RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE) if provider is None else provider()
            if not isinstance(stats, RuntimeEngineStats):
                raise TypeError("engine stats provider must return RuntimeEngineStats")
        except Exception:
            logger.exception("Engine stats collection failed; exporting unavailable/NaN state.")
            stats = RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)

        for state in RuntimeEngineState:
            self._engine_state.labels(state=state.value).set(1 if state is stats.state else 0)
        for field, gauge in self._engine_numeric.items():
            value = getattr(stats, field)
            gauge.set(float("nan") if value is None else value)

    @staticmethod
    def _renderer_kind(kind: str) -> str:
        if kind not in _RENDERER_KINDS:
            raise ValueError("renderer kind must be chat, raw_text, or count_tokens")
        return kind

    def renderer_wait_started(self) -> None:
        self._renderer_waiting.inc()

    def renderer_wait_finished(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._renderer_waiting.dec()
        self._renderer_queue.observe(elapsed_seconds)

    def renderer_started(self, kind: str) -> None:
        normalized = self._renderer_kind(kind)
        self._renderer_in_flight.inc()
        self._renderer_requests.labels(kind=normalized).inc()

    def renderer_finished(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._renderer_in_flight.dec()
        self._renderer_execution.observe(elapsed_seconds)

    def request_started(self) -> None:
        self._active.inc()

    def request_rejected(self) -> None:
        self._requests.labels(status="rejected").inc()

    def request_failed_before_start(self) -> None:
        self._requests.labels(status="failed").inc()

    def capture_failed(self) -> None:
        self._capture_failures.inc()

    def recovery_attempt(self, kind: str, cause: str) -> None:
        self._recovery_attempts.labels(kind=kind, cause=cause).inc()

    def recovery_success(self, kind: str, cause: str) -> None:
        self._recovery_success.labels(kind=kind, cause=cause).inc()

    def recovery_exhausted(self, kind: str, cause: str) -> None:
        self._recovery_exhausted.labels(kind=kind, cause=cause).inc()

    def recovery_skipped(self, reason: str) -> None:
        self._recovery_skipped.labels(reason=reason).inc()

    def unclassified_terminal(self, surface: str) -> None:
        if surface not in {"serving", "submission"}:
            raise ValueError("surface must be serving or submission")
        self._unclassified_terminal.labels(surface=surface).inc()

    def request_finished(self, status: str, elapsed_seconds: float) -> None:
        if status not in _TERMINAL_STATUSES - {"rejected"}:
            raise ValueError("status must be completed, failed, or cancelled")
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._active.dec()
        self._requests.labels(status=status).inc()
        self._request_latency.observe(elapsed_seconds)

    def observe_ttfe(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._ttfe.observe(elapsed_seconds)

    def observe_tool_start(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._tool_start.observe(elapsed_seconds)

    def observe_backend(self, timing: GenerationTiming, usage: TokenUsage) -> None:
        if not isinstance(timing, GenerationTiming):
            raise TypeError("timing must be GenerationTiming")
        if not isinstance(usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")

        if timing.queue_seconds is not None:
            self._backend_queue.observe(timing.queue_seconds)
        if timing.prefill_seconds is not None:
            self._backend_prefill.observe(timing.prefill_seconds)
        if timing.generation_seconds is not None:
            self._backend_generation.observe(timing.generation_seconds)

        if usage.input_tokens is not None:
            self._input_tokens.inc(usage.input_tokens)
        if usage.cached_input_tokens is not None:
            self._cached_input_tokens.inc(usage.cached_input_tokens)
        if usage.output_tokens is not None:
            self._output_tokens.inc(usage.output_tokens)

        if (
            usage.input_tokens is not None
            and usage.input_tokens > 0
            and usage.cached_input_tokens is not None
        ):
            self._cached_ratio.observe(usage.cached_input_tokens / usage.input_tokens)

        if (
            timing.prefill_seconds is not None
            and timing.prefill_seconds > 0
            and usage.input_tokens is not None
            and usage.cached_input_tokens is not None
        ):
            new_prefill = usage.input_tokens - usage.cached_input_tokens
            self._prefill_rate.observe(new_prefill / timing.prefill_seconds)

        if (
            timing.generation_seconds is not None
            and timing.generation_seconds > 0
            and usage.output_tokens is not None
        ):
            self._decode_rate.observe(usage.output_tokens / timing.generation_seconds)

    def render(self) -> bytes:
        self._refresh_engine_stats()
        return generate_latest(self.registry)

    def render_text(self) -> str:
        return self.render().decode("utf-8")
