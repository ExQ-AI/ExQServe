"""Low-cardinality Prometheus metrics for protocol-neutral serving."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected"})


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
            registry=self.registry,
        )
        self._ttfe = Histogram(
            "exqserve_time_to_first_semantic_event_seconds",
            "Time from submit start to first client-meaningful semantic event.",
            registry=self.registry,
        )
        self._tool_start = Histogram(
            "exqserve_time_to_tool_call_start_seconds",
            "Time from submit start to first tool-call start.",
            registry=self.registry,
        )
        self._backend_queue = Histogram(
            "exqserve_backend_queue_seconds",
            "Measured backend queue duration.",
            registry=self.registry,
        )
        self._backend_prefill = Histogram(
            "exqserve_backend_prefill_seconds",
            "Measured backend prefill duration.",
            registry=self.registry,
        )
        self._backend_generation = Histogram(
            "exqserve_backend_generation_seconds",
            "Measured backend generation duration.",
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
            registry=self.registry,
        )
        self._decode_rate = Histogram(
            "exqserve_decode_tokens_per_second",
            "Measured output tokens per backend generation second.",
            registry=self.registry,
        )
        self._capture_failures = Counter(
            "exqserve_capture_failures",
            "Capture sink failures isolated from serving responses.",
            registry=self.registry,
        )

    def request_started(self) -> None:
        self._active.inc()

    def request_rejected(self) -> None:
        self._requests.labels(status="rejected").inc()

    def request_failed_before_start(self) -> None:
        self._requests.labels(status="failed").inc()

    def capture_failed(self) -> None:
        self._capture_failures.inc()

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
        return generate_latest(self.registry)

    def render_text(self) -> str:
        return self.render().decode("utf-8")
