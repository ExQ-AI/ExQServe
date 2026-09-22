from __future__ import annotations

from exqserve.runtime.exllamav3 import _trim_cuda_allocator_cache_if_pressured

_MIB = 1024**2
_GIB = 1024**3


class _FakeMemory:
    def __init__(self, owner: _FakeCuda) -> None:
        self.owner = owner

    def get_allocator_backend(self) -> str:
        self.owner.backend_calls += 1
        if self.owner.backend_failure:
            raise RuntimeError("backend lookup failed")
        return self.owner.backend


class _FakeCuda:
    def __init__(
        self,
        *,
        free_bytes: int = 640 * _MIB,
        allocated_bytes: int = 20 * _GIB,
        reserved_bytes: int = 20 * _GIB + 720 * _MIB,
        backend: str = "cudaMallocAsync",
        backend_api: bool = True,
        backend_failure: bool = False,
        query_failure: bool = False,
        trim_failure: bool = False,
        device_stats: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        self.free_bytes = free_bytes
        self.allocated_bytes = allocated_bytes
        self.reserved_bytes = reserved_bytes
        self.backend = backend
        self.backend_failure = backend_failure
        self.query_failure = query_failure
        self.trim_failure = trim_failure
        self.device_stats = device_stats
        self.backend_calls = 0
        self.query_calls = 0
        self.trim_calls = 0
        self.memory = _FakeMemory(self) if backend_api else object()

    def is_available(self) -> bool:
        return True

    def _stats(self, device_id: int) -> tuple[int, int, int]:
        if self.device_stats is not None:
            return self.device_stats[device_id]
        return self.free_bytes, self.allocated_bytes, self.reserved_bytes

    def mem_get_info(self, device_id: int) -> tuple[int, int]:
        self.query_calls += 1
        if self.query_failure:
            raise RuntimeError("accounting failed")
        free_bytes, _, _ = self._stats(device_id)
        return free_bytes, 24 * _GIB

    def memory_allocated(self, device_id: int) -> int:
        _, allocated_bytes, _ = self._stats(device_id)
        return allocated_bytes

    def memory_reserved(self, device_id: int) -> int:
        _, _, reserved_bytes = self._stats(device_id)
        return reserved_bytes

    def empty_cache(self) -> None:
        self.trim_calls += 1
        if self.trim_failure:
            raise RuntimeError("trim failed")


def _run(
    cuda: _FakeCuda,
    *,
    device_ids: tuple[int, ...] = (0,),
    active_jobs: int = 0,
    pending_jobs: int = 0,
) -> bool:
    return _trim_cuda_allocator_cache_if_pressured(
        cuda,
        device_ids,
        active_jobs=active_jobs,
        pending_jobs=pending_jobs,
    )


def test_idle_low_headroom_substantial_cached_free_trims_once() -> None:
    cuda = _FakeCuda()

    assert _run(cuda) is True
    assert cuda.trim_calls == 1


def test_two_pressured_devices_still_trigger_one_allocator_global_trim() -> None:
    cuda = _FakeCuda(
        device_stats={
            0: (640 * _MIB, 20 * _GIB, 20 * _GIB + 720 * _MIB),
            1: (512 * _MIB, 18 * _GIB, 18 * _GIB + 640 * _MIB),
        }
    )

    assert _run(cuda, device_ids=(0, 1)) is True
    assert cuda.trim_calls == 1


def test_one_pressured_device_triggers_one_global_trim_after_scanning_configured_devices() -> None:
    cuda = _FakeCuda(
        device_stats={
            0: (1536 * _MIB, 20 * _GIB, 20 * _GIB + 720 * _MIB),
            1: (640 * _MIB, 18 * _GIB, 18 * _GIB + 640 * _MIB),
        }
    )

    assert _run(cuda, device_ids=(0, 1)) is True
    assert cuda.trim_calls == 1


def test_native_allocator_fails_closed_before_pressure_accounting() -> None:
    cuda = _FakeCuda(backend="native")

    assert _run(cuda) is False
    assert cuda.backend_calls == 1
    assert cuda.query_calls == 0
    assert cuda.trim_calls == 0


def test_unknown_allocator_backend_api_fails_closed() -> None:
    cuda = _FakeCuda(backend_api=False)

    assert _run(cuda) is False
    assert cuda.query_calls == 0
    assert cuda.trim_calls == 0


def test_allocator_backend_lookup_failure_fails_safe() -> None:
    cuda = _FakeCuda(backend_failure=True)

    assert _run(cuda) is False
    assert cuda.query_calls == 0
    assert cuda.trim_calls == 0


def test_idle_healthy_headroom_does_not_trim() -> None:
    cuda = _FakeCuda(free_bytes=1536 * _MIB)

    assert _run(cuda) is False
    assert cuda.trim_calls == 0


def test_idle_reclaimable_below_policy_threshold_does_not_trim() -> None:
    cuda = _FakeCuda(reserved_bytes=20 * _GIB + 200 * _MIB)

    assert _run(cuda) is False
    assert cuda.trim_calls == 0


def test_active_backend_job_does_not_query_or_trim() -> None:
    cuda = _FakeCuda(free_bytes=128 * _MIB, reserved_bytes=21 * _GIB)

    assert _run(cuda, active_jobs=1) is False
    assert cuda.backend_calls == 0
    assert cuda.query_calls == 0
    assert cuda.trim_calls == 0


def test_pending_backend_job_does_not_query_or_trim() -> None:
    cuda = _FakeCuda(free_bytes=128 * _MIB, reserved_bytes=21 * _GIB)

    assert _run(cuda, pending_jobs=1) is False
    assert cuda.backend_calls == 0
    assert cuda.query_calls == 0
    assert cuda.trim_calls == 0


def test_cuda_accounting_failure_fails_safe() -> None:
    cuda = _FakeCuda(
        free_bytes=128 * _MIB,
        reserved_bytes=21 * _GIB,
        query_failure=True,
    )

    assert _run(cuda) is False
    assert cuda.trim_calls == 0


def test_cuda_trim_failure_fails_safe() -> None:
    cuda = _FakeCuda(
        free_bytes=128 * _MIB,
        reserved_bytes=21 * _GIB,
        trim_failure=True,
    )

    assert _run(cuda) is False
    assert cuda.trim_calls == 1
