from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from exqserve.runtime.contracts import ExLlamaV3LoadConfig, RuntimeGenerationRequest
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime, RuntimeSession
from tests.runtime.test_adapter import _backend, _FakeTorch, _reset_factories


class _RecoveryIterationTask:
    def __init__(self) -> None:
        self._done = False
        self._callbacks: list[object] = []

    def done(self) -> bool:
        return self._done

    def add_done_callback(self, callback: object) -> None:
        self._callbacks.append(callback)

    def finish(self) -> None:
        self._done = True
        for callback in tuple(self._callbacks):
            assert callable(callback)
            callback(self)


class _RecoverySyncGenerator:
    def __init__(self) -> None:
        self.clear_queue_calls = 0
        self.clear_queue_error: BaseException | None = None

    def clear_queue(self) -> None:
        self.clear_queue_calls += 1
        if self.clear_queue_error is not None:
            raise self.clear_queue_error


class _RecoveryAsyncGenerator:
    instances: ClassVar[list[_RecoveryAsyncGenerator]] = []
    fail_construction_at: ClassVar[int | None] = None
    start_dead_at: ClassVar[int | None] = None
    expose_iteration_task: ClassVar[bool] = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        index = len(type(self).instances) + 1
        if type(self).fail_construction_at == index:
            raise RuntimeError("replacement construction failed")
        self.args = args
        self.kwargs = kwargs
        self.error: BaseException | None = None
        if type(self).start_dead_at == index:
            self.error = RuntimeError("replacement started dead")
        self.generator = _RecoverySyncGenerator()
        self.close_calls = 0
        self.close_error: BaseException | None = None
        self.close_entered: asyncio.Event | None = None
        self.close_release: asyncio.Event | None = None
        if type(self).expose_iteration_task:
            self.iteration_task = _RecoveryIterationTask()
        type(self).instances.append(self)

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_entered is not None:
            self.close_entered.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


def _reset_recovery_generators() -> None:
    _RecoveryAsyncGenerator.instances.clear()
    _RecoveryAsyncGenerator.fail_construction_at = None
    _RecoveryAsyncGenerator.start_dead_at = None
    _RecoveryAsyncGenerator.expose_iteration_task = False


def _make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config: ExLlamaV3LoadConfig | None = None,
) -> ExLlamaV3Runtime:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    _reset_recovery_generators()
    backend = _backend()
    backend.AsyncGenerator = _RecoveryAsyncGenerator
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(config or ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
    return runtime


def _submit(runtime: ExLlamaV3Runtime, request_id: str) -> RuntimeSession:
    return runtime.submit(RuntimeGenerationRequest(request_id, (1, 2, 3), 8))


async def _poison_and_wait(runtime: ExLlamaV3Runtime, generator: _RecoveryAsyncGenerator) -> None:
    generator.error = RuntimeError("shared generator failed")
    runtime._begin_generator_recovery(generator)
    task = runtime._recovery_task
    assert task is not None
    await task


def test_generator_recovery_replaces_only_generator_and_reuses_loaded_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        resources = runtime._resources
        assert resources is not None
        model = resources.model
        cache = resources.cache
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]

        failed.error = RuntimeError("shared generator failed")
        runtime._begin_generator_recovery(failed)

        assert runtime._generator_state is module._GeneratorLifecycleState.RECOVERING
        assert runtime._generator is None
        assert runtime._quarantined_generator is failed
        task = runtime._recovery_task
        assert task is not None
        await task

        replacement = runtime._generator
        assert isinstance(replacement, _RecoveryAsyncGenerator)
        assert replacement is not failed
        assert runtime._generator_state is module._GeneratorLifecycleState.READY
        assert runtime.is_healthy is True
        assert runtime._quarantined_generator is None
        assert runtime._resources is resources
        assert runtime._resources.model is model
        assert runtime._resources.cache is cache
        assert failed.close_calls == 1
        assert failed.generator.clear_queue_calls == 1
        assert replacement.args == failed.args
        assert replacement.kwargs == failed.kwargs
        await runtime.close()

    asyncio.run(scenario())


def test_generator_recovery_supports_two_independent_poison_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        first = _RecoveryAsyncGenerator.instances[-1]
        await _poison_and_wait(runtime, first)
        second = runtime._generator
        assert isinstance(second, _RecoveryAsyncGenerator)
        await _poison_and_wait(runtime, second)
        third = runtime._generator

        assert isinstance(third, _RecoveryAsyncGenerator)
        assert len(_RecoveryAsyncGenerator.instances) == 3
        assert first.close_calls == 1
        assert first.generator.clear_queue_calls == 1
        assert second.close_calls == 1
        assert second.generator.clear_queue_calls == 1
        assert runtime.is_healthy is True
        await runtime.close()

    asyncio.run(scenario())


def test_duplicate_failure_notifications_start_only_one_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        session = _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.error = RuntimeError("shared generator failed")

        callback = session._on_backend_failure
        assert callable(callback)
        callback()
        first_task = runtime._recovery_task
        assert first_task is not None
        callback()
        assert runtime._recovery_task is first_task
        await first_task

        callback()
        assert len(_RecoveryAsyncGenerator.instances) == 2
        assert runtime.is_healthy is True
        await runtime.close()

    asyncio.run(scenario())


def test_non_poison_failure_notification_does_not_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        session = _submit(runtime, "req-a")
        generator = _RecoveryAsyncGenerator.instances[-1]
        callback = session._on_backend_failure
        assert callable(callback)

        callback()

        assert runtime._generator is generator
        assert runtime._recovery_task is None
        assert len(_RecoveryAsyncGenerator.instances) == 1
        assert runtime.is_healthy is True
        await runtime.close()

    asyncio.run(scenario())


def test_submit_time_known_poison_guard_starts_recovery_and_rejects_racing_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.error = RuntimeError("shared generator failed")

        with pytest.raises(RuntimeError, match="recovering"):
            _submit(runtime, "req-b")

        assert runtime._generator_state is module._GeneratorLifecycleState.RECOVERING
        assert runtime._quarantined_generator is failed
        task = runtime._recovery_task
        assert task is not None
        await task
        assert runtime._generator_state is module._GeneratorLifecycleState.READY
        assert len(_RecoveryAsyncGenerator.instances) == 2
        await runtime.close()

    asyncio.run(scenario())


def test_iteration_task_observer_proactively_starts_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _RecoveryAsyncGenerator.expose_iteration_task = True
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.error = RuntimeError("shared generator failed")
        iteration_task = failed.iteration_task

        iteration_task.finish()

        assert runtime._generator_state is module._GeneratorLifecycleState.RECOVERING
        task = runtime._recovery_task
        assert task is not None
        await task
        assert runtime._generator_state is module._GeneratorLifecycleState.READY
        assert runtime.is_healthy is True
        assert len(_RecoveryAsyncGenerator.instances) == 2
        await runtime.close()

    asyncio.run(scenario())


def test_failed_quiescence_retains_quarantine_until_close_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.close_error = RuntimeError("close failed")
        await _poison_and_wait(runtime, failed)

        assert runtime._generator_state is module._GeneratorLifecycleState.FAILED
        assert runtime._generator is None
        assert runtime._quarantined_generator is failed
        assert runtime._resources is not None
        assert len(_RecoveryAsyncGenerator.instances) == 1

        failed.close_error = None
        await runtime.close()

        assert failed.close_calls == 2
        assert failed.generator.clear_queue_calls == 1
        assert runtime._resources is None
        assert runtime._quarantined_generator is None

    asyncio.run(scenario())


def test_failed_queue_cleanup_retains_quarantine_until_close_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.generator.clear_queue_error = RuntimeError("clear failed")
        await _poison_and_wait(runtime, failed)

        assert runtime._generator_state is module._GeneratorLifecycleState.FAILED
        assert runtime._quarantined_generator is failed
        assert failed.close_calls == 1
        assert failed.generator.clear_queue_calls == 1

        failed.generator.clear_queue_error = None
        await runtime.close()

        assert failed.close_calls == 2
        assert failed.generator.clear_queue_calls == 2
        assert runtime._resources is None

    asyncio.run(scenario())


def test_close_during_recovery_prevents_replacement_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        failed.close_entered = asyncio.Event()
        failed.close_release = asyncio.Event()
        failed.error = RuntimeError("shared generator failed")
        runtime._begin_generator_recovery(failed)
        recovery_task = runtime._recovery_task
        assert recovery_task is not None

        await failed.close_entered.wait()
        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert runtime._closing is True
        assert close_task.done() is False

        failed.close_release.set()
        await close_task

        assert recovery_task.done() is True
        assert len(_RecoveryAsyncGenerator.instances) == 1
        assert failed.generator.clear_queue_calls == 1
        assert runtime._resources is None
        assert runtime._generator is None
        assert runtime._quarantined_generator is None
        assert runtime._recovery_task is None

    asyncio.run(scenario())


def test_sysmem_kv_poison_fails_closed_without_constructing_second_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(
            monkeypatch,
            ExLlamaV3LoadConfig(
                "/models/qwen",
                cache_tokens=1024,
                sysmem_kv_cache_mb=512,
            ),
        )
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        await _poison_and_wait(runtime, failed)

        assert runtime._generator_state is module._GeneratorLifecycleState.FAILED
        assert runtime.is_healthy is False
        assert runtime._generator is None
        assert runtime._quarantined_generator is failed
        assert len(_RecoveryAsyncGenerator.instances) == 1
        assert failed.close_calls == 1
        assert failed.generator.clear_queue_calls == 1
        with pytest.raises(RuntimeError, match="unhealthy"):
            _submit(runtime, "req-b")
        with pytest.raises(RuntimeError, match="restart the server process"):
            await runtime.close()
        assert runtime._resources is not None
        assert runtime._quarantined_generator is failed
        assert len(_RecoveryAsyncGenerator.instances) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "config",
    [
        ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024, mtp_enabled=True),
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            draft_model_directory="/models/draft",
            draft_tokens=4,
        ),
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            ngram_match_min=3,
            ngram_draft_size=4,
        ),
    ],
    ids=("mtp", "external-draft", "ngram"),
)
def test_recovery_uses_identical_generator_configuration(
    monkeypatch: pytest.MonkeyPatch,
    config: ExLlamaV3LoadConfig,
) -> None:
    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch, config)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        await _poison_and_wait(runtime, failed)
        replacement = runtime._generator

        assert isinstance(replacement, _RecoveryAsyncGenerator)
        assert replacement.args == failed.args
        assert replacement.kwargs == failed.kwargs
        await runtime.close()

    asyncio.run(scenario())


def test_known_dead_replacement_is_never_published_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        _RecoveryAsyncGenerator.start_dead_at = 2
        await _poison_and_wait(runtime, failed)

        dead_replacement = _RecoveryAsyncGenerator.instances[-1]
        assert len(_RecoveryAsyncGenerator.instances) == 2
        assert runtime._generator_state is module._GeneratorLifecycleState.FAILED
        assert runtime._generator is None
        assert runtime.is_healthy is False
        assert dead_replacement.close_calls == 1
        assert dead_replacement.generator.clear_queue_calls == 1
        await runtime.close()

    asyncio.run(scenario())


def test_replacement_factory_failure_leaves_runtime_failed_but_closable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    async def scenario() -> None:
        runtime = _make_runtime(monkeypatch)
        _submit(runtime, "req-a")
        failed = _RecoveryAsyncGenerator.instances[-1]
        _RecoveryAsyncGenerator.fail_construction_at = 2
        await _poison_and_wait(runtime, failed)

        assert runtime._generator_state is module._GeneratorLifecycleState.FAILED
        assert runtime._generator is None
        assert runtime._quarantined_generator is None
        assert len(_RecoveryAsyncGenerator.instances) == 1
        await runtime.close()
        assert runtime._resources is None

    asyncio.run(scenario())
