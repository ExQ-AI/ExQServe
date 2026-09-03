from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import GenerationEvent
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.model import ServedModelInfo
from exqserve.core.request import CanonicalRequest
from exqserve.server.model_manager import ActiveModelBundle, ModelManager, ModelManagerState
from exqserve.serving.contracts import RawServingRequest, ServingRejected, ServingRequest


class _Session:
    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        async def stream() -> AsyncIterator[GenerationEvent]:
            if False:
                yield  # pragma: no cover
        return stream()

    async def cancel(self) -> None:
        return None


class _Engine:
    async def submit(self, request: ServingRequest) -> _Session:
        return _Session()

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return 1


class _BlockingEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.two_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, request: ServingRequest) -> _Session:
        self.calls += 1
        if self.calls == 2:
            self.two_entered.set()
        await self.release.wait()
        return _Session()

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return 1


class _CloseAwareEngine:
    def __init__(self, close_event: asyncio.Event) -> None:
        self.close_event = close_event
        self.entered = asyncio.Event()

    async def submit(self, request: ServingRequest) -> _Session:
        self.entered.set()
        await self.close_event.wait()
        raise RuntimeError("controller closed")

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return 1


class _RawEngine:
    async def submit(self, request: RawServingRequest) -> _Session:
        return _Session()


class _Controller:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log
        self.in_flight = 0

    async def close(self) -> None:
        self.log.append(f"controller:{self.name}")


class _CloseAwareController(_Controller):
    def __init__(self, name: str, log: list[str], close_event: asyncio.Event) -> None:
        super().__init__(name, log)
        self.close_event = close_event

    async def close(self) -> None:
        await super().close()
        self.close_event.set()


class _FailingController(_Controller):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("controller close failed")


class _Runtime:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log
        self.is_ready = True
        self.is_healthy = True

    async def close(self) -> None:
        self.log.append(f"runtime:{self.name}")
        self.is_ready = False


class _UnresolvedCloseRuntime(_Runtime):
    async def close(self) -> None:
        self.log.append(f"runtime:{self.name}")
        self.is_ready = False
        self.is_healthy = False
        raise RuntimeError("restart the server process")


def _bundle(name: str, log: list[str]) -> ActiveModelBundle:
    return ActiveModelBundle(
        management_id=name,
        served_model=ServedModelInfo(name, created=1, context_length=4096),
        runtime=_Runtime(name, log),
        controller=_Controller(name, log),
        engine=_Engine(),
        raw_engine=_RawEngine(),
    )


def _unresolved_bundle(name: str, log: list[str]) -> ActiveModelBundle:
    return ActiveModelBundle(
        management_id=name,
        served_model=ServedModelInfo(name, created=1, context_length=4096),
        runtime=_UnresolvedCloseRuntime(name, log),
        controller=_Controller(name, log),
        engine=_Engine(),
        raw_engine=_RawEngine(),
    )


def _request(model: str = "first", request_id: str = "req") -> ServingRequest:
    return ServingRequest(
        CanonicalRequest(
            request_id,
            model,
            (MessageItem(MessageRole.USER, "hi"),),
        ),
        ReasoningPolicy(),
        ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), False),
        max_output_tokens=1,
    )


def test_switch_closes_old_controller_before_runtime_and_publishes_new_model(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        built: list[str] = []

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            built.append(model_id)
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            _bundle("first", log),
            build,
        )
        result = await manager.switch("second")

        assert result.state is ModelManagerState.READY
        assert result.current_model == "second"
        assert manager.current_model() is not None
        assert manager.current_model().id == "second"  # type: ignore[union-attr]
        assert built == ["second"]
        assert log == ["controller:first", "runtime:first"]

    asyncio.run(scenario())


def test_unload_keeps_manager_alive_and_load_recovers(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            _bundle("first", log),
            build,
        )
        unloaded = await manager.unload()
        assert unloaded.state is ModelManagerState.UNLOADED
        assert manager.current_model() is None
        assert manager.is_ready is False

        loaded = await manager.load("second")
        assert loaded.state is ModelManagerState.READY
        assert loaded.current_model == "second"
        assert manager.is_ready is True

    asyncio.run(scenario())


def test_unload_close_failure_preserves_bundle_and_blocks_reload(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        built: list[str] = []
        initial = _unresolved_bundle("first", log)

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            built.append(model_id)
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            initial,
            build,
        )

        with pytest.raises(RuntimeError, match="restart the server process"):
            await manager.unload()

        assert manager.state is ModelManagerState.ERROR
        assert manager.current_runtime is initial.runtime
        assert manager.current_model() == initial.served_model
        assert manager.is_ready is False
        with pytest.raises(RuntimeError, match="prior runtime ownership is unresolved"):
            await manager.load("second")
        assert built == []

    asyncio.run(scenario())


def test_switch_close_failure_preserves_bundle_and_blocks_replacement(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        built: list[str] = []
        initial = _unresolved_bundle("first", log)

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            built.append(model_id)
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            initial,
            build,
        )

        with pytest.raises(RuntimeError, match="restart the server process"):
            await manager.switch("second")

        assert manager.state is ModelManagerState.ERROR
        assert manager.current_runtime is initial.runtime
        assert manager.current_model() == initial.served_model
        assert manager.is_ready is False
        with pytest.raises(RuntimeError, match="prior runtime ownership is unresolved"):
            await manager.load("second")
        assert built == []

    asyncio.run(scenario())


def test_controller_close_failure_does_not_poison_released_runtime_ownership(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        built: list[str] = []
        initial = ActiveModelBundle(
            management_id="first",
            served_model=ServedModelInfo("first", created=1, context_length=4096),
            runtime=_Runtime("first", log),
            controller=_FailingController("first", log),
            engine=_Engine(),
            raw_engine=_RawEngine(),
        )

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            built.append(model_id)
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            initial,
            build,
        )

        with pytest.raises(RuntimeError, match="controller close failed"):
            await manager.unload()

        assert manager.state is ModelManagerState.ERROR
        assert manager.current_runtime is None
        assert manager.current_model() is None
        loaded = await manager.load("second")
        assert loaded.state is ModelManagerState.READY
        assert built == ["second"]
        assert log[:2] == ["controller:first", "runtime:first"]

    asyncio.run(scenario())


def test_switch_builder_failure_remains_reloadable_after_successful_close(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        attempts = 0

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("build failed")
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            _bundle("first", log),
            build,
        )

        with pytest.raises(RuntimeError, match="build failed"):
            await manager.switch("second")

        assert manager.state is ModelManagerState.ERROR
        assert manager.current_runtime is None
        assert manager.current_model() is None
        loaded = await manager.load("second")
        assert loaded.state is ModelManagerState.READY
        assert loaded.current_model == "second"
        assert attempts == 2

    asyncio.run(scenario())


def test_unknown_or_unsafe_model_id_is_rejected_without_build(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        builds = 0

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            nonlocal builds
            builds += 1
            return _bundle(model_id, log)

        manager = ModelManager({"first": tmp_path / "first"}, _bundle("first", log), build)
        with pytest.raises(KeyError):
            await manager.switch("../outside")
        assert builds == 0

    asyncio.run(scenario())


def test_concurrent_submissions_do_not_serialize_on_model_lifecycle_lock(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        engine = _BlockingEngine()
        bundle = ActiveModelBundle(
            management_id="first",
            served_model=ServedModelInfo("first", created=1, context_length=4096),
            runtime=_Runtime("first", log),
            controller=_Controller("first", log),
            engine=engine,
            raw_engine=_RawEngine(),
        )

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            return _bundle(model_id, log)

        manager = ModelManager({"first": tmp_path / "first"}, bundle, build)
        first = asyncio.create_task(manager.submit(_request(request_id="req-1")))
        second = asyncio.create_task(manager.submit(_request(request_id="req-2")))
        await asyncio.wait_for(engine.two_entered.wait(), timeout=0.5)
        assert engine.calls == 2
        engine.release.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())


def test_switch_can_close_controller_while_submission_is_waiting_in_engine(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        close_event = asyncio.Event()
        controller = _CloseAwareController("first", log, close_event)
        engine = _CloseAwareEngine(close_event)
        bundle = ActiveModelBundle(
            management_id="first",
            served_model=ServedModelInfo("first", created=1, context_length=4096),
            runtime=_Runtime("first", log),
            controller=controller,
            engine=engine,
            raw_engine=_RawEngine(),
        )

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            bundle,
            build,
        )
        submission = asyncio.create_task(manager.submit(_request()))
        await engine.entered.wait()
        switching = asyncio.create_task(manager.switch("second"))
        await asyncio.wait_for(close_event.wait(), timeout=0.5)
        assert manager.state is ModelManagerState.SWITCHING
        with pytest.raises(RuntimeError, match="controller closed"):
            await submission
        result = await asyncio.wait_for(switching, timeout=0.5)
        assert result.state is ModelManagerState.READY
        assert result.current_model == "second"
        assert log[:2] == ["controller:first", "runtime:first"]

    asyncio.run(scenario())


def test_close_during_switch_cannot_resurrect_manager(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        build_entered = asyncio.Event()
        release_build = asyncio.Event()

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            build_entered.set()
            await release_build.wait()
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            _bundle("first", log),
            build,
        )
        switching = asyncio.create_task(manager.switch("second"))
        await asyncio.wait_for(build_entered.wait(), timeout=0.5)

        closing = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not closing.done()

        release_build.set()
        switched = await asyncio.wait_for(switching, timeout=0.5)
        assert switched.state is ModelManagerState.READY
        await asyncio.wait_for(closing, timeout=0.5)

        assert manager.state is ModelManagerState.CLOSED
        assert manager.current_model() is None
        assert log == [
            "controller:first",
            "runtime:first",
            "controller:second",
            "runtime:second",
        ]

    asyncio.run(scenario())


def test_close_during_load_cannot_resurrect_manager(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        build_entered = asyncio.Event()
        release_build = asyncio.Event()

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            build_entered.set()
            await release_build.wait()
            return _bundle(model_id, log)

        manager = ModelManager(
            {"first": tmp_path / "first", "second": tmp_path / "second"},
            _bundle("first", log),
            build,
        )
        await manager.unload()

        loading = asyncio.create_task(manager.load("second"))
        await asyncio.wait_for(build_entered.wait(), timeout=0.5)
        closing = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not closing.done()

        release_build.set()
        loaded = await asyncio.wait_for(loading, timeout=0.5)
        assert loaded.state is ModelManagerState.READY
        await asyncio.wait_for(closing, timeout=0.5)

        assert manager.state is ModelManagerState.CLOSED
        assert manager.current_model() is None
        assert log == [
            "controller:first",
            "runtime:first",
            "controller:second",
            "runtime:second",
        ]

    asyncio.run(scenario())


def test_submit_rejects_retryably_while_unloaded(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []

        async def build(model_id: str, _path: Path) -> ActiveModelBundle:
            return _bundle(model_id, log)

        manager = ModelManager({"first": tmp_path / "first"}, _bundle("first", log), build)
        await manager.unload()
        request = ServingRequest(
            CanonicalRequest(
                "req",
                "first",
                (MessageItem(MessageRole.USER, "hi"),),
            ),
            ReasoningPolicy(),
            ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), False),
            max_output_tokens=1,
        )
        with pytest.raises(ServingRejected) as raised:
            await manager.submit(request)
        assert raised.value.error.code == "model_not_ready"
        assert raised.value.error.retryable is True

    asyncio.run(scenario())
