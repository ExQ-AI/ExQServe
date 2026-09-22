"""Single-owner ASGI disconnect race for non-streaming protocol responses."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from fastapi import Request


async def _wait_for_disconnect(request: Request) -> None:
    """Own request.receive() after the request body has been fully consumed."""

    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _cleanup_cancellable_result(value: object) -> bool:
    cancel_result = getattr(value, "cancel", None)
    if not callable(cancel_result):
        return False
    cleanup_task = asyncio.ensure_future(cancel_result())
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            continue
    cleanup_task.result()
    return True


async def race_nonstream_disconnect[T](request: Request, result: Awaitable[T]) -> T:
    """Race request work against disconnect while preserving terminal-vs-setup semantics.

    Non-stream terminal values (for example a response dictionary) win a simultaneous
    disconnect. A returned cancellable session is still pre-response setup, so a
    disconnect consumed in the same turn cancels that session instead of handing it off.
    """

    result_task = asyncio.ensure_future(result)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    handed_off = False
    cleaned = False
    try:
        done, _ = await asyncio.wait(
            {result_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if result_task in done:
            value = await result_task
            if disconnect_task in done:
                cleaned = await _cleanup_cancellable_result(value)
                if cleaned:
                    raise asyncio.CancelledError
            handed_off = True
            return value

        result_task.cancel()
        await asyncio.gather(result_task, return_exceptions=True)
        raise asyncio.CancelledError
    finally:
        for task in (disconnect_task, result_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(disconnect_task, result_task, return_exceptions=True)

        if not handed_off and not cleaned and result_task.done() and not result_task.cancelled():
            exception = result_task.exception()
            if exception is None:
                await _cleanup_cancellable_result(result_task.result())
