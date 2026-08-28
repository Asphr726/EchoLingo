from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ...models import CanonicalTranscriptEvent


_END = object()


class AsrEventQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[CanonicalTranscriptEvent | object] = asyncio.Queue()
        self._ended = False

    async def put(self, event: CanonicalTranscriptEvent) -> None:
        if not self._ended:
            await self._queue.put(event)

    async def end(self) -> None:
        if not self._ended:
            self._ended = True
            await self._queue.put(_END)

    async def events(self) -> AsyncIterator[CanonicalTranscriptEvent]:
        while True:
            event = await self._queue.get()
            if event is _END:
                break
            assert isinstance(event, CanonicalTranscriptEvent)
            yield event

