from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from .protocol import PROTOCOL_VERSION, ProtocolError, decode_audio_packet
from .session import DesktopInferenceSession


def _event(event_type: str, payload: Any) -> str:
    return json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False)


class SidecarConnection:
    def __init__(self, websocket: ServerConnection, token: str) -> None:
        self.websocket = websocket
        self.expected_token = token
        self.authenticated = False
        self.session: DesktopInferenceSession | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sender: asyncio.Task[None] | None = None
        self.background_tasks: set[asyncio.Task[Any]] = set()

    async def run(self) -> None:
        self.sender = asyncio.create_task(self._send_events())
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    await self._audio(message)
                else:
                    await self._command(json.loads(message))
        except (json.JSONDecodeError, ProtocolError, KeyError, ValueError) as error:
            await self.websocket.send(
                _event("error", {"code": "invalid_request", "message": str(error), "recoverable": True})
            )
        finally:
            if self.session is not None:
                await self.session.close()
            for task in self.background_tasks:
                task.cancel()
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            if self.sender is not None:
                self.sender.cancel()
                await asyncio.gather(self.sender, return_exceptions=True)

    async def _send_events(self) -> None:
        while True:
            await self.websocket.send(json.dumps(await self.events.get(), ensure_ascii=False))

    async def _command(self, command: dict[str, Any]) -> None:
        command_type = command.get("type")
        payload = command.get("payload", {})
        if not self.authenticated:
            if command_type != "hello":
                raise ProtocolError("hello must be the first sidecar command")
            if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
                raise ProtocolError("sidecar protocol version mismatch")
            if payload.get("authentication_token") != self.expected_token:
                raise ProtocolError("sidecar authentication failed")
            self.authenticated = True
            await self.websocket.send(
                _event("hello_accepted", {"protocol_version": PROTOCOL_VERSION})
            )
            return
        if command_type == "start_session":
            if self.session is not None:
                raise ProtocolError("a sidecar session is already active")
            self.session = await DesktopInferenceSession.create(payload, self.events)
            await self.websocket.send(
                _event("ready", {"session_id": payload["session_id"], "route": self.session.route})
            )
        elif command_type == "pause":
            self._require_session().paused = True
        elif command_type == "resume":
            self._require_session().paused = False
        elif command_type == "finish_session":
            session = self._require_session()
            await asyncio.wait_for(session.finish(), timeout=10.0)
            await self.websocket.send(
                _event("session_finished", {"session_id": payload["session_id"]})
            )
            self.session = None
            task = asyncio.create_task(self._run_alignment(session))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        elif command_type == "shutdown":
            await self.websocket.close(code=1000, reason="sidecar shutdown")
        else:
            raise ProtocolError(f"unknown sidecar command: {command_type}")

    async def _audio(self, data: bytes) -> None:
        if not self.authenticated:
            raise ProtocolError("audio received before authentication")
        await self._require_session().push(decode_audio_packet(data))

    def _require_session(self) -> DesktopInferenceSession:
        if self.session is None:
            raise ProtocolError("no active sidecar session")
        return self.session

    async def _run_alignment(self, session: DesktopInferenceSession) -> None:
        try:
            await session.align()
        except asyncio.CancelledError:
            if session.alignment_capture is not None:
                session.alignment_capture.discard()
            raise
        except Exception as error:
            if session.alignment_capture is not None:
                session.alignment_capture.discard()
            await self.events.put(
                {
                    "type": "error",
                    "payload": {
                        "code": "alignment_failed",
                        "message": str(error),
                        "recoverable": True,
                    },
                }
            )


async def run_server(host: str, port: int, token: str) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("inference sidecar must bind to loopback")
    async with serve(lambda ws: SidecarConnection(ws, token).run(), host, port) as server:
        sockets = server.sockets or []
        selected_port = sockets[0].getsockname()[1] if sockets else port
        print(json.dumps({"status": "ready", "port": selected_port}), flush=True)
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="echolingo-sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    token = os.environ.get("ECHOLINGO_IPC_TOKEN")
    if not token:
        parser.error("ECHOLINGO_IPC_TOKEN is required")
    asyncio.run(run_server(args.host, args.port, token))
    return 0
