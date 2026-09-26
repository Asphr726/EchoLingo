from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from ..assistant.service import AssistantService, request_id_of
from ..backends import registry
from .cloud_probe import probe_cloud
from .parent_watchdog import start_parent_watchdog_from_environment
from .power_throttling import disable_power_throttling
from .protocol import PROTOCOL_VERSION, ProtocolError, decode_audio_packet
from .session import DesktopInferenceSession

log = logging.getLogger("echolingo.sidecar")

# Features the shell may rely on.
CAPABILITIES = ["assistant.v1", "asr_context.v1"]
# Assistant requests carry transcripts of long sessions.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _event(event_type: str, payload: Any) -> str:
    return json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False)


class SidecarConnection:
    def __init__(
        self,
        websocket: ServerConnection,
        token: str,
        *,
        assistant: AssistantService | None = None,
    ) -> None:
        self.websocket = websocket
        self.expected_token = token
        self.authenticated = False
        self.session: DesktopInferenceSession | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sender: asyncio.Task[None] | None = None
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.assistant = assistant or AssistantService()
        self.assistant_tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        self.sender = asyncio.create_task(self._send_events())
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    await self._audio(message)
                else:
                    await self._command(json.loads(message))
        except ConnectionClosed:
            # The shell's readiness probe and a normal shutdown both close
            # without a command; nothing to report.
            pass
        except (json.JSONDecodeError, ProtocolError, KeyError, ValueError) as error:
            await self._send_error("invalid_request", str(error))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._send_error("backend_start_failed", str(error))
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

    async def _send_error(self, code: str, message: str) -> None:
        try:
            await self.websocket.send(
                _event("error", {"code": code, "message": message, "recoverable": True})
            )
        except ConnectionClosed:
            log.warning("sidecar error not delivered (%s): %s", code, message)

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
                _event(
                    "hello_accepted",
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "providers_digest": registry.catalog_digest(),
                        "capabilities": list(CAPABILITIES),
                    },
                )
            )
            return
        if command_type == "start_session":
            if self.session is not None:
                raise ProtocolError("a sidecar session is already active")
            self.session = await DesktopInferenceSession.create(payload, self.events)
            await self.websocket.send(
                _event("ready", {"session_id": payload["session_id"], "route": self.session.route})
            )
        elif command_type == "plan_session":
            if self.session is not None:
                raise ProtocolError("cannot plan while a sidecar session is active")
            plan = await asyncio.to_thread(DesktopInferenceSession.plan, payload)
            await self.websocket.send(_event("route_plan", plan))
        elif command_type == "probe_cloud":
            if self.session is not None:
                raise ProtocolError("cannot probe cloud while a session is active")
            asr_provider = payload.get("asr_provider")
            translation_provider = payload.get("translation_provider")
            if asr_provider is None and translation_provider is None:
                # Legacy shape from earlier shells: Qwen Cloud only.
                asr_provider = "qwen_cloud"
                if payload.get("include_translation"):
                    translation_provider = "qwen_cloud"
            result = await probe_cloud(asr_provider, translation_provider)
            await self.websocket.send(
                _event(
                    "cloud_probe_result",
                    {"request_id": payload["request_id"], "result": result},
                )
            )
        elif command_type == "pause":
            self._require_session().paused = True
        elif command_type == "resume":
            self._require_session().paused = False
        elif command_type == "finish_session":
            session = self._require_session()
            await asyncio.wait_for(session.finish(), timeout=10.0)
            # Through the ordered event queue, so it follows the last
            # transcript/translation events that finish() enqueued.
            self.events.put_nowait(
                {"type": "session_finished", "payload": {"session_id": payload["session_id"]}}
            )
            self.session = None
            task = asyncio.create_task(self._run_alignment(session))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        elif command_type == "assistant_request":
            self._start_assistant(payload)
        elif command_type == "assistant_cancel":
            self._cancel_assistant(payload)
        elif command_type == "shutdown":
            await self.websocket.close(code=1000, reason="sidecar shutdown")
        else:
            raise ProtocolError(f"unknown sidecar command: {command_type}")

    def _start_assistant(self, payload: Any) -> None:
        """Run an assistant task in the background.

        Validation happens inside the task, which always answers with one
        ``assistant_result``; nothing here raises into the command loop.
        """
        request_id = request_id_of(payload)
        if request_id and request_id in self.assistant_tasks:
            log.warning("assistant request %s is already running; duplicate ignored", request_id)
            return
        answered = False

        def put(event: dict[str, Any]) -> None:
            nonlocal answered
            if event.get("type") == "assistant_result":
                answered = True
            self.events.put_nowait(event)

        def settle(done: asyncio.Task[None]) -> None:
            # A cancel that arrives before the task first runs never enters
            # ``handle``; the request still ends in exactly one result.
            if done.cancelled() and not answered:
                put(
                    {
                        "type": "assistant_result",
                        "payload": {
                            "request_id": request_id,
                            "ok": False,
                            "result": {
                                "code": "cancelled",
                                "message": "The assistant task was cancelled.",
                            },
                        },
                    }
                )

        task = asyncio.create_task(self.assistant.handle(payload, put))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        task.add_done_callback(_consume_task_outcome)
        task.add_done_callback(settle)
        if request_id:
            self.assistant_tasks[request_id] = task

            def forget(done: asyncio.Task[None], request_id: str = request_id) -> None:
                if self.assistant_tasks.get(request_id) is done:
                    del self.assistant_tasks[request_id]

            task.add_done_callback(forget)

    def _cancel_assistant(self, payload: Any) -> None:
        request_id = request_id_of(payload)
        task = self.assistant_tasks.get(request_id) if request_id else None
        if task is None:
            log.debug("assistant cancel for an unknown or finished request")
            return
        task.cancel()

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


def _consume_task_outcome(task: asyncio.Task[Any]) -> None:
    """Retrieve a finished task's exception so asyncio does not log it as
    never retrieved (assistant tasks report failures as results)."""
    if not task.cancelled():
        task.exception()


async def run_server(host: str, port: int, token: str) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("inference sidecar must bind to loopback")
    async with serve(
        lambda ws: SidecarConnection(ws, token).run(),
        host,
        port,
        max_size=MAX_MESSAGE_BYTES,
    ) as server:
        sockets = server.sockets or []
        selected_port = sockets[0].getsockname()[1] if sockets else port
        print(json.dumps({"status": "ready", "port": selected_port}), flush=True)
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    start_parent_watchdog_from_environment()
    # stderr is captured by the desktop shell into logs/sidecar.log; stdout
    # stays reserved for the JSON ready line.
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("ECHOLINGO_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Per-connection chatter from the WebSocket library is not useful in
    # logs/sidecar.log; provider probes and sessions log through "echolingo.*".
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # Also for the Conda launch, where the Desktop's power-throttling opt-out
    # reaches only `conda` (a no-op outside Windows).
    disable_power_throttling()
    parser = argparse.ArgumentParser(prog="echolingo-sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    token = os.environ.get("ECHOLINGO_IPC_TOKEN")
    if not token:
        parser.error("ECHOLINGO_IPC_TOKEN is required")
    asyncio.run(run_server(args.host, args.port, token))
    return 0
