from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence

from .parent_watchdog import watch_parent


def _child_creationflags() -> int:
    # The wrapper runs without a console on Windows; a console child such as
    # llama-server.exe would otherwise open a window of its own.
    if sys.platform == "win32":
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0x0800_0000))
    return 0


def _stop_signals() -> list[int]:
    signals = [signal.SIGTERM, signal.SIGINT]
    # Windows delivers console break events as SIGBREAK. TerminateProcess runs
    # no handler at all; the parent poll below and the Desktop's job object
    # cover that case.
    if hasattr(signal, "SIGBREAK"):
        signals.append(signal.SIGBREAK)
    return signals


def run_child_until_parent_exit(
    command: Sequence[str],
    *,
    parent_process_id: int | None = None,
    poll_seconds: float = 0.25,
    alive: Callable[[int], bool] | None = None,
) -> int:
    """Run a native model server and reap it when the Desktop owner disappears."""
    if not command:
        raise ValueError("watch-process requires a child command")
    parent_process_id = parent_process_id or int(os.environ["ECHOLINGO_PARENT_PID"])
    # Set up before the child starts (on Windows: one handle to the owner).
    parent_alive = (
        watch_parent(parent_process_id)
        if alive is None
        else functools.partial(alive, parent_process_id)
    )
    child = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        creationflags=_child_creationflags(),
    )
    stopping = threading.Event()

    def request_stop(_signal: int, _frame: object) -> None:
        stopping.set()

    previous = {number: signal.signal(number, request_stop) for number in _stop_signals()}
    try:
        while child.poll() is None:
            if stopping.is_set() or not parent_alive():
                child.terminate()
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5.0)
                break
            time.sleep(poll_seconds)
        return int(child.returncode or 0)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)
