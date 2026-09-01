from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence

from .parent_watchdog import parent_process_alive


def run_child_until_parent_exit(
    command: Sequence[str],
    *,
    parent_process_id: int | None = None,
    poll_seconds: float = 0.25,
    alive: Callable[[int], bool] = parent_process_alive,
) -> int:
    """Run a native model server and reap it when the Desktop owner disappears."""
    if not command:
        raise ValueError("watch-process requires a child command")
    parent_process_id = parent_process_id or int(os.environ["ECHOLINGO_PARENT_PID"])
    child = subprocess.Popen(list(command), stdin=subprocess.DEVNULL)
    stopping = threading.Event()

    def request_stop(_signal: int, _frame: object) -> None:
        stopping.set()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        while child.poll() is None:
            if stopping.is_set() or not alive(parent_process_id):
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
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)
