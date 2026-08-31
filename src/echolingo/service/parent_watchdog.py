from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

_started = False
_lock = threading.Lock()


def parent_process_alive(
    process_id: int, *, probe: Callable[[int, int], None] = os.kill
) -> bool:
    try:
        probe(process_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def start_parent_watchdog_from_environment() -> None:
    """Exit a packaged worker when its owning Desktop process disappears."""
    raw = os.getenv("ECHOLINGO_PARENT_PID")
    if not raw:
        return
    try:
        process_id = int(raw)
    except ValueError:
        return
    if process_id <= 1 or process_id == os.getpid():
        return

    global _started
    with _lock:
        if _started:
            return
        _started = True

    def monitor() -> None:
        while parent_process_alive(process_id):
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(
        target=monitor,
        name="echolingo-parent-watchdog",
        daemon=True,
    ).start()
