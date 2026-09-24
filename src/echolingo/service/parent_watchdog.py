from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

_started = False
_lock = threading.Lock()

# Windows access rights, error and wait codes used by the liveness probe.
_SYNCHRONIZE = 0x0010_0000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
_WAIT_TIMEOUT = 0x0000_0102


class _Kernel32Processes:
    """The three kernel32 calls the Windows probe needs, bound lazily."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._last_error: Callable[[], int] = ctypes.get_last_error  # type: ignore[attr-defined]

    def open_process(self, process_id: int) -> tuple[Any, int]:
        handle = self._kernel32.OpenProcess(
            _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
        )
        return (handle, 0) if handle else (None, self._last_error())

    def wait(self, handle: Any) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, 0))

    def close(self, handle: Any) -> None:
        self._kernel32.CloseHandle(handle)


_kernel32: _Kernel32Processes | None = None


def windows_process_alive(process_id: int, *, api: Any | None = None) -> bool:
    """Probe a process through a SYNCHRONIZE handle (Windows only)."""
    global _kernel32
    if api is None:
        if _kernel32 is None:
            _kernel32 = _Kernel32Processes()
        api = _kernel32
    handle, error = api.open_process(process_id)
    if handle is None:
        # ERROR_INVALID_PARAMETER: no process has this id any more. Access
        # denied (a protected or foreign process) and anything unexpected keep
        # the worker alive; the Desktop's job object still reaps it.
        return error != _ERROR_INVALID_PARAMETER
    try:
        return api.wait(handle) == _WAIT_TIMEOUT
    finally:
        api.close(handle)


def parent_process_alive(
    process_id: int, *, probe: Callable[[int, int], None] | None = None
) -> bool:
    if probe is None:
        if sys.platform == "win32":
            # os.kill(pid, 0) sends CTRL_C_EVENT on Windows instead of probing.
            return windows_process_alive(process_id)
        probe = os.kill
    try:
        probe(process_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # An unexplained probe failure must not stop a working session.
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
