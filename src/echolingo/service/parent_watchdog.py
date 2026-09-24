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
_ERROR_INVALID_PARAMETER = 87
_WAIT_OBJECT_0 = 0x0000_0000


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


def _kernel32_api() -> _Kernel32Processes:
    global _kernel32
    if _kernel32 is None:
        _kernel32 = _Kernel32Processes()
    return _kernel32


def _signalled(api: Any, handle: Any) -> bool:
    # Only WAIT_OBJECT_0 means the process has exited. WAIT_TIMEOUT is a
    # running process; WAIT_FAILED is unknown and must not end a session.
    return api.wait(handle) == _WAIT_OBJECT_0


def windows_process_alive(process_id: int, *, api: Any | None = None) -> bool:
    """One-shot probe through a SYNCHRONIZE handle (Windows only)."""
    api = api if api is not None else _kernel32_api()
    handle, error = api.open_process(process_id)
    if handle is None:
        # ERROR_INVALID_PARAMETER: no process has this id any more. Access
        # denied (a protected or foreign process) and anything unexpected keep
        # the worker alive; the Desktop's job object still reaps it.
        return error != _ERROR_INVALID_PARAMETER
    try:
        return not _signalled(api, handle)
    finally:
        api.close(handle)


class WindowsProcessWatch:
    """One handle to the owner, opened once and waited on for its lifetime.

    The open handle pins the process object, so a recycled PID can never pass
    for the owner, and polling needs no OpenProcess per check.
    """

    def __init__(self, process_id: int, *, api: Any | None = None) -> None:
        self.process_id = process_id
        self._api = api if api is not None else _kernel32_api()
        self._handle, error = self._api.open_process(process_id)
        self._exited = self._handle is None and error == _ERROR_INVALID_PARAMETER

    def alive(self) -> bool:
        if self._exited:
            return False
        if self._handle is None:
            # No handle (access denied): fall back to probing by id.
            return windows_process_alive(self.process_id, api=self._api)
        if not _signalled(self._api, self._handle):
            return True
        self._exited = True
        self.close()
        return False

    def close(self) -> None:
        if self._handle is not None:
            self._api.close(self._handle)
            self._handle = None


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


def watch_parent(process_id: int, *, api: Any | None = None) -> Callable[[], bool]:
    """Liveness check for the owner, set up once when a watchdog starts."""
    if sys.platform == "win32" or api is not None:
        return WindowsProcessWatch(process_id, api=api).alive
    return lambda: parent_process_alive(process_id)


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
    parent_alive = watch_parent(process_id)

    def monitor() -> None:
        while parent_alive():
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(
        target=monitor,
        name="echolingo-parent-watchdog",
        daemon=True,
    ).start()
