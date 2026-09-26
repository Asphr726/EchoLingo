"""Keep Windows from throttling EchoLingo's inference processes.

Windows applies power throttling (EcoQoS) to windowless background processes,
and more so on battery or the Balanced power plan: lower CPU clocks,
efficiency cores and coarser timers. On a hybrid-core laptop it slowed the
local ASR server and llama-server (and the GPU work they feed) by more than
half. ``SetProcessInformation(ProcessPowerThrottling)`` opts a process out.
Every call here is best effort: a failure is logged and never raised, and
other platforms are left alone.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# From processthreadsapi.h: the PROCESS_INFORMATION_CLASS value and the flags.
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
_PROCESS_SET_INFORMATION = 0x0200

# A mechanism named in ControlMask but cleared in StateMask is turned off for
# the process. Windows 10 does not know the timer flag and rejects the whole
# request, so the execution-speed opt-out alone is the fallback.
OPT_OUT_CONTROL_MASKS = (
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
)


class ProcessPowerThrottlingState(ctypes.Structure):
    """``PROCESS_POWER_THROTTLING_STATE``: three ULONGs (32-bit on Windows)."""

    _fields_ = [
        ("Version", ctypes.c_uint32),
        ("ControlMask", ctypes.c_uint32),
        ("StateMask", ctypes.c_uint32),
    ]


class _Kernel32PowerThrottling:
    """The kernel32 calls the opt-out needs, bound lazily."""

    def __init__(self, kernel32: Any | None = None) -> None:
        if kernel32 is None:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.argtypes = ()
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.SetProcessInformation.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            )
            kernel32.SetProcessInformation.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32

    def current_process(self) -> Any:
        # A pseudo handle: it needs no closing.
        return self._kernel32.GetCurrentProcess()

    def open_process(self, process_id: int) -> Any:
        return self._kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, process_id)

    def set_information(self, handle: Any, state: ProcessPowerThrottlingState) -> bool:
        return bool(
            self._kernel32.SetProcessInformation(
                handle, PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)
            )
        )

    def close(self, handle: Any) -> None:
        self._kernel32.CloseHandle(handle)

    @staticmethod
    def last_error() -> int:
        get_last_error = getattr(ctypes, "get_last_error", None)
        return int(get_last_error()) if get_last_error is not None else 0


_kernel32: _Kernel32PowerThrottling | None = None


def _kernel32_api() -> _Kernel32PowerThrottling:
    global _kernel32
    if _kernel32 is None:
        _kernel32 = _Kernel32PowerThrottling()
    return _kernel32


def _opt_out(api: Any, handle: Any, target: str) -> bool:
    for control_mask in OPT_OUT_CONTROL_MASKS:
        state = ProcessPowerThrottlingState(
            Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=control_mask,
            StateMask=0,
        )
        if api.set_information(handle, state):
            return True
    logger.warning(
        "could not disable power throttling for %s (error %s)", target, api.last_error()
    )
    return False


def disable_power_throttling(*, api: Any | None = None) -> bool:
    """Opt this process out; True when Windows accepted it."""
    if sys.platform != "win32" and api is None:
        return False
    try:
        api = api if api is not None else _kernel32_api()
        return _opt_out(api, api.current_process(), "this process")
    except Exception as error:  # noqa: BLE001 - an opt-out must never stop a worker
        logger.warning("could not disable power throttling: %s", error)
        return False


def disable_power_throttling_for_process(process_id: int, *, api: Any | None = None) -> bool:
    """Opt a child process out by id; True when Windows accepted it."""
    if sys.platform != "win32" and api is None:
        return False
    try:
        api = api if api is not None else _kernel32_api()
        handle = api.open_process(process_id)
        if not handle:
            logger.warning(
                "could not disable power throttling for process %s (error %s)",
                process_id,
                api.last_error(),
            )
            return False
        try:
            return _opt_out(api, handle, f"process {process_id}")
        finally:
            api.close(handle)
    except Exception as error:  # noqa: BLE001 - an opt-out must never stop a worker
        logger.warning("could not disable power throttling for process %s: %s", process_id, error)
        return False
