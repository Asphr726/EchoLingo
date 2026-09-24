import os
import subprocess
import sys

import pytest

from echolingo.service import parent_watchdog
from echolingo.service.parent_watchdog import (
    WindowsProcessWatch,
    parent_process_alive,
    watch_parent,
    windows_process_alive,
)


def test_parent_watchdog_distinguishes_live_missing_and_inaccessible_processes() -> None:
    assert parent_process_alive(42, probe=lambda _pid, _signal: None)

    def missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    def inaccessible(_pid: int, _signal: int) -> None:
        raise PermissionError

    def unexpected(_pid: int, _signal: int) -> None:
        raise OSError("probe failed")

    assert not parent_process_alive(42, probe=missing)
    assert parent_process_alive(42, probe=inaccessible)
    assert parent_process_alive(42, probe=unexpected)


def test_default_probe_sees_a_live_process_and_a_finished_child() -> None:
    assert parent_process_alive(os.getpid())

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert not parent_process_alive(child.pid)


def test_windows_default_probe_never_signals_the_process(monkeypatch) -> None:
    probed: list[int] = []
    monkeypatch.setattr(parent_watchdog.sys, "platform", "win32")
    monkeypatch.setattr(
        parent_watchdog, "windows_process_alive", lambda pid: probed.append(pid) or True
    )

    def kill(_pid: int, _signal: int) -> None:
        raise AssertionError("os.kill(pid, 0) sends CTRL_C_EVENT on Windows")

    monkeypatch.setattr(parent_watchdog.os, "kill", kill)
    assert parent_process_alive(1234)
    assert probed == [1234]


WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
WAIT_FAILED = 0xFFFF_FFFF


class FakeKernel32:
    def __init__(self, *, error: int = 0, waits: list[int] | None = None) -> None:
        self.error = error
        self.waits = list(waits or [WAIT_TIMEOUT])
        self.opened = 0
        self.closed: list[object] = []

    def open_process(self, _process_id: int):
        self.opened += 1
        return (None, self.error) if self.error else (f"handle-{self.opened}", 0)

    def wait(self, _handle) -> int:
        return self.waits.pop(0) if len(self.waits) > 1 else self.waits[0]

    def close(self, handle) -> None:
        self.closed.append(handle)


def test_windows_probe_maps_open_and_wait_results() -> None:
    running = FakeKernel32(waits=[WAIT_TIMEOUT])
    assert windows_process_alive(7, api=running)
    assert running.closed == ["handle-1"]

    exited = FakeKernel32(waits=[WAIT_OBJECT_0])
    assert not windows_process_alive(7, api=exited)
    assert exited.closed == ["handle-1"]

    # WAIT_FAILED says nothing about the process: keep running.
    assert windows_process_alive(7, api=FakeKernel32(waits=[WAIT_FAILED]))
    assert not windows_process_alive(7, api=FakeKernel32(error=87))  # invalid parameter
    assert windows_process_alive(7, api=FakeKernel32(error=5))  # access denied


def test_windows_watch_keeps_one_handle_to_the_owner() -> None:
    api = FakeKernel32(waits=[WAIT_TIMEOUT, WAIT_FAILED, WAIT_TIMEOUT, WAIT_OBJECT_0])
    alive = watch_parent(7, api=api)

    assert [alive(), alive(), alive()] == [True, True, True]
    assert api.opened == 1 and api.closed == []
    assert alive() is False
    assert api.closed == ["handle-1"]
    # Once the owner has exited the watch stays down without new handles.
    assert alive() is False
    assert api.opened == 1


def test_windows_watch_of_a_missing_or_protected_owner() -> None:
    missing = FakeKernel32(error=87)
    assert WindowsProcessWatch(7, api=missing).alive() is False

    # Access denied: no handle to keep, so each check probes by id instead.
    protected = FakeKernel32(error=5)
    watch = WindowsProcessWatch(7, api=protected)
    assert watch.alive() and watch.alive()
    assert protected.opened == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX probe")
def test_posix_watch_probes_the_owner_by_id() -> None:
    assert watch_parent(os.getpid())()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert watch_parent(child.pid)() is False
