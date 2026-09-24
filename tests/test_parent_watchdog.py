import os
import subprocess
import sys

from echolingo.service import parent_watchdog
from echolingo.service.parent_watchdog import parent_process_alive, windows_process_alive


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


class FakeKernel32:
    def __init__(self, *, error: int = 0, wait_result: int = 0x102) -> None:
        self.error = error
        self.wait_result = wait_result
        self.closed: list[object] = []

    def open_process(self, _process_id: int):
        return (None, self.error) if self.error else ("handle", 0)

    def wait(self, _handle) -> int:
        return self.wait_result

    def close(self, handle) -> None:
        self.closed.append(handle)


def test_windows_probe_maps_open_and_wait_results() -> None:
    running = FakeKernel32(wait_result=0x102)  # WAIT_TIMEOUT
    assert windows_process_alive(7, api=running)
    assert running.closed == ["handle"]

    exited = FakeKernel32(wait_result=0)  # WAIT_OBJECT_0
    assert not windows_process_alive(7, api=exited)
    assert exited.closed == ["handle"]

    assert not windows_process_alive(7, api=FakeKernel32(error=87))  # invalid parameter
    assert windows_process_alive(7, api=FakeKernel32(error=5))  # access denied
