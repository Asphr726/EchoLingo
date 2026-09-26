import subprocess
import sys
import time

from echolingo.service import process_watchdog
from echolingo.service.process_watchdog import run_child_until_parent_exit


def test_process_watchdog_reaps_native_child_when_owner_is_missing() -> None:
    started = time.monotonic()
    result = run_child_until_parent_exit(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        parent_process_id=999_999,
        poll_seconds=0.01,
        alive=lambda _pid: False,
    )

    assert result != 0
    assert time.monotonic() - started < 5.0


def test_process_watchdog_returns_the_child_exit_code_while_owner_lives() -> None:
    result = run_child_until_parent_exit(
        [sys.executable, "-c", "raise SystemExit(3)"],
        parent_process_id=999_999,
        poll_seconds=0.01,
        alive=lambda _pid: True,
    )

    assert result == 3


def test_native_child_gets_no_console_window_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(process_watchdog.sys, "platform", "win32")
    assert process_watchdog._child_creationflags() == getattr(
        subprocess, "CREATE_NO_WINDOW", 0x0800_0000
    )
    monkeypatch.setattr(process_watchdog.sys, "platform", "linux")
    assert process_watchdog._child_creationflags() == 0


def test_default_liveness_is_set_up_once_before_the_child_starts(monkeypatch) -> None:
    watched: list[int] = []

    def watch_parent(process_id: int):
        watched.append(process_id)
        return lambda: False

    monkeypatch.setattr(process_watchdog, "watch_parent", watch_parent)
    result = run_child_until_parent_exit(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        parent_process_id=4321,
        poll_seconds=0.01,
    )

    assert result != 0
    assert watched == [4321]


def test_the_native_child_is_opted_out_of_power_throttling(monkeypatch) -> None:
    opted_out: list[int] = []
    monkeypatch.setattr(
        process_watchdog, "disable_power_throttling_for_process", opted_out.append
    )
    started: list[subprocess.Popen] = []
    popen = subprocess.Popen

    def record(*arguments, **options):
        child = popen(*arguments, **options)
        started.append(child)
        return child

    monkeypatch.setattr(process_watchdog.subprocess, "Popen", record)
    result = run_child_until_parent_exit(
        [sys.executable, "-c", "raise SystemExit(0)"],
        parent_process_id=999_999,
        poll_seconds=0.01,
        alive=lambda _pid: True,
    )

    assert result == 0
    assert opted_out == [started[0].pid]
