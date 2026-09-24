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
