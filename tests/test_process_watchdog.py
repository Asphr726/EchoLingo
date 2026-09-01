import time

from echolingo.service.process_watchdog import run_child_until_parent_exit


def test_process_watchdog_reaps_native_child_when_owner_is_missing() -> None:
    started = time.monotonic()
    result = run_child_until_parent_exit(
        ["/bin/sleep", "30"],
        parent_process_id=999_999,
        poll_seconds=0.01,
        alive=lambda _pid: False,
    )

    assert result != 0
    assert time.monotonic() - started < 2.0
