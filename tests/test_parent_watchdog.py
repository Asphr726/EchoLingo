from echolingo.service.parent_watchdog import parent_process_alive


def test_parent_watchdog_distinguishes_live_missing_and_inaccessible_processes() -> None:
    assert parent_process_alive(42, probe=lambda _pid, _signal: None)

    def missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    def inaccessible(_pid: int, _signal: int) -> None:
        raise PermissionError

    assert not parent_process_alive(42, probe=missing)
    assert parent_process_alive(42, probe=inaccessible)
