import ctypes
from pathlib import Path

import pytest

from echolingo.service import power_throttling
from echolingo.service.power_throttling import (
    ProcessPowerThrottlingState,
    _Kernel32PowerThrottling,
    disable_power_throttling,
    disable_power_throttling_for_process,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROCESS = -1


class FakeKernel32:
    """Records the kernel32 calls; SetProcessInformation answers `results`."""

    def __init__(self, results: tuple[int, ...] = (1,), open_handle: int = 0x1234) -> None:
        self.results = list(results)
        self.open_handle = open_handle
        self.requests: list[tuple[object, int, int, tuple[int, int, int]]] = []
        self.opened: list[tuple[int, bool, int]] = []
        self.closed: list[object] = []

    def GetCurrentProcess(self) -> int:  # noqa: N802 - kernel32 name
        return CURRENT_PROCESS

    def OpenProcess(self, access: int, inherit: bool, process_id: int) -> int:  # noqa: N802
        self.opened.append((access, inherit, process_id))
        return self.open_handle

    def SetProcessInformation(  # noqa: N802
        self, handle: object, information_class: int, pointer: object, size: int
    ) -> int:
        state = pointer._obj  # type: ignore[attr-defined] - the byref() target
        self.requests.append(
            (handle, information_class, size, (state.Version, state.ControlMask, state.StateMask))
        )
        return self.results.pop(0) if self.results else 0

    def CloseHandle(self, handle: object) -> int:  # noqa: N802
        self.closed.append(handle)
        return 1


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> FakeKernel32:
    kernel32 = FakeKernel32()
    monkeypatch.setattr(power_throttling.sys, "platform", "win32")
    monkeypatch.setattr(
        power_throttling, "_kernel32_api", lambda: _Kernel32PowerThrottling(kernel32=kernel32)
    )
    return kernel32


def test_the_state_matches_the_windows_structure() -> None:
    assert ctypes.sizeof(ProcessPowerThrottlingState) == 12
    assert power_throttling.PROCESS_POWER_THROTTLING == 4
    assert power_throttling.OPT_OUT_CONTROL_MASKS == (0x5, 0x1)


def test_the_current_process_opts_out_of_execution_speed_and_timer_throttling(
    windows: FakeKernel32,
) -> None:
    assert disable_power_throttling() is True

    assert windows.requests == [(CURRENT_PROCESS, 4, 12, (1, 0x5, 0))]


def test_windows_10_falls_back_to_the_execution_speed_opt_out(windows: FakeKernel32) -> None:
    windows.results = [0, 1]

    assert disable_power_throttling() is True

    assert [request[3] for request in windows.requests] == [(1, 0x5, 0), (1, 0x1, 0)]


def test_a_child_is_opted_out_through_its_own_handle(windows: FakeKernel32) -> None:
    assert disable_power_throttling_for_process(4321) is True

    assert windows.opened == [(0x0200, False, 4321)]
    assert windows.requests == [(0x1234, 4, 12, (1, 0x5, 0))]
    assert windows.closed == [0x1234]


def test_refused_opt_outs_are_logged_not_raised(
    windows: FakeKernel32, caplog: pytest.LogCaptureFixture
) -> None:
    windows.results = [0, 0]
    assert disable_power_throttling() is False
    assert len(windows.requests) == 2
    assert "could not disable power throttling" in caplog.text

    windows.results = [0, 0]
    assert disable_power_throttling_for_process(4321) is False
    # The handle is closed on failure too.
    assert windows.closed == [0x1234]

    windows.open_handle = 0
    windows.requests.clear()
    assert disable_power_throttling_for_process(4321) is False
    assert windows.requests == []
    assert windows.closed == [0x1234]


def test_errors_from_the_system_call_are_swallowed(windows: FakeKernel32) -> None:
    def fail(*_arguments: object) -> int:
        raise OSError("SetProcessInformation is unavailable")

    windows.SetProcessInformation = fail  # type: ignore[method-assign]

    assert disable_power_throttling() is False
    assert disable_power_throttling_for_process(4321) is False


def test_a_missing_kernel32_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(power_throttling.sys, "platform", "win32")
    monkeypatch.setattr(power_throttling, "_kernel32", None)
    monkeypatch.delattr(power_throttling.ctypes, "WinDLL", raising=False)

    assert disable_power_throttling() is False
    assert disable_power_throttling_for_process(4321) is False


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_other_platforms_are_left_alone(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    def unexpected() -> None:
        raise AssertionError("kernel32 must not be loaded")

    monkeypatch.setattr(power_throttling.sys, "platform", platform)
    monkeypatch.setattr(power_throttling, "_kernel32_api", unexpected)

    assert disable_power_throttling() is False
    assert disable_power_throttling_for_process(4321) is False


def test_the_packaged_entry_point_opts_out_before_any_mode_runs() -> None:
    source = (ROOT / "packaging/sidecar_entry.py").read_text()
    main = source[source.index("def main() -> int:") :]
    assert main.index("disable_power_throttling()") < main.index('"watch-process"')
