from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

from echolingo import __version__
from echolingo.service import self_test

ROOT = Path(__file__).resolve().parents[1]
CUDA_KEYS = {
    "torch_version",
    "cuda_version",
    "available",
    "device_name",
    "capability",
    "bf16",
    "model_dtype",
    "arch_list",
    "matmul_ok",
    "error",
}


def _passing() -> str:
    return "fine"


def _failing() -> str:
    raise ImportError("No module named 'onnxruntime'")


def _noisy() -> str:
    print("library chatter on stdout")
    return "noisy"


def _stub_checks(monkeypatch, **checks) -> None:
    monkeypatch.setattr(self_test, "default_checks", lambda: dict(checks))
    monkeypatch.setattr(self_test, "qwen_runtime_checks", lambda: {"qwen_server": _passing})


def _only_json_line(output: str) -> dict:
    lines = output.splitlines()
    assert len(lines) == 1, output
    return json.loads(lines[0])


class _FakeCuda:
    def __init__(self, *, available: bool, native_bf16: bool = True) -> None:
        self.available = available
        self.native_bf16 = native_bf16

    def get_arch_list(self) -> list[str]:
        return ["sm_75", "sm_80", "sm_120"] if self.available else []

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, _index: int) -> str:
        return "NVIDIA GeForce RTX 3060"

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return (8, 6)

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        return self.native_bf16


CU130_FLAGS = "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120"


def _fake_torch(*, available: bool, cuda_version: str | None = "13.0", native_bf16: bool = True):
    torch = types.SimpleNamespace(
        __version__="2.13.0+cu130" if cuda_version else "2.13.0+cpu",
        version=types.SimpleNamespace(cuda=cuda_version),
        cuda=_FakeCuda(available=available, native_bf16=native_bf16),
        float32="float32",
        bfloat16="bfloat16",
        float16="float16",
    )
    if cuda_version:
        # CPU builds have no CUDA arch flags at all.
        torch._C = types.SimpleNamespace(_cuda_getArchFlags=lambda: CU130_FLAGS)
    return torch


def test_self_test_prints_one_json_object_and_exits_zero_when_all_checks_pass(
    monkeypatch, capfd
) -> None:
    _stub_checks(monkeypatch, numpy=_passing, ssl=_passing, noisy=_noisy)

    code = self_test.main([])

    captured = capfd.readouterr()
    report = _only_json_line(captured.out)
    assert code == 0
    assert set(report) == {
        "ok", "version", "platform", "frozen", "utf8_mode", "ssl_ca_certs", "checks", "cuda"
    }
    assert report["ok"] is True
    assert report["version"] == __version__ == "0.3.1"
    assert report["frozen"] is False
    assert isinstance(report["utf8_mode"], bool)
    assert isinstance(report["ssl_ca_certs"], int)
    assert report["checks"]["numpy"] == {"ok": True, "detail": "fine"}
    assert report["cuda"] is None
    # Whatever a library prints goes to stderr, never into the JSON channel.
    assert "library chatter" in captured.err


@pytest.mark.skipif(sys.platform == "win32", reason="libc printf via ctypes")
def test_buffered_native_stdout_never_trails_the_json_line(monkeypatch, capfd) -> None:
    import ctypes

    libc = ctypes.CDLL(None)

    def native_chatter() -> str:
        # Buffered by C stdio (stdout is not a terminal here) until flushed.
        libc.printf(b"native chatter")
        return "native"

    _stub_checks(monkeypatch, native=native_chatter)

    assert self_test.main([]) == 0
    libc.fflush(None)

    captured = capfd.readouterr()
    assert _only_json_line(captured.out)["checks"]["native"]["ok"] is True
    assert "native chatter" in captured.err


def test_one_failing_check_fails_the_run_but_every_check_still_runs(monkeypatch, capfd) -> None:
    _stub_checks(monkeypatch, onnxruntime=_failing, numpy=_passing)

    code = self_test.main([])

    report = _only_json_line(capfd.readouterr().out)
    assert code == 1
    assert report["ok"] is False
    assert report["checks"]["onnxruntime"]["ok"] is False
    assert "ImportError" in report["checks"]["onnxruntime"]["detail"]
    assert report["checks"]["numpy"]["ok"] is True


def test_qwen_runtime_installs_the_shims_before_importing_the_server_stack(monkeypatch) -> None:
    order: list[str] = []

    def runtime_check() -> str:
        order.append("import")
        return "ok"

    report = self_test.run_self_test(
        qwen_runtime=True,
        install_qwen_shims=lambda: order.append("shims"),
        checks={"numpy": _passing},
        runtime_checks={"qwen_server": runtime_check},
    )

    assert order == ["shims", "import"]
    assert report["ok"] is True
    assert set(report["checks"]) == {"numpy", "qwen_server"}


def test_cli_parsing_modules_are_imported_with_an_empty_command_line(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "argv_probe_module.py").write_text(
        "import sys\nSEEN = list(sys.argv)\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "argv_probe_module", raising=False)
    monkeypatch.setattr(sys, "argv", ["echolingo-sidecar", "self-test", "--qwen-runtime"])

    module = self_test.import_without_cli_arguments("argv_probe_module")

    assert module.SEEN == ["echolingo-sidecar"]
    assert sys.argv == ["echolingo-sidecar", "self-test", "--qwen-runtime"]


def test_cuda_report_without_a_gpu_is_not_a_failure(capfd, monkeypatch) -> None:
    _stub_checks(monkeypatch, numpy=_passing)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, cuda_version=None))

    code = self_test.main(["--cuda"])

    report = _only_json_line(capfd.readouterr().out)
    assert code == 0
    cuda = report["cuda"]
    assert set(cuda) == CUDA_KEYS
    assert cuda["available"] is False
    assert cuda["cuda_version"] is None
    assert cuda["torch_version"] == "2.13.0+cpu"
    assert cuda["device_name"] is None and cuda["capability"] is None
    assert cuda["bf16"] is None and cuda["matmul_ok"] is None and cuda["error"] is None
    assert cuda["model_dtype"] is None and cuda["arch_list"] == []
    assert report["checks"]["cuda"]["ok"] is True


def test_cuda_build_without_a_gpu_still_reports_its_compiled_archs() -> None:
    report = self_test.run_self_test(
        cuda=True, checks={}, torch_module=_fake_torch(available=False)
    )

    cuda = report["cuda"]
    assert report["ok"] is True and cuda["available"] is False
    assert cuda["arch_list"] == CU130_FLAGS.split()


def test_arch_flags_are_normalised() -> None:
    assert self_test.normalise_arch_flags(" sm_75 sm_80  SM_90a compute_120 sm_80") == [
        "sm_75", "sm_80", "sm_90a", "compute_120"
    ]
    assert self_test.normalise_arch_flags("7.5;8.6;12.0+PTX") == [
        "sm_75", "sm_86", "sm_120", "compute_120"
    ]
    assert self_test.normalise_arch_flags(None) == []
    assert self_test.normalise_arch_flags("-O3 garbage") == []


def test_cuda_report_describes_a_usable_gpu(monkeypatch) -> None:
    dtypes: list[str] = []

    def matmul(_torch, device, dtype) -> bool:
        assert device == "cuda"
        dtypes.append(dtype)
        return True

    monkeypatch.setattr(self_test, "_matmul_matches", matmul)
    report = self_test.run_self_test(
        cuda=True, checks={}, torch_module=_fake_torch(available=True, native_bf16=True)
    )

    cuda = report["cuda"]
    assert report["ok"] is True
    assert cuda["available"] is True
    assert cuda["cuda_version"] == "13.0"
    assert cuda["device_name"] == "NVIDIA GeForce RTX 3060"
    assert cuda["capability"] == "8.6"
    assert cuda["bf16"] is True
    assert cuda["model_dtype"] == "bfloat16"
    assert cuda["arch_list"] == ["sm_75", "sm_80", "sm_120"]
    assert cuda["matmul_ok"] is True
    # float32 plus the dtype the Qwen server loads the model in.
    assert dtypes == ["float32", "bfloat16"]

    # Without native bf16 the model runs in float32 (never float16).
    dtypes.clear()
    report = self_test.run_self_test(
        cuda=True, checks={}, torch_module=_fake_torch(available=True, native_bf16=False)
    )
    assert report["cuda"]["bf16"] is False
    assert report["cuda"]["model_dtype"] == "float32"
    assert dtypes == ["float32"]


def test_cuda_matmul_failure_fails_the_self_test(monkeypatch) -> None:
    def broken(_torch, _device, _dtype) -> bool:
        raise RuntimeError("CUDA error: no kernel image is available for execution")

    monkeypatch.setattr(self_test, "_matmul_matches", broken)
    report = self_test.run_self_test(
        cuda=True, checks={}, torch_module=_fake_torch(available=True)
    )

    assert report["ok"] is False
    assert report["cuda"]["matmul_ok"] is False
    assert "no kernel image" in report["cuda"]["error"]
    assert report["checks"]["cuda"]["ok"] is False


def test_matmul_probe_is_exact_in_model_dtypes() -> None:
    torch = pytest.importorskip("torch")
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert self_test._matmul_matches(torch, "cpu", dtype)


def test_light_checks_pass_in_this_environment(monkeypatch) -> None:
    # What the frozen entry point does; CI interpreters' default CA paths vary.
    monkeypatch.setenv("SSL_CERT_FILE", pytest.importorskip("certifi").where())
    report = self_test.run_self_test(
        checks={
            "numpy": self_test.check_numpy,
            "httpx": self_test.check_httpx,
            "websockets": self_test.check_websockets,
            "ssl": self_test.check_ssl,
        }
    )
    assert report["ok"] is True, report["checks"]
    assert report["ssl_ca_certs"] > 0
    assert "SSL_CERT_FILE" in report["checks"]["ssl"]["detail"]


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "echolingo_sidecar_entry_under_test", ROOT / "packaging/sidecar_entry.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_dispatches_self_test_with_the_qwen_shims(monkeypatch) -> None:
    entry = _load_entrypoint()
    seen: dict = {}

    def fake_main(argv, *, install_qwen_shims):
        seen.update(argv=argv, shims=install_qwen_shims)
        return 7

    monkeypatch.setattr(self_test, "main", fake_main)
    # Leave pytest's capture streams alone.
    monkeypatch.setattr(entry, "_configure_standard_streams", lambda: None)
    monkeypatch.delenv("ECHOLINGO_PARENT_PID", raising=False)
    monkeypatch.setattr(sys, "argv", ["echolingo-sidecar", "self-test", "--qwen-runtime", "--cuda"])

    assert entry.main() == 7
    assert seen["argv"] == ["--qwen-runtime", "--cuda"]
    assert seen["shims"] is entry._install_qwen_import_shims


def test_frozen_entrypoint_uses_the_bundled_ca_file_unless_one_is_set(monkeypatch) -> None:
    certifi = pytest.importorskip("certifi")
    entry = _load_entrypoint()
    monkeypatch.setattr(sys, "platform", "darwin")

    environ: dict[str, str] = {}
    entry._configure_certificate_bundle(environ)
    assert environ == {}  # source runs keep OpenSSL defaults

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    entry._configure_certificate_bundle(environ)
    assert environ == {"SSL_CERT_FILE": certifi.where()}

    environ = {"SSL_CERT_FILE": "/custom/ca.pem"}
    entry._configure_certificate_bundle(environ)
    assert environ == {"SSL_CERT_FILE": "/custom/ca.pem"}

    monkeypatch.setattr(sys, "platform", "win32")
    environ = {}
    entry._configure_certificate_bundle(environ, linux_bundles=())
    assert environ == {"SSL_CERT_FILE": certifi.where()}


def test_frozen_linux_entrypoint_prefers_the_distribution_bundle(monkeypatch, tmp_path) -> None:
    certifi = pytest.importorskip("certifi")
    entry = _load_entrypoint()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    system_bundle = tmp_path / "ca-certificates.crt"
    system_bundle.write_text("", encoding="utf-8")
    missing = str(tmp_path / "missing.pem")

    environ: dict[str, str] = {}
    entry._configure_certificate_bundle(environ, linux_bundles=(missing, str(system_bundle)))
    assert environ == {"SSL_CERT_FILE": str(system_bundle)}

    # No distribution bundle: an explicit SSL_CERT_DIR is used as is ...
    environ = {"SSL_CERT_DIR": str(tmp_path)}
    entry._configure_certificate_bundle(environ, linux_bundles=(missing,))
    assert environ == {"SSL_CERT_DIR": str(tmp_path)}

    # ... and otherwise certifi is the fallback.
    environ = {}
    entry._configure_certificate_bundle(environ, linux_bundles=(missing,))
    assert environ == {"SSL_CERT_FILE": certifi.where()}


def test_entrypoint_leaves_the_process_environment_alone_when_not_frozen(monkeypatch) -> None:
    entry = _load_entrypoint()
    monkeypatch.delattr(sys, "frozen", raising=False)
    before = dict(entry.os.environ)
    entry._configure_certificate_bundle()
    assert dict(entry.os.environ) == before


def test_entrypoint_switches_standard_streams_to_utf8(monkeypatch) -> None:
    entry = _load_entrypoint()
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    entry._configure_standard_streams()

    assert (stdout.encoding, stdout.errors) == ("utf-8", "backslashreplace")
    assert (stderr.encoding, stderr.errors) == ("utf-8", "backslashreplace")
    print("講義 강의", file=stdout)
    stdout.flush()
    assert stdout.buffer.getvalue().decode("utf-8").strip() == "講義 강의"
