from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_gpu_pack() -> ModuleType:
    path = ROOT / "scripts" / "build_gpu_pack.py"
    spec = importlib.util.spec_from_file_location("echolingo_scripts_build_gpu_pack", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gpu_pack = load_gpu_pack()

CU130_ARCH_LIST = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
# Incompressible, so the tiny test archive still spans several parts.
PAYLOAD = random.Random(7).randbytes(6000)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_release_asset_names_follow_the_contract() -> None:
    archive = gpu_pack.archive_name("0.2.0", "windows-x64")
    assert archive == "echolingo-gpu-pack-0.2.0-windows-x64.tar.zst"
    assert gpu_pack.part_name(archive, 1) == archive + ".part01"
    assert gpu_pack.part_name(archive, 12) == archive + ".part12"
    assert gpu_pack.manifest_name("0.2.0", "linux-x64") == "echolingo-gpu-pack-0.2.0-linux-x64.json"
    assert gpu_pack.PART_BYTES == 1_610_612_736


def test_part_writer_splits_at_the_part_size(tmp_path) -> None:
    data = bytes(range(256)) * 41  # 10 496 bytes
    writer = gpu_pack.PartWriter(tmp_path, "pack.tar.zst", part_bytes=4096)
    for offset in range(0, len(data), 1000):
        writer.write(data[offset : offset + 1000])
    result = writer.close()
    assert [part["name"] for part in result["parts"]] == [
        "pack.tar.zst.part01",
        "pack.tar.zst.part02",
        "pack.tar.zst.part03",
    ]
    assert [part["bytes"] for part in result["parts"]] == [4096, 4096, 2304]
    assert result["archive_bytes"] == len(data)
    assert result["archive_sha256"] == sha256(data)
    joined = b""
    for part in result["parts"]:
        content = (tmp_path / part["name"]).read_bytes()
        assert part["sha256"] == sha256(content)
        joined += content
    assert joined == data


def test_part_writer_never_leaves_an_empty_trailing_part(tmp_path) -> None:
    writer = gpu_pack.PartWriter(tmp_path, "pack.tar.zst", part_bytes=8)
    writer.write(b"x" * 16)
    result = writer.close()
    assert [part["bytes"] for part in result["parts"]] == [8, 8]
    with pytest.raises(RuntimeError):
        gpu_pack.PartWriter(tmp_path / "empty", "pack.tar.zst", part_bytes=8).close()


def test_cuda_arch_list_must_cover_turing_through_blackwell() -> None:
    assert gpu_pack.missing_capabilities(CU130_ARCH_LIST) == []
    # SASS runs on later minors of the same major; PTX JIT-compiles forward.
    assert gpu_pack.missing_capabilities(["sm_75", "sm_80", "sm_90", "compute_90"]) == []
    assert gpu_pack.missing_capabilities(["sm_80", "sm_90"]) == ["7.5", "12.0"]
    assert gpu_pack.missing_capabilities([]) == list(gpu_pack.REQUIRED_CAPABILITIES)


def test_torch_build_check_rejects_the_wrong_wheel() -> None:
    good = {"torch_version": "2.13.0+cu130", "cuda_version": "13.0", "arch_list": CU130_ARCH_LIST}
    gpu_pack.check_torch_build(good)
    for broken in (
        dict(good, torch_version="2.13.0+cpu"),
        dict(good, cuda_version=None),
        dict(good, cuda_version="12.8"),
        dict(good, arch_list=["sm_80", "sm_90"]),
    ):
        with pytest.raises(RuntimeError):
            gpu_pack.check_torch_build(broken)


def test_manifests_carry_the_contract_fields() -> None:
    torch_info = {"torch_version": "2.13.0+cu130", "cuda_version": "13.0", "arch_list": CU130_ARCH_LIST}
    inner = gpu_pack.inner_manifest(
        version="0.2.0", platform_name="windows-x64", torch_info=torch_info, unpacked_bytes=123
    )
    assert inner == {
        "schema_version": 1,
        "app_version": "0.2.0",
        "platform": "windows-x64",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "min_driver_version": "580.65",
        "min_compute_capability": "7.5",
        "arch_list": CU130_ARCH_LIST,
        "unpacked_bytes": 123,
    }
    archive = {
        "archive": "echolingo-gpu-pack-0.2.0-windows-x64.tar.zst",
        "archive_bytes": 10,
        "archive_sha256": "a" * 64,
        "parts": [{"name": "x.part01", "bytes": 10, "sha256": "b" * 64}],
    }
    outer = gpu_pack.outer_manifest(inner, archive)
    assert {key: outer[key] for key in inner} == inner
    assert {key: outer[key] for key in archive} == archive
    assert "| x.part01 |" in gpu_pack.report(outer)


def _fake_pack(root: Path) -> Path:
    pack = root / "pack"
    (pack / "sidecar" / "_internal").mkdir(parents=True)
    (pack / "llama.cpp").mkdir()
    (pack / "sidecar" / "echolingo-sidecar").write_bytes(b"sidecar")
    (pack / "sidecar" / "_internal" / "torch_cuda.bin").write_bytes(PAYLOAD)
    (pack / "llama.cpp" / "llama-server").write_bytes(b"server")
    (pack / "llama.cpp" / "libggml-vulkan.so").write_bytes(b"vulkan" * 1000)
    (pack / "pack.json").write_text(json.dumps({"app_version": "0.2.0"}))
    return pack


def _compressor_available() -> bool:
    if shutil.which("zstd"):
        return True
    return importlib.util.find_spec("zstandard") is not None


@pytest.mark.skipif(not _compressor_available(), reason="needs the zstd CLI or zstandard")
def test_archive_round_trips_through_parts(tmp_path) -> None:
    pack = _fake_pack(tmp_path)
    out = tmp_path / "out"
    archive = gpu_pack.build_archive(pack, out, "pack.tar.zst", part_bytes=1024)
    assert len(archive["parts"]) > 1
    assert all(part["bytes"] <= 1024 for part in archive["parts"])
    manifest = {"app_version": "0.2.0", **archive}
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    root = gpu_pack.extract_pack(manifest_path, tmp_path / "unpacked with space 目录")
    assert root.name == gpu_pack.PACK_ROOT
    assert (root / "sidecar" / "_internal" / "torch_cuda.bin").read_bytes() == PAYLOAD
    assert (root / "llama.cpp" / "libggml-vulkan.so").read_bytes() == b"vulkan" * 1000


@pytest.mark.skipif(not _compressor_available(), reason="needs the zstd CLI or zstandard")
def test_extraction_rejects_a_corrupt_part(tmp_path) -> None:
    pack = _fake_pack(tmp_path)
    out = tmp_path / "out"
    archive = gpu_pack.build_archive(pack, out, "pack.tar.zst", part_bytes=1024)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({"app_version": "0.2.0", **archive}))
    second = out / archive["parts"][1]["name"]
    corrupted = bytearray(second.read_bytes())
    corrupted[0] ^= 0xFF
    second.write_bytes(bytes(corrupted))
    with pytest.raises(RuntimeError, match="SHA256"):
        gpu_pack.extract_pack(manifest_path, tmp_path / "unpacked")


def test_windows_unpack_paths_stay_below_max_path(tmp_path) -> None:
    pack = _fake_pack(tmp_path)
    assert "longest unpacked Windows path" in gpu_pack.check_windows_path_length(pack)
    deep = pack / "sidecar" / ("d" * 80) / ("f" * 90)
    deep.parent.mkdir()
    deep.write_text("x")
    with pytest.raises(RuntimeError, match="characters on Windows"):
        gpu_pack.check_windows_path_length(pack)


def test_layout_check_requires_the_vulkan_runtime(tmp_path) -> None:
    pack = _fake_pack(tmp_path)
    if sys.platform == "win32":
        pytest.skip("fake pack uses POSIX executable names")
    gpu_pack.check_layout(pack)
    (pack / "llama.cpp" / "libggml-vulkan.so").unlink()
    with pytest.raises(RuntimeError, match="Vulkan"):
        gpu_pack.check_layout(pack)


def test_layout_check_rejects_symlinks_leaving_the_pack(tmp_path) -> None:
    if sys.platform == "win32":
        pytest.skip("symlinks need privileges on Windows")
    pack = _fake_pack(tmp_path)
    (pack / "sidecar" / "_internal" / "libok.so").symlink_to("torch_cuda.bin")
    gpu_pack.check_layout(pack)
    (pack / "sidecar" / "_internal" / "libevil.so").symlink_to("../../../outside")
    with pytest.raises(RuntimeError, match="outside the pack"):
        gpu_pack.check_layout(pack)
