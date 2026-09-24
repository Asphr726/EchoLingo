from __future__ import annotations

import importlib
import sys
from pathlib import Path

from echolingo.runtime.ascii_paths import NAGISA_DATA_FILES, TaggerBaseFinder, nagisa_ascii_base


def _nagisa_install(base: Path) -> None:
    data = base / "data"
    data.mkdir(parents=True)
    for name in NAGISA_DATA_FILES:
        (data / name).write_bytes(name.encode() * 3)


def test_ascii_paths_are_left_alone(tmp_path) -> None:
    assert nagisa_ascii_base(tmp_path / "EchoLingo" / "nagisa", short_path=lambda _path: None) is None


def test_the_short_path_is_preferred(tmp_path) -> None:
    base = tmp_path / "安装 测试" / "nagisa"
    _nagisa_install(base)
    assert nagisa_ascii_base(base, short_path=lambda _path: "C:\\Users\\5B89~1\\nagisa") == Path(
        "C:\\Users\\5B89~1\\nagisa"
    )


def test_data_is_copied_when_the_volume_has_no_short_names(tmp_path) -> None:
    base = tmp_path / "张三" / "nagisa"
    _nagisa_install(base)
    program_data = tmp_path / "ProgramData"
    alias = nagisa_ascii_base(base, short_path=lambda _path: None, program_data=str(program_data))
    assert alias == program_data / "EchoLingo" / "nagisa"
    for name in NAGISA_DATA_FILES:
        assert (alias / "data" / name).read_bytes() == (base / "data" / name).read_bytes()
    # A second start reuses the copy.
    assert nagisa_ascii_base(base, short_path=lambda _path: None, program_data=str(program_data)) == alias


def test_no_alias_when_program_data_is_not_ascii_either(tmp_path) -> None:
    base = tmp_path / "张三" / "nagisa"
    _nagisa_install(base)
    assert (
        nagisa_ascii_base(base, short_path=lambda _path: None, program_data=str(tmp_path / "数据"))
        is None
    )


def test_the_base_is_rebased_before_the_package_loads_its_model(tmp_path, monkeypatch) -> None:
    # A stand-in for nagisa: the package builds its default tagger on import,
    # reading the model path from tagger.base.
    package = tmp_path / "fake_nagisa_pkg"
    package.mkdir()
    (package / "tagger.py").write_text(
        "import os\n"
        "base = os.path.dirname(os.path.abspath(__file__))\n"
        "class Tagger:\n"
        "    def __init__(self):\n"
        "        self.params = base + '/data/nagisa_v001.model'\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "from fake_nagisa_pkg.tagger import Tagger\ntagger = Tagger()\n", encoding="utf-8"
    )
    alias = tmp_path / "ascii-alias"
    finder = TaggerBaseFinder(lambda _base: alias, module="fake_nagisa_pkg.tagger")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    try:
        module = importlib.import_module("fake_nagisa_pkg")
        assert module.tagger.params == f"{alias}/data/nagisa_v001.model"
    finally:
        for name in ("fake_nagisa_pkg", "fake_nagisa_pkg.tagger"):
            sys.modules.pop(name, None)
