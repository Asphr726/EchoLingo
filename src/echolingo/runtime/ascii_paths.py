"""ASCII paths for native code that opens files through narrow C strings.

nagisa (the Japanese tokenizer behind the forced aligner) loads its model
through DyNet, which cannot open a path the Windows ANSI code page cannot
encode, such as ``C:\\Users\\张三\\AppData\\Local\\EchoLingo\\sidecar\\…``.
nagisa finds its data relative to ``nagisa.tagger.base`` and loads the model
as soon as the package is imported, so the base is repointed while
``nagisa.tagger`` is imported: to the folder's 8.3 short name or, on a volume
without short names, to a copy of the data under ``%ProgramData%``.
"""

from __future__ import annotations

import importlib.abc
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

NAGISA_DATA_FILES = ("nagisa_v001.dict", "nagisa_v001.hp", "nagisa_v001.model")


def windows_short_path(path: str) -> str | None:
    """The 8.3 form of an existing path, or None when the volume has none."""
    import ctypes
    from ctypes import wintypes

    get_short_path = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    get_short_path.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    get_short_path.restype = wintypes.DWORD
    size = get_short_path(path, None, 0)
    if size == 0:
        return None
    buffer = ctypes.create_unicode_buffer(size)
    if get_short_path(path, buffer, size) == 0:
        return None
    return buffer.value


def nagisa_ascii_base(
    base: Path,
    *,
    short_path: Callable[[str], str | None] = windows_short_path,
    program_data: str | None = None,
) -> Path | None:
    """An ASCII directory holding ``data/<nagisa files>`` for ``base``.

    None means nothing needs to change (the path is already ASCII) or no ASCII
    location could be found; nagisa then keeps its own path.
    """
    if str(base).isascii():
        return None
    short = short_path(str(base))
    if short and short.isascii():
        return Path(short)
    root = Path(program_data or os.environ.get("ProgramData") or r"C:\ProgramData") / "EchoLingo" / "nagisa"
    if not str(root).isascii():
        return None
    try:
        data = root / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name in NAGISA_DATA_FILES:
            source = base / "data" / name
            target = data / name
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                continue
            partial = target.with_name(name + ".partial")
            shutil.copyfile(source, partial)
            os.replace(partial, target)
    except OSError:
        return None
    return root


class _RebasingLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, resolve: Callable[[Path], Path | None]) -> None:
        self._loader = loader
        self._resolve = resolve

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        base = getattr(module, "base", None)
        if isinstance(base, str):
            alias = self._resolve(Path(base))
            if alias is not None:
                module.base = str(alias)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._loader, name)


class TaggerBaseFinder(importlib.abc.MetaPathFinder):
    """Rebases ``<module>.base`` right after the module executes."""

    def __init__(self, resolve: Callable[[Path], Path | None], module: str = "nagisa.tagger") -> None:
        self._resolve = resolve
        self._module = module

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname != self._module:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None:
            spec.loader = _RebasingLoader(spec.loader, self._resolve)
        return spec


def install_nagisa_ascii_paths() -> None:
    """Make nagisa's model loadable from a non-ASCII folder (Windows only)."""
    if sys.platform != "win32" or "nagisa.tagger" in sys.modules:
        return
    if any(isinstance(finder, TaggerBaseFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, TaggerBaseFinder(nagisa_ascii_base))
