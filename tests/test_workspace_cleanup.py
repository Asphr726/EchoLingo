"""Deletion safety and publication gating, without network or real cleanup."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('clean_workspace', Path(__file__).parents[1] / 'scripts/clean_workspace.py')
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def test_cleanup_preserves_models_evidence_and_sources(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for name in ['target/debug', 'node_modules/pkg', 'src/__pycache__', 'runs/lecture/__pycache__', 'models/cache', '.agents/skills/__pycache__']:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / 'src/source.py').write_text('source')
    paths = clean.candidates(tmp_path)
    assert set(paths) == {tmp_path/'target', tmp_path/'node_modules', tmp_path/'src/__pycache__'}
    (tmp_path/'target/source.txt').write_text('keep')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'target/source.txt'], check=True)
    with pytest.raises(ValueError, match='tracked files'):
        clean.candidates(tmp_path)


def test_cleanup_rejects_symlink_targets(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'target').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        clean.candidates(tmp_path)


def test_cleanup_refuses_draft_or_stale_updater(tmp_path, monkeypatch):
    config = tmp_path/'apps/desktop/src-tauri/tauri.conf.json'
    config.parent.mkdir(parents=True)
    config.write_text('{"version":"0.4.0"}')
    monkeypatch.setattr(clean, 'gh', lambda *args: json.dumps({'tagName':'v0.4.0','isDraft':True,'publishedAt':None}))
    with pytest.raises(ValueError, match='not been published'):
        clean.verify_release('v0.4.0', tmp_path)
    with pytest.raises(ValueError, match='match this checkout'):
        clean.verify_release('v0.3.1', tmp_path)


def test_apply_runs_verification_before_deleting_and_dry_run_is_safe(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'target').mkdir()
    (tmp_path/'target/cache').write_text('regenerable')
    (tmp_path/'runs').mkdir()
    evidence = tmp_path/'runs/evidence.json'
    evidence.write_text('preserve')
    monkeypatch.setattr(clean, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['clean_workspace.py'])
    assert clean.main() == 0
    assert (tmp_path/'target/cache').is_file()
    checked = []
    monkeypatch.setattr(clean, 'verify_release', lambda tag, root: checked.append((tag, root)))
    monkeypatch.setattr('sys.argv', ['clean_workspace.py', '--apply', '--release-tag', 'v0.4.0'])
    assert clean.main() == 0
    assert checked == [('v0.4.0', tmp_path)]
    assert not (tmp_path/'target').exists()
    assert evidence.read_text() == 'preserve'
