"""Actual source metadata and transport ordering; no native model fixtures."""

import ast
import json
import os
import sys
import time
from pathlib import Path

import pytest

from scripts import kaggle_advanced_training as runner
from scripts import stage_advanced_training as stager

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(
    os.environ.get(
        "VIDEO_ORIGINAL_SOURCE",
        "D:/CodexData/portfolio/semantic-video-retriever/data/interim/research-cloud-encoding-v5/source",
    )
)
CACHE = ROOT / "data/interim/research-cloud-encoding-v5/terminal-output-v1"


def actual_metadata():
    source = ORIGINAL / "video-source-manifest.json"
    if not source.is_file():
        pytest.skip("Actual original immutable source unavailable")
    assert runner.sha(source) == stager.ORIGINAL_SOURCE_SHA
    return (
        json.loads(source.read_bytes()),
        json.loads((CACHE / "research-training-protocol.json").read_bytes()),
        json.loads((CACHE / "research-features/manifest.json").read_bytes()),
    )


def _isolated_test_actual_accounting_selects_original_preparation(tmp_path):
    original, protocol, cache = actual_metadata()
    rows = [
        row
        for row in original["files"]
        if row["sha256"] == protocol["accounting_sha256"]
        and Path(row["path"]).name == "source-accounting.json"
    ]
    assert len(rows) == 1
    accounting = rows[0]
    folder = Path(accounting["path"]).parent
    selected = [
        accounting,
        *[
            row
            for row in original["files"]
            if row["path"] == (folder / "prepared/manifest.json").as_posix()
        ],
    ]
    assert len(selected) == 2
    for row in selected:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (ORIGINAL / row["flat_name"]).read_bytes()
        target.write_bytes(raw)
        assert runner.sha(target) == row["sha256"]
    assert (
        runner.resolve_preparation(tmp_path, original, protocol, cache)
        == tmp_path / folder
    )
    with pytest.raises(ValueError, match="accounting"):
        runner.resolve_preparation(
            tmp_path,
            {"files": [row for row in original["files"] if row is not accounting]},
            protocol,
            cache,
        )
    assert "torch" not in sys.modules and "numpy" not in sys.modules


def test_actual_accounting_selects_original_preparation(tmp_path):
    actual_metadata()
    # Collection may legitimately load native libraries in other tests.
    # Preserve this entire original test in an isolated interpreter instead.
    import subprocess
    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root),str(root/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
import pytest
module=importlib.import_module('test_advanced_training_transport')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    module._isolated_test_actual_accounting_selects_original_preparation(Path(sys.argv[2]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('actual-isolated-test-PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), str(tmp_path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "actual-isolated-test-PASS"


def test_actual_runner_render_roundtrip_and_timeout():
    raw = (ROOT / stager.OWN[0]).read_bytes().replace(b"\r\n", b"\n")
    token = b"EXPECTED_ADVANCED_TRAIN_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    actual_pin = runner.sha(ROOT / stager.OWN[0])
    rendered = raw.replace(
        token, ("EXPECTED_ADVANCED_TRAIN_PLAN_SHA256 = " + repr(actual_pin)).encode()
    )
    ast.parse(rendered)
    assert b"\r" not in rendered
    assert 0 < runner.remaining(time.monotonic()) <= 36000
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.remaining(time.monotonic() - 36001)
    tree = ast.parse(raw)
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    pip = next(node for node in calls if ast.unparse(node.func) == "subprocess.run")
    assert (
        ast.unparse(next(k.value for k in pip.keywords if k.arg == "timeout"))
        == "remaining(started)"
    )
    first_guard = min(n.lineno for n in calls if ast.unparse(n.func) == "raw_guard")
    helper = next(n.lineno for n in calls if ast.unparse(n.func) == "module")
    native = next(
        n.lineno for n in calls if ast.unparse(n.func) == "advanced_training.train"
    )
    assert first_guard < helper < native
    imports = [
        n.lineno
        for n in ast.walk(run)
        if isinstance(n, ast.ImportFrom) and n.module == "src.models"
    ]
    assert first_guard < min(imports) < native
    assert sum(ast.unparse(n.func) == "guard" and n.lineno > native for n in calls) >= 1


def test_review_gate_precedes_output_creation_and_reserved_metadata_excluded():
    source = (ROOT / "scripts/stage_advanced_training.py").read_text()
    tree = ast.parse(source)
    stage = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage"
    )
    calls = [n for n in ast.walk(stage) if isinstance(n, ast.Call)]
    review = next(
        n.lineno
        for n in calls
        if ast.unparse(n.func) == "portability.read_bound_json"
        and n.args
        and ast.unparse(n.args[0]) == "source_review_path"
    )
    mkdir = min(n.lineno for n in calls if ast.unparse(n.func).endswith(".mkdir"))
    assert review < mkdir
    assert 'upload = output / "upload-metadata"' in source
    assert (
        '"dataset-metadata.json"'
        not in source.split("payload = {", 1)[1].split("plan = {", 1)[0]
    )
    assert stager.source_names(ROOT, {"files": {}}) == set(stager.OWN)
