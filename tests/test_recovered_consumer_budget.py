"""Whole-service timing controls only; no market/model fixtures or native loads."""

import ast
import importlib
import time
from pathlib import Path

import pytest

MODULES = ("scripts.kaggle_video_query_proof",)


@pytest.mark.parametrize("name", MODULES)
def test_original_monotonic_start_and_external_positive_budget(name):
    module = importlib.import_module(name)
    now = time.monotonic()
    assert 36000 < module.remaining_budget(now, 36001) <= 36001
    with pytest.raises(TimeoutError, match="exhausted"):
        module.remaining_budget(max(0, now - 2), 1)
    for invalid in (None, False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_seconds"):
            module.remaining_budget(now, invalid)
    with pytest.raises(ValueError, match="monotonic"):
        module.remaining_budget(float("nan"), 1800)


@pytest.mark.parametrize("name", MODULES)
def test_deadline_covers_native_operation_and_disarms_on_failure(name):
    module = importlib.import_module(name)
    tree = ast.parse(Path(module.__file__).read_bytes())
    budget = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "service_budget"
    )
    assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(budget))
    calls = [n for n in ast.walk(budget) if isinstance(n, ast.Call)]
    assert any(
        ast.unparse(n.func) == "signal.setitimer"
        and len(n.args) == 2
        and ast.unparse(n.args[1]) == "remaining()"
        for n in calls
    )
    assert any(
        ast.unparse(n.func) == "signal.setitimer"
        and len(n.args) == 2
        and isinstance(n.args[1], ast.Constant)
        and n.args[1].value == 0
        for n in calls
    )
    source = Path(module.__file__).read_text()
    assert "with service_budget(started," in source
    if name != "scripts.recovered_global_selection":
        runs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "subprocess.run"
        ]
        assert runs and all(
            any(
                k.arg == "timeout" and ast.unparse(k.value) == "remaining()"
                for k in n.keywords
            )
            for n in runs
        )
