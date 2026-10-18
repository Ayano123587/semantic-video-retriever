"""Actual recorded metadata contracts and software mutations; no native loads."""

import ast
import copy
import hashlib
import json
import os
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from src.models import research_search as search

PRIVATE = Path(os.environ.get("VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private"))


def read_actual_frozen_inventory(row):
    # Project the exact recorded private prefix; never rewrite frozen JSON.
    historical = PureWindowsPath(row["path"])
    prefix = PureWindowsPath("D:/CodexData/portfolio/quant-research-lab-private")
    relative = historical.relative_to(prefix)
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Invalid historical inventory path")
    path = PRIVATE.joinpath(*relative.parts)
    if (
        not path.resolve().is_relative_to(PRIVATE.resolve())
        or path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
    ):
        raise ValueError("Contained actual inventory file required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Actual frozen inventory SHA differs")
    return json.loads(raw)


@pytest.fixture
def recorded():
    base = PRIVATE / "video-recovered-inventory-v1"
    request = base / "root-global-selection-request-v3.json"
    diagnostic = (
        base
        / "global-selection-publication-v2/terminal-diagnostic-v1/recovered-global-choice-runtime-diagnostic.json"
    )
    if not request.is_file() or not diagnostic.is_file():
        pytest.skip("Actual frozen encoding and recorded runtime metadata unavailable")
    approval = json.loads(request.read_bytes())
    runtime = json.loads(diagnostic.read_bytes())["actual_runtime"]
    freeze = json.loads(
        (base / "root-complete-inventories-freeze-v1.json").read_bytes()
    )
    identity = approval["environment"]
    for row in freeze["inventories"].values():
        assert read_actual_frozen_inventory(row)["environment"] == identity
    policy = {
        "schema": search.RECOVERED_POLICY_SCHEMA,
        "environment": copy.deepcopy(identity),
        "selection_runtime": copy.deepcopy(runtime),
    }
    # This bag replays recorded metadata fields; it is not a fresh native observation.
    observed = {
        "environment": {
            **{key: copy.deepcopy(identity[key]) for key in search.ENCODER_ENV_KEYS},
            "python": runtime["python"],
        },
        "training_runtime": copy.deepcopy(runtime),
    }
    return policy, observed


def _isolated_test_actual_full_encoding_provenance_stays_separate_from_observables(
    recorded, monkeypatch
):
    policy, observed = recorded
    frozen = copy.deepcopy(policy["environment"])
    assert "python" not in frozen and "encoder" in frozen
    monkeypatch.setattr(search, "runtime_observation", lambda: copy.deepcopy(observed))
    search.require_runtime(policy)
    state = search.runtime_state(policy)
    assert set(state["encoder_runtime"]) == set(search.ENCODER_ENV_KEYS)
    assert state["selection_runtime"] == policy["selection_runtime"]
    assert policy["environment"] == frozen and "python" not in policy["environment"]
    assert "torch" not in sys.modules and "numpy" not in sys.modules


def test_actual_full_encoding_provenance_stays_separate_from_observables(recorded, monkeypatch):
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
module=importlib.import_module('test_recovered_consumer_runtime')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    module._isolated_test_actual_full_encoding_provenance_stays_separate_from_observables(module.recorded.__wrapped__(), patch)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('actual-isolated-test-PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "actual-isolated-test-PASS"


def _isolated_test_every_observable_encoder_field_mutation_blocks_before_native(
    recorded, monkeypatch, field
):
    policy, observed = recorded
    observed["environment"][field] = None
    monkeypatch.setattr(search, "runtime_observation", lambda: observed)
    with pytest.raises(ValueError, match="Exact frozen numerical/encoder runtime"):
        search.require_runtime(policy)
    assert "torch" not in sys.modules and "numpy" not in sys.modules


@pytest.mark.parametrize("field", search.ENCODER_ENV_KEYS)
def test_every_observable_encoder_field_mutation_blocks_before_native(recorded, monkeypatch, field):
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
module=importlib.import_module('test_recovered_consumer_runtime')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    module._isolated_test_every_observable_encoder_field_mutation_blocks_before_native(module.recorded.__wrapped__(), patch, sys.argv[2])
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('actual-isolated-test-PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), field],
        check=True, capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "actual-isolated-test-PASS"


def test_unrecognized_observable_field_and_python_disagreement_fail(
    recorded, monkeypatch
):
    policy, observed = recorded
    extra = copy.deepcopy(observed)
    extra["environment"]["encoder"] = policy["environment"]["encoder"]
    monkeypatch.setattr(search, "runtime_observation", lambda: extra)
    with pytest.raises(ValueError, match="field inventory"):
        search.require_runtime(policy)
    wrong = copy.deepcopy(observed)
    wrong["environment"]["python"] = policy["selection_runtime"]["numpy"]
    monkeypatch.setattr(search, "runtime_observation", lambda: wrong)
    with pytest.raises(ValueError, match="Python inventories disagree"):
        search.require_runtime(policy)


def _isolated_test_frozen_interop_initialized_once_before_any_native_objects(
    recorded, monkeypatch
):
    policy, _ = recorded
    expected = policy["selection_runtime"]["interop_threads"]
    assert expected == 2
    state = {"interop": expected + 1}
    calls = []

    def set_interop(value):
        state["interop"] = value
        calls.append(("interop", value))

    controls = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("threads", value)),
        get_num_interop_threads=lambda: state["interop"],
        set_num_interop_threads=set_interop,
        use_deterministic_algorithms=lambda value: calls.append(
            ("deterministic", value)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", controls)
    search.initialize_runtime(policy)
    search.initialize_runtime(policy)
    assert calls.count(("interop", 2)) == 1 and state["interop"] == 2
    assert calls[:3] == [("threads", 2), ("interop", 2), ("deterministic", True)]
    assert not hasattr(controls, "load") and "numpy" not in sys.modules


def test_frozen_interop_initialized_once_before_any_native_objects(recorded, monkeypatch):
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
module=importlib.import_module('test_recovered_consumer_runtime')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    module._isolated_test_frozen_interop_initialized_once_before_any_native_objects(module.recorded.__wrapped__(), patch)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('actual-isolated-test-PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "actual-isolated-test-PASS"


def test_proof_runtime_projection_guard_precedes_native_loading():
    root = Path(search.__file__).resolve().parents[2]
    proof = ast.parse((root / "scripts/kaggle_video_query_proof.py").read_bytes())
    guard = next(
        node
        for node in proof.body
        if isinstance(node, ast.FunctionDef) and node.name == "guard_runtime"
    )
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "search.frozen_runtime_state"
        for node in ast.walk(guard)
    )
