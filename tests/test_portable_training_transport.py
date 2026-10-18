"""Source/actual-metadata transport checks; no fabricated native or price data."""

import ast
import json
import subprocess
import time
from pathlib import Path

import pytest

from scripts import kaggle_portable_training as transport
from scripts import stage_portable_training as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path("D:/CodexData/portfolio/quant-research-lab-private")


def source(name):
    return ast.parse((ROOT / name).read_bytes())


def function(name):
    return next(
        n
        for n in source("scripts/kaggle_portable_training.py").body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_actual_checkpoint_freeze_and_provider_are_external_inputs():
    freeze_path = (
        PRIVATE
        / "video-checkpoint-dataset-stage-v1/root-checkpoint-data-freeze-v1.json"
    )
    if not freeze_path.is_file():
        pytest.skip("Actual independently verified provider metadata required")
    freeze = stager.portability.read_bound_json(
        freeze_path, stager.CHECKPOINT_FREEZE_SHA
    )
    assert len(freeze["remote_files"]) == 19
    assert freeze["version"] == 1 and freeze["dataset_id"] == 12656520
    assert freeze["private"] is True and freeze["status"] == "ready"


def test_stager_refuses_current_unreviewed_dirty_checkout_before_output(tmp_path, monkeypatch):
    # Exercise the dirty-status refusal independently of archive/Git checkout.
    def dirty(command, **kwargs):
        assert command == ["git", "status", "--porcelain"]
        return b" M scripts/stage_portable_training.py\n"
    monkeypatch.setattr(stager.subprocess, "check_output", dirty)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Clean committed"):
        stager.stage(
            output,
            phase="phase1",
            policy_path=None,
            policy_sha256=None,
            service_path=None,
            service_sha256=None,
            source_manifest_path=None,
            source_review_path=None,
            source_review_sha256=None,
            inventory_path=None,
            checkpoint_freeze_path=None,
            provider_path=None,
            inspection_freeze_path=None,
            inspection_report_path=None,
            amd_observation_path=None,
            dataset_id=None,
            kernel_id=None,
        )
    assert not output.exists()


def test_actual_previous_plan_cannot_authorize_new_portability():
    candidates = list(
        PRIVATE.glob("video-gateb-phase1-stage-v1/dataset/recovery-plan.json")
    )
    if not candidates:
        pytest.skip("Actual historical transport plan unavailable")
    with pytest.raises((ValueError, FileNotFoundError)):
        transport.authenticate(candidates[0].parent, transport.sha(candidates[0]))


def test_plain_render_template_has_exact_one_binding_and_compiles():
    raw = (
        (ROOT / "scripts/kaggle_portable_training.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    token = b"EXPECTED_PORTABLE_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    # Actual source digest is used solely as a software render parameter, never
    # as an approval or a purported actual new service plan.
    rendered = raw.replace(
        token,
        (
            "EXPECTED_PORTABLE_PLAN_SHA256 = "
            + repr(transport.sha(ROOT / "scripts/kaggle_portable_training.py"))
        ).encode(),
    )
    compile(rendered, "rendered-runtime.py", "exec")


def test_all_raw_input_checks_precede_helper_and_native_imports():
    run = function("run")
    calls = [
        (n.lineno, ast.unparse(n.func))
        for n in ast.walk(run)
        if isinstance(n, ast.Call)
    ]
    raw = min(line for line, name in calls if name == "raw_inputs")
    helper = min(line for line, name in calls if name == "load_module")
    project_import = min(
        n.lineno
        for n in ast.walk(run)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("src")
    )
    assert raw < helper < project_import
    gate = ast.unparse(function("raw_inputs"))
    assert 'freeze["remote_files"]' in gate or "freeze['remote_files']" in gate
    assert 'source_manifest["files"]' in gate or "source_manifest['files']" in gate
    assert 'cached["files"]' in gate or "cached['files']" in gate
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom))
        for n in ast.walk(function("raw_inputs"))
    )


def test_actual_runtime_diagnostic_written_before_runtime_gate_and_train():
    run = function("run")
    calls = [
        (n.lineno, ast.unparse(n.func))
        for n in ast.walk(run)
        if isinstance(n, ast.Call)
    ]
    diagnostic = min(
        n.lineno
        for n in ast.walk(run)
        if isinstance(n, ast.Constant) and n.value == "runtime-diagnostic.json"
    )
    gate = min(line for line, name in calls if name == "portability.runtime_guard")
    train = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "portable_training.train"
    )
    assert diagnostic < gate < train.lineno
    arguments = next(
        n.value
        for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "arguments" for t in n.targets)
    )
    assert any(
        isinstance(k, ast.Constant)
        and k.value == "service_started"
        and isinstance(v, ast.Name)
        and v.id == "started"
        for k, v in zip(arguments.keys, arguments.values)
    )
    assert any(
        k.arg is None and isinstance(k.value, ast.Name) and k.value.id == "arguments"
        for k in train.keywords
    )


def test_restore_dispatch_is_distinct_from_training_branch():
    run = function("run")
    branch = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "plan['action'] == 'restore_only'"
        and any(
            isinstance(c, ast.Call)
            and ast.unparse(c.func) == "portable_training.restore_only"
            for c in ast.walk(n)
        )
    )
    smoke_calls = {
        ast.unparse(n.func)
        for statement in branch.body
        for n in ast.walk(statement)
        if isinstance(n, ast.Call)
    }
    train_calls = {
        ast.unparse(n.func)
        for statement in branch.orelse
        for n in ast.walk(statement)
        if isinstance(n, ast.Call)
    }
    assert (
        "portable_training.restore_only" in smoke_calls
        and "portable_training.train" not in smoke_calls
    )
    assert (
        "portable_training.train" in train_calls
        and "portable_training.restore_only" not in train_calls
    )


def test_bootstrap_and_postflight_failures_keep_bounded_failure_receipt(
    tmp_path, monkeypatch
):
    # Actual source-only early guard failure, no native imports or inventeddata.
    monkeypatch.setattr(
        transport,
        "unique",
        lambda *_: (_ for _ in ()).throw(ValueError("actual guard exercise")),
    )
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="actual guard"):
        transport.run(tmp_path, output)
    status = json.loads((output / "portable-status.json").read_bytes())
    assert status["scientific_complete"] is False
    assert status["native_loading_started"] is False
    assert status["error_type"] == "ValueError" and len(status["traceback"]) <= 12000


@pytest.mark.parametrize("name", ["../native.pt", "phase/native.pt", "C:/native.pt"])
def test_unsafe_transport_parameter_refused(name):
    with pytest.raises(ValueError):
        transport.safe_name(name, flat=True)


def test_actual_monotonic_positive_and_expired_bootstrap_budgets(monkeypatch):
    actual_start = time.monotonic()
    assert 0 < transport.remaining_budget(actual_start, 1800) <= 1800
    assert 0 < transport.remaining_budget(actual_start, 36000) <= 36000
    # Monotonic origin is platform-dependent; deterministic software-clock test.
    monkeypatch.setattr(transport.time, "monotonic", lambda: 40000.0)
    with pytest.raises(ValueError, match="budget exhausted"):
        transport.remaining_budget(40000.0 - 1801, 1800)


def test_pip_timeout_uses_original_start_and_entire_plan_budget():
    run = function("run")
    pip = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "subprocess.run"
    )
    timeout = next(k.value for k in pip.keywords if k.arg == "timeout")
    assert ast.unparse(timeout) == "remaining_budget(started, plan['max_seconds'])"
    assert not any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "started" for t in n.targets)
        for n in ast.walk(run)
        if getattr(n, "lineno", 0) > pip.lineno
    )


def test_actual_fresh_namespace_checkout_has_complete_source_inventory():
    fresh = PRIVATE / "video-portability-readiness-219a0b4"
    actual = PRIVATE / "video-portable-restore-v2"
    if not fresh.is_dir() or not (actual / "root-policy.json").is_file():
        pytest.skip("Actual fresh checkout and root-authored restore metadata required")
    portability = stager.portability
    evidence = portability.load_evidence(
        PRIVATE
        / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json",
        PRIVATE
        / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json",
        PRIVATE
        / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json",
    )
    policy_sha = "76481000952884b49203be09ed8867f97ea0d2851a438100efd019c0be64f916"
    portability.authenticate_policy(actual / "root-policy.json", policy_sha, evidence)
    service = portability.authenticate_restore_service(
        actual / "phase1-restore-service.json",
        "981eefde446cc94190803854c36d75c7b3d4bdec3863d6115947d7aa3d193ea7",
        policy_sha,
        "phase1",
        evidence,
    )
    manifest = portability.read_bound_json(
        actual / "phase1-source-manifest.json", service["source_manifest_sha256"]
    )
    assert len(manifest["files"]) == 20
    assert not (fresh / "scripts/__init__.py").exists()
    names = stager.source_names(fresh, manifest)
    assert "scripts/__init__.py" not in names
    assert names == set(manifest["files"]) | set(stager.OWN)
    assert all((fresh / name).is_file() for name in names)
    assert all(
        portability.sha(fresh / name) == pin for name, pin in manifest["files"].items()
    )
