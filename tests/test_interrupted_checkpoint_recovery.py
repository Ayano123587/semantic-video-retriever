"""Checks against the actual terminal audit; never open native checkpoints."""

import ast
import hashlib
import os
from pathlib import Path

import pytest

from scripts import interrupted_checkpoint_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/interim/old-training-terminal-audit-v1/terminal-audit-report.json"


def audit_sha():
    if not AUDIT.is_file():
        pytest.skip("Actual private terminal-audit metadata required")
    return hashlib.sha256(AUDIT.read_bytes()).hexdigest()


@pytest.mark.parametrize("phase,count", [("phase1", 10), ("phase2", 8)])
def test_actual_root_checkpoint_freeze_mapping_and_refusals(phase, count):
    import copy
    import json

    from scripts import kaggle_interrupted_attestation as runner

    path = Path(
        os.environ.get(
            "VIDEO_CHECKPOINT_FREEZE",
            ROOT.parents[1]
            / "portfolio/quant-research-lab-private/video-checkpoint-dataset-stage-v1/root-checkpoint-data-freeze-v1.json",
        )
    )
    if not path.exists():
        pytest.skip("Actual root-authored checkpoint freeze required")
    freeze = recovery._json(
        path, "58ce0269abad44dfcc5dbcf866f94e83c40e7824095d770f96ccbb0c40bde87c"
    )
    provider = json.loads(
        (path.parent / "payload/provider-provenance.json").read_bytes()
    )
    reference = recovery.audit_reference(AUDIT, audit_sha(), phase)
    mapping = runner.checkpoint_mapping(freeze, provider, phase, reference)
    assert len(mapping) == count and set(mapping) == set(recovery._required(reference))
    for mutation in ("missing", "extra", "duplicate", "unsafe", "private", "version"):
        bad_freeze, bad_provider = copy.deepcopy(freeze), copy.deepcopy(provider)
        items = bad_provider["jobs"][0]["files"]
        if mutation == "missing":
            items.pop()
        elif mutation == "extra":
            items.append(copy.deepcopy(items[0]))
        elif mutation == "duplicate":
            items[1]["upload_name"] = items[0]["upload_name"]
        elif mutation == "unsafe":
            items[0]["upload_name"] = "../" + items[0]["upload_name"]
        elif mutation == "private":
            bad_freeze["private"] = False
        else:
            bad_freeze["version"] = 2
        with pytest.raises(ValueError):
            runner.checkpoint_mapping(bad_freeze, bad_provider, phase, reference)


@pytest.mark.parametrize(
    "phase,count,batches", [("phase1", 8, 40929), ("phase2", 6, 22028)]
)
def test_actual_cancelled_parent_metadata(phase, count, batches):
    reference = recovery.audit_reference(AUDIT, audit_sha(), phase)
    assert reference["completed_seeds"] == count
    assert reference["batches"] == batches
    assert len(reference["completed_seed_files"]) == count
    assert reference["terminal_status"] == "CANCEL_ACKNOWLEDGED"
    assert reference["snapshot_sha256"] == recovery.PARENTS[phase]["snapshot_sha256"]
    assert len(recovery._required(reference)) == count + 2


def test_external_hash_is_required_for_actual_audit():
    with pytest.raises(recovery.RecoveryError, match="SHA256 required"):
        recovery.audit_reference(AUDIT, None, "phase1")


def test_actual_host_metadata_cannot_authorize_native_hashing():
    with pytest.raises(recovery.RecoveryError, match="cloud-only"):
        recovery.cloud_hash_inventory(AUDIT, audit_sha(), "phase1", AUDIT.parent)


def test_no_native_imports_and_callback_is_between_byte_guards():
    tree = ast.parse(Path(recovery.__file__).read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = {alias.name.split(".")[0] for node in imports for alias in node.names}
    assert not imported.intersection({"torch", "numpy", "pickle", "joblib"})
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "guarded_recover"
    )
    guards = sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "authenticate_frozen_inventory"
    )
    callback = next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resume_callback"
    )
    assert len(guards) == 2 and guards[0] < callback < guards[1]
    assert any(
        isinstance(node, ast.Try) and node.finalbody for node in ast.walk(function)
    )
    source = ast.get_source_segment(Path(recovery.__file__).read_text(), function)
    assert "max_seconds=remaining" in source
    assert "time.monotonic() - service_started" in source


def test_gate_a_runner_is_hash_only_and_writes_failure_receipt():
    path = ROOT / "scripts/kaggle_interrupted_attestation.py"
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"load", "train", "guarded_recover"}
        for node in calls
    )
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assert any(
        isinstance(node, ast.Try) and node.handlers and node.finalbody
        for node in ast.walk(run)
    )
    compile(path.read_text(), str(path), "exec")


def test_gate_a_stager_refuses_current_uncommitted_sources(tmp_path, monkeypatch):
    from scripts import stage_interrupted_attestation as stager
    stage = stager.stage
    def dirty(command, **kwargs):
        assert command == ["git", "status", "--porcelain"]
        return " M scripts/stage_interrupted_attestation.py\n"
    monkeypatch.setattr(stager.subprocess, "check_output", dirty)

    with pytest.raises(ValueError, match="Clean committed"):
        stage(
            tmp_path / "never-created",
            "phase1",
            AUDIT,
            audit_sha(),
            None,
            None,
            None,
            None,
            "unused/unused",
            "unused/unused",
        )
    assert not (tmp_path / "never-created").exists()


def test_flat_roundtrip_actual_metadata_and_exact_header_discovery(tmp_path):
    import json

    from scripts import kaggle_interrupted_attestation as runner

    reference = recovery.audit_reference(AUDIT, audit_sha(), "phase1")
    header = (
        AUDIT.parent / "phase1/small-metadata/research-phase1/checkpoint-000047.json"
    )
    if not header.is_file():
        pytest.skip("Actual private checkpoint header required")
    package = tmp_path / "package"
    package.mkdir()
    files = {}
    for index, path in enumerate([AUDIT, header, Path(recovery.__file__)]):
        raw = path.read_bytes()
        name = f"payload-{index:03d}" + path.suffix
        (package / name).write_bytes(raw)
        files[path.name] = {
            "upload_name": name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    raw = json.dumps(
        {"schema": "interrupted_attestation_transport_v1", "files": files}
    ).encode()
    (package / "attestation-plan.json").write_bytes(raw)
    assert (
        runner.authenticate(package, hashlib.sha256(raw).hexdigest())["files"] == files
    )
    actual_mount = tmp_path / "actual-parent"
    actual_mount.mkdir()
    (actual_mount / header.name).write_bytes(header.read_bytes())
    assert runner.discover_parent(tmp_path, reference) == actual_mount
    with pytest.raises(ValueError, match="unique"):
        runner.discover_parent(package, reference)
    second = tmp_path / "second-parent"
    second.mkdir()
    (second / header.name).write_bytes(header.read_bytes())
    with pytest.raises(ValueError, match="unique"):
        runner.discover_parent(tmp_path, reference)


@pytest.mark.parametrize("invalid", ["missing", "extra", "duplicate", "unsafe"])
def test_actual_flat_native_mapping_rejects_invalid_software_paths_before_cloud(
    invalid,
):
    import json

    provider_path = Path(
        os.environ.get(
            "VIDEO_CHECKPOINT_PROVIDER",
            ROOT.parents[1]
            / "portfolio/quant-research-lab-private/video-checkpoint-dataset-stage-v1/payload/provider-provenance.json",
        )
    )
    if not provider_path.exists():
        pytest.skip("Actual reviewed provider provenance required")
    provider = json.loads(provider_path.read_bytes())
    job = next(job for job in provider["jobs"] if job["phase"] == "phase1")
    mapping = {
        Path(item["logical_path"]).name: item["upload_name"] for item in job["files"]
    }
    keys = list(mapping)
    if invalid == "missing":
        mapping.pop(keys[0])
    elif invalid == "extra":
        mapping["not-an-approved-role"] = keys[0]
    elif invalid == "duplicate":
        mapping[keys[1]] = mapping[keys[0]]
    else:
        mapping[keys[0]] = "../" + mapping[keys[0]]
    with pytest.raises(
        recovery.RecoveryError, match="mapping required|filenames required"
    ):
        recovery.cloud_hash_inventory(
            AUDIT, audit_sha(), "phase1", AUDIT.parent, mapping
        )


def test_actual_gateb_contract_refuses_missing_identity_before_native_or_callback(
    monkeypatch,
):
    import time

    private = ROOT.parents[1] / "portfolio/quant-research-lab-private"
    contract_path = (
        private / "video-phase1-hashcheck-publication-v1/root-recovery-contract-v1.json"
    )
    if not contract_path.exists():
        pytest.skip("Actual root recovery contract required")
    contract = recovery._json(
        contract_path,
        "26419519b1f79b20ded547875dc9d5b3e999eaadf31137e7086964687cf34fc9",
    )
    assert contract["reference"] == recovery.audit_reference(
        AUDIT, audit_sha(), "phase1"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid identity must not reach native path or consumer")

    monkeypatch.setattr(recovery, "_cloud_parent", forbidden)
    with pytest.raises(recovery.RecoveryError, match="complete training identity"):
        recovery.guarded_recover(
            contract_path,
            "26419519b1f79b20ded547875dc9d5b3e999eaadf31137e7086964687cf34fc9",
            audit_path=AUDIT,
            audit_sha256=audit_sha(),
            phase="phase1",
            parent_phase_directory=AUDIT.parent,
            inventory_path=private
            / "video-phase1-hashcheck-publication-v1/terminal-output-v1/interrupted-inventory.json",
            inventory_sha256=contract["inventory_sha256"],
            training_identity={},
            resume_callback=forbidden,
            service_started=time.monotonic(),
            native_paths={},
        )


def test_gateb_actual_inventory_flatmap_forwarded_without_native_reads(monkeypatch):
    import json

    private = ROOT.parents[1] / "portfolio/quant-research-lab-private"
    inventory_path = (
        private
        / "video-phase1-hashcheck-publication-v1/terminal-output-v1/interrupted-inventory.json"
    )
    if not inventory_path.exists():
        pytest.skip("Actual completed cloud inventory required")
    inventory = json.loads(inventory_path.read_bytes())
    provider = json.loads(
        (
            private
            / "video-checkpoint-dataset-stage-v1/payload/provider-provenance.json"
        ).read_bytes()
    )
    mapping = {
        Path(item["logical_path"]).name: item["upload_name"]
        for item in provider["jobs"][0]["files"]
    }

    def metadata_only(audit, pin, phase, parent, native_paths):
        assert native_paths == mapping
        return inventory

    monkeypatch.setattr(recovery, "cloud_hash_inventory", metadata_only)
    assert (
        recovery.authenticate_frozen_inventory(
            AUDIT,
            audit_sha(),
            "phase1",
            AUDIT.parent,
            inventory_path,
            "c1dd92f7b95c566bbaa36937a7b20c018a78afaed9a225ba1d421b1d2b1e9d0d",
            mapping,
        )
        == inventory
    )


def test_gateb_runner_actual_render_and_preload_source_order():
    path = ROOT / "scripts/kaggle_interrupted_recovery.py"
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    token = b"EXPECTED_RECOVERY_PLAN_SHA256" + b" = None"
    assert raw.count(token) == 1
    rendered = raw.replace(
        token,
        b"EXPECTED_RECOVERY_PLAN_SHA256 = '26419519b1f79b20ded547875dc9d5b3e999eaadf31137e7086964687cf34fc9'",
    )
    compile(rendered, str(path), "exec")
    tree = ast.parse(raw)
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    guard = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "guarded_recover"
    )
    callback = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.FunctionDef) and node.name == "resume_callback"
    )
    assert "training_identity=identity" in ast.get_source_segment(raw.decode(), run)
    assert "Original complete training identity/runtime mismatch" in raw.decode()
    assert guard > callback.end_lineno
    callback_calls = [node for node in ast.walk(callback) if isinstance(node, ast.Call)]
    train = next(
        node.lineno
        for node in callback_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "train"
    )
    assert (
        min(
            node.lineno
            for node in callback_calls
            if isinstance(node.func, ast.Name) and node.func.id == "checkpoint_bytes"
        )
        < train
    )
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in ("load", "fit")
        for node in calls
    )
