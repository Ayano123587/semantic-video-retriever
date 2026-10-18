"""Pure transport tests; opaque software fixtures never load learned models."""

import json
from pathlib import Path
import pytest
from scripts import research_test_transport as t


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def choice(tmp_path):
    inventories = {}
    phase_reports = {}
    env = {"cpu": "fixture"}
    for phase, count in (("phase1", 9), ("phase2", 12)):
        report_file = tmp_path / (phase + "-report.json")
        write(
            report_file,
            {
                "completed_fits": count,
                "required_fits": count,
                "test_targets_used": False,
                "continuation_identity": {"runtime": {"cpu": "fixture"}},
            },
        )
        phase_reports[phase] = {"path": str(report_file), "sha256": t.sha(report_file)}
        path = tmp_path / (phase + ".json")
        write(
            path,
            {
                "phase": phase,
                "files": {"results.json": phase_reports[phase]["sha256"]},
                "completed_fits": count,
                "validation_queries": 3132,
                "test_targets_used": False,
                "learned_files_deserialized": False,
                "cache_manifest_sha256": t.CACHE_SHA,
                "protocol_sha256": "protocol",
                "environment": env,
            },
        )
        inventories[phase] = {"path": str(path), "sha256": t.sha(path)}
    report = {
        "test_targets_used": False,
        "baseline": {"queries": 3132},
        "selected_validation": {"queries": 3132},
        "phases": [
            {"phase": phase, "report_sha256": phase_reports[phase]["sha256"]}
            for phase in ("phase1", "phase2")
        ],
        "cache_manifest_sha256": t.CACHE_SHA,
        "protocol_sha256": "protocol",
        "selected_checkpoint_sha256": "opaquecheckpoint",
    }
    report_path = tmp_path / "results.json"
    write(report_path, report)
    dispatch_path = tmp_path / "dispatch.json"
    write(
        dispatch_path,
        {
            "schema": "both_phase_global_selection_dispatch_v1",
            "result": report,
            "test_targets_used": False,
            "validation_queries": 3132,
            "inventory_sha256": {
                phase: row["sha256"] for phase, row in inventories.items()
            },
        },
    )
    return {
        "global_choice": {
            "report_path": str(report_path),
            "report_sha256": t.sha(report_path),
            "dispatch_path": str(dispatch_path),
            "dispatch_sha256": t.sha(dispatch_path),
            "checkpoint_sha256": "opaquecheckpoint",
            "kernel": "owner/global-choice/1",
        },
        "inventories": inventories,
        "phase_reports": phase_reports,
        "protocol_sha256": "protocol",
        "environment": env,
        "training_runtime": {"cpu": "fixture"},
    }


def test_actual_global_metadata_barrier(choice):
    assert t.checked_global_choice(choice)["baseline"]["queries"] == 3132


@pytest.mark.parametrize(
    "fault", ["missing", "partial", "subset", "inventory", "checkpoint"]
)
def test_global_metadata_incomplete_or_changed_rejected(choice, fault):
    if fault == "missing":
        Path(choice["global_choice"]["report_path"]).unlink()
    elif fault == "inventory":
        Path(choice["inventories"]["phase2"]["path"]).write_bytes(b"changed")
    elif fault == "checkpoint":
        choice["global_choice"]["checkpoint_sha256"] = "changed"
    else:
        path = Path(choice["global_choice"]["report_path"])
        report = json.loads(path.read_bytes())
        if fault == "partial":
            report["phases"].pop()
        else:
            report["selected_validation"]["queries"] = 64
        write(path, report)
        choice["global_choice"]["report_sha256"] = t.sha(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        t.checked_global_choice(choice)


@pytest.fixture
def packed(tmp_path, monkeypatch, choice):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (prepared / "media.json").write_bytes(b"opaque prepared data")
    (cache / "test-moments.npy").write_bytes(b"not a valid native array")
    write(
        prepared / "manifest.json",
        {"files": {"media.json": t.sha(prepared / "media.json")}},
    )
    write(
        cache / "manifest.json",
        {"files": {"test-moments.npy": t.sha(cache / "test-moments.npy")}},
    )
    official = tmp_path / "official.json"
    official.write_bytes(b"invalid JSON proves no target parser used while packing")
    cohort = tmp_path / "cohort.json"
    cohort.write_bytes(b"opaque cohort fixture")
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    for key, path in (
        ("PREPARED_SHA", prepared / "manifest.json"),
        ("CACHE_SHA", cache / "manifest.json"),
        ("OFFICIAL_SHA", official),
        ("COHORT_SHA", cohort),
    ):
        monkeypatch.setattr(t, key, t.sha(path))
    approval = choice
    approval["global_choice"]["checkpoint_sha256"] = t.sha(checkpoint)
    report_path = Path(approval["global_choice"]["report_path"])
    report = json.loads(report_path.read_bytes())
    report["cache_manifest_sha256"] = t.CACHE_SHA
    report["selected_checkpoint_sha256"] = t.sha(checkpoint)
    write(report_path, report)
    approval["global_choice"]["report_sha256"] = t.sha(report_path)
    for row in approval["inventories"].values():
        path = Path(row["path"])
        value = json.loads(path.read_bytes())
        value["cache_manifest_sha256"] = t.CACHE_SHA
        write(path, value)
        row["sha256"] = t.sha(path)
    dispatch_path = Path(approval["global_choice"]["dispatch_path"])
    dispatch = json.loads(dispatch_path.read_bytes())
    dispatch["result"] = report
    dispatch["inventory_sha256"] = {
        phase: row["sha256"] for phase, row in approval["inventories"].items()
    }
    write(dispatch_path, dispatch)
    approval["global_choice"]["dispatch_sha256"] = t.sha(dispatch_path)

    def freeze(source, cache_path, selected):
        assert t.sha(source / "manifest.json") == t.PREPARED_SHA
        assert t.sha(source / "test-cohort.json") == t.COHORT_SHA
        write(
            source / t.POLICY, {"test_targets_used": False, "selected": t.sha(selected)}
        )
        return {"test_targets_used": False}

    output = tmp_path / "packed"
    receipt = t.pack(
        prepared, cache, checkpoint, official, cohort, {}, output, approval, freeze
    )
    return output, receipt, approval


def test_pack_never_parses_gold_or_changes_prepared_manifest(packed):
    output, receipt, global_approval = packed
    assert receipt["test_gold_parsed"] is False
    assert t.sha(output / "private-prepared/manifest.json") == t.PREPARED_SHA
    t.authenticate_stage(
        output / "cloud-test-stage.json",
        receipt["stage_sha256"],
        receipt["policy_sha256"],
    )


def test_changed_raw_gold_rejected_before_policy_callback(tmp_path, monkeypatch):
    raw = tmp_path / "raw.json"
    raw.write_bytes(b"different raw bytes")
    with pytest.raises(ValueError, match="raw official"):
        t.pack(
            tmp_path,
            tmp_path,
            "missing.pt",
            raw,
            "missing.json",
            {},
            tmp_path / "output",
            {},
            lambda *_: pytest.fail(
                "Policy freeze invoked before raw test authentication"
            ),
        )


def test_failed_release_preserves_marker_and_forbids_retry(tmp_path, packed):
    output, receipt, global_approval = packed
    workspace = tmp_path / "workspace"
    t.restore_stage(
        output / "cloud-test-stage.json",
        receipt["stage_sha256"],
        receipt["policy_sha256"],
        workspace,
    )
    approval = dict(
        global_approval,
        stage_sha256=receipt["stage_sha256"],
        policy_sha256=receipt["policy_sha256"],
    )
    result = tmp_path / "receipt.json"

    def failing_scientific_release(source, cache, checkpoint, official, report):
        with (source / t.MARKER).open("x") as handle:
            handle.write("attempt before parser")
        assert (source / t.MARKER).exists()
        json.loads(Path(official).read_bytes())

    with pytest.raises(json.JSONDecodeError):
        t.release_once(workspace, approval, failing_scientific_release, result)
    assert (result.parent / t.MARKER).exists()
    assert json.loads(result.read_bytes())["marker_preserved"] is True
    with pytest.raises(ValueError, match="never retry"):
        t.release_once(
            workspace, approval, lambda *_: pytest.fail("Retried holdout"), result
        )


def test_stage_tamper_fails_before_restore(tmp_path, packed):
    output, receipt, global_approval = packed
    (output / "selected.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload changed"):
        t.restore_stage(
            output / "cloud-test-stage.json",
            receipt["stage_sha256"],
            receipt["policy_sha256"],
            tmp_path / "workspace",
        )
    assert not (tmp_path / "workspace").exists()


def test_complete_cloud_runner_render_and_metadata_only_stage(
    tmp_path, choice, monkeypatch
):
    from scripts import stage_research_test_cloud as staging
    from scripts import kaggle_research_test_cloud as cloud

    monkeypatch.setattr(staging.training, "PROTOCOL_SHA", "protocol")
    cohort = tmp_path / "cohort.json"
    cohort.write_bytes(b"metadata-only cohort fixture")
    monkeypatch.setattr(t, "COHORT_SHA", t.sha(cohort))
    contract = dict(
        choice,
        schema="root_frozen_cloud_test_approval_v1",
        action="pack",
        cohort_sha256=t.COHORT_SHA,
        official_test_sha256=t.OFFICIAL_SHA,
        source_manifest_sha256=staging.training.SOURCE_SHA,
        cohort_path=str(cohort),
    )
    approval = tmp_path / "approval.json"
    write(approval, contract)
    root = Path(staging.__file__).resolve().parents[1]

    def tracked(args, **kwargs):
        if "status" in args:
            return b""
        return (root / args[-1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(staging.subprocess, "check_output", tracked)
    original = t.sha

    def metadata_only(path):
        assert not str(path).endswith(
            (".pt", ".npy")
        ), "Host stager accessed native data"
        return original(path)

    monkeypatch.setattr(t, "sha", metadata_only)
    result = staging.stage(
        tmp_path / "stage", approval, t.sha(approval), "owner/private-test-source"
    )
    monkeypatch.setattr(cloud, "EXPECTED_MANIFEST_SHA256", result["manifest_sha256"])
    manifest = tmp_path / "stage/dataset/cloud-test-transport.json"
    runtime = tmp_path / "stage/run.py"
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes().replace(b"\n", b"\r\n"))
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes() + b" ")
    with pytest.raises(ValueError, match="runtime"):
        cloud.checked_payload(manifest, runtime)


def test_test_source_closure_includes_transitive_project_imports():
    import ast
    from scripts.stage_research_test_cloud import RELEASE_CODE

    root = Path(__file__).resolve().parents[1]
    for name in RELEASE_CODE:
        for node in ast.walk(ast.parse((root / name).read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "src."
            ):
                relative = node.module.replace(".", "/")
                target = relative + ".py"
                if not (root / target).exists():
                    target = relative + "/__init__.py"
                assert target in RELEASE_CODE, (name, target)


def test_runtime_gate_precedes_scientific_policy_and_release_calls():
    import ast
    import inspect
    from scripts import kaggle_research_test_cloud as cloud

    tree = ast.parse(inspect.getsource(cloud.main))
    runtime_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_runtime"
    ]
    scientific_import = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.models.moment_evaluation"
    )
    assert min(runtime_calls) < scientific_import
