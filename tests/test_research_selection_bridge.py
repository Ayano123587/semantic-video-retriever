"""Opaque file fixtures only: no arrays, models, training or test labels."""

import copy
import json
from pathlib import Path
import pytest
from scripts import research_selection_bridge as b


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def both(tmp_path):
    configs = [{"name": name} for name in ("lr001", "lr0003", "decay01")]
    scientific = {
        "source_manifest_sha256": "source",
        "cache_manifest_sha256": "cache",
        "protocol_sha256": "protocol",
        "environment": {"cpu": "fixture"},
        "implementation_sha256": {"science.py": "unchanged"},
        "coverage": {"retained_media": 7454},
        "encoding_identity": {"model": "fixture"},
        "baseline": {"queries": 3132},
        "phase1_configs": configs,
        "training_runtime": {"cpu": "fixture"},
    }
    phases = {}
    for phase in ("phase1", "phase2"):
        directory = tmp_path / phase
        directory.mkdir()
        count = b.PHASE_GRIDS[phase] * 3
        identity = {
            key: scientific[key]
            for key in (
                "cache_manifest_sha256",
                "protocol_sha256",
                "implementation_sha256",
                "coverage",
                "encoding_identity",
            )
        }
        identity.update(phase=phase, runtime=scientific["training_runtime"])
        report = {
            "completed_fits": count,
            "required_fits": count,
            "test_targets_used": False,
            "cache_manifest_sha256": "cache",
            "protocol_sha256": "protocol",
            "continuation_identity": identity,
            "baseline": {"queries": 3132},
            "selected_validation": {"queries": 3132},
            "checkpoint_replay_max_abs_error": 0.0,
            "runs": [
                {
                    "seeds": [{"seed": seed} for seed in b.SEEDS],
                    **({"config": config} if phase == "phase1" else {"lambda": weight}),
                }
                for config, weight in zip(
                    configs if phase == "phase1" else [None] * 4,
                    range(3) if phase == "phase1" else [0, 0.5, 1, 2],
                )
            ],
        }
        if phase == "phase2":
            report["policies"] = [
                {
                    "lambda": weight,
                    "adapted_score_weight": alpha,
                    "validation": {"queries": 3132},
                }
                for weight in (0, 0.5, 1, 2)
                for alpha in (0.25, 0.5, 0.75, 1.0)
            ]
        (directory / "selected.pt").write_bytes(b"opaque selected fixture")
        (directory / "checkpoint-000001.pt").write_bytes(b"opaque final fixture")
        report["selected_checkpoint_sha256"] = b.sha(directory / "selected.pt")
        header = {
            "stage": "done",
            "completed_seeds": count,
            "identity_sha256": b.fingerprint(identity),
            "snapshot_sha256": b.sha(directory / "checkpoint-000001.pt"),
        }
        write(directory / "checkpoint-000001.json", header)
        write(directory / "results.json", report)
        for grid in range(b.PHASE_GRIDS[phase]):
            for seed in b.SEEDS:
                (directory / f"config{grid}-seed{seed}.pt").write_bytes(
                    b"opaque seed fixture"
                )
        dispatch = {
            key: scientific[key]
            for key in (
                "source_manifest_sha256",
                "cache_manifest_sha256",
                "protocol_sha256",
                "environment",
            )
        }
        dispatch.update(
            phase=phase,
            result=report,
            test_annotations_used=False,
            train_queries=23202,
            validation_queries=3132,
            transport_sha256="a" * 64,
        )
        write(directory / "dispatch.json", dispatch)
        contract = {
            "phase": phase,
            "kernel": "owner/actual/1",
            "report_sha256": b.sha(directory / "results.json"),
            "dispatch_sha256": b.sha(directory / "dispatch.json"),
            "final_header": "checkpoint-000001.json",
            "final_header_sha256": b.sha(directory / "checkpoint-000001.json"),
            "transport_sha256": "a" * 64,
        }
        phases[phase] = {
            "directory": directory,
            "dispatch": directory / "dispatch.json",
            "contract": contract,
        }
    return phases, scientific


def freeze(tmp_path, phases, scientific):
    actual = b.authenticate_both(phases, scientific)
    frozen = {}
    for phase, value in actual.items():
        path = tmp_path / (phase + "-frozen.json")
        write(path, value)
        frozen[phase] = {"path": path, "sha256": b.sha(path)}
    return frozen


def test_hash_only_complete_attestation_and_selection_boundary(tmp_path, both):
    phases, scientific = both
    frozen = freeze(tmp_path, phases, scientific)
    assert (
        b.guarded_choose(phases, scientific, frozen, lambda: "selected after barrier")
        == "selected after barrier"
    )
    assert b.authenticate_both(phases, scientific)["phase2"]["completed_fits"] == 12


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "policies",
        "seeds",
        "subset",
        "runtime",
        "source",
        "done",
        "selected",
        "missingseed",
        "reordered",
    ],
)
def test_invalid_phase_never_reaches_loader(tmp_path, both, fault):
    phases, scientific = both
    frozen = freeze(tmp_path, phases, scientific)
    row = phases["phase2"]
    directory = row["directory"]
    report = json.loads((directory / "results.json").read_bytes())
    dispatch = json.loads(row["dispatch"].read_bytes())
    if fault == "partial":
        report["completed_fits"] -= 1
    elif fault == "policies":
        report["policies"].pop()
    elif fault == "seeds":
        report["runs"][0]["seeds"].reverse()
    elif fault == "subset":
        report["selected_validation"]["queries"] = 64
    elif fault == "runtime":
        dispatch["environment"] = {"cpu": "different"}
    elif fault == "source":
        dispatch["source_manifest_sha256"] = "changed"
    elif fault == "done":
        header = directory / "checkpoint-000001.json"
        value = json.loads(header.read_bytes())
        value["stage"] = "fit"
        write(header, value)
        row["contract"]["final_header_sha256"] = b.sha(header)
    elif fault == "selected":
        (directory / "selected.pt").write_bytes(b"tampered")
    elif fault == "missingseed":
        (directory / "config0-seed42.pt").unlink()
    else:
        report["policies"].reverse()
    dispatch["result"] = report
    write(directory / "results.json", report)
    write(row["dispatch"], dispatch)
    row["contract"]["report_sha256"] = b.sha(directory / "results.json")
    row["contract"]["dispatch_sha256"] = b.sha(row["dispatch"])
    with pytest.raises(ValueError):
        b.guarded_choose(
            phases,
            scientific,
            frozen,
            lambda: pytest.fail("Loader invoked before both inventories authenticated"),
        )


@pytest.mark.parametrize(
    "fault", ["missingphase", "missinginventory", "changedseed", "changedinventory"]
)
def test_root_frozen_inventory_barrier(tmp_path, both, fault):
    phases, scientific = both
    frozen = freeze(tmp_path, phases, scientific)
    if fault == "missingphase":
        phases.pop("phase1")
    elif fault == "missinginventory":
        frozen.pop("phase1")
    elif fault == "changedseed":
        (phases["phase1"]["directory"] / "config0-seed42.pt").write_bytes(
            b"changed learned bytes"
        )
    else:
        frozen["phase1"]["path"].write_bytes(b"changed inventory")
    with pytest.raises(ValueError):
        b.guarded_choose(
            phases, scientific, frozen, lambda: pytest.fail("Native loader invoked")
        )


def test_bridge_module_has_no_native_deserialization_imports():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(b))
    names = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not {"torch", "numpy", "pickle"}.intersection(names)


def test_metadata_only_phase_barrier_does_not_hash_native_files(both, monkeypatch):
    phases, scientific = both
    row = phases["phase1"]
    original = b.sha

    def metadata_only(path):
        if str(path).endswith(".pt"):
            pytest.fail("Native bytes accessed on metadata-only host path")
        return original(path)

    monkeypatch.setattr(b, "sha", metadata_only)
    result = b.phase_attestation(
        row["directory"],
        row["dispatch"],
        row["contract"],
        scientific,
        hash_learned=False,
    )
    assert result["completed_fits"] == 9


def staged_fixture(tmp_path, both, monkeypatch):
    from scripts import stage_research_bridge as stage

    phases, scientific = both
    for key, value in (
        ("SOURCE_SHA", "source"),
        ("CACHE_SHA", "cache"),
        ("PROTOCOL_SHA", "protocol"),
    ):
        monkeypatch.setattr(stage.training, key, value)
    approval = {
        "schema": "root_frozen_research_bridge_inputs_v1",
        "mode": "inventory",
        "scientific": scientific,
        "phases": {},
    }
    for phase, row in phases.items():
        approval["phases"][phase] = dict(
            row["contract"],
            report_path=str(row["directory"] / "results.json"),
            dispatch_path=str(row["dispatch"]),
        )
    path = tmp_path / "approval.json"
    write(path, approval)
    root = Path(stage.__file__).resolve().parents[1]

    def tracked(args, **kwargs):
        if "status" in args:
            return b""
        return (root / args[-1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(stage.subprocess, "check_output", tracked)
    return stage, path, approval


def test_real_full_runner_stage_render_roundtrip(tmp_path, both, monkeypatch):
    from scripts import kaggle_research_bridge as cloud

    stage, approval, _ = staged_fixture(tmp_path, both, monkeypatch)
    result = stage.stage(
        tmp_path / "stage", approval, b.sha(approval), "owner/transport-fixture"
    )
    monkeypatch.setattr(cloud, "EXPECTED_MANIFEST_SHA256", result["manifest_sha256"])
    manifest = tmp_path / "stage/dataset/research-bridge-manifest.json"
    runtime = tmp_path / "stage/run.py"
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes().replace(b"\n", b"\r\n"))
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes() + b" ")
    with pytest.raises(ValueError, match="runtime"):
        cloud.checked_payload(manifest, runtime)


@pytest.mark.parametrize("fault", ["missing", "partial"])
def test_stager_refuses_missing_or_partial_actual_parents(
    tmp_path, both, monkeypatch, fault
):
    stage, path, approval = staged_fixture(tmp_path, both, monkeypatch)
    report = Path(approval["phases"]["phase1"]["report_path"])
    if fault == "missing":
        report.unlink()
    else:
        data = json.loads(report.read_bytes())
        data["completed_fits"] = 1
        write(report, data)
        approval["phases"]["phase1"]["report_sha256"] = b.sha(report)
        write(path, approval)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.stage(tmp_path / "stage", path, b.sha(path), "owner/transport-fixture")
    assert not (tmp_path / "stage").exists()


def test_selection_never_accepts_inventory_mode_without_root_freezes(both):
    phases, scientific = both
    with pytest.raises(ValueError, match="root-frozen"):
        b.guarded_choose(
            phases,
            scientific,
            None,
            lambda: pytest.fail("Unauthenticated native loader invoked"),
        )


def test_phase_training_runtime_mismatch_precedes_learned_hashes_and_loader(
    tmp_path, both, monkeypatch
):
    phases, scientific = both
    frozen = freeze(tmp_path, phases, scientific)
    row = phases["phase2"]
    report_path = row["directory"] / "results.json"
    report = json.loads(report_path.read_bytes())
    report["continuation_identity"]["runtime"] = {"cpu": "different training CPU"}
    write(report_path, report)
    dispatch = json.loads(row["dispatch"].read_bytes())
    assert dispatch["environment"] == scientific["environment"]
    dispatch["result"] = report
    write(row["dispatch"], dispatch)
    header_path = row["directory"] / row["contract"]["final_header"]
    header = json.loads(header_path.read_bytes())
    header["identity_sha256"] = b.fingerprint(report["continuation_identity"])
    write(header_path, header)
    row["contract"].update(
        report_sha256=b.sha(report_path),
        dispatch_sha256=b.sha(row["dispatch"]),
        final_header_sha256=b.sha(header_path),
    )
    original = b.sha

    def reject_phase2_learned_hash(path):
        if Path(path).parent == row["directory"] and Path(path).suffix == ".pt":
            pytest.fail("Mismatched training runtime reached learned file hashing")
        return original(path)

    monkeypatch.setattr(b, "sha", reject_phase2_learned_hash)
    with pytest.raises(ValueError, match="learned training numerical runtime"):
        b.guarded_choose(
            phases,
            scientific,
            frozen,
            lambda: pytest.fail("Mismatched runtime reached native loader"),
        )


def test_callback_mutation_is_rejected_before_result_release(tmp_path, both):
    phases, scientific = both
    frozen = freeze(tmp_path, phases, scientific)

    def tampering_callback():
        (phases["phase2"]["directory"] / "selected.pt").write_bytes(
            b"mutated during callback"
        )
        return "must not be released"

    with pytest.raises(ValueError, match="learned artifact changed"):
        b.guarded_choose(phases, scientific, frozen, tampering_callback)
