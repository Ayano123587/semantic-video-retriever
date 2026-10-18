"""Hash-only completed-phase attestation and guarded global-choice boundary."""

import hashlib
import json
import math
from pathlib import Path

SEEDS = (42, 1337, 2026)
PHASE_GRIDS = {"phase1": 3, "phase2": 4}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError("Externally frozen metadata fingerprint differs")
    return json.loads(Path(path).read_bytes())


def phase_attestation(
    directory, dispatch_path, contract, scientific, hash_learned=True
):
    """Never deserialize learned files. Require actual complete metadata first."""
    phase = contract["phase"]
    if phase not in PHASE_GRIDS:
        raise ValueError("Unknown phase")
    directory = Path(directory)
    report = read_bound(directory / "results.json", contract["report_sha256"])
    dispatch = read_bound(dispatch_path, contract["dispatch_sha256"])
    header_name = contract["final_header"]
    if (
        Path(header_name).name != header_name
        or not header_name.startswith("checkpoint-")
        or not header_name.endswith(".json")
    ):
        raise ValueError("Safe final checkpoint header required")
    header = read_bound(directory / header_name, contract["final_header_sha256"])
    expected_fits = PHASE_GRIDS[phase] * len(SEEDS)
    if any(
        type(report.get(key)) is not int or report[key] != expected_fits
        for key in ("completed_fits", "required_fits")
    ):
        raise ValueError("Both complete fixed fit grids required")
    if (
        report.get("test_targets_used") is not False
        or dispatch.get("test_annotations_used") is not False
    ):
        raise ValueError("No-test phase proof required")
    if (
        dispatch.get("phase") != phase
        or dispatch.get("result") != report
        or dispatch.get("train_queries") != 23202
        or dispatch.get("validation_queries") != 3132
    ):
        raise ValueError("Actual complete full-cohort dispatch required")
    for key in ("source_manifest_sha256", "cache_manifest_sha256", "protocol_sha256"):
        if dispatch.get(key) != scientific[key]:
            raise ValueError("Original source/cache/protocol changed")
    if dispatch.get("transport_sha256") != contract["transport_sha256"]:
        raise ValueError("Actual executed phase transport differs")
    if dispatch.get("environment") != scientific["environment"]:
        raise ValueError("Complete phases require the frozen numerical environment")
    for key in ("cache_manifest_sha256", "protocol_sha256"):
        if report.get(key) != scientific[key]:
            raise ValueError("Report scientific identity differs")
    identity = report.get("continuation_identity", {})
    if (
        identity.get("phase") != phase
        or identity.get("implementation_sha256") != scientific["implementation_sha256"]
        or identity.get("coverage") != scientific["coverage"]
        or identity.get("encoding_identity") != scientific["encoding_identity"]
    ):
        raise ValueError("Complete source/runtime continuation identity required")
    if (
        identity.get("cache_manifest_sha256") != scientific["cache_manifest_sha256"]
        or identity.get("protocol_sha256") != scientific["protocol_sha256"]
    ):
        raise ValueError("Continuation protocol/cache differs")
    if identity.get("runtime") != scientific["training_runtime"]:
        raise ValueError("Exact learned training numerical runtime required")
    if (
        header.get("stage") != "done"
        or type(header.get("completed_seeds")) is not int
        or header["completed_seeds"] != expected_fits
        or header.get("identity_sha256") != fingerprint(identity)
    ):
        raise ValueError("Actual final done checkpoint barrier required")
    if (
        report.get("baseline", {}).get("queries") != 3132
        or report.get("selected_validation", {}).get("queries") != 3132
    ):
        raise ValueError("Full 3132 validation queries required")
    if report.get("baseline") != scientific["baseline"]:
        raise ValueError("Frozen full-cohort baseline differs")
    replay = report.get("checkpoint_replay_max_abs_error")
    if (
        isinstance(replay, bool)
        or not isinstance(replay, (int, float))
        or not math.isfinite(replay)
        or replay > 1e-6
        or replay < 0
    ):
        raise ValueError("Saved selected-policy replay proof required")
    runs = report.get("runs", [])
    if len(runs) != PHASE_GRIDS[phase] or any(
        [item.get("seed") for item in run.get("seeds", [])] != list(SEEDS)
        for run in runs
    ):
        raise ValueError("Complete ordered config/seed reports required")
    if phase == "phase1":
        if [run.get("config") for run in runs] != scientific["phase1_configs"]:
            raise ValueError("Fixed phase1 config order required")
    else:
        if [run.get("lambda") for run in runs] != [0, 0.5, 1, 2]:
            raise ValueError("Fixed phase2 lambda order required")
        policies = report.get("policies", [])
        if [
            (row.get("lambda"), row.get("adapted_score_weight")) for row in policies
        ] != [
            (weight, alpha)
            for weight in (0, 0.5, 1, 2)
            for alpha in (0.25, 0.5, 0.75, 1.0)
        ] or any(
            row.get("validation", {}).get("queries") != 3132 for row in policies
        ):
            raise ValueError("All 16 ordered full-validation policies required")
    if not hash_learned:
        return {
            "phase": phase,
            "completed_fits": expected_fits,
            "validation_queries": 3132,
        }
    files = {
        "results.json": contract["report_sha256"],
        header_name: contract["final_header_sha256"],
    }
    snapshot = Path(header_name).with_suffix(".pt").name
    files[snapshot] = header["snapshot_sha256"]
    files["selected.pt"] = report["selected_checkpoint_sha256"]
    for name, expected in files.items():
        if sha(directory / name) != expected:
            raise ValueError("Selected/final learned artifact changed")
    expected_names = [
        f"config{grid}-seed{seed}.pt"
        for grid in range(PHASE_GRIDS[phase])
        for seed in SEEDS
    ]
    actual_names = sorted(path.name for path in directory.glob("config*-seed*.pt"))
    if actual_names != sorted(expected_names):
        raise ValueError("Complete learned seed inventory required")
    for name in expected_names:
        files[name] = sha(directory / name)
    return {
        "schema": "complete_research_phase_inventory_v1",
        "phase": phase,
        "kernel": contract["kernel"],
        "dispatch_sha256": contract["dispatch_sha256"],
        "transport_sha256": contract["transport_sha256"],
        "files": files,
        "completed_fits": expected_fits,
        "validation_queries": 3132,
        "source_manifest_sha256": scientific["source_manifest_sha256"],
        "cache_manifest_sha256": scientific["cache_manifest_sha256"],
        "protocol_sha256": scientific["protocol_sha256"],
        "environment": scientific["environment"],
        "test_targets_used": False,
        "learned_files_deserialized": False,
    }


def authenticate_both(phase_inputs, scientific, frozen_inventories=None):
    if set(phase_inputs) != {"phase1", "phase2"}:
        raise ValueError("BOTH actual complete phases required")
    if frozen_inventories is not None and set(frozen_inventories) != set(phase_inputs):
        raise ValueError("BOTH externally frozen inventories required before loading")
    inventories = {}
    for phase in ("phase1", "phase2"):
        row = phase_inputs[phase]
        if row["contract"]["phase"] != phase:
            raise ValueError("Phase input role differs")
        actual = phase_attestation(
            row["directory"], row["dispatch"], row["contract"], scientific
        )
        if frozen_inventories is not None:
            frozen = read_bound(
                frozen_inventories[phase]["path"], frozen_inventories[phase]["sha256"]
            )
            if actual != frozen:
                raise ValueError(
                    "Root-frozen actual learned inventory differs before load"
                )
        inventories[phase] = actual
    return inventories


def guarded_choose(phase_inputs, scientific, frozen_inventories, choose_callback):
    if frozen_inventories is None:
        raise ValueError(
            "Both externally root-frozen inventories required for selection"
        )
    authenticate_both(phase_inputs, scientific, frozen_inventories)
    result = choose_callback()
    authenticate_both(phase_inputs, scientific, frozen_inventories)
    return result
