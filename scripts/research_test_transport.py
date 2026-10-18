"""Stdlib boundaries for cloud-only policy packing and one-shot holdout release."""

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

COHORT_SHA = "6e4dcdfe816138cae7803db0977d9b9300a3593cd0d49167b64b2b270d49be38"
OFFICIAL_SHA = "1891c04ec48b3d364c739594b2b6413806b74bd9027c092d896e7ebb930ff1cd"
PREPARED_SHA = "ad99d633cee842d81a237dcd95e75f39edcf1589392963a3f5310fbf0cbda09b"
CACHE_SHA = "ea73dced79e83cfc735652d6f8e702011d4e663c3fd9307ef010f66b9120a785"
MARKER = "test-release-marker.json"
POLICY = "frozen-test-policy.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_json(path, expected):
    if sha(path) != expected:
        raise ValueError("Actual externally frozen metadata differs")
    return json.loads(Path(path).read_bytes())


def checked_global_choice(approval, recovered_checker=None):
    """Host path reads only actual small frozen metadata, never weights/arrays."""
    if (
        approval.get("schema") == "root_frozen_cloud_test_approval_v2"
        or approval.get("phase_provenance_schema") == "recovered_v2"
    ):
        if recovered_checker is None:
            from src.models.recovered_policy import (
                checked_global_choice as recovered_checker,
            )
        return recovered_checker(approval)
    global_choice = approval["global_choice"]
    report = bound_json(global_choice["report_path"], global_choice["report_sha256"])
    dispatch = bound_json(
        global_choice["dispatch_path"], global_choice["dispatch_sha256"]
    )
    if (
        dispatch.get("schema") != "both_phase_global_selection_dispatch_v1"
        or dispatch.get("result") != report
        or dispatch.get("test_targets_used") is not False
        or dispatch.get("validation_queries") != 3132
    ):
        raise ValueError("Actual complete both-phase global selection required")
    if (
        report.get("test_targets_used") is not False
        or report.get("baseline", {}).get("queries") != 3132
        or report.get("selected_validation", {}).get("queries") != 3132
        or [phase.get("phase") for phase in report.get("phases", [])]
        != ["phase1", "phase2"]
    ):
        raise ValueError("Both complete full-validation phase choices required")
    if (
        report.get("cache_manifest_sha256") != CACHE_SHA
        or report.get("protocol_sha256") != approval["protocol_sha256"]
        or report.get("selected_checkpoint_sha256")
        != global_choice["checkpoint_sha256"]
    ):
        raise ValueError("Frozen global learned policy identity differs")
    for phase, count in (("phase1", 9), ("phase2", 12)):
        row = approval["inventories"][phase]
        inventory = bound_json(row["path"], row["sha256"])
        if (
            inventory.get("phase") != phase
            or inventory.get("completed_fits") != count
            or inventory.get("validation_queries") != 3132
            or inventory.get("test_targets_used") is not False
            or inventory.get("learned_files_deserialized") is not False
            or dispatch.get("inventory_sha256", {}).get(phase) != row["sha256"]
        ):
            raise ValueError("Both actual root-frozen completed inventories required")
        phase_row = approval["phase_reports"][phase]
        phase_report = bound_json(phase_row["path"], phase_row["sha256"])
        evidence = next(row for row in report["phases"] if row["phase"] == phase)
        if (
            evidence.get("report_sha256") != phase_row["sha256"]
            or inventory.get("files", {}).get("results.json") != phase_row["sha256"]
        ):
            raise ValueError(
                "Global choice/learned inventory must bind actual phase report"
            )
        if (
            phase_report.get("completed_fits") != count
            or phase_report.get("required_fits") != count
            or phase_report.get("test_targets_used") is not False
            or phase_report.get("continuation_identity", {}).get("runtime")
            != approval["training_runtime"]
        ):
            raise ValueError(
                "Actual phase training runtime and complete fit proof required"
            )
        if (
            inventory.get("cache_manifest_sha256") != CACHE_SHA
            or inventory.get("protocol_sha256") != approval["protocol_sha256"]
            or inventory.get("environment") != approval["environment"]
        ):
            raise ValueError("Global phase scientific/runtime identity differs")
    return report


def execution_runtime(approval):
    recovered = (
        approval.get("schema")
        in ("root_frozen_cloud_test_approval_v2", "cloud_frozen_test_stage_v2")
        or approval.get("phase_provenance_schema") == "recovered_v2"
    )
    return approval["selection_runtime"] if recovered else approval["training_runtime"]


def checked_raw_test(path):
    """Hash before any caller is permitted to parse official holdout bytes."""
    if sha(path) != OFFICIAL_SHA:
        raise ValueError("Pinned raw official test bytes differ")


def authenticate_stage(manifest, expected_sha, policy_sha):
    stage = bound_json(manifest, expected_sha)
    if (
        stage.get("schema")
        not in ("cloud_frozen_test_stage_v1", "cloud_frozen_test_stage_v2")
        or stage.get("test_gold_parsed") is not False
        or stage.get("official_test_sha256") != OFFICIAL_SHA
        or stage.get("cohort_sha256") != COHORT_SHA
        or stage.get("frozen_policy_sha256") != policy_sha
    ):
        raise ValueError("Actual root-frozen cloud policy/stage required")
    seen = set()
    flat_seen = set()
    for row in stage["files"]:
        relative = PurePosixPath(row["path"])
        flat = row["flat_name"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row["path"]
            or ":" in row["path"]
            or row["path"] in seen
            or Path(flat).name != flat
            or flat in flat_seen
        ):
            raise ValueError("Safe unique cloud stage paths required")
        seen.add(row["path"])
        flat_seen.add(flat)
        if sha(Path(manifest).parent / flat) != row["sha256"]:
            raise ValueError("Frozen cloud stage payload changed before native load")
    required = {
        "prepared/manifest.json",
        "cache/manifest.json",
        "prepared/test-cohort.json",
        "prepared/" + POLICY,
        "selected.pt",
        "official-test.json",
    }
    if not required.issubset(seen):
        raise ValueError("Complete frozen stage required")
    return stage


def pack(
    prepared,
    cache,
    checkpoint,
    official_test,
    cohort,
    code,
    output,
    approval,
    freeze_callback,
):
    """Cloud only. Official test bytes are copied and hashed but never decoded."""
    prepared, cache, output = Path(prepared), Path(cache), Path(output)
    if output.exists() or (prepared / MARKER).exists() or (prepared / POLICY).exists():
        raise ValueError("Fresh unused prepared source and pack output required")
    checked_raw_test(official_test)
    checked_global_choice(approval)
    if (
        sha(cohort) != COHORT_SHA
        or sha(prepared / "manifest.json") != PREPARED_SHA
        or sha(cache / "manifest.json") != CACHE_SHA
        or sha(checkpoint) != approval["global_choice"]["checkpoint_sha256"]
    ):
        raise ValueError("Actual frozen model/cohort/source/cache inputs required")
    prepared_manifest = json.loads((prepared / "manifest.json").read_bytes())
    cache_manifest = json.loads((cache / "manifest.json").read_bytes())
    for directory, metadata in ((prepared, prepared_manifest), (cache, cache_manifest)):
        for name, expected in metadata["files"].items():
            if Path(name).name != name or sha(directory / name) != expected:
                raise ValueError("Prepared/cache bytes differ before policy freeze")
    output.mkdir(parents=True)
    source = output / "private-prepared"
    source.mkdir()
    for name in ("manifest.json", *prepared_manifest["files"]):
        shutil.copyfile(prepared / name, source / name)
    shutil.copyfile(cohort, source / "test-cohort.json")
    if sha(source / "manifest.json") != PREPARED_SHA:
        raise ValueError("Prepared manifest cannot change when attaching sidecar")
    policy = freeze_callback(source, cache, checkpoint)
    policy_sha = sha(source / POLICY)
    entries = []

    def copy(original, relative):
        flat = relative.replace("/", "__")
        shutil.copyfile(original, output / flat)
        entries.append(
            {"path": relative, "flat_name": flat, "sha256": sha(output / flat)}
        )

    for relative, original in code.items():
        copy(original, relative)
    for name in (
        "manifest.json",
        *prepared_manifest["files"],
        "test-cohort.json",
        POLICY,
    ):
        copy(source / name, "prepared/" + name)
    for name in ("manifest.json", *cache_manifest["files"]):
        copy(cache / name, "cache/" + name)
    copy(checkpoint, "selected.pt")
    copy(official_test, "official-test.json")
    stage = {
        "schema": "cloud_frozen_test_stage_v2"
        if approval.get("schema") == "root_frozen_cloud_test_approval_v2"
        else "cloud_frozen_test_stage_v1",
        "files": entries,
        "frozen_policy_sha256": policy_sha,
        "global_choice_checkpoint_sha256": approval["global_choice"][
            "checkpoint_sha256"
        ],
        "global_choice_report_sha256": approval["global_choice"]["report_sha256"],
        "global_choice_dispatch_sha256": approval["global_choice"]["dispatch_sha256"],
        "environment": approval["environment"],
        (
            "selection_runtime"
            if approval.get("schema") == "root_frozen_cloud_test_approval_v2"
            else "training_runtime"
        ): execution_runtime(approval),
        "official_test_sha256": OFFICIAL_SHA,
        "cohort_sha256": COHORT_SHA,
        "test_gold_parsed": False,
        "identities": policy,
    }
    manifest = output / "cloud-test-stage.json"
    manifest.write_text(
        json.dumps(stage, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "stage_sha256": sha(manifest),
        "policy_sha256": policy_sha,
        "test_gold_parsed": False,
    }


def restore_stage(manifest, expected_sha, policy_sha, workspace):
    stage = authenticate_stage(manifest, expected_sha, policy_sha)
    workspace = Path(workspace)
    workspace.mkdir(exist_ok=False)
    for row in stage["files"]:
        target = workspace / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(manifest).parent / row["flat_name"], target)
    checked_raw_test(workspace / "official-test.json")
    if sha(workspace / "prepared" / POLICY) != policy_sha:
        raise ValueError("Restored root-frozen policy changed")
    return stage


def release_once(workspace, approval, release_callback, receipt_path):
    """Scientific release creates its exclusive marker before parsing labels."""
    workspace, receipt_path = Path(workspace), Path(receipt_path)
    source = workspace / "prepared"
    if receipt_path.exists() or (source / MARKER).exists():
        raise ValueError("One-shot release already attempted; never retry")
    checked_raw_test(workspace / "official-test.json")
    checked_global_choice(approval)
    if (
        sha(workspace / "selected.pt") != approval["global_choice"]["checkpoint_sha256"]
        or sha(source / "manifest.json") != PREPARED_SHA
        or sha(workspace / "cache/manifest.json") != CACHE_SHA
        or sha(source / "test-cohort.json") != COHORT_SHA
    ):
        raise ValueError(
            "Frozen global model/source/cache/cohort differ before holdout release"
        )
    if sha(source / POLICY) != approval["policy_sha256"]:
        raise ValueError("Root-frozen policy differs before holdout parsing")
    status = "failed"
    try:
        result = release_callback(
            source,
            workspace / "cache",
            workspace / "selected.pt",
            workspace / "official-test.json",
            receipt_path.with_name("video-final-test.json"),
        )
        status = "complete"
        return result
    finally:
        marker = source / MARKER
        if marker.exists():
            shutil.copyfile(marker, receipt_path.with_name(MARKER))
        receipt_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "stage_sha256": approval["stage_sha256"],
                    "policy_sha256": approval["policy_sha256"],
                    "official_test_sha256": OFFICIAL_SHA,
                    "marker_preserved": marker.exists(),
                    "automatic_retry_permitted": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
