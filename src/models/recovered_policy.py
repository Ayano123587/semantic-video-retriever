"""Recovered global policy metadata; no model or array imports."""

import hashlib
import json
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_bound_json(path, expected):
    require(
        isinstance(expected, str)
        and re.fullmatch("[0-9a-f]{64}", expected)
        and sha(path) == expected,
        "Actual externally frozen recovered metadata differs",
    )
    return json.loads(Path(path).read_bytes())


def is_recovered(approval):
    return (
        approval.get("schema")
        in (
            "globally_selected_recovered_video_query_policy_v2",
            "root_frozen_cloud_test_approval_v2",
            "cloud_frozen_test_stage_v2",
        )
        or approval.get("phase_provenance_schema") == "recovered_v2"
    )


def execution_runtime(approval):
    return (
        approval["selection_runtime"]
        if is_recovered(approval)
        else approval["training_runtime"]
    )


def query_contract(policy):
    require(
        policy.get("schema") == "globally_selected_recovered_video_query_policy_v2",
        "Explicit recovered query policy required",
    )
    return {
        "phase_provenance_schema": "recovered_v2",
        "global_choice": {
            "report_path": policy["report_path"],
            "report_sha256": policy["report_sha256"],
            "dispatch_path": policy["dispatch_path"],
            "dispatch_sha256": policy["dispatch_sha256"],
            "checkpoint_sha256": policy["checkpoint_sha256"],
        },
        "inventories": policy["inventories"],
        "phase_reports": policy["phase_reports"],
        "source_manifest_sha256": policy["source_manifest_sha256"],
        "cache_manifest_sha256": policy["cache_manifest_sha256"],
        "protocol_sha256": policy["protocol_sha256"],
        "environment": policy["environment"],
        "selection_runtime": policy["selection_runtime"],
    }


def checked_global_choice(approval):
    """Shared future production/test consumer; actual global freeze required."""
    require(
        isinstance(approval.get("environment"), dict)
        and isinstance(approval.get("selection_runtime"), dict),
        "Actual encoding environment and selection runtime dictionaries required",
    )
    global_choice = approval["global_choice"]
    report = read_bound_json(
        global_choice["report_path"], global_choice["report_sha256"]
    )
    dispatch = read_bound_json(
        global_choice["dispatch_path"], global_choice["dispatch_sha256"]
    )
    require(
        dispatch.get("schema") == "both_phase_recovered_global_selection_dispatch_v2"
        and dispatch["result"] == report
        and dispatch["validation_queries"] == 3132
        and dispatch["training"] is False
        and dispatch["scoring"] is True
        and dispatch["test_targets_used"] is False
        and dispatch["holdout_released"] is False,
        "Actual recovered global-selection execution required",
    )
    require(
        report["test_targets_used"] is False
        and report["baseline"]["queries"] == 3132
        and report["selected_validation"]["queries"] == 3132
        and [row["phase"] for row in report["phases"]] == ["phase1", "phase2"]
        and report["selected_checkpoint_sha256"] == global_choice["checkpoint_sha256"],
        "Actual full global policy identity required",
    )
    require(
        report["cache_manifest_sha256"] == approval["cache_manifest_sha256"]
        and report["protocol_sha256"] == approval["protocol_sha256"],
        "Actual global report scientific identity differs",
    )
    require(
        set(approval["inventories"])
        == set(approval["phase_reports"])
        == {"phase1", "phase2"},
        "Both actual recovered inventory/report roles required",
    )
    for phase, count in (("phase1", 9), ("phase2", 12)):
        row = approval["inventories"][phase]
        inventory = read_bound_json(row["path"], row["sha256"])
        phase_meta = approval["phase_reports"][phase]
        phase_report = read_bound_json(phase_meta["path"], phase_meta["sha256"])
        phase_choice = next(item for item in report["phases"] if item["phase"] == phase)
        require(
            inventory["schema"] == "complete_recovered_phase_inventory_v2"
            and inventory["environment"] == approval["environment"]
            and inventory["phase"] == phase
            and inventory["completed_fits"] == count
            and inventory["validation_queries"] == 3132
            and inventory["test_targets_used"] is False
            and inventory["learned_files_deserialized"] is False
            and inventory["files"]["results.json"]
            == phase_meta["sha256"]
            == phase_choice["report_sha256"]
            and dispatch["inventory_sha256"][phase] == row["sha256"],
            "Actual complete recovered phase inventory differs",
        )
        require(
            phase_choice["checkpoint_sha256"]
            == phase_report["selected_checkpoint_sha256"]
            == inventory["files"]["selected.pt"]
            and phase_choice["validation"] == phase_report["selected_validation"],
            "Actual global phase selection evidence differs",
        )
        require(
            phase_report["completed_fits"] == phase_report["required_fits"] == count
            and phase_report["test_targets_used"] is False
            and phase_report["actual_execution_runtime"]
            == inventory["actual_execution_runtime"]
            == dispatch["phase_execution_runtime"][phase]
            and fingerprint(phase_report["origin_identity"])
            == inventory["origin_identity_sha256"]
            == dispatch["phase_origin_identity_sha256"][phase]
            and phase_report["execution_lineage"] == inventory["execution_lineage"],
            "Actual separate origin/execution provenance differs",
        )
        for key in (
            "source_manifest_sha256",
            "cache_manifest_sha256",
            "protocol_sha256",
        ):
            require(
                inventory[key] == approval[key] == dispatch[key],
                "Frozen global scientific provenance differs",
            )
    require(
        dispatch["selection_runtime"] == approval["selection_runtime"],
        "Actual separate selection execution runtime differs",
    )
    return report
