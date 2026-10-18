"""Distinct schema3 provenance for the independently restored ten-seed parent.

The original portability service, lineage and loader remain unchanged. A restore
receipt authenticates history; a separate externally pinned service authorizes
continuation. No model or array module is imported at this metadata boundary.
"""

import copy
import json
import shutil
import tempfile
from pathlib import Path

from src.models import cpu_portability as portability

RESTORE_FREEZE_SHA = "f9580a2e7bd8802182ec7c35aec2cf4ee7d6283c44ad1b1c330b34911e2f1343"
CONTRACT_SHA = "358bf6f334bec6dab320637bddc1164ba69e3962ef65a955a43422d5622268e5"
DATA_FREEZE_SHA = "eeb0a1c5fefb0f6ff236ce53461264a760fe3595bd59259acab2e2b721955f67"


def load_context(paths, original):
    require = portability.require
    require(
        set(paths) == {"freeze", "inspection", "restore", "status", "contract", "data"},
        "Complete externally authenticated advanced evidence required",
    )
    freeze = portability.read_bound_json(paths["freeze"], RESTORE_FREEZE_SHA)
    contract = portability.read_bound_json(paths["contract"], CONTRACT_SHA)
    data = portability.read_bound_json(paths["data"], DATA_FREEZE_SHA)
    artifacts = {
        name: portability.read_bound_json(paths[key], freeze["artifacts_sha256"][name])
        for key, name in (
            ("inspection", "inspection.json"),
            ("restore", "restore-result.json"),
            ("status", "worker-status.json"),
        )
    }
    report, restored, status = (
        artifacts[name]
        for name in ("inspection.json", "restore-result.json", "worker-status.json")
    )
    old = original["report"]["phases"]["phase2"]
    require(
        freeze["schema"] == "root_advanced_native_restore_freeze_v1"
        and freeze["pass"] is True
        and freeze["training_authorized"] is False
        and freeze["kernel_id"] == 139284290
        and freeze["version"] == 1,
        "Actual independently reviewed restore receipt required",
    )
    require(
        report["origin_identity"] == old["origin_identity"]
        and report["origin_spec"] == old["origin_spec"]
        and report["completed"][:6] == old["completed"],
        "Original identity/spec/six completed payloads changed",
    )
    require(
        status["complete"] is True
        and status["restored"] is True
        and status["parent_progress"] == contract["actual_progress"]
        and len(report["completed"]) == len(restored["completed"]) == 10,
        "Complete actual ten-seed boundary proof required",
    )
    for result in (report, restored):
        require(
            result["training"] is False
            and result["scoring"] is False
            and result["test_annotations_used"] is False
            and result["active"] is None
            and result["next_seed"] == 1337
            and result["completed_optimizer_steps"] == result["batches"] == 34122
            and result["actual_runtime"] == status["actual_runtime"],
            "Exact actual restored boundary required",
        )
    for key in (
        "active_optimizer",
        "optimizer_steps",
        "generator",
        "permutation",
        "epoch",
        "cursor",
    ):
        require(restored[key] is None, "Absent active state must remain absent")
    require(
        report["rng"] == report["completed"][-1]["rng"] == restored["rng"],
        "Actual restored global RNG proof differs",
    )
    for index, row in enumerate(report["completed"]):
        name = contract["completed_names"][index]
        require(
            row["name"] == name
            and row["sha256"] == data["files"][name]["sha256"]
            and restored["completed"][index]["name"] == name
            and restored["completed"][index]["identity"] == row["identity"]
            and restored["completed"][index]["sha256"] == row["sha256"]
            and restored["completed"][index]["restored"] is True,
            "Every independently restored native seed must remain bound",
        )
    require(
        sum(row["optimizer_steps"] for row in restored["completed"]) == 34122,
        "Actual completed optimizer total differs",
    )
    return {
        "freeze": freeze,
        "report": report,
        "restore": restored,
        "status": status,
        "contract": contract,
        "data": data,
    }


def authenticate_service(path, pin, policy_sha256, original, context):
    service = portability.read_bound_json(path, pin)
    required = {
        "schema",
        "action",
        "training_authorized",
        "restore_authorized",
        "phase",
        "max_seconds",
        "test_annotations_used",
        "portable_policy_sha256",
        "origin_identity_sha256",
        "origin_inspection_freeze_sha256",
        "advanced_restore_freeze_sha256",
        "source_manifest_sha256",
        "parent_schema",
        "parent_checkpoint_sha256",
        "parent_header_sha256",
        "parent_inventory_sha256",
        "parent_lineage_sha256",
        "parent_snapshot_name",
        "parent_native_paths",
    }
    portability.require(
        set(service) == required
        and service["schema"] == "advanced_parent_training_service_v3"
        and service["action"] == "continue_training"
        and service["training_authorized"] is True
        and service["restore_authorized"] is False
        and service["phase"] == "phase2"
        and service["max_seconds"] == 36000
        and service["test_annotations_used"] is False,
        "Separate actual advanced-parent training authorization required",
    )
    portability.require(
        service["advanced_restore_freeze_sha256"] == RESTORE_FREEZE_SHA
        and service["origin_inspection_freeze_sha256"]
        == portability.INSPECTION_FREEZE_SHA
        and service["portable_policy_sha256"] == policy_sha256
        and service["origin_identity_sha256"]
        == portability.fingerprint(context["report"]["origin_identity"])
        == portability.fingerprint(
            original["report"]["phases"]["phase2"]["origin_identity"]
        ),
        "Independent advanced origin/runtime-policy evidence required",
    )
    for key in (
        "source_manifest_sha256",
        "parent_checkpoint_sha256",
        "parent_header_sha256",
        "parent_inventory_sha256",
    ):
        portability.require(
            portability.valid_sha(service[key]),
            "Externally frozen actual service SHA required",
        )
    portability.require(
        type(service["parent_schema"]) is int
        and service["parent_schema"] in (1, 3)
        and (
            (service["parent_schema"] == 1 and service["parent_lineage_sha256"] is None)
            or (
                service["parent_schema"] == 3
                and portability.valid_sha(service["parent_lineage_sha256"])
            )
        ),
        "Distinct actual original or schema3 parent required",
    )
    mapping = service["parent_native_paths"]
    portability.require(
        isinstance(mapping, dict)
        and service["parent_snapshot_name"] in mapping
        and len(set(mapping.values())) == len(mapping)
        and all(
            isinstance(name, str)
            and name
            and Path(name).name == name
            and "/" not in name
            and "\\" not in name
            and ":" not in name
            for name in (*mapping, *mapping.values())
        ),
        "Exact flat native-role mapping required",
    )
    return service


def authenticate_parent(resume, inventory_path, service, context):
    require = portability.require
    resume = Path(resume)
    parent = portability.cloud_directory(resume.parent)
    inventory = portability.read_bound_json(
        inventory_path, service["parent_inventory_sha256"]
    )
    require(
        set(inventory) == {"schema", "files"}
        and inventory["schema"] == "advanced_checkpoint_parent_inventory_v3",
        "Externally frozen advanced native inventory required",
    )
    files, mapping = inventory["files"], service["parent_native_paths"]
    require(
        set(files) == set(mapping)
        and resume.name == mapping[service["parent_snapshot_name"]],
        "Exact selected parent inventory and mapping required",
    )
    for name, entry in files.items():
        path = parent / mapping[name]
        require(
            set(entry) == {"bytes", "sha256"}
            and type(entry["bytes"]) is int
            and entry["bytes"] > 0
            and portability.valid_sha(entry["sha256"])
            and not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == entry["bytes"]
            and portability.sha(path) == entry["sha256"],
            "All native sibling bytes must authenticate before any deserialization",
        )
    snapshot = service["parent_snapshot_name"]
    header_name = Path(snapshot).with_suffix(".json").name
    require(
        header_name in files
        and files[snapshot]["sha256"] == service["parent_checkpoint_sha256"]
        and files[header_name]["sha256"] == service["parent_header_sha256"],
        "Exact parent snapshot/header required",
    )
    header = json.loads((parent / mapping[header_name]).read_bytes())
    if service["parent_schema"] == 1:
        expected = {
            name: entry
            for name, entry in context["data"]["files"].items()
            if name != "provider-provenance.json"
        }
        require(
            files == expected
            and snapshot == context["contract"]["snapshot_name"]
            and header
            == {
                "snapshot_sha256": context["contract"]["snapshot_sha256"],
                "identity_sha256": context["contract"]["origin_identity_sha256"],
                "completed_seeds": 10,
                "batches": 34122,
                "stage": "fit",
            },
            "Only the actual independently restored checkpoint35 boundary may migrate",
        )
        for name, entry in context["data"]["files"].items():
            path = parent / name
            require(
                not path.is_symlink()
                and path.stat().st_size == entry["bytes"]
                and portability.sha(path) == entry["sha256"],
                "Entire original thirteen-file provider dataset must authenticate",
            )
    else:
        require(
            set(header)
            == {
                "schema",
                "snapshot_sha256",
                "origin_identity_sha256",
                "portable_policy_sha256",
                "lineage_sha256",
                "execution_runtime_sha256",
                "completed_seeds",
                "batches",
                "stage",
            }
            and header["schema"] == 3
            and header["snapshot_sha256"] == service["parent_checkpoint_sha256"]
            and header["origin_identity_sha256"] == service["origin_identity_sha256"]
            and header["portable_policy_sha256"] == service["portable_policy_sha256"]
            and header["lineage_sha256"] == service["parent_lineage_sha256"]
            and type(header["completed_seeds"]) is int
            and 10 <= header["completed_seeds"] <= 12
            and type(header["batches"]) is int
            and header["batches"] >= 34122
            and header["stage"] in ("fit", "candidate", "final"),
            "Actual externally frozen unfinished schema3 parent required",
        )
        spec = context["report"]["origin_spec"]
        names = [
            f"config{grid}-seed{seed}.pt"
            for grid in range(len(spec["configs"]))
            for seed in spec["seeds"]
        ]
        require(
            set(files) == {snapshot, header_name, *names[: header["completed_seeds"]]},
            "Complete exact current schema3 native prefix required",
        )
    return inventory, header


def seed_identity(origin, spec, grid, seed, lineage, policy_sha256):
    return {
        "schema": 3,
        "origin_identity_sha256": portability.fingerprint(origin),
        "portable_policy_sha256": policy_sha256,
        "advanced_restore_freeze_sha256": RESTORE_FREEZE_SHA,
        "config_index": grid,
        "config": spec["configs"][grid],
        "seed": seed,
        "execution_lineage_sha256": portability.fingerprint(lineage),
        "finishing_runtime_sha256": lineage[-1]["actual_runtime_sha256"],
    }


def validate_lineage(events, phase, policy, original, policy_sha256, context):
    require = portability.require
    require(
        phase == "phase2" and isinstance(events, list) and 1 <= len(events) <= 256,
        "Actual bounded advanced lineage required",
    )
    previous, batches = None, 34122
    for event in events:
        require(
            set(event)
            == {
                "schema",
                "origin_identity_sha256",
                "portable_policy_sha256",
                "advanced_restore_freeze_sha256",
                "source_manifest_sha256",
                "parent_checkpoint_sha256",
                "parent_header_sha256",
                "previous_event_sha256",
                "actual_runtime",
                "actual_runtime_sha256",
                "start_progress",
            }
            and event["schema"] == "advanced_cpu_execution_lineage_event_v3"
            and event["origin_identity_sha256"]
            == portability.fingerprint(context["report"]["origin_identity"])
            and event["advanced_restore_freeze_sha256"] == RESTORE_FREEZE_SHA
            and event["portable_policy_sha256"] == policy_sha256
            and event["previous_event_sha256"] == previous,
            "Actual ordered advanced lineage evidence required",
        )
        portability.runtime_guard(event["actual_runtime"], phase, policy, original)
        require(
            event["actual_runtime_sha256"]
            == portability.fingerprint(event["actual_runtime"])
            and all(
                portability.valid_sha(event[key])
                for key in (
                    "source_manifest_sha256",
                    "parent_checkpoint_sha256",
                    "parent_header_sha256",
                )
            ),
            "Actual source/native/runtime lineage hashes required",
        )
        progress = event["start_progress"]
        require(
            set(progress)
            == {"batches", "grid", "seed_index", "stage", "epoch", "cursor"}
            and all(
                type(progress[key]) is int and progress[key] >= 0
                for key in ("batches", "grid", "seed_index")
            )
            and progress["batches"] >= batches
            and progress["grid"] <= 4
            and progress["seed_index"] <= 3
            and progress["stage"] in ("fit", "candidate", "final")
            and (
                (progress["epoch"] is None and progress["cursor"] is None)
                or (
                    type(progress["epoch"]) is int
                    and 1 <= progress["epoch"] <= 50
                    and type(progress["cursor"]) is int
                    and 0 <= progress["cursor"] <= 23202
                )
            ),
            "Actual non-regressing advanced cursor required",
        )
        if previous is None:
            require(
                event["parent_checkpoint_sha256"]
                == context["contract"]["snapshot_sha256"]
                and event["parent_header_sha256"]
                == context["contract"]["header_sha256"]
                and progress
                == {
                    "batches": 34122,
                    "grid": 3,
                    "seed_index": 1,
                    "stage": "fit",
                    "epoch": None,
                    "cursor": None,
                },
                "Advanced lineage must begin at the actual restored ten-seed boundary",
            )
        previous, batches = portability.fingerprint(event), progress["batches"]
    return previous


def append_execution(
    events, service, actual_runtime, progress, phase, policy, original, context
):
    result = copy.deepcopy(events)
    previous = (
        validate_lineage(
            result, phase, policy, original, service["portable_policy_sha256"], context
        )
        if result
        else None
    )
    result.append(
        {
            "schema": "advanced_cpu_execution_lineage_event_v3",
            "origin_identity_sha256": service["origin_identity_sha256"],
            "portable_policy_sha256": service["portable_policy_sha256"],
            "advanced_restore_freeze_sha256": RESTORE_FREEZE_SHA,
            "source_manifest_sha256": service["source_manifest_sha256"],
            "parent_checkpoint_sha256": service["parent_checkpoint_sha256"],
            "parent_header_sha256": service["parent_header_sha256"],
            "previous_event_sha256": previous,
            "actual_runtime": copy.deepcopy(actual_runtime),
            "actual_runtime_sha256": portability.fingerprint(actual_runtime),
            "start_progress": copy.deepcopy(progress),
        }
    )
    validate_lineage(
        result, phase, policy, original, service["portable_policy_sha256"], context
    )
    return result


def checked_resume(
    resume,
    output,
    origin,
    spec,
    policy,
    original,
    service,
    inventory_path,
    guard,
    exact,
    context,
):
    from src.models.portable_training import progress, validate_cursor

    guard()
    inventory, header = authenticate_parent(resume, inventory_path, service, context)
    mounted, logical = (
        Path(resume).parent,
        Path(tempfile.mkdtemp(prefix="advanced-parent-native-")),
    )
    for name, entry in inventory["files"].items():
        destination = logical / name
        shutil.copyfile(mounted / service["parent_native_paths"][name], destination)
        portability.require(
            destination.stat().st_size == entry["bytes"]
            and portability.sha(destination) == entry["sha256"],
            "Private native copy must preserve authenticated bytes",
        )
    guard()
    snapshot = logical / service["parent_snapshot_name"]
    if service["parent_schema"] == 1:
        state = exact.checked_resume(
            snapshot, service["parent_checkpoint_sha256"], origin, spec, output
        )
        portability.require(
            progress(state)
            == {
                "batches": 34122,
                "grid": 3,
                "seed_index": 1,
                "stage": "fit",
                "epoch": None,
                "cursor": None,
            }
            and state["active"] is None,
            "Only the independently restored boundary may initialize schema3",
        )
        state["origin_identity"] = state.pop("identity")
        state["schema"] = 3
        state["advanced"] = {
            "policy_sha256": service["portable_policy_sha256"],
            "restore_freeze_sha256": RESTORE_FREEZE_SHA,
            "origin_inspection_freeze_sha256": portability.INSPECTION_FREEZE_SHA,
            "origin_checkpoint_sha256": context["contract"]["snapshot_sha256"],
            "execution_lineage": [],
        }
    else:
        state = exact.torch.load(snapshot, map_location="cpu", weights_only=True)
        portability.require(
            state.get("schema") == 3
            and "identity" not in state
            and "portable" not in state
            and state["origin_identity"] == origin
            and state["spec"] == spec
            and exact.finite_tree(state),
            "Exact truthful schema3 origin/spec/native state required",
        )
        meta = state["advanced"]
        portability.require(
            set(meta)
            == {
                "policy_sha256",
                "restore_freeze_sha256",
                "origin_inspection_freeze_sha256",
                "origin_checkpoint_sha256",
                "execution_lineage",
            }
            and meta["policy_sha256"] == service["portable_policy_sha256"]
            and meta["restore_freeze_sha256"] == RESTORE_FREEZE_SHA
            and meta["origin_inspection_freeze_sha256"]
            == portability.INSPECTION_FREEZE_SHA
            and meta["origin_checkpoint_sha256"]
            == context["contract"]["snapshot_sha256"],
            "Advanced native provenance differs",
        )
        events = meta["execution_lineage"]
        validate_lineage(
            events,
            "phase2",
            policy,
            original,
            service["portable_policy_sha256"],
            context,
        )
        portability.require(
            portability.fingerprint(events)
            == service["parent_lineage_sha256"]
            == header["lineage_sha256"]
            and header["execution_runtime_sha256"]
            == events[-1]["actual_runtime_sha256"]
            and state["batches"] == header["batches"]
            and state["stage"] == header["stage"]
            and len(state["completed"]) == header["completed_seeds"],
            "Frozen actual schema3 lineage/progress differs",
        )
    validate_cursor(state, spec)
    for index, record in enumerate(state["completed"]):
        grid, position = divmod(index, len(spec["seeds"]))
        name = f"config{grid}-seed{spec['seeds'][position]}.pt"
        portability.require(
            record["name"] == name and exact.digest(logical / name) == record["sha256"],
            "Complete unchanged native prefix required",
        )
        if index < 10:
            observed = context["report"]["completed"][index]
            portability.require(
                record["identity"] == observed["identity"]
                and record["sha256"] == observed["sha256"],
                "Original ten actual seeds must never be rewritten",
            )
        guard()
        saved = exact.torch.load(logical / name, map_location="cpu", weights_only=True)
        portability.require(
            saved["identity"] == record["identity"] and exact.finite_tree(saved),
            "Actual completed native payload required",
        )
        if index >= 10:
            events = saved["advanced_lineage"]
            portability.require(
                saved["schema"] == 3
                and events == state["advanced"]["execution_lineage"][: len(events)],
                "Actual seed execution lineage prefix required",
            )
            validate_lineage(
                events,
                "phase2",
                policy,
                original,
                service["portable_policy_sha256"],
                context,
            )
            portability.require(
                record["identity"]
                == seed_identity(
                    origin,
                    spec,
                    grid,
                    spec["seeds"][position],
                    events,
                    service["portable_policy_sha256"],
                ),
                "Truthful new seed identity required",
            )
        if service["parent_schema"] == 3:
            shutil.copyfile(logical / name, Path(output) / name)
            portability.require(
                exact.digest(Path(output) / name) == record["sha256"],
                "Output prefix native copy changed",
            )
    guard()
    return state
