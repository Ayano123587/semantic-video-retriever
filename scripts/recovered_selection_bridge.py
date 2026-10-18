"""Truthful recovered-phase metadata and pre-load global-selection boundary.

No old dispatch or identity is synthesized. The original chooser remains the
only scientific selection implementation. This module never imports native
array/model packages or deserializes learned files.
"""

import math
from pathlib import Path

from scripts import research_selection_bridge as original_bridge
from src.models import advanced_parent as advanced
from src.models import cpu_portability as portability

SEEDS = (42, 1337, 2026)
GRIDS = {"phase1": 3, "phase2": 4}


def phase_metadata(
    directory,
    wrapper_path,
    proof_path,
    service_path,
    contract,
    scientific,
    policy,
    evidence,
    advanced_context=None,
):
    """Authenticate actual small metadata; no native files are opened here."""
    require = portability.require
    phase = contract["phase"]
    require(phase in GRIDS, "Explicit completed phase required")
    schema = 2 if phase == "phase1" else 3
    directory = Path(directory)
    report = original_bridge.read_bound(
        directory / "results.json", contract["report_sha256"]
    )
    wrapper = original_bridge.read_bound(wrapper_path, contract["wrapper_sha256"])
    proof = original_bridge.read_bound(proof_path, contract["execution_proof_sha256"])
    header_name = contract["final_header"]
    require(
        Path(header_name).name == header_name
        and header_name.startswith("checkpoint-")
        and header_name.endswith(".json"),
        "Safe actual final header required",
    )
    header = original_bridge.read_bound(
        directory / header_name, contract["final_header_sha256"]
    )
    require(
        proof["id"] == contract["kernel_id"]
        and proof["canonical"] == contract["kernel"]
        and proof["version"] == contract["version"]
        and proof["source_sha256"] == contract["transport_sha256"]
        and proof["datasets"] == contract["datasets"]
        and proof["encoding_parent"] == contract["encoding_parent"]
        and proof["private"] is True
        and proof["cpu"] is True
        and proof["internet"] is True,
        "Actual accepted recovered execution required",
    )
    require(
        wrapper["stage"] == "service_returned"
        and wrapper["scientific_complete"] is True
        and wrapper["test_annotations_used"] is False
        and wrapper["result"] == report,
        "Actual complete recovered wrapper required",
    )
    entry_flag = (
        "native_loading_started"
        if phase == "phase1"
        else "native_training_entry_started"
    )
    require(wrapper[entry_flag] is True, "Actual native execution entry required")
    fits = GRIDS[phase] * len(SEEDS)
    require(
        type(report["completed_fits"]) is int
        and report["completed_fits"] == report["required_fits"] == fits
        and report["test_targets_used"] is False,
        "Full fixed fit grid and no-test proof required",
    )
    origin = report["origin_identity"]
    require(
        origin == evidence["report"]["phases"][phase]["origin_identity"],
        "Immutable inspected origin differs",
    )
    for key in ("cache_manifest_sha256", "protocol_sha256"):
        require(
            report[key] == origin[key] == scientific[key],
            "Original cache/protocol differs",
        )
    for key in ("implementation_sha256", "coverage", "encoding_identity"):
        require(origin[key] == scientific[key], "Original scientific identity differs")
    require(
        report["baseline"] == scientific["baseline"]
        and origin["runtime"] == scientific["training_runtime"],
        "Original full baseline/runtime differs",
    )
    policy_sha = contract["policy_sha256"]
    if phase == "phase1":
        service = portability.authenticate_service(
            service_path, contract["service_sha256"], policy_sha, phase, evidence
        )
        portability.validate_lineage(
            report["execution_lineage"], phase, policy, evidence, policy_sha
        )
    else:
        require(
            advanced_context is not None,
            "Actual independently restored advanced parent required",
        )
        service = advanced.authenticate_service(
            service_path,
            contract["service_sha256"],
            policy_sha,
            evidence,
            advanced_context,
        )
        advanced.validate_lineage(
            report["execution_lineage"],
            phase,
            policy,
            evidence,
            policy_sha,
            advanced_context,
        )
    last = report["execution_lineage"][-1]
    runtime = report["actual_execution_runtime"]
    portability.runtime_guard(runtime, phase, policy, evidence)
    require(
        runtime == last["actual_runtime"] == contract["execution_runtime"],
        "Actual execution runtime differs",
    )
    require(
        last["source_manifest_sha256"] == service["source_manifest_sha256"]
        and last["parent_checkpoint_sha256"] == service["parent_checkpoint_sha256"]
        and last["parent_header_sha256"] == service["parent_header_sha256"],
        "Actual service/lineage differs",
    )
    identity = report["continuation_identity"]
    expected_identity = {
        "schema": schema,
        "origin_identity_sha256": portability.fingerprint(origin),
        "portable_policy_sha256": policy_sha,
        "lineage_sha256": portability.fingerprint(report["execution_lineage"]),
        "actual_runtime_sha256": portability.fingerprint(runtime),
    }
    require(
        identity == expected_identity,
        "Truthful recovered continuation identity required",
    )
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
        and header["schema"] == schema
        and header["stage"] == "done"
        and header["completed_seeds"] == fits
        and header["origin_identity_sha256"]
        == expected_identity["origin_identity_sha256"]
        and header["portable_policy_sha256"] == policy_sha
        and header["lineage_sha256"] == expected_identity["lineage_sha256"]
        and header["execution_runtime_sha256"]
        == expected_identity["actual_runtime_sha256"],
        "Actual recovered final done header required",
    )
    runs = report["runs"]
    require(len(runs) == GRIDS[phase], "Full configuration grid required")
    spec = evidence["report"]["phases"][phase]["origin_spec"]
    require(
        [run["config"] for run in runs] == spec["configs"]
        if phase == "phase1"
        else [run["lambda"] for run in runs] == [0, 0.5, 1, 2],
        "Original ordered configurations required",
    )
    known = (
        evidence["report"]["phases"][phase]["completed"]
        if phase == "phase1"
        else advanced_context["report"]["completed"]
    )
    known = {row["name"]: row for row in known}
    epochs = 0
    for grid, run in enumerate(runs):
        require(
            [row["seed"] for row in run["seeds"]] == list(SEEDS),
            "All ordered three seeds required",
        )
        for row in run["seeds"]:
            history = row["history"]
            require(
                isinstance(history, list)
                and 5 <= len(history) <= 50
                and [x["epoch"] for x in history] == list(range(1, len(history) + 1)),
                "Complete actual epoch history required",
            )
            values = [x["validation_corpus_consensus_r1_iou50"] for x in history]
            require(
                all(
                    math.isfinite(x["train_loss"])
                    and x["train_loss"] >= 0
                    and math.isfinite(v)
                    and 0 <= v <= 1
                    for x, v in zip(history, values, strict=True)
                ),
                "Finite actual training history required",
            )
            best = values.index(max(values)) + 1
            require(
                row["best_epoch"] == best
                and (len(history) == 50 or len(history) - best >= 8),
                "Original earliest-best/stop rule required",
            )
            name = f"config{grid}-seed{row['seed']}.pt"
            if name in known:
                require(
                    portability.fingerprint(history) == known[name]["history_sha256"]
                    and best == known[name]["best_epoch"],
                    "Inspected completed seed history changed",
                )
            for count in range(1, len(history)):
                early = values[:count].index(max(values[:count])) + 1
                require(
                    count < 5 or count - early < 8,
                    "Training continued after original early stop",
                )
            epochs += len(history)
    require(
        type(header["batches"]) is int
        and header["batches"] == epochs * math.ceil(23202 / 64),
        "Complete actual optimizer batch count required",
    )
    metrics = [report["baseline"], report["selected_validation"]]
    if phase == "phase2":
        require(
            [(x["lambda"], x["adapted_score_weight"]) for x in report["policies"]]
            == [(w, a) for w in (0, 0.5, 1, 2) for a in (0.25, 0.5, 0.75, 1.0)],
            "All sixteen ordered policies required",
        )
        candidates = report["policies"]
    else:
        candidates = runs
    chosen_name = "frozen_clip"
    chosen_metrics = report["baseline"]
    for row in candidates:
        measured = row["validation"]
        eligible = (
            measured["corpus_consensus_r1_iou50"]
            - report["baseline"]["corpus_consensus_r1_iou50"]
            >= 0.01 - 1e-12
            and measured["within_video_official_miou"]
            >= report["baseline"]["within_video_official_miou"] - 0.01 - 1e-12
        )
        require(
            row["promotion_eligible"] is eligible, "Original promotion policy differs"
        )
        if (
            eligible
            and measured["corpus_consensus_r1_iou50"]
            > chosen_metrics["corpus_consensus_r1_iou50"]
        ):
            chosen_metrics = measured
            chosen_name = (
                row["config"]["name"]
                if phase == "phase1"
                else f"lambda{row['lambda']}-alpha{row['adapted_score_weight']}"
            )
        metrics.append(measured)
    require(
        report["selected"] == chosen_name
        and report["selected_validation"] == chosen_metrics,
        "Original eligible winner and stable tie order required",
    )
    require(
        all(x["queries"] == 3132 and x["candidate_moments"] == 15927 for x in metrics),
        "Full available validation geometry required",
    )
    replay = report["checkpoint_replay_max_abs_error"]
    require(
        type(replay) in (int, float) and math.isfinite(replay) and 0 <= replay <= 1e-6,
        "Saved selected-policy replay required",
    )
    return report, header


def phase_attestation(
    directory,
    wrapper_path,
    proof_path,
    service_path,
    contract,
    scientific,
    policy,
    evidence,
    advanced_context=None,
):
    """Cloud hash-only inventory; no learned deserialization or chooser call."""
    report, header = phase_metadata(
        directory,
        wrapper_path,
        proof_path,
        service_path,
        contract,
        scientific,
        policy,
        evidence,
        advanced_context,
    )
    directory = portability.cloud_directory(directory)
    phase = contract["phase"]
    header_name = contract["final_header"]
    pins = {
        "results.json": contract["report_sha256"],
        header_name: contract["final_header_sha256"],
        Path(header_name).with_suffix(".pt").name: header["snapshot_sha256"],
        "selected.pt": report["selected_checkpoint_sha256"],
    }
    names = [
        f"config{grid}-seed{seed}.pt" for grid in range(GRIDS[phase]) for seed in SEEDS
    ]
    known = (
        evidence["report"]["phases"][phase]["completed"]
        if phase == "phase1"
        else advanced_context["report"]["completed"]
    )
    known = {row["name"]: row["sha256"] for row in known}
    portability.require(
        sorted(p.name for p in directory.glob("config*-seed*.pt")) == sorted(names),
        "Exact whole learned seed inventory required",
    )
    for name in [*pins, *names]:
        path = directory / name
        portability.require(
            path.is_file() and not path.is_symlink(),
            "Actual regular learned files required",
        )
        actual = original_bridge.sha(path)
        if name in known:
            portability.require(
                actual == known[name], "Inspected completed native seed changed"
            )
        if name in pins:
            portability.require(
                actual == pins[name], "Frozen final/selected native bytes differ"
            )
        else:
            pins[name] = actual
    return {
        "schema": "complete_recovered_phase_inventory_v2",
        "phase": phase,
        "kernel": contract["kernel"],
        "kernel_id": contract["kernel_id"],
        "version": contract["version"],
        "wrapper_sha256": contract["wrapper_sha256"],
        "transport_sha256": contract["transport_sha256"],
        "execution_proof_sha256": contract["execution_proof_sha256"],
        "files": pins,
        "completed_fits": report["completed_fits"],
        "validation_queries": 3132,
        "source_manifest_sha256": scientific["source_manifest_sha256"],
        "cache_manifest_sha256": scientific["cache_manifest_sha256"],
        "protocol_sha256": scientific["protocol_sha256"],
        "environment": scientific["environment"],
        "origin_identity_sha256": portability.fingerprint(report["origin_identity"]),
        "actual_execution_runtime": report["actual_execution_runtime"],
        "execution_lineage": report["execution_lineage"],
        "test_targets_used": False,
        "learned_files_deserialized": False,
    }


def authenticate_both(
    inputs, scientific, policy, evidence, advanced_context, frozen=None
):
    portability.require(
        set(inputs) == {"phase1", "phase2"},
        "BOTH actual complete recovered phases required",
    )
    if frozen is not None:
        portability.require(
            set(frozen) == set(inputs), "BOTH actual root-frozen inventories required"
        )
    inventories = {}
    for phase, row in inputs.items():
        portability.require(row["contract"]["phase"] == phase, "Phase role differs")
        actual = phase_attestation(
            **row,
            scientific=scientific,
            policy=policy,
            evidence=evidence,
            advanced_context=advanced_context,
        )
        if frozen is not None:
            portability.require(
                actual
                == original_bridge.read_bound(
                    frozen[phase]["path"], frozen[phase]["sha256"]
                ),
                "Root-frozen actual inventory changed",
            )
        inventories[phase] = actual
    return inventories


def guarded_choose(
    inputs, scientific, policy, evidence, advanced_context, frozen, choose_callback
):
    portability.require(
        frozen is not None and set(frozen) == {"phase1", "phase2"},
        "BOTH actual root-frozen inventories required before chooser",
    )
    authenticate_both(inputs, scientific, policy, evidence, advanced_context, frozen)
    result = choose_callback()
    authenticate_both(inputs, scientific, policy, evidence, advanced_context, frozen)
    return result
