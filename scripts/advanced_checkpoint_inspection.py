"""Inspect and restore an externally authenticated newer parent, without updates.

The caller authenticates its committed source package, root contract, original
source/protocol and all thirteen mounted dataset files before importing this
module. This worker repeats the byte guard before and after native reads. A
newly emitted inspection report is evidence, never its own approval to train.
"""

import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def inspect_restore(parent, contract, evidence, policy, guard, *, started, diagnostic):
    """Cloud-only metadata and real state restoration; no model forward call."""
    require(
        sys.platform == "linux"
        and Path(parent).resolve().is_relative_to(Path("/kaggle/input")),
        "Actual Linux input dataset required",
    )
    require(
        contract["schema"] == "advanced_phase2_native_inspection_v1"
        and contract["artifact_revision"] == 2
        and contract["action"] == "inspect_and_restore_only"
        and contract["training"] is False
        and contract["scoring"] is False
        and contract["test_annotations_used"] is False
        and contract["max_seconds"] == 1800,
        "Explicit independent root inspection and restore contract required",
    )
    require(0 <= time.monotonic() - started < 1800, "Original startup budget required")
    guard()
    # All thirteen raw files and scientific source must pass the caller's full
    # externally pinned gate before even importing native software.
    from src.models import cpu_portability as portability
    from src.models import resumable_training as exact

    exact.torch.set_num_threads(2)
    exact.torch.use_deterministic_algorithms(True)
    actual = exact.runtime()
    portability.runtime_guard(actual, "phase2", policy, evidence)
    inspected = evidence["report"]["phases"]["phase2"]
    origin = inspected["origin_identity"]
    spec = exact.production_spec("phase2")
    require(
        exact.fingerprint(origin) == contract["origin_identity_sha256"]
        and spec == inspected["origin_spec"],
        "Independent original identity and specification required",
    )
    parent = Path(parent)
    snapshot = parent / contract["snapshot_name"]
    private = Path(tempfile.mkdtemp(prefix="advanced-parent-restoration-"))
    guard()
    state = exact.checked_resume(
        snapshot, contract["snapshot_sha256"], origin, spec, private
    )
    guard()
    header = json.loads(snapshot.with_suffix(".json").read_bytes())
    active_value = state.get("active")
    actual_progress = {
        "stage": state.get("stage"),
        "batches": state.get("batches"),
        "completed_count": len(state["completed"]),
        "grid": state.get("grid"),
        "seed_index": state.get("seed_index"),
        "policy_index": state.get("policy_index"),
        "active_present": active_value is not None,
        "active_epoch": active_value.get("epoch")
        if isinstance(active_value, dict)
        else None,
        "active_cursor": active_value.get("cursor")
        if isinstance(active_value, dict)
        else None,
        "header_stage": header.get("stage"),
        "header_batches": header.get("batches"),
        "header_completed_seeds": header.get("completed_seeds"),
        "header_progress_matches_state": header.get("stage") == state.get("stage")
        and header.get("batches") == state.get("batches")
        and header.get("completed_seeds") == len(state["completed"]),
    }
    diagnostic(actual_progress)
    require(
        actual_progress == contract["actual_progress"]
        and state["stage"] == "fit"
        and state["batches"] == contract["batches"]
        and len(state["completed"]) == len(contract["completed_names"]) == 10
        and state["active"] is None
        and state["grid"] == 3
        and state["seed_index"] == 1
        and state["policy_index"] == 0
        and spec["seeds"][state["seed_index"]] == contract["next_seed"] == 1337,
        "Exact actual newer parent progress required",
    )
    tensor_count = 0

    def geometry(value, depth=0):
        nonlocal tensor_count
        require(depth <= 12, "Bounded state metadata depth required")
        if isinstance(value, exact.torch.Tensor):
            tensor_count += 1
            require(
                tensor_count <= 8192 and value.ndim <= 8,
                "Bounded tensor metadata required",
            )
            raw = (
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(exact.torch.uint8)
                .numpy()
                .tobytes()
            )
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        if isinstance(value, dict):
            require(len(value) <= 4096, "Bounded mapping required")
            return {str(key): geometry(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            require(len(value) <= 4096, "Bounded sequence required")
            return [geometry(item, depth + 1) for item in value]
        require(
            value is None or isinstance(value, (str, int, float, bool)),
            "Plain state metadata required",
        )
        return value

    def rng_metadata(value):
        return {
            "torch": geometry(value["torch"]),
            "numpy": geometry(value["numpy"]),
            "python_sha256": exact.fingerprint(value["python"]),
        }

    def optimizer_metadata(value):
        return {
            "param_groups": geometry(value["param_groups"]),
            "state": {
                str(key): {"step": float(item["step"]), "geometry": geometry(item)}
                for key, item in value["state"].items()
            },
        }

    def restore_optimizer(model, saved, config, steps):
        optimizer = exact.torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
        optimizer.load_state_dict(saved)
        require(
            optimizer.state_dict()["param_groups"] == groups,
            "Original optimizer groups/order required",
        )
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                item = optimizer.state.get(parameter, {})
                require(
                    set(item) == {"step", "exp_avg", "exp_avg_sq"}
                    and item["step"].numel() == 1
                    and float(item["step"]) == steps
                    and all(
                        item[key].shape == parameter.shape
                        and item[key].dtype == parameter.dtype
                        for key in ("exp_avg", "exp_avg_sq")
                    ),
                    "Original AdamW state/moment/step geometry required",
                )
        require(
            optimizer_metadata(optimizer.state_dict()) == optimizer_metadata(saved),
            "Optimizer restoration must preserve state bytes",
        )
        return optimizer

    completed, restored = [], []
    for index, record in enumerate(state["completed"]):
        name = contract["completed_names"][index]
        require(
            record["name"] == name
            and record["sha256"] == contract["files"][name]["sha256"],
            "Every completed native file must match independently frozen bytes",
        )
        guard()
        require(
            exact.digest(private / name) == record["sha256"],
            "Authenticated private completed-state copy required",
        )
        saved = exact.torch.load(private / name, map_location="cpu", weights_only=True)
        grid, position = divmod(index, len(spec["seeds"]))
        bound = exact.seed_identity(origin, spec, grid, spec["seeds"][position])
        require(
            saved["identity"] == record["identity"] == bound
            and exact.finite_tree(saved),
            "Actual completed seed identity/finite state required",
        )
        report = {
            "name": name,
            "sha256": record["sha256"],
            "identity": bound,
            "best_epoch": saved["best_epoch"],
            "history_count": len(saved["history"]),
            "history_sha256": exact.fingerprint(saved["history"]),
            "model": geometry(saved["state"]),
            "final_model": geometry(saved["final_model"]),
            "optimizer": optimizer_metadata(saved["optimizer"]),
            "generator": geometry(saved["generator"]),
            "rng": rng_metadata(saved["rng"]),
        }
        if index < 6:
            require(
                report == inspected["completed"][index],
                "Original six independent inspected seeds changed",
            )
        model = exact.Adapter()
        model.load_state_dict(saved["state"], strict=True)
        require(
            geometry(model.state_dict()) == report["model"],
            "Best model restoration differs",
        )
        model.load_state_dict(saved["final_model"], strict=True)
        require(
            geometry(model.state_dict()) == report["final_model"],
            "Final model restoration differs",
        )
        steps = saved["history"][-1]["epoch"] * math.ceil(23202 / spec["batch"])
        restore_optimizer(model, saved["optimizer"], spec["configs"][grid], steps)
        generator = exact.torch.Generator()
        generator.set_state(saved["generator"])
        require(
            geometry(generator.get_state()) == report["generator"],
            "Saved generator restoration differs",
        )
        exact.restore_rng(saved["rng"])
        require(
            rng_metadata(exact.rng_state()) == report["rng"],
            "Saved global RNG restoration differs",
        )
        completed.append(report)
        restored.append(
            {
                "name": name,
                "sha256": record["sha256"],
                "identity": bound,
                "restored": True,
                "optimizer_steps": steps,
            }
        )
    total_steps = sum(row["optimizer_steps"] for row in restored)
    require(
        total_steps == state["batches"] == contract["batches"] == 34122,
        "Completed optimizer steps must exhaust actual checkpoint batches",
    )
    expected_rng = rng_metadata(state["rng"])
    require(
        expected_rng == completed[-1]["rng"],
        "Boundary checkpoint global RNG must equal the last completed seed",
    )
    exact.restore_rng(state["rng"])
    require(
        rng_metadata(exact.rng_state()) == expected_rng,
        "Boundary global RNG restoration differs",
    )
    guard()
    require(
        exact.runtime() == actual and time.monotonic() - started < 1800,
        "Stable approved runtime and original service deadline required",
    )
    common = {
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "actual_runtime": actual,
        "origin_identity_sha256": exact.fingerprint(origin),
        "snapshot_sha256": contract["snapshot_sha256"],
        "bitwise_equivalence_claimed": False,
    }
    inspection = {
        "schema": "advanced_phase2_native_inspection_result_v1",
        **common,
        "origin_identity": origin,
        "origin_spec": spec,
        "stage": state["stage"],
        "batches": state["batches"],
        "grid": state["grid"],
        "seed_index": state["seed_index"],
        "policy_index": state["policy_index"],
        "completed_seeds": 10,
        "completed": completed,
        "active": None,
        "next_seed": contract["next_seed"],
        "completed_optimizer_steps": total_steps,
        "rng": expected_rng,
    }
    restoration = {
        "schema": "advanced_phase2_restore_result_v1",
        **common,
        "restored": True,
        "completed": restored,
        "active": None,
        "active_optimizer": None,
        "optimizer_steps": None,
        "generator": None,
        "permutation": None,
        "completed_optimizer_steps": total_steps,
        "next_seed": contract["next_seed"],
        "rng": expected_rng,
        "epoch": None,
        "cursor": None,
        "batches": state["batches"],
    }
    return inspection, restoration
