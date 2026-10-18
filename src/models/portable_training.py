"""Versioned CPU-portable provenance boundary around unchanged training math.

The original exact trainer remains untouched. Native handling is cloud-only;
policy, service, source and sibling bytes are authenticated before math imports.
"""

import ast
import copy
import importlib.metadata
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

from src.models import cpu_portability as portability


def source_closure(root, starting):
    root = Path(root)
    pending = list(starting)
    found = set()

    def add_module(module):
        if not module.startswith("src"):
            return
        path = root.joinpath(*module.split("."))
        candidate = (
            path.with_suffix(".py")
            if path.with_suffix(".py").is_file()
            else path / "__init__.py"
        )
        if candidate.is_file():
            pending.append(candidate.relative_to(root).as_posix())

    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        path = root / name
        for ancestor in path.parents:
            if ancestor == root:
                break
            init = ancestor / "__init__.py"
            if init.is_file():
                pending.append(init.relative_to(root).as_posix())
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src")
            ):
                add_module(node.module)
                for alias in node.names:
                    add_module(node.module + "." + alias.name)
    return found


def authenticate_source(path, service, origin):
    manifest = portability.read_bound_json(path, service["source_manifest_sha256"])
    portability.require(
        manifest.get("schema") == "portable_training_source_manifest_v2",
        "Portable source manifest required",
    )
    root = Path(__file__).resolve().parents[2]
    scientific = {
        "src/" + name: pin for name, pin in origin["implementation_sha256"].items()
    }
    expected = source_closure(
        root,
        [
            *scientific,
            "src/models/cpu_portability.py",
            "src/models/portable_training.py",
        ],
    )
    portability.require(
        set(manifest["files"]) == expected,
        "Complete exact transitive source closure required",
    )
    for name, pin in manifest["files"].items():
        relative = Path(name)
        portability.require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in name,
            "Safe source path required",
        )
        target = root / name
        portability.require(
            not target.is_symlink() and portability.sha(target) == pin,
            "Frozen source bytes differ",
        )
        if name in scientific:
            portability.require(
                pin == scientific[name], "Original scientific implementation changed"
            )
    return manifest


def seed_identity_v2(origin, spec, grid, seed, lineage, policy_sha256):
    return {
        "schema": 2,
        "origin_identity_sha256": portability.fingerprint(origin),
        "portable_policy_sha256": policy_sha256,
        "config_index": grid,
        "config": spec["configs"][grid],
        "seed": seed,
        "execution_lineage_sha256": portability.fingerprint(lineage),
        "finishing_runtime_sha256": lineage[-1]["actual_runtime_sha256"],
    }


def progress(state):
    active = state["active"]
    return {
        "batches": state["batches"],
        "grid": state["grid"],
        "seed_index": state["seed_index"],
        "stage": state["stage"],
        "epoch": active["epoch"] if active else None,
        "cursor": active["cursor"] if active else None,
    }


def validate_cursor(state, spec):
    require = portability.require
    require(
        state["stage"] in ("fit", "candidate", "final"), "Invalid continuation stage"
    )
    require(
        type(state["grid"]) is int and 0 <= state["grid"] <= len(spec["configs"]),
        "Invalid grid cursor",
    )
    require(
        type(state["seed_index"]) is int
        and 0 <= state["seed_index"] <= len(spec["seeds"]),
        "Invalid seed cursor",
    )
    if state["stage"] == "fit":
        require(
            state["grid"] < len(spec["configs"])
            and state["seed_index"] < len(spec["seeds"]),
            "Invalid active fit cursor",
        )
    if state["stage"] == "candidate":
        require(
            state["grid"] < len(spec["configs"])
            and state["seed_index"] == len(spec["seeds"])
            and state["active"] is None,
            "Invalid candidate cursor",
        )
    if state["stage"] == "final":
        require(
            state["grid"] == len(spec["configs"])
            and state["seed_index"] == 0
            and state["active"] is None,
            "Invalid final cursor",
        )
    require(
        len(state["runs"]) == state["grid"]
        and len(state["completed"])
        == state["grid"] * len(spec["seeds"]) + state["seed_index"],
        "Incomplete candidate/seed coverage",
    )
    if spec["phase"] == "phase2":
        require(
            len(state["policies"])
            == state["grid"] * len(spec["alphas"]) + state["policy_index"],
            "Incomplete policy coverage",
        )


def checked_portable_resume(
    resume,
    output,
    origin,
    spec,
    policy,
    evidence,
    service,
    inventory_path,
    guard,
    exact,
):
    guard()
    inventory, header = portability.authenticate_parent(
        resume, inventory_path, service, evidence, spec["phase"]
    )
    # Kaggle's checkpoint dataset uses flat transport names. Reconstruct only
    # authenticated logical siblings in a fresh cloud-private directory; never
    # rewrite native objects or their historical JSON headers.
    mounted = Path(resume).parent
    logical = Path(tempfile.mkdtemp(prefix="portable-parent-"))
    for name, entry in inventory["files"].items():
        destination = logical / name
        shutil.copyfile(mounted / service["parent_native_paths"][name], destination)
        portability.require(
            destination.stat().st_size == entry["bytes"]
            and portability.sha(destination) == entry["sha256"],
            "Logical checkpoint copy differs from frozen mounted bytes",
        )
    guard()
    resume = logical / service["parent_snapshot_name"]
    if service["parent_schema"] == 1:
        # Deliberately validate the real historical identity, never pretend it is
        # the current runtime. The separately authorized v2 policy governs that.
        state = exact.checked_resume(
            resume, service["parent_checkpoint_sha256"], origin, spec, output
        )
        state["origin_identity"] = state.pop("identity")
        state["schema"] = 2
        state["portable"] = {
            "policy_sha256": service["portable_policy_sha256"],
            "inspection_freeze_sha256": portability.INSPECTION_FREEZE_SHA,
            "origin_checkpoint_sha256": service["parent_checkpoint_sha256"],
            "execution_lineage": [],
        }
    else:
        state = exact.torch.load(resume, map_location="cpu", weights_only=True)
        require = portability.require
        require(
            state.get("schema") == 2
            and "identity" not in state
            and state["origin_identity"] == origin
            and state["spec"] == spec
            and exact.finite_tree(state),
            "Portable origin/spec/state differs",
        )
        meta = state["portable"]
        require(
            set(meta)
            == {
                "policy_sha256",
                "inspection_freeze_sha256",
                "origin_checkpoint_sha256",
                "execution_lineage",
            }
            and meta["policy_sha256"] == service["portable_policy_sha256"]
            and meta["inspection_freeze_sha256"] == portability.INSPECTION_FREEZE_SHA
            and meta["origin_checkpoint_sha256"]
            == evidence["report"]["phases"][spec["phase"]]["reference"][
                "snapshot_sha256"
            ],
            "Portable provenance differs",
        )
        lineage = meta["execution_lineage"]
        portability.validate_lineage(
            lineage, spec["phase"], policy, evidence, service["portable_policy_sha256"]
        )
        require(
            portability.fingerprint(lineage) == service["parent_lineage_sha256"]
            and header["execution_runtime_sha256"]
            == lineage[-1]["actual_runtime_sha256"],
            "Frozen parent lineage/runtime differs",
        )
        require(
            state["batches"] == header["batches"]
            and state["stage"] == header["stage"]
            and len(state["completed"]) == header["completed_seeds"],
            "Portable header progress differs",
        )
        validate_cursor(state, spec)
        original = evidence["report"]["phases"][spec["phase"]]["completed"]
        for index, record in enumerate(state["completed"]):
            grid, position = divmod(index, len(spec["seeds"]))
            seed = spec["seeds"][position]
            name = f"config{grid}-seed{seed}.pt"
            require(
                record["name"] == name
                and exact.digest(Path(resume).parent / name) == record["sha256"],
                "Completed seed prefix/hash mismatch",
            )
            if index < len(original):
                require(
                    record["identity"] == original[index]["identity"]
                    and record["sha256"] == original[index]["sha256"],
                    "Immutable original completed seed changed",
                )
            guard()
            saved = exact.torch.load(
                Path(resume).parent / name, map_location="cpu", weights_only=True
            )
            require(
                saved["identity"] == record["identity"] and exact.finite_tree(saved),
                "Completed payload identity/nonfinite mismatch",
            )
            if index >= len(original):
                events = saved["portable_lineage"]
                require(
                    saved["schema"] == 2 and events == lineage[: len(events)],
                    "Seed execution lineage is not an actual prefix",
                )
                portability.validate_lineage(
                    events,
                    spec["phase"],
                    policy,
                    evidence,
                    service["portable_policy_sha256"],
                )
                require(
                    record["identity"]
                    == seed_identity_v2(
                        origin,
                        spec,
                        grid,
                        seed,
                        events,
                        service["portable_policy_sha256"],
                    ),
                    "Portable seed execution identity differs",
                )
            shutil.copyfile(Path(resume).parent / name, Path(output) / name)
    validate_cursor(state, spec)
    return state


def candidate_bytes(records, output, exact):
    for record in records:
        portability.require(
            exact.digest(Path(output) / record["name"]) == record["sha256"],
            "Candidate artifact changed before deserialization",
        )


def train(
    preparation,
    cache,
    output,
    protocol,
    phase,
    *,
    policy_path,
    policy_sha256,
    evidence_paths,
    service_contract_path,
    service_contract_sha256,
    source_manifest_path,
    parent_inventory_path,
    resume,
    metadata="data/external/didemo",
    service_started=None,
):
    started = time.monotonic() if service_started is None else service_started
    portability.require(
        isinstance(started, (float, int))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= started <= time.monotonic(),
        "Actual monotonic service start required",
    )
    evidence = portability.load_evidence(
        evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
    )
    policy = portability.authenticate_policy(policy_path, policy_sha256, evidence)
    service = portability.authenticate_service(
        service_contract_path, service_contract_sha256, policy_sha256, phase, evidence
    )
    origin = evidence["report"]["phases"][phase]["origin_identity"]

    def guard():
        authenticate_source(source_manifest_path, service, origin)
        portability.authenticate_parent(
            resume, parent_inventory_path, service, evidence, phase
        )
        portability.require(
            portability.sha(protocol) == policy["origin_protocol_sha256"]
            and portability.sha(Path(cache) / "manifest.json")
            == policy["cache_manifest_sha256"],
            "Original cache/protocol metadata changed",
        )
        cached = json.loads((Path(cache) / "manifest.json").read_bytes())
        for name, pin in cached["files"].items():
            portability.require(
                Path(name).name == name and portability.sha(Path(cache) / name) == pin,
                "Original native cache bytes changed",
            )

    guard()  # Includes Linux/input and every native sibling BEFORE Torch import.
    from src.models import resumable_training as exact

    exact.torch.set_num_threads(2)
    exact.torch.use_deterministic_algorithms(True)
    actual = exact.runtime()
    portability.runtime_guard(actual, phase, policy, evidence)
    encoded = origin["encoding_identity"]
    for distribution, key in (
        ("transformers", "transformers"),
        ("pillow", "pillow"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ):
        portability.require(
            importlib.metadata.version(distribution) == encoded[key],
            "Original encoding software differs",
        )
    import imageio_ffmpeg

    portability.require(
        imageio_ffmpeg.get_ffmpeg_version() == encoded["ffmpeg"],
        "Original FFmpeg build differs",
    )
    from src.models.research_followup import checked_protocol

    data, manifest, frozen = checked_protocol(
        preparation, cache, protocol, policy["origin_protocol_sha256"], metadata
    )
    portability.require(
        len(data["train"]["x"]) == 23202 and len(data["val"]["x"]) == 3132,
        "Full unchanged training/validation query cohorts required",
    )
    portability.require(
        exact.production_spec(phase)
        == evidence["report"]["phases"][phase]["origin_spec"],
        "Original production grid changed",
    )
    portability.require(
        time.monotonic() - started < 36000,
        "Startup-inclusive service deadline exhausted before engine",
    )
    result = engine(
        data,
        manifest,
        frozen["baseline"],
        output,
        origin,
        exact.production_spec(phase),
        resume=resume,
        resume_sha256=service["parent_checkpoint_sha256"],
        max_seconds=36000,
        service_started=started,
        policy=policy,
        evidence=evidence,
        service=service,
        inventory_path=parent_inventory_path,
        guard=guard,
        execution_runtime=actual,
    )
    guard()
    portability.require(
        exact.runtime() == actual, "Execution runtime changed during service"
    )
    return result


# The engine below is an explicit development-time fork of the frozen original
# function. Source parity tests protect the training, restore and scoring blocks.


def restore_only(
    cache,
    protocol,
    phase,
    *,
    policy_path,
    policy_sha256,
    evidence_paths,
    service_contract_path,
    service_contract_sha256,
    source_manifest_path,
    parent_inventory_path,
    resume,
    service_started=None,
):
    """Authenticate and restore real state without a forward call or update."""
    started = time.monotonic() if service_started is None else service_started
    portability.require(
        isinstance(started, (float, int))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= started <= time.monotonic(),
        "Actual monotonic service start required",
    )
    evidence = portability.load_evidence(
        evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
    )
    policy = portability.authenticate_policy(policy_path, policy_sha256, evidence)
    service = portability.authenticate_restore_service(
        service_contract_path, service_contract_sha256, policy_sha256, phase, evidence
    )
    origin = evidence["report"]["phases"][phase]["origin_identity"]
    inspected = evidence["report"]["phases"][phase]
    portability.require(
        service["parent_schema"] == 1,
        "Restore smoke requires actual inspected original parent",
    )

    def guard():
        authenticate_source(source_manifest_path, service, origin)
        portability.authenticate_parent(
            resume, parent_inventory_path, service, evidence, phase
        )
        portability.require(
            portability.sha(protocol) == policy["origin_protocol_sha256"]
            and portability.sha(Path(cache) / "manifest.json")
            == policy["cache_manifest_sha256"],
            "Original cache/protocol metadata changed",
        )

    guard()
    from src.models import resumable_training as exact

    exact.torch.set_num_threads(2)
    exact.torch.use_deterministic_algorithms(True)
    actual = exact.runtime()
    portability.runtime_guard(actual, phase, policy, evidence)
    spec = exact.production_spec(phase)
    for distribution, key in (
        ("transformers", "transformers"),
        ("pillow", "pillow"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ):
        portability.require(
            importlib.metadata.version(distribution)
            == origin["encoding_identity"][key],
            "Original encoding software differs",
        )
    import imageio_ffmpeg

    portability.require(
        imageio_ffmpeg.get_ffmpeg_version() == origin["encoding_identity"]["ffmpeg"],
        "Original FFmpeg build differs",
    )
    portability.require(
        spec == inspected["origin_spec"],
        "Actual original production specification required",
    )
    portability.require(
        time.monotonic() - started < 1800, "Restore startup deadline exhausted"
    )
    private = Path(tempfile.mkdtemp(prefix="portable-restore-only-"))
    state = checked_portable_resume(
        resume,
        private,
        origin,
        spec,
        policy,
        evidence,
        service,
        parent_inventory_path,
        guard,
        exact,
    )

    def geometry(value):
        if isinstance(value, exact.torch.Tensor):
            raw = value.detach().cpu().contiguous().numpy().tobytes()
            import hashlib

            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(key): geometry(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [geometry(item) for item in value]
        return value

    def rng_metadata(value):
        return {
            "torch": geometry(value["torch"]),
            "python_sha256": exact.fingerprint(value["python"]),
            "numpy": geometry(value["numpy"]),
        }

    def optimizer_metadata(value):
        return {
            "param_groups": geometry(value["param_groups"]),
            "state": {
                str(key): {"step": float(item["step"]), "geometry": geometry(item)}
                for key, item in value["state"].items()
            },
        }

    def restore_optimizer(model, saved, config, expected_steps):
        optimizer = exact.torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        expected_groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
        optimizer.load_state_dict(saved)
        portability.require(
            optimizer.state_dict()["param_groups"] == expected_groups,
            "Saved optimizer hyperparameters/parameter order differ",
        )
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                item = optimizer.state.get(parameter, {})
                portability.require(
                    set(item) == {"step", "exp_avg", "exp_avg_sq"}
                    and item["step"].numel() == 1
                    and float(item["step"]) == expected_steps
                    and all(
                        item[key].shape == parameter.shape
                        and item[key].dtype == parameter.dtype
                        for key in ("exp_avg", "exp_avg_sq")
                    ),
                    "Saved optimizer moments/step geometry differs",
                )
        portability.require(
            optimizer_metadata(optimizer.state_dict()) == optimizer_metadata(saved),
            "Optimizer restoration changed saved bytes/metadata",
        )
        return optimizer

    completed = []
    for index, record in enumerate(state["completed"]):
        guard()
        saved = exact.torch.load(
            private / record["name"], map_location="cpu", weights_only=True
        )
        observed = inspected["completed"][index]
        config = spec["configs"][index // len(spec["seeds"])]
        model = exact.Adapter()
        model.load_state_dict(saved["state"], strict=True)
        portability.require(
            geometry(model.state_dict()) == observed["model"],
            "Completed best model differs from inspected native state",
        )
        model.load_state_dict(saved["final_model"], strict=True)
        portability.require(
            geometry(model.state_dict()) == observed["final_model"],
            "Completed final model differs from inspected native state",
        )
        steps = saved["history"][-1]["epoch"] * math.ceil(23202 / spec["batch"])
        optimizer = restore_optimizer(model, saved["optimizer"], config, steps)
        portability.require(
            optimizer_metadata(optimizer.state_dict()) == observed["optimizer"],
            "Completed optimizer differs from actual inspection",
        )
        generator = exact.torch.Generator()
        generator.set_state(saved["generator"])
        portability.require(
            geometry(generator.get_state()) == observed["generator"],
            "Completed generator restore differs",
        )
        exact.restore_rng(saved["rng"])
        portability.require(
            rng_metadata(exact.rng_state()) == observed["rng"],
            "Completed global RNG restore differs",
        )
        completed.append(
            {
                "name": record["name"],
                "sha256": record["sha256"],
                "identity": record["identity"],
                "restored": True,
                "optimizer_steps": steps,
            }
        )
    active = state["active"]
    exact.checked_active(active, spec, 23202)
    model = exact.Adapter()
    model.load_state_dict(active["model"], strict=True)
    if active["best_state"] is not None:
        model.load_state_dict(active["best_state"], strict=True)
        portability.require(
            geometry(model.state_dict()) == inspected["active"]["best_state"],
            "Active best model differs",
        )
        model.load_state_dict(active["model"], strict=True)
    portability.require(
        geometry(model.state_dict()) == inspected["active"]["model"],
        "Active model restore differs",
    )
    steps = (active["epoch"] - 1) * math.ceil(23202 / spec["batch"]) + math.ceil(
        active["cursor"] / spec["batch"]
    )
    optimizer = restore_optimizer(
        model, active["optimizer"], spec["configs"][state["grid"]], steps
    )
    portability.require(
        optimizer_metadata(optimizer.state_dict()) == inspected["active"]["optimizer"],
        "Active optimizer differs",
    )
    generator = exact.torch.Generator()
    generator.set_state(active["generator"])
    portability.require(
        geometry(generator.get_state()) == inspected["active"]["generator"]
        and geometry(active["permutation"]) == inspected["active"]["permutation"],
        "Active generator/permutation differs",
    )
    exact.restore_rng(state["rng"])
    restored_rng = rng_metadata(exact.rng_state())
    portability.require(
        restored_rng == inspected["rng"], "Active global RNG restore differs"
    )
    guard()
    portability.require(exact.runtime() == actual, "Restore execution runtime changed")
    portability.require(
        time.monotonic() - started < 1800, "Restore-only service deadline exhausted"
    )
    return {
        "schema": "portable_restore_only_result_v2",
        "action": "restore_only",
        "phase": phase,
        "restored": True,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "origin_identity_sha256": exact.fingerprint(origin),
        "actual_runtime": actual,
        "policy_sha256": policy_sha256,
        "service_sha256": service_contract_sha256,
        "parent_snapshot_sha256": service["parent_checkpoint_sha256"],
        "completed": completed,
        "active_progress": progress(state),
        "optimizer_steps": steps,
        "generator": geometry(generator.get_state()),
        "permutation": geometry(active["permutation"]),
        "rng": restored_rng,
        "elapsed_seconds": time.monotonic() - started,
        "bitwise_equivalence_claimed": False,
    }


def engine(
    data,
    manifest,
    baseline,
    output,
    identity,
    spec,
    *,
    resume=None,
    resume_sha256=None,
    max_seconds=None,
    checkpoint_every_batches=None,
    checkpoint_every_seconds=300.0,
    max_batches=None,
    service_started=None,
    policy,
    evidence,
    service,
    inventory_path,
    guard,
    execution_runtime,
):
    """Unchanged scientific update loop with explicit v2 provenance boundaries."""
    guard()
    import numpy as np
    import torch

    from src.models import resumable_training as exact
    from src.models.resumable_training import (
        Adapter,
        checked_active,
        conditional_loss,
        digest,
        eligible,
        evaluate,
        fingerprint,
        json_new,
        multipositive_loss,
        primary,
        restore_rng,
        rng_state,
        runtime,
        save_new,
        score_policy,
    )

    started = time.monotonic() if service_started is None else service_started
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if (
        torch.get_default_dtype() != torch.float32
        or str(torch.get_default_device()) != "cpu"
    ):
        raise ValueError("CPU float32 training required")
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh continuation output required")
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Positive finite service window required")
    if checkpoint_every_batches is not None and (
        type(checkpoint_every_batches) is not int or checkpoint_every_batches < 1
    ):
        raise ValueError("Positive checkpoint interval required")
    if (
        isinstance(checkpoint_every_seconds, bool)
        or not math.isfinite(checkpoint_every_seconds)
        or checkpoint_every_seconds <= 0
    ):
        raise ValueError("Positive checkpoint time interval required")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("Positive batch limit required")
    if (resume is None) != (resume_sha256 is None):
        raise ValueError("Resume path and fingerprint required together")
    identity = copy.deepcopy(identity)
    portability.require(
        identity == evidence["report"]["phases"][spec["phase"]]["origin_identity"],
        "Exact immutable origin identity required",
    )
    portability.runtime_guard(execution_runtime, spec["phase"], policy, evidence)
    portability.require(
        runtime() == execution_runtime
        and spec == evidence["report"]["phases"][spec["phase"]]["origin_spec"],
        "Actual execution runtime/spec changed",
    )
    training, validation = data["train"], data["val"]
    valid = torch.from_numpy(training["positive_index"].valid)
    query_ids = np.flatnonzero(training["positive_index"].valid)
    x = training["x"][valid]
    v = training["v"]
    if not len(x):
        raise ValueError("No training query has a consensus-positive moment")
    output.mkdir(parents=True)
    if resume is None:
        raise ValueError(
            "Authentic prior snapshot required; scratch restart is not this protocol"
        )
    state = checked_portable_resume(
        resume,
        output,
        identity,
        spec,
        policy,
        evidence,
        service,
        inventory_path,
        guard,
        exact,
    )
    state["portable"]["execution_lineage"] = portability.append_execution(
        state["portable"]["execution_lineage"],
        service,
        execution_runtime,
        progress(state),
        spec["phase"],
        policy,
        evidence,
    )
    if state["baseline"] != baseline:
        raise ValueError("Resume baseline differs")
    window_batches = 0
    snapshots = 0
    model = None
    optimizer = None
    generator = None
    last_checkpoint = time.monotonic()

    # Validation and completed-policy replay are intentionally indivisible work
    # units. Budget checks occur before them, never halfway through a metric.
    def checkpoint(stopped):
        nonlocal snapshots, last_checkpoint
        if state["active"] is not None:
            state["active"].update(
                model=copy.deepcopy(model.state_dict()),
                optimizer=copy.deepcopy(optimizer.state_dict()),
                generator=generator.get_state(),
            )
        state["rng"] = rng_state()
        saved = copy.deepcopy(state)
        saved["service_seconds"] = state["service_seconds"] + time.monotonic() - started
        guard()
        portability.require(
            runtime() == execution_runtime,
            "Execution runtime changed before checkpoint",
        )
        portability.validate_lineage(
            saved["portable"]["execution_lineage"],
            spec["phase"],
            policy,
            evidence,
            service["portable_policy_sha256"],
        )
        snapshots += 1
        path = output / f"checkpoint-{snapshots:06d}.pt"
        sha = save_new(path, saved)
        last_checkpoint = time.monotonic()
        json_new(
            path.with_suffix(".json"),
            {
                "schema": 2,
                "snapshot_sha256": sha,
                "origin_identity_sha256": fingerprint(identity),
                "portable_policy_sha256": service["portable_policy_sha256"],
                "lineage_sha256": fingerprint(saved["portable"]["execution_lineage"]),
                "execution_runtime_sha256": fingerprint(execution_runtime),
                "completed_seeds": len(state["completed"]),
                "batches": state["batches"],
                "stage": state["stage"],
            },
        )
        if stopped:
            result = {
                "status": "paused_service_window",
                "snapshot": path.name,
                "snapshot_sha256": sha,
                "completed_seeds": len(state["completed"]),
                "batches": state["batches"],
                "test_targets_used": False,
            }
            names = [path.name, path.with_suffix(".json").name] + [
                r["name"] for r in state["completed"]
            ]
            inventory = {
                "schema": "portable_checkpoint_inventory_v2",
                "files": {
                    name: {
                        "sha256": digest(output / name),
                        "bytes": (output / name).stat().st_size,
                    }
                    for name in names
                },
            }
            json_new(output / "continuation-inventory.json", inventory)
            result.update(
                schema=2,
                origin_identity_sha256=fingerprint(identity),
                portable_policy_sha256=service["portable_policy_sha256"],
                lineage_sha256=fingerprint(state["portable"]["execution_lineage"]),
                inventory_sha256=digest(output / "continuation-inventory.json"),
                bitwise_equivalence_claimed=False,
            )
            json_new(output / "continuation.json", result)
            return result

    def boundary():
        return (
            max_seconds is not None and time.monotonic() - started >= max_seconds
        ) or (max_batches is not None and window_batches >= max_batches)

    # Loading constructors must not advance the resumed random streams.
    if state["active"] is not None:
        active = state["active"]
        checked_active(active, spec, len(x))
        config = spec["configs"][state["grid"]]
        model = Adapter()
        model.load_state_dict(active["model"], strict=True)
        if active["best_state"] is not None:
            model.load_state_dict(active["best_state"], strict=True)
            model.load_state_dict(active["model"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        expected_groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
        optimizer.load_state_dict(active["optimizer"])
        if optimizer.state_dict()["param_groups"] != expected_groups:
            raise ValueError("Saved optimizer hyperparameters differ")
        expected_steps = (active["epoch"] - 1) * math.ceil(
            len(x) / spec["batch"]
        ) + math.ceil(active["cursor"] / spec["batch"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                saved = optimizer.state.get(parameter, {})
                if (
                    set(saved) != {"step", "exp_avg", "exp_avg_sq"}
                    or saved["step"].numel() != 1
                    or float(saved["step"]) != expected_steps
                    or any(
                        saved[key].shape != parameter.shape
                        or saved[key].dtype != parameter.dtype
                        for key in ("exp_avg", "exp_avg_sq")
                    )
                ):
                    raise ValueError("Saved optimizer moments/step geometry differ")
        generator = torch.Generator()
        generator.set_state(active["generator"])
    restore_rng(state["rng"])
    while state["grid"] < len(spec["configs"]):
        if boundary():
            return checkpoint(True)
        grid = state["grid"]
        config = spec["configs"][grid]
        if state["stage"] == "fit":
            seed = spec["seeds"][state["seed_index"]]
            if state["active"] is None:
                torch.manual_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                model = Adapter()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                )
                state["active"] = {
                    "epoch": 1,
                    "cursor": 0,
                    "permutation": None,
                    "total": 0.0,
                    "history": [],
                    "best": -1.0,
                    "best_epoch": 0,
                    "best_state": None,
                }
            active = state["active"]
            if active["permutation"] is None:
                active["permutation"] = torch.randperm(len(x), generator=generator)
            if active["cursor"] < len(x):
                model.train()
                ix = active["permutation"][
                    active["cursor"] : active["cursor"] + spec["batch"]
                ]
                optimizer.zero_grad()
                logits = model(x[ix]) @ v.T / spec["temperature"]
                positive = torch.from_numpy(
                    training["positive_index"].mask(query_ids[ix.numpy()])
                )
                if spec["phase"] == "phase1":
                    loss = multipositive_loss(logits, positive)
                else:
                    loss = conditional_loss(logits, positive, torch.ones_like(positive))
                    if config["lambda"]:
                        allowed = torch.from_numpy(
                            training["positive_index"].same_video_mask(
                                query_ids[ix.numpy()]
                            )
                        )
                        loss = loss + config["lambda"] * conditional_loss(
                            logits, positive, allowed
                        )
                loss.backward()
                if not torch.isfinite(loss) or not all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise ValueError("Nonfinite training")
                optimizer.step()
                active["total"] += float(loss.detach()) * len(ix)
                active["cursor"] += len(ix)
                state["batches"] += 1
                window_batches += 1
                if (
                    checkpoint_every_batches is not None
                    and state["batches"] % checkpoint_every_batches == 0
                ) or time.monotonic() - last_checkpoint >= checkpoint_every_seconds:
                    checkpoint(False)
                continue
            value = primary(model, validation)
            epoch = active["epoch"]
            active["history"].append(
                {
                    "epoch": epoch,
                    "train_loss": active["total"] / len(x),
                    "validation_corpus_consensus_r1_iou50": value,
                }
            )
            if value > active["best"]:
                active["best"] = value
                active["best_epoch"] = epoch
                active["best_state"] = copy.deepcopy(model.state_dict())
            print(
                json.dumps(
                    {
                        "phase": spec["phase"],
                        "config": config["name"],
                        "seed": seed,
                        **active["history"][-1],
                    }
                ),
                flush=True,
            )
            done = epoch >= spec["maximum_epochs"] or (
                epoch >= spec["minimum_epochs"]
                and epoch - active["best_epoch"] >= spec["patience"]
            )
            if done:
                bound = seed_identity_v2(
                    identity,
                    spec,
                    grid,
                    seed,
                    state["portable"]["execution_lineage"],
                    service["portable_policy_sha256"],
                )
                name = f"config{grid}-seed{seed}.pt"
                payload = {
                    "identity": bound,
                    "state": active["best_state"],
                    "best_epoch": active["best_epoch"],
                    "history": active["history"],
                    "final_model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "generator": generator.get_state(),
                    "rng": rng_state(),
                }
                payload.update(
                    schema=2,
                    portable_lineage=copy.deepcopy(
                        state["portable"]["execution_lineage"]
                    ),
                )
                sha = save_new(output / name, payload)
                state["completed"].append(
                    {"name": name, "sha256": sha, "identity": bound}
                )
                state["seed_index"] += 1
                state["active"] = None
                model = None
                optimizer = None
                generator = None
                if state["seed_index"] == len(spec["seeds"]):
                    state["stage"] = "candidate"
                checkpoint(False)
            else:
                active.update(epoch=epoch + 1, cursor=0, permutation=None, total=0.0)
            continue
        # Every completed seed is hash-bound and re-used; no partial seed/grid
        # is eligible for a selected model.
        records = state["completed"][
            grid * len(spec["seeds"]) : (grid + 1) * len(spec["seeds"])
        ]
        candidate_bytes(records, output, exact)
        completed = [
            torch.load(output / r["name"], map_location="cpu", weights_only=True)
            for r in records
        ]
        states = [item["state"] for item in completed]
        alpha = spec["alphas"][state["policy_index"]]
        bundle = {"schema": 1, "states": states, "adapted_score_weight": alpha}
        measured = evaluate(
            score_policy(bundle, validation["x"], validation["v"]),
            validation["candidates"],
            validation["queries"],
        )
        adopted = eligible(measured, baseline)
        pvalue = measured["corpus_consensus_r1_iou50"]
        if spec["phase"] == "phase2":
            state["policies"].append(
                {
                    "lambda": config["lambda"],
                    "adapted_score_weight": alpha,
                    "validation": measured,
                    "promotion_eligible": adopted,
                }
            )
        state["candidate_metrics"].append(measured)
        if adopted and pvalue > state["best_promoted"]:
            state.update(
                best_promoted=pvalue,
                winner_states=states,
                winner_alpha=alpha,
                winner_metrics=measured,
                winner_name=config["name"]
                if spec["phase"] == "phase1"
                else f"lambda{config['lambda']}-alpha{alpha}",
            )
        state["policy_index"] += 1
        if state["policy_index"] == len(spec["alphas"]):
            seed_reports = [
                {
                    "seed": seed,
                    "best_epoch": item["best_epoch"],
                    "history": item["history"],
                }
                for seed, item in zip(spec["seeds"], completed)
            ]
            if spec["phase"] == "phase1":
                state["runs"].append(
                    {
                        "config": config,
                        "seeds": seed_reports,
                        "validation": measured,
                        "promotion_eligible": adopted,
                    }
                )
            else:
                state["runs"].append(
                    {"lambda": config["lambda"], "seeds": seed_reports}
                )
            state.update(
                grid=grid + 1,
                seed_index=0,
                stage="fit",
                policy_index=0,
                candidate_metrics=[],
            )
        checkpoint(False)
    state["stage"] = "final"
    if boundary():
        return checkpoint(True)
    if spec["phase"] == "phase1":
        for name in ("random_control", "chronological_control"):
            if name in state["controls"]:
                continue
            if boundary():
                return checkpoint(True)
            shape = (len(validation["x"]), len(validation["v"]))
            values = (
                np.random.default_rng(42).random(shape)
                if name == "random_control"
                else np.broadcast_to(-np.arange(shape[1], dtype=float), shape)
            )
            state["controls"][name] = evaluate(
                values, validation["candidates"], validation["queries"]
            )
            checkpoint(False)
    required = len(spec["configs"]) * len(spec["seeds"])
    if (
        len(state["completed"]) != required
        or len(state["runs"]) != len(spec["configs"])
        or (
            spec["phase"] == "phase2"
            and len(state["policies"]) != len(spec["configs"]) * len(spec["alphas"])
        )
    ):
        raise ValueError("Cannot finalize an incomplete fit/policy grid")
    for record in state["completed"]:
        if (
            not (output / record["name"]).is_file()
            or digest(output / record["name"]) != record["sha256"]
        ):
            raise ValueError("Cannot finalize missing/changed completed seed")
    bundle = {
        "schema": 1,
        "winner": state["winner_name"],
        "states": state["winner_states"],
        "adapted_score_weight": state["winner_alpha"],
        "encoder": manifest["encoder"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "validation": state["winner_metrics"],
        "training_consensus_no_positive_queries": int((~valid).sum()),
        "test_targets_used": False,
    }
    sha = save_new(output / "selected.pt", bundle)
    restored = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
    delta = float(
        np.max(
            np.abs(
                score_policy(bundle, validation["x"][:3], validation["v"])
                - score_policy(restored, validation["x"][:3], validation["v"])
            )
        )
    )
    if delta > 1e-6:
        raise ValueError("Saved policy replay mismatch")
    state["stage"] = "done"
    checkpoint(False)
    result = {
        "baseline": baseline,
        "runs": state["runs"],
        "selected": state["winner_name"],
        "selected_validation": state["winner_metrics"],
        "protocol_sha256": identity["protocol_sha256"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "selected_checkpoint_sha256": sha,
        "checkpoint_replay_max_abs_error": delta,
        "test_targets_used": False,
        "service_seconds": state["service_seconds"] + time.monotonic() - started,
        "completed_fits": len(state["completed"]),
        "required_fits": len(spec["configs"]) * len(spec["seeds"]),
        "continuation_identity": {
            "schema": 2,
            "origin_identity_sha256": fingerprint(identity),
            "portable_policy_sha256": service["portable_policy_sha256"],
            "lineage_sha256": fingerprint(state["portable"]["execution_lineage"]),
            "actual_runtime_sha256": fingerprint(execution_runtime),
        },
        "origin_identity": identity,
        "execution_lineage": state["portable"]["execution_lineage"],
        "actual_execution_runtime": execution_runtime,
        "bitwise_equivalence_claimed": False,
        **state["controls"],
    }
    if spec["phase"] == "phase2":
        result["policies"] = state["policies"]
    json_new(output / "results.json", result)
    return result
