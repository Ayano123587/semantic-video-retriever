"""Schema3 advanced-parent continuation; unchanged audited training mathematics.

The explicit engine fork changes provenance only. Whole-function AST comparison
against the frozen portable engine rejects any optimizer, loss, epoch, scoring,
validation, promotion, or selected-bundle change.
"""

import copy
import importlib.metadata
import json
import math
import time
from pathlib import Path

from src.models import advanced_parent as advanced
from src.models import cpu_portability as portability
from src.models.portable_training import candidate_bytes, progress, source_closure


def authenticate_source(path, service, origin):
    manifest = portability.read_bound_json(path, service["source_manifest_sha256"])
    portability.require(
        manifest["schema"] == "advanced_training_source_manifest_v3",
        "Distinct advanced source closure required",
    )
    root = Path(__file__).resolve().parents[2]
    scientific = {
        "src/" + name: pin for name, pin in origin["implementation_sha256"].items()
    }
    expected = source_closure(
        root,
        [
            *scientific,
            "src/models/advanced_parent.py",
            "src/models/advanced_training.py",
        ],
    )
    portability.require(
        set(manifest["files"]) == expected,
        "Complete actual transitive advanced source closure required",
    )
    for name, pin in manifest["files"].items():
        relative = Path(name)
        portability.require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in name,
            "Safe reviewed source path required",
        )
        portability.require(
            not (root / relative).is_symlink()
            and portability.sha(root / relative) == pin,
            "Frozen advanced source bytes differ",
        )
        if name in scientific:
            portability.require(
                pin == scientific[name], "Original scientific source must remain exact"
            )
    return manifest


def train(
    preparation,
    cache,
    output,
    protocol,
    *,
    policy_path,
    policy_sha256,
    evidence_paths,
    advanced_paths,
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
        isinstance(started, (int, float))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= started <= time.monotonic(),
        "Actual original service start required",
    )
    original = portability.load_evidence(
        evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
    )
    context = advanced.load_context(advanced_paths, original)
    policy = portability.authenticate_policy(policy_path, policy_sha256, original)
    service = advanced.authenticate_service(
        service_contract_path, service_contract_sha256, policy_sha256, original, context
    )
    origin = context["report"]["origin_identity"]

    def guard():
        advanced.load_context(advanced_paths, original)
        advanced.authenticate_service(
            service_contract_path,
            service_contract_sha256,
            policy_sha256,
            original,
            context,
        )
        authenticate_source(source_manifest_path, service, origin)
        advanced.authenticate_parent(resume, parent_inventory_path, service, context)
        portability.require(
            portability.sha(protocol) == policy["origin_protocol_sha256"]
            and portability.sha(Path(cache) / "manifest.json")
            == policy["cache_manifest_sha256"],
            "Original cache/protocol metadata changed",
        )
        cached = json.loads((Path(cache) / "manifest.json").read_bytes())
        for name, pin in cached["files"].items():
            portability.require(
                Path(name).name == name
                and not (Path(cache) / name).is_symlink()
                and portability.sha(Path(cache) / name) == pin,
                "All original native cache bytes required",
            )

    guard()
    from src.models import resumable_training as exact

    exact.torch.set_num_threads(2)
    exact.torch.use_deterministic_algorithms(True)
    actual = exact.runtime()
    portability.runtime_guard(actual, "phase2", policy, original)
    for distribution, key in (
        ("transformers", "transformers"),
        ("pillow", "pillow"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ):
        portability.require(
            importlib.metadata.version(distribution)
            == origin["encoding_identity"][key],
            "Original encoding software required",
        )
    import imageio_ffmpeg

    portability.require(
        imageio_ffmpeg.get_ffmpeg_version() == origin["encoding_identity"]["ffmpeg"],
        "Original FFmpeg build required",
    )
    from src.models.research_followup import checked_protocol

    data, manifest, frozen = checked_protocol(
        preparation, cache, protocol, policy["origin_protocol_sha256"], metadata
    )
    spec = exact.production_spec("phase2")
    portability.require(
        len(data["train"]["x"]) == 23202
        and len(data["val"]["x"]) == 3132
        and spec == context["report"]["origin_spec"],
        "Entire original cohorts and phase2 grid required",
    )
    portability.require(
        time.monotonic() - started < 36000,
        "Startup-inclusive training budget exhausted",
    )
    result = engine(
        data,
        manifest,
        frozen["baseline"],
        output,
        origin,
        spec,
        resume=resume,
        resume_sha256=service["parent_checkpoint_sha256"],
        max_seconds=36000,
        service_started=started,
        policy=policy,
        evidence=original,
        service=service,
        inventory_path=parent_inventory_path,
        guard=guard,
        execution_runtime=actual,
        advanced_context=context,
    )
    guard()
    portability.require(
        exact.runtime() == actual, "Execution runtime changed during continuation"
    )
    return result


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
    advanced_context,
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
    state = advanced.checked_resume(
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
        advanced_context,
    )
    state["advanced"]["execution_lineage"] = advanced.append_execution(
        state["advanced"]["execution_lineage"],
        service,
        execution_runtime,
        progress(state),
        spec["phase"],
        policy,
        evidence,
        advanced_context,
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
        advanced.validate_lineage(
            saved["advanced"]["execution_lineage"],
            spec["phase"],
            policy,
            evidence,
            service["portable_policy_sha256"],
            advanced_context,
        )
        snapshots += 1
        path = output / f"checkpoint-{snapshots:06d}.pt"
        sha = save_new(path, saved)
        last_checkpoint = time.monotonic()
        json_new(
            path.with_suffix(".json"),
            {
                "schema": 3,
                "snapshot_sha256": sha,
                "origin_identity_sha256": fingerprint(identity),
                "portable_policy_sha256": service["portable_policy_sha256"],
                "lineage_sha256": fingerprint(saved["advanced"]["execution_lineage"]),
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
                "schema": "advanced_checkpoint_inventory_v3",
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
                schema=3,
                origin_identity_sha256=fingerprint(identity),
                portable_policy_sha256=service["portable_policy_sha256"],
                lineage_sha256=fingerprint(state["advanced"]["execution_lineage"]),
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
                bound = advanced.seed_identity(
                    identity,
                    spec,
                    grid,
                    seed,
                    state["advanced"]["execution_lineage"],
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
                    schema=3,
                    advanced_lineage=copy.deepcopy(
                        state["advanced"]["execution_lineage"]
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
            "schema": 3,
            "origin_identity_sha256": fingerprint(identity),
            "portable_policy_sha256": service["portable_policy_sha256"],
            "lineage_sha256": fingerprint(state["advanced"]["execution_lineage"]),
            "actual_runtime_sha256": fingerprint(execution_runtime),
        },
        "origin_identity": identity,
        "execution_lineage": state["advanced"]["execution_lineage"],
        "actual_execution_runtime": execution_runtime,
        "bitwise_equivalence_claimed": False,
        **state["controls"],
    }
    if spec["phase"] == "phase2":
        result["policies"] = state["policies"]
    json_new(output / "results.json", result)
    return result
