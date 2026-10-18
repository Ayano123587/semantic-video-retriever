"""Cloud-only authenticated checkpoint metadata inspection; no fitting or scoring."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_INSPECTION_PLAN_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def authenticate(package, expected):
    raw = (Path(package) / "inspection-plan.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Exact root inspection plan required")
    plan = json.loads(raw)
    if plan["schema"] != "video_checkpoint_inspection_transport_v1":
        raise ValueError("Inspection transport required")
    names = set()
    for name, entry in plan["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Unsafe inspection path")
        flat = entry["upload_name"]
        if Path(flat).name != flat or "/" in flat or "\\" in flat or flat in names:
            raise ValueError("Unique flat payload required")
        names.add(flat)
        path = Path(package) / flat
        if (
            path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or sha(path) != entry["sha256"]
        ):
            raise ValueError("Inspection payload differs")
    return plan


def validate_contract(contract, recovery, freeze_sha256, audit_sha256, inventories):
    expected = {
        "schema": "video_checkpoint_inspection_contract_v1",
        "action": "inspect_original_checkpoint_metadata_no_training",
        "checkpoint_freeze_sha256": freeze_sha256,
        "audit_sha256": audit_sha256,
        "inventories": inventories,
        "origin_identity_sha256": {
            phase: value["identity_sha256"] for phase, value in recovery.PARENTS.items()
        },
        "source_manifest_sha256": recovery.SOURCE_SHA,
        "cache_manifest_sha256": recovery.CACHE_SHA,
        "protocol_sha256": recovery.PROTOCOL_SHA,
        "cpu_policy": "record_actual_cpu_for_inspection_only",
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "max_seconds": 1800,
    }
    if contract != expected:
        raise ValueError("Exact actual root inspection-only contract required")


def diff_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = prefix + "." + str(key) if prefix else str(key)
            result.extend(
                [path]
                if key not in left or key not in right
                else diff_paths(left[key], right[key], path)
            )
        return result
    return [] if type(left) is type(right) and left == right else [prefix]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(input_root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "stage": "start",
        "training": False,
        "scoring": False,
        "native_loading_started": False,
        "inspection_complete": False,
    }

    def record():
        (output / "inspection-status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    def deadline():
        remaining = 1800 - (time.monotonic() - started)
        if remaining <= 0:
            raise ValueError("Inspection service budget exhausted")
        return remaining

    record()
    try:
        matches = [
            p
            for p in Path(input_root).rglob("inspection-plan.json")
            if sha(p) == EXPECTED_INSPECTION_PLAN_SHA256
        ]
        if len(matches) != 1:
            raise ValueError("One exact inspection transport required")
        package = matches[0].parent
        plan = authenticate(package, EXPECTED_INSPECTION_PLAN_SHA256)
        private = Path(tempfile.mkdtemp(prefix="video-checkpoint-inspection-"))
        source = private / "source"
        source.mkdir()
        for name, entry in plan["files"].items():
            raw = (package / entry["upload_name"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Inspection payload changed")
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        template = (source / "inspection-runner.py").read_bytes()
        token = b"EXPECTED_INSPECTION_PLAN_SHA256" + b" = None"
        if template.count(token) != 1 or Path(__file__).read_bytes().replace(
            b"\r\n", b"\n"
        ) != template.replace(
            token,
            (
                "EXPECTED_INSPECTION_PLAN_SHA256 = "
                + repr(EXPECTED_INSPECTION_PLAN_SHA256)
            ).encode(),
        ):
            raise ValueError("Executed inspection runner differs")
        recovery = load_module(
            "recovery", source / "interrupted_checkpoint_recovery.py"
        )
        gatea = load_module("gatea", source / "kaggle_interrupted_attestation.py")
        legacy = load_module("legacy", source / "kaggle_research_training.py")
        contract = recovery._json(
            source / "root-inspection-contract.json", plan["inspection_contract_sha256"]
        )
        validate_contract(
            contract,
            recovery,
            plan["checkpoint_freeze_sha256"],
            plan["audit_sha256"],
            plan["inventories"],
        )
        freeze = recovery._json(
            source / "checkpoint-freeze.json", contract["checkpoint_freeze_sha256"]
        )
        providers = [
            p
            for p in Path(input_root).rglob("provider-provenance.json")
            if sha(p) == freeze["provider_provenance_sha256"]
        ]
        if len(providers) != 1:
            raise ValueError("One exact checkpoint dataset required")
        parent = recovery._cloud_parent(providers[0].parent)
        provider = json.loads(providers[0].read_bytes())
        audit = source / "terminal-audit-report.json"
        mappings = {}
        references = {}
        for phase in ("phase1", "phase2"):
            references[phase] = recovery.audit_reference(
                audit, contract["audit_sha256"], phase
            )
            mappings[phase] = gatea.checkpoint_mapping(
                freeze, provider, phase, references[phase]
            )

        def checkpoint_bytes():
            deadline()
            recovery._cloud_parent(parent)
            for name, entry in freeze["remote_files"].items():
                path = parent / name
                if (
                    path.is_symlink()
                    or path.stat().st_size != entry["bytes"]
                    or sha(path) != entry["sha256"]
                ):
                    raise ValueError("All nineteen checkpoint dataset bytes required")
            for phase in ("phase1", "phase2"):
                recovery.authenticate_frozen_inventory(
                    audit,
                    contract["audit_sha256"],
                    phase,
                    parent,
                    source / (phase + "-inventory.json"),
                    contract["inventories"][phase],
                    native_paths=mappings[phase],
                )

        checkpoint_bytes()
        if (
            sha(source / "cache-manifest.json") != legacy.CACHE_SHA
            or sha(source / "protocol.json") != legacy.PROTOCOL_SHA
        ):
            raise ValueError("Actual original cache/protocol metadata required")
        encoded = json.loads((source / "cache-manifest.json").read_bytes())
        protocol = json.loads((source / "protocol.json").read_bytes())
        original = legacy.unique_hash(
            Path(input_root).rglob("video-source-manifest.json"), legacy.SOURCE_SHA
        )
        workspace = private / "original-scientific-source"
        manifest = legacy.restore_source(original, workspace)

        def source_bytes():
            authenticate(package, EXPECTED_INSPECTION_PLAN_SHA256)
            for name, entry in plan["files"].items():
                if sha(source / name) != entry["sha256"]:
                    raise ValueError("Materialized inspection source changed")
            if sha(original) != legacy.SOURCE_SHA:
                raise ValueError("Original source manifest changed")
            for row in manifest["files"]:
                if sha(workspace / row["path"]) != row["sha256"]:
                    raise ValueError("Original scientific source changed")

        if platform.python_version() != legacy.PYTHON:
            raise ValueError("Exact original Python required")
        status.update(stage="software_bootstrap", expected_python=legacy.PYTHON)
        record()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers==4.57.6",
                "pillow==12.3.0",
                "imageio==2.37.4",
                "imageio-ffmpeg==0.6.0",
            ],
            check=True,
            timeout=deadline(),
        )
        expected = encoded["encoding_identity"]
        packages = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "transformers",
                "pillow",
                "imageio",
                "imageio-ffmpeg",
            )
        }
        package_keys = {
            "torch": "torch",
            "numpy": "numpy",
            "transformers": "transformers",
            "pillow": "pillow",
            "imageio": "imageio",
            "imageio-ffmpeg": "imageio_ffmpeg",
        }
        status.update(
            stage="software_preimport",
            packages=packages,
            expected_packages={
                name: expected[key] for name, key in package_keys.items()
            },
        )
        record()
        if any(packages[name] != expected[key] for name, key in package_keys.items()):
            raise ValueError("Exact original numerical software required")
        checkpoint_bytes()
        source_bytes()
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        import imageio_ffmpeg
        import torch

        if (
            hashlib.sha256(torch.__config__.show().encode()).hexdigest()
            != expected["torch_build_sha256"]
            or imageio_ffmpeg.get_ffmpeg_version() != expected["ffmpeg"]
        ):
            raise ValueError("Exact original Torch/FFmpeg builds required")
        from src.models import resumable_training as trainer

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        actual_runtime = trainer.runtime()
        status.update(
            stage="authenticated_native_inspection",
            native_loading_started=True,
            actual_runtime=actual_runtime,
        )
        record()
        tensor_count = 0

        def geometry(value, depth=0):
            nonlocal tensor_count
            if depth > 12:
                raise ValueError("State geometry depth bound exceeded")
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                if tensor_count > 4096 or value.ndim > 8:
                    raise ValueError("State geometry tensor bound exceeded")
                raw = (
                    value.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
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
                if len(value) > 4096:
                    raise ValueError("State geometry mapping bound exceeded")
                return {
                    str(key): geometry(item, depth + 1) for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                if len(value) > 4096:
                    raise ValueError("State geometry sequence bound exceeded")
                return [geometry(item, depth + 1) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise ValueError("Unexpected state metadata type")

        def rng_metadata(rng):
            return {
                "torch": geometry(rng["torch"]),
                "python_sha256": fingerprint(rng["python"]),
                "numpy": geometry(rng["numpy"]),
            }

        def optimizer_metadata(optimizer):
            groups = optimizer["param_groups"]
            result = {"param_groups": groups, "state": {}}
            for key, value in optimizer["state"].items():
                result["state"][str(key)] = {
                    "step": float(value["step"]),
                    "geometry": geometry(value),
                }
            return result

        report = {
            "schema": "video_checkpoint_inspection_result_v1",
            "inspection_contract_sha256": plan["inspection_contract_sha256"],
            "actual_runtime": actual_runtime,
            "packages": packages,
            "training": False,
            "scoring": False,
            "test_annotations_used": False,
            "phases": {},
        }
        for phase in ("phase1", "phase2"):
            reference = references[phase]
            mapping = mappings[phase]
            checkpoint_bytes()
            source_bytes()
            state = torch.load(
                parent / mapping[reference["snapshot"]],
                map_location="cpu",
                weights_only=True,
            )
            origin = state["identity"]
            if fingerprint(origin) != contract["origin_identity_sha256"][phase]:
                raise ValueError("Original full checkpoint identity mismatch")
            spec = state["spec"]
            if spec != trainer.production_spec(phase) or origin[
                "spec_sha256"
            ] != fingerprint(spec):
                raise ValueError("Original scientific spec mismatch")
            base = {
                "schema": 1,
                "phase": phase,
                "protocol_sha256": legacy.PROTOCOL_SHA,
                "cache_manifest_sha256": protocol["source_manifest_sha256"],
                "implementation_sha256": protocol["implementation_sha256"],
                "coverage": protocol["coverage"],
                "encoding_identity": protocol["encoding_identity"],
                "runtime": origin["runtime"],
                "spec_sha256": fingerprint(spec),
            }
            if (
                origin != base
                or state["schema"] != 1
                or state["stage"] != "fit"
                or state["batches"] != reference["batches"]
            ):
                raise ValueError(
                    "Original checkpoint scientific identity/progress mismatch"
                )
            if len(state["completed"]) != reference[
                "completed_seeds"
            ] or not trainer.finite_tree(state):
                raise ValueError("Original completed prefix/nonfinite state mismatch")
            active = state["active"]
            if active is None:
                raise ValueError("Actual active interrupted seed required")
            active_report = {
                "epoch": active["epoch"],
                "cursor": active["cursor"],
                "best_epoch": active["best_epoch"],
                "history_count": len(active["history"]),
                "history_sha256": fingerprint(active["history"]),
                "model": geometry(active["model"]),
                "best_state": geometry(active["best_state"]),
                "optimizer": optimizer_metadata(active["optimizer"]),
                "generator": geometry(active["generator"]),
                "permutation": geometry(active["permutation"]),
            }
            completed = []
            for index, record_seed in enumerate(state["completed"]):
                grid, seed_index = divmod(index, len(spec["seeds"]))
                seed = spec["seeds"][seed_index]
                name = f"config{grid}-seed{seed}.pt"
                bound = trainer.seed_identity(origin, spec, grid, seed)
                if (
                    record_seed["name"] != name
                    or record_seed["identity"] != bound
                    or record_seed["sha256"]
                    != freeze["remote_files"][mapping[name]]["sha256"]
                ):
                    raise ValueError(
                        "Original completed seed identity/hash prefix mismatch"
                    )
                checkpoint_bytes()
                source_bytes()
                saved = torch.load(
                    parent / mapping[name], map_location="cpu", weights_only=True
                )
                if saved["identity"] != bound or not trainer.finite_tree(saved):
                    raise ValueError(
                        "Completed seed payload identity/nonfinite mismatch"
                    )
                completed.append(
                    {
                        "name": name,
                        "sha256": record_seed["sha256"],
                        "identity": saved["identity"],
                        "best_epoch": saved["best_epoch"],
                        "history_count": len(saved["history"]),
                        "history_sha256": fingerprint(saved["history"]),
                        "model": geometry(saved["state"]),
                        "final_model": geometry(saved["final_model"]),
                        "optimizer": optimizer_metadata(saved["optimizer"]),
                        "generator": geometry(saved["generator"]),
                        "rng": rng_metadata(saved["rng"]),
                    }
                )
            report["phases"][phase] = {
                "reference": reference,
                "origin_identity": origin,
                "origin_spec": spec,
                "runtime_diff_paths": diff_paths(origin["runtime"], actual_runtime),
                "stage": state["stage"],
                "batches": state["batches"],
                "grid": state["grid"],
                "seed_index": state["seed_index"],
                "policy_index": state["policy_index"],
                "completed_seeds": len(completed),
                "active": active_report,
                "rng": rng_metadata(state["rng"]),
                "completed": completed,
            }
        checkpoint_bytes()
        source_bytes()
        if trainer.runtime() != actual_runtime:
            raise ValueError("Inspection runtime changed")
        raw = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
        if len(raw) > 2_000_000:
            raise ValueError("Bounded inspection report exceeded")
        (output / "checkpoint-inspection.json").write_bytes(raw)
        status.update(
            stage="inspection_complete",
            inspection_complete=True,
            report_sha256=hashlib.sha256(raw).hexdigest(),
            elapsed_seconds=time.monotonic() - started,
            tensor_metadata_count=tensor_count,
        )
        record()
    except Exception as error:
        status.update(
            stage="failed",
            inspection_complete=False,
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/checkpoint-inspection")
