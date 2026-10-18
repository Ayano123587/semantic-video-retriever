"""Prospective interrupted recovery; byte and identity gates precede native loads."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_RECOVERY_PLAN_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticate(package, expected):
    raw = (Path(package) / "recovery-plan.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Exact root recovery plan required")
    plan = json.loads(raw)
    if (
        plan["schema"] != "interrupted_recovery_transport_v1"
        or plan["max_seconds"] != 36000
    ):
        raise ValueError("Interrupted recovery service contract required")
    for name, entry in plan["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe logical source")
        flat = entry["upload_name"]
        if Path(flat).name != flat or "/" in flat or "\\" in flat:
            raise ValueError("Flat payload required")
        path = Path(package) / flat
        if (
            path.is_symlink()
            or path.stat().st_size != entry["bytes"]
            or sha(path) != entry["sha256"]
        ):
            raise ValueError("Recovery payload differs")
    return plan


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
        "phase": "bootstrap",
        "scientific_complete": False,
        "native_loading_started": False,
    }

    def record():
        (output / "recovery-status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    record()
    try:
        matches = [
            p
            for p in Path(input_root).rglob("recovery-plan.json")
            if sha(p) == EXPECTED_RECOVERY_PLAN_SHA256
        ]
        if len(matches) != 1:
            raise ValueError("One exact root recovery transport required")
        package = matches[0].parent
        plan = authenticate(package, EXPECTED_RECOVERY_PLAN_SHA256)
        source = output / "authenticated-source"
        source.mkdir()
        for name, entry in plan["files"].items():
            raw = (package / entry["upload_name"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Payload changed")
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        template = (source / "recovery-runner.py").read_bytes()
        token = b"EXPECTED_RECOVERY_PLAN_SHA256" + b" = None"
        if template.count(token) != 1 or Path(__file__).read_bytes().replace(
            b"\r\n", b"\n"
        ) != template.replace(
            token,
            (
                "EXPECTED_RECOVERY_PLAN_SHA256 = " + repr(EXPECTED_RECOVERY_PLAN_SHA256)
            ).encode(),
        ):
            raise ValueError("Executed recovery runner differs")
        recovery = load_module(
            "interrupted_recovery", source / "interrupted_checkpoint_recovery.py"
        )
        gatea = load_module("gatea", source / "kaggle_interrupted_attestation.py")
        legacy = load_module("legacy_training", source / "kaggle_research_training.py")
        audit = source / "terminal-audit-report.json"
        reference = recovery.audit_reference(audit, plan["audit_sha256"], plan["phase"])
        freeze = recovery._json(
            source / "checkpoint-dataset-freeze.json", plan["checkpoint_freeze_sha256"]
        )
        providers = [
            p
            for p in Path(input_root).rglob("provider-provenance.json")
            if sha(p) == freeze["provider_provenance_sha256"]
        ]
        if len(providers) != 1:
            raise ValueError("One root-frozen checkpoint dataset required")
        parent = providers[0].parent
        mapping = gatea.checkpoint_mapping(
            freeze, json.loads(providers[0].read_bytes()), plan["phase"], reference
        )

        def checkpoint_bytes():
            for name, entry in freeze["remote_files"].items():
                path = parent / name
                if (
                    path.is_symlink()
                    or path.stat().st_size != entry["bytes"]
                    or sha(path) != entry["sha256"]
                ):
                    raise ValueError("Frozen checkpoint dataset changed")
            recovery.authenticate_frozen_inventory(
                audit,
                plan["audit_sha256"],
                plan["phase"],
                parent,
                source / "root-inventory.json",
                plan["inventory_sha256"],
                native_paths=mapping,
            )

        checkpoint_bytes()
        original_source = legacy.unique_hash(
            Path(input_root).rglob("video-source-manifest.json"), legacy.SOURCE_SHA
        )
        cache_manifest = legacy.unique_hash(
            Path(input_root).rglob("research-features/manifest.json"), legacy.CACHE_SHA
        )
        protocol = legacy.unique_hash(
            Path(input_root).rglob("research-training-protocol.json"),
            legacy.PROTOCOL_SHA,
        )
        encoded, frozen = legacy.authenticate_cache(cache_manifest.parent, protocol)
        private = Path(tempfile.mkdtemp(prefix="video-interrupted-recovery-"))
        workspace = private / "original-scientific-source"
        legacy.restore_source(original_source, workspace)
        scientific_manifest = json.loads(original_source.read_bytes())

        def scientific_bytes():
            if sha(original_source) != legacy.SOURCE_SHA:
                raise ValueError("Original scientific manifest changed")
            for row in scientific_manifest["files"]:
                if sha(workspace / row["path"]) != row["sha256"]:
                    raise ValueError("Original scientific bytes changed")

        if platform.python_version() != legacy.PYTHON:
            raise ValueError("Original Python interpreter required")
        status.update(phase="package_bootstrap", python=platform.python_version())
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
        )
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        scientific_bytes()
        import imageio_ffmpeg
        import numpy as np
        import torch

        from src.models.moment_encoder import cpu_identity

        actual = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "pillow": importlib.metadata.version("pillow"),
            "imageio": importlib.metadata.version("imageio"),
            "imageio_ffmpeg": importlib.metadata.version("imageio-ffmpeg"),
            "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
            "torch_build_sha256": hashlib.sha256(
                torch.__config__.show().encode()
            ).hexdigest(),
            "cpu_identity": cpu_identity(),
        }
        legacy.check_environment(
            actual,
            encoded["encoding_identity"],
            output / "environment-diagnostic.json",
            plan["phase"],
        )
        # Definition imports do not load cache arrays or checkpoint tensors.
        scientific_bytes()
        from src.models import resumable_training as trainer

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        identity = {
            "schema": 1,
            "phase": plan["phase"],
            "protocol_sha256": legacy.PROTOCOL_SHA,
            "cache_manifest_sha256": frozen["source_manifest_sha256"],
            "implementation_sha256": frozen["implementation_sha256"],
            "coverage": frozen["coverage"],
            "encoding_identity": frozen["encoding_identity"],
            "runtime": trainer.runtime(),
            "spec_sha256": trainer.fingerprint(trainer.production_spec(plan["phase"])),
        }
        status.update(
            phase="original_training_identity_guard",
            actual_training_identity=identity,
            actual_identity_sha256=trainer.fingerprint(identity),
            expected_identity_sha256=reference["training_identity_sha256"],
        )
        record()
        if trainer.fingerprint(identity) != reference["training_identity_sha256"]:
            raise ValueError("Original complete training identity/runtime mismatch")

        def resume_callback(resume, resume_sha256, max_seconds):
            checkpoint_bytes()
            legacy.authenticate_cache(cache_manifest.parent, protocol)
            authenticate(package, EXPECTED_RECOVERY_PLAN_SHA256)
            scientific_bytes()
            # Only cloud copies, after the root inventory and every byte were authenticated.
            logical = private / "authenticated-resume"
            logical.mkdir()
            for name, flat in mapping.items():
                shutil.copyfile(parent / flat, logical / name)
                if sha(logical / name) != freeze["remote_files"][flat]["sha256"]:
                    raise ValueError("Resume copy differs")
            if resume != parent / mapping[reference["snapshot"]]:
                raise ValueError("Exact original flat snapshot required")
            status.update(phase="training", native_loading_started=True)
            record()
            return trainer.train(
                workspace / "data/processed/didemo-research-v5",
                cache_manifest.parent,
                output / ("research-" + plan["phase"]),
                protocol,
                legacy.PROTOCOL_SHA,
                plan["phase"],
                resume=logical / reference["snapshot"],
                resume_sha256=resume_sha256,
                max_seconds=legacy.remaining_seconds(started, time.monotonic()),
                checkpoint_every_seconds=300.0,
            )

        result = recovery.guarded_recover(
            source / "root-recovery-contract.json",
            plan["recovery_contract_sha256"],
            audit_path=audit,
            audit_sha256=plan["audit_sha256"],
            phase=plan["phase"],
            parent_phase_directory=parent,
            inventory_path=source / "root-inventory.json",
            inventory_sha256=plan["inventory_sha256"],
            training_identity=identity,
            resume_callback=resume_callback,
            service_started=started,
            native_paths=mapping,
        )
        checkpoint_bytes()
        legacy.authenticate_cache(cache_manifest.parent, protocol)
        authenticate(package, EXPECTED_RECOVERY_PLAN_SHA256)
        for name, entry in plan["files"].items():
            if sha(source / name) != entry["sha256"]:
                raise ValueError("Materialized transport source changed")
        scientific_bytes()
        if trainer.runtime() != identity["runtime"]:
            raise ValueError("Training runtime changed")
        training_output = output / ("research-" + plan["phase"])
        if result.get("status") == "paused_service_window":
            continuation = {
                "continuation_sha256": sha(training_output / "continuation.json"),
                "files": {
                    p.name: sha(p)
                    for p in sorted(training_output.iterdir())
                    if p.is_file() and p.suffix in (".pt", ".json")
                },
            }
            (training_output / "continuation-inventory.json").write_text(
                json.dumps(continuation, indent=2) + "\n"
            )
        complete = (
            result.get("completed_fits") == result.get("required_fits")
            and result.get("required_fits") == (9 if plan["phase"] == "phase1" else 12)
            and result.get("test_targets_used") is False
        )
        status.update(
            phase="service_returned",
            result=result,
            scientific_complete=complete,
            elapsed_seconds=time.monotonic() - started,
        )
        record()
    except Exception as error:
        status.update(
            phase="failed",
            scientific_complete=False,
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/interrupted-recovery")
