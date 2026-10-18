"""Prospective v2 continuation: authenticate bytes before project/native imports."""

import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_PORTABLE_PLAN_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def remaining_budget(started, max_seconds):
    """Remaining wall time from the original service start, never a reset."""
    now = time.monotonic()
    require(
        isinstance(started, (int, float))
        and not isinstance(started, bool)
        and math.isfinite(started)
        and 0 <= started <= now,
        "Actual monotonic service start required",
    )
    require(
        type(max_seconds) is int and max_seconds in (1800, 36000),
        "Explicit approved whole-service budget required",
    )
    remaining = max_seconds - (now - started)
    require(remaining > 0, "Original service budget exhausted before bootstrap")
    return remaining


def safe_name(name, flat=False):
    value = Path(name)
    require(
        isinstance(name, str)
        and name
        and not value.is_absolute()
        and ".." not in value.parts
        and "\\" not in name
        and ":" not in name,
        "Safe relative payload path required",
    )
    if flat:
        require(value.name == name and "/" not in name, "Flat payload required")
    return name


def check_file(path, entry):
    require(
        path.is_file()
        and not path.is_symlink()
        and type(entry["bytes"]) is int
        and path.stat().st_size == entry["bytes"]
        and sha(path) == entry["sha256"],
        "Frozen raw bytes differ",
    )


def authenticate(package, expected):
    package = Path(package)
    require(
        sha(package / "portable-plan.json") == expected, "External plan SHA required"
    )
    plan = json.loads((package / "portable-plan.json").read_bytes())
    require(
        plan["schema"] == "portable_training_transport_v2"
        and plan["action"] in ("continue_training", "restore_only")
        and plan["max_seconds"]
        == (36000 if plan["action"] == "continue_training" else 1800)
        and plan["test_annotations_used"] is False,
        "Prospectively reviewed v2 transport required",
    )
    names = []
    for name, entry in plan["files"].items():
        safe_name(name)
        names.append(safe_name(entry["upload_name"], flat=True))
        check_file(package / entry["upload_name"], entry)
    require(len(names) == len(set(names)), "Unique transport filenames required")
    return plan


def unique(root, name, pin):
    found = [
        path for path in Path(root).rglob(name) if path.is_file() and sha(path) == pin
    ]
    require(len(found) == 1, "One exact authenticated mounted artifact required")
    result = found[0]
    require(
        sys.platform == "linux"
        and result.resolve().is_relative_to(Path("/kaggle/input")),
        "Actual Linux Kaggle input required",
    )
    for path in (result, *result.parents):
        require(not stat.S_ISLNK(path.lstat().st_mode), "Symlink input forbidden")
    return result


def raw_inputs(input_root, plan, source):
    """Pure stdlib full-byte gate, deliberately before helper or native imports."""
    freeze = json.loads((source / "checkpoint-freeze.json").read_bytes())
    provider = unique(
        input_root, "provider-provenance.json", freeze["provider_provenance_sha256"]
    )
    for name, entry in freeze["remote_files"].items():
        check_file(provider.parent / safe_name(name, flat=True), entry)
    original = unique(
        input_root, "video-source-manifest.json", plan["original_source_sha256"]
    )
    source_manifest = json.loads(original.read_bytes())
    for row in source_manifest["files"]:
        safe_name(row["path"])
        require(
            Path(row["path"]).name not in ("test.json", "test_data.json"),
            "No test annotations in training source",
        )
        path = original.parent / safe_name(row["flat_name"], flat=True)
        require(
            not path.is_symlink() and sha(path) == row["sha256"],
            "Original source bytes differ",
        )
    cache = unique(input_root, "manifest.json", plan["cache_manifest_sha256"])
    cached = json.loads(cache.read_bytes())
    require(
        cached["status"] == "complete"
        and cached["smoke_only"] is False
        and cached["test_text_or_targets_encoded"] is False,
        "Complete original no-test-text cache required",
    )
    for name, pin in cached["files"].items():
        path = cache.parent / safe_name(name, flat=True)
        require(
            name not in ("test-text.npy", "test-queries.json")
            and not path.is_symlink()
            and sha(path) == pin,
            "Original cache bytes differ",
        )
    protocol = unique(
        input_root, "research-training-protocol.json", plan["protocol_sha256"]
    )
    return provider, original, cache, protocol


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
        "stage": "bootstrap",
        "scientific_complete": False,
        "native_loading_started": False,
        "test_annotations_used": False,
    }

    def record():
        (output / "portable-status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    record()
    try:
        package = unique(
            input_root, "portable-plan.json", EXPECTED_PORTABLE_PLAN_SHA256
        ).parent
        plan = authenticate(package, EXPECTED_PORTABLE_PLAN_SHA256)
        source = Path(tempfile.mkdtemp(prefix="portable-transport-"))
        for name, entry in plan["files"].items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / entry["upload_name"]).read_bytes())
            check_file(target, entry)
        token = b"EXPECTED_PORTABLE_PLAN_SHA256" + b" = None"
        template = (source / "portable-runner.py").read_bytes()
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_PORTABLE_PLAN_SHA256 = "
                    + repr(EXPECTED_PORTABLE_PLAN_SHA256)
                ).encode(),
            ),
            "Executed portable runner differs",
        )
        # Even importing our stdlib helpers waits for every native sibling,
        # source byte, cache byte, protocol and externally bound evidence file.
        provider, original, cache, protocol = raw_inputs(input_root, plan, source)
        legacy = load_module(
            "portable_legacy_transport", source / "legacy-transport.py"
        )
        workspace = Path(tempfile.mkdtemp(prefix="portable-source-")) / "project"
        legacy.restore_source(original, workspace)
        manifest = json.loads((source / "source-manifest.json").read_bytes())
        for name, pin in manifest["files"].items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            supplied = source / "project" / name
            require(sha(supplied) == pin, "Externally frozen portable source changed")
            if target.exists():
                require(
                    sha(target) == pin,
                    "Original scientific source cannot be overwritten",
                )
            else:
                target.write_bytes(supplied.read_bytes())
        require(
            platform.python_version() == legacy.PYTHON, "Exact origin Python required"
        )
        status.update(stage="package_bootstrap", python=platform.python_version())
        record()
        bootstrap = subprocess.run(
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
            capture_output=True,
            check=False,
            text=True,
            timeout=remaining_budget(started, plan["max_seconds"]),
        )
        status.update(
            bootstrap_returncode=bootstrap.returncode,
            bootstrap_stdout=bootstrap.stdout[-2000:],
            bootstrap_stderr=bootstrap.stderr[-5000:],
        )
        record()
        bootstrap.check_returncode()
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        # Full post-bootstrap preimport byte authentication. Encoding provenance
        # remains Intel; the legacy CPU checker is never patched or bypassed.
        raw_inputs(input_root, plan, source)
        authenticate(package, EXPECTED_PORTABLE_PLAN_SHA256)
        from src.models import cpu_portability as portability
        from src.models import portable_training

        evidence_paths = {
            "freeze": source / "inspection-freeze.json",
            "report": source / "inspection-report.json",
            "amd": source / "amd-observation.json",
        }
        evidence = portability.load_evidence(
            evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
        )
        policy = portability.authenticate_policy(
            source / "policy.json", plan["policy_sha256"], evidence
        )
        service = portability.authenticate_service(
            source / "service.json",
            plan["service_sha256"],
            plan["policy_sha256"],
            plan["phase"],
            evidence,
            action=plan["action"],
        )
        portable_training.authenticate_source(
            source / "source-manifest.json",
            service,
            evidence["report"]["phases"][plan["phase"]]["origin_identity"],
        )
        resume = (
            provider.parent
            / service["parent_native_paths"][service["parent_snapshot_name"]]
        )
        portability.authenticate_parent(
            resume, source / "inventory.json", service, evidence, plan["phase"]
        )
        from src.models import resumable_training as trainer

        trainer.torch.set_num_threads(2)
        trainer.torch.use_deterministic_algorithms(True)
        actual = trainer.runtime()
        diagnostic = {
            "actual": actual,
            "expected_origin": evidence["report"]["phases"][plan["phase"]][
                "origin_identity"
            ]["runtime"],
            "allowed_cpu_descriptor_paths": policy["allowed_cpu_descriptor_paths"],
            "native_loading_started": False,
        }
        diagnostic["differing_paths"] = portability.differences(
            diagnostic["expected_origin"], actual
        )
        (output / "runtime-diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n"
        )
        portability.runtime_guard(actual, plan["phase"], policy, evidence)
        status.update(
            stage="native_restore_only"
            if plan["action"] == "restore_only"
            else "native_restore_and_training",
            actual_runtime_sha256=portability.fingerprint(actual),
            native_loading_started=True,
            origin_identity_sha256=service["origin_identity_sha256"],
        )
        record()
        arguments = {
            "policy_path": source / "policy.json",
            "policy_sha256": plan["policy_sha256"],
            "evidence_paths": evidence_paths,
            "service_contract_path": source / "service.json",
            "service_contract_sha256": plan["service_sha256"],
            "source_manifest_path": source / "source-manifest.json",
            "parent_inventory_path": source / "inventory.json",
            "resume": resume,
            "service_started": started,
        }
        if plan["action"] == "restore_only":
            result = portable_training.restore_only(
                cache.parent, protocol, plan["phase"], **arguments
            )
            (output / "restore-only-result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        else:
            result = portable_training.train(
                workspace / "data/processed/didemo-research-v5",
                cache.parent,
                output / ("research-" + plan["phase"]),
                protocol,
                plan["phase"],
                **arguments,
            )
        status["stage"] = "postflight"
        record()
        raw_inputs(input_root, plan, source)
        authenticate(package, EXPECTED_PORTABLE_PLAN_SHA256)
        for name, entry in plan["files"].items():
            check_file(source / name, entry)
        portable_training.authenticate_source(
            source / "source-manifest.json",
            service,
            evidence["report"]["phases"][plan["phase"]]["origin_identity"],
        )
        require(trainer.runtime() == actual, "Runtime changed during service")
        complete = (
            plan["action"] == "continue_training"
            and result.get("required_fits") == (9 if plan["phase"] == "phase1" else 12)
            and result.get("completed_fits") == result.get("required_fits")
            and result.get("test_targets_used") is False
        )
        status.update(
            stage="service_returned",
            result=result,
            scientific_complete=complete,
            restore_complete=plan["action"] == "restore_only"
            and result.get("restored") is True,
            elapsed_seconds=time.monotonic() - started,
            bitwise_equivalence_claimed=False,
        )
        record()
        return status
    except Exception as error:
        status.update(
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
            elapsed_seconds=time.monotonic() - started,
            scientific_complete=False,
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/portable-training")
