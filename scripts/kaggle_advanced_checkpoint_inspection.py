"""Byte-authenticated advanced parent inspection and restore only."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256 = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(name, flat=False):
    path = Path(name)
    require(
        isinstance(name, str)
        and name
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name,
        "Safe relative path required",
    )
    require(not flat or path.name == name, "Flat payload required")
    return name


def check(path, entry):
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == entry["bytes"]
        and sha(path) == entry["sha256"],
        "Externally frozen raw bytes differ",
    )


def unique(root, name, pin):
    matches = [
        path for path in Path(root).rglob(name) if path.is_file() and sha(path) == pin
    ]
    require(len(matches) == 1, "One exact mounted payload required")
    result = matches[0]
    require(
        sys.platform == "linux"
        and result.resolve().is_relative_to(Path("/kaggle/input")),
        "Actual Linux Kaggle input required",
    )
    require(
        all(
            not stat.S_ISLNK(path.lstat().st_mode) for path in (result, *result.parents)
        ),
        "No symlink mounts allowed",
    )
    return result


def authenticate(package, pin):
    require(
        sha(package / "advanced-inspection-plan.json") == pin,
        "External plan SHA required",
    )
    plan = json.loads((package / "advanced-inspection-plan.json").read_bytes())
    require(
        plan["schema"] == "advanced_phase2_native_inspection_transport_v1"
        and plan["training"] is False
        and plan["scoring"] is False
        and plan["test_annotations_used"] is False
        and plan["max_seconds"] == 1800,
        "Explicit inspection and restore transport required",
    )
    uploads = []
    for name, entry in plan["files"].items():
        safe(name)
        uploads.append(safe(entry["upload_name"], flat=True))
        check(package / entry["upload_name"], entry)
    require(len(set(uploads)) == len(uploads), "Unique payload names required")
    return plan


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "stage": "start",
        "complete": False,
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "native_loading_started": False,
    }

    def record():
        (output / "status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    def remaining():
        value = 1800 - (time.monotonic() - started)
        require(value > 0, "Whole service deadline exhausted")
        return value

    record()
    try:
        package = unique(
            root,
            "advanced-inspection-plan.json",
            EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256,
        ).parent
        plan = authenticate(package, EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256)
        private = Path(tempfile.mkdtemp(prefix="advanced-native-inspection-"))
        source = private / "source"
        source.mkdir()
        for name, entry in plan["files"].items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((package / entry["upload_name"]).read_bytes())
        template = (
            (source / "inspection-runner.py").read_bytes().replace(b"\r\n", b"\n")
        )
        token = b"EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256 = "
                    + repr(EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256)
                ).encode(),
            ),
            "Actual rendered runner differs",
        )
        contract = json.loads((source / "root-contract.json").read_bytes())
        freeze = json.loads((source / "checkpoint-freeze.json").read_bytes())
        require(
            contract["artifact_revision"] == 2
            and sha(source / "actual-progress-diagnostic.json")
            == contract["actual_progress_diagnostic_sha256"]
            and json.loads((source / "actual-progress-diagnostic.json").read_bytes())[
                "parent_progress"
            ]
            == contract["actual_progress"],
            "Actual independently reviewed completed-seed boundary required",
        )
        require(
            sha(source / "root-contract.json") == plan["contract_sha256"]
            and sha(source / "checkpoint-freeze.json")
            == contract["checkpoint_data_freeze_sha256"],
            "Actual root contract and dataset freeze required",
        )
        require(
            freeze["schema"] == "video_advanced_phase2_checkpoint_data_freeze_v1"
            and freeze["private"] is True
            and freeze["status"] == "READY"
            and freeze["version"] == 1
            and freeze["completed_seed_models"] == 10
            and len(freeze["files"]) == 13
            and freeze["files"] == contract["files"],
            "Full actual thirteen-file checkpoint freeze required",
        )
        parent = unique(
            root,
            "provider-provenance.json",
            freeze["files"]["provider-provenance.json"]["sha256"],
        ).parent
        original = unique(
            root, "video-source-manifest.json", contract["original_source_sha256"]
        )
        manifest = json.loads(original.read_bytes())

        def raw_guard():
            authenticate(package, EXPECTED_ADVANCED_INSPECTION_PLAN_SHA256)
            for name, entry in plan["files"].items():
                check(source / name, entry)
            require(
                sha(original) == contract["original_source_sha256"],
                "Original source manifest changed",
            )
            for row in manifest["files"]:
                require(
                    Path(safe(row["path"])).name not in ("test.json", "test_data.json"),
                    "No test data allowed",
                )
                require(
                    sha(original.parent / safe(row["flat_name"], flat=True))
                    == row["sha256"],
                    "Original source bytes changed",
                )
            for name, entry in freeze["files"].items():
                check(parent / safe(name, flat=True), entry)
            require(
                sha(source / "cache-manifest.json") == contract["cache_manifest_sha256"]
                and sha(source / "protocol.json") == contract["protocol_sha256"],
                "Original cache/protocol metadata required",
            )

        raw_guard()
        legacy = module(
            "advanced_original_transport", source / "kaggle_research_training.py"
        )
        workspace = private / "project"
        legacy.restore_source(original, workspace)
        core = workspace / "src/models/cpu_portability.py"
        require(
            not core.exists(),
            "Prospective module must not replace original scientific source",
        )
        core.write_bytes((source / "cpu_portability.py").read_bytes())

        def guard():
            raw_guard()
            for row in manifest["files"]:
                require(
                    sha(workspace / row["path"]) == row["sha256"],
                    "Restored scientific source changed",
                )
            require(
                sha(core) == sha(source / "cpu_portability.py"),
                "Prospective runtime guard changed",
            )

        require(
            platform.python_version() == legacy.PYTHON,
            "Original Python version required",
        )
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
            timeout=remaining(),
        )
        encoded = json.loads((source / "cache-manifest.json").read_bytes())[
            "encoding_identity"
        ]
        packages = {
            key: importlib.metadata.version(name)
            for name, key in (
                ("torch", "torch"),
                ("numpy", "numpy"),
                ("transformers", "transformers"),
                ("pillow", "pillow"),
                ("imageio", "imageio"),
                ("imageio-ffmpeg", "imageio_ffmpeg"),
            )
        }
        status.update(
            stage="software_preimport",
            packages=packages,
            expected_packages={key: encoded[key] for key in packages},
        )
        record()
        require(
            all(value == encoded[key] for key, value in packages.items()),
            "Exact numerical software required",
        )
        guard()
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        import imageio_ffmpeg

        from src.models import cpu_portability as portability
        from src.models import resumable_training as exact

        evidence = portability.load_evidence(
            source / "inspection-freeze.json",
            source / "inspection-report.json",
            source / "amd-diagnostic.json",
        )
        policy = portability.authenticate_policy(
            source / "root-policy.json", contract["policy_sha256"], evidence
        )
        exact.torch.set_num_threads(2)
        exact.torch.use_deterministic_algorithms(True)
        actual = exact.runtime()
        status.update(
            stage="runtime_pre_native",
            actual_runtime=actual,
            origin_runtime=evidence["report"]["phases"]["phase2"]["origin_identity"][
                "runtime"
            ],
        )
        record()
        portability.runtime_guard(actual, "phase2", policy, evidence)
        require(
            hashlib.sha256(exact.torch.__config__.show().encode()).hexdigest()
            == encoded["torch_build_sha256"]
            and imageio_ffmpeg.get_ffmpeg_version() == encoded["ffmpeg"],
            "Original native builds required",
        )
        guard()
        remaining()
        worker = module(
            "advanced_native_worker", source / "advanced_checkpoint_inspection.py"
        )
        status.update(
            stage="native_inspection_and_restore", native_loading_started=True
        )
        record()

        def parent_diagnostic(value):
            raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
            require(len(raw) < 4096, "Bounded scalar progress diagnostic required")
            (output / "parent-progress.json").write_bytes(raw)
            status["parent_progress"] = value
            record()

        inspection, restored = worker.inspect_restore(
            parent,
            contract,
            evidence,
            policy,
            guard,
            started=started,
            diagnostic=parent_diagnostic,
        )
        guard()
        require(exact.runtime() == actual, "Postflight runtime changed")
        remaining()
        for name, result in (
            ("inspection.json", inspection),
            ("restore-result.json", restored),
        ):
            raw = json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
            require(len(raw) < 2_000_000, "Bounded metadata result required")
            (output / name).write_bytes(raw)
        status.update(
            stage="complete",
            complete=True,
            restored=True,
            inspection_sha256=sha(output / "inspection.json"),
            restore_sha256=sha(output / "restore-result.json"),
        )
        record()
    except Exception as error:
        status.update(
            stage="failed",
            complete=False,
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/advanced-checkpoint-inspection")
