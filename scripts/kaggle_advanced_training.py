"""Initial actual checkpoint35 schema3 continuation transport; no scratch path."""

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

EXPECTED_ADVANCED_TRAIN_PLAN_SHA256 = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(name):
    require(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and "\\" not in name
        and ":" not in name,
        "Safe relative payload path required",
    )
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
    found = [p for p in Path(root).rglob(name) if p.is_file() and sha(p) == pin]
    require(len(found) == 1, "One exact actual mounted artifact required")
    path = found[0]
    require(
        sys.platform == "linux"
        and path.resolve().is_relative_to(Path("/kaggle/input"))
        and all(not stat.S_ISLNK(p.lstat().st_mode) for p in (path, *path.parents)),
        "Actual non-symlink Linux Kaggle input required",
    )
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def authenticate(package, pin):
    require(
        sha(package / "advanced-training-plan.json") == pin,
        "External reviewed plan required",
    )
    plan = json.loads((package / "advanced-training-plan.json").read_bytes())
    require(
        plan["schema"] == "advanced_training_transport_v3"
        and plan["action"] == "continue_training"
        and plan["max_seconds"] == 36000
        and plan["test_annotations_used"] is False,
        "Distinct actual reviewed schema3 training plan required",
    )
    uploads = []
    for name, entry in plan["files"].items():
        safe(name)
        upload = safe(entry["upload_name"])
        require(Path(upload).name == upload, "Flat transport payload required")
        uploads.append(upload)
        check(package / upload, entry)
    require(len(uploads) == len(set(uploads)), "Unique raw source payloads required")
    return plan


def remaining(started):
    value = 36000 - (time.monotonic() - started)
    require(value > 0, "Original whole-service startup budget exhausted")
    return value


def resolve_preparation(workspace, original, protocol, cache):
    matches = [
        row
        for row in original["files"]
        if row["sha256"] == protocol["accounting_sha256"]
        and Path(row["path"]).name == "source-accounting.json"
    ]
    require(len(matches) == 1, "One original accounting-bound preparation required")
    path = Path(workspace) / safe(matches[0]["path"])
    require(
        sha(path) == protocol["accounting_sha256"], "Original accounting bytes required"
    )
    preparation = path.parent
    require(
        sha(preparation / "prepared/manifest.json")
        == protocol["preparation_manifest_sha256"]
        == cache["source_manifest_sha256"],
        "Original full prepared/cache binding required",
    )
    return preparation


def run(root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "stage": "start",
        "scientific_complete": False,
        "training_authorized": True,
        "native_training_entry_started": False,
        "test_annotations_used": False,
    }

    def record():
        (output / "status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    record()
    try:
        package = unique(
            root, "advanced-training-plan.json", EXPECTED_ADVANCED_TRAIN_PLAN_SHA256
        ).parent
        plan = authenticate(package, EXPECTED_ADVANCED_TRAIN_PLAN_SHA256)
        private = Path(tempfile.mkdtemp(prefix="advanced-continuation-"))
        source = private / "source"
        source.mkdir()
        for name, entry in plan["files"].items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / entry["upload_name"]).read_bytes())
        template = (
            (source / "advanced-train-runner.py").read_bytes().replace(b"\r\n", b"\n")
        )
        token = b"EXPECTED_ADVANCED_TRAIN_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_ADVANCED_TRAIN_PLAN_SHA256 = "
                    + repr(EXPECTED_ADVANCED_TRAIN_PLAN_SHA256)
                ).encode(),
            ),
            "Actual rendered continuation runner differs",
        )
        service = json.loads((source / "service.json").read_bytes())
        require(
            sha(source / "service.json") == plan["service_sha256"]
            and service["parent_schema"] == 1,
            "This initial transport requires the actual independently restored original boundary",
        )
        freeze = json.loads((source / "checkpoint-freeze.json").read_bytes())
        require(
            sha(source / "checkpoint-freeze.json") == plan["checkpoint_freeze_sha256"]
            and len(freeze["files"]) == 13,
            "Actual full thirteen-file native dataset freeze required",
        )
        provider = unique(
            root,
            "provider-provenance.json",
            freeze["files"]["provider-provenance.json"]["sha256"],
        )
        parent = provider.parent
        original_path = unique(
            root, "video-source-manifest.json", plan["original_source_sha256"]
        )
        original = json.loads(original_path.read_bytes())
        cache_path = unique(root, "manifest.json", plan["cache_manifest_sha256"])
        cache = json.loads(cache_path.read_bytes())
        protocol = json.loads((source / "protocol.json").read_bytes())

        def raw_guard():
            authenticate(package, EXPECTED_ADVANCED_TRAIN_PLAN_SHA256)
            for name, entry in plan["files"].items():
                check(source / name, entry)
            require(
                sha(original_path) == plan["original_source_sha256"]
                and sha(cache_path) == plan["cache_manifest_sha256"]
                and sha(source / "protocol.json") == plan["protocol_sha256"],
                "Actual original source/cache/protocol changed",
            )
            for name, entry in freeze["files"].items():
                require(
                    Path(safe(name)).name == name, "Flat native dataset paths required"
                )
                check(parent / name, entry)
            for row in original["files"]:
                safe(row["path"])
                require(
                    Path(row["path"]).name not in ("test.json", "test_data.json"),
                    "No test annotations permitted",
                )
                flat = safe(row["flat_name"])
                require(
                    Path(flat).name == flat
                    and not (original_path.parent / flat).is_symlink()
                    and sha(original_path.parent / flat) == row["sha256"],
                    "All original source/prepared bytes required",
                )
            for name, pin in cache["files"].items():
                require(
                    Path(safe(name)).name == name
                    and not (cache_path.parent / name).is_symlink()
                    and sha(cache_path.parent / name) == pin,
                    "All native cache bytes must authenticate before imports",
                )

        raw_guard()
        legacy = module(
            "advanced_original_transport", source / "kaggle_research_training.py"
        )
        workspace = private / "project"
        legacy.restore_source(original_path, workspace)
        for name in (
            "cpu_portability.py",
            "portable_training.py",
            "advanced_parent.py",
            "advanced_training.py",
        ):
            target = workspace / "src/models" / name
            require(
                not target.exists(),
                "Prospective module must not overwrite original scientific source",
            )
            target.write_bytes((source / name).read_bytes())
        preparation = resolve_preparation(workspace, original, protocol, cache)

        def guard():
            raw_guard()
            for row in original["files"]:
                require(
                    sha(workspace / row["path"]) == row["sha256"],
                    "Restored original source/preparation changed",
                )
            for name in (
                "cpu_portability.py",
                "portable_training.py",
                "advanced_parent.py",
                "advanced_training.py",
            ):
                require(
                    sha(workspace / "src/models" / name) == sha(source / name),
                    "Prospective source changed",
                )

        require(
            platform.python_version() == legacy.PYTHON,
            "Original numerical Python required",
        )
        status.update(stage="bootstrap", expected_python=legacy.PYTHON)
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
            timeout=remaining(started),
        )
        encoded = cache["encoding_identity"]
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
            "Original numerical packages required",
        )
        guard()
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        import imageio_ffmpeg

        from src.models import advanced_parent as advanced
        from src.models import advanced_training
        from src.models import cpu_portability as portability
        from src.models import resumable_training as exact

        evidence_paths = {
            "freeze": source / "inspection-freeze.json",
            "report": source / "inspection-report.json",
            "amd": source / "amd-diagnostic.json",
        }
        advanced_paths = {
            "freeze": source / "advanced-freeze.json",
            "inspection": source / "advanced-inspection.json",
            "restore": source / "advanced-restore.json",
            "status": source / "advanced-status.json",
            "contract": source / "advanced-contract.json",
            "data": source / "checkpoint-freeze.json",
        }
        old = portability.load_evidence(
            evidence_paths["freeze"], evidence_paths["report"], evidence_paths["amd"]
        )
        context = advanced.load_context(advanced_paths, old)
        policy = portability.authenticate_policy(
            source / "root-policy.json", plan["policy_sha256"], old
        )
        advanced.authenticate_service(
            source / "service.json",
            plan["service_sha256"],
            plan["policy_sha256"],
            old,
            context,
        )
        exact.torch.set_num_threads(2)
        exact.torch.use_deterministic_algorithms(True)
        actual = exact.runtime()
        status.update(
            stage="runtime_pre_native",
            actual_runtime=actual,
            origin_runtime=context["report"]["origin_identity"]["runtime"],
        )
        record()
        portability.runtime_guard(actual, "phase2", policy, old)
        require(
            hashlib.sha256(exact.torch.__config__.show().encode()).hexdigest()
            == encoded["torch_build_sha256"]
            and imageio_ffmpeg.get_ffmpeg_version() == encoded["ffmpeg"],
            "Original native numerical builds required",
        )
        advanced_training.authenticate_source(
            source / "source-manifest.json",
            service,
            context["report"]["origin_identity"],
        )
        resume = (
            parent / service["parent_native_paths"][service["parent_snapshot_name"]]
        )
        advanced.authenticate_parent(
            resume, source / "parent-inventory.json", service, context
        )
        guard()
        remaining(started)
        status.update(stage="native_continuation", native_training_entry_started=True)
        record()
        result = advanced_training.train(
            preparation,
            cache_path.parent,
            output / "research-phase2",
            source / "protocol.json",
            policy_path=source / "root-policy.json",
            policy_sha256=plan["policy_sha256"],
            evidence_paths=evidence_paths,
            advanced_paths=advanced_paths,
            service_contract_path=source / "service.json",
            service_contract_sha256=plan["service_sha256"],
            source_manifest_path=source / "source-manifest.json",
            parent_inventory_path=source / "parent-inventory.json",
            resume=resume,
            service_started=started,
        )
        guard()
        advanced_training.authenticate_source(
            source / "source-manifest.json",
            service,
            context["report"]["origin_identity"],
        )
        require(exact.runtime() == actual, "Postflight numerical runtime changed")
        status.update(
            stage="service_returned",
            result=result,
            scientific_complete=result.get("completed_fits")
            == result.get("required_fits")
            == 12
            and len(result.get("policies", [])) == 16
            and result.get("test_targets_used") is False,
        )
        record()
    except Exception as error:
        status.update(
            stage="failed",
            scientific_complete=False,
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/advanced-training")
