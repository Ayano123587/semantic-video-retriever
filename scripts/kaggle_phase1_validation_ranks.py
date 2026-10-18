"""Authenticated phase1 validation replay transport; no training or test reads."""

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

EXPECTED_PHASE1_RANK_PLAN_SHA256 = None


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
        "Safe relative input path required",
    )
    return name


def check(path, entry):
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == entry["bytes"]
        and sha(path) == entry["sha256"],
        "Externally pinned raw bytes differ",
    )


def unique(root, name, pin):
    found = [p for p in Path(root).rglob(name) if p.is_file() and sha(p) == pin]
    require(len(found) == 1, "One exact mounted input required")
    path = found[0]
    require(
        sys.platform == "linux"
        and path.resolve().is_relative_to(Path("/kaggle/input"))
        and all(not stat.S_ISLNK(p.lstat().st_mode) for p in (path, *path.parents)),
        "Actual non-symlink Linux input required",
    )
    return path


def authenticate(package, pin):
    require(
        sha(package / "phase1-rank-plan.json") == pin,
        "External reviewed plan SHA required",
    )
    plan = json.loads((package / "phase1-rank-plan.json").read_bytes())
    require(
        plan["schema"] == "phase1_validation_rank_transport_v1"
        and plan["max_seconds"] == 1800
        and plan["training"] is False
        and plan["test_annotations_used"] is False,
        "Explicit val-only export transport required",
    )
    uploads = []
    for name, entry in plan["files"].items():
        safe(name)
        upload = safe(entry["upload_name"])
        require(Path(upload).name == upload, "Flat transport payload required")
        uploads.append(upload)
        check(package / upload, entry)
    require(len(uploads) == len(set(uploads)), "Unique mounted payloads required")
    return plan


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def resolve_prepared(workspace, source_manifest, cache_manifest, freeze):
    relative = safe(freeze["prepared_manifest_relative_path"])
    require(
        Path(relative).name == "manifest.json"
        and len(source_manifest["files"])
        == freeze["source_transport_total_files"]
        == 41,
        "Actual complete original source transport required",
    )
    matches = [row for row in source_manifest["files"] if row["path"] == relative]
    require(
        len(matches) == 1
        and matches[0]["sha256"]
        == cache_manifest["source_manifest_sha256"]
        == freeze["prepared_manifest_sha256"],
        "Exactly one externally bound prepared-manifest path required",
    )
    path = Path(workspace) / relative
    require(
        not path.is_symlink() and sha(path) == freeze["prepared_manifest_sha256"],
        "Restored exact prepared manifest bytes required",
    )
    return path.parent


def run(root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "stage": "start",
        "complete": False,
        "training": False,
        "test_annotations_used": False,
        "native_loading_started": False,
    }

    def record():
        (output / "status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False) + "\n"
        )

    def remaining():
        value = 1800 - (time.monotonic() - started)
        require(value > 0, "Original whole-service budget exhausted")
        return value

    record()
    try:
        package = unique(
            root, "phase1-rank-plan.json", EXPECTED_PHASE1_RANK_PLAN_SHA256
        ).parent
        plan = authenticate(package, EXPECTED_PHASE1_RANK_PLAN_SHA256)
        private = Path(tempfile.mkdtemp(prefix="phase1-val-rank-proof-"))
        source = private / "source"
        source.mkdir()
        for name, entry in plan["files"].items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / entry["upload_name"]).read_bytes())
        template = (source / "rank-runner.py").read_bytes().replace(b"\r\n", b"\n")
        token = b"EXPECTED_PHASE1_RANK_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_PHASE1_RANK_PLAN_SHA256 = "
                    + repr(EXPECTED_PHASE1_RANK_PLAN_SHA256)
                ).encode(),
            ),
            "Actual executed runner differs",
        )
        freeze = json.loads((source / "parent-freeze.json").read_bytes())
        require(
            sha(source / "parent-freeze.json") == plan["parent_freeze_sha256"]
            and freeze["schema"] == "video_completed_phase1_validation_parent_freeze_v1"
            and freeze["parent_status"] == "COMPLETE"
            and freeze["parent"]["id"] == 139270426
            and freeze["parent"]["version"] == 1,
            "Actual independently frozen complete parent required",
        )
        selected = unique(root, "selected.pt", freeze["selected_checkpoint_sha256"])
        results = selected.parent / "results.json"
        header = selected.parent / Path(freeze["final_header_path"]).name
        parent_status = selected.parent.parent / "portable-status.json"
        cache_manifest_path = unique(
            root, "manifest.json", freeze["cache_manifest_sha256"]
        )
        cache = cache_manifest_path.parent
        cache_manifest = json.loads(cache_manifest_path.read_bytes())
        original = unique(
            root, "video-source-manifest.json", freeze["source_manifest_sha256"]
        )
        source_manifest = json.loads(original.read_bytes())

        def raw_guard():
            authenticate(package, EXPECTED_PHASE1_RANK_PLAN_SHA256)
            for name, entry in plan["files"].items():
                check(source / name, entry)
            for path, pin in (
                (selected, freeze["selected_checkpoint_sha256"]),
                (results, freeze["results_sha256"]),
                (header, freeze["final_header_sha256"]),
                (parent_status, freeze["status_sha256"]),
                (cache_manifest_path, freeze["cache_manifest_sha256"]),
                (original, freeze["source_manifest_sha256"]),
                (source / "protocol.json", freeze["protocol_sha256"]),
            ):
                require(
                    not path.is_symlink() and sha(path) == pin,
                    "Full original parent/cache/protocol bytes changed",
                )
            for name, pin in cache_manifest["files"].items():
                path = cache / safe(name)
                require(
                    not path.is_symlink() and sha(path) == pin,
                    "Entire cache inventory must authenticate before native imports",
                )
            for row in source_manifest["files"]:
                safe(row["path"])
                require(
                    Path(row["path"]).name not in ("test.json", "test_data.json"),
                    "No test annotation payloads permitted",
                )
                flat = safe(row["flat_name"])
                require(
                    Path(flat).name == flat
                    and not (original.parent / flat).is_symlink()
                    and sha(original.parent / flat) == row["sha256"],
                    "Original source/prepared-cohort bytes changed",
                )

        raw_guard()
        legacy = module("phase1_val_legacy", source / "kaggle_research_training.py")
        workspace = private / "project"
        legacy.restore_source(original, workspace)
        prepared = resolve_prepared(workspace, source_manifest, cache_manifest, freeze)
        prospective = {
            "src/models/cpu_portability.py": "cpu_portability.py",
            "scripts/video_query_proof.py": "video_query_proof.py",
        }
        for name, upload in prospective.items():
            target = workspace / name
            require(
                not target.exists(),
                "Prospective helper must not overwrite original scientific source",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / upload).read_bytes())

        def guard():
            raw_guard()
            for row in source_manifest["files"]:
                require(
                    sha(workspace / row["path"]) == row["sha256"],
                    "Restored original source/cohort changed",
                )
            for name, upload in prospective.items():
                require(
                    sha(workspace / name) == sha(source / upload),
                    "Prospective helper source changed",
                )

        require(platform.python_version() == legacy.PYTHON, "Original Python required")
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
            timeout=remaining(),
        )
        expected = cache_manifest["encoding_identity"]
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
            expected_packages={key: expected[key] for key in packages},
        )
        record()
        require(
            all(value == expected[key] for key, value in packages.items()),
            "Exact original software required",
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
            source / "root-policy.json", freeze["portable_policy_sha256"], evidence
        )
        exact.torch.set_num_threads(2)
        exact.torch.use_deterministic_algorithms(True)
        actual = exact.runtime()
        status.update(
            stage="runtime_pre_native",
            actual_runtime=actual,
            origin_runtime=evidence["report"]["phases"]["phase1"]["origin_identity"][
                "runtime"
            ],
        )
        record()
        portability.runtime_guard(actual, "phase1", policy, evidence)
        require(
            hashlib.sha256(exact.torch.__config__.show().encode()).hexdigest()
            == expected["torch_build_sha256"]
            and imageio_ffmpeg.get_ffmpeg_version() == expected["ffmpeg"],
            "Original numerical native builds required",
        )
        actual_header = json.loads(header.read_bytes())
        actual_status = json.loads(parent_status.read_bytes())
        require(
            actual_header["stage"] == "done"
            and actual_header["completed_seeds"] == 9
            and actual_status["scientific_complete"] is True
            and actual_status["stage"] == "service_returned",
            "Actual complete parent state required",
        )
        guard()
        remaining()
        worker = module("phase1_rank_worker", source / "phase1_validation_ranks.py")
        status.update(stage="native_validation_export", native_loading_started=True)
        record()
        result = worker.export(
            cache,
            prepared,
            selected,
            results,
            output / "validation",
            source / "parent-freeze.json",
            plan["parent_freeze_sha256"],
            guard,
            exact.runtime,
            started=started,
            max_seconds=1800,
        )
        guard()
        require(exact.runtime() == actual, "Postflight runtime changed")
        remaining()
        status.update(
            stage="complete",
            complete=True,
            validation_manifest_sha256=sha(output / "validation/manifest.json"),
            queries=result["queries"],
            candidate_moments=result["candidate_moments"],
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
    run("/kaggle/input", "/kaggle/working/phase1-validation-rank-proof")
