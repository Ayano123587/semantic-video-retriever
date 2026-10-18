"""Separately approved full-validation chooser; no training or holdout reads."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

EXPECTED_GLOBAL_PLAN_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def authenticate(package, pin):
    raw = package / "recovered-global-plan.json"
    require(sha(raw) == pin, "Externally bound global plan required")
    plan = json.loads(raw.read_bytes())
    require(
        plan["schema"] == "recovered_global_transport_v1"
        and plan["action"] == "choose_full_validation"
        and plan["training"] is False
        and plan["scoring"] is True
        and plan["test_annotations_used"] is False
        and type(plan["max_seconds"]) is int
        and plan["max_seconds"] > 0,
        "Separate scoring-only transport required",
    )
    uploads = []
    for name, row in plan["files"].items():
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and "\\" not in name
            and ":" not in name,
            "Safe logical source path required",
        )
        upload = row["upload_name"]
        require(Path(upload).name == upload, "Flat source payload required")
        path = package / upload
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"],
            "Every source/metadata byte must authenticate before imports",
        )
        uploads.append(upload)
    require(len(set(uploads)) == len(uploads), "Unique source payload required")
    return plan


def authenticate_accounting(science, preparation, accounting_sha256):
    """Hash original real selection/journal bytes before any native operation."""
    base = Path(science).resolve()
    preparation = Path(preparation).resolve()
    require(
        base.is_dir() and preparation.is_relative_to(base / "data"),
        "Authenticated scientific data tree required",
    )
    audit_path = preparation / "source-accounting.json"
    require(
        not audit_path.is_symlink() and sha(audit_path) == accounting_sha256,
        "Frozen real source accounting required",
    )
    inputs = json.loads(audit_path.read_bytes()).get("input_sha256", {})
    require(
        isinstance(inputs, dict) and len(inputs) >= 2,
        "Original selection and complete source journal required",
    )
    resolved = set()
    for name, expected in inputs.items():
        require(
            isinstance(name, str) and name and ":" not in name,
            "Safe original accounting input key required",
        )
        relative = PurePosixPath(name.replace("\\", "/"))
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parts
            and relative.parts[0] == "data",
            "Accounting input must remain inside scientific data",
        )
        unnormalized = base / Path(*relative.parts)
        path = unnormalized.resolve()
        require(
            path.is_relative_to(base / "data")
            and path not in resolved
            and path.is_file()
            and all(not p.is_symlink() for p in (unnormalized, *unnormalized.parents)),
            "Unique regular contained accounting inputs required",
        )
        require(sha(path) == expected, "Original real accounting input bytes differ")
        resolved.add(path)
    return inputs


@contextmanager
def scientific_directory(science):
    previous = Path.cwd()
    base = Path(science).resolve()
    require(
        base.is_dir() and not Path(science).is_symlink(),
        "Authenticated scientific directory required",
    )
    os.chdir(base)
    try:
        require(
            Path.cwd().resolve() == base,
            "Exact scientific accounting working directory required",
        )
        yield
    finally:
        os.chdir(previous)


def run():
    started = time.monotonic()
    out = Path("/kaggle/working/recovered-global-status.json")
    state = {
        "status": "starting",
        "training": False,
        "scoring": True,
        "test_annotations_used": False,
        "native_loading_started": False,
    }

    def record():
        out.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    record()
    try:
        root = Path("/kaggle/input")
        candidates = [
            p
            for p in root.rglob("recovered-global-plan.json")
            if p.is_file() and sha(p) == EXPECTED_GLOBAL_PLAN_SHA256
        ]
        require(len(candidates) == 1, "One exact reviewed global transport required")
        package = candidates[0].parent
        plan = authenticate(package, EXPECTED_GLOBAL_PLAN_SHA256)
        workspace = Path(tempfile.mkdtemp(prefix="recovered-global-"))
        for name, row in plan["files"].items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / row["upload_name"]).read_bytes())
        template = (
            (workspace / "runner-template.py").read_bytes().replace(b"\r\n", b"\n")
        )
        token = b"EXPECTED_GLOBAL_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_GLOBAL_PLAN_SHA256 = " + repr(EXPECTED_GLOBAL_PLAN_SHA256)
                ).encode(),
            ),
            "Exact LF rendered global runner required",
        )
        sys.path.insert(0, str(workspace / "project"))
        from scripts import kaggle_recovered_phase_inventory as inventory
        from scripts import kaggle_research_training as legacy
        from scripts import recovered_global_selection as selection

        request_path = workspace / "request.json"
        request = json.loads(request_path.read_bytes())
        require(
            sha(request_path) == plan["request_sha256"]
            and request["max_seconds"] == plan["max_seconds"],
            "Actual global request required",
        )
        with selection.service_budget(started, request["max_seconds"]) as remaining:

            def raw_guard():
                remaining()
                authenticate(package, EXPECTED_GLOBAL_PLAN_SHA256)
                for name, row in plan["files"].items():
                    require(
                        sha(workspace / name) == row["sha256"],
                        "Materialized global source/metadata changed",
                    )

            raw_guard()
            contract_path = workspace / "hash-contract.json"
            contract = json.loads(contract_path.read_bytes())
            require(
                sha(contract_path) == request["hash_contract_sha256"],
                "Actual historical hash contract required",
            )
            freeze_path = workspace / "inventory-freeze.json"
            freeze = json.loads(freeze_path.read_bytes())
            require(
                sha(freeze_path) == request["inventory_freeze_sha256"]
                and freeze["passed"] is True
                and freeze["hash_contract_sha256"] == request["hash_contract_sha256"],
                "Actual root inventory freeze required",
            )
            paths = {
                name: workspace / "metadata" / name
                for name in contract["metadata_sha256"]
            }
            directories = {
                phase: inventory.mounted(
                    root, "results.json", row["report_sha256"]
                ).parent
                for phase, row in contract["phases"].items()
            }
            inventory_paths = {
                phase: workspace / (phase + "-inventory.json")
                for phase in ("phase1", "phase2")
            }
            for phase, path in inventory_paths.items():
                require(
                    sha(path)
                    == request["inventories"][phase]["sha256"]
                    == freeze["inventories"][phase]["sha256"],
                    "Actual externally frozen native inventory required",
                )
            selection.request_inputs(
                request_path,
                plan["request_sha256"],
                contract_path,
                paths,
                directories,
                inventory_paths,
            )
            source = inventory.mounted(
                root, "video-source-manifest.json", legacy.SOURCE_SHA
            )
            science = workspace / "scientific"
            legacy.restore_source(source, science)
            cache_manifest = inventory.mounted(root, "manifest.json", legacy.CACHE_SHA)
            protocol = inventory.mounted(
                root, "research-training-protocol.json", legacy.PROTOCOL_SHA
            )
            preparation = science / "data/processed/didemo-research-v5"
            frozen_protocol, _encoded = selection.raw_scientific_inputs(
                preparation,
                cache_manifest.parent,
                protocol,
                paths,
                contract["scientific"],
            )
            accounting_inputs = authenticate_accounting(
                science, preparation, frozen_protocol["accounting_sha256"]
            )
            state.update(accounting_input_sha256=accounting_inputs)
            environment = request["environment"]
            packages = [
                name + "==" + environment[key]
                for name, key in [
                    ("transformers", "transformers"),
                    ("pillow", "pillow"),
                    ("imageio", "imageio"),
                    ("imageio-ffmpeg", "imageio_ffmpeg"),
                ]
            ]
            state.update(
                status="bootstrap",
                expected_selection_runtime=request["selection_runtime"],
                expected_environment=environment,
            )
            record()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *packages],
                check=True,
                timeout=remaining(),
            )
            os.environ["OMP_NUM_THREADS"] = "2"
            os.environ["MKL_NUM_THREADS"] = "2"
            raw_guard()
            state.update(status="native_scoring", native_loading_started=True)
            record()
            # choose owns the same whole-service deadline; disarm this outer timer first.
        with scientific_directory(science):
            result = selection.choose(
                preparation,
                cache_manifest.parent,
                directories,
                "/kaggle/working/recovered-global-choice",
                protocol,
                request_path=request_path,
                request_sha256=plan["request_sha256"],
                hash_contract_path=contract_path,
                metadata_paths=paths,
                metadata=science / "data/external/didemo",
                service_started=started,
                inventory_paths=inventory_paths,
            )
        with selection.service_budget(started, request["max_seconds"]):
            authenticate_accounting(
                science, preparation, frozen_protocol["accounting_sha256"]
            )
            authenticate(package, EXPECTED_GLOBAL_PLAN_SHA256)
            for name, row in plan["files"].items():
                require(
                    sha(workspace / name) == row["sha256"],
                    "Global postflight materialized source changed",
                )
            state.update(
                status="complete",
                dispatch_sha256=sha(
                    Path(
                        "/kaggle/working/recovered-global-choice/recovered-global-dispatch.json"
                    )
                ),
                result=result,
            )
            record()
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
        )
        record()
        raise


if __name__ == "__main__":
    run()
