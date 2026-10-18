"""Cloud-only policy pack, followed by separately approved one-shot release."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

EXPECTED_MANIFEST_SHA256 = None


def remaining_budget(started, maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Required positive root max_seconds budget")
    if (
        type(started) not in (int, float)
        or not math.isfinite(started)
        or not 0 <= started <= time.monotonic()
    ):
        raise ValueError("Original service monotonic start required")
    value = maximum - (time.monotonic() - started)
    if value <= 0:
        raise TimeoutError("Original whole-service budget exhausted")
    return value


@contextmanager
def service_budget(started, maximum):
    """Root-bound whole-service deadline, including startup and native scoring."""

    def remaining():
        return remaining_budget(started, maximum)

    remaining()
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("Fresh Linux process timer required before native loading")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("Original whole-service budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    try:
        yield remaining
        remaining()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(paths, expected):
    matches = [path for path in paths if sha(path) == expected]
    if len(matches) != 1:
        raise ValueError("One exact actual mounted artifact required")
    return matches[0]


def checked_payload(manifest, runtime):
    if sha(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Reviewed cloud-test transport required")
    plan = json.loads(Path(manifest).read_bytes())
    if plan.get("schema") != "cloud_test_transport_v1":
        raise ValueError("Cloud test schema differs")
    for name, expected in plan["files"].items():
        if Path(name).name != name or sha(Path(manifest).parent / name) != expected:
            raise ValueError("Cloud test payload changed before imports")
    template = (
        Path(manifest).parent / "scripts__kaggle_research_test_cloud.py"
    ).read_bytes()
    token = b"EXPECTED_MANIFEST_SHA256" + b" = None"
    expected = template.replace(b"\r\n", b"\n").replace(
        token, ("EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256)).encode()
    )
    actual = Path(runtime).read_bytes().replace(b"\r\n", b"\n")
    if (
        b"\r" in expected
        or b"\r" in actual
        or template.count(token) != 1
        or actual != expected
    ):
        raise ValueError("Only exact reviewed LF/CRLF runtime permitted")
    return plan


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relocate_metadata(approval, payload):
    approval = copy.deepcopy(approval)
    approval["global_choice"]["report_path"] = str(payload / "global-report.json")
    approval["global_choice"]["dispatch_path"] = str(payload / "global-dispatch.json")
    for phase in ("phase1", "phase2"):
        approval["inventories"][phase]["path"] = str(
            payload / (phase + "-inventory.json")
        )
        approval["phase_reports"][phase]["path"] = str(
            payload / (phase + "-report.json")
        )
    if approval.get("schema") == "root_frozen_cloud_test_approval_v2":
        approval["production_package_policy"]["path"] = str(
            payload / "production-package-policy.json"
        )
    return approval


def checked_production_terminal(approval, terminal_freeze):
    if approval.get("schema") != "root_frozen_cloud_test_approval_v2":
        return None
    expected = "d7797f6b1fbbe2e94c1ed1e9dad4a1e5979aff71cbcc8a6908347c79bad36465"
    if (
        approval.get("production_proof_complete_freeze_required") is not True
        or approval.get("root_approval_issued") is not True
        or approval.get("proposal_only") is not False
        or approval.get("action") not in ("pack", "release")
        or approval["production_terminal_freeze"]["sha256"] != expected
        or sha(terminal_freeze) != expected
    ):
        raise ValueError("Actual completed root-frozen production proof required")
    frozen = json.loads(Path(terminal_freeze).read_bytes())
    if (
        frozen.get("schema") != "root_full_video_production_terminal_freeze_v2"
        or frozen.get("scientific_production_proof_complete") is not True
        or frozen.get("actual_all3132_query_ids_independently_replayed") is not True
        or frozen.get("runtime_exact_before_and_after") is not True
        or frozen.get("all_eight_prospective_package_pins_exact") is not True
        or frozen.get("all_seven_prospective_asset_hashes_exact") is not True
        or frozen.get("original_23_production_modules_current_unchanged") is not True
        or frozen.get("full_retained_media_scope") != 7454
        or frozen.get("actual_proof", {}).get("queries") != 3132
        or frozen["actual_proof"].get("same_vector_stable_all_split_topk_matches")
        != 3132
        or frozen["actual_proof"].get("selected_validation_metrics_replayed")
        is not True
        or frozen["actual_proof"].get("test_descriptions_or_targets_used") is not False
    ):
        raise ValueError("Full real production proof prerequisite differs")
    return frozen


def checked_production_packages(approval, package_policy):
    if approval.get("schema") != "root_frozen_cloud_test_approval_v2":
        return None
    row = approval["production_package_policy"]
    if (
        row["sha256"]
        != "963869875500fe123bd487b0aa5423e25a87f23b26ac62e19062bdc1a957e790"
        or sha(package_policy) != row["sha256"]
    ):
        raise ValueError("Actual root-frozen prospective package policy differs")
    frozen = json.loads(Path(package_policy).read_bytes())
    packages = approval["production_packages"]
    names = {
        "torch",
        "numpy",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
    }
    if (
        frozen.get("schema") != "root_prospective_video_assets_freeze_v2"
        or frozen.get("PASS") is not True
        or frozen["production_packages"] != packages
        or not isinstance(packages, dict)
        or set(packages) != names
        or frozen["selection_runtime"] != operation_runtime(approval)
    ):
        raise ValueError(
            "All eight frozen packages and actual selection runtime required"
        )
    for name, key in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
        ("pillow", "pillow"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ):
        if packages[name] != approval["environment"][key]:
            raise ValueError(
                "Frozen package pins differ from original encoder identity"
            )
    if not packages["torch"].endswith("+cpu"):
        raise ValueError("Exact CPU Torch binding required")
    return packages


def prepared_pack_directory(workspace, source_manifest, cache_manifest, approval):
    relative = "data/processed/didemo-research-v5/prepared/manifest.json"
    expected = "ad99d633cee842d81a237dcd95e75f39edcf1589392963a3f5310fbf0cbda09b"
    if (
        sha(source_manifest) != approval["source_manifest_sha256"]
        or sha(cache_manifest) != approval["cache_manifest_sha256"]
    ):
        raise ValueError(
            "Original source/cache metadata differs before prepared lookup"
        )
    source = json.loads(Path(source_manifest).read_bytes())
    encoded = json.loads(Path(cache_manifest).read_bytes())
    rows = [row for row in source["files"] if row["path"] == relative]
    path = Path(workspace) / relative
    if (
        len(rows) != 1
        or rows[0]["sha256"] != expected
        or encoded["source_manifest_sha256"] != expected
        or not path.is_file()
        or path.is_symlink()
        or Path(workspace).resolve() not in path.resolve().parents
        or any(parent.is_symlink() for parent in path.parents)
        or sha(path) != expected
    ):
        raise ValueError("Exact authenticated nested prepared manifest required")
    return path.parent


def check_runtime(training, approval):
    expected = operation_runtime(approval)
    if (
        type(expected.get("threads")) is not int
        or expected["threads"] != 2
        or type(expected.get("interop_threads")) is not int
        or expected["interop_threads"] != 2
        or expected.get("deterministic") is not True
    ):
        raise ValueError(
            "Only frozen two-thread/interoperability deterministic controls supported"
        )
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
    training.check_environment(
        actual,
        approval["environment"],
        "/kaggle/working/test-runtime-diagnostic.json",
        approval["action"],
    )
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != expected["interop_threads"]:
        torch.set_num_interop_threads(expected["interop_threads"])
    torch.use_deterministic_algorithms(True)
    from src.models.resumable_training import runtime

    if runtime() != operation_runtime(approval):
        raise ValueError(
            "Frozen numerical runtime differs before test policy operations"
        )


def operation_runtime(approval):
    return (
        approval["selection_runtime"]
        if approval.get("schema") == "root_frozen_cloud_test_approval_v2"
        else approval["training_runtime"]
    )


def main():
    started = time.monotonic()
    root = Path("/kaggle/input")
    manifest = unique(root.rglob("cloud-test-transport.json"), EXPECTED_MANIFEST_SHA256)
    plan = checked_payload(manifest, __file__)
    payload = manifest.parent
    transport = load_helper(
        payload / "scripts__research_test_transport.py", "reviewed_test_transport"
    )
    training = load_helper(
        payload / "scripts__kaggle_research_training.py", "reviewed_training_transport"
    )
    approval = relocate_metadata(
        transport.bound_json(payload / "approval.json", plan["approval_sha256"]),
        payload,
    )
    with service_budget(started, approval["max_seconds"]) as remaining:
        checker = None
        if approval.get("schema") == "root_frozen_cloud_test_approval_v2":
            checker = load_helper(
                payload / "src__models__recovered_policy.py",
                "reviewed_recovered_global_metadata",
            ).checked_global_choice
        transport.checked_global_choice(approval, checker)
        checked_production_terminal(
            approval, payload / "production-terminal-freeze.json"
        )
        packages = checked_production_packages(
            approval, payload / "production-package-policy.json"
        )
        workspace = Path("/kaggle/working/cloud-test-workspace")
        if approval["action"] == "pack":
            source_manifest = unique(
                root.rglob("video-source-manifest.json"), training.SOURCE_SHA
            )
            training.restore_source(source_manifest, workspace)
            cache_manifest = unique(
                root.rglob("research-features/manifest.json"), transport.CACHE_SHA
            )
            protocol = unique(
                root.rglob("research-training-protocol.json"), training.PROTOCOL_SHA
            )
            training.authenticate_cache(cache_manifest.parent, protocol)
            checkpoint = unique(
                root.rglob("selected.pt"),
                approval["global_choice"]["checkpoint_sha256"],
            )
            # No test JSON parser is used: retrieve bounded pinned raw bytes only.
            official_test = Path("/kaggle/working/official-test-raw.json")
            request = Request(
                "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/b6a555c8134581305d0ed4716fbc192860e0b88c/data/test_data.json",
                headers={"User-Agent": "didemo-cloud-test-pack/1"},
            )
            with urlopen(request, timeout=60) as response:
                content = response.read(15_000_001)
            if len(content) > 15_000_000:
                raise ValueError("Unexpected official metadata size")
            official_test.write_bytes(content)
            transport.checked_raw_test(official_test)
        elif approval["action"] == "release":
            stage_manifest = unique(
                root.rglob("cloud-test-stage.json"), approval["stage_sha256"]
            )
            frozen_stage = transport.restore_stage(
                stage_manifest,
                approval["stage_sha256"],
                approval["policy_sha256"],
                workspace,
            )
            if (
                frozen_stage["global_choice_checkpoint_sha256"]
                != approval["global_choice"]["checkpoint_sha256"]
                or frozen_stage["global_choice_report_sha256"]
                != approval["global_choice"]["report_sha256"]
                or frozen_stage["global_choice_dispatch_sha256"]
                != approval["global_choice"]["dispatch_sha256"]
                or frozen_stage["environment"] != approval["environment"]
                or transport.execution_runtime(frozen_stage)
                != operation_runtime(approval)
            ):
                raise ValueError(
                    "Actual frozen stage/global policy/runtime identities differ"
                )
        else:
            raise ValueError("Unknown approved action")
        for row in plan["sources"]:
            if row["path"].startswith("src/"):
                target = workspace / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if approval["action"] == "release" and (
                    not target.exists() or sha(target) != row["sha256"]
                ):
                    raise ValueError(
                        "Release source differs from root-frozen packed source"
                    )
                if approval["action"] == "pack":
                    shutil.copyfile(payload / row["flat_name"], target)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.57.6",
            "pillow==12.3.0",
            "imageio==2.37.4",
            "imageio-ffmpeg==0.6.0",
        ]
        if packages is not None:
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                *[name + "==" + packages[name] for name in sorted(packages)],
            ]
        subprocess.run(
            command,
            check=True,
            timeout=remaining(),
        )
        if (
            packages is not None
            and {name: importlib.metadata.version(name) for name in packages}
            != packages
        ):
            raise ValueError(
                "Actual eight production packages differ before native loading"
            )
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        check_runtime(training, approval)
        from src.models.moment_evaluation import freeze, release

        if approval["action"] == "pack":
            code = {name: workspace / name for name in plan["release_code"]}
            prepared = (
                prepared_pack_directory(
                    workspace, source_manifest, cache_manifest, approval
                )
                if approval.get("schema") == "root_frozen_cloud_test_approval_v2"
                else workspace / "data/processed/didemo-research-v5"
            )
            receipt = transport.pack(
                prepared,
                cache_manifest.parent,
                checkpoint,
                official_test,
                payload / "test-cohort.json",
                code,
                "/kaggle/working/cloud-packed-test-stage",
                approval,
                freeze,
            )
            Path("/kaggle/working/cloud-test-pack-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        else:
            release_operation = release
            if approval.get("schema") == "root_frozen_cloud_test_approval_v2":
                from src.models.release_evidence import release_with_rank_evidence

                release_operation = release_with_rank_evidence
            transport.release_once(
                workspace,
                approval,
                release_operation,
                "/kaggle/working/video-test-release-receipt.json",
            )
        checked_payload(manifest, __file__)
        for row in plan["sources"]:
            if (
                row["path"].startswith("src/")
                and sha(workspace / row["path"]) != row["sha256"]
            ):
                raise ValueError(
                    "Scientific source changed during cloud test operation"
                )
        check_runtime(training, approval)
        if approval["action"] == "pack":
            training.authenticate_cache(cache_manifest.parent, protocol)
            transport.checked_raw_test(official_test)
        else:
            transport.authenticate_stage(
                stage_manifest, approval["stage_sha256"], approval["policy_sha256"]
            )
            for row in frozen_stage["files"]:
                if sha(workspace / row["path"]) != row["sha256"]:
                    raise ValueError(
                        "Restored release inputs changed during evaluation"
                    )
        Path("/kaggle/working/cloud-test-execution.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "action": approval["action"],
                    "transport_sha256": EXPECTED_MANIFEST_SHA256,
                    "approval_sha256": plan["approval_sha256"],
                    "scientific_source_postflight_passed": True,
                    "runtime_postflight_passed": True,
                    "input_postflight_passed": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
