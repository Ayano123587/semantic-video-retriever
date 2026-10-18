"""Prospective processor inventory, then separately frozen full-query replay."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = None


OBSERVATION_DIRECTORY = Path("/kaggle/working")
OBSERVATION = {"phase": "start", "native_loading_started": False}


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


def diagnose(phase, **fields):
    OBSERVATION.update(phase=phase, **fields)
    path = OBSERVATION_DIRECTORY / "video-query-proof-diagnostic.json"
    path.write_text(
        json.dumps(OBSERVATION, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def installed_packages():
    values = {}
    for name in (
        "torch",
        "numpy",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def bootstrap_packages(policy, remaining, production_packages=None):
    environment = policy["environment"]
    packages = [
        name + "==" + environment[key]
        for name, key in (
            ("transformers", "transformers"),
            ("pillow", "pillow"),
            ("imageio", "imageio"),
            ("imageio-ffmpeg", "imageio_ffmpeg"),
        )
    ]
    extra_arguments = []
    if production_packages is not None:
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
            not isinstance(production_packages, dict)
            or set(production_packages) != names
            or any(
                not isinstance(value, str) or not value
                for value in production_packages.values()
            )
        ):
            raise ValueError(
                "All eight separately root-frozen production packages required"
            )
        for name, key in (
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("transformers", "transformers"),
            ("pillow", "pillow"),
            ("imageio", "imageio"),
            ("imageio-ffmpeg", "imageio_ffmpeg"),
        ):
            if production_packages[name] != environment[key]:
                raise ValueError(
                    "Production package pins differ from original encoder identity"
                )
        if not production_packages["torch"].endswith("+cpu"):
            raise ValueError("Exact frozen CPU Torch wheel required")
        packages = [name + "==" + production_packages[name] for name in sorted(names)]
        extra_arguments = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    diagnose(
        "bootstrap",
        expected_runtime={
            "environment": environment,
            (
                "selection_runtime"
                if policy.get("schema")
                == "globally_selected_recovered_video_query_policy_v2"
                else "training_runtime"
            ): policy["selection_runtime"]
            if policy.get("schema")
            == "globally_selected_recovered_video_query_policy_v2"
            else policy["training_runtime"],
        },
        actual_packages_before_bootstrap=installed_packages(),
        requested_packages=packages,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *extra_arguments, *packages],
        check=True,
        timeout=remaining(),
    )
    diagnose("bootstrap_complete", actual_packages_after_bootstrap=installed_packages())


def guard_runtime(search, policy, phase, initialize=True):
    expected = {
        "environment": policy["environment"],
        (
            "selection_runtime"
            if policy.get("schema")
            == "globally_selected_recovered_video_query_policy_v2"
            else "training_runtime"
        ): policy["selection_runtime"]
        if policy.get("schema") == "globally_selected_recovered_video_query_policy_v2"
        else policy["training_runtime"],
    }
    if policy.get("schema") == "globally_selected_recovered_video_query_policy_v2":
        expected = search.frozen_runtime_state(policy)
    diagnose(
        phase + "_before_observation",
        expected_runtime=expected,
        actual_packages_at_runtime_check=installed_packages(),
    )
    before = search.runtime_state(policy)
    diagnose(phase + "_before_initialization", runtime_before=before)
    if initialize:
        search.initialize_runtime(policy)
    after = search.runtime_state(policy)
    diagnose(
        phase + "_before_comparison",
        runtime_after=after,
        runtime_differing_sections=[
            key for key in expected if after.get(key) != expected[key]
        ],
    )
    search.require_runtime(policy)
    diagnose(phase + "_passed")


def run_with_receipt(operation, output=None):
    """No success is written until the complete operation and postflight return."""
    global OBSERVATION_DIRECTORY
    OBSERVATION_DIRECTORY = (
        Path(output) if output is not None else Path("/kaggle/working")
    )
    OBSERVATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    success = OBSERVATION_DIRECTORY / "video-query-proof-execution.json"
    failure = OBSERVATION_DIRECTORY / "video-query-proof-failure.json"
    if success.exists() or failure.exists():
        raise ValueError("Fresh outcome paths required; never overwrite prior proof")
    OBSERVATION.clear()
    OBSERVATION.update(phase="start", native_loading_started=False)
    diagnose("start")
    try:
        receipt = operation()
        if not isinstance(receipt, dict) or receipt.get("status") != "complete":
            raise ValueError("Complete operation receipt required")
        success.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return receipt
    except Exception as exc:
        diagnostic = OBSERVATION_DIRECTORY / "video-query-proof-diagnostic.json"
        # Exception text and subprocess output may contain URLs or credentials;
        # record only safe type/return-code plus the bounded numerical inventory.
        receipt = {
            "schema": "video_query_proof_failure_v1",
            "status": "failed",
            "transport_sha256": EXPECTED_MANIFEST_SHA256,
            "phase": OBSERVATION["phase"],
            "error_type": type(exc).__name__,
            "returncode": getattr(exc, "returncode", None),
            "native_loading_started": OBSERVATION["native_loading_started"],
            "diagnostic_sha256": sha(diagnostic),
            "success_receipt_written": False,
        }
        failure.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        raise


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(paths, expected):
    found = [path for path in paths if sha(path) == expected]
    if len(found) != 1:
        raise ValueError("One actual mounted artifact required")
    return found[0]


def checked_payload(manifest, runtime):
    if sha(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Reviewed query proof transport required")
    plan = json.loads(Path(manifest).read_bytes())
    if plan.get("schema") != "video_query_proof_transport_v1":
        raise ValueError("Query proof schema differs")
    for name, expected in plan["files"].items():
        if Path(name).name != name or sha(Path(manifest).parent / name) != expected:
            raise ValueError("Query proof payload changed before imports")
    raw = (Path(manifest).parent / "scripts__kaggle_video_query_proof.py").read_bytes()
    token = b"EXPECTED_MANIFEST_SHA256" + b" = None"
    expected = raw.replace(b"\r\n", b"\n").replace(
        token, ("EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256)).encode()
    )
    actual = Path(runtime).read_bytes().replace(b"\r\n", b"\n")
    if (
        raw.count(token) != 1
        or b"\r" in expected
        or b"\r" in actual
        or actual != expected
    ):
        raise ValueError("Exact LF/CRLF query proof runner required")
    return plan


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relocate_policy(policy, payload):
    value = copy.deepcopy(policy)
    for key, name in (
        ("report_path", "global-report.json"),
        ("dispatch_path", "global-dispatch.json"),
        ("protocol_path", "protocol.json"),
    ):
        value[key] = str(payload / name)
    for phase in ("phase1", "phase2"):
        value["inventories"][phase]["path"] = str(payload / (phase + "-inventory.json"))
        value["phase_reports"][phase]["path"] = str(payload / (phase + "-report.json"))
    return value


def download_assets(model_id, revision, required, optional):
    from huggingface_hub import hf_hub_download, list_repo_files

    names = set(list_repo_files(model_id, revision=revision))
    if not required.issubset(names):
        raise ValueError("Pinned model lacks required processor assets")
    return {
        name: Path(hf_hub_download(model_id, name, revision=revision))
        for name in sorted((required | optional) & names)
    }


def frozen_inventory_path(payload, root, expected_sha256):
    """V2 consumes authenticated staged bytes, independent of provider basename."""
    staged = payload / "frozen-query-assets.json"
    if staged.is_file():
        if staged.is_symlink() or sha(staged) != expected_sha256:
            raise ValueError("Staged root-frozen asset inventory differs")
        if (
            json.loads(staged.read_bytes()).get("schema")
            == "prospective_video_asset_inventory_v2"
        ):
            return staged
    return unique(root.rglob("query-asset-inventory.json"), expected_sha256)


def prepared_directory(workspace, source_manifest, cache_manifest, policy):
    """The immutable source has two equal-hash aliases; bind the reviewed path."""
    relative = "data/processed/didemo-research-v5/prepared/manifest.json"
    if (
        sha(source_manifest) != policy["source_manifest_sha256"]
        or sha(cache_manifest) != policy["cache_manifest_sha256"]
    ):
        raise ValueError(
            "Original source/cache metadata differs before prepared lookup"
        )
    source = json.loads(source_manifest.read_bytes())
    encoded = json.loads(cache_manifest.read_bytes())
    rows = [row for row in source["files"] if row["path"] == relative]
    expected = policy["prepared_manifest_sha256"]
    if (
        len(rows) != 1
        or rows[0]["sha256"] != expected
        or encoded["source_manifest_sha256"] != expected
    ):
        raise ValueError("Exact reviewed prepared path/source/cache identity required")
    path = workspace / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or workspace.resolve() not in path.resolve().parents
        or any(parent.is_symlink() for parent in path.parents)
        or sha(path) != expected
    ):
        raise ValueError("Contained regular authenticated prepared manifest required")
    return path.parent


def main():
    started = time.monotonic()
    root = Path("/kaggle/input")
    manifest = unique(
        root.rglob("video-query-proof-transport.json"), EXPECTED_MANIFEST_SHA256
    )
    plan = checked_payload(manifest, __file__)
    payload = manifest.parent
    training = load_helper(
        payload / "scripts__kaggle_research_training.py", "query_training_transport"
    )
    approval_raw = payload / "approval.json"
    if sha(approval_raw) != plan["approval_sha256"]:
        raise ValueError("Actual root query-proof approval differs")
    approval = json.loads(approval_raw.read_bytes())
    with service_budget(started, approval["max_seconds"]) as remaining:
        policy_raw = payload / "query-policy.json"
        if (
            sha(policy_raw) != plan["query_policy_sha256"]
            or sha(policy_raw) != approval["query_policy_sha256"]
        ):
            raise ValueError("Actual frozen production query policy differs")
        original_policy = json.loads(policy_raw.read_bytes())
        policy = relocate_policy(original_policy, payload)
        diagnose(
            "authenticated_approval",
            transport_sha256=EXPECTED_MANIFEST_SHA256,
            root_query_policy_sha256=plan["query_policy_sha256"],
            action=approval["action"],
            expected_runtime={
                "environment": policy["environment"],
                (
                    "selection_runtime"
                    if policy.get("schema")
                    == "globally_selected_recovered_video_query_policy_v2"
                    else "training_runtime"
                ): policy["selection_runtime"]
                if policy.get("schema")
                == "globally_selected_recovered_video_query_policy_v2"
                else policy["training_runtime"],
            },
        )
        workspace = Path("/kaggle/working/query-proof-workspace")
        original_source = unique(
            root.rglob("video-source-manifest.json"), training.SOURCE_SHA
        )
        training.restore_source(original_source, workspace)
        for row in plan["source_files"]:
            target = workspace / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(payload / row["flat_name"], target)
        local_policy = workspace / "query-policy-local-paths.json"
        local_policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        local_policy_sha = sha(local_policy)
        bootstrap_packages(policy, remaining, approval.get("production_packages"))
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        from scripts import video_query_proof as proof
        from src.config import MODEL_ID, MODEL_REVISION
        from src.models import research_search as search

        cache_manifest = unique(
            root.rglob("research-features/manifest.json"),
            policy["cache_manifest_sha256"],
        )
        cache = cache_manifest.parent
        checkpoint = unique(root.rglob("selected.pt"), policy["checkpoint_sha256"])
        prepared = prepared_directory(
            workspace, original_source, cache_manifest, policy
        )
        receipts = (
            workspace / "data/processed/didemo-research-v5/resolved-receipts.jsonl"
        )
        # Original prepared manifest declares the root-reviewed receipt bytes.
        receipt_hash = json.loads((prepared / "manifest.json").read_bytes())[
            "receipt_sha256"
        ]
        receipts = unique(workspace.rglob("resolved-receipts.jsonl"), receipt_hash)
        diagnose("authenticate_full_query_inputs")
        checked = search.metadata_inputs(
            cache, prepared, receipts, checkpoint, local_policy, local_policy_sha
        )
        guard_runtime(search, policy, "startup_runtime")
        versions = proof.package_versions()
        frozen_assets = None
        if approval["action"] == "proof":
            frozen_path = frozen_inventory_path(
                payload, root, approval["asset_inventory_sha256"]
            )
            frozen_assets = search.bound(
                frozen_path, approval["asset_inventory_sha256"]
            )
            frozen_packages = (
                approval.get("production_packages")
                if frozen_assets.get("schema") == "prospective_video_asset_inventory_v2"
                else frozen_assets["packages"]
            )
            if versions != frozen_packages:
                raise ValueError(
                    "Prospective frozen processor dependency versions differ before asset/model loading"
                )
        elif approval["action"] != "inventory":
            raise ValueError("Unknown approved query proof action")
        assets = download_assets(
            MODEL_ID, MODEL_REVISION, proof.ASSET_REQUIRED, proof.ASSET_OPTIONAL
        )
        inventory = proof.asset_inventory(assets, versions, MODEL_ID, MODEL_REVISION)
        output = Path("/kaggle/working/query-assets")
        output.mkdir(exist_ok=False)
        files = output / "files"
        files.mkdir()
        for name, path in assets.items():
            shutil.copyfile(path, files / name)
        (output / "query-asset-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        result = None
        if approval["action"] == "proof":
            proof.verify_assets(
                assets,
                versions,
                frozen_assets,
                MODEL_ID,
                MODEL_REVISION,
                approval.get("production_packages"),
            )
            encoded = json.loads(cache_manifest.read_bytes())
            prepared_metadata = json.loads((prepared / "manifest.json").read_bytes())
            rows = proof.validation_rows(prepared, cache, encoded, prepared_metadata)
            diagnose("production_native_loading", native_loading_started=True)
            index = search.ResearchSearch(
                cache, prepared, receipts, checkpoint, local_policy, local_policy_sha
            )
            import numpy as np

            from src.models.encoder import CLIPEncoder

            encoder = CLIPEncoder("cpu")
            if any(
                encoder.receipt != part.encoder_receipt
                for part in index.indexes.values()
            ):
                raise ValueError("Actual query encoder receipt differs")
            cached = np.load(cache / "val-text.npy", allow_pickle=False)
            result = proof.replay(
                index,
                encoder,
                rows,
                cached,
                encoded["text_truncation"]["val"],
                checked[1],
                (checked[2], checked[3], checked[4]),
            )
        diagnose("postflight_inputs")
        checked_payload(manifest, __file__)
        for row in plan["source_files"]:
            if sha(workspace / row["path"]) != row["sha256"]:
                raise ValueError("Proof source changed during operation")
        search.metadata_inputs(
            cache, prepared, receipts, checkpoint, local_policy, local_policy_sha
        )
        guard_runtime(search, policy, "postflight_runtime", initialize=False)
        proof.verify_assets(
            assets, proof.package_versions(), inventory, MODEL_ID, MODEL_REVISION
        )
        if frozen_assets is not None:
            proof.verify_assets(
                assets,
                proof.package_versions(),
                frozen_assets,
                MODEL_ID,
                MODEL_REVISION,
                approval.get("production_packages"),
            )
        receipt = {
            "schema": "video_query_proof_execution_v1",
            "action": approval["action"],
            "status": "complete",
            "transport_sha256": EXPECTED_MANIFEST_SHA256,
            "root_query_policy_sha256": plan["query_policy_sha256"],
            "local_path_projection_sha256": local_policy_sha,
            "asset_inventory_sha256": sha(output / "query-asset-inventory.json"),
            "historical_tokenizer_inventory_claimed": False,
            "test_descriptions_or_targets_used": False,
            "result": result,
            "postflight_source_inputs_runtime_assets_passed": True,
        }
        return receipt


if __name__ == "__main__":
    run_with_receipt(main)
