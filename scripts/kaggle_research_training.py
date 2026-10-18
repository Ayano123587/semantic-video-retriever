"""Authenticated CPU service window for the unchanged full-research trainer."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time

EXPECTED_TRANSPORT_SHA256 = None
SOURCE_SHA = "265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76"
CACHE_SHA = "ea73dced79e83cfc735652d6f8e702011d4e663c3fd9307ef010f66b9120a785"
PROTOCOL_SHA = "d81c9ef5aa881b04440806702312023135aea8eab9b6a4ba9c10fcab335a39a3"
ENCODING_KERNEL = "aikyatansinha/portfolio-didemo-full-research-encoding-v5/1"
SOURCE_DATASET = "aikyatansinha/portfolio-didemo-full-research-source-v5"
PYTHON = "3.13.15"  # Prospective service pin; encoding did not record Python.


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique_hash(paths, expected):
    matches = [p for p in paths if sha(p) == expected]
    if len(matches) != 1:
        raise ValueError("One exact authenticated input required")
    return matches[0]


def canonical_runner_bytes(content):
    """Transport only: allow CRLF to LF and reject every remaining CR byte."""
    canonical = content.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise ValueError("Lone CR is forbidden in runner transport")
    return canonical


def check_transport(path, rendered):
    if sha(path) != EXPECTED_TRANSPORT_SHA256:
        raise ValueError("Reviewed transport SHA required")
    plan = json.loads(Path(path).read_bytes())
    expected = dict(
        schema="full_research_training_transport_v2",
        runner_normalization="crlf_to_lf_only_reject_lone_cr_v1",
        source_manifest_sha256=SOURCE_SHA,
        cache_manifest_sha256=CACHE_SHA,
        protocol_sha256=PROTOCOL_SHA,
        encoding_kernel=ENCODING_KERNEL,
        source_dataset=SOURCE_DATASET,
        python=PYTHON,
        max_seconds=36000,
        test_annotations_used=False,
    )
    if any(
        type(plan.get(k)) is not type(v) or plan.get(k) != v
        for k, v in expected.items()
    ) or plan.get("phase") not in ("phase1", "phase2"):
        raise ValueError("Fixed source/cache/phase/service contract required")
    raw = Path(path).parent / "training-runner.py"
    if sha(raw) != plan.get("runner_template_sha256"):
        raise ValueError("Bound runner bytes changed")
    token = b"EXPECTED_TRANSPORT_SHA256" + b" = None"
    content = raw.read_bytes()
    canonical = canonical_runner_bytes(content)
    if hashlib.sha256(canonical).hexdigest() != plan.get(
        "runner_template_canonical_sha256"
    ):
        raise ValueError("Bound canonical runner bytes changed")
    if content.count(token) != 1 or canonical_runner_bytes(
        Path(rendered).read_bytes()
    ) != canonical.replace(
        token,
        ("EXPECTED_TRANSPORT_SHA256 = " + repr(EXPECTED_TRANSPORT_SHA256)).encode(),
    ):
        raise ValueError("Rendered runner differs from reviewed template")
    resume = plan.get("resume")
    if resume is not None:
        if (
            set(resume)
            != {
                "kernel_source",
                "snapshot",
                "snapshot_sha256",
                "receipt_sha256",
                "snapshot_receipt_sha256",
                "inventory_sha256",
            }
            or not isinstance(resume["kernel_source"], str)
            or len(resume["kernel_source"].split("/")) != 3
        ):
            raise ValueError("Explicit versioned continuation contract required")
        if Path(resume["snapshot"]).name != resume["snapshot"] or any(
            len(resume[k]) != 64
            for k in (
                "snapshot_sha256",
                "receipt_sha256",
                "snapshot_receipt_sha256",
                "inventory_sha256",
            )
        ):
            raise ValueError("Safe exact continuation identity required")
    return plan


def restore_source(path, workspace):
    if sha(path) != SOURCE_SHA:
        raise ValueError("Original executed source manifest required")
    manifest = json.loads(Path(path).read_bytes())
    contents = {}
    flat_seen = set()
    if manifest.get("source_commit") != "686a022d97c540f5f713d7dee5e64dcf7f6d4575":
        raise ValueError("Original executed source commit required")
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        flat = row["flat_name"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row["path"]
            or ":" in row["path"]
            or row["path"] in contents
            or flat in flat_seen
            or Path(flat).name != flat
            or "\\" in flat
            or ":" in flat
        ):
            raise ValueError("Unsafe or duplicate source transport path")
        flat_seen.add(flat)
        raw = (Path(path).parent / flat).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Original source/proof payload changed")
        if relative.name == "test_data.json" or relative.name == "test.json":
            raise ValueError("Training transport cannot contain test annotations")
        contents[row["path"]] = raw
    workspace = Path(workspace)
    workspace.mkdir(exist_ok=False)
    for name, raw in contents.items():
        target = workspace.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return manifest


def authenticate_cache(cache, protocol):
    """Hash native files without parsing arrays or any test annotations."""
    cache = Path(cache)
    if sha(cache / "manifest.json") != CACHE_SHA or sha(protocol) != PROTOCOL_SHA:
        raise ValueError("Actual complete encoding cache/protocol required")
    manifest = json.loads((cache / "manifest.json").read_bytes())
    frozen = json.loads(Path(protocol).read_bytes())
    if (
        manifest.get("status") != "complete"
        or manifest.get("smoke_only") is not False
        or manifest.get("test_text_or_targets_encoded") is not False
        or frozen.get("source_manifest_sha256") != CACHE_SHA
    ):
        raise ValueError("Complete no-test-label cache required")
    for name, expected in manifest["files"].items():
        if (
            Path(name).name != name
            or name in ("test-text.npy", "test-queries.json")
            or sha(cache / name) != expected
        ):
            raise ValueError("Changed or forbidden native cache file")
    if frozen.get("coverage") != {
        "candidate_count": 7592,
        "retained_media": 7454,
        "decoder_sources": 7464,
        "decoder_excluded": 10,
    }:
        raise ValueError("Full7454 cohort required")
    return manifest, frozen


def check_environment(actual, identity, diagnostic_path=None, phase=None):
    keys = [
        "torch",
        "numpy",
        "transformers",
        "pillow",
        "imageio",
        "imageio_ffmpeg",
        "ffmpeg",
        "torch_build_sha256",
        "cpu_identity",
    ]
    expected = {key: identity.get(key) for key in keys}
    expected["python"] = PYTHON
    differences = [key for key in sorted(expected) if actual.get(key) != expected[key]]
    if diagnostic_path is not None:
        diagnostic = dict(
            schema="research_training_environment_diagnostic_v1",
            phase=phase,
            source_manifest_sha256=SOURCE_SHA,
            cache_manifest_sha256=CACHE_SHA,
            protocol_sha256=PROTOCOL_SHA,
            transport_sha256=EXPECTED_TRANSPORT_SHA256,
            expected=expected,
            actual={key: actual.get(key) for key in expected},
            differing_keys=differences,
            status="pass" if not differences else "mismatch",
            training_module_imported=False,
            model_weights_loaded=False,
            training_started=False,
        )
        Path(diagnostic_path).write_text(
            json.dumps(diagnostic, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if actual.get("python") != PYTHON or any(
        actual.get(k) != identity.get(k) for k in keys
    ):
        raise ValueError(
            "Exact recorded numerical environment/CPU required; no compatibility override"
        )
    return actual


def remaining_seconds(started, now):
    remaining = 36000 - (now - started)
    if remaining <= 0:
        raise ValueError("Service deadline exhausted before training")
    return remaining


def authenticate_continuation_inventory(receipt, ref):
    inventory_path = receipt.parent / "continuation-inventory.json"
    if sha(inventory_path) != ref["inventory_sha256"]:
        raise ValueError("Externally bound continuation inventory differs")
    inventory = json.loads(inventory_path.read_bytes())
    if inventory.get("continuation_sha256") != ref["receipt_sha256"]:
        raise ValueError("Continuation inventory parent differs")
    for name, expected in inventory["files"].items():
        if Path(name).name != name or sha(receipt.parent / name) != expected:
            raise ValueError("Continuation file changed before any checkpoint load")
    if (
        ref["snapshot"] not in inventory["files"]
        or Path(ref["snapshot"]).with_suffix(".json").name not in inventory["files"]
    ):
        raise ValueError("Continuation inventory omits snapshot or sibling receipt")
    header = receipt.parent / Path(ref["snapshot"]).with_suffix(".json")
    if sha(header) != ref["snapshot_receipt_sha256"]:
        raise ValueError("Externally bound snapshot header changed")
    metadata = json.loads(header.read_bytes())
    completed = [
        name
        for name in inventory["files"]
        if name.startswith("config") and name.endswith(".pt")
    ]
    if (
        type(metadata.get("completed_seeds")) is not int
        or len(completed) != metadata["completed_seeds"]
    ):
        raise ValueError("Continuation inventory omits completed seed artifacts")


def main():
    started = time.monotonic()
    root = Path("/kaggle/input")
    transport = unique_hash(
        root.rglob("training-transport.json"), EXPECTED_TRANSPORT_SHA256
    )
    plan = check_transport(transport, Path(__file__))
    if platform.python_version() != PYTHON:
        raise ValueError("Prospective Python service pin differs")
    source = unique_hash(root.rglob("video-source-manifest.json"), SOURCE_SHA)
    cache_manifest = unique_hash(
        root.rglob("research-features/manifest.json"), CACHE_SHA
    )
    protocol = unique_hash(root.rglob("research-training-protocol.json"), PROTOCOL_SHA)
    cache = cache_manifest.parent
    encoded, frozen = authenticate_cache(cache, protocol)
    if encoded["encoding_identity"] != plan.get("encoding_identity"):
        raise ValueError("Transport environment binding differs")
    workspace = Path("/kaggle/working/research-training-workspace")
    restore_source(source, workspace)
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
    import torch
    import numpy as np
    import imageio_ffmpeg
    from src.models.moment_encoder import cpu_identity

    actual = dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        transformers=importlib.metadata.version("transformers"),
        pillow=importlib.metadata.version("pillow"),
        imageio=importlib.metadata.version("imageio"),
        imageio_ffmpeg=importlib.metadata.version("imageio-ffmpeg"),
        ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        cpu_identity=cpu_identity(),
    )
    check_environment(
        actual,
        encoded["encoding_identity"],
        diagnostic_path="/kaggle/working/training-environment-diagnostic.json",
        phase=plan["phase"],
    )
    from src.models.research_followup import checked_preparation

    preparation = workspace / "data/processed/didemo-research-v5"
    prepared, coverage = checked_preparation(preparation)
    if (
        prepared["counts"]["train"]["queries"] != 23202
        or prepared["counts"]["val"]["queries"] != 3132
    ):
        raise ValueError("All23202train/3132validation queries required")
    resume_path = None
    resume_sha = None
    if plan["resume"] is not None:
        ref = plan["resume"]
        receipt = unique_hash(root.rglob("continuation.json"), ref["receipt_sha256"])
        authenticate_continuation_inventory(receipt, ref)
        state = json.loads(receipt.read_bytes())
        if (
            state.get("snapshot") != ref["snapshot"]
            or state.get("snapshot_sha256") != ref["snapshot_sha256"]
        ):
            raise ValueError("Explicit continuation receipt differs")
        resume_path = receipt.parent / ref["snapshot"]
        resume_sha = ref["snapshot_sha256"]
        if sha(resume_path.with_suffix(".json")) != ref["snapshot_receipt_sha256"]:
            raise ValueError(
                "Continuation sibling receipt changed before checkpoint load"
            )
        if sha(resume_path) != resume_sha:
            raise ValueError("Continuation snapshot changed")
    from src.models.resumable_training import train

    output = Path("/kaggle/working") / ("research-" + plan["phase"])
    result = train(
        preparation,
        cache,
        output,
        protocol,
        PROTOCOL_SHA,
        plan["phase"],
        resume=resume_path,
        resume_sha256=resume_sha,
        max_seconds=remaining_seconds(started, time.monotonic()),
        checkpoint_every_seconds=300.0,
    )
    authenticate_cache(cache, protocol)
    if result.get("status") == "paused_service_window":
        inventory = dict(
            continuation_sha256=sha(output / "continuation.json"),
            files={
                p.name: sha(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.suffix in (".pt", ".json")
            },
        )
        (output / "continuation-inventory.json").write_text(
            json.dumps(inventory, indent=2, allow_nan=False) + "\n"
        )
    receipt = dict(
        transport_sha256=EXPECTED_TRANSPORT_SHA256,
        source_manifest_sha256=SOURCE_SHA,
        cache_manifest_sha256=CACHE_SHA,
        protocol_sha256=PROTOCOL_SHA,
        phase=plan["phase"],
        runner_normalization=plan["runner_normalization"],
        runner_runtime_raw_sha256=sha(Path(__file__)),
        runner_runtime_canonical_sha256=hashlib.sha256(
            canonical_runner_bytes(Path(__file__).read_bytes())
        ).hexdigest(),
        environment=actual,
        coverage=coverage,
        train_queries=23202,
        validation_queries=3132,
        test_annotations_used=False,
        service_elapsed_seconds=time.monotonic() - started,
        result=result,
    )
    Path("/kaggle/working/training-dispatch-result.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
