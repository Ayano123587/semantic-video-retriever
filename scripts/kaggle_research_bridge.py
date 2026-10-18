"""Hash-only complete-phase inventory, then separately authorized global choice."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(paths, expected):
    matches = [path for path in paths if sha(path) == expected]
    if len(matches) != 1:
        raise ValueError("One exact mounted artifact required")
    return matches[0]


def checked_payload(manifest, runtime):
    if sha(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Actual reviewed bridge manifest required")
    plan = json.loads(Path(manifest).read_bytes())
    if plan.get("schema") != "research_bridge_transport_v1":
        raise ValueError("Bridge transport schema differs")
    for name, expected in plan["files"].items():
        if Path(name).name != name or sha(Path(manifest).parent / name) != expected:
            raise ValueError("Bridge payload differs before imports")
    template = (Path(manifest).parent / "kaggle_research_bridge.py").read_bytes()
    token = b"EXPECTED_MANIFEST_SHA256" + b" = None"
    canonical = lambda raw: raw.replace(b"\r\n", b"\n")
    expected = canonical(template).replace(
        token, ("EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256)).encode()
    )
    raw = canonical(Path(runtime).read_bytes())
    if (
        b"\r" in raw
        or b"\r" in expected
        or template.count(token) != 1
        or raw != expected
    ):
        raise ValueError("Only exact reviewed LF/CRLF bridge runtime accepted")
    return plan


def main():
    root = Path("/kaggle/input")
    manifest = unique(
        root.rglob("research-bridge-manifest.json"), EXPECTED_MANIFEST_SHA256
    )
    transport = checked_payload(manifest, __file__)
    payload = manifest.parent
    sys.path.insert(0, str(payload))
    import research_selection_bridge as bridge
    import kaggle_research_training as training

    approval = bridge.read_bound(
        payload / "approval.json", transport["approval_sha256"]
    )
    scientific = approval["scientific"]
    for key, expected in {
        "source_manifest_sha256": training.SOURCE_SHA,
        "cache_manifest_sha256": training.CACHE_SHA,
        "protocol_sha256": training.PROTOCOL_SHA,
    }.items():
        if scientific.get(key) != expected:
            raise ValueError("Frozen scientific source differs")
    source = unique(root.rglob("video-source-manifest.json"), training.SOURCE_SHA)
    cache_manifest = unique(
        root.rglob("research-features/manifest.json"), training.CACHE_SHA
    )
    protocol = unique(
        root.rglob("research-training-protocol.json"), training.PROTOCOL_SHA
    )
    encoded, frozen = training.authenticate_cache(cache_manifest.parent, protocol)
    if (
        scientific["encoding_identity"] != encoded["encoding_identity"]
        or scientific["implementation_sha256"] != frozen["implementation_sha256"]
        or scientific["coverage"] != frozen["coverage"]
        or scientific["baseline"] != frozen["baseline"]
        or scientific["phase1_configs"] != frozen["phase1"]["configs"]
    ):
        raise ValueError("Full actual cache/protocol scientific identity differs")
    phase_inputs = {}
    for phase in ("phase1", "phase2"):
        contract = approval["phases"][phase]
        report = unique(root.rglob("results.json"), contract["report_sha256"])
        dispatch = unique(
            root.rglob("training-dispatch-result.json"), contract["dispatch_sha256"]
        )
        phase_inputs[phase] = {
            "directory": report.parent,
            "dispatch": dispatch,
            "contract": contract,
        }
    inventories = None
    if approval["mode"] == "choose":
        inventories = {
            phase: {
                "path": payload / (phase + "-inventory.json"),
                "sha256": approval["inventories"][phase]["sha256"],
            }
            for phase in ("phase1", "phase2")
        }
    elif approval["mode"] != "inventory":
        raise ValueError("Unknown reviewed bridge action")
    # No native arrays or checkpoints have been parsed above this barrier.
    attested = bridge.authenticate_both(phase_inputs, scientific, inventories)
    output = Path("/kaggle/working/both-phase-inventories")
    output.mkdir(exist_ok=False)
    for phase, inventory in attested.items():
        (output / (phase + "-inventory.json")).write_text(
            json.dumps(inventory, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if approval["mode"] == "inventory":
        return
    workspace = Path("/kaggle/working/global-selection-workspace")
    restored_manifest = training.restore_source(source, workspace)
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
        scientific["environment"],
        "/kaggle/working/selection-environment-diagnostic.json",
        "global",
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    from src.models.resumable_training import runtime

    if runtime() != scientific["training_runtime"]:
        raise ValueError("Frozen training numerical runtime differs before native load")
    from src.models.research_followup import choose

    def select():
        return choose(
            workspace / "data/processed/didemo-research-v5",
            cache_manifest.parent,
            phase_inputs["phase1"]["directory"],
            phase_inputs["phase2"]["directory"],
            "/kaggle/working/global-selection",
            protocol,
            training.PROTOCOL_SHA,
            approval["phases"]["phase1"]["report_sha256"],
            approval["phases"]["phase2"]["report_sha256"],
        )

    result = bridge.guarded_choose(phase_inputs, scientific, inventories, select)
    training.authenticate_cache(cache_manifest.parent, protocol)
    checked_payload(manifest, __file__)
    if sha(source) != training.SOURCE_SHA or sha(protocol) != training.PROTOCOL_SHA:
        raise ValueError("Scientific metadata changed during selection")
    for row in restored_manifest["files"]:
        if sha(workspace / row["path"]) != row["sha256"]:
            raise ValueError("Restored scientific source changed during selection")
    if runtime() != scientific["training_runtime"]:
        raise ValueError("Numerical runtime changed during global selection")
    receipt = {
        "schema": "both_phase_global_selection_dispatch_v1",
        "approval_sha256": transport["approval_sha256"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "inventory_sha256": {
            phase: inventories[phase]["sha256"] for phase in inventories
        },
        "validation_queries": 3132,
        "test_targets_used": False,
        "result": result,
    }
    Path("/kaggle/working/global-selection-dispatch.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
