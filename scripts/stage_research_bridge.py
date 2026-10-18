"""Prepare reviewed hash-only attestation or global-selection transport. No jobs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from scripts import research_selection_bridge as bridge
from scripts import kaggle_research_training as training


def stage(output, approval, approval_sha256, dataset_id):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh stage required")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Reviewed clean checkout required")
    contract = bridge.read_bound(approval, approval_sha256)
    if (
        contract.get("schema") != "root_frozen_research_bridge_inputs_v1"
        or contract.get("mode") not in ("inventory", "choose")
        or set(contract.get("phases", {})) != {"phase1", "phase2"}
    ):
        raise ValueError("Actual BOTH-phase root-frozen inputs required")
    scientific = contract["scientific"]
    for key, expected in {
        "source_manifest_sha256": training.SOURCE_SHA,
        "cache_manifest_sha256": training.CACHE_SHA,
        "protocol_sha256": training.PROTOCOL_SHA,
    }.items():
        if scientific.get(key) != expected:
            raise ValueError("Immutable executed scientific pins required")
    if contract["mode"] == "choose" and set(contract.get("inventories", {})) != {
        "phase1",
        "phase2",
    }:
        raise ValueError("Both actual root-frozen inventories required")
    for phase, row in contract["phases"].items():
        if row.get("phase") != phase or len(row.get("kernel", "").split("/")) != 3:
            raise ValueError("Exact versioned actual phase mount required")
        for key in (
            "dispatch_sha256",
            "report_sha256",
            "final_header_sha256",
            "transport_sha256",
        ):
            if (
                not isinstance(row.get(key), str)
                or len(row[key]) != 64
                or any(c not in "0123456789abcdef" for c in row[key])
            ):
                raise ValueError("Actual metadata fingerprints required")
    # Metadata-only host preflight: actual parent reports/dispatch/done header
    # must already exist. Learned files are hashed only in the cloud runner.
    for phase, row in contract["phases"].items():
        report_path = Path(row["report_path"])
        if report_path.name != "results.json":
            raise ValueError("Actual complete results metadata path required")
        bridge.phase_attestation(
            report_path.parent,
            row["dispatch_path"],
            row,
            scientific,
            hash_learned=False,
        )
    root = Path(__file__).resolve().parents[1]
    names = [
        "scripts/research_selection_bridge.py",
        "scripts/kaggle_research_bridge.py",
        "scripts/stage_research_bridge.py",
        "scripts/kaggle_research_training.py",
    ]
    payloads = {"approval.json": Path(approval).read_bytes()}
    for name in names:
        content = (root / name).read_bytes()
        if content != subprocess.check_output(["git", "show", "HEAD:" + name]):
            raise ValueError("Exact reviewed transport source required")
        payloads[Path(name).name] = content
    if contract["mode"] == "choose":
        for phase, row in contract["inventories"].items():
            value = bridge.read_bound(row["path"], row["sha256"])
            if (
                value.get("phase") != phase
                or value.get("learned_files_deserialized") is not False
            ):
                raise ValueError("Actual hash-only inventory required")
            payloads[phase + "-inventory.json"] = Path(row["path"]).read_bytes()
    output.mkdir(parents=True)
    dataset = output / "dataset"
    dataset.mkdir()
    for name, content in payloads.items():
        (dataset / name).write_bytes(content)
    plan = {
        "schema": "research_bridge_transport_v1",
        "approval_sha256": approval_sha256,
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payloads.items()
        },
    }
    manifest = dataset / "research-bridge-manifest.json"
    manifest.write_bytes((json.dumps(plan, indent=2) + "\n").encode())
    digest = bridge.sha(manifest)
    token = b"EXPECTED_MANIFEST_SHA256 = None"
    template = payloads["kaggle_research_bridge.py"]
    if template.count(token) != 1:
        raise ValueError("One manifest binding required")
    rendered = training.canonical_runner_bytes(template).replace(
        token, ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode()
    )
    (output / "run.py").write_bytes(rendered)
    owner = dataset_id.split("/")[0]
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Private DiDeMo Both Phase " + contract["mode"],
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": owner + "/portfolio-didemo-global-" + contract["mode"] + "-v1",
                "title": "DiDeMo Full Both Phase " + contract["mode"],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset_id, training.SOURCE_DATASET],
                "kernel_sources": [training.ENCODING_KERNEL]
                + [
                    contract["phases"][phase]["kernel"]
                    for phase in ("phase1", "phase2")
                ],
                "competition_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_sha256": digest,
        "runner_sha256": bridge.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "approval", "approval-sha256", "dataset-id"):
        parser.add_argument("--" + field, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
