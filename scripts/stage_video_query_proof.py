"""Prepare prospective processor inventory or full-validation query proof only."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_research_training as training
from scripts import research_test_transport as test_transport
from src.models import research_search as search

PROJECT_SOURCE_FILES = (
    *search.SOURCE_FILES,
    "scripts/video_query_proof.py",
    "scripts/stage_video_query_proof.py",
    "scripts/kaggle_video_query_proof.py",
    "scripts/kaggle_research_training.py",
    "scripts/research_test_transport.py",
)


def metadata_contract(policy):
    if policy.get("schema") == search.RECOVERED_POLICY_SCHEMA:
        from src.models.recovered_policy import query_contract

        return query_contract(policy)
    return {
        "global_choice": {
            "report_path": policy["report_path"],
            "report_sha256": policy["report_sha256"],
            "dispatch_path": policy["dispatch_path"],
            "dispatch_sha256": policy["dispatch_sha256"],
            "checkpoint_sha256": policy["checkpoint_sha256"],
        },
        "inventories": policy["inventories"],
        "phase_reports": policy["phase_reports"],
        "protocol_sha256": policy["protocol_sha256"],
        "environment": policy["environment"],
        "training_runtime": policy["training_runtime"],
    }


def stage(output, approval, approval_sha256, dataset_id):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh query proof stage required")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Reviewed clean source required")
    contract = search.bound(approval, approval_sha256)
    if (
        contract.get("schema") != "root_frozen_video_query_proof_v1"
        or contract.get("action") not in ("inventory", "proof")
        or len(contract.get("global_kernel", "").split("/")) != 3
    ):
        raise ValueError("Actual frozen global choice and prospective action required")
    policy = search.bound(
        contract["query_policy_path"], contract["query_policy_sha256"]
    )
    if policy.get("schema") not in (
        "globally_selected_video_query_policy_v1",
        search.RECOVERED_POLICY_SCHEMA,
    ):
        raise ValueError("Actual approved production query policy required")
    test_transport.checked_global_choice(metadata_contract(policy))
    protocol = search.bound(policy["protocol_path"], policy["protocol_sha256"])
    if protocol.get("source_manifest_sha256") != policy["cache_manifest_sha256"]:
        raise ValueError("Actual full cache/protocol binding required")
    payloads = {
        "approval.json": Path(approval).read_bytes(),
        "query-policy.json": Path(contract["query_policy_path"]).read_bytes(),
        "protocol.json": Path(policy["protocol_path"]).read_bytes(),
        "global-report.json": Path(policy["report_path"]).read_bytes(),
        "global-dispatch.json": Path(policy["dispatch_path"]).read_bytes(),
    }
    for phase in ("phase1", "phase2"):
        payloads[phase + "-inventory.json"] = Path(
            policy["inventories"][phase]["path"]
        ).read_bytes()
        payloads[phase + "-report.json"] = Path(
            policy["phase_reports"][phase]["path"]
        ).read_bytes()
    mounts = [training.ENCODING_KERNEL, contract["global_kernel"]]
    if contract["action"] == "proof":
        inventory = search.bound(
            contract["asset_inventory_path"], contract["asset_inventory_sha256"]
        )
        if (
            inventory.get("schema")
            not in {
                "prospective_video_query_assets_v1",
                "prospective_video_asset_inventory_v2",
            }
            or inventory.get("historical_encoding_assets_attested") is not False
            or len(contract.get("asset_kernel", "").split("/")) != 3
        ):
            raise ValueError(
                "Actual separately frozen prospective asset inventory required"
            )
        if inventory.get("schema") == "prospective_video_asset_inventory_v2":
            packages = contract.get("production_packages")
            if (
                not isinstance(packages, dict)
                or set(packages)
                != {
                    "torch",
                    "numpy",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                    "pillow",
                    "imageio",
                    "imageio-ffmpeg",
                }
                or any(
                    not isinstance(value, str) or not value
                    for value in packages.values()
                )
            ):
                raise ValueError("All eight root-frozen production packages required")
            if (
                inventory.get("native_runtime_measured") is not False
                or inventory.get("native_loading") is not False
            ):
                raise ValueError("Prospective metadata-only inventory required")
        payloads["frozen-query-assets.json"] = Path(
            contract["asset_inventory_path"]
        ).read_bytes()
        mounts.append(contract["asset_kernel"])
    root = Path(__file__).resolve().parents[1]
    names = (
        *PROJECT_SOURCE_FILES,
        *[
            name
            for name in search.query_source_files(policy)
            if name not in PROJECT_SOURCE_FILES
        ],
    )
    rows = []
    for name in names:
        raw = (root / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", "HEAD:" + name]):
            raise ValueError("Exact reviewed proof source closure required")
        if (
            name in policy["source_files"]
            and hashlib.sha256(raw).hexdigest() != policy["source_files"][name]
        ):
            raise ValueError("Approved production source differs")
        flat = name.replace("/", "__")
        payloads[flat] = raw
        rows.append(
            {"path": name, "flat_name": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    output.mkdir(parents=True)
    dataset = output / "dataset"
    dataset.mkdir()
    for name, raw in payloads.items():
        (dataset / name).write_bytes(raw)
    plan = {
        "schema": "video_query_proof_transport_v1",
        "approval_sha256": approval_sha256,
        "query_policy_sha256": contract["query_policy_sha256"],
        "source_files": rows,
        "files": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()
        },
    }
    manifest = dataset / "video-query-proof-transport.json"
    manifest.write_bytes((json.dumps(plan, indent=2) + "\n").encode())
    digest = search.sha(manifest)
    template = payloads["scripts__kaggle_video_query_proof.py"]
    token = b"EXPECTED_MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One exact manifest literal required")
    (output / "run.py").write_bytes(
        training.canonical_runner_bytes(template).replace(
            token, ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode()
        )
    )
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Private Video Query " + contract["action"],
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
                "id": dataset_id.split("/")[0]
                + "/portfolio-video-query-"
                + contract["action"]
                + "-v1",
                "title": "Video Query " + contract["action"],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset_id, training.SOURCE_DATASET],
                "kernel_sources": mounts,
                "competition_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_sha256": digest,
        "runner_sha256": search.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "approval", "approval-sha256", "dataset-id"):
        parser.add_argument("--" + key, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
