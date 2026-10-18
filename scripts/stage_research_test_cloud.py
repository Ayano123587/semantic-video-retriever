"""Metadata-only host staging for cloud policy packing or one-shot test release."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_research_training as training
from scripts import research_test_transport as transport

RELEASE_CODE = [
    "src/__init__.py",
    "src/config.py",
    "src/data/__init__.py",
    "src/data/didemo.py",
    "src/data/test_cohort.py",
    "src/models/__init__.py",
    "src/models/encoder.py",
    "src/models/moments.py",
    "src/models/moment_training.py",
    "src/models/moment_evaluation.py",
    "src/models/moment_encoder.py",
    "src/models/resumable_training.py",
    "src/models/localization_followup.py",
    "src/models/positive_index.py",
    "src/models/research_followup.py",
    "src/models/encoding_progress.py",
    "src/data/research_receipts.py",
    "src/data/decoder_recovery.py",
    "src/data/decoder_preflight.py",
    "src/data/prepare_moments.py",
]


def stage(output, approval, approval_sha256, dataset_id):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh test transport required")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Exact reviewed clean checkout required")
    contract = transport.bound_json(approval, approval_sha256)
    if (
        contract.get("schema")
        not in (
            "root_frozen_cloud_test_approval_v1",
            "root_frozen_cloud_test_approval_v2",
        )
        or contract.get("action") not in ("pack", "release")
        or contract.get("cohort_sha256") != transport.COHORT_SHA
        or contract.get("official_test_sha256") != transport.OFFICIAL_SHA
        or contract.get("source_manifest_sha256") != training.SOURCE_SHA
        or contract.get("protocol_sha256") != training.PROTOCOL_SHA
    ):
        raise ValueError("Actual root-frozen scientific/test approval required")
    transport.checked_global_choice(contract)
    payloads = {"approval.json": Path(approval).read_bytes()}
    if contract.get("schema") == "root_frozen_cloud_test_approval_v2":
        from scripts.kaggle_research_test_cloud import (
            checked_production_packages,
            checked_production_terminal,
        )

        package_policy = Path(contract["production_package_policy"]["path"])
        checked_production_packages(contract, package_policy)
        payloads["production-package-policy.json"] = package_policy.read_bytes()
        terminal_freeze = Path(contract["production_terminal_freeze"]["path"])
        checked_production_terminal(contract, terminal_freeze)
        payloads["production-terminal-freeze.json"] = terminal_freeze.read_bytes()
    metadata = {
        "global-report.json": contract["global_choice"]["report_path"],
        "global-dispatch.json": contract["global_choice"]["dispatch_path"],
    }
    for phase in ("phase1", "phase2"):
        metadata[phase + "-inventory.json"] = contract["inventories"][phase]["path"]
        metadata[phase + "-report.json"] = contract["phase_reports"][phase]["path"]
    for name, path in metadata.items():
        payloads[name] = Path(path).read_bytes()
    if contract["action"] == "pack":
        cohort = Path(contract["cohort_path"])
        if (
            transport.sha(cohort) != transport.COHORT_SHA
            or len(contract["global_choice"]["kernel"].split("/")) != 3
        ):
            raise ValueError(
                "Actual prospective cohort and versioned global choice required"
            )
        payloads["test-cohort.json"] = cohort.read_bytes()
        mounts = [training.ENCODING_KERNEL, contract["global_choice"]["kernel"]]
    else:
        frozen_stage = transport.bound_json(
            contract["stage_metadata_path"], contract["stage_sha256"]
        )
        if (
            frozen_stage.get("frozen_policy_sha256") != contract["policy_sha256"]
            or frozen_stage.get("global_choice_checkpoint_sha256")
            != contract["global_choice"]["checkpoint_sha256"]
            or frozen_stage.get("global_choice_report_sha256")
            != contract["global_choice"]["report_sha256"]
            or frozen_stage.get("test_gold_parsed") is not False
            or len(contract["stage_kernel"].split("/")) != 3
        ):
            raise ValueError("Actual cloud-packed root-frozen policy/stage required")
        payloads["frozen-stage-metadata.json"] = Path(
            contract["stage_metadata_path"]
        ).read_bytes()
        mounts = [contract["stage_kernel"]]
    root = Path(__file__).resolve().parents[1]
    release_code = list(RELEASE_CODE)
    if contract.get("schema") == "root_frozen_cloud_test_approval_v2":
        release_code.append("src/models/recovered_policy.py")
        release_code.append("src/models/release_evidence.py")
    source_paths = [
        "scripts/research_test_transport.py",
        "scripts/kaggle_research_test_cloud.py",
        "scripts/stage_research_test_cloud.py",
        "scripts/kaggle_research_training.py",
        *release_code,
    ]
    source_rows = []
    for name in source_paths:
        content = (root / name).read_bytes()
        if content != subprocess.check_output(["git", "show", "HEAD:" + name]):
            raise ValueError("Exact reviewed source closure required")
        flat = name.replace("/", "__")
        payloads[flat] = content
        source_rows.append(
            {
                "path": name,
                "flat_name": flat,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    output.mkdir(parents=True)
    dataset = output / "dataset"
    dataset.mkdir()
    for name, content in payloads.items():
        (dataset / name).write_bytes(content)
    plan = {
        "schema": "cloud_test_transport_v1",
        "approval_sha256": approval_sha256,
        "sources": source_rows,
        "release_code": release_code,
        "files": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()
        },
    }
    manifest = dataset / "cloud-test-transport.json"
    manifest.write_bytes((json.dumps(plan, indent=2) + "\n").encode())
    digest = transport.sha(manifest)
    template = payloads["scripts__kaggle_research_test_cloud.py"]
    token = b"EXPECTED_MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One literal runner binding required")
    (output / "run.py").write_bytes(
        training.canonical_runner_bytes(template).replace(
            token, ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode()
        )
    )
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Private DiDeMo Cloud Test " + contract["action"],
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
                + "/portfolio-didemo-cloud-test-"
                + contract["action"]
                + "-v1",
                "title": "DiDeMo Cloud Test " + contract["action"],
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset_id]
                + ([training.SOURCE_DATASET] if contract["action"] == "pack" else []),
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
        "runner_sha256": transport.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "approval", "approval-sha256", "dataset-id"):
        parser.add_argument("--" + key, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
