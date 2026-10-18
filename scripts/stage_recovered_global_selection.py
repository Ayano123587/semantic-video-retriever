"""Stage only after a separate actual recovered global-choice root approval."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_recovered_global_selection as worker
from scripts import kaggle_recovered_phase_inventory as inventory
from scripts.stage_recovered_phase_inventory import (
    source_names as inventory_source_names,
)
from src.models import cpu_portability as portability
from src.models.portable_training import source_closure

OWN = (
    "scripts/kaggle_recovered_global_selection.py",
    "scripts/stage_recovered_global_selection.py",
    "scripts/recovered_global_selection.py",
    "scripts/kaggle_research_training.py",
)


def source_names(root, original_manifest):
    return inventory_source_names(root, original_manifest) | source_closure(root, OWN)


def stage(
    output,
    *,
    approval_path,
    approval_sha256,
    hash_contract_path,
    inventory_freeze_path,
    inventory_paths,
    metadata_paths,
    phase_directories,
    source_review_path,
    source_review_sha256,
    dataset_id,
    kernel_id,
):
    root = Path(__file__).resolve().parents[1]
    require = portability.require
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean isolated reviewed source checkout required",
    )
    approval = portability.read_bound_json(approval_path, approval_sha256)
    require(
        approval.get("schema") == "root_recovered_global_selection_v1"
        and approval.get("proposal_only") is not True
        and approval["action"] == "choose_full_validation"
        and approval["training"] is False
        and approval["scoring"] is True
        and approval["test_annotations_used"] is False
        and approval["validation_queries"] == 3132
        and type(approval["max_seconds"]) is int
        and approval["max_seconds"] > 0,
        "Separate actual full-validation root approval required",
    )
    contract = portability.read_bound_json(
        hash_contract_path, approval["hash_contract_sha256"]
    )
    inventory.metadata_gate(metadata_paths, contract, phase_directories)
    require(
        approval["environment"] == contract["scientific"]["encoding_identity"],
        "Exact full original encoding environment required",
    )
    freeze = portability.read_bound_json(
        inventory_freeze_path, approval["inventory_freeze_sha256"]
    )
    require(
        freeze["schema"] == "root_complete_recovered_inventory_freeze_v1"
        and freeze["passed"] is True
        and freeze["hash_contract_sha256"] == approval["hash_contract_sha256"]
        and set(inventory_paths)
        == set(approval["inventories"])
        == {"phase1", "phase2"},
        "Actual both-phase root inventory freeze required",
    )
    for phase, path in inventory_paths.items():
        require(
            portability.sha(path)
            == freeze["inventories"][phase]["sha256"]
            == approval["inventories"][phase]["sha256"],
            "Actual frozen native inventory bytes differ",
        )
    require(
        isinstance(approval["environment"], dict)
        and isinstance(approval["selection_runtime"], dict),
        "Separate actual encoding and selection runtimes required",
    )
    original = json.loads(
        Path(metadata_paths["original-source-manifest.json"]).read_bytes()
    )
    names = source_names(root, original)
    hashes = {name: portability.sha(root / name) for name in sorted(names)}
    require(
        approval["source_sha256"] == hashes,
        "Actual full global source closure must be externally frozen",
    )
    for name in names:
        require(
            (root / name).read_bytes()
            == subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root),
            "Exact committed raw source required",
        )
    for row in original["files"]:
        if row["path"].startswith("src/") and row["path"].endswith(".py"):
            require(
                hashes[row["path"]] == row["sha256"],
                "All original scientific source must remain exact",
            )
    for name in OWN:
        require(
            subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset"),
            "Raw reviewed source attributes required",
        )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    review = portability.read_bound_json(source_review_path, source_review_sha256)
    require(
        review
        == {
            "schema": "recovered_global_source_review_v1",
            "head": head,
            "source_sha256": hashes,
            "approval_sha256": approval_sha256,
            "action": "choose_full_validation",
            "training": False,
            "scoring": True,
            "test_annotations_used": False,
        },
        "Actual root source/contract review required",
    )
    require(
        dataset_id != kernel_id
        and len(dataset_id.split("/")[-1]) <= 50
        and len(kernel_id.split("/")[-1]) <= 50,
        "Distinct bounded source/kernel identifiers required",
    )
    payload = {
        "request.json": Path(approval_path).read_bytes(),
        "hash-contract.json": Path(hash_contract_path).read_bytes(),
        "inventory-freeze.json": Path(inventory_freeze_path).read_bytes(),
        **{
            phase + "-inventory.json": Path(path).read_bytes()
            for phase, path in inventory_paths.items()
        },
        "source-review.json": Path(source_review_path).read_bytes(),
        "runner-template.py": (root / OWN[0]).read_bytes(),
    }
    payload.update(
        {
            "metadata/" + name: Path(path).read_bytes()
            for name, path in metadata_paths.items()
        }
    )
    payload.update({"project/" + name: (root / name).read_bytes() for name in names})
    plan = {
        "schema": "recovered_global_transport_v1",
        "max_seconds": approval["max_seconds"],
        "action": "choose_full_validation",
        "training": False,
        "scoring": True,
        "test_annotations_used": False,
        "request_sha256": approval_sha256,
        "source_review_sha256": source_review_sha256,
        "head": head,
        "files": {
            name: {
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for index, (name, raw) in enumerate(payload.items())
        },
    }
    output = Path(output)
    output.mkdir(exist_ok=False)
    package = output / "dataset"
    package.mkdir()
    for name, raw in payload.items():
        (package / plan["files"][name]["upload_name"]).write_bytes(raw)
    (package / "recovered-global-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = portability.sha(package / "recovered-global-plan.json")
    worker.authenticate(package, digest)
    template = payload["runner-template.py"].replace(b"\r\n", b"\n")
    token = b"EXPECTED_GLOBAL_PLAN_SHA256" + b" = None"
    require(
        template.count(token) == 1 and b"\r" not in template,
        "One exact canonical LF runner binding required",
    )
    rendered = template.replace(
        token, ("EXPECTED_GLOBAL_PLAN_SHA256 = " + repr(digest)).encode()
    )
    compile(rendered, "run.py", "exec")
    (output / "run.py").write_bytes(rendered)
    kernel = {
        "id": kernel_id,
        "title": kernel_id.split("/")[-1].replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [
            dataset_id,
            "aikyatansinha/portfolio-didemo-full-research-source-v5",
        ],
        "kernel_sources": [
            *inventory.PHASE_KERNELS.values(),
            "aikyatansinha/portfolio-didemo-full-research-encoding-v5/1",
        ],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    upload = output / "upload-metadata"
    upload.mkdir()
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Recovered global choice source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": digest,
        "runner_sha256": portability.sha(output / "run.py"),
        "payload_files": len(payload),
        "source_files": len(names),
        "uploaded": False,
        "dispatched": False,
    }
