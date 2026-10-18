"""Metadata-only clean-source staging for a byte-only recovered inventory job."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_recovered_phase_inventory as worker
from src.models import cpu_portability as portability
from src.models.portable_training import source_closure

OWN = (
    "scripts/kaggle_recovered_phase_inventory.py",
    "scripts/stage_recovered_phase_inventory.py",
    "scripts/recovered_selection_bridge.py",
)


def source_names(root, original_manifest):
    """Include every original source and transitive actual local import."""
    root = Path(root)
    pending = list(OWN) + [
        row["path"]
        for row in original_manifest["files"]
        if row["path"].startswith("src/") and row["path"].endswith(".py")
    ]
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        pending.extend(source_closure(root, [name]) - found)
        for node in ast.walk(ast.parse((root / name).read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [
                    node.module,
                    *[node.module + "." + alias.name for alias in node.names],
                ]
            for module in modules:
                if not module.startswith(("src.", "scripts.")):
                    continue
                candidate = root.joinpath(*module.split(".")).with_suffix(".py")
                if candidate.is_file():
                    pending.append(candidate.relative_to(root).as_posix())
    if (root / "scripts/__init__.py").is_file():
        found.add("scripts/__init__.py")
    return found


def stage(
    output,
    *,
    approval_path,
    approval_sha256,
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
    worker.metadata_gate(metadata_paths, approval, phase_directories)
    original = json.loads(
        Path(metadata_paths["original-source-manifest.json"]).read_bytes()
    )
    names = source_names(root, original)
    hashes = {name: portability.sha(root / name) for name in sorted(names)}
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
            "schema": "recovered_inventory_source_review_v1",
            "head": head,
            "source_sha256": hashes,
            "approval_sha256": approval_sha256,
            "action": "hash_only",
            "training": False,
            "scoring": False,
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
        "approval.json": Path(approval_path).read_bytes(),
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
        "schema": "recovered_inventory_transport_v1",
        "max_seconds": 1800,
        "action": "hash_only",
        "training": False,
        "scoring": False,
        "test_annotations_used": False,
        "approval_sha256": approval_sha256,
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
    (package / "recovered-inventory-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = portability.sha(package / "recovered-inventory-plan.json")
    worker.authenticate(package, digest)
    template = payload["runner-template.py"].replace(b"\r\n", b"\n")
    token = b"EXPECTED_INVENTORY_PLAN_SHA256" + b" = None"
    require(
        template.count(token) == 1 and b"\r" not in template,
        "One exact canonical LF runner binding required",
    )
    rendered = template.replace(
        token, ("EXPECTED_INVENTORY_PLAN_SHA256 = " + repr(digest)).encode()
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
        "enable_internet": False,
        "dataset_sources": [dataset_id],
        "kernel_sources": list(worker.PHASE_KERNELS.values()),
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    upload = output / "upload-metadata"
    upload.mkdir()
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Recovered video byte inventory source",
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
