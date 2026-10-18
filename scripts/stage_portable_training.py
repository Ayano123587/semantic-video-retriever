"""Stage only an actually approved CPU-portable service; no network actions."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_portable_training as transport
from src.models import cpu_portability as portability
from src.models import portable_training

CHECKPOINT_FREEZE_SHA = (
    "58ce0269abad44dfcc5dbcf866f94e83c40e7824095d770f96ccbb0c40bde87c"
)
ORIGINAL_SOURCE_SHA = "265b18be07ec7ec87d60231175963eeafe63e3fc6547a019e206517e2bcc0a76"
OWN = (
    "scripts/kaggle_portable_training.py",
    "scripts/stage_portable_training.py",
    "scripts/kaggle_research_training.py",
    "src/models/cpu_portability.py",
    "src/models/portable_training.py",
)


def source_names(root, manifest):
    """Bind an initializer only when the actual package supplies one."""
    names = set(manifest["files"]) | set(OWN)
    if (Path(root) / "scripts/__init__.py").is_file():
        names.add("scripts/__init__.py")
    return names


def stage(
    output,
    *,
    phase,
    policy_path,
    policy_sha256,
    service_path,
    service_sha256,
    source_manifest_path,
    source_review_path,
    source_review_sha256,
    inventory_path,
    checkpoint_freeze_path,
    provider_path,
    inspection_freeze_path,
    inspection_report_path,
    amd_observation_path,
    dataset_id,
    kernel_id,
    action="continue_training",
):
    root = Path(__file__).resolve().parents[1]
    require = portability.require
    require(phase in ("phase1", "phase2"), "Explicit phase required")
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean committed reviewed source checkout required",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    evidence = portability.load_evidence(
        inspection_freeze_path, inspection_report_path, amd_observation_path
    )
    policy = portability.authenticate_policy(policy_path, policy_sha256, evidence)
    service = portability.authenticate_service(
        service_path, service_sha256, policy_sha256, phase, evidence, action=action
    )
    manifest = portable_training.authenticate_source(
        source_manifest_path,
        service,
        evidence["report"]["phases"][phase]["origin_identity"],
    )
    require(
        service["parent_schema"] == 1,
        "First portable service must use inspected original checkpoint; future v2 parent transport requires separate review",
    )
    freeze = portability.read_bound_json(checkpoint_freeze_path, CHECKPOINT_FREEZE_SHA)
    provider = portability.read_bound_json(
        provider_path, freeze["provider_provenance_sha256"]
    )
    require(
        freeze["private"] is True
        and freeze["version"] == 1
        and freeze["dataset_id"] == 12656520
        and len(freeze["remote_files"]) == 19,
        "Actual reviewed private checkpoint dataset required",
    )
    # Parent map is an externally frozen service field, not derived from a
    # guessed future mount or from the provider's directory spelling.
    inventory = portability.read_bound_json(
        inventory_path, service["parent_inventory_sha256"]
    )
    mapping = service["parent_native_paths"]
    require(
        set(mapping) == set(inventory["files"]),
        "Exact approved phase inventory mapping required",
    )
    for logical, flat in mapping.items():
        require(
            flat in freeze["remote_files"]
            and freeze["remote_files"][flat] == inventory["files"][logical],
            "Phase inventory differs from actual remote frozen bytes",
        )
    require(
        len(mapping) == (10 if phase == "phase1" else 8),
        "Exact original seed prefix coverage required",
    )
    original = evidence["report"]["phases"][phase]
    require(
        service["parent_checkpoint_sha256"] == original["reference"]["snapshot_sha256"]
        and service["parent_header_sha256"] == original["reference"]["header_sha256"]
        and service["parent_snapshot_name"] == original["reference"]["snapshot"],
        "Actual inspected native parent required",
    )
    # Provider provenance is preserved byte-for-byte; its full actual structure
    # is root-bound by the checkpoint dataset freeze, never invented here.
    require(isinstance(provider, dict), "Actual provider provenance required")
    names = source_names(root, manifest)
    source_hashes = {name: portability.sha(root / name) for name in sorted(names)}
    for name in names:
        require(
            subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root)
            == (root / name).read_bytes(),
            "Reviewed committed raw source differs",
        )
    for name in OWN:
        require(
            subprocess.check_output(
                ["git", "check-attr", "text", "--", name], cwd=root, text=True
            )
            .strip()
            .endswith(": unset"),
            "Explicit raw transport source attributes required",
        )
    review = portability.read_bound_json(source_review_path, source_review_sha256)
    require(
        review
        == {
            "schema": "portable_training_source_review_v2",
            "head": head,
            "source_sha256": source_hashes,
            "phase": phase,
            "action": action,
            "policy_sha256": policy_sha256,
            "service_sha256": service_sha256,
            "source_manifest_sha256": service["source_manifest_sha256"],
            "checkpoint_freeze_sha256": CHECKPOINT_FREEZE_SHA,
            "inspection_freeze_sha256": portability.INSPECTION_FREEZE_SHA,
            "inventory_sha256": service["parent_inventory_sha256"],
        },
        "Actual root source/policy/service review required",
    )
    require(
        dataset_id != kernel_id and dataset_id != freeze["dataset"],
        "Distinct source dataset and kernel identities required",
    )
    payload = {
        "policy.json": Path(policy_path).read_bytes(),
        "service.json": Path(service_path).read_bytes(),
        "source-manifest.json": Path(source_manifest_path).read_bytes(),
        "source-review.json": Path(source_review_path).read_bytes(),
        "inventory.json": Path(inventory_path).read_bytes(),
        "checkpoint-freeze.json": Path(checkpoint_freeze_path).read_bytes(),
        "provider-reviewed.json": Path(provider_path).read_bytes(),
        "inspection-freeze.json": Path(inspection_freeze_path).read_bytes(),
        "inspection-report.json": Path(inspection_report_path).read_bytes(),
        "amd-observation.json": Path(amd_observation_path).read_bytes(),
        "legacy-transport.py": (
            root / "scripts/kaggle_research_training.py"
        ).read_bytes(),
        "portable-runner.py": (root / "scripts/kaggle_portable_training.py")
        .read_bytes()
        .replace(b"\r\n", b"\n"),
    }
    payload.update({"project/" + name: (root / name).read_bytes() for name in names})
    plan = {
        "schema": "portable_training_transport_v2",
        "phase": phase,
        "action": action,
        "max_seconds": service["max_seconds"],
        "source_head": head,
        "source_review_sha256": source_review_sha256,
        "policy_sha256": policy_sha256,
        "service_sha256": service_sha256,
        "checkpoint_freeze_sha256": CHECKPOINT_FREEZE_SHA,
        "source_manifest_sha256": service["source_manifest_sha256"],
        "original_source_sha256": ORIGINAL_SOURCE_SHA,
        "cache_manifest_sha256": policy["cache_manifest_sha256"],
        "protocol_sha256": policy["origin_protocol_sha256"],
        "test_annotations_used": False,
        "files": {
            name: {
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for index, (name, raw) in enumerate(payload.items())
        },
    }
    # Nothing is created until every actual approval and immutable input passes.
    output = Path(output)
    output.mkdir(exist_ok=False)
    package = output / "dataset"
    package.mkdir()
    for name, raw in payload.items():
        (package / plan["files"][name]["upload_name"]).write_bytes(raw)
    (package / "portable-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    digest = portability.sha(package / "portable-plan.json")
    transport.authenticate(package, digest)
    token = b"EXPECTED_PORTABLE_PLAN_SHA256" + b" = None"
    require(
        payload["portable-runner.py"].count(token) == 1,
        "One literal plan binding required",
    )
    rendered = payload["portable-runner.py"].replace(
        token, ("EXPECTED_PORTABLE_PLAN_SHA256 = " + repr(digest)).encode()
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
            freeze["dataset"],
            "aikyatansinha/portfolio-video-source-v5",
        ],
        "kernel_sources": ["aikyatansinha/portfolio-didemo-full-encoding-v5/1"],
        "competition_sources": [],
    }
    # These must stay aligned with the original reviewed transport constants.
    from scripts import kaggle_research_training as legacy

    kernel["dataset_sources"][-1] = legacy.SOURCE_DATASET
    kernel["kernel_sources"] = [legacy.ENCODING_KERNEL]
    (output / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    upload = output / "upload-metadata"
    upload.mkdir()
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Frozen portable source " + dataset_id.split("/")[-1],
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": digest,
        "runner_sha256": portability.sha(output / "run.py"),
        "uploaded": False,
        "dispatched": False,
    }
