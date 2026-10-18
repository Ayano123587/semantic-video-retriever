"""Stage independently reviewed complete validation replay; never dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import kaggle_phase1_validation_ranks as runner

OWN = (
    "scripts/phase1_validation_ranks.py",
    "scripts/kaggle_phase1_validation_ranks.py",
    "scripts/stage_phase1_validation_ranks.py",
    "scripts/kaggle_research_training.py",
    "src/models/cpu_portability.py",
    "scripts/video_query_proof.py",
)
METADATA = {
    "parent-freeze.json",
    "protocol.json",
    "root-policy.json",
    "inspection-freeze.json",
    "inspection-report.json",
    "amd-diagnostic.json",
}
PARENT_FREEZE_SHA = "3426b130f02b38563a581b17b61af70f6674aea461a700e395ea5b458acefb3e"


def validate_parent(value):
    runner.require(
        value["schema"] == "video_completed_phase1_validation_parent_freeze_v1"
        and value["artifact_revision"] == 2
        and value["action"] == "export_existing_full_validation_rankings_only"
        and value["parent_status"] == "COMPLETE"
        and value["parent"]["id"] == 139270426
        and value["parent"]["version"] == 1
        and value["validation_queries"] == 3132
        and value["candidate_moments"] == 15927
        and value["training"] is False
        and value["test_annotations_used"] is False
        and value["validation_replay_independently_complete"] is False,
        "Actual independently frozen complete phase1 parent required",
    )


def stage(
    output,
    metadata_paths,
    metadata_sha256,
    source_review_path,
    source_review_sha256,
    dataset_id,
    kernel_id,
):
    root = Path(__file__).resolve().parents[1]
    runner.require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=root),
        "Clean committed reviewed checkout required",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    runner.require(
        set(metadata_paths) == set(metadata_sha256) == METADATA,
        "Complete externally pinned input metadata required",
    )
    payload = {}
    for name, path in metadata_paths.items():
        raw = Path(path).read_bytes()
        runner.require(
            hashlib.sha256(raw).hexdigest() == metadata_sha256[name],
            "Actual root input bytes changed",
        )
        payload[name] = raw
    runner.require(
        metadata_sha256["parent-freeze.json"] == PARENT_FREEZE_SHA,
        "Actual root parent freeze required",
    )
    parent = json.loads(payload["parent-freeze.json"])
    validate_parent(parent)
    runner.require(
        metadata_sha256["protocol.json"] == parent["protocol_sha256"]
        and metadata_sha256["root-policy.json"] == parent["portable_policy_sha256"],
        "Original protocol and approved runtime policy required",
    )
    sources = {}
    for name in OWN:
        raw = (root / name).read_bytes()
        runner.require(
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root) == raw,
            "Exact committed raw source required",
        )
        attrs = subprocess.check_output(
            ["git", "check-attr", "text", "--", name], cwd=root, text=True
        )
        runner.require(
            attrs.strip().endswith(": unset"), "Explicit raw source attributes required"
        )
        sources[name] = hashlib.sha256(raw).hexdigest()
    raw = Path(source_review_path).read_bytes()
    runner.require(
        hashlib.sha256(raw).hexdigest() == source_review_sha256,
        "Actual source review SHA required",
    )
    runner.require(
        json.loads(raw)
        == {
            "schema": "phase1_validation_rank_source_review_v1",
            "head": head,
            "source_sha256": sources,
            "parent_freeze_sha256": PARENT_FREEZE_SHA,
        },
        "Actual independent root source review required",
    )
    payload["root-source-review.json"] = raw
    for name in OWN:
        mounted = "rank-runner.py" if name == OWN[1] else Path(name).name
        payload[mounted] = (root / name).read_bytes()
    runner.require(
        dataset_id != kernel_id, "Distinct source dataset and kernel identity required"
    )
    plan = {
        "schema": "phase1_validation_rank_transport_v1",
        "source_head": head,
        "parent_freeze_sha256": PARENT_FREEZE_SHA,
        "source_review_sha256": source_review_sha256,
        "training": False,
        "test_annotations_used": False,
        "max_seconds": 1800,
        "files": {
            name: {
                "upload_name": f"payload-{index:03d}" + Path(name).suffix,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
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
    raw = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    (package / "phase1-rank-plan.json").write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    template = payload["rank-runner.py"].replace(b"\r\n", b"\n")
    token = b"EXPECTED_PHASE1_RANK_PLAN_SHA256" + b" = None"
    runner.require(template.count(token) == 1, "One literal plan binding required")
    rendered = template.replace(
        token, ("EXPECTED_PHASE1_RANK_PLAN_SHA256 = " + repr(pin)).encode()
    )
    (output / "run.py").write_bytes(rendered)
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
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
                    parent["parent"]["canonical"] + "/1",
                    "aikyatansinha/portfolio-didemo-full-research-encoding-v5/1",
                ],
                "competition_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    uploads = output / "upload-metadata"
    uploads.mkdir()
    (uploads / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": "Phase 1 full validation rank source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "plan_sha256": pin,
        "runner_sha256": hashlib.sha256(rendered).hexdigest(),
        "payload_files": len(payload),
        "uploaded": False,
        "dispatched": False,
    }
