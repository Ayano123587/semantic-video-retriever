"""Prepare a private authenticated training service window; never dispatch it."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts import kaggle_research_training as runner


def stage(
    output,
    phase,
    cache_manifest,
    protocol,
    dataset_id,
    resume_receipt=None,
    resume_receipt_sha256=None,
    resume_kernel=None,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh training stage required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed transport before private metadata reads")
    if phase not in ("phase1", "phase2"):
        raise ValueError("Fixed training phase required")
    if (
        runner.sha(cache_manifest) != runner.CACHE_SHA
        or runner.sha(protocol) != runner.PROTOCOL_SHA
    ):
        raise ValueError("Root-frozen actual cache/protocol metadata required")
    cache = json.loads(Path(cache_manifest).read_bytes())
    frozen = json.loads(Path(protocol).read_bytes())
    if (
        cache.get("status") != "complete"
        or cache.get("smoke_only") is not False
        or cache.get("test_text_or_targets_encoded") is not False
        or frozen.get("source_manifest_sha256") != runner.CACHE_SHA
    ):
        raise ValueError("Actual complete cache metadata required")
    root = Path(__file__).resolve().parents[1]
    for name, expected in frozen["implementation_sha256"].items():
        if runner.sha(root / "src" / name) != expected:
            raise ValueError("Executed scientific bytes changed")
    path = root / "scripts/kaggle_research_training.py"
    template = path.read_bytes()
    committed = subprocess.check_output(
        ["git", "show", "HEAD:scripts/kaggle_research_training.py"]
    )
    if template != committed:
        raise ValueError("Exact tracked runner required")
    raw_template_sha256 = hashlib.sha256(template).hexdigest()
    template = runner.canonical_runner_bytes(template)
    token = b"EXPECTED_TRANSPORT_SHA256" + b" = None"
    if template.count(token) != 1:
        raise ValueError("One literal runner binding required")
    resume = None
    if any(
        v is not None for v in (resume_receipt, resume_receipt_sha256, resume_kernel)
    ):
        if (
            not all(
                v is not None
                for v in (resume_receipt, resume_receipt_sha256, resume_kernel)
            )
            or runner.sha(resume_receipt) != resume_receipt_sha256
        ):
            raise ValueError("All exact reviewed continuation identities required")
        state = json.loads(Path(resume_receipt).read_bytes())
        if state.get("status") != "paused_service_window" or Path(
            state.get("snapshot", "")
        ).name != state.get("snapshot"):
            raise ValueError("Actual safe paused continuation required")
        if (
            len(resume_kernel.split("/")) != 3
            or len(state.get("snapshot_sha256", "")) != 64
        ):
            raise ValueError("Versioned kernel and snapshot SHA required")
        inventory_path = Path(resume_receipt).parent / "continuation-inventory.json"
        inventory = json.loads(inventory_path.read_bytes())
        if inventory.get("continuation_sha256") != resume_receipt_sha256:
            raise ValueError("Continuation inventory receipt differs")
        snapshot_receipt = Path(resume_receipt).parent / Path(
            state["snapshot"]
        ).with_suffix(".json")
        snapshot_state = json.loads(snapshot_receipt.read_bytes())
        if snapshot_state.get("snapshot_sha256") != state["snapshot_sha256"]:
            raise ValueError("Snapshot sibling receipt differs")
        resume = dict(
            inventory_sha256=runner.sha(inventory_path),
            snapshot_receipt_sha256=runner.sha(snapshot_receipt),
            kernel_source=resume_kernel,
            snapshot=state["snapshot"],
            snapshot_sha256=state["snapshot_sha256"],
            receipt_sha256=resume_receipt_sha256,
        )
    plan = dict(
        schema="full_research_training_transport_v2",
        runner_normalization="crlf_to_lf_only_reject_lone_cr_v1",
        tracked_runner_raw_sha256=raw_template_sha256,
        runner_template_canonical_sha256=hashlib.sha256(template).hexdigest(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_manifest_sha256=runner.SOURCE_SHA,
        cache_manifest_sha256=runner.CACHE_SHA,
        protocol_sha256=runner.PROTOCOL_SHA,
        encoding_kernel=runner.ENCODING_KERNEL,
        source_dataset=runner.SOURCE_DATASET,
        python=runner.PYTHON,
        python_pin_scope="prospective service pin; encoding did not record Python",
        phase=phase,
        max_seconds=36000,
        test_annotations_used=False,
        runner_template_sha256=hashlib.sha256(template).hexdigest(),
        encoding_identity=cache["encoding_identity"],
        resume=resume,
    )
    output.mkdir(parents=True)
    dataset = output / "dataset"
    dataset.mkdir()
    (dataset / "training-runner.py").write_bytes(template)
    manifest = dataset / "training-transport.json"
    manifest.write_bytes((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode())
    digest = runner.sha(manifest)
    (output / "run.py").write_bytes(
        template.replace(
            token, ("EXPECTED_TRANSPORT_SHA256 = " + repr(digest)).encode()
        )
    )
    owner = dataset_id.split("/")[0]
    title = f"Portfolio DiDeMo Full Research {phase.title()} V1"
    metadata = dict(
        id=owner + f"/portfolio-didemo-full-research-{phase}-v1",
        title=title,
        code_file="run.py",
        language="python",
        kernel_type="script",
        is_private=True,
        enable_gpu=False,
        enable_internet=True,
        dataset_sources=[dataset_id, runner.SOURCE_DATASET],
        kernel_sources=[runner.ENCODING_KERNEL] + ([resume_kernel] if resume else []),
        competition_sources=[],
    )
    (dataset / "dataset-metadata.json").write_bytes(
        (
            json.dumps(
                dict(
                    id=dataset_id,
                    title=f"Private DiDeMo {phase} Training Transport",
                    licenses=[dict(name="other")],
                ),
                indent=2,
            )
            + "\n"
        ).encode()
    )
    (output / "kernel-metadata.json").write_bytes(
        (json.dumps(metadata, indent=2) + "\n").encode()
    )
    return dict(
        phase=phase,
        transport_sha256=digest,
        runner_sha256=runner.sha(output / "run.py"),
        uploaded=False,
        dispatched=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["output", "phase", "cache-manifest", "protocol", "dataset-id"]:
        p.add_argument("--" + name, required=True)
    for name in ["resume-receipt", "resume-receipt-sha256", "resume-kernel"]:
        p.add_argument("--" + name)
    print(json.dumps(stage(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
