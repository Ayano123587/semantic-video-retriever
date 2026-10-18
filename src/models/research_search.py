"""Authenticated global-policy search over the complete available video corpus.

The CLI initializes the frozen two-thread intra/inter-op deterministic controls. Set
OMP_NUM_THREADS, MKL_NUM_THREADS and PYTHONHASHSEED before launch to their recorded
execution-runtime values; leave a variable unset when its recorded value is null.
Hardware, package versions and environment must match the frozen execution.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from pathlib import Path

FULL_COUNTS = {
    "train": {"videos": 5916, "source_ids": 5987, "moments": 119292},
    "val": {"videos": 797, "source_ids": 823, "moments": 15927},
    "test": {"videos": 741, "source_ids": 761, "moments": 15039},
}
SPLITS = ("train", "val", "test")
SOURCE_FILES = (
    "src/__init__.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/models/moment_encoder.py",
    "src/models/resumable_training.py",
    "src/models/positive_index.py",
    "src/models/research_followup.py",
    "src/models/encoding_progress.py",
    "src/data/research_receipts.py",
    "src/data/decoder_recovery.py",
    "src/data/decoder_preflight.py",
    "src/data/prepare_moments.py",
    "src/config.py",
    "src/data/didemo.py",
    "src/data/test_cohort.py",
    "src/models/encoder.py",
    "src/models/moments.py",
    "src/models/moment_training.py",
    "src/models/moment_search.py",
    "src/models/localization_followup.py",
    "src/models/moment_evaluation.py",
    "src/models/research_search.py",
)


RECOVERED_POLICY_SCHEMA = "globally_selected_recovered_video_query_policy_v2"
RECOVERED_SOURCE_FILES = (*SOURCE_FILES, "src/models/recovered_policy.py")


def query_source_files(policy):
    return (
        RECOVERED_SOURCE_FILES
        if policy.get("schema") == RECOVERED_POLICY_SCHEMA
        else SOURCE_FILES
    )


def numerical_runtime(policy):
    return (
        policy["selection_runtime"]
        if policy.get("schema") == RECOVERED_POLICY_SCHEMA
        else policy["training_runtime"]
    )


def encoder_runtime_projection(environment, *, observed=False):
    if not isinstance(environment, dict) or not set(ENCODER_ENV_KEYS).issubset(
        environment
    ):
        raise ValueError("Complete observable encoder runtime required")
    if observed and set(environment) != set(ENCODER_ENV_KEYS) | {"python"}:
        raise ValueError("Exact observed encoder runtime field inventory required")
    return {key: environment[key] for key in ENCODER_ENV_KEYS}


def frozen_runtime_state(policy):
    if policy.get("schema") == RECOVERED_POLICY_SCHEMA:
        return {
            "encoder_runtime": encoder_runtime_projection(policy["environment"]),
            "selection_runtime": numerical_runtime(policy),
        }
    return {
        "environment": policy["environment"],
        "training_runtime": numerical_runtime(policy),
    }


def runtime_state(policy):
    observed = runtime_observation()
    if policy.get("schema") == RECOVERED_POLICY_SCHEMA:
        if observed["environment"].get("python") != observed["training_runtime"].get(
            "python"
        ):
            raise ValueError("Observed Python inventories disagree")
        return {
            "encoder_runtime": encoder_runtime_projection(
                observed["environment"], observed=True
            ),
            "selection_runtime": observed["training_runtime"],
        }
    return observed


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(path, expected):
    if (
        not isinstance(expected, str)
        or not re.fullmatch("[0-9a-f]{64}", expected)
        or sha(path) != expected
    ):
        raise ValueError("Externally frozen artifact fingerprint differs")
    return json.loads(Path(path).read_bytes())


def query_text(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Nonempty query required")
    return query


ENCODER_ENV_KEYS = (
    "torch",
    "numpy",
    "transformers",
    "pillow",
    "imageio",
    "imageio_ffmpeg",
    "ffmpeg",
    "torch_build_sha256",
    "cpu_identity",
)


def runtime_observation():
    """Read existing recorded fields; import definitions but never load weights."""
    import imageio_ffmpeg
    import numpy as np
    import torch

    from src.models.moment_encoder import cpu_identity
    from src.models.resumable_training import runtime

    environment = {
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
    return {"environment": environment, "training_runtime": runtime()}


def initialize_runtime(policy):
    """Apply only controls explicitly used by the frozen training/choice bridge."""
    expected = numerical_runtime(policy)
    if (
        type(expected.get("threads")) is not int
        or expected["threads"] != 2
        or type(expected.get("interop_threads")) is not int
        or expected["interop_threads"] != 2
        or expected.get("deterministic") is not True
    ):
        raise ValueError(
            "Only frozen two-thread/interoperability deterministic query controls supported"
        )
    import torch

    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != expected["interop_threads"]:
        torch.set_num_interop_threads(expected["interop_threads"])
    torch.use_deterministic_algorithms(True)


def require_runtime(policy):
    expected = frozen_runtime_state(policy)
    if runtime_state(policy) != expected:
        raise ValueError(
            "Exact frozen numerical/encoder runtime required before native loading"
        )


def metadata_inputs(cache, prepared, receipts, checkpoint, approval, approval_sha256):
    """Authenticate small metadata and learned bytes BEFORE any native loader."""
    cache, prepared = Path(cache), Path(prepared)
    policy = bound(approval, approval_sha256)
    if (
        policy.get("schema")
        not in ("globally_selected_video_query_policy_v1", RECOVERED_POLICY_SCHEMA)
        or policy.get("test_targets_used_for_selection") is not False
    ):
        raise ValueError("Actual frozen globally selected query policy required")
    if not isinstance(policy.get("source_manifest_sha256"), str) or not re.fullmatch(
        "[0-9a-f]{64}", policy["source_manifest_sha256"]
    ):
        raise ValueError("Externally frozen training source identity required")
    root = Path(__file__).resolve().parents[2]
    if set(policy.get("source_files", {})) != set(query_source_files(policy)):
        raise ValueError("Exact production source closure required")
    for name, expected in policy["source_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Reviewed query/scoring source changed")
    if policy.get("schema") == RECOVERED_POLICY_SCHEMA:
        from src.models.recovered_policy import checked_global_choice, query_contract

        report = checked_global_choice(query_contract(policy))
    else:
        report = bound(policy["report_path"], policy["report_sha256"])
        dispatch = bound(policy["dispatch_path"], policy["dispatch_sha256"])
        if (
            dispatch.get("schema") != "both_phase_global_selection_dispatch_v1"
            or dispatch.get("result") != report
            or dispatch.get("test_targets_used") is not False
            or dispatch.get("validation_queries") != 3132
        ):
            raise ValueError("Actual full global-selection execution proof required")
        if (
            report.get("test_targets_used") is not False
            or report.get("baseline", {}).get("queries") != 3132
            or report.get("selected_validation", {}).get("queries") != 3132
            or [row.get("phase") for row in report.get("phases", [])]
            != ["phase1", "phase2"]
        ):
            raise ValueError("Both full-validation phase choices required")
        if (
            report.get("cache_manifest_sha256") != policy["cache_manifest_sha256"]
            or report.get("protocol_sha256") != policy["protocol_sha256"]
            or report.get("selected_checkpoint_sha256") != policy["checkpoint_sha256"]
        ):
            raise ValueError("Frozen global policy identities differ")
        if set(policy.get("inventories", {})) != {"phase1", "phase2"}:
            raise ValueError("Both completed learned inventories required")
        if not isinstance(policy.get("environment"), dict) or not isinstance(
            policy.get("training_runtime"), dict
        ):
            raise ValueError("Actual frozen runtime inventories required")
        if set(policy.get("phase_reports", {})) != {"phase1", "phase2"}:
            raise ValueError("Both actual runtime-bearing phase reports required")
        for phase, fits in (("phase1", 9), ("phase2", 12)):
            row = policy["inventories"][phase]
            inventory = bound(row["path"], row["sha256"])
            phase_report = next(
                item for item in report["phases"] if item["phase"] == phase
            )
            phase_meta = policy["phase_reports"][phase]
            phase_results = bound(phase_meta["path"], phase_meta["sha256"])
            if (
                phase_meta["sha256"] != phase_report.get("report_sha256")
                or phase_results.get("continuation_identity", {}).get("runtime")
                != policy["training_runtime"]
                or phase_results.get("completed_fits") != fits
                or phase_results.get("required_fits") != fits
                or inventory.get("environment") != policy["environment"]
            ):
                raise ValueError("Actual phase runtime/dispatch evidence differs")
            if (
                inventory.get("phase") != phase
                or inventory.get("source_manifest_sha256")
                != policy["source_manifest_sha256"]
                or inventory.get("learned_files_deserialized") is not False
                or inventory.get("completed_fits") != fits
                or inventory.get("validation_queries") != 3132
                or inventory.get("test_targets_used") is not False
                or inventory.get("cache_manifest_sha256")
                != policy["cache_manifest_sha256"]
                or inventory.get("protocol_sha256") != policy["protocol_sha256"]
                or inventory.get("files", {}).get("results.json")
                != phase_report.get("report_sha256")
                or dispatch.get("inventory_sha256", {}).get(phase) != row["sha256"]
            ):
                raise ValueError("Complete authenticated phase inventory differs")
    if sha(checkpoint) != policy["checkpoint_sha256"]:
        raise ValueError("Globally selected checkpoint changed before loading")
    encoded = bound(cache / "manifest.json", policy["cache_manifest_sha256"])
    source = bound(prepared / "manifest.json", policy["prepared_manifest_sha256"])
    protocol = bound(policy["protocol_path"], policy["protocol_sha256"])
    if protocol.get("source_manifest_sha256") != policy[
        "cache_manifest_sha256"
    ] or protocol.get("encoding_identity") != encoded.get("encoding_identity"):
        raise ValueError("Actual frozen protocol/encoding runtime differs")
    identity = protocol["encoding_identity"]
    if (
        policy.get("schema") == RECOVERED_POLICY_SCHEMA
        and identity != policy["environment"]
    ):
        raise ValueError("Complete encoding/cache/inventory provenance differs")
    if any(
        key not in identity or identity[key] != policy["environment"].get(key)
        for key in ENCODER_ENV_KEYS
    ) or identity.get("encoder") != encoded.get("encoder"):
        raise ValueError("Frozen encoder numerical inventory differs")
    if (
        encoded.get("status") != "complete"
        or encoded.get("smoke_only") is not False
        or encoded.get("test_text_or_targets_encoded") is not False
        or encoded.get("source_manifest_sha256") != policy["prepared_manifest_sha256"]
    ):
        raise ValueError("Complete non-smoke no-test-label feature cache required")
    if sha(receipts) != source.get("receipt_sha256"):
        raise ValueError("Prepared source receipt binding differs")
    media = bound(prepared / "media.json", source["files"]["media.json"])
    frames = bound(cache / "frames.json", encoded["files"]["frames.json"])
    videos = {}
    hashes = {}
    aliases = {}
    durations = {}
    for row in media:
        key = (row["split"], row["video"])
        if (
            row["split"] not in SPLITS
            or key in videos
            or type(row.get("num_segments")) is not int
            or not 1 <= row["num_segments"] <= 6
            or not re.fullmatch("[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ValueError("Unique valid retained media geometry required")
        if row["sha256"] in hashes:
            raise ValueError("Canonical duplicate or cross-split media hash")
        if any(
            not isinstance(row.get(field), str) or not row[field].strip()
            for field in ("license", "original_url")
        ):
            raise ValueError("Source license and attribution required")
        videos[key] = row
        hashes[row["sha256"]] = key
        aliases[key] = []
    receipt_seen = set()
    for line in Path(receipts).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row.get("split"), row.get("video"))
        if key in receipt_seen:
            raise ValueError("Duplicate source terminal receipt")
        receipt_seen.add(key)
        canonical = hashes.get(row.get("sha256"))
        if row.get("status") == "complete" and canonical is not None:
            if (
                key[0] != canonical[0]
                or not isinstance(key[1], str)
                or not key[1].strip()
            ):
                raise ValueError("Source alias split or identity mismatch")
            aliases[canonical].append(key[1])
    for row in frames:
        key = (row["split"], row["video"])
        duration = row.get("duration")
        if (
            key not in videos
            or key in durations
            or row.get("source_sha256") != videos[key]["sha256"]
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= (videos[key]["num_segments"] - 1) * 5
        ):
            raise ValueError("Complete source-bound unique frame geometry required")
        durations[key] = duration
    if set(durations) != set(videos):
        raise ValueError("All retained media need frame durations")
    for split in SPLITS:
        subset = [key for key in videos if key[0] == split]
        expected = FULL_COUNTS[split]
        if (
            len(subset) != expected["videos"]
            or sum(len(aliases[key]) for key in subset) != expected["source_ids"]
            or any(key[1] not in aliases[key] for key in subset)
        ):
            raise ValueError("Complete retained videos and source aliases required")
        name = split + "-candidates.json"
        candidates = bound(cache / name, encoded["files"][name])
        geometry = [
            (key[1], [a, end])
            for key in subset
            for a in range(videos[key]["num_segments"])
            for end in range(a, videos[key]["num_segments"])
        ]
        if (
            len(candidates) != expected["moments"]
            or [(row.get("video"), row.get("moment")) for row in candidates] != geometry
        ):
            raise ValueError("Full ordered triangular moment geometry required")
        native_name = split + "-moments.npy"
        if sha(cache / native_name) != encoded["files"][native_name]:
            raise ValueError("Authenticated native moment bytes differ before loading")
    for names in aliases.values():
        names.sort()
    return policy, report, videos, aliases, durations


class ResearchSearch:
    def __init__(
        self,
        cache,
        prepared,
        receipts,
        checkpoint,
        approval,
        approval_sha256,
        split="all",
    ):
        if split not in ("all", *SPLITS):
            raise ValueError("Unknown media split")
        policy, report, videos, aliases, durations = metadata_inputs(
            cache, prepared, receipts, checkpoint, approval, approval_sha256
        )
        initialize_runtime(policy)
        require_runtime(policy)
        from src.models.moment_evaluation import winner_family
        from src.models.moment_search import MomentSearch

        self.indexes = {}
        for name in SPLITS if split == "all" else (split,):
            index = MomentSearch(cache, checkpoint, policy["checkpoint_sha256"], name)
            family = winner_family(index.bundle)
            if (
                index.bundle.get("global_protocol_sha256") != policy["protocol_sha256"]
                or index.bundle.get("winner") != report["selected"]
                or len(index.bundle["states"]) != (0 if family == "frozen_clip" else 3)
            ):
                raise ValueError("Globally frozen winner family/ensemble differs")
            self.indexes[name] = index
        self.policy = policy
        self.videos = videos
        self.aliases = aliases
        self.durations = durations

    def search_vector(self, vector, top_k=5):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("Positive integer top-k required")
        matches = []
        for split, index in self.indexes.items():
            for row in index.search_vector(vector, top_k):
                key = (split, row["video"])
                source = self.videos[key]
                matches.append(
                    {
                        **row,
                        "split": split,
                        "source_sha256": source["sha256"],
                        "source_video_ids": self.aliases[key],
                        "license": source["license"],
                        "original_url": source["original_url"],
                    }
                )
        return sorted(matches, key=lambda row: -row["score"])[:top_k]

    def search(self, query, top_k=5, encoder=None):
        query_text(query)
        if type(top_k) is not int or top_k < 1:
            raise ValueError("Positive integer top-k required")
        from src.models.encoder import CLIPEncoder

        encoder = encoder or CLIPEncoder()
        if any(
            encoder.receipt != index.encoder_receipt for index in self.indexes.values()
        ):
            raise ValueError("Query encoder identity mismatch")
        return self.search_vector(encoder.text(query), top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    for name in (
        "cache",
        "prepared",
        "receipts",
        "checkpoint",
        "approval",
        "approval-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--top-k", type=int, default=5)
    args = vars(parser.parse_args())
    query = query_text(args.pop("query"))
    top_k = args.pop("top_k")
    if top_k < 1:
        raise ValueError("Positive integer top-k required")
    index = ResearchSearch(**args)
    print(
        json.dumps(
            {
                "checkpoint_sha256": index.policy["checkpoint_sha256"],
                "matches": index.search(query, top_k),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
