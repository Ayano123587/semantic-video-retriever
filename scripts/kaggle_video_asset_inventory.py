"""Standalone, stdlib-only prospective CLIP metadata inventory; never loads models."""

import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CONTRACT = None  # Root-reviewed literal inserted only after actual approval.
REQUIRED = {
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
}
OPTIONAL = {"tokenizer.json", "special_tokens_map.json"}
PACKAGES = (
    "torch",
    "numpy",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "pillow",
    "imageio",
    "imageio-ffmpeg",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def authenticate_inputs(contract, root):
    """Every metadata byte is authenticated before any JSON content is trusted."""
    if (
        contract.get("schema") != "root_video_asset_inventory_v1"
        or contract.get("action") != "inventory_metadata_only"
    ):
        raise ValueError("Actual root metadata-only contract required")
    if (
        contract.get("training") is not False
        or contract.get("native_loading") is not False
        or contract.get("test_used") is not False
    ):
        raise ValueError("Metadata-only boundaries required")
    if (
        contract.get("model_id") != "openai/clip-vit-base-patch32"
        or contract.get("revision") != "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    ):
        raise ValueError("Original immutable encoder revision required")
    if (
        contract.get("weight_sha256")
        != "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
    ):
        raise ValueError("Original future model-load obligation required")
    rows = contract.get("metadata_inputs")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("Actual metadata input pins required")
    found = {}
    for role, row in rows.items():
        if (
            not isinstance(role, str)
            or not isinstance(row, dict)
            or set(row) != {"basename", "sha256", "bytes"}
        ):
            raise ValueError("Exact metadata pin fields required")
        if Path(row["basename"]).name != row["basename"]:
            raise ValueError("Safe metadata basename required")
        matches = []
        for path in root.rglob(row["basename"]):
            if (
                path.is_symlink()
                or not path.is_file()
                or root.resolve() not in path.resolve().parents
            ):
                continue
            if path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError("Unique exact root-bound metadata input required: " + role)
        found[role] = matches[0]
    return found


def asset_names(repository):
    names = [row["rfilename"] for row in repository["siblings"]]
    if len(names) != len(set(names)) or not REQUIRED.issubset(names):
        raise ValueError("Unique complete pinned repository metadata required")
    # New tokenizer/processor machinery requires conscious review, never omission.
    relevant = {
        name
        for name in names
        if any(
            token in name.lower()
            for token in ("tokeniz", "processor", "vocab", "merges", "special_tokens")
        )
        or name.endswith(("config.json", ".py"))
    }
    if relevant - REQUIRED - OPTIONAL:
        raise ValueError("Unreviewed processor/tokenizer asset or executable source")
    return names, sorted((REQUIRED | OPTIONAL).intersection(names))


def _run(contract, output, input_root):
    started = time.monotonic()
    maximum = contract.get("max_seconds")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Positive root service budget required")

    def remaining():
        value = maximum - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Whole metadata service budget exhausted")
        return value

    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("Fresh Linux process deadline required")

    def expired(_signum, _frame):
        raise TimeoutError("Whole metadata service budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining())

    inputs = authenticate_inputs(contract, input_root)
    output.mkdir(parents=True, exist_ok=False)

    def fetch(url, limit):
        request = urllib.request.Request(
            url, headers={"User-Agent": "video-asset-metadata-audit/1"}
        )
        with urllib.request.urlopen(request, timeout=min(60, remaining())) as response:
            data = response.read(limit + 1)
        remaining()
        if len(data) > limit:
            raise ValueError("Bounded metadata response exceeded")
        return data

    model = contract["model_id"]
    revision = contract["revision"]
    api_bytes = fetch(
        "https://huggingface.co/api/models/" + model + "/revision/" + revision,
        2_000_000,
    )
    repository = json.loads(api_bytes)
    if repository.get("sha") != revision:
        raise ValueError("Actual repository resolved commit differs")
    names, selected = asset_names(repository)
    assets = output / "files"
    assets.mkdir()
    files = {}
    for name in selected:
        raw = fetch(
            "https://huggingface.co/"
            + model
            + "/resolve/"
            + revision
            + "/"
            + urllib.parse.quote(name, safe=""),
            8_000_000,
        )
        path = assets / name
        path.write_bytes(raw)
        files[name] = {"sha256": digest(path), "bytes": len(raw)}
    (output / "repository-response.json").write_bytes(api_bytes)
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    receipt = {
        "schema": "prospective_video_asset_inventory_v2",
        "contract_sha256": canonical(contract),
        "model_id": model,
        "revision": revision,
        "resolved_commit": repository["sha"],
        "repository_files": names,
        "repository_response_sha256": hashlib.sha256(api_bytes).hexdigest(),
        "files": files,
        "installed_packages_observed": packages,
        "stdlib_runtime_observed": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "linux_cpuinfo": Path("/proc/cpuinfo").read_text(encoding="utf-8")
            if Path("/proc/cpuinfo").is_file()
            else None,
        },
        "selection_runtime_expected": contract["selection_runtime"],
        "native_runtime_measured": False,
        "historical_encoding_assets_attested": False,
        "model_weights_downloaded": False,
        "native_loading": False,
        "training": False,
        "test_used": False,
        "future_weight_sha256_obligation": contract["weight_sha256"],
    }
    if any(
        name in sys.modules for name in ("torch", "numpy", "transformers", "tokenizers")
    ):
        raise ValueError("Metadata inventory imported native/model dependencies")
    for role, path in inputs.items():
        row = contract["metadata_inputs"][role]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Metadata changed during inventory")
    for name, expected in files.items():
        path = assets / name
        if (
            path.is_symlink()
            or {"sha256": digest(path), "bytes": path.stat().st_size} != expected
        ):
            raise ValueError("Downloaded asset changed during inventory")
    if (
        digest(output / "repository-response.json")
        != hashlib.sha256(api_bytes).hexdigest()
    ):
        raise ValueError("Repository listing changed during inventory")
    remaining()
    path = output / "inventory.json"
    path.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "COMPLETE").write_text(digest(path) + "\n", encoding="ascii")
    return receipt


def run(contract, output, input_root):
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("Fresh Linux process deadline required")
    try:
        return _run(contract, output, input_root)
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    try:
        run(CONTRACT, Path("/kaggle/working/video-assets"), Path("/kaggle/input"))
    except Exception as error:
        Path("/kaggle/working/video-asset-failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error)[:2000],
                    "complete": False,
                    "native_loading": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
