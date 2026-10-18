"""Separate recovered-phase global-choice execution and metadata consumer.

The original scientific chooser is unchanged. Actual native inventories and a
separate root request must exist before its model/array imports are reachable.
"""

import importlib.metadata
import json
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from scripts import kaggle_recovered_phase_inventory as inventory_worker
from scripts import recovered_selection_bridge as bridge
from scripts.stage_recovered_phase_inventory import source_names
from src.models import cpu_portability as portability
from src.models.portable_training import source_closure
from src.models.recovered_policy import checked_global_choice as checked_global_choice  # noqa: PLC0414


def remaining_budget(started, maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Required positive root max_seconds budget")
    if (
        type(started) not in (int, float)
        or not math.isfinite(started)
        or not 0 <= started <= time.monotonic()
    ):
        raise ValueError("Original service monotonic start required")
    value = maximum - (time.monotonic() - started)
    if value <= 0:
        raise TimeoutError("Original whole-service budget exhausted")
    return value


@contextmanager
def service_budget(started, maximum):
    """Root-bound whole-service deadline, including startup and native scoring."""

    def remaining():
        return remaining_budget(started, maximum)

    remaining()
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("Fresh Linux process timer required before native loading")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("Original whole-service budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    try:
        yield remaining
        remaining()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def request_inputs(
    request_path,
    request_sha256,
    hash_contract_path,
    metadata_paths,
    phase_directories,
    inventory_paths=None,
):
    request = portability.read_bound_json(request_path, request_sha256)
    portability.require(
        request["schema"] == "root_recovered_global_selection_v1"
        and request.get("proposal_only") is not True
        and request["action"] == "choose_full_validation"
        and request["validation_queries"] == 3132
        and request["training"] is False
        and request["scoring"] is True
        and request["test_annotations_used"] is False
        and type(request["max_seconds"]) is int
        and request["max_seconds"] > 0,
        "Separate actual root full-validation selection request required",
    )
    portability.require(
        set(request["inventories"]) == {"phase1", "phase2"},
        "Both actual root-frozen inventories required before imports",
    )
    hash_contract = portability.read_bound_json(
        hash_contract_path, request["hash_contract_sha256"]
    )
    inputs, scientific, policy, evidence, context = inventory_worker.metadata_gate(
        metadata_paths, hash_contract, phase_directories
    )
    portability.require(
        isinstance(request.get("selection_runtime"), dict)
        and request.get("environment") == scientific["encoding_identity"],
        "Actual separate selection runtime and full original encoding environment required",
    )
    original = json.loads(
        Path(metadata_paths["original-source-manifest.json"]).read_bytes()
    )
    root = Path(__file__).resolve().parents[1]
    names = source_names(root, original) | source_closure(
        root,
        [
            "scripts/recovered_global_selection.py",
            "scripts/kaggle_recovered_global_selection.py",
            "scripts/stage_recovered_global_selection.py",
            "scripts/kaggle_research_training.py",
        ],
    )
    portability.require(
        set(request["source_sha256"]) == names,
        "Exact transitive selection source closure required",
    )
    for name, pin in request["source_sha256"].items():
        portability.require(
            not (root / name).is_symlink() and portability.sha(root / name) == pin,
            "Actual reviewed selection source bytes differ",
        )
    frozen = request["inventories"]
    if inventory_paths is not None:
        portability.require(
            set(inventory_paths) == set(frozen),
            "Both local inventory projections required",
        )
        frozen = {
            phase: {**row, "path": str(inventory_paths[phase])}
            for phase, row in frozen.items()
        }
    inventories = bridge.authenticate_both(
        inputs, scientific, policy, evidence, context, frozen
    )
    return request, inputs, scientific, policy, evidence, context, inventories


def raw_scientific_inputs(
    preparation, cache, protocol_path, metadata_paths, scientific
):
    """Authenticate every available original preparation/cache byte, no arrays."""
    preparation, cache = Path(preparation), Path(cache)
    protocol = portability.read_bound_json(protocol_path, scientific["protocol_sha256"])
    encoded = portability.read_bound_json(
        cache / "manifest.json", scientific["cache_manifest_sha256"]
    )
    original = json.loads(
        Path(metadata_paths["original-source-manifest.json"]).read_bytes()
    )
    rows = [
        row
        for row in original["files"]
        if row["sha256"] == protocol["accounting_sha256"]
        and Path(row["path"]).name == "source-accounting.json"
    ]
    portability.require(
        len(rows) == 1, "One actual accounting-bound preparation required"
    )
    prefix = Path(rows[0]["path"]).parent.as_posix() + "/"
    selected = [row for row in original["files"] if row["path"].startswith(prefix)]
    portability.require(
        bool(selected), "Complete original preparation inventory required"
    )
    for row in selected:
        relative = row["path"][len(prefix) :]
        inventory_worker.safe(relative)
        path = preparation / relative
        portability.require(
            path.is_file()
            and not path.is_symlink()
            and portability.sha(path) == row["sha256"],
            "Original preparation bytes differ before native imports",
        )
    portability.require(
        portability.sha(preparation / "prepared/manifest.json")
        == protocol["preparation_manifest_sha256"]
        == encoded["source_manifest_sha256"],
        "Original cache/preparation binding required",
    )
    for name, pin in encoded["files"].items():
        portability.require(
            Path(name).name == name
            and not (cache / name).is_symlink()
            and portability.sha(cache / name) == pin,
            "All native cache bytes must authenticate before imports",
        )
    return protocol, encoded


def choose(
    preparation,
    cache,
    phase_directories,
    output,
    protocol_path,
    *,
    request_path,
    request_sha256,
    hash_contract_path,
    metadata_paths,
    metadata,
    service_started=None,
    inventory_paths=None,
):
    """Cloud-only scoring; no fitting and no holdout annotations are accepted."""
    started = time.monotonic() if service_started is None else service_started
    portability.require(
        type(started) in (int, float) and 0 <= started <= time.monotonic(),
        "Original actual service start required",
    )
    output = Path(output)
    portability.require(
        not output.exists(), "Fresh immutable global selection output required"
    )

    def guard():
        context = request_inputs(
            request_path,
            request_sha256,
            hash_contract_path,
            metadata_paths,
            phase_directories,
            inventory_paths,
        )
        raw_scientific_inputs(
            preparation, cache, protocol_path, metadata_paths, context[2]
        )
        return context

    request, inputs, scientific, policy, evidence, advanced_context, inventories = (
        guard()
    )
    portability.require(
        time.monotonic() - started < request["max_seconds"],
        "Whole selection startup budget exhausted",
    )
    with service_budget(started, request["max_seconds"]):
        from src.models import research_followup
        from src.models import resumable_training as trainer

        trainer.torch.set_num_threads(2)
        if trainer.torch.get_num_interop_threads() != 2:
            trainer.torch.set_num_interop_threads(2)
        trainer.torch.use_deterministic_algorithms(True)
        actual = trainer.runtime()
        encoded = json.loads(Path(metadata_paths["cache-manifest.json"]).read_bytes())[
            "encoding_identity"
        ]
        packages = {
            key: importlib.metadata.version(name)
            for name, key in (
                ("torch", "torch"),
                ("numpy", "numpy"),
                ("transformers", "transformers"),
                ("pillow", "pillow"),
                ("imageio", "imageio"),
                ("imageio-ffmpeg", "imageio_ffmpeg"),
            )
        }
        diagnostic = {
            "actual_runtime": actual,
            "expected_runtime": request["selection_runtime"],
            "actual_packages": packages,
            "expected_packages": {key: encoded[key] for key in packages},
            "training": False,
            "test_annotations_used": False,
            "native_model_or_array_loaded": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + "-runtime-diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n"
        )
        portability.require(
            actual == request["selection_runtime"]
            and all(value == encoded[key] for key, value in packages.items()),
            "Separate exact frozen global selection numerical runtime required",
        )
        guard()

        def unchanged_choose():
            return research_followup.choose(
                preparation,
                cache,
                phase_directories["phase1"],
                phase_directories["phase2"],
                output,
                protocol_path,
                scientific["protocol_sha256"],
                inputs["phase1"]["contract"]["report_sha256"],
                inputs["phase2"]["contract"]["report_sha256"],
                metadata,
            )

        result = bridge.guarded_choose(
            inputs,
            scientific,
            policy,
            evidence,
            advanced_context,
            (
                {
                    phase: {**row, "path": str(inventory_paths[phase])}
                    for phase, row in request["inventories"].items()
                }
                if inventory_paths is not None
                else request["inventories"]
            ),
            unchanged_choose,
        )
        guard()
        portability.require(
            trainer.runtime() == actual,
            "Actual selection runtime changed during scoring",
        )
        portability.require(
            result["test_targets_used"] is False
            and result["baseline"]["queries"] == 3132
            and result["selected_validation"]["queries"] == 3132
            and result["selected_checkpoint_sha256"]
            == portability.sha(output / "selected.pt"),
            "Complete actual selected global policy required",
        )
        dispatch = {
            "schema": "both_phase_recovered_global_selection_dispatch_v2",
            "result": result,
            "request_sha256": request_sha256,
            "inventory_sha256": {
                phase: row["sha256"] for phase, row in request["inventories"].items()
            },
            "source_manifest_sha256": scientific["source_manifest_sha256"],
            "cache_manifest_sha256": scientific["cache_manifest_sha256"],
            "protocol_sha256": scientific["protocol_sha256"],
            "selection_runtime": actual,
            "phase_execution_runtime": {
                phase: row["actual_execution_runtime"]
                for phase, row in inventories.items()
            },
            "phase_origin_identity_sha256": {
                phase: row["origin_identity_sha256"]
                for phase, row in inventories.items()
            },
            "source_sha256": request["source_sha256"],
            "validation_queries": 3132,
            "training": False,
            "scoring": True,
            "test_targets_used": False,
            "holdout_released": False,
        }
        target = output / "recovered-global-dispatch.json"
        target.write_text(json.dumps(dispatch, indent=2, allow_nan=False) + "\n")
        (output / "COMPLETE").write_text(portability.sha(target) + "\n")
        return dispatch
