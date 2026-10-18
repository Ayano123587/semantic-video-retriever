"""Actual frozen inspection metadata; no native artifact loading or fitting."""

import copy
import os
from pathlib import Path

import pytest

from src.models import cpu_portability as portability


@pytest.fixture(scope="module")
def evidence():
    private = Path(
        os.environ.get(
            "VIDEO_PORTABILITY_EVIDENCE",
            "D:/CodexData/portfolio/quant-research-lab-private",
        )
    )
    paths = (
        private
        / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json",
        private
        / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json",
        private
        / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json",
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("Actual independently frozen inspection metadata required")
    return portability.load_evidence(*paths)


def test_actual_both_origins_and_completed_prefix(evidence):
    phases = evidence["report"]["phases"]
    assert [phases[p]["completed_seeds"] for p in ("phase1", "phase2")] == [8, 6]
    assert [phases[p]["batches"] for p in ("phase1", "phase2")] == [40929, 22028]
    assert evidence["freeze"]["training_authorized"] is False


@pytest.mark.parametrize("phase", ["phase1", "phase2"])
def test_actual_intel_execution_runtime_is_exact(evidence, phase):
    policy = portability.policy_description(evidence)
    actual = evidence["report"]["phases"][phase]["origin_identity"]["runtime"]
    proof = portability.runtime_guard(actual, phase, policy, evidence)
    assert proof["changed_paths"] == []
    assert proof["bitwise_equivalence_claimed"] is False


def test_cpu_allowlist_contains_only_actual_observations(evidence):
    policy = portability.policy_description(evidence)
    intel = evidence["report"]["phases"]["phase1"]["origin_identity"]["runtime"]["cpu"][
        "linux_cpu"
    ]
    amd = evidence["amd_observation"]["actual"]["cpu_identity"]["cpuinfo"]
    assert policy["allowed_cpu_profiles"] == [
        {key: source[key] for key in portability.DESCRIPTORS} for source in (intel, amd)
    ]
    assert set(policy["allowed_cpu_descriptor_paths"]) == set(portability.CPU_PATHS)


@pytest.mark.parametrize(
    "key", ["threads", "interop_threads", "deterministic", "torch_build"]
)
def test_non_cpu_metadata_fault_refused(evidence, key):
    # Software metadata fault injection, never a fabricated execution or dataset.
    actual = copy.deepcopy(
        evidence["report"]["phases"]["phase1"]["origin_identity"]["runtime"]
    )
    actual.pop(key)
    with pytest.raises(portability.PortabilityError, match="non-CPU"):
        portability.runtime_guard(
            actual, "phase1", portability.policy_description(evidence), evidence
        )


def test_removed_cpu_descriptor_refused(evidence):
    actual = copy.deepcopy(
        evidence["report"]["phases"]["phase1"]["origin_identity"]["runtime"]
    )
    actual["cpu"]["linux_cpu"].pop("flags")
    with pytest.raises((portability.PortabilityError, KeyError)):
        portability.runtime_guard(
            actual, "phase1", portability.policy_description(evidence), evidence
        )


def test_actual_inspection_freeze_cannot_be_training_service(evidence):
    private = Path(
        os.environ.get(
            "VIDEO_PORTABILITY_EVIDENCE",
            "D:/CodexData/portfolio/quant-research-lab-private",
        )
    )
    with pytest.raises(portability.PortabilityError, match="service contract"):
        portability.authenticate_service(
            private
            / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json",
            portability.INSPECTION_FREEZE_SHA,
            portability.fingerprint(portability.policy_description(evidence)),
            "phase1",
            evidence,
        )


def test_host_native_paths_are_forbidden():
    if os.name == "nt":
        with pytest.raises(portability.PortabilityError, match="cloud-only"):
            portability.cloud_directory(Path(__file__).parent)


def test_inspected_real_optimizer_steps_match_restore_formula(evidence):
    import math

    for observed in evidence["report"]["phases"].values():
        batches = math.ceil(23202 / observed["origin_spec"]["batch"])
        active = observed["active"]
        expected = (active["epoch"] - 1) * batches + math.ceil(
            active["cursor"] / observed["origin_spec"]["batch"]
        )
        assert all(
            item["step"] == expected for item in active["optimizer"]["state"].values()
        )
        for saved in observed["completed"]:
            assert all(
                item["step"] == saved["history_count"] * batches
                for item in saved["optimizer"]["state"].values()
            )


def test_actual_inspected_optimizer_groups_normalize_native_beta_tuple(evidence):
    import ast
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "src/models/portable_training.py"
    tree = ast.parse(path.read_bytes())
    smoke = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "restore_only"
    )
    geometry = next(
        node
        for node in smoke.body
        if isinstance(node, ast.FunctionDef) and node.name == "geometry"
    )
    metadata = next(
        node
        for node in smoke.body
        if isinstance(node, ast.FunctionDef) and node.name == "optimizer_metadata"
    )
    returned = next(
        node.value for node in metadata.body if isinstance(node, ast.Return)
    )
    group_value = next(
        value
        for key, value in zip(returned.keys, returned.values)
        if key.value == "param_groups"
    )
    assert ast.unparse(group_value) == "geometry(value['param_groups'])"
    # Execute only the existing metadata normalizer. An empty isinstance type
    # tuple guarantees that no tensor branch or native library is touched.
    namespace = {"exact": SimpleNamespace(torch=SimpleNamespace(Tensor=()))}
    exec(  # noqa: S102 - reviewed project metadata function, never input artifact code
        compile(
            ast.Module(body=[copy.deepcopy(geometry)], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    for phase in evidence["report"]["phases"].values():
        for completed in phase["completed"]:
            inspected = completed["optimizer"]["param_groups"]
            native_representation = copy.deepcopy(inspected)
            for group in native_representation:
                group["betas"] = tuple(group["betas"])
            assert native_representation != inspected
            assert namespace["geometry"](native_representation) == inspected
