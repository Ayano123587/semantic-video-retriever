"""Opaque processor assets and tiny query fixtures; no pretrained weights/data."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import video_query_proof as p
from test_research_search import artifact as search_artifact, index as query_index
from test_research_test_transport import choice as metadata_choice


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def assets(tmp_path):
    paths = {}
    for name in p.ASSET_REQUIRED | p.ASSET_OPTIONAL:
        path = tmp_path / name
        path.write_bytes(("opaque processor asset " + name).encode())
        paths[name] = path
    versions = {name: "software fixture" for name in p.PACKAGES}
    inventory = p.asset_inventory(paths, versions, "fixture-model", "fixture-revision")
    return paths, versions, inventory


def test_prospective_inventory_never_claims_historical_assets(assets):
    paths, versions, inventory = assets
    assert inventory["historical_encoding_assets_attested"] is False
    assert (
        p.verify_assets(paths, versions, inventory, "fixture-model", "fixture-revision")
        == inventory
    )


@pytest.mark.parametrize("fault", ["missing", "changed", "version", "revision"])
def test_frozen_asset_inventory_rejects_drift_before_any_model(assets, fault):
    paths, versions, inventory = assets
    revision = "fixture-revision"
    if fault == "missing":
        paths.pop("merges.txt")
    elif fault == "changed":
        paths["vocab.json"].write_bytes(b"changed processor bytes")
    elif fault == "version":
        versions["tokenizers"] = "different runtime"
    else:
        revision = "other revision"
    with pytest.raises(ValueError):
        p.verify_assets(paths, versions, inventory, "fixture-model", revision)


@pytest.fixture
def full_query_fixture(search_artifact, monkeypatch):
    from src.models.localization_followup import score_policy
    from src.models.moments import evaluate

    monkeypatch.setattr(p, "QUERY_COUNT", 65)
    index = query_index(search_artifact)
    rows = [
        {
            "id": "validation:" + str(i),
            "text": "query " + str(i),
            "video": "val-video",
            "times": [[0, 0]],
        }
        for i in range(65)
    ]
    vectors = np.zeros((65, 512), dtype=np.float32)
    for i in range(65):
        vectors[i, i % 2] = 1

    class Tokenizer:
        def __call__(self, texts, truncation=False):
            assert truncation is False
            return {"input_ids": [[0, 1, 2] for _ in texts]}

    class Processor:
        tokenizer = Tokenizer()

    class Encoder:
        receipt = search_artifact["bundle"]["encoder"]
        processor = Processor()
        batches = []
        scalar_calls = 0

        def texts(self, texts):
            self.batches.append(len(texts))
            return np.stack([vectors[int(text.split()[-1])] for text in texts])

        def text(self, text):
            self.scalar_calls += 1
            return vectors[int(text.split()[-1])].copy()

    encoder = Encoder()
    part = index.indexes["val"]
    metrics = evaluate(
        score_policy(part.bundle, torch.from_numpy(vectors), part.features),
        part.candidates,
        rows,
    )
    return (
        index,
        encoder,
        rows,
        vectors,
        {"queries": 65, "over_token_limit": 0, "maximum_tokens": 3},
        {"selected_validation": metrics},
        (index.videos, index.aliases, index.durations),
    )


def test_every_query_original_batch32_actual_production_route_and_same_vector_reference(
    full_query_fixture,
):
    index, encoder, rows, vectors, truncation, report, metadata = full_query_fixture
    result = p.replay(index, encoder, rows, vectors, truncation, report, metadata)
    assert encoder.batches == [32, 32, 1]
    assert encoder.scalar_calls == 65
    assert result["same_vector_stable_all_split_topk_matches"] == 65
    assert result["selected_validation_metrics_replayed"] is True
    assert result["test_descriptions_or_targets_used"] is False
    assert result["all_split_benchmark_score_claimed"] is False
    assert result["different_vector_rank_bitwise_claimed"] is False


@pytest.mark.parametrize(
    "fault", ["batchdrift", "scalardrift", "truncation", "metric", "citation"]
)
def test_full_query_proof_rejects_real_boundary_failures(
    full_query_fixture, fault, monkeypatch
):
    index, encoder, rows, vectors, truncation, report, metadata = full_query_fixture
    if fault == "batchdrift":
        vectors = vectors.copy()
        vectors[0, 0] += 0.01
    elif fault == "scalardrift":
        original = encoder.text

        def changed(text):
            value = original(text)
            value[3] += 0.001
            return value / np.linalg.norm(value)

        monkeypatch.setattr(encoder, "text", changed)
    elif fault == "truncation":
        truncation["maximum_tokens"] = 99
    elif fault == "metric":
        report["selected_validation"]["corpus_consensus_r1_iou50"] = 0.012345
    else:
        original = index.search

        def changed(*args, **kwargs):
            matches = original(*args, **kwargs)
            matches[0]["source_sha256"] = "tampered"
            return matches

        monkeypatch.setattr(index, "search", changed)
    with pytest.raises(ValueError):
        p.replay(index, encoder, rows, vectors, truncation, report, metadata)


def test_scalar_batch_roundoff_uses_same_scalar_vector_for_reference(
    full_query_fixture, monkeypatch
):
    index, encoder, rows, vectors, truncation, report, metadata = full_query_fixture
    original = encoder.text

    def tiny_change(text):
        value = original(text)
        value[3] += 1e-6
        return value / np.linalg.norm(value)

    monkeypatch.setattr(encoder, "text", tiny_change)
    result = p.replay(index, encoder, rows, vectors, truncation, report, metadata)
    assert 0 < result["scalar_batch_max_abs_error"] <= p.EMBEDDING_TOLERANCE
    assert result["same_vector_stable_all_split_topk_matches"] == 65


def test_validation_identity_guard_checks_full_order_and_cached_raw_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(p, "QUERY_COUNT", 2)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [{"id": str(i), "text": "fixture query"} for i in range(2)]
    write(prepared / "val.json", rows)
    write(cache / "val-queries.json", rows)
    (cache / "val-text.npy").write_bytes(b"opaque array bytes, never parsed")
    a = {"files": {"val.json": p.sha(prepared / "val.json")}}
    b = {
        "files": {
            name: p.sha(cache / name) for name in ("val-queries.json", "val-text.npy")
        }
    }
    assert p.validation_rows(prepared, cache, b, a) == rows
    rows.reverse()
    write(cache / "val-queries.json", rows)
    b["files"]["val-queries.json"] = p.sha(cache / "val-queries.json")
    with pytest.raises(ValueError, match="ordered unique"):
        p.validation_rows(prepared, cache, b, a)


@pytest.fixture
def stage_inputs(tmp_path, metadata_choice):
    from src.models import research_search as search

    choice = metadata_choice
    protocol = tmp_path / "frozen-protocol.json"
    write(
        protocol,
        {
            "source_manifest_sha256": (
                p.CACHE_SHA
                if hasattr(p, "CACHE_SHA")
                else "ea73dced79e83cfc735652d6f8e702011d4e663c3fd9307ef010f66b9120a785"
            )
        },
    )
    choice["protocol_sha256"] = p.sha(protocol)
    for row in choice["inventories"].values():
        path = Path(row["path"])
        value = json.loads(path.read_bytes())
        value["protocol_sha256"] = p.sha(protocol)
        write(path, value)
        row["sha256"] = p.sha(path)
    report_path = Path(choice["global_choice"]["report_path"])
    report = json.loads(report_path.read_bytes())
    report["protocol_sha256"] = p.sha(protocol)
    report["selected_checkpoint_sha256"] = "c" * 64
    write(report_path, report)
    choice["global_choice"]["checkpoint_sha256"] = "c" * 64
    choice["global_choice"]["report_sha256"] = p.sha(report_path)
    dispatch_path = Path(choice["global_choice"]["dispatch_path"])
    dispatch = json.loads(dispatch_path.read_bytes())
    dispatch["result"] = report
    dispatch["inventory_sha256"] = {
        phase: row["sha256"] for phase, row in choice["inventories"].items()
    }
    write(dispatch_path, dispatch)
    choice["global_choice"]["dispatch_sha256"] = p.sha(dispatch_path)
    root = Path(search.__file__).resolve().parents[2]
    policy = {
        "schema": "globally_selected_video_query_policy_v1",
        **choice["global_choice"],
        "inventories": choice["inventories"],
        "phase_reports": choice["phase_reports"],
        "environment": choice["environment"],
        "training_runtime": choice["training_runtime"],
        "protocol_path": str(protocol),
        "protocol_sha256": p.sha(protocol),
        "cache_manifest_sha256": report["cache_manifest_sha256"],
        "source_files": {name: p.sha(root / name) for name in search.SOURCE_FILES},
    }
    query_policy = tmp_path / "query-policy.json"
    write(query_policy, policy)
    return query_policy, root


@pytest.mark.parametrize("action", ["inventory", "proof"])
def test_full_real_runner_render_and_realistic_dataset_mount(
    tmp_path, stage_inputs, assets, monkeypatch, action
):
    from scripts import stage_video_query_proof as staging
    from scripts import kaggle_video_query_proof as cloud

    query_policy, root = stage_inputs
    contract = {
        "schema": "root_frozen_video_query_proof_v1",
        "action": action,
        "global_kernel": "owner/actual-global/1",
        "query_policy_path": str(query_policy),
        "query_policy_sha256": p.sha(query_policy),
    }
    if action == "proof":
        asset_file = tmp_path / "asset-inventory.json"
        write(asset_file, assets[2])
        contract.update(
            asset_inventory_path=str(asset_file),
            asset_inventory_sha256=p.sha(asset_file),
            asset_kernel="owner/actual-assets/1",
        )
    approval = tmp_path / "approval.json"
    write(approval, contract)

    def tracked(args, **kwargs):
        if "status" in args:
            return b""
        return (root / args[-1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(staging.subprocess, "check_output", tracked)
    result = staging.stage(
        tmp_path / "stage", approval, p.sha(approval), "owner/private-query-source"
    )
    manifest = tmp_path / "stage/dataset/video-query-proof-transport.json"
    runtime = tmp_path / "stage/run.py"
    plan = json.loads(manifest.read_bytes())
    assert "dataset-metadata.json" not in plan["files"]
    # Kaggle consumes this reserved upload file; a real mount omits it.
    (manifest.parent / "dataset-metadata.json").unlink()
    monkeypatch.setattr(cloud, "EXPECTED_MANIFEST_SHA256", result["manifest_sha256"])
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes().replace(b"\n", b"\r\n"))
    cloud.checked_payload(manifest, runtime)
    runtime.write_bytes(runtime.read_bytes() + b" ")
    with pytest.raises(ValueError, match="runner"):
        cloud.checked_payload(manifest, runtime)


def test_proof_stager_cannot_skip_actual_frozen_asset_inventory(
    tmp_path, stage_inputs, monkeypatch
):
    from scripts import stage_video_query_proof as staging

    query_policy, root = stage_inputs
    contract = {
        "schema": "root_frozen_video_query_proof_v1",
        "action": "proof",
        "global_kernel": "owner/global/1",
        "query_policy_path": str(query_policy),
        "query_policy_sha256": p.sha(query_policy),
        "asset_inventory_path": str(tmp_path / "nonexistent.json"),
        "asset_inventory_sha256": "0" * 64,
        "asset_kernel": "owner/assets/1",
    }
    approval = tmp_path / "approval.json"
    write(approval, contract)
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *_a, **_k: b"")
    with pytest.raises(FileNotFoundError):
        staging.stage(
            tmp_path / "stage", approval, p.sha(approval), "owner/private-query-source"
        )
    assert not (tmp_path / "stage").exists()


def test_complete_host_stager_and_runtime_project_source_closure():
    import ast
    from scripts.stage_video_query_proof import PROJECT_SOURCE_FILES
    from src.models.research_search import RECOVERED_SOURCE_FILES

    root = Path(__file__).resolve().parents[1]
    from src.models import research_search as search
    stage_tree = ast.parse((root / "scripts/stage_video_query_proof.py").read_text())
    stage_function = next(n for n in stage_tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage")
    assembly = next(n for n in stage_function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "names" for t in n.targets))
    # Execute the real source assembly expression, not a test-created union.
    assembly_module = ast.fix_missing_locations(ast.Module(body=[assembly], type_ignores=[]))
    closure = set()
    for policy in ({}, {"schema": search.RECOVERED_POLICY_SCHEMA}):
        namespace = {"PROJECT_SOURCE_FILES": PROJECT_SOURCE_FILES, "search": search, "policy": policy}
        exec(compile(assembly_module, "actual-stager-source-assembly", "exec"), namespace)
        names = namespace["names"]
        expected = (*PROJECT_SOURCE_FILES, *(name for name in search.query_source_files(policy) if name not in PROJECT_SOURCE_FILES))
        assert names == expected and len(names) == len(set(names))
        assert set(search.query_source_files(policy)).issubset(names)
        closure.update(names)
    assert set(RECOVERED_SOURCE_FILES).issubset(closure)
    for name in closure:
        for node in ast.walk(ast.parse((root / name).read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if module == "scripts":
                dependencies = ["scripts/" + alias.name + ".py" for alias in node.names]
            elif module.startswith(("src.", "scripts.")):
                relative = module.replace(".", "/")
                target = relative + ".py"
                if not (root / target).exists():
                    target = relative + "/__init__.py"
                dependencies = [target]
            else:
                continue
            for target in dependencies:
                assert target in closure, (name, target)


def rendered_runner(tmp_path):
    from scripts import kaggle_video_query_proof as cloud

    template = Path(cloud.__file__).read_bytes()
    token = b"EXPECTED_MANIFEST_SHA256" + b" = None"
    assert template.count(token) == 1
    runtime = tmp_path / "rendered-runtime.py"
    runtime.write_bytes(
        template.replace(
            token, ("EXPECTED_MANIFEST_SHA256 = " + repr("f" * 64)).encode()
        )
    )
    namespace = {"__name__": "rendered_query_proof_fixture", "__file__": str(runtime)}
    exec(compile(runtime.read_bytes(), str(runtime), "exec"), namespace)
    namespace["installed_packages"] = lambda: {
        "torch": "fixture",
        "transformers": "fixture",
    }
    return namespace


def test_actual_rendered_runner_bootstrap_failure_writes_safe_diagnostic_before_exit(
    tmp_path, monkeypatch
):
    runtime = rendered_runner(tmp_path)
    policy = {
        "environment": {
            key: "fixture"
            for key in ("transformers", "pillow", "imageio", "imageio_ffmpeg")
        },
        "training_runtime": {"threads": 2, "deterministic": True},
    }
    output = tmp_path / "out"

    def bootstrap_failure(*_a, **_k):
        diagnostic = json.loads(
            (output / "video-query-proof-diagnostic.json").read_bytes()
        )
        assert diagnostic["phase"] == "bootstrap"
        assert diagnostic["expected_runtime"]["training_runtime"]["threads"] == 2
        assert diagnostic["actual_packages_before_bootstrap"]["torch"] == "fixture"
        raise runtime["subprocess"].CalledProcessError(
            7, ["pip", "secret=DO_NOT_RECORD"]
        )

    monkeypatch.setattr(runtime["subprocess"], "run", bootstrap_failure)
    with pytest.raises(runtime["subprocess"].CalledProcessError):
        runtime["run_with_receipt"](
            lambda: runtime["bootstrap_packages"](policy, lambda: 60), output
        )
    failure = output / "video-query-proof-failure.json"
    assert json.loads(failure.read_bytes())["returncode"] == 7
    assert json.loads(failure.read_bytes())["native_loading_started"] is False
    assert "DO_NOT_RECORD" not in failure.read_text()
    assert not (output / "video-query-proof-execution.json").exists()


def test_actual_rendered_runtime_mismatch_diagnostic_precedes_native_loads(tmp_path):
    runtime = rendered_runner(tmp_path)
    output = tmp_path / "out"
    policy = {
        "environment": {"cpu": "expected CPU"},
        "training_runtime": {"threads": 2, "deterministic": True},
    }

    class Search:
        @staticmethod
        def runtime_state(_policy):
            return {
                "environment": {"cpu": "different actual CPU"},
                "training_runtime": policy["training_runtime"],
            }

        @staticmethod
        def initialize_runtime(_policy):
            return None

        @staticmethod
        def require_runtime(_policy):
            diagnostic = json.loads(
                (output / "video-query-proof-diagnostic.json").read_bytes()
            )
            assert (
                diagnostic["runtime_before"]["environment"]["cpu"]
                == "different actual CPU"
            )
            assert (
                diagnostic["runtime_after"]["environment"]["cpu"]
                == "different actual CPU"
            )
            assert (
                diagnostic["expected_runtime"]["environment"]["cpu"] == "expected CPU"
            )
            assert diagnostic["runtime_differing_sections"] == ["environment"]
            raise ValueError("Exact runtime differs")

    def operation():
        runtime["guard_runtime"](Search, policy, "startup_runtime")
        pytest.fail("Native loader reached after runtime mismatch")

    with pytest.raises(ValueError):
        runtime["run_with_receipt"](operation, output)
    assert (
        json.loads((output / "video-query-proof-failure.json").read_bytes())["phase"]
        == "startup_runtime_before_comparison"
    )
    assert not (output / "video-query-proof-execution.json").exists()


def test_rendered_late_postflight_failure_cannot_publish_success(tmp_path):
    runtime = rendered_runner(tmp_path)
    output = tmp_path / "out"

    def operation():
        runtime["diagnose"]("postflight_assets", native_loading_started=True)
        raise ValueError("Secret message DO_NOT_RECORD")

    with pytest.raises(ValueError):
        runtime["run_with_receipt"](operation, output)
    failure = output / "video-query-proof-failure.json"
    assert json.loads(failure.read_bytes())["native_loading_started"] is True
    assert "DO_NOT_RECORD" not in failure.read_text()
    assert not (output / "video-query-proof-execution.json").exists()


def test_postflight_runtime_check_never_resets_mutated_controls(tmp_path):
    runtime = rendered_runner(tmp_path)
    output = tmp_path / "out"
    policy = {"environment": {}, "training_runtime": {"threads": 2}}

    class Search:
        @staticmethod
        def runtime_state(_policy):
            return {"environment": {}, "training_runtime": {"threads": 99}}

        @staticmethod
        def initialize_runtime(_policy):
            pytest.fail("Postflight reset hides control drift")

        @staticmethod
        def require_runtime(_policy):
            raise ValueError("Postflight control drift")

    with pytest.raises(ValueError):
        runtime["run_with_receipt"](
            lambda: runtime["guard_runtime"](
                Search, policy, "postflight_runtime", initialize=False
            ),
            output,
        )
    assert not (output / "video-query-proof-execution.json").exists()
