"""Actual provider-returned processor metadata; no model or array parsing."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_script(name):
    source = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_v2_inventory_uses_authenticated_staged_payload(tmp_path):
    actual = Path(
        os.environ.get(
            "VIDEO_ASSET_INVENTORY",
            "D:/CodexData/portfolio/quant-research-lab-private/video-prospective-assets-v2/publication-v1/terminal-output-v1/video-assets/inventory.json",
        )
    )
    if not actual.is_file():
        pytest.skip("Actual metadata-only provider inventory is not supplied")
    raw = actual.read_bytes()
    inventory = json.loads(raw)
    assert inventory["schema"] == "prospective_video_asset_inventory_v2"
    assert (
        hashlib.sha256(raw).hexdigest()
        == "63bdfe504891dec767decb4e1bedb5a2d44bb2e226123c2d208beb0d9afecb1f"
    )
    source = Path(__file__).resolve().parents[1] / "scripts/kaggle_video_query_proof.py"
    spec = importlib.util.spec_from_file_location("actual_asset_consumer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = tmp_path / "payload"
    payload.mkdir()
    staged = payload / "frozen-query-assets.json"
    staged.write_bytes(raw)
    root = tmp_path / "input"
    (root / "video-assets").mkdir(parents=True)
    (root / "video-assets/inventory.json").write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert module.frozen_inventory_path(payload, root, expected) == staged
    with pytest.raises(ValueError, match="differs"):
        module.frozen_inventory_path(payload, root, "0" * 64)


def test_actual_assets_and_root_package_pins_are_separate(monkeypatch):
    base = Path(
        "D:/CodexData/portfolio/quant-research-lab-private/video-prospective-assets-v2"
    )
    freeze_path = base / "root-prospective-assets-freeze-v2.json"
    if not freeze_path.is_file():
        pytest.skip("Actual independently reviewed asset freeze is not supplied")
    freeze = json.loads(freeze_path.read_bytes())
    inventory_path = Path(freeze["inventory_path"])
    inventory = json.loads(inventory_path.read_bytes())
    proof = load_script("video_query_proof")
    paths = {
        name: inventory_path.parent / "files" / name for name in inventory["files"]
    }
    packages = freeze["production_packages"]
    assert (
        proof.verify_assets(
            paths, packages, inventory, freeze["model_id"], freeze["revision"], packages
        )
        == inventory
    )
    with pytest.raises(ValueError, match="Separately root-frozen"):
        proof.verify_assets(
            paths, packages, inventory, freeze["model_id"], freeze["revision"]
        )
    assert (
        inventory["installed_packages_observed"]["transformers"]
        != packages["transformers"]
    )
    worker = load_script("kaggle_video_query_proof")
    observed = []
    monkeypatch.setattr(worker, "diagnose", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda command, **kwargs: observed.append((command, kwargs)),
    )
    global_freeze = json.loads(
        (
            base.parent
            / "video-recovered-inventory-v1/root-global-selection-terminal-freeze-v1.json"
        ).read_bytes()
    )
    policy = {
        "schema": "globally_selected_recovered_video_query_policy_v2",
        "environment": global_freeze["environment"],
        "selection_runtime": global_freeze["actual_runtime"],
    }
    worker.bootstrap_packages(policy, lambda: 77, packages)
    command, options = observed[0]
    assert all(name + "==" + version in command for name, version in packages.items())
    assert "https://download.pytorch.org/whl/cpu" in command
    assert options == {"check": True, "timeout": 77}


def test_actual_proof_closure_covers_local_project_imports():
    from scripts import stage_video_query_proof as stage
    from src.models import research_search as search

    root = Path(__file__).resolve().parents[1]
    names = set(stage.PROJECT_SOURCE_FILES) | set(search.RECOVERED_SOURCE_FILES)
    assert len(names) == 28
    for name in names:
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level:
                    module = ".".join(
                        name.removesuffix(".py").split("/")[: -node.level]
                        + module.split(".")
                    )
                candidates = [module + "." + alias.name for alias in node.names]
                candidates.append(module)
            elif isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            else:
                continue
            for module in candidates:
                if module.startswith(("scripts.", "src.")):
                    path = module.replace(".", "/") + ".py"
                    if (root / path).is_file():
                        assert path in names, (name, path)


def test_actual_proof_imports_are_pre_native():
    root = Path(__file__).resolve().parents[1]
    code = "import sys,pathlib; root=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(root)); import scripts.stage_video_query_proof as stage, scripts.kaggle_video_query_proof as worker, scripts.video_query_proof as proof; from src.models import research_search,recovered_policy; assert not {'torch','numpy','transformers','tokenizers'}.intersection(sys.modules); assert all(root in pathlib.Path(m.__file__).resolve().parents for m in (stage,worker,proof,research_search,recovered_policy)); print('pre-native import PASS')"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == "pre-native import PASS"


def test_actual_prepared_aliases_resolve_exact_reviewed_nested_path(tmp_path):
    original = Path(
        "D:/CodexData/portfolio/semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json"
    )
    policy_path = Path(
        "D:/CodexData/portfolio/quant-research-lab-private/video-production-query-proof-v2/root-query-policy-v2.json"
    )
    cache = Path(
        "D:/CodexData/portfolio-improvements/semantic-video-resumable/data/interim/research-cloud-encoding-v5/terminal-output-v1/research-features/manifest.json"
    )
    if not all(path.is_file() for path in (original, policy_path, cache)):
        pytest.skip("Actual immutable source/cache/root policy metadata unavailable")
    source = json.loads(original.read_bytes())
    policy = json.loads(policy_path.read_bytes())
    aliases = [
        row
        for row in source["files"]
        if row["sha256"] == policy["prepared_manifest_sha256"]
    ]
    assert {row["path"] for row in aliases} == {
        "prepared/manifest.json",
        "data/processed/didemo-research-v5/prepared/manifest.json",
    }
    for row in aliases:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (original.parent / row["flat_name"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        target.write_bytes(raw)
    worker = load_script("kaggle_video_query_proof")
    prepared = worker.prepared_directory(tmp_path, original, cache, policy)
    assert prepared == tmp_path / "data/processed/didemo-research-v5/prepared"
    assert not (tmp_path / "data/processed/didemo-research-v5/manifest.json").exists()
    raw_manifest = prepared / "manifest.json"
    assert worker.sha(raw_manifest) == policy["prepared_manifest_sha256"]
    metadata = json.loads(raw_manifest.read_bytes())
    (receipt_row,) = [
        row for row in source["files"] if row["sha256"] == metadata["receipt_sha256"]
    ]
    receipt = tmp_path / receipt_row["path"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes((original.parent / receipt_row["flat_name"]).read_bytes())
    assert (
        worker.unique(
            tmp_path.rglob("resolved-receipts.jsonl"), metadata["receipt_sha256"]
        )
        == receipt
    )
    with pytest.raises(ValueError, match="source/cache"):
        worker.prepared_directory(
            tmp_path, original, cache, {**policy, "source_manifest_sha256": "0" * 64}
        )
