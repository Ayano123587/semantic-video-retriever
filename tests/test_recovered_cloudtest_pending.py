"""Held future release tests; paired with uncommitted cloud test consumer."""

import ast
import json
from pathlib import Path

from test_recovered_consumer_budget import (
    test_deadline_covers_native_operation_and_disarms_on_failure as _deadline,
)
from test_recovered_consumer_budget import (
    test_original_monotonic_start_and_external_positive_budget as _budget,
)

from src.models import research_search as search


def test_cloud_interop_precedes_native_loading():
    root = Path(search.__file__).resolve().parents[2]
    tree = ast.parse((root / "scripts/kaggle_research_test_cloud.py").read_bytes())
    check = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_runtime"
    )
    calls = [node for node in ast.walk(check) if isinstance(node, ast.Call)]
    setter = next(
        node
        for node in calls
        if ast.unparse(node.func) == "torch.set_num_interop_threads"
    )
    runtime = next(node for node in calls if ast.unparse(node.func) == "runtime")
    assert setter.lineno < runtime.lineno
    assert any(
        isinstance(node, ast.If)
        and "torch.get_num_interop_threads()" in ast.unparse(node.test)
        and any(call is setter for call in ast.walk(node))
        for node in ast.walk(check)
    )


def test_cloud_original_budget():
    _budget("scripts.kaggle_research_test_cloud")


def test_cloud_deadline():
    _deadline("scripts.kaggle_research_test_cloud")


def test_actual_package_policy_is_bound_before_any_native_loading():
    import pytest

    from scripts import kaggle_research_test_cloud as worker

    private = Path("D:/CodexData/portfolio/quant-research-lab-private")
    policy_path = (
        private / "video-prospective-assets-v2/root-prospective-assets-freeze-v2.json"
    )
    if not policy_path.is_file():
        pytest.skip("Actual independently reviewed package policy unavailable")
    frozen = json.loads(policy_path.read_bytes())
    global_freeze = json.loads(
        (
            private
            / "video-recovered-inventory-v1/root-global-selection-terminal-freeze-v1.json"
        ).read_bytes()
    )
    # Metadata projection for the pure checker, not an issued release approval.
    metadata = {
        "schema": "root_frozen_cloud_test_approval_v2",
        "environment": global_freeze["environment"],
        "selection_runtime": global_freeze["actual_runtime"],
        "production_packages": frozen["production_packages"],
        "production_package_policy": {
            "path": str(policy_path),
            "sha256": "963869875500fe123bd487b0aa5423e25a87f23b26ac62e19062bdc1a957e790",
        },
    }
    assert (
        worker.checked_production_packages(metadata, policy_path)
        == frozen["production_packages"]
    )
    wrong = {
        **metadata,
        "production_package_policy": {"path": str(policy_path), "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="policy differs"):
        worker.checked_production_packages(wrong, policy_path)
    assert (
        worker.checked_production_packages(
            {"schema": "root_frozen_cloud_test_approval_v1"}, Path("not-read")
        )
        is None
    )


def test_actual_nested_pack_prepared_aliases(tmp_path):
    import pytest

    from scripts import kaggle_research_test_cloud as worker

    root = Path(
        "D:/CodexData/portfolio/semantic-video-retriever/data/interim/research-cloud-encoding-v5/source"
    )
    source_manifest = root / "video-source-manifest.json"
    cache_manifest = Path(
        "D:/CodexData/portfolio-improvements/semantic-video-resumable/data/interim/research-cloud-encoding-v5/terminal-output-v1/research-features/manifest.json"
    )
    if not source_manifest.is_file() or not cache_manifest.is_file():
        pytest.skip("Actual immutable source/cache metadata unavailable")
    source = json.loads(source_manifest.read_bytes())
    expected = "ad99d633cee842d81a237dcd95e75f39edcf1589392963a3f5310fbf0cbda09b"
    aliases = [row for row in source["files"] if row["sha256"] == expected]
    assert len(aliases) == 2
    for row in aliases:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / row["flat_name"]).read_bytes())
    metadata = {
        "source_manifest_sha256": worker.sha(source_manifest),
        "cache_manifest_sha256": worker.sha(cache_manifest),
    }
    assert (
        worker.prepared_pack_directory(
            tmp_path, source_manifest, cache_manifest, metadata
        )
        == tmp_path / "data/processed/didemo-research-v5/prepared"
    )


def test_recovered_release_binds_rank_evidence_module_and_callback():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    worker = ast.parse((root / "scripts/kaggle_research_test_cloud.py").read_text())
    stager = ast.parse((root / "scripts/stage_research_test_cloud.py").read_text())
    imports = [node for node in ast.walk(worker) if isinstance(node, ast.ImportFrom)]
    assert any(
        node.module == "src.models.release_evidence"
        and [alias.name for alias in node.names] == ["release_with_rank_evidence"]
        for node in imports
    )
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    release_call = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "release_once"
    )
    assert isinstance(release_call.args[2], ast.Name)
    assert release_call.args[2].id == "release_operation"
    assert "src/models/release_evidence.py" in {
        node.value
        for node in ast.walk(stager)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_actual_completed_production_prerequisite_is_not_informational():
    import pytest

    from scripts import kaggle_research_test_cloud as worker

    path = Path(
        "D:/CodexData/portfolio/quant-research-lab-private/video-production-query-proof-v2/root-production-terminal-freeze-v2.json"
    )
    if not path.is_file():
        pytest.skip("Actual completed production evidence unavailable")
    # Pure-checker metadata projection only; this is not a release approval.
    metadata = {
        "schema": "root_frozen_cloud_test_approval_v2",
        "root_approval_issued": True,
        "proposal_only": False,
        "action": "pack",
        "production_proof_complete_freeze_required": True,
        "production_terminal_freeze": {
            "sha256": "d7797f6b1fbbe2e94c1ed1e9dad4a1e5979aff71cbcc8a6908347c79bad36465"
        },
    }
    assert (
        worker.checked_production_terminal(metadata, path)[
            "scientific_production_proof_complete"
        ]
        is True
    )
    for field, value in (
        ("production_proof_complete_freeze_required", False),
        ("root_approval_issued", False),
        ("proposal_only", True),
    ):
        with pytest.raises(ValueError, match="production proof required"):
            worker.checked_production_terminal({**metadata, field: value}, path)


def test_production_prerequisite_precedes_bootstrap():
    from scripts import kaggle_research_test_cloud as worker

    tree = ast.parse(Path(worker.__file__).read_bytes())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    guard = next(
        n for n in calls if ast.unparse(n.func) == "checked_production_terminal"
    )
    bootstrap = next(n for n in calls if ast.unparse(n.func) == "subprocess.run")
    assert guard.lineno < bootstrap.lineno
