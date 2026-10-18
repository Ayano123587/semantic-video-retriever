"""Actual saved validation rank evidence; no native arrays or test annotations."""

import ast
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

from src.models import release_evidence as evidence


def test_original_guard_marker_encoder_scorer_metric_body_is_preserved():
    root = Path(__file__).resolve().parents[1]
    before = ast.parse(
        (root / "src/models/moment_evaluation.py").read_text(encoding="utf-8")
    )
    after = ast.parse(
        (root / "src/models/release_evidence.py").read_text(encoding="utf-8")
    )
    original = next(
        node
        for node in before.body
        if isinstance(node, ast.FunctionDef) and node.name == "release"
    )
    wrapper = next(
        node
        for node in after.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "release_with_rank_evidence"
    )
    # Explicit aliases bind exactly the original APIs, not replacements.
    for name in (
        "verified_inputs",
        "read_json",
        "checked_test_cohort",
        "interval",
        "pinned_encoder",
        "scores_for",
        "policy_scores",
        "evaluate",
    ):
        alias = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        )
        assert ast.unparse(alias.value) == "original." + name
    start = next(
        index
        for index, node in enumerate(wrapper.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "(source, output)"
    )
    additions = {
        "baseline_scores",
        "baseline_metrics",
        "selected_scores",
        "selected_metrics",
        "report_policy_sha256",
    }
    filtered = []
    for node in wrapper.body[start:]:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in additions
        ):
            continue
        if (
            isinstance(node, ast.Delete)
            and ast.unparse(node.targets[0]) == "baseline_scores"
        ):
            continue
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "report['rank_evidence']"
        ):
            continue
        filtered.append(copy.deepcopy(node))
    substitutions = {
        "baseline_metrics": "evaluate(scores_for([], x, v), candidates, queries)",
        "selected_metrics": "evaluate(policy_scores(selected, x, v), candidates, queries)",
        "report_policy_sha256": "digest(source / POLICY)",
    }

    class RestoreExpressions(ast.NodeTransformer):
        def visit_Name(self, node):
            return (
                ast.parse(substitutions[node.id], mode="eval").body
                if isinstance(node.ctx, ast.Load) and node.id in substitutions
                else node
            )

    normalized = [RestoreExpressions().visit(node) for node in filtered]
    original_body = original.body[1:]  # original docstring only
    assert len(normalized) == len(original_body)
    assert [ast.dump(node, include_attributes=False) for node in normalized] == [
        ast.dump(node, include_attributes=False) for node in original_body
    ]
    score_operations = [
        ast.unparse(node.value)
        for node in wrapper.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in additions - {"report_policy_sha256"}
    ]
    assert score_operations == [
        "scores_for([], x, v)",
        "evaluate(baseline_scores, candidates, queries)",
        "policy_scores(selected, x, v)",
        "evaluate(selected_scores, candidates, queries)",
    ]


def actual_validation_directory():
    return Path(os.environ.get(
        "VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private"
    )) / (
        "video-phase1-full-validation-rank-export-v1/publication-v1/"
        "observation-20261017T190046/validation"
    )


def _isolated_test_all_actual_validation_ranks_replay_original_metrics():
    root = actual_validation_directory()
    if not (root / "manifest.json").is_file():
        pytest.skip("Actual full validation rank export is not supplied")
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert (root / "COMPLETE").read_text().strip() == evidence.digest(
        root / "manifest.json"
    )
    for name, row in manifest["files"].items():
        path = root / name
        assert (
            evidence.digest(path) == row["sha256"]
            and path.stat().st_size == row["bytes"]
        )
    assert (
        evidence.digest(root / "candidate-keys.json")
        == "55318838aa14039ae50641729d3a953631a154149471ab5abe0b92c6fdfa5a4d"
    )
    candidates = json.loads((root / "candidate-keys.json").read_bytes())
    assert len(candidates) == 15927 and len({row["video"] for row in candidates}) == 797
    source = Path(os.environ.get(
        "VIDEO_VALIDATION_SOURCE",
        "D:/CodexData/portfolio/semantic-video-retriever/data/interim/research-cloud-encoding-v5/source",
    ))
    source_manifest = json.loads((source / "video-source-manifest.json").read_bytes())
    (source_row,) = [
        row
        for row in source_manifest["files"]
        if row["path"] == "data/processed/didemo-research-v5/prepared/val.json"
    ]
    actual_path = source / source_row["flat_name"]
    assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == source_row["sha256"]
    queries = json.loads(actual_path.read_bytes())
    with (root / "validation-ranks.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert [row["id"] for row in rows] == [row["id"] for row in queries]
    assert [(row["video"], row["references"]) for row in rows] == [
        (row["video"], row["times"]) for row in queries
    ]
    replay = evidence.replay_rank_evidence(
        candidates, rows, 3132, [row["id"] for row in queries]
    )
    metrics = json.loads((root / "metrics.json").read_bytes())
    for name, value in replay.items():
        assert math.isclose(value, metrics[name], rel_tol=0, abs_tol=1e-12), name
    assert "torch" not in sys.modules and "numpy" not in sys.modules


def test_all_actual_validation_ranks_replay_original_metrics():
    if not (actual_validation_directory() / "manifest.json").is_file():
        pytest.skip("Actual full validation rank export is not supplied")
    # Collection may legitimately load native libraries in other tests.
    # Preserve this entire original test in an isolated interpreter instead.
    import subprocess
    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root),str(root/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
import pytest
module=importlib.import_module('test_release_evidence')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    module._isolated_test_all_actual_validation_ranks_replay_original_metrics()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print('actual-isolated-test-PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "actual-isolated-test-PASS"
