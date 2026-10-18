"""Actual ten-seed restore receipts and original metadata; no model fixtures."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.models import advanced_parent as advanced
from src.models import cpu_portability as portability

PRIVATE = Path(os.environ.get("VIDEO_PRIVATE", "D:/CodexData/portfolio/quant-research-lab-private"))


def evidence():
    path = PRIVATE / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json"
    if not path.is_file():
        pytest.skip("Actual private original evidence unavailable")
    return portability.load_evidence(path,
        PRIVATE / "video-portability-inspection-v1/publication-v1/observation-20261017T140733/checkpoint-inspection.json",
        PRIVATE / "video-gateb-phase1-publication-v1/environment-terminal-v1/environment-diagnostic.json")


def paths():
    base = PRIVATE / "video-advanced-phase2-parent-v1"
    observation = base / "native-inspection-publication-v3/observation-20261017T192355"
    return {"freeze": base / "root-actual-native-restore-freeze-v1.json",
            "inspection": observation / "inspection.json", "restore": observation / "restore-result.json",
            "status": observation / "worker-status.json", "contract": base / "root-native-contract-v2.json",
            "data": base / "checkpoint-dataset-v1/root-data-freeze-v1.json"}


def test_actual_advanced_evidence_authenticates_without_native_imports():
    first = PRIVATE / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json"
    if not first.is_file():
        pytest.skip("Actual private original evidence unavailable")
    # Full-suite collection legitimately imports Torch elsewhere. Prove this
    # metadata-only authentication in a fresh isolated interpreter instead.
    code = r"""
import os
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.environ["VIDEO_PRIVATE"] = sys.argv[2]
sys.path.insert(0, str(root))
assert "torch" not in sys.modules and "numpy" not in sys.modules
scope = runpy.run_path(str(root / "tests/test_advanced_parent.py"))
assert "torch" not in sys.modules and "numpy" not in sys.modules
context = scope["advanced"].load_context(scope["paths"](), scope["evidence"]())
assert len(context["report"]["completed"]) == 10
assert context["restore"]["completed_optimizer_steps"] == 34122
assert context["report"]["active"] is None and context["report"]["next_seed"] == 1337
assert "torch" not in sys.modules and "numpy" not in sys.modules
print("actual_advanced_context_authenticated_without_native_imports")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parents[1]), str(PRIVATE)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    assert result.stdout.strip() == "actual_advanced_context_authenticated_without_native_imports"


def test_original_actual_six_seed_service_cannot_authorize_advanced_continuation():
    original = evidence()
    context = advanced.load_context(paths(), original)
    path = PRIVATE / "video-portable-restore-v3/phase2-restore-service.json"
    if not path.is_file():
        pytest.skip("Actual original restore-only service unavailable")
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Separate actual advanced-parent training authorization"):
        advanced.authenticate_service(path, pin, context["contract"]["policy_sha256"], original, context)


def test_actual_inspection_approval_is_not_training_authorization():
    original = evidence()
    actual_paths = paths()
    context = advanced.load_context(actual_paths, original)
    path = actual_paths["freeze"]
    with pytest.raises(ValueError, match="Separate actual advanced-parent training authorization"):
        advanced.authenticate_service(path, advanced.RESTORE_FREEZE_SHA, context["contract"]["policy_sha256"], original, context)


def test_mismatched_real_evidence_paths_fail_before_native_loading():
    actual_paths = paths()
    actual_paths["freeze"] = PRIVATE / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json"
    with pytest.raises(ValueError, match="Externally frozen metadata changed"):
        advanced.load_context(actual_paths, evidence())
