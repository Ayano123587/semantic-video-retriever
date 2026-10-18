"""Actual metadata-role refusals and immutable chooser source boundary."""

import ast
import os
import sys
from pathlib import Path

import pytest
from test_recovered_inventory_transport import PRIVATE, actual_metadata_contract

from scripts import recovered_global_selection as selection
from src.models import cpu_portability as portability
from src.models import recovered_policy


def _isolated_test_actual_hash_only_contract_cannot_authorize_global_native_load(tmp_path):
    paths, _contract, directories = actual_metadata_contract()
    approval = PRIVATE / "video-recovered-inventory-v1/root-hash-only-contract-v1.json"
    if not approval.is_file():
        pytest.skip("Actual root hash-only approval unavailable")
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="root full-validation selection request"):
        selection.choose(
            tmp_path,
            tmp_path,
            directories,
            output,
            paths["protocol.json"],
            request_path=approval,
            request_sha256=portability.sha(approval),
            hash_contract_path=approval,
            metadata_paths=paths,
            metadata=tmp_path,
        )
    assert not output.exists()
    assert "torch" not in sys.modules and "numpy" not in sys.modules

def test_actual_hash_only_contract_cannot_authorize_global_native_load(tmp_path):
    import subprocess
    if not (PRIVATE / "video-portability-inspection-v1/root-inspection-result-freeze-v1.json").is_file():
        pytest.skip("Actual original private evidence unavailable")
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from test_recovered_global_selection import _isolated_test_actual_hash_only_contract_cannot_authorize_global_native_load
_isolated_test_actual_hash_only_contract_cannot_authorize_global_native_load(Path(sys.argv[2]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(root), str(tmp_path)], check=True, capture_output=True, text=True, timeout=120)



def test_actual_phase_outputs_are_not_global_dispatch():
    paths, _contract, _directories = actual_metadata_contract()
    with pytest.raises(ValueError, match="recovered global-selection execution"):
        selection.checked_global_choice(
            {
                "environment": {},
                "selection_runtime": {},
                "global_choice": {
                    "report_path": paths["phase1-results.json"],
                    "report_sha256": portability.sha(paths["phase1-results.json"]),
                    "dispatch_path": paths["phase2-wrapper.json"],
                    "dispatch_sha256": portability.sha(paths["phase2-wrapper.json"]),
                },
            }
        )


def test_native_import_and_unchanged_choose_follow_actual_inventory_guard():
    tree = ast.parse(Path(selection.__file__).read_bytes())
    choose = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose"
    )
    calls = [
        (n.lineno, ast.unparse(n.func))
        for n in ast.walk(choose)
        if isinstance(n, ast.Call)
    ]
    native = min(
        n.lineno
        for n in ast.walk(choose)
        if isinstance(n, ast.ImportFrom) and n.module == "src.models"
    )
    assert min(line for line, name in calls if name == "guard") < native
    assert sum(name == "research_followup.choose" for _line, name in calls) == 1
    assert not any(
        name.endswith((".train", ".fit", ".backward", ".step")) for _line, name in calls
    )
    guarded = next(
        n
        for n in ast.walk(choose)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "bridge.guarded_choose"
    )
    assert ast.unparse(guarded.args[-1]) == "unchanged_choose"
    assert any(line > guarded.lineno for line, name in calls if name == "guard")


def test_consumer_preserves_separate_phase_and_selection_runtimes():
    source = (
        Path(selection.__file__).read_text()
        + Path(recovered_policy.__file__).read_text()
    )
    assert '"phase_execution_runtime"' in source and '"selection_runtime"' in source
    assert 'phase_report["actual_execution_runtime"]' in source
    assert 'phase_report["origin_identity"]' in source
    assert 'get("runtime")' not in source


def test_consumer_rejects_missing_encoding_and_execution_roles_before_report_read():
    with pytest.raises(ValueError, match="environment and selection runtime"):
        recovered_policy.checked_global_choice({"global_choice": {}})


def test_global_transport_is_scoring_only_with_budgeted_bootstrap_and_postflight():
    from scripts import kaggle_recovered_global_selection as worker

    tree = ast.parse(Path(worker.__file__).read_bytes())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(
        ast.unparse(n.func).endswith((".fit", ".train", ".backward", ".step"))
        for n in calls
    )
    run = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    sources = ast.unparse(run)
    assert sources.index("authenticate(package") < sources.index(
        "from scripts import recovered_global_selection"
    )
    assert (
        sources.index("selection.request_inputs")
        < sources.index("subprocess.run")
        < sources.index("selection.choose")
    )
    assert sources.rindex("authenticate(package") > sources.index("selection.choose")
    pip = next(n for n in calls if ast.unparse(n.func) == "subprocess.run")
    assert any(
        k.arg == "timeout" and ast.unparse(k.value) == "remaining()"
        for k in pip.keywords
    )
    assert "test_annotations_used" in Path(worker.__file__).read_text()


def test_worker_final_postflight_and_success_share_original_service_deadline():
    from scripts import kaggle_recovered_global_selection as worker

    tree = ast.parse(Path(worker.__file__).read_bytes())
    blocks = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call)
            and ast.unparse(i.context_expr.func) == "selection.service_budget"
            for i in n.items
        )
    ]
    assert len(blocks) == 2
    final = max(blocks, key=lambda n: n.lineno)
    assert (
        ast.unparse(final.items[0].context_expr)
        == "selection.service_budget(started, request['max_seconds'])"
    )
    source = ast.unparse(final)
    assert (
        "authenticate(package" in source
        and "status='complete'" in source
        and "record()" in source
    )


def _isolated_test_actual_full_staged_project_closure_resolves_every_local_import():
    import json

    from scripts import stage_recovered_global_selection as stage

    root = Path(stage.__file__).resolve().parents[1]
    original = Path(os.environ.get(
        "VIDEO_ORIGINAL_SOURCE_MANIFEST",
        PRIVATE.parent / "semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json",
    ))
    if not original.is_file():
        pytest.skip("Actual immutable original source manifest unavailable")
    names = stage.source_names(root, json.loads(original.read_bytes()))
    assert len(names) == 31 and "scripts/kaggle_research_training.py" in names
    missing = set()
    for name in names:
        tree = ast.parse((root / name).read_bytes())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [
                    node.module,
                    *[node.module + "." + alias.name for alias in node.names],
                ]
            for module in modules:
                if not module.startswith(("src.", "scripts.")):
                    continue
                candidate = root.joinpath(*module.split(".")).with_suffix(".py")
                if (
                    candidate.is_file()
                    and candidate.relative_to(root).as_posix() not in names
                ):
                    missing.add(candidate.relative_to(root).as_posix())
                parts = module.split(".")
                for count in range(1, len(parts)):
                    init = root.joinpath(*parts[:count], "__init__.py")
                    if (
                        init.is_file()
                        and init.relative_to(root).as_posix() not in names
                    ):
                        missing.add(init.relative_to(root).as_posix())
    assert not missing, sorted(missing)
    assert "torch" not in sys.modules and "numpy" not in sys.modules

def test_actual_full_staged_project_closure_resolves_every_local_import():
    import subprocess
    original = Path(os.environ.get(
        "VIDEO_ORIGINAL_SOURCE_MANIFEST",
        PRIVATE.parent / "semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json",
    ))
    if not original.is_file():
        pytest.skip("Actual immutable original source manifest unavailable")
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from test_recovered_global_selection import _isolated_test_actual_full_staged_project_closure_resolves_every_local_import
_isolated_test_actual_full_staged_project_closure_resolves_every_local_import()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True, capture_output=True, text=True, timeout=120)



def test_actual_materialized_project_imports_in_isolated_pre_native_process(tmp_path):
    import json
    import subprocess

    from scripts import stage_recovered_global_selection as stage

    root = Path(stage.__file__).resolve().parents[1]
    original = Path(os.environ.get(
        "VIDEO_ORIGINAL_SOURCE_MANIFEST",
        PRIVATE.parent / "semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json",
    ))
    if not original.is_file():
        pytest.skip("Actual immutable original source manifest unavailable")
    names = stage.source_names(root, json.loads(original.read_bytes()))
    project = tmp_path / "project"
    for name in names:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    modules = sorted(
        name[:-3].replace("/", ".")
        for name in names
        if name.startswith("scripts/")
        and name.endswith(".py")
        and not name.endswith("/__init__.py")
    )
    code = """import importlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(project))
modules=json.loads(sys.argv[2])
for name in modules:
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(project),name
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'modules':len(modules),'native_imported':False}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(project), json.dumps(modules)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt = json.loads(result.stdout)
    assert receipt == {"modules": len(modules), "native_imported": False}
    assert "scripts.kaggle_research_training" in modules


def actual_accounting_tree(tmp_path):
    import json
    from pathlib import PurePosixPath

    from scripts import kaggle_recovered_global_selection as worker

    source = (
        PRIVATE.parent
        / "semantic-video-retriever/data/interim/research-cloud-encoding-v5/source"
    )
    manifest_path = source / "video-source-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Actual immutable source transport unavailable")
    manifest = json.loads(manifest_path.read_bytes())
    rows = {row["path"]: row for row in manifest["files"]}
    account = rows["data/processed/didemo-research-v5/source-accounting.json"]
    raw = (source / account["flat_name"]).read_bytes()
    assert worker.sha(source / account["flat_name"]) == account["sha256"]
    audit = json.loads(raw)
    science = tmp_path / "scientific"
    selected = [account]
    for name, pin in audit["input_sha256"].items():
        row = rows[PurePosixPath(name.replace("\\", "/")).as_posix()]
        assert row["sha256"] == pin
        selected.append(row)
    for row in selected:
        data = (source / row["flat_name"]).read_bytes()
        assert worker.sha(source / row["flat_name"]) == row["sha256"]
        target = science / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return (
        science,
        science / Path(account["path"]).parent,
        account["sha256"],
        audit["input_sha256"],
        manifest_path,
        manifest,
    )


def _isolated_test_real_accounting_paths_resolve_only_from_authenticated_science(tmp_path):
    from pathlib import PurePosixPath

    from scripts import kaggle_recovered_global_selection as worker

    science, prepared, pin, inputs, manifest_path, manifest = actual_accounting_tree(
        tmp_path
    )
    source_root = Path(worker.__file__).resolve().parents[1]
    original = source_root / "src/models/research_followup.py"
    row = next(
        row
        for row in manifest["files"]
        if row["path"] == "src/models/research_followup.py"
    )
    assert worker.sha(original) == row["sha256"]
    tree = ast.parse(original.read_bytes())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "accounting_path"
    )
    namespace = {"Path": Path, "PurePosixPath": PurePosixPath}
    exec(  # noqa: S102 - immutable original stdlib-only path resolver
        compile(ast.Module(body=[function], type_ignores=[]), str(original), "exec"),
        namespace,
    )
    legacy = namespace["accounting_path"]
    before = Path.cwd()
    assert worker.authenticate_accounting(science, prepared, pin) == inputs
    with worker.scientific_directory(science):
        assert Path.cwd().resolve() == science.resolve()
        for name, expected in inputs.items():
            assert worker.sha(legacy(name)) == expected
    assert Path.cwd() == before
    with pytest.raises(ValueError, match="Frozen real source accounting"):
        worker.authenticate_accounting(science, prepared, worker.sha(manifest_path))
    with (
        pytest.raises(RuntimeError, match="software exception"),
        worker.scientific_directory(science),
    ):
        raise RuntimeError("software exception")
    assert Path.cwd() == before
    assert "torch" not in sys.modules and "numpy" not in sys.modules

def test_real_accounting_paths_resolve_only_from_authenticated_science(tmp_path):
    import subprocess
    # Same original full source-transport gate. Metadata-only CI does not invent
    # the actual accounting/annotation inputs required by this integration test.
    manifest = PRIVATE.parent / "semantic-video-retriever/data/interim/research-cloud-encoding-v5/source/video-source-manifest.json"
    if not manifest.is_file():
        pytest.skip("Actual immutable source transport unavailable")
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from test_recovered_global_selection import _isolated_test_real_accounting_paths_resolve_only_from_authenticated_science
_isolated_test_real_accounting_paths_resolve_only_from_authenticated_science(Path(sys.argv[2]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(root), str(tmp_path)], check=True, capture_output=True, text=True, timeout=120)



def test_accounting_preflight_and_scientific_cwd_precede_unchanged_choose():
    from scripts import kaggle_recovered_global_selection as worker

    tree = ast.parse(Path(worker.__file__).read_bytes())
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    before = min(
        node.lineno
        for node in calls
        if ast.unparse(node.func) == "authenticate_accounting"
    )
    pip = min(
        node.lineno for node in calls if ast.unparse(node.func) == "subprocess.run"
    )
    assert before < pip
    context = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func) == "scientific_directory"
            for item in node.items
        )
    )
    assert any(
        isinstance(node, ast.Call) and ast.unparse(node.func) == "selection.choose"
        for node in ast.walk(context)
    )
