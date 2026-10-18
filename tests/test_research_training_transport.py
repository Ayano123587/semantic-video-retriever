"""Pure transport fixtures: no native arrays, checkpoints, fitting, or network."""

import hashlib
import json
from pathlib import Path
import pytest
from scripts import kaggle_research_training as r
from scripts import stage_research_training as s


def write(path, value):
    path.write_bytes((json.dumps(value) + "\n").encode())


@pytest.fixture
def transport(tmp_path, monkeypatch):
    template = b"EXPECTED_TRANSPORT_SHA256 = None\n"
    raw = tmp_path / "training-runner.py"
    raw.write_bytes(template)
    plan = dict(
        schema="full_research_training_transport_v2",
        runner_normalization="crlf_to_lf_only_reject_lone_cr_v1",
        runner_template_canonical_sha256=hashlib.sha256(template).hexdigest(),
        source_manifest_sha256=r.SOURCE_SHA,
        cache_manifest_sha256=r.CACHE_SHA,
        protocol_sha256=r.PROTOCOL_SHA,
        encoding_kernel=r.ENCODING_KERNEL,
        source_dataset=r.SOURCE_DATASET,
        python=r.PYTHON,
        max_seconds=36000,
        test_annotations_used=False,
        phase="phase1",
        resume=None,
        runner_template_sha256=r.sha(raw),
    )
    manifest = tmp_path / "training-transport.json"
    write(manifest, plan)
    digest = r.sha(manifest)
    monkeypatch.setattr(r, "EXPECTED_TRANSPORT_SHA256", digest)
    rendered = tmp_path / "run.py"
    rendered.write_bytes(template.replace(b"None", repr(digest).encode()))
    return manifest, rendered, plan


def test_exact_rendered_runner(transport):
    manifest, rendered, plan = transport
    assert r.check_transport(manifest, rendered) == plan


@pytest.mark.parametrize("fault", ["render", "template", "manifest"])
def test_changed_transport_refused(transport, fault):
    manifest, rendered, _ = transport
    target = (
        rendered
        if fault == "render"
        else manifest.parent / "training-runner.py" if fault == "template" else manifest
    )
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        r.check_transport(manifest, rendered)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "other"),
        ("max_seconds", True),
        ("test_annotations_used", True),
        ("encoding_kernel", "unversioned"),
    ],
)
def test_transport_semantic_contract(transport, monkeypatch, field, value):
    manifest, rendered, plan = transport
    plan[field] = value
    write(manifest, plan)
    monkeypatch.setattr(r, "EXPECTED_TRANSPORT_SHA256", r.sha(manifest))
    with pytest.raises(ValueError):
        r.check_transport(manifest, rendered)


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "payload.py"
    raw.write_bytes(b"unchanged scientific bytes\r\n")
    manifest = tmp_path / "video-source-manifest.json"
    value = dict(
        source_commit="686a022d97c540f5f713d7dee5e64dcf7f6d4575",
        files=[dict(path="src/fixture.py", flat_name=raw.name, sha256=r.sha(raw))],
    )
    write(manifest, value)
    monkeypatch.setattr(r, "SOURCE_SHA", r.sha(manifest))
    return manifest, raw, value, tmp_path / "workspace"


def test_source_raw_bytes_preserved(source):
    manifest, raw, _, output = source
    r.restore_source(manifest, output)
    assert (output / "src/fixture.py").read_bytes() == raw.read_bytes()


@pytest.mark.parametrize("fault", ["payload", "duplicate", "escape", "gold"])
def test_source_failure_precedes_workspace_copy(source, monkeypatch, fault):
    manifest, raw, value, output = source
    if fault == "payload":
        raw.write_bytes(b"changed")
    elif fault == "duplicate":
        value["files"].append(value["files"][0])
    elif fault == "escape":
        value["files"][0]["path"] = "../escape"
    else:
        value["files"][0]["path"] = "data/test_data.json"
    write(manifest, value)
    monkeypatch.setattr(r, "SOURCE_SHA", r.sha(manifest))
    with pytest.raises(ValueError):
        r.restore_source(manifest, output)
    assert not output.exists()


def test_native_file_hash_before_any_array_parser(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    native = cache / "train-moments.npy"
    native.write_bytes(b"not a valid array; hash-only fixture")
    manifest = cache / "manifest.json"
    write(
        manifest,
        dict(
            status="complete",
            smoke_only=False,
            test_text_or_targets_encoded=False,
            files={native.name: r.sha(native)},
        ),
    )
    monkeypatch.setattr(r, "CACHE_SHA", r.sha(manifest))
    protocol = tmp_path / "protocol.json"
    write(
        protocol,
        dict(
            source_manifest_sha256=r.CACHE_SHA,
            coverage=dict(
                candidate_count=7592,
                retained_media=7454,
                decoder_sources=7464,
                decoder_excluded=10,
            ),
        ),
    )
    monkeypatch.setattr(r, "PROTOCOL_SHA", r.sha(protocol))
    r.authenticate_cache(cache, protocol)
    native.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="native cache"):
        r.authenticate_cache(cache, protocol)


def test_exact_environment_and_deadline_guard():
    identity = {
        k: "fixture"
        for k in [
            "torch",
            "numpy",
            "transformers",
            "pillow",
            "imageio",
            "imageio_ffmpeg",
            "ffmpeg",
            "torch_build_sha256",
            "cpu_identity",
        ]
    }
    actual = dict(python=r.PYTHON, **identity)
    assert r.check_environment(actual, identity) == actual
    with pytest.raises(ValueError):
        r.check_environment(dict(actual, numpy="other"), identity)
    assert r.remaining_seconds(100, 120) == 35980
    with pytest.raises(ValueError):
        r.remaining_seconds(0, 36000)


def test_ambiguous_input_refused(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    for p in paths:
        p.write_bytes(b"same")
    with pytest.raises(ValueError):
        r.unique_hash(paths, r.sha(paths[0]))


def test_dirty_stager_refuses_private_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(s.subprocess, "check_output", lambda *a, **k: " M source")
    monkeypatch.setattr(r, "sha", lambda *_: pytest.fail("Read before clean Git"))
    with pytest.raises(ValueError, match="Commit reviewed"):
        s.stage(tmp_path / "fresh", "phase1", "missing", "missing", "private/dataset")


def test_continuation_inventory_authenticates_all_files_before_load(tmp_path):
    receipt = tmp_path / "continuation.json"
    write(receipt, {"status": "paused_service_window"})
    checkpoint = tmp_path / "checkpoint-000001.pt"
    checkpoint.write_bytes(b"opaque snapshot, not deserialized")
    header = checkpoint.with_suffix(".json")
    write(header, {"snapshot_sha256": r.sha(checkpoint), "completed_seeds": 1})
    completed = tmp_path / "config0-seed1.pt"
    completed.write_bytes(b"opaque completed seed")
    inventory = tmp_path / "continuation-inventory.json"
    write(
        inventory,
        {
            "continuation_sha256": r.sha(receipt),
            "files": {p.name: r.sha(p) for p in (checkpoint, header, completed)},
        },
    )
    ref = {
        "receipt_sha256": r.sha(receipt),
        "inventory_sha256": r.sha(inventory),
        "snapshot": checkpoint.name,
        "snapshot_receipt_sha256": r.sha(header),
    }
    r.authenticate_continuation_inventory(receipt, ref)
    completed.write_bytes(b"tampered completed sibling")
    with pytest.raises(ValueError, match="before any checkpoint load"):
        r.authenticate_continuation_inventory(receipt, ref)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_runtime_newline_roundtrip_only(transport, newline):
    manifest, rendered, plan = transport
    rendered.write_bytes(rendered.read_bytes().replace(b"\n", newline))
    assert r.check_transport(manifest, rendered) == plan


@pytest.mark.parametrize("edit", [b" ", b"# extra code\n", b"\r"])
def test_runtime_whitespace_code_and_lone_cr_rejected(transport, edit):
    manifest, rendered, _ = transport
    rendered.write_bytes(rendered.read_bytes() + edit)
    with pytest.raises(ValueError):
        r.check_transport(manifest, rendered)


def test_canonical_template_hash_is_required(transport):
    manifest, rendered, _ = transport
    template = manifest.parent / "training-runner.py"
    assert (
        r.canonical_runner_bytes(template.read_bytes().replace(b"\n", b"\r\n"))
        == template.read_bytes()
    )
    with pytest.raises(ValueError, match="Lone CR"):
        r.canonical_runner_bytes(b"arbitrary\rcontent")


@pytest.mark.parametrize("mismatch", [False, True])
def test_environment_diagnostic_precedes_training_boundary(tmp_path, mismatch):
    identity = {
        key: "fixture"
        for key in (
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
    }
    actual = dict(python=r.PYTHON, **identity)
    if mismatch:
        actual["cpu_identity"] = "different CPU fixture"
    diagnostic = tmp_path / "environment.json"
    if mismatch:
        with pytest.raises(ValueError, match="Exact recorded"):
            r.check_environment(actual, identity, diagnostic, "phase2")
    else:
        r.check_environment(actual, identity, diagnostic, "phase1")
    result = json.loads(diagnostic.read_bytes())
    assert result["differing_keys"] == (["cpu_identity"] if mismatch else [])
    assert result["status"] == ("mismatch" if mismatch else "pass")
    assert result["actual"]["cpu_identity"] == actual["cpu_identity"]
    assert result["expected"]["cpu_identity"] == identity["cpu_identity"]
    assert result["source_manifest_sha256"] == r.SOURCE_SHA
    assert result["cache_manifest_sha256"] == r.CACHE_SHA
    assert result["protocol_sha256"] == r.PROTOCOL_SHA
    assert result["training_module_imported"] is False
    assert result["training_started"] is False
    assert result["model_weights_loaded"] is False


def test_environment_diagnostic_guard_precedes_trainer_import_and_fit():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(r.main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    guard = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "check_environment"
    )
    trainer_import = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.models.resumable_training"
    )
    fit = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "train"
    )
    assert guard.lineno < trainer_import.lineno < fit.lineno
    assert any(keyword.arg == "diagnostic_path" for keyword in guard.keywords)
