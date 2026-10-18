"""AST parity of original scientific blocks without importing Torch or arrays."""

import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def engine(name):
    tree = ast.parse((ROOT / "src/models" / name).read_bytes())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "engine"
    )


def same(left, right):
    assert ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


class ProvenanceOnly(ast.NodeTransformer):
    """Remove precisely the three reviewed new artifact-provenance statements."""

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name) and call.func.id == "candidate_bytes":
                return None
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "payload"
                and call.func.attr == "update"
            ):
                assert {k.arg for k in call.keywords} == {"schema", "portable_lineage"}
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "seed_identity_v2":
            assert len(node.args) == 6 and not node.keywords
            node.func.id = "seed_identity"
            node.args = node.args[:4]
        return self.generic_visit(node)


def test_entire_training_policy_loop_ast_exact_after_explicit_provenance_boundaries():
    old, new = engine("resumable_training.py"), engine("portable_training.py")
    old_loop = next(n for n in old.body if isinstance(n, ast.While))
    new_loop = next(n for n in new.body if isinstance(n, ast.While))
    same(old_loop, ProvenanceOnly().visit(copy.deepcopy(new_loop)))


def test_optimizer_model_rng_restore_ast_exact():
    old, new = engine("resumable_training.py"), engine("portable_training.py")

    def restoration(function):
        marker = next(
            i for i, n in enumerate(function.body) if isinstance(n, ast.While)
        )
        return function.body[marker - 2 : marker]

    same(
        ast.Module(body=restoration(old), type_ignores=[]),
        ast.Module(body=restoration(new), type_ignores=[]),
    )


def test_budget_boundary_and_complete_phase1_controls_ast_exact():
    old, new = engine("resumable_training.py"), engine("portable_training.py")
    for name in ("boundary",):
        same(
            next(
                n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name
            ),
            next(
                n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name
            ),
        )

    def controls(function):
        loop = next(i for i, n in enumerate(function.body) if isinstance(n, ast.While))
        return function.body[loop + 1 : loop + 4]

    same(
        ast.Module(body=controls(old), type_ignores=[]),
        ast.Module(body=controls(new), type_ignores=[]),
    )


def test_final_selected_bundle_and_validation_replay_ast_exact():
    old, new = engine("resumable_training.py"), engine("portable_training.py")

    def tail(function):
        loop = next(i for i, n in enumerate(function.body) if isinstance(n, ast.While))
        return function.body[loop + 4 :]

    old_tail, new_tail = tail(old), tail(new)
    # Result provenance is the only changed tail statement; all other final
    # completeness, bytes, selected bundle, replay and metric math must match.
    old_science = [
        n
        for n in old_tail
        if not (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
        )
    ]
    new_science = [
        n
        for n in new_tail
        if not (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
        )
    ]
    same(
        ast.Module(body=old_science, type_ignores=[]),
        ast.Module(body=new_science, type_ignores=[]),
    )


def test_raw_guard_and_logical_copy_before_native_parent_loading():
    tree = ast.parse((ROOT / "src/models/portable_training.py").read_bytes())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "checked_portable_resume"
    )
    calls = [
        (n.lineno, ast.unparse(n.func))
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
    ]
    raw = min(line for line, name in calls if name == "portability.authenticate_parent")
    copied = min(line for line, name in calls if name == "shutil.copyfile")
    loaded = min(
        line
        for line, name in calls
        if name in ("exact.checked_resume", "exact.torch.load")
    )
    assert raw < copied < loaded
    assert any(
        name == "portability.sha" and raw < line < loaded for line, name in calls
    )


def test_final_result_scientific_fields_ast_exact():
    def result_dictionary(function):
        return next(
            n.value
            for n in function.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
        )

    original = result_dictionary(engine("resumable_training.py"))
    portable = result_dictionary(engine("portable_training.py"))
    provenance = {
        "continuation_identity",
        "origin_identity",
        "execution_lineage",
        "actual_execution_runtime",
        "bitwise_equivalence_claimed",
    }

    def scientific(dictionary):
        pairs = [
            (key, value)
            for key, value in zip(dictionary.keys, dictionary.values)
            if key is None
            or not isinstance(key, ast.Constant)
            or key.value not in provenance
        ]
        return ast.Dict(
            keys=[key for key, _ in pairs], values=[value for _, value in pairs]
        )

    same(scientific(original), scientific(portable))


def test_restore_only_has_no_model_execution_optimizer_update_or_scoring():
    tree = ast.parse((ROOT / "src/models/portable_training.py").read_bytes())
    smoke = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "restore_only"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(smoke) if isinstance(n, ast.Call)]
    forbidden = {
        "forward",
        "backward",
        "step",
        "evaluate",
        "score_policy",
        "primary",
        "train",
        "engine",
        "checked_protocol",
        "load_cache",
    }
    assert not any(name.split(".")[-1] in forbidden for name in calls)
    assert "model" not in calls
    assert "exact.checked_active" in calls
    assert "optimizer.load_state_dict" in calls and "model.load_state_dict" in calls
    assert "exact.restore_rng" in calls and "generator.set_state" in calls
    assert "portability.authenticate_restore_service" in calls
    statements = ast.unparse(smoke)
    assert "'training': False" in statements and "'scoring': False" in statements


def test_restore_adamw_constructor_and_load_match_original_exactly():
    tree = ast.parse((ROOT / "src/models/portable_training.py").read_bytes())
    smoke = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "restore_only"
    )
    constructor = next(
        n
        for n in ast.walk(smoke)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "exact.torch.optim.AdamW"
    )
    constructor = copy.deepcopy(constructor)
    constructor.func = ast.parse("torch.optim.AdamW", mode="eval").body
    original = next(
        n
        for n in ast.walk(engine("resumable_training.py"))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.optim.AdamW"
    )
    same(original, constructor)
