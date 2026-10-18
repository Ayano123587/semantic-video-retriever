"""Whole-engine provenance-only AST parity; no native arrays or execution."""

import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def engine(name):
    tree = ast.parse((ROOT / "src/models" / name).read_bytes())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "engine")


class ProvenanceOnly(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "engine":
            index = next(i for i, arg in enumerate(node.args.kwonlyargs) if arg.arg == "advanced_context")
            node.args.kwonlyargs.pop(index)
            node.args.kw_defaults.pop(index)
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and node.slice.value == "advanced":
            node.slice.value = "portable"
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "advanced":
            name = node.func.attr
            if name == "seed_identity":
                assert len(node.args) == 6
                node.func = ast.Name(id="seed_identity_v2", ctx=ast.Load())
            else:
                assert name in {"checked_resume", "append_execution", "validate_lineage"}
                assert isinstance(node.args[-1], ast.Name) and node.args[-1].id == "advanced_context"
                node.args.pop()
                node.func = (ast.Name(id="checked_portable_resume", ctx=ast.Load()) if name == "checked_resume"
                             else ast.Attribute(value=ast.Name(id="portability", ctx=ast.Load()), attr=name, ctx=ast.Load()))
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == "advanced_lineage":
            node.arg = "portable_lineage"
        if node.arg == "schema" and isinstance(node.value, ast.Constant) and node.value.value == 3:
            node.value.value = 2
        return self.generic_visit(node)

    def visit_Dict(self, node):
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "schema" and isinstance(value, ast.Constant):
                if value.value == 3:
                    value.value = 2
                elif value.value == "advanced_checkpoint_inventory_v3":
                    value.value = "portable_checkpoint_inventory_v2"
        return self.generic_visit(node)


def test_entire_engine_ast_equal_after_explicit_provenance_only_allowlist():
    old = engine("portable_training.py")
    new = ProvenanceOnly().visit(copy.deepcopy(engine("advanced_training.py")))
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_entire_optimizer_loss_validation_promotion_loop_ast_exact():
    old, new = engine("portable_training.py"), engine("advanced_training.py")
    old_loop = next(node for node in old.body if isinstance(node, ast.While))
    new_loop = next(node for node in new.body if isinstance(node, ast.While))
    assert ast.dump(old_loop, include_attributes=False) == ast.dump(ProvenanceOnly().visit(copy.deepcopy(new_loop)), include_attributes=False)


def test_all_native_parent_bytes_and_copies_guarded_before_loading():
    tree = ast.parse((ROOT / "src/models/advanced_parent.py").read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "checked_resume")
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(function) if isinstance(node, ast.Call)]
    authenticated = min(line for line, name in calls if name == "authenticate_parent")
    copied = min(line for line, name in calls if name == "shutil.copyfile")
    loaded = min(line for line, name in calls if name in {"exact.checked_resume", "exact.torch.load"})
    assert authenticated < copied < loaded
    assert any(authenticated < line < loaded for line, name in calls if name == "portability.sha")


def test_native_import_follows_full_guard_in_training_entrypoint():
    tree = ast.parse((ROOT / "src/models/advanced_training.py").read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    native = next(node.lineno for node in function.body if isinstance(node, ast.ImportFrom) and node.module == "src.models")
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == "guard"
               and node.lineno < native for node in function.body)
