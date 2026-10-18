"""Source-only preflight: no remote assets or native market/model fixtures."""

import ast
import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/kaggle_video_asset_inventory.py"


def test_standalone_inventory_imports_only_standard_library():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] in sys.stdlib_module_names
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in sys.stdlib_module_names
    spec = importlib.util.spec_from_file_location("asset_inventory_preflight", SOURCE)
    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    spec.loader.exec_module(module)
    assert not {"torch", "numpy", "transformers", "tokenizers"}.intersection(
        set(sys.modules) - before
    )
    assert module.CONTRACT is None


def test_deadline_and_authentication_precede_network_and_completion():
    source = SOURCE.read_text(encoding="utf-8")
    run = source[source.index("def _run(") : source.index('if __name__ == "__main__"')]
    assert (
        run.index("signal.setitimer")
        < run.index("authenticate_inputs(contract")
        < run.index("api_bytes = fetch")
    )
    assert run.index("for role, path in inputs.items()") < run.index(
        'output / "COMPLETE"'
    )
    assert '"native_runtime_measured": False' in run
    assert '"historical_encoding_assets_attested": False' in run
    assert '"model_weights_downloaded": False' in run
    assert (
        "from_pretrained" not in source
        and "torch.load" not in source
        and "np.load" not in source
    )
