"""Validate published executed notebooks against their actual raw-byte receipt."""
import hashlib
import json
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
expected = {
    "notebooks/00_complete_workflow.ipynb",
    "notebooks/01_real_source_eda.ipynb",
    "notebooks/02_full_training.ipynb",
    "notebooks/03_final_evaluation.ipynb",
    "notebooks/04_full_corpus_encoding.ipynb",
    "notebooks/05_advanced_training_analysis.ipynb",
}
receipt = json.loads((root / "docs/notebook-execution.json").read_bytes())
rows = receipt["notebooks"]
if len(rows) != 6 or {row["notebook"] for row in rows} != expected:
    raise ValueError("The actual receipt must cover all six current notebooks")
if {path.relative_to(root).as_posix() for path in (root / "notebooks").glob("*.ipynb")} != expected:
    raise ValueError("Obsolete or missing notebook files")

for row in rows:
    path = root / row["notebook"]
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular published notebook required")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != row["sha256"]:
        raise ValueError("Notebook raw bytes differ from the actual receipt")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code or len(code) != row["executed_code_cells"]:
        raise ValueError("Actual executed cell count differs")
    figures = 0
    for cell in code:
        if type(cell.execution_count) is not int or cell.execution_count < 1:
            raise ValueError("An actual code cell was not executed")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValueError("Notebook contains an execution error")
            figures += "image/png" in output.get("data", {})
    if figures != row["visible_plots"] or figures < 1:
        raise ValueError("Actual notebook figure count differs")
print("All six notebook schemas, raw hashes, executed cells and figures passed.")
