"""Publish and execute analysis of the completed advanced validation experiment."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nb
from jupyter_client import KernelManager
from nbclient import NotebookClient

PINS = {
    "results.json": "49a03fb7b212a61c4ad7a2c8ca745f37490723333f1680169ce7ee9eb7c9dfe4",
    "status.json": "94ddb25a8aa60e4b31e8e9285157d604006b5462ac115eabd8ed99e16c09a681",
    "checkpoint-000022.json": "5d6dbda5ebae2c04a6305b14824ea963500140b2fc334e4a7006c85d0d38b15b",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CELLS = [
    (
        "markdown",
        """# Advanced video retrieval training

This notebook reads the saved results of twelve completed training runs and
sixteen validation policies. Training used 23,202 queries. Evaluation uses all
3,132 eligible validation queries and all 15,927 moments in the validation
corpus. Corpus retrieval metrics are custom; within-video metrics use the
official formula. This is the complete eligible licensed validation population,
not a claim that every video from the original public benchmark was available.

The training curves below are recorded epoch observations, not reconstructed
or simulated runs. No training or test evaluation occurs in this analysis.
""",
    ),
    (
        "code",
        """from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'pyproject.toml').exists())
report = root/'reports/advanced_validation_v1'
raw = (report/'export-manifest.json').read_bytes()
manifest = json.loads(raw)
assert manifest['complete'] and not manifest['test_evaluated']
assert (report/'COMPLETE').read_text().strip() == hashlib.sha256(raw).hexdigest()
for name, pin in manifest['outputs_sha256'].items():
    assert hashlib.sha256((report/name).read_bytes()).hexdigest() == pin
result = json.loads((report/'results.json').read_bytes())
assert result['completed_fits'] == result['required_fits'] == 12
assert len(result['runs']) == 4 and len(result['policies']) == 16
assert result['test_targets_used'] is False
histories, seeds = [], []
for run in result['runs']:
    assert [s['seed'] for s in run['seeds']] == [42, 1337, 2026]
    for seed in run['seeds']:
        history = seed['history']
        assert [h['epoch'] for h in history] == list(range(1, len(history)+1))
        seeds.append(dict(penalty=run['lambda'], seed=seed['seed'],
            completed_epochs=len(history), selected_epoch=seed['best_epoch']))
        histories.extend(dict(penalty=run['lambda'], seed=seed['seed'], **h) for h in history)
history = pd.DataFrame(histories)
display(pd.DataFrame(seeds))
print('Selected policy:', result['selected'])
""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (penalty, group) in zip(axes.flat, history.groupby('penalty'), strict=True):
    for seed, rows in group.groupby('seed'):
        ax.plot(rows.epoch, rows.train_loss, marker='.', label=f'Seed {seed}')
    ax.set_title(f'Penalty weight {penalty:g}')
    ax.set_xlabel('Completed epoch')
    ax.set_ylabel('Recorded training objective')
    ax.legend(fontsize=8)
fig.suptitle('All twelve actual training histories; objectives differ across penalty weights')
fig.tight_layout()
plt.show()
""",
    ),
    (
        "code",
        """primary = 'corpus_consensus_r1_iou50'
baseline = result['baseline'][primary]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
for ax, (penalty, group) in zip(axes.flat, history.groupby('penalty'), strict=True):
    for seed, rows in group.groupby('seed'):
        ax.plot(rows.epoch, rows.validation_corpus_consensus_r1_iou50,
                marker='.', label=f'Seed {seed}')
    ax.axhline(baseline, color='black', linestyle='--', label='Frozen CLIP')
    ax.set_title(f'Penalty weight {penalty:g}')
    ax.set_xlabel('Completed epoch')
    ax.set_ylabel('Validation corpus R@1 at IoU 0.5')
    ax.legend(fontsize=8)
fig.suptitle('Validation histories; these curves participated in early stopping')
fig.tight_layout()
plt.show()
""",
    ),
    (
        "code",
        """policies = pd.DataFrame([
    dict(penalty=p['lambda'], adapted_weight=p['adapted_score_weight'],
         eligible=p['promotion_eligible'], **p['validation']) for p in result['policies']])
assert policies.queries.eq(3132).all() and policies.candidate_moments.eq(15927).all()
assert not policies.eligible.any() and result['selected'] == 'frozen_clip'
assert result['selected_validation'] == result['baseline']
display(policies[['penalty','adapted_weight',primary,'corpus_consensus_r5_iou50',
                  'within_video_official_rank1','eligible']])
table = policies.pivot(index='penalty', columns='adapted_weight', values=primary)
fig, ax = plt.subplots(figsize=(8, 5))
image = ax.imshow(table.to_numpy(), cmap='viridis')
ax.set_xticks(range(4), table.columns)
ax.set_yticks(range(4), table.index)
ax.set_xlabel('Adapted-score mixture weight')
ax.set_ylabel('Training penalty weight')
midpoint = (table.to_numpy().min()+table.to_numpy().max())/2
for i in range(4):
    for j in range(4):
        value = table.iloc[i,j]
        ax.text(j,i,f'{value:.4f}',ha='center',va='center',
                color='black' if value > midpoint else 'white')
fig.colorbar(image, ax=ax, label='Validation corpus R@1 at IoU 0.5')
ax.set_title(f'All sixteen policies; frozen CLIP reference {baseline:.4f}')
fig.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        """The best mixture improves the primary validation rate by only
about 0.002. It does not meet the registered promotion requirement, so the
selected model remains frozen CLIP. A decreasing training loss alone does not
show improved retrieval.

These figures reproduce saved training and validation summaries. Full dense
score matrices for all sixteen policies have not been independently replayed.
Production parity and the sealed test evaluation are separate deliverables.
""",
    ),
]


def build(source):
    source = Path(source)
    root = Path(__file__).resolve().parents[1]
    for name, pin in PINS.items():
        assert digest(source / name) == pin
    audit = json.loads((source / "agent-metadata-grid-audit.json").read_bytes())
    assert audit["status"] == "PASS" and audit["fits"] == 12 and audit["policies"] == 16
    assert audit["results_sha256"] == PINS["results.json"]
    report = root / "reports/advanced_validation_v1"
    report.mkdir(parents=True, exist_ok=False)
    (report / "results.json").write_bytes((source / "results.json").read_bytes())
    manifest = {
        "complete": True,
        "test_evaluated": False,
        "kernel_id": 139291990,
        "version": 1,
        "fits": 12,
        "policies": 16,
        "validation_queries": 3132,
        "candidate_moments": 15927,
        "native_artifacts_independently_verified": False,
        "all_policy_dense_scores_independently_replayed": False,
        "terminal_metadata_sha256": PINS,
        "outputs_sha256": {"results.json": digest(report / "results.json")},
    }
    (report / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (report / "COMPLETE").write_text(digest(report / "export-manifest.json") + "\n")
    document = nb.v4.new_notebook(
        cells=[
            getattr(nb.v4, "new_" + kind + "_cell")(source) for kind, source in CELLS
        ]
    )
    document.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        NotebookClient(
            document,
            km=manager,
            timeout=240,
            startup_timeout=180,
            resources={"metadata": {"path": str(root)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    path = root / "notebooks/05_advanced_training_analysis.ipynb"
    nb.write(document, path)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    build(**vars(parser.parse_args()))
