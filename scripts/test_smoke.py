"""CPU-only structural checks: sources parse, trainer APIs match TRL 1.13 /
transformers 5, and the Colab bundles are in sync with the sources.

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train",
    "03b_dpo_variants", "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
    "07_grpo_bonus",
]
SOURCES = [REPO / "notebooks" / f"{nb}.py" for nb in NOTEBOOKS] + sorted((REPO / "scripts").glob("*.py")) + sorted(
    (REPO / "lab22").glob("*.py")
)


def test_sources_exist_and_parse():
    for p in SOURCES:
        assert p.exists(), f"missing {p}"
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def test_no_removed_trainer_arguments():
    # tokenizer= (TRL >= 0.13), warmup_ratio (transformers 5), max_prompt_length (TRL 1.x DPOConfig).
    banned = re.compile(r"\btokenizer\s*=\s*tokenizer\b|\bwarmup_ratio\s*=|\bmax_prompt_length\s*=")
    offenders = [str(p.relative_to(REPO)) for p in SOURCES if banned.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"removed trainer arguments in {offenders}"


def test_no_hardcoded_judge_model():
    for p in SOURCES:
        if p.name == "test_smoke.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "gpt-4o-mini" not in text and "claude-haiku" not in text, f"hard-coded judge id in {p}"


def test_colab_bundles_are_valid_and_current():
    from build_colab import render

    for tier, path in (("T4", "Lab22_DPO_T4.ipynb"), ("BIGGPU", "Lab22_DPO_BigGPU.ipynb")):
        on_disk = json.loads((REPO / "colab" / path).read_text(encoding="utf-8"))
        assert on_disk == render(tier), f"colab/{path} is stale: run `make colab`"
        ids = [cell["id"] for cell in on_disk["cells"]]
        assert len(ids) == len(set(ids))
        for cell in on_disk["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if source.startswith("%%writefile"):
                    source = source.split("\n", 1)[1]
                compile(source, f"{path}:{cell['id']}", "exec")


def test_completed_student_loss_has_stable_gradients():
    torch = pytest.importorskip("torch")
    tree = ast.parse(SOURCES[0].read_text(encoding="utf-8"))
    loss_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "my_dpo_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[loss_function], type_ignores=[]), str(SOURCES[0]), "exec"), namespace)
    loss = namespace["my_dpo_loss"]
    pc = torch.tensor([-10.0], requires_grad=True)
    pr = torch.tensor([-10.0], requires_grad=True)
    reference = torch.tensor([-10.0])
    value = loss(pc, pr, reference, reference)
    assert value.item() == pytest.approx(0.69314718)
    value.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # descent raises chosen, lowers rejected
    for margin in (-10000.0, 10000.0):
        assert torch.isfinite(loss(torch.tensor([margin]), torch.zeros(1), torch.zeros(1), torch.zeros(1)))


def test_colab_run_all_skips_disabled_bonus_and_core_runs_first():
    from build_colab import BONUS_FLAGS, render

    nb = render("T4")
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    assert next(i for i, s in enumerate(sources) if s.startswith("---\n# ⏵ `notebooks/04_")) < next(
        i for i, s in enumerate(sources) if s.startswith("---\n# ⏵ `notebooks/03b_")
    )
    guarded = [s for s in sources if s.startswith("if RUN_BONUS[")]
    assert guarded
    namespace = {"RUN_BONUS": dict.fromkeys(BONUS_FLAGS.values(), False)}
    for source in guarded:
        exec(compile(source, "bonus-cell", "exec"), namespace)
    assert set(namespace) == {"RUN_BONUS", "__builtins__"}
    for source in guarded:
        tree = ast.parse(source)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)


def test_colab_export_keeps_evidence_and_excludes_weights_and_secrets(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from zipfile import ZipFile

    from build_colab import submission_cells

    work = tmp_path / "lab22"
    for relative in (
        "adapters/dpo/dpo_metrics.json", "adapters/dpo/adapter_model.safetensors",
        "data/pref/eval.parquet", "submission/REFLECTION.md", ".env", "models/sft-merged/config.json",
    ):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example", encoding="utf-8")
    downloads = []
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(files=SimpleNamespace(download=downloads.append)))
    exec("".join(submission_cells()[-1]["source"]), {"WORK": work})
    with ZipFile(downloads[0]) as bundle:
        assert set(bundle.namelist()) == {
            "adapters/dpo/dpo_metrics.json", "data/pref/eval.parquet", "submission/REFLECTION.md",
        }
