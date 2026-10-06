"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/mace_materials_colab.ipynb`
(review PR #9: MMC-M1..M4, MMC-m1..m7).

Everything here runs in CI's dependency budget (numpy, ase, pytest): the notebook's own cells are read from the
committed JSON, Section 4 is executed in a namespace of the carried modules' names, and the pipeline's chunking and
adapt guard are exercised with a stand-in `predict`. The one model-backed test (two consecutive adaptations with
different trainable scopes) is skipped unless mace-torch and the converted weights are present.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import pytest

from mace_materials_pipeline import metrics as metrics_module
from mace_materials_pipeline import pipeline as pipeline_module
from mace_materials_pipeline import samples as samples_module
from mace_materials_pipeline.pipeline import MAX_ATOMS_PER_CALL, MAX_STRUCTURES_PER_CALL, MaceMaterialsPipeline
from mace_materials_pipeline.samples import (
    MAX_DATASET_RECORDS,
    load_byod_dataset,
    split_dataset,
    split_report,
    validate_dataset,
    write_dataset_xyz,
)

ROOT = Path(__file__).resolve().parents[1]
# Structure validation and extended-XYZ I/O use ase (installed in CI); skip those tests where it is absent.
needs_ase = pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="ase not installed")
NOTEBOOK =ROOT / "tutorials" / "mace_materials_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_mace_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and _source(cell).startswith(heading):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _kernel_cells() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]


def _cu_al(n_cu: int, n_al: int, shift: float = 0.0) -> dict:
    """A 32-atom fcc 2x2x2 supercell (a = 3.8 A) with the first n_cu sites Cu and the rest Al, plus a small shift."""
    a = 3.8
    base = [[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    positions = [[x + i * a + shift, y + j * a, z + k * a] for i in range(2) for j in range(2) for k in range(2) for x, y, z in base]
    return {"symbols": ["Cu"] * n_cu + ["Al"] * n_al, "positions": positions, "cell": [[2 * a, 0, 0], [0, 2 * a, 0], [0, 0, 2 * a]], "pbc": [True, True, True]}


def _labelled(name: str, n_cu: int, n_al: int, k: int = 0, group: str | None = None) -> dict:
    record = {**_cu_al(n_cu, n_al, shift=0.01 * k), "name": name, "energy": 0.1 * k + n_cu * 0.01, "forces": [[0.0, 0.0, 0.0]] * 32}
    if group is not None:
        record["group"] = group
    return record


# --- MMC-M1: isolated uv environment, hash lock without the sdist-only dependency -------------------------------------


def test_exactly_two_kernel_cells_build_and_route_to_an_isolated_python_3_12():
    kernel = _kernel_cells()
    assert len(kernel) == 2
    install, router = kernel
    for needed in ("MANAGED_PYTHON = '3.12.12'", '"--managed-python"', '"--require-hashes"', '"--only-binary", ":all:"', '"--no-deps", "-r", str(lock_path)', "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install, needed
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED=\"1\"" in router
    runtime = _code_after("### Record the runtime")
    assert "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in runtime


def test_carried_lock_is_the_committed_lock_resolves_mace_as_wheels_and_matches_the_t4_lock_versions():
    install = _kernel_cells()[0]
    carried = re.search(r"^LOCK_TEXT = r'''(.*?)'''$", install, re.M | re.S).group(1)
    committed = LOCK.read_text(encoding="utf-8")
    assert carried == committed
    build = _load_tool("build_notebook")
    build.check_lock(build._pins(ROOT), committed)  # every pyproject pin at its version, every entry hashed
    locked = build.lock_packages(committed)
    assert len(locked) == 73
    assert locked["mace-torch"] == "0.3.16" and locked["e3nn"] == "0.4.4" and locked["ase"] == "3.29.0"
    for name in ("matscipy", "scipy", "torch-ema", "torchmetrics", "h5py", "lmdb", "opt-einsum", "pandas", "matplotlib"):
        assert name in locked, name
    assert "python-hostlist" not in locked  # sdist only; excluded through lock_no_deps
    # The 39 packages shared with ast-audio-classification-pipeline's T4-passed lock (16eee39) sit at its versions.
    for name, version in {"torch": "2.14.0", "torchvision": "0.29.0", "numpy": "2.5.3", "safetensors": "0.8.0", "triton": "3.8.0", "cuda-bindings": "13.4.3", "nvidia-cudnn-cu13": "9.24.0.43"}.items():
        assert locked[name] == version, name
    assert locked["huggingface-hub"] == "1.32.0"  # this repository's own pin


def test_lock_no_deps_refuses_a_lock_that_still_pins_an_excluded_package():
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    ctx = {"lock_text": LOCK.read_text(encoding="utf-8") + "python-hostlist==2.3.0 \\\n    --hash=sha256:" + "0" * 64 + "\n"}
    with pytest.raises(SystemExit, match="lock_no_deps names packages the lock still pins"):
        build._isolated_install(ctx, template, "PINS = []")
    ok = build._isolated_install({"lock_text": LOCK.read_text(encoding="utf-8")}, template, "PINS = []")
    assert '"--no-deps", "-r", str(lock_path)]' in ok


def test_mace_tolerates_a_missing_hostlist_module():
    """The exclusion is safe only because mace imports hostlist inside try/except; check the pinned source says so."""
    overrides = (ROOT / "tutorials" / "requirements-colab-overrides.txt").read_text(encoding="utf-8")
    assert 'python-hostlist ; sys_platform == "never"' in overrides
    assert "try: import hostlist / except ImportError: hostlist = None" in overrides


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: a library that calls importlib.util.find_spec("google.colab") raises on a spec-less stub."""
    router = _kernel_cells()[1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_no_restart_or_in_kernel_install_text_and_spec_2_2():
    md = _markdown()
    for stale in ("Restart the runtime", "the cell stops with a restart instruction", "installs the pinned dependencies", "re-run from that cell", "Google Colab or Jupyter, Python 3.12"):
        assert stale not in md, stale
    assert "DIMER Notebook Specification 2.2" in md and "Linux x86_64" in md
    assert json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]["notebook_spec"] == "2.2"
    restart_doc = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "an interpreter restart after the install is expected" not in restart_doc
    for name in ("STATUS.md", "README.md"):
        assert "**Candidate**" in (ROOT / name).read_text(encoding="utf-8"), name


# --- MMC-M2 / MMC-M3: every frozen-model and adaptation section starts from the pinned base ---------------------------


def test_sections_5_to_7_return_to_the_pinned_base_and_adapt_refuses_an_adapted_pipeline():
    reset_if_adapted = "if pipe.adapter is not None:\n    print({'reset_to_pretrained': pipe.reset_to_pretrained()})"
    assert _code_after("## 5.").count(reset_if_adapted) == 1
    assert _code_after("## 6.").startswith(reset_if_adapted)
    s7 = _code_after("## 7.")
    assert s7.index("print({'reset_to_pretrained': pipe.reset_to_pretrained()})") < s7.index("adapt_result = pipe.adapt(")
    pipe = MaceMaterialsPipeline(model=None, config={}, device="cpu", weights_dir=Path("w"), source="stand-in", adapter={"best_epoch": 3})
    with pytest.raises(ValueError, match=r"already adapted; call pipe.reset_to_pretrained\(\)"):
        pipe.adapt([], [])


def test_export_replaces_the_previous_adapter_only_after_reload_parity():
    s9 = _code_after("## 9.")
    body = s9[s9.index("def export_and_reload("): s9.index("artifact_dir = Path(OUTPUTS['adapter'])")]
    order = [body.index(k) for k in ("model_pipe.save_artifact(staging", "MaceMaterialsPipeline.from_artifact(staging", "raise AssertionError(", "shutil.rmtree(target", "staging.rename(target)")]
    assert order == sorted(order)
    assert "shutil.rmtree(artifact_dir" not in s9


def test_activity_trains_a_separate_pipeline_from_the_same_starting_point():
    s10 = _code_after("## 10.")
    assert "activity_pipe = MaceMaterialsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in s10
    assert "pipe.adapt(" not in s10.replace("activity_pipe.adapt(", "")
    assert "assert start_diff < 1e-6" in s10 and "export_and_reload(activity_pipe, OUTPUTS['activity_adapter']" in s10
    assert "ACTIVITY_TRAINABLE_BLOCKS = 0  # @param [0, 1, 2]" in s10


# --- MMC-M4 / MMC-m1: grouped BYOD split, per-split minimums, ceilings and chunked evaluation -------------------------


@needs_ase
def test_group_split_keeps_every_trajectory_on_one_side():
    records = [_labelled(f"t{g}-{k}", 16, 16, k, group=g) for g in "ABCD" for k in range(6)]
    splits = split_dataset(records, seed=0, group_by="auto")
    report = split_report(splits, group_by="auto")
    assert report["mode"] == "group" and report["groups_in_more_than_one_split"] == []
    seen = [g for name in ("train", "val", "test") for g in report["groups"][name]]
    assert sorted(seen) == ["A", "B", "C", "D"]
    assert report["sizes"] == {"train": 12, "val": 6, "test": 6}


@needs_ase
def test_auto_without_tags_groups_by_composition_and_one_per_composition_completes():
    comps = [(32, 0), (16, 16), (24, 8), (8, 24), (28, 4), (4, 28), (20, 12), (12, 20)]
    records = [_labelled(f"c{i}", cu, al, i) for i, (cu, al) in enumerate(comps)]
    splits = split_dataset(records, seed=0, group_by="auto")
    report = split_report(splits, group_by="auto")
    assert report["mode"] == "composition" and not report["groups_in_more_than_one_split"]
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (4, 2, 2)


@needs_ase
def test_four_compositions_of_two_complete_grouped_and_are_refused_stratified_with_the_rule_named():
    comps = [(32, 0), (16, 16), (24, 8), (8, 24)]
    records = [_labelled(f"c{i}-{k}", cu, al, k) for i, (cu, al) in enumerate(comps) for k in range(2)]
    grouped = split_dataset(records, seed=42, group_by="auto")
    assert min(len(grouped["train"]), 4) == 4 and grouped["val"] and grouped["test"]
    with pytest.raises(ValueError, match=r"the validation split has 0 structure\(s\); at least 1 are needed"):
        split_dataset(records, seed=42, group_by="stratified")


@needs_ase
def test_group_split_refusals_name_the_rule():
    two_groups = [_labelled(f"t{g}-{k}", 16, 16, k, group=g) for g in "AB" for k in range(4)]
    with pytest.raises(ValueError, match="needs at least 3 groups"):
        split_dataset(two_groups, group_by="group")
    partial = [_labelled(f"x{k}", 16, 16, k, group="A" if k % 2 else None) for k in range(8)]
    with pytest.raises(ValueError, match="needs a group tag on every structure"):
        split_dataset(partial, group_by="group")
    # Two Cu-only groups and one small alloy group: whenever a Cu-only group is the training split, Al is uncovered.
    al_uncovered = [_labelled(f"a{k}", 32, 0, k, group="A") for k in range(6)] + [_labelled(f"b{k}", 32, 0, k, group="B") for k in range(6)] + [_labelled(f"m{k}", 16, 16, k, group="C") for k in range(2)]
    messages = []
    for seed in range(10):
        try:
            split_dataset(al_uncovered, seed=seed, group_by="group")
        except ValueError as exc:
            messages.append(str(exc))
    assert any(re.search(r"elements \['Al'\] occur in the (val|test) split but not in training", m) for m in messages), messages
    with pytest.raises(ValueError, match="group_by must be one of"):
        split_dataset(two_groups, group_by="random")


def test_the_generated_sample_keeps_its_stratified_split():
    s4 = _code_after("## 4.")
    assert "split_mode = 'stratified'  # every generated structure is an independent draw" in s4
    assert "BYOD_SPLIT = 'auto'" in s4 and "split_mode = BYOD_SPLIT" in s4
    assert "assert not split_summary['groups_in_more_than_one_split']" in s4


@needs_ase
def test_dataset_ceiling_is_stated_and_enforced():
    records = [_labelled(f"r{k}", 16, 16, k) for k in range(MAX_DATASET_RECORDS + 1)]
    with pytest.raises(ValueError, match=f"exceed the dataset ceiling of {MAX_DATASET_RECORDS}"):
        validate_dataset(records)
    md = _markdown()
    assert "A dataset needs 8 to 1,000 labelled structures" in md and "at least 4 training, 1 validation and 1 test structure" in md


@needs_ase
def test_evaluate_scores_any_number_of_structures_in_chunks_within_the_per_call_ceilings():
    calls = []

    class Stand(MaceMaterialsPipeline):
        def predict(self, structures, *, batch_size=8):
            assert len(structures) <= MAX_STRUCTURES_PER_CALL
            assert sum(len(s["symbols"]) for s in structures) <= MAX_ATOMS_PER_CALL
            calls.append(len(structures))
            return {"results": [{"energy": s["energy"] + len(s["symbols"]) * 0.5, "forces": [[0.1, 0.0, 0.0]] * len(s["symbols"])} for s in structures]}

    pipe = Stand(model=None, config={}, device="cpu", weights_dir=Path("w"), source="stand-in")
    records = [_labelled(f"r{k}", 16, 16, k) for k in range(201)]
    metrics = pipe.evaluate(records)
    assert sum(calls) == 201 and max(calls) <= MAX_STRUCTURES_PER_CALL and len(calls) == math.ceil(201 / MAX_STRUCTURES_PER_CALL)
    assert metrics["n_structures"] == 201 and abs(metrics["energy_mae_per_atom"] - 0.5) < 1e-12
    assert abs(metrics["force_mae"] - 0.1 / 3) < 1e-12
    with pytest.raises(ValueError, match="every structure needs energy and forces"):
        pipe.evaluate([{**records[0], "energy": None}])


# --- MMC-m2: BYOD through a path on any runtime, named outputs, new (unscored) structures -----------------------------


def _section4_namespace() -> dict:
    ns = {}
    for module in (metrics_module, pipeline_module, samples_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    return ns


def _run_section4(tmp_path, monkeypatch, **fields) -> dict:
    src = _code_after("## 4.")
    for name, value in fields.items():
        src, n = re.subn(rf"^{name} = .*$", lambda _m, line=f"{name} = {value}": line, src, flags=re.M)
        assert n == 1, name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)  # no Colab: the upload branch must not be reached
    ns = _section4_namespace()
    exec(compile(src, "section-4", "exec"), ns)
    return ns


@needs_ase
def test_byod_path_runs_section_4_without_google_colab_and_names_outputs_as_byod(tmp_path, monkeypatch):
    records = [_labelled(f"t{g}-{k}", 16, 16, k, group=g) for g in "ABCD" for k in range(4)]
    write_dataset_xyz(records, tmp_path / "mine.xyz")
    ns = _run_section4(tmp_path, monkeypatch, USE_BYOD="True", BYOD_PATH=repr(str(tmp_path / "mine.xyz")))
    assert ns["split_summary"]["mode"] == "group" and not ns["split_summary"]["groups_in_more_than_one_split"]
    assert ns["OUTPUTS"]["dataset"] == "outputs/mace_materials_byod_dataset.xyz" and (tmp_path / ns["OUTPUTS"]["dataset"]).is_file()
    assert all("_byod_" in path for path in ns["OUTPUTS"].values())
    assert not (tmp_path / "outputs" / "mace_materials_sample_dataset.xyz").exists()
    assert ns["data_source"].startswith("BYOD file mine.xyz")


@needs_ase
def test_byod_errors_are_actionable_without_colab(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="set BYOD_PATH to the path of one extended-XYZ file"):
        _run_section4(tmp_path, monkeypatch, USE_BYOD="True")
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _run_section4(tmp_path, monkeypatch, USE_BYOD="True", BYOD_PATH="'missing.xyz'")


@needs_ase
def test_trajectory_tags_survive_load_and_write(tmp_path):
    records = [_labelled(f"t{g}-{k}", 16, 16, k, group=g) for g in "AB" for k in range(2)]
    path = write_dataset_xyz(records, tmp_path / "x.xyz")
    loaded = load_byod_dataset(path)
    assert [r["group"] for r in loaded] == ["A", "A", "B", "B"]
    text = path.read_text(encoding="utf-8").replace("group=", "trajectory=")
    (tmp_path / "y.xyz").write_text(text, encoding="utf-8")
    assert [r["group"] for r in load_byod_dataset(tmp_path / "y.xyz")] == ["A", "A", "B", "B"]


def test_byod_new_structures_are_fresh_unscored_geometries_and_byod_asserts_are_readings():
    s9 = _code_after("## 9.")
    byod = s9[s9.index("if USE_BYOD:"): s9.index("else:")]
    assert "for record in test_records[:3]" in byod and "new_rng.normal(0.0, 0.03" in byod and "'energy'" not in byod
    assert "new_metrics = pipe.evaluate(new_records) if labelled_new else None" in s9
    s8 = _code_after("## 8.")
    assert s8.index("if USE_BYOD:") < s8.index("assert test_metrics['energy_mae_per_atom'] < baseline_composition")


# --- MMC-m3 / MMC-m5 / MMC-m6 / MMC-m7: learner-facing statements and the guided layer --------------------------------


def test_uncertainty_torch_load_side_effect_and_data_contract_are_stated():
    md = _markdown()
    assert "**No uncertainty, no domain flag:**" in md and "no per-prediction uncertainty" in md
    assert "`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`" in md and "pass `weights_only=True` yourself" in md
    assert "`{symbols, positions, cell, pbc}`" in md and "{{" not in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "UNC1–UNC6" in registry


def test_guided_layer_and_infrastructure_labels():
    md = _markdown()
    for marker, minimum in (("**Who this is for.**", 1), ("**Input → Model → Output.**", 1), ("**How to use this notebook.**", 1), ("**Roadmap:**", 1), ("**Predict before running:**", 6), ("**What to notice:**", 7), ("<summary>Check your reasoning</summary>", 7), ("## 10. Your turn — change one thing", 1), ("**Predict → Change one thing → Run → Observe → Explain**", 1), ("## Troubleshooting", 1), ("## Glossary", 1), ("## Conclusion (your notes)", 1), ("> **Infrastructure.**", 3)):
        assert md.count(marker) >= minimum, marker
    carried = [c for c in _cells() if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    assert len(carried) == 3 and all(_source(c).startswith("# @title Infrastructure: carried module") and c["metadata"].get("cellView") == "form" for c in carried)


# --- model-backed (local only): two consecutive adaptations start from the same model ---------------------------------


def test_consecutive_adaptations_start_from_the_same_frozen_model_and_both_reload_with_parity(tmp_path):
    pytest.importorskip("mace")
    from mace_materials_pipeline import CONVERTED_WEIGHTS_NAME, DEFAULT_WEIGHTS_DIR, build_sample_structure, to_atoms
    from mace_materials_pipeline.samples import _emt_energy_forces

    if not (DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_NAME).is_file():
        pytest.skip("converted weights not staged")
    import random

    rng = random.Random(21)
    records = []
    for i in range(6):
        record = build_sample_structure("Cu16Al16", strain=rng.uniform(-0.02, 0.02), sigma=0.1, rng=rng)
        record["energy"], record["forces"] = _emt_energy_forces(to_atoms(record))
        record["name"] = f"r-{i}"
        records.append(record)
    pipe = MaceMaterialsPipeline.from_pretrained()
    first = pipe.adapt(records[:4], records[4:], epochs=1, trainable_blocks=1, batch_size=2)
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt(records[:4], records[4:], epochs=1, trainable_blocks=0, batch_size=2)
    assert pipe.reset_to_pretrained()["was_adapted"] is True and pipe.adapter is None
    second = pipe.adapt(records[:4], records[4:], epochs=1, trainable_blocks=0, batch_size=2)
    assert second["history"][0]["val"]["energy_mae_per_atom"] == first["history"][0]["val"]["energy_mae_per_atom"]
    assert second["history"][0]["val"]["force_mae"] == first["history"][0]["val"]["force_mae"]
    artifact = pipe.save_artifact(tmp_path / "adapter")
    reloaded = MaceMaterialsPipeline.from_artifact(artifact)
    a = pipe.predict(records[4:])["results"]
    b = reloaded.predict(records[4:])["results"]
    assert max(abs(x["energy"] - y["energy"]) for x, y in zip(a, b, strict=True)) < 1e-9
