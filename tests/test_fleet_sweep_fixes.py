"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['mace_materials_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'mace_materials_colab': 5}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


NB = NOTEBOOKS[0]


def _learner_code() -> str:
    return "\n".join(c["source"] for c in _code_cells(_nb(NB)) if not c["metadata"].get("dimer", {}).get("embedded_module"))


def test_swp_r_source_build_is_limited_to_the_named_sdist():
    """mace-torch needs python-hostlist, which has no wheel: only that hash-locked sdist may be built."""
    kernel = _cell(_nb(NB), "# dimer: kernel cell")
    assert "'--require-hashes', '--only-binary', ':all:', '--no-binary', 'python-hostlist'," in kernel
    assert kernel.count("'--no-binary'") == 1
    assert re.search(r"^python-hostlist==\S+ \\\n    --hash=sha256:[0-9a-f]{64}$", LOCK.read_text(encoding="utf-8"), re.M)


# --- SWP-A: quality outcomes are recorded verdicts, never asserts -------------------------------------------------


def test_swp_a_no_quality_assert_remains():
    code = _learner_code()
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"_mae|baseline|zero_shot_test|test_metrics", line)]
    assert quality == []
    # Physics and reload-parity checks are contract integrity and stay hard.
    assert "assert physics['rotation_energy_diff'] < 1e-8 and physics['rotation_force_diff'] < 1e-8" in code
    assert "assert parity['max_abs_energy_diff'] < 1e-9 and parity['max_abs_force_diff'] < 1e-9" in code


def test_swp_a_negative_results_are_recorded_and_do_not_stop_the_notebook():
    s6 = _cell(_nb(NB), "zero_shot_test = pipe.evaluate(test_records)")
    frozen_line = next(line for line in s6.splitlines() if line.startswith("frozen_verdict = "))
    s8 = _cell(_nb(NB), "test_metrics = pipe.evaluate(test_records)")
    block = s8[s8.index("energy_verdict = ") : s8.index("for metric, values in comparison.items():")]
    for adapted_e, adapted_f, expected in ((0.2, 0.3, ("not below the composition baseline", "not below the frozen model")), (0.007, 0.078, ("improved", "improved"))):
        ns = {
            "zero_shot_test": {"force_mae": 0.2079, "energy_mae_per_atom": 3.98},
            "baseline_zero_force": {"force_mae": 0.7376},
            "baseline_composition": {"energy_mae_per_atom": 0.0929},
            "test_metrics": {"energy_mae_per_atom": adapted_e, "force_mae": adapted_f},
            "comparison": {},
        }
        exec(frozen_line, ns)
        exec(block, ns)
        verdicts = ns["comparison"]["verdicts"]
        assert (verdicts["adapted_energy_vs_composition_baseline"], verdicts["adapted_force_vs_frozen_model"]) == expected
        assert verdicts["frozen_vs_zero_force_baseline"] == "force MAE below the zero-force baseline"
    assert s8.index("comparison['verdicts'] = ") < s8.index("json.dump(evaluation_report")


# --- SWP-F: Sections 6 and 7 always start from the frozen model --------------------------------------------------


def test_swp_f_frozen_pipeline_reloads_an_adapted_pipeline(capsys):
    s6 = _cell(_nb(NB), "def frozen_pipeline():")
    helper = s6[s6.index("def frozen_pipeline():") : s6.index("\n\n\nfrozen_pipeline()")]
    loads = []

    class Stand:
        @staticmethod
        def from_pretrained(weights_dir, device, report):
            loads.append((weights_dir, device))
            return types.SimpleNamespace(adapter=None)

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"pipe": types.SimpleNamespace(adapter={"best_epoch": 8}), "MaceMaterialsPipeline": Stand, "WEIGHTS_DIR": "w", "torch": torch}
    exec(helper, ns)
    ns["frozen_pipeline"]()
    assert loads == [("w", "cpu")] and ns["pipe"].adapter is None
    assert "Reloaded the frozen foundation model" in capsys.readouterr().out
    ns["frozen_pipeline"]()
    assert len(loads) == 1


def test_swp_f_scoring_and_training_cells_call_frozen_pipeline_first():
    s6 = _cell(_nb(NB), "zero_shot_test = pipe.evaluate(test_records)")
    assert s6.index("\nfrozen_pipeline()\n") < s6.index("zero_shot_test = pipe.evaluate(")
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("frozen_pipeline()") < s7.index("adapt_result = pipe.adapt(")


# --- SWP-B: BYOD path, guarded upload, refusals that name the file ------------------------------------------------


def _byod(monkeypatch, path: str, tmp_path) -> dict:
    from mace_materials_pipeline.samples import load_byod_dataset

    source = _cell(_nb(NB), "BYOD_PATH = ''")
    block = source[source.index("def byod_file(") : source.index("    data_source = 'BYOD (' + file_name + ')'")]
    block = block.replace("os.makedirs('outputs', exist_ok=True)\n", "")
    monkeypatch.chdir(tmp_path)
    ns = {"Path": Path, "USE_BYOD": True, "BYOD_PATH": path, "load_byod_dataset": load_byod_dataset}
    exec(compile(block, "<section 4 BYOD>", "exec"), ns)
    return ns


def test_swp_b_byod_path_works_outside_colab(monkeypatch, tmp_path):
    from mace_materials_pipeline.samples import generate_sample_dataset, write_dataset_xyz

    _no_colab(monkeypatch)
    data = tmp_path / "mine.xyz"
    write_dataset_xyz(generate_sample_dataset(seed=1)[:3], data)
    ns = _byod(monkeypatch, str(data), tmp_path)
    assert ns["file_name"] == "mine.xyz" and len(ns["records"]) == 3


def test_swp_b_refusals_name_the_file_and_the_rule(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.xyz.* is not a file"):
        _byod(monkeypatch, str(tmp_path / "missing.xyz"), tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _byod(monkeypatch, "", tmp_path)
    wrong = tmp_path / "data.csv"
    wrong.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^data\.csv: not a readable labelled extended-XYZ dataset .*extended XYZ"):
        _byod(monkeypatch, str(wrong), tmp_path)
    broken = tmp_path / "broken.xyz"
    broken.write_text("2\nLattice=\"x\"\nCu 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^broken\.xyz: not a readable labelled extended-XYZ dataset"):
        _byod(monkeypatch, str(broken), tmp_path)


def test_swp_b_cancelled_or_multiple_uploads_are_refused(monkeypatch, tmp_path):
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"a.xyz": b"", "b.xyz": b""})
    with pytest.raises(RuntimeError, match="got 2"):
        _byod(monkeypatch, "", tmp_path)


def test_swp_g_checkpoint_answers_agree_with_the_recorded_run():
    """Numbers in the worked answers are the recorded 2026-09-19 Kaggle T4 run's and the 2026-09-18 pre-flight's."""
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    assert "energy_mae_per_atom: {composition_baseline: 0.09291, frozen_model: 3.981, adapted: 0.00725}" in record
    assert "force_mae: {zero_force_baseline: 0.7376, frozen_model: 0.2079, adapted: 0.07803}" in record
    assert "calibration Al +3.768 / Cu +4.176 eV" in record
    assert "max_abs_energy_diff: 1.421e-14, max_abs_force_diff: 7.772e-15" in record
    for quoted in ("3.981 eV/atom", "0.09291", "0.2079 eV/Å", "0.7376", "0.00725 eV/atom", "0.07803 eV/Å", "Al +3.768 eV and Cu +4.176 eV", "1.4e-14 eV", "7.8e-15 eV/Å"):
        assert quoted in markdown, quoted
