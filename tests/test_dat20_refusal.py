"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from mace_materials_pipeline import metrics as metrics_module
from mace_materials_pipeline import pipeline as pipeline_module
from mace_materials_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/mace_materials_colab.ipynb"
BYOD_CELL = "mace_materials-17"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "mace_materials"
EARLIER = {"records": "old", "splits": "old", "test_records": "old", "test_metrics": "old", "adapt_result": "old", "result_payload": "old"}
needs_ase = pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="ase not installed")


def _namespace(extra: dict | None = None) -> dict:
    ns: dict = {}
    for module in (metrics_module, pipeline_module, samples_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update({**EARLIER, "pipe": "kept", **(extra or {})})
    return ns


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_byod_adapter").mkdir(parents=True)
    for name in ("byod_dataset.xyz", "byod_evaluation_report.json", "byod_result.json", "sample_dataset.xyz", "result.json"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str = "", *, use_byod: bool = True, colab=None, source: str | None = None, extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    source = source[: source.index("write_dataset_xyz(records, OUTPUTS['dataset'])")]
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = _namespace(extra)
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


KEPT = [f"{STEM}_result.json", f"{STEM}_sample_dataset.xyz"]  # the sample's outputs are labelled as the sample's


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "words.xyz").write_text("hello world\n", encoding="utf-8")
    (tmp_path / "count_only.xyz").write_text("3\n", encoding="utf-8")
    (tmp_path / "badnum.xyz").write_text("1\nProperties=species:S:1:pos:R:3 energy=-1.0\nCu a b c\n", encoding="utf-8")
    (tmp_path / "empty.xyz").write_text("", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return {
        "not extended XYZ (the scripted /etc/hostname run)": ("hostname", "BYOD datasets must be extended XYZ"),
        "not an xyz header (XYZError, an OSError)": ("words.xyz", "Expected xyz header"),
        "truncated frame (RuntimeError from StopIteration)": ("count_only.xyz", "the file ends inside a frame"),
        "a coordinate that is not a number": ("badnum.xyz", "could not convert string to float"),
        "no frames": ("empty.xyz", "at least"),
        "a folder": ("folder", "is not a file in this runtime"),
        "missing": ("nope.xyz", "is not a file in this runtime"),
    }


CASES = ["not extended XYZ (the scripted /etc/hostname run)", "not an xyz header (XYZError, an OSError)", "truncated frame (RuntimeError from StopIteration)",
         "a coordinate that is not a number", "no frames", "a folder", "missing"]


def _assert_refused(caught, file_name, rule, tmp_path):
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == KEPT


@needs_ase
@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_byod_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    _assert_refused(caught, file_name, rule, tmp_path)


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.xyz'", 1)
    ns = _namespace()
    with pytest.raises(ValueError, match="^nope.xyz: BYOD_PATH 'nope.xyz' is not a file"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path)
    _assert_refused(caught, "(no file)", "no Colab upload dialog", tmp_path)


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "no file uploaded"), ({"a.xyz": b"1", "b.xyz": b"2"}, "a.xyz, b.xyz", "2 files uploaded")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, colab=_colab(uploaded))
    _assert_refused(caught, file_name, rule, tmp_path)


@needs_ase
def test_valid_byod_file_still_splits(monkeypatch, tmp_path):
    samples_module.write_dataset_xyz(samples_module.generate_sample_dataset(seed=1), tmp_path / "mine.xyz")
    source = _cell(BYOD_CELL).replace("BYOD_SPLIT = 'auto'", "BYOD_SPLIT = 'stratified'", 1)  # the sample has no groups
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "mine.xyz"), source=source)
    assert ns["data_source"].startswith("BYOD file mine.xyz") and ns["train_records"] and ns["test_records"]


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "mine.xyz").write_text("1\n\nCu 0 0 0\n", encoding="utf-8")

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "mine.xyz"), extra={"load_byod_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_byod_result.json").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("sample generator bug")

    with pytest.raises(ValueError, match="sample generator bug") as caught:
        _section4(monkeypatch, tmp_path, use_byod=False, extra={"generate_sample_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)


@needs_ase
@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: ValueError: BYOD datasets must be extended XYZ")
