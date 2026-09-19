"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot and the converted pair are verified —
and the pickled source is statically audited — before torch, e3nn or mace are imported.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import MODEL_LIBRARIES, fcc_structure
from mace_materials_pipeline import validate_dataset, validate_inputs

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mace_materials_pipeline"


def test_package_import_does_not_import_model_libraries(forbid_model_imports):
    import importlib

    import mace_materials_pipeline

    importlib.reload(mace_materials_pipeline)


def test_no_module_level_model_imports():
    """Every torch / mace / e3nn / safetensors / huggingface_hub import is inside a function body."""
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_structures_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="outside the 89 elements"):
        validate_inputs([{"symbols": ["At"], "positions": [[0, 0, 0]]}])
    with pytest.raises(ValueError, match="energy and forces are required"):
        validate_dataset([fcc_structure()] * 8)
