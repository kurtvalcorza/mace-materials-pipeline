import builtins

import pytest

MODEL_LIBRARIES = {"torch", "mace", "e3nn", "safetensors", "huggingface_hub"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def cubic_cell(a: float = 3.6) -> list[list[float]]:
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


def fcc_structure(symbol: str = "Cu", a: float = 3.6, **extra):
    """Four-atom conventional fcc cell as a plain structure mapping (no ase needed)."""
    return {
        "symbols": [symbol] * 4,
        "positions": [[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        "cell": cubic_cell(a),
        "pbc": [True, True, True],
        **extra,
    }
