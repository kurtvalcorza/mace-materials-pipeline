"""Energy / force regression metrics and the trivial baselines every adapted number is read against.

Energies are compared per atom (eV/atom) because total energies scale with system size; forces are
compared per Cartesian component (eV/Å). Two baselines need no model: the **composition baseline**
fits one reference energy per element to the training set (the classical "E0" regression, which
cannot see atomic displacements at all) and the **zero-force baseline** predicts every force as zero,
so its MAE is the mean absolute reference force. The third comparison point — the frozen foundation
model after E0 calibration — comes from the pipeline itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _flatten(rows: Sequence[Sequence[float]]) -> list[float]:
    return [float(x) for row in rows for x in row]


def _mae(errors: Sequence[float]) -> float:
    return sum(abs(e) for e in errors) / len(errors) if errors else math.nan


def _rmse(errors: Sequence[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan


def regression_metrics(references: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-atom energy and per-component force errors between labelled structures and predictions."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions differ in length")
    if not references:
        raise ValueError("at least one structure is required")
    energy_errors: list[float] = []
    force_errors: list[float] = []
    n_atoms = 0
    for ref, pred in zip(references, predictions, strict=True):
        if ref.get("energy") is None or ref.get("forces") is None:
            raise ValueError(f"{ref.get('name', 'structure')}: energy and forces are required")
        count = len(ref["symbols"])
        n_atoms += count
        energy_errors.append((float(pred["energy"]) - float(ref["energy"])) / count)
        pf, rf = _flatten(pred["forces"]), _flatten(ref["forces"])
        if len(pf) != len(rf):
            raise ValueError("force arrays differ in shape")
        force_errors.extend(p - r for p, r in zip(pf, rf, strict=True))
    return {
        "n_structures": len(references),
        "n_atoms": n_atoms,
        "energy_mae_per_atom": _mae(energy_errors),
        "energy_rmse_per_atom": _rmse(energy_errors),
        "force_mae": _mae(force_errors),
        "force_rmse": _rmse(force_errors),
        "units": {"energy": "eV/atom", "forces": "eV/Å"},
    }


def _lstsq(rows: Sequence[Sequence[float]], targets: Sequence[float]) -> list[float]:
    """Least squares via numpy (a light dependency already required for structures)."""
    import numpy as np

    solution, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)
    return [float(x) for x in solution]


def composition_baseline(train: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit one energy per element on `train` (E = Σ n_element × E0_element) and score `evaluation`.

    It is blind to geometry by construction: every structure of the same composition gets the same
    energy, and its force prediction is identically zero."""
    if not train or not evaluation:
        raise ValueError("train and evaluation sets must be non-empty")
    elements = sorted({s for record in train for s in record["symbols"]})
    rows = [[record["symbols"].count(el) for el in elements] for record in train]
    e0 = _lstsq(rows, [float(record["energy"]) for record in train])
    predictions = []
    for record in evaluation:
        unknown = sorted(set(record["symbols"]) - set(elements))
        if unknown:
            raise ValueError(f"elements {unknown} absent from the training set; the composition baseline cannot score them")
        energy = sum(record["symbols"].count(el) * e for el, e in zip(elements, e0, strict=True))
        predictions.append({"energy": energy, "forces": [[0.0, 0.0, 0.0] for _ in record["symbols"]]})
    metrics = regression_metrics(evaluation, predictions)
    metrics["e0_ev"] = dict(zip(elements, e0, strict=True))
    metrics["baseline"] = "composition (per-element E0 regression, zero forces)"
    return metrics


def zero_force_baseline(evaluation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Force MAE / RMSE of predicting zero force everywhere: the magnitude a model has to beat."""
    if not evaluation:
        raise ValueError("evaluation set must be non-empty")
    components = [x for record in evaluation for x in _flatten(record["forces"])]
    return {
        "n_structures": len(evaluation),
        "force_mae": _mae(components),
        "force_rmse": _rmse(components),
        "units": {"forces": "eV/Å"},
        "baseline": "zero forces",
    }
