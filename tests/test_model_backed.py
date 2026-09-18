"""Model-backed checks that run only where mace-torch and the converted weights are present (local
pre-flight; skipped in the lightweight CI). They exercise the physics contract of the rebuilt model —
rotation equivariance, permutation invariance, analytic-vs-finite-difference forces — and one short
adaptation with artifact round trip."""

from __future__ import annotations

import json
import random

import pytest

from mace_materials_pipeline import (
    CONVERTED_WEIGHTS_NAME,
    DEFAULT_WEIGHTS_DIR,
    MaceMaterialsPipeline,
    build_sample_structure,
    to_atoms,
)
from mace_materials_pipeline.samples import _emt_energy_forces

pytest.importorskip("mace")
np = pytest.importorskip("numpy")
if not (DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_NAME).is_file():
    pytest.skip("converted weights not staged", allow_module_level=True)


@pytest.fixture(scope="module")
def pipe():
    return MaceMaterialsPipeline.from_pretrained()


@pytest.fixture(scope="module")
def structure():
    rng = random.Random(3)
    record = build_sample_structure("Cu16Al16", strain=0.01, sigma=0.1, rng=rng)
    record["energy"], record["forces"] = _emt_energy_forces(to_atoms(record))
    return record


def test_rebuilt_model_identity(pipe):
    assert pipe.source.startswith("converted")
    assert sum(p.numel() for p in pipe.model.parameters()) == 8_221_984
    assert not any(p.requires_grad for p in pipe.model.parameters())


def test_predictions_are_equivariant_and_forces_are_the_gradient(pipe, structure):
    base = pipe.predict([structure])["results"][0]
    assert base["stress"] is not None and len(base["node_energies"]) == 32
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = {
        **structure,
        "positions": (np.array(structure["positions"]) @ q.T).tolist(),
        "cell": (np.array(structure["cell"]) @ q.T).tolist(),
    }
    rot = pipe.predict([rotated])["results"][0]
    assert abs(rot["energy"] - base["energy"]) < 1e-9
    assert np.abs(np.array(rot["forces"]) - np.array(base["forces"]) @ q.T).max() < 1e-9
    perm = rng.permutation(32)
    permuted = {
        **structure,
        "symbols": [structure["symbols"][i] for i in perm],
        "positions": [structure["positions"][i] for i in perm],
    }
    pm = pipe.predict([permuted])["results"][0]
    assert abs(pm["energy"] - base["energy"]) < 1e-9
    h = 1e-4
    plus = np.array(structure["positions"])
    plus[5, 1] += h
    minus = np.array(structure["positions"])
    minus[5, 1] -= h
    e_plus = pipe.predict([{**structure, "positions": plus.tolist()}])["results"][0]["energy"]
    e_minus = pipe.predict([{**structure, "positions": minus.tolist()}])["results"][0]["energy"]
    assert abs(-(e_plus - e_minus) / (2 * h) - base["forces"][5][1]) < 1e-6
    molecule = {"symbols": ["O", "H", "H"], "positions": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}
    assert pipe.predict([molecule])["results"][0]["stress"] is None


def test_short_adaptation_and_artifact_round_trip(pipe, tmp_path):
    rng = random.Random(11)
    records = []
    for i in range(6):
        record = build_sample_structure("Cu32", strain=rng.uniform(-0.02, 0.02), sigma=0.1, rng=rng)
        record["energy"], record["forces"] = _emt_energy_forces(to_atoms(record))
        record["name"] = f"cu-{i}"
        records.append(record)
    before = pipe.evaluate(records[4:])
    result = pipe.adapt(records[:4], records[4:], epochs=1, trainable_blocks=0, batch_size=2)
    assert result["n_trainable"] == 2_192 and result["calibration"]["elements"] == ["Cu"]
    assert result["history"][0]["note"].startswith("frozen model")
    after = pipe.evaluate(records[4:])
    # E0 calibration alone removes the level-of-theory offset (eV/atom) between PBE and EMT
    assert before["energy_mae_per_atom"] > 1.0 and after["energy_mae_per_atom"] < 0.5
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert all(name.startswith(("readouts.", "atomic_energies_fn.", "scale_shift.")) for name in manifest["tensors"])
    reloaded = MaceMaterialsPipeline.from_artifact(artifact)
    a = pipe.predict(records[4:])["results"]
    b = reloaded.predict(records[4:])["results"]
    assert max(abs(x["energy"] - y["energy"]) for x, y in zip(a, b, strict=True)) == 0.0
