"""Offline tests for the dataset contract, the EMT-labelled sample, stratified splits, metrics,
baselines, extended-XYZ I/O and artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import json
import math
import random

import pytest

from conftest import fcc_structure
from mace_materials_pipeline import (
    CONVERTED_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_COMPOSITIONS,
    MaceMaterialsPipeline,
    build_sample_structure,
    composition_baseline,
    composition_key,
    dataset_digest,
    generate_sample_dataset,
    load_byod_dataset,
    regression_metrics,
    split_dataset,
    structure_digest,
    validate_dataset,
    write_dataset_xyz,
    zero_force_baseline,
)
from mace_materials_pipeline import pipeline as pl
from mace_materials_pipeline import samples as sm

# Geometry digest of the default sample (labels excluded): pins the generator, not EMT's floating point.
SAMPLE_GEOMETRY_DIGEST = "24a8ba06e692016155b5ebc351a838953c8e9a26c7353b77be106b30cded8ae2"


@pytest.fixture(scope="module")
def sample():
    return generate_sample_dataset(seed=42)


# --- sample dataset -----------------------------------------------------------------------------------


def test_sample_dataset_is_deterministic_labelled_and_balanced(sample, forbid_model_imports):
    assert len(sample) == 48
    report = validate_dataset(sample)
    assert report["compositions"] == {"Cu32": 16, "Al32": 16, "Al16Cu16": 16}
    assert report["elements"] == ["Al", "Cu"] and report["max_atoms"] == 32 and report["labelled"]
    again = generate_sample_dataset(seed=42)
    assert dataset_digest(again) == report["digest"]
    assert [structure_digest(s) for s in again] == [structure_digest(s) for s in sample]
    assert generate_sample_dataset(seed=1)[0]["positions"] != sample[0]["positions"]
    assert all(s["label_source"] == sm.SAMPLE_LABEL_SOURCE for s in sample)


def test_sample_geometry_digest_is_pinned(sample, forbid_model_imports):
    geometry = json.dumps([structure_digest(s) for s in sample]).encode()
    import hashlib

    assert hashlib.sha256(geometry).hexdigest() == SAMPLE_GEOMETRY_DIGEST


def test_sample_structures_are_rattled_strained_fcc_cells(sample, forbid_model_imports):
    for structure in sample:
        assert structure["n_atoms"] if "n_atoms" in structure else len(structure["symbols"]) == 32
        assert structure["pbc"] == [True, True, True]
        assert abs(structure["strain"]) <= sm.STRAIN_RANGE and structure["rattle_sigma"] in sm.RATTLE_SIGMAS
        cell = structure["cell"]
        assert cell[0][1] == 0.0 and cell[0][0] == pytest.approx(cell[1][1]) == pytest.approx(cell[2][2])
        # forces are the negative gradient of a translation-invariant energy: they sum to ~0
        for axis in range(3):
            assert abs(sum(f[axis] for f in structure["forces"])) < 1e-6
    energies = {c: [s["energy"] / 32 for s in sample if s["composition"] == c] for c in SAMPLE_COMPOSITIONS}
    # EMT cohesive-scale energies for these metals are small positive numbers per atom near equilibrium
    assert all(-1.0 < e < 1.0 for values in energies.values() for e in values)
    # strain and rattle move the energy; the spread within a composition is what the fine-tune must learn
    assert all(max(v) - min(v) > 0.01 for v in energies.values())


def test_composition_baseline_is_blind_to_geometry(sample, forbid_model_imports):
    splits = split_dataset(sample, seed=0)
    baseline = composition_baseline(splits["train"], splits["test"])
    assert set(baseline["e0_ev"]) == {"Al", "Cu"}
    assert baseline["force_mae"] == pytest.approx(zero_force_baseline(splits["test"])["force_mae"])
    # same composition -> same predicted energy, so the error is the within-composition spread
    assert 0.0 < baseline["energy_mae_per_atom"] < 0.5
    with pytest.raises(ValueError, match="absent from the training set"):
        composition_baseline(splits["train"], [fcc_structure("Ni", energy=0.0, forces=[[0, 0, 0]] * 4)])


def test_build_sample_structure_rejects_unknown_composition(forbid_model_imports):
    with pytest.raises(ValueError, match="composition must be one of"):
        build_sample_structure("Fe32", strain=0.0, sigma=0.1, rng=random.Random(0))
    with pytest.raises(ValueError, match="multiple of 3"):
        generate_sample_dataset(size=4)


# --- dataset validation and split -------------------------------------------------------------------


def test_validate_dataset_rejections_are_actionable(forbid_model_imports):
    good = fcc_structure(energy=-1.0, forces=[[0, 0, 0]] * 4)
    with pytest.raises(ValueError, match="at least 8 are required"):
        validate_dataset([good] * 3)
    with pytest.raises(ValueError, match="energy and forces are required"):
        validate_dataset([fcc_structure()] * 8)
    with pytest.raises(ValueError, match="duplicate structure name"):
        validate_dataset([{**good, "name": "same"}] * 8)
    assert validate_dataset([fcc_structure()] * 8, require_labels=False)["labelled"] is False
    report = validate_dataset([{**good, "name": f"s{i}"} for i in range(8)])
    assert report["compositions"] == {"Cu4": 8} and report["n_atoms"] == 32


def test_composition_key_is_hill_ordered(forbid_model_imports):
    assert composition_key({"symbols": ["Cu", "Al", "Cu"]}) == "Al1Cu2"


def test_split_is_stratified_disjoint_and_seeded(sample, forbid_model_imports):
    splits = split_dataset(sample, val_fraction=0.2, test_fraction=0.25, seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"train": 27, "val": 9, "test": 12}
    names = [s["name"] for part in splits.values() for s in part]
    assert len(names) == len(set(names)) == 48
    for part in splits.values():
        assert {s["composition"] for s in part} == {"Cu32", "Al32", "Al16Cu16"}
    assert [s["name"] for s in split_dataset(sample, seed=0)["test"]] == [s["name"] for s in splits["test"]]
    assert [s["name"] for s in split_dataset(sample, seed=1)["test"]] != [s["name"] for s in splits["test"]]
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(sample, val_fraction=0.5, test_fraction=0.5)


# --- metrics ------------------------------------------------------------------------------------------


def test_regression_metrics_units_and_errors(forbid_model_imports):
    ref = [fcc_structure(energy=-4.0, forces=[[1, 0, 0]] * 4)]
    pred = [{"energy": -8.0, "forces": [[0, 0, 0]] * 4}]
    metrics = regression_metrics(ref, pred)
    assert metrics["energy_mae_per_atom"] == pytest.approx(1.0) and metrics["energy_rmse_per_atom"] == pytest.approx(1.0)
    assert metrics["force_mae"] == pytest.approx(1 / 3) and metrics["force_rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert metrics["n_atoms"] == 4 and metrics["units"] == {"energy": "eV/atom", "forces": "eV/Å"}
    with pytest.raises(ValueError, match="differ in length"):
        regression_metrics(ref, [])
    with pytest.raises(ValueError, match="energy and forces are required"):
        regression_metrics([fcc_structure()], pred)


# --- extended XYZ -------------------------------------------------------------------------------------


def test_xyz_round_trip_and_rejections(sample, tmp_path, forbid_model_imports):
    path = write_dataset_xyz(sample[:6], tmp_path / "sample.xyz")
    back = load_byod_dataset(path)
    assert len(back) == 6
    for original, loaded in zip(sample[:6], back, strict=True):
        assert loaded["symbols"] == original["symbols"] and loaded["pbc"] == original["pbc"]
        assert loaded["energy"] == pytest.approx(original["energy"], abs=1e-6)
        assert (
            max(
                abs(a - b)
                for fa, fb in zip(loaded["forces"], original["forces"], strict=True)
                for a, b in zip(fa, fb, strict=True)
            )
            < 1e-6
        )
        assert (
            max(
                abs(a - b)
                for pa, pb in zip(loaded["positions"], original["positions"], strict=True)
                for a, b in zip(pa, pb, strict=True)
            )
            < 1e-6
        )
    assert validate_dataset(back, min_records=6)["labelled"]
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.xyz")
    (tmp_path / "data.csv").write_text("a,b\n")
    with pytest.raises(ValueError, match="extended XYZ"):
        load_byod_dataset(tmp_path / "data.csv")
    unlabelled = write_dataset_xyz([fcc_structure(name="cu4")], tmp_path / "geom.xyz")
    assert load_byod_dataset(unlabelled)[0]["energy"] is None


# --- artifacts ----------------------------------------------------------------------------------------


class _NoModel:
    def state_dict(self):
        return {}


def _pipeline_without_model():
    return MaceMaterialsPipeline(model=_NoModel(), config={}, device="cpu", weights_dir=pl.DEFAULT_WEIGHTS_DIR, source="test")


def test_save_artifact_requires_adaptation(forbid_model_imports):
    with pytest.raises(ValueError, match="call adapt"):
        _pipeline_without_model().save_artifact("unused")


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": pl.ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "converted_sha256": dict(CONVERTED_SHA256)},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({**manifest, "base_model": {**manifest["base_model"], "revision": "0" * 40}})
    )
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    bad_digests = {**manifest["base_model"], "converted_sha256": {k: "0" * 64 for k in CONVERTED_SHA256}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "base_model": bad_digests}))
    with pytest.raises(ValueError, match="converted-base digests"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_adapt_validates_hyperparameters_before_model_work(forbid_model_imports):
    pipe = _pipeline_without_model()
    records = [fcc_structure(energy=-1.0, forces=[[0, 0, 0]] * 4, name=f"s{i}") for i in range(4)]
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(records, trainable_blocks=5)
    with pytest.raises(ValueError, match="at least 4"):
        pipe.adapt(records[:2])
