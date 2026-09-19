"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import mace_materials_pipeline as pkg
from mace_materials_pipeline import INPUT_SCHEMA, MAX_STRUCTURES_PER_CALL, MIN_INTERATOMIC_DISTANCE, validate_inputs


def test_input_schema_names_the_ceilings():
    assert INPUT_SCHEMA["structures"] == [1, MAX_STRUCTURES_PER_CALL]
    assert INPUT_SCHEMA["min_interatomic_distance_angstrom"] == MIN_INTERATOMIC_DISTANCE
    assert "89 elements" in INPUT_SCHEMA["elements"]
    assert "eV" in INPUT_SCHEMA["units"]


def test_validate_inputs_reports_names_and_periodicity():
    report = validate_inputs([{"symbols": ["He"], "positions": [[0.0, 0.0, 0.0]], "name": "lonely"}])
    assert report == {
        "n_structures": 1,
        "n_atoms": 1,
        "elements": ["He"],
        "periodic": [False],
        "min_distance": float("inf"),
        "labelled": 0,
    }


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "MaceMaterialsPipeline" in pkg.__all__ and "audit_model_file" in pkg.__all__
