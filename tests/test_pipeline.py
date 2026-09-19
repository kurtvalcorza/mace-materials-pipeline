"""Offline tests: identity, snapshot verification and staging, the static pickle audit, converted-config
drift refusal, structure validation and the ase round trip. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from conftest import cubic_cell, fcc_structure
from mace_materials_pipeline import (
    CONVERTED_CONFIG_NAME,
    CONVERTED_SHA256,
    CONVERTED_WEIGHTS_NAME,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SCHEMA,
    MANIFEST_NAME,
    MAX_ATOMS_PER_STRUCTURE,
    MAX_STRUCTURES_PER_CALL,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PICKLE_AUDIT_SHA256,
    SOURCE_MODEL_NAME,
    SOURCE_MODEL_SHA256,
    SUPPORTED_ATOMIC_NUMBERS,
    audit_model_file,
    from_atoms,
    stage_missing_files,
    structure_digest,
    to_atoms,
    validate_inputs,
    verify_converted,
    verify_snapshot,
)
from mace_materials_pipeline import pipeline as pl

# --- identity -------------------------------------------------------------------------------------


def test_identity_constants_and_manifest():
    assert MODEL_ID == "mace-foundations/mace-mp-0"
    assert len(MODEL_REVISION) == 40 and MODEL_REVISION.lower() == MODEL_REVISION
    assert MODEL_LICENSE == "mit"
    assert MODEL_KEY == "mace-mp-0b2-small"
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "dimer_hf_snapshot"
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert set(by_path) == {"README.md", SOURCE_MODEL_NAME}
    assert by_path[SOURCE_MODEL_NAME]["sha256"] == SOURCE_MODEL_SHA256
    assert by_path[SOURCE_MODEL_NAME]["bytes"] == pl.SOURCE_MODEL_BYTES
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert len(SUPPORTED_ATOMIC_NUMBERS) == pl.NUM_ELEMENTS == 89
    assert 84 not in SUPPORTED_ATOMIC_NUMBERS and 94 in SUPPORTED_ATOMIC_NUMBERS
    assert set(CONVERTED_SHA256) == {CONVERTED_CONFIG_NAME, CONVERTED_WEIGHTS_NAME}
    assert all(len(d) == 64 for d in (*CONVERTED_SHA256.values(), PICKLE_AUDIT_SHA256, SOURCE_MODEL_SHA256))


def test_committed_converted_config_matches_the_pinned_digest_and_constants():
    path = DEFAULT_WEIGHTS_DIR / CONVERTED_CONFIG_NAME
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CONVERTED_SHA256[CONVERTED_CONFIG_NAME]
    assert len(data) == pl.CONVERTED_BYTES[CONVERTED_CONFIG_NAME]
    config = pl._config_from_json(json.loads(data))
    assert config["interaction_cls"] == "RealAgnosticDensityResidualInteractionBlock"
    assert config["pair_repulsion"] is True and config["distance_transform"] == "Agnesi"
    assert config["gate"] == "silu" and config["radial_type"] == "bessel"


# --- snapshot verification and staging ------------------------------------------------------------


def _write_snapshot(root, *, tamper=False, model_bytes=b"x" * 16):
    root.mkdir(parents=True, exist_ok=True)
    readme = b"---\nlicense: mit\n---\n"
    (root / "README.md").write_bytes(readme)
    (root / SOURCE_MODEL_NAME).write_bytes(model_bytes)
    files = [
        {"path": "README.md", "bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()},
        {"path": SOURCE_MODEL_NAME, "bytes": len(model_bytes), "sha256": hashlib.sha256(model_bytes).hexdigest()},
    ]
    if tamper:
        files[0]["sha256"] = "0" * 64
    (root / MANIFEST_NAME).write_text(
        json.dumps({"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}), encoding="utf-8"
    )


def test_verify_snapshot_checks_digests_and_the_pinned_source_constant(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path)
    # the digest matches the manifest but not the pinned constant: refused
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_refuses_missing_manifest_or_wrong_identity(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        verify_snapshot(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}))
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": []}))
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(tmp_path)


def test_verify_converted_checks_both_files(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / CONVERTED_CONFIG_NAME).write_bytes(b"{}")
    (tmp_path / CONVERTED_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="size"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path)
    (tmp_path / SOURCE_MODEL_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def downloader(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(b"x" * 16)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == [SOURCE_MODEL_NAME]
    assert fetched == [SOURCE_MODEL_NAME]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []


def test_stage_refuses_manifest_for_another_model(tmp_path, forbid_model_imports):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": "0" * 40, "files": []}))
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


def test_from_pretrained_refuses_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError):
        pl.MaceMaterialsPipeline.from_pretrained(weights_dir=tmp_path)
    _write_snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        pl.MaceMaterialsPipeline.from_pretrained(weights_dir=tmp_path)
    # DIMER-hosted case: no manifest, converted pair required and digest-checked
    (tmp_path / MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        pl.MaceMaterialsPipeline.from_pretrained(weights_dir=tmp_path, require_source=False)


def test_require_source_false_never_stages_even_when_the_manifest_is_present(tmp_path, monkeypatch, forbid_model_imports):
    """A checkout keeps the committed manifest beside the converted pair and no pickle: the converted-only
    path must not touch the source at all (it used to fall into stage_missing_files whenever the manifest existed)."""
    _write_snapshot(tmp_path)
    (tmp_path / SOURCE_MODEL_NAME).unlink()

    def refuse(*args, **kwargs):
        raise AssertionError("stage_missing_files must not run with require_source=False")

    monkeypatch.setattr(pl, "stage_missing_files", refuse)
    # only the converted pair is looked at: with it absent the error names the converted file, not the source
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        pl.MaceMaterialsPipeline.from_pretrained(weights_dir=tmp_path, require_source=False)
    # and the default (require_source=True) still goes through staging
    with pytest.raises(AssertionError, match="must not run"):
        pl.MaceMaterialsPipeline.from_pretrained(weights_dir=tmp_path)


def test_convert_model_refuses_a_wrong_sized_source_before_unpickling(tmp_path, forbid_model_imports):
    (tmp_path / SOURCE_MODEL_NAME).write_bytes(b"not a model")
    with pytest.raises(ValueError, match="size"):
        pl.convert_model(tmp_path)


# --- static pickle audit ----------------------------------------------------------------------------


def _torch_like_archive(pickle_bytes: bytes, extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model/data.pkl", pickle_bytes)
        archive.writestr("model/version", b"3\n")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _global(module: str, name: str) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def test_audit_accepts_allow_listed_globals_and_reports_them(tmp_path, forbid_model_imports):
    pickle_bytes = (
        b"\x80\x02" + _global("torch._utils", "_rebuild_tensor_v2") + _global("mace.modules.models", "ScaleShiftMACE") + b"."
    )
    path = tmp_path / SOURCE_MODEL_NAME
    path.write_bytes(_torch_like_archive(pickle_bytes))
    report = audit_model_file(path)
    assert report["globals"] == ["mace.modules.models.ScaleShiftMACE", "torch._utils._rebuild_tensor_v2"]
    assert report["violations"] == [] and len(report["audit_sha256"]) == 64


def test_audit_refuses_globals_outside_the_allow_list(tmp_path, forbid_model_imports):
    pickle_bytes = b"\x80\x02" + _global("os", "system") + b"."
    path = tmp_path / SOURCE_MODEL_NAME
    path.write_bytes(_torch_like_archive(pickle_bytes))
    with pytest.raises(ValueError, match="global outside allow-list: os.system"):
        audit_model_file(path)


def test_audit_refuses_a_stack_global_and_exec_tokens_in_generated_code(tmp_path, forbid_model_imports):
    # STACK_GLOBAL with two unicode strings (protocol 4): builtins.eval
    stack_global = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93."
    path = tmp_path / SOURCE_MODEL_NAME
    path.write_bytes(_torch_like_archive(stack_global))
    with pytest.raises(ValueError, match="builtins.eval"):
        audit_model_file(path)
    code = "def forward(self, x):\n    return subprocess.run(x)\n"
    encoded = code.encode()
    fx_code = b"\x80\x02" + b"X" + len(encoded).to_bytes(4, "little") + encoded + b"."
    path.write_bytes(_torch_like_archive(fx_code))
    with pytest.raises(ValueError, match="forbidden token 'subprocess'"):
        audit_model_file(path)


def test_audit_refuses_an_import_block_outside_the_allow_list(tmp_path, forbid_model_imports):
    block = b"import torch\nimport os\n"
    pickle_bytes = b"\x80\x02" + b"X" + len(block).to_bytes(4, "little") + block + b"."
    path = tmp_path / SOURCE_MODEL_NAME
    path.write_bytes(_torch_like_archive(pickle_bytes))
    with pytest.raises(ValueError, match="import line outside allow-list: import os"):
        audit_model_file(path)


def test_audit_walks_nested_torchscript_archives(tmp_path, forbid_model_imports):
    nested = _torch_like_archive(
        b"\x80\x02" + _global("torch.jit._pickle", "restore_type_tag") + b".",
        {"model/code/x.py": b"def forward(self):\n    return os.getcwd()\n"},
    )
    text = nested.decode("latin-1").encode("utf-8")
    outer = b"\x80\x02" + b"X" + len(text).to_bytes(4, "little") + text + b"."
    path = tmp_path / SOURCE_MODEL_NAME
    path.write_bytes(_torch_like_archive(outer))
    with pytest.raises(ValueError, match="forbidden token 'os' in torchscript source"):
        audit_model_file(path)


@pytest.mark.skipif(not (DEFAULT_WEIGHTS_DIR / SOURCE_MODEL_NAME).is_file(), reason="source .model not staged")
def test_audit_of_the_real_source_matches_the_pinned_digest(forbid_model_imports):
    report = audit_model_file()
    assert report["audit_sha256"] == PICKLE_AUDIT_SHA256
    assert report["violations"] == []
    assert report["nested_archives"] == 18 and report["fx_code_strings"] == 28 and report["torchscript_sources"] == 30
    roots = {name.split(".")[0] for name in report["globals"]}
    assert roots <= {"torch", "e3nn", "mace", "collections", "_codecs", "__torch__", "__builtin__"}


# --- converted-config drift --------------------------------------------------------------------------


def test_config_drift_is_refused_before_model_imports(forbid_model_imports):
    config = json.loads((DEFAULT_WEIGHTS_DIR / CONVERTED_CONFIG_NAME).read_text(encoding="utf-8"))
    for key, bad in (("r_max", 6.0), ("num_interactions", 3), ("hidden_irreps", "64x0e"), ("dtype", "float32")):
        with pytest.raises(ValueError, match=f"converted config {key}"):
            pl._config_from_json({**config, key: bad})
    with pytest.raises(ValueError, match="atomic_numbers"):
        pl._config_from_json({**config, "atomic_numbers": list(range(1, 90))})


# --- structure validation ---------------------------------------------------------------------------


def test_validate_inputs_manifest_and_acceptance(forbid_model_imports):
    assert INPUT_SCHEMA["structures"] == [1, MAX_STRUCTURES_PER_CALL]
    report = validate_inputs(
        [fcc_structure(), {"symbols": ["O", "H", "H"], "positions": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}]
    )
    assert report["n_structures"] == 2 and report["n_atoms"] == 7
    assert report["elements"] == ["Cu", "H", "O"] and report["periodic"] == [True, False]
    assert report["labelled"] == 0 and 0.9 < report["min_distance"] < 1.0


@pytest.mark.parametrize(
    ("structure", "message"),
    [
        ({"symbols": ["Po"], "positions": [[0, 0, 0]]}, "outside the 89 elements"),
        ({"symbols": ["Xx"], "positions": [[0, 0, 0]]}, "not an element symbol"),
        ({"symbols": ["Cu", "Cu"], "positions": [[0, 0, 0], [0.3, 0, 0]]}, "below the 0.5"),
        ({"symbols": ["Cu"], "positions": [[0, 0]]}, "positions shape"),
        ({"symbols": ["Cu"], "positions": [[0, 0, float("nan")]]}, "finite"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "pbc": True}, "no cell is given"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "cell": [[1, 0, 0], [0, 1, 0]], "pbc": True}, "3x3"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "cell": [[300, 0, 0], [0, 3, 0], [0, 0, 3]], "pbc": True}, "exceeds 200"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "cell": [[0.4, 0, 0], [0, 3, 0], [0, 0, 3]], "pbc": True}, "cell height"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "cell": [[3, 0, 0], [3, 0, 0], [0, 0, 3]], "pbc": True}, "singular"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "pbc": [True, False]}, "three bools"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "energy": float("inf")}, "energy must be a finite"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "forces": [[0, 0]]}, "forces must be a finite"),
        ({"symbols": ["Cu"], "positions": [[0, 0, 0]], "name": 5}, "name must be a string"),
        ({"symbols": [], "positions": []}, "non-empty"),
        (
            {"symbols": ["Cu"] * (MAX_ATOMS_PER_STRUCTURE + 1), "positions": [[0, 0, 0]] * (MAX_ATOMS_PER_STRUCTURE + 1)},
            "exceed the ceiling",
        ),
    ],
)
def test_validate_inputs_rejections(structure, message, forbid_model_imports):
    with pytest.raises(ValueError, match=message):
        validate_inputs([structure])


def test_validate_inputs_call_ceilings(forbid_model_imports):
    with pytest.raises(ValueError, match="at least one structure"):
        validate_inputs([])
    with pytest.raises(ValueError, match="list of structure mappings"):
        validate_inputs(fcc_structure())
    with pytest.raises(ValueError, match=f"exceed the ceiling of {MAX_STRUCTURES_PER_CALL}"):
        validate_inputs([fcc_structure()] * (MAX_STRUCTURES_PER_CALL + 1))
    big = {"symbols": ["Cu"] * 512, "positions": [[3.0 * i, 0, 0] for i in range(512)]}
    with pytest.raises(ValueError, match="atoms in one call exceed"):
        validate_inputs([big] * 9)


def test_periodic_images_count_for_the_distance_floor(forbid_model_imports):
    # two atoms 3.3 Å apart inside a 3.6 Å cell are 0.3 Å apart through the boundary
    structure = {"symbols": ["Cu", "Cu"], "positions": [[0, 0, 0], [3.3, 0, 0]], "cell": cubic_cell(3.6), "pbc": True}
    with pytest.raises(ValueError, match="below the 0.5"):
        validate_inputs([structure])
    structure["pbc"] = False
    assert validate_inputs([structure])["min_distance"] == pytest.approx(3.3)


def test_structure_digest_ignores_labels_and_extra_keys(forbid_model_imports):
    base = fcc_structure()
    assert structure_digest(base) == structure_digest({**base, "energy": -1.0, "forces": [[0, 0, 0]] * 4, "name": "x"})
    assert structure_digest(base) != structure_digest({**base, "symbols": ["Al"] * 4})


def test_ase_round_trip_keeps_labels(forbid_model_imports):
    structure = fcc_structure(energy=-14.5, forces=[[0.1, 0, 0]] * 4, name="cu4")
    atoms = to_atoms(structure)
    back = from_atoms(atoms)
    assert back["symbols"] == structure["symbols"] and back["pbc"] == [True, True, True]
    assert back["energy"] == -14.5 and back["forces"][0] == [0.1, 0.0, 0.0] and back["name"] == "cu4"
    assert back["cell"] == cubic_cell(3.6)


def test_trainable_prefixes_are_bounded(forbid_model_imports):
    assert pl._trainable_prefixes(0) == pl.TRAINABLE_ALWAYS
    assert pl._trainable_prefixes(1) == (*pl.TRAINABLE_ALWAYS, "interactions.1.", "products.1.")
    assert len(pl._trainable_prefixes(2)) == len(pl.TRAINABLE_ALWAYS) + 4
    with pytest.raises(ValueError, match="trainable_blocks"):
        pl._trainable_prefixes(3)
