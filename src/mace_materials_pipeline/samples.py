"""Tutorial dataset, dataset validation, stratified split and extended-XYZ I/O for the MACE pipeline.

The default dataset is generated in code (no download): rattled and strained face-centred-cubic
2×2×2 supercells of copper, aluminium and a random Cu₁₆Al₁₆ substitutional alloy, **labelled with
ASE's effective-medium-theory potential (EMT)**. EMT is a real, deterministic, analytic interatomic
potential for these metals — not a DFT code and not a surrogate of MACE — so the labels are a
different level of theory from the PBE data MACE-MP-0 was trained on. That is exactly the situation a
foundation potential is fine-tuned in: the frozen model already predicts sensible forces, but its
energies sit on a different reference and its curvature differs, and the adaptation has to close the
gap on held-out structures.

Design choices made so the trivial baselines sit where they should:

* Every structure of one composition has the **same atom count and the same element multiset**, so a
  per-element energy regression (the composition baseline) predicts one energy per composition and
  cannot see the rattle or the strain at all.
* Displacement amplitudes and strains are drawn independently of composition, so nothing about a
  structure's label is recoverable from which elements it contains.
* Structures are labelled with forces as well as energies, so the force error — where the foundation
  model already starts well and must improve further — is measured alongside the energy error.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_ATOMS_PER_STRUCTURE, _check_structure, from_atoms, structure_digest, to_atoms

SAMPLE_COMPOSITIONS = ("Cu32", "Al32", "Cu16Al16")
SAMPLE_SIZE = 48
LATTICE_CONSTANTS = {"Cu": 3.60, "Al": 4.05}  # Å, fcc conventional cell
SUPERCELL = (2, 2, 2)  # 32 atoms
STRAIN_RANGE = 0.03  # isotropic, ±3 %
RATTLE_SIGMAS = (0.05, 0.10, 0.15)  # Å, Gaussian displacement per Cartesian component
SAMPLE_LABEL_SOURCE = "ase.calculators.emt.EMT"
MIN_RECORDS = 8
MAX_DATASET_RECORDS = 1_000  # BYOD ceiling: about an hour of CPU fine-tuning at the default epochs
# Smallest split sizes the notebook runs end to end with: adapt() needs 4 training structures, Section 8 scores a
# non-empty validation split and every section reads a non-empty test split.
MIN_SPLIT_SIZES = {"train": 4, "val": 1, "test": 1}
# How split_dataset keeps related structures together (SPL3/SPL5). "stratified" treats every structure as an
# independent draw and is only right for data like the generated sample; frames of one trajectory or relaxation are
# near-duplicates and must not straddle splits, so BYOD defaults to "auto".
SPLIT_MODES = ("auto", "group", "composition", "stratified")
GROUP_INFO_KEYS = ("group", "trajectory")  # extended-XYZ per-frame info keys read by load_byod_dataset


def _emt_energy_forces(atoms: Any) -> tuple[float, list[list[float]]]:
    from ase.calculators.emt import EMT

    atoms = atoms.copy()
    atoms.calc = EMT()
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces().tolist()
    atoms.calc = None
    return energy, forces


def build_sample_structure(composition: str, *, strain: float, sigma: float, rng: random.Random) -> dict[str, Any]:
    """One rattled, strained fcc 2×2×2 supercell of the named composition (geometry only, no labels)."""
    import numpy as np
    from ase.build import bulk

    if composition not in SAMPLE_COMPOSITIONS:
        raise ValueError(f"composition must be one of {SAMPLE_COMPOSITIONS}")
    if composition == "Cu16Al16":
        a0 = 0.5 * (LATTICE_CONSTANTS["Cu"] + LATTICE_CONSTANTS["Al"])  # Vegard's law
        atoms = bulk("Cu", "fcc", a=a0, cubic=True) * SUPERCELL
        symbols = ["Cu"] * len(atoms)
        for index in rng.sample(range(len(atoms)), 16):
            symbols[index] = "Al"
        atoms.set_chemical_symbols(symbols)
    else:
        element = composition[:2]
        atoms = bulk(element, "fcc", a=LATTICE_CONSTANTS[element], cubic=True) * SUPERCELL
    atoms.set_cell(atoms.cell * (1.0 + strain), scale_atoms=True)
    displacement = np.array([[rng.gauss(0.0, sigma) for _ in range(3)] for _ in range(len(atoms))])
    atoms.positions = atoms.positions + displacement
    structure = from_atoms(atoms)
    structure["name"] = f"{composition}-s{strain:+.4f}-r{sigma:.2f}"
    structure["composition"] = composition
    structure["strain"] = strain
    structure["rattle_sigma"] = sigma
    return structure


def generate_sample_dataset(*, seed: int = 42, size: int = SAMPLE_SIZE) -> list[dict[str, Any]]:
    """Deterministic EMT-labelled dataset: `size` structures cycling through the three compositions."""
    if not isinstance(size, int) or not 3 <= size <= 96 or size % 3:
        raise ValueError("size must be a multiple of 3 in 3..96")
    rng = random.Random(seed)
    records = []
    for index in range(size):
        composition = SAMPLE_COMPOSITIONS[index % len(SAMPLE_COMPOSITIONS)]
        strain = rng.uniform(-STRAIN_RANGE, STRAIN_RANGE)
        sigma = RATTLE_SIGMAS[rng.randrange(len(RATTLE_SIGMAS))]
        structure = build_sample_structure(composition, strain=strain, sigma=sigma, rng=rng)
        structure["name"] = f"{structure['name']}-{index:02d}"
        energy, forces = _emt_energy_forces(to_atoms(structure))
        structure["energy"] = energy
        structure["forces"] = forces
        structure["label_source"] = SAMPLE_LABEL_SOURCE
        records.append(structure)
    return records


def composition_key(structure: Mapping[str, Any]) -> str:
    """Hill-order formula string, e.g. Al16Cu16, used to stratify splits."""
    counts: dict[str, int] = {}
    for symbol in structure["symbols"]:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a labelled dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of structure mappings")
    if len(records) < min_records:
        raise ValueError(f"{len(records)} records; at least {min_records} are required")
    if len(records) > MAX_DATASET_RECORDS:
        raise ValueError(f"{len(records)} records exceed the dataset ceiling of {MAX_DATASET_RECORDS}; use a subset")
    checked = []
    names: set[str] = set()
    for index, record in enumerate(records):
        structure = _check_structure(record, index)
        if require_labels and (structure["energy"] is None or structure["forces"] is None):
            raise ValueError(f"{structure['name']}: energy and forces are required")
        if structure["name"] in names:
            raise ValueError(f"duplicate structure name {structure['name']!r}")
        names.add(structure["name"])
        structure["composition"] = composition_key(structure)
        for key in ("strain", "rattle_sigma", "label_source"):
            if key in record:
                structure[key] = record[key]
        group = record.get("group")
        if group is not None:
            if not isinstance(group, (str, int)) or isinstance(group, bool) or not str(group).strip():
                raise ValueError(f"{structure['name']}: group must be a non-empty string or integer")
            structure["group"] = str(group)
        checked.append(structure)
    compositions: dict[str, int] = {}
    for structure in checked:
        compositions[structure["composition"]] = compositions.get(structure["composition"], 0) + 1
    return {
        "records": checked,
        "n_records": len(checked),
        "n_atoms": sum(s["n_atoms"] for s in checked),
        "compositions": compositions,
        "elements": sorted({s for c in checked for s in c["symbols"]}),
        "max_atoms": max(s["n_atoms"] for s in checked),
        "labelled": all(s["energy"] is not None and s["forces"] is not None for s in checked),
        "digest": dataset_digest(checked),
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over per-structure geometry digests and labels rounded to 1e-6."""
    parts = []
    for record in records:
        energy = None if record.get("energy") is None else round(float(record["energy"]), 6)
        forces = None if record.get("forces") is None else [[round(float(x), 6) for x in row] for row in record["forces"]]
        parts.append([structure_digest(record), energy, forces])
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode("utf-8")).hexdigest()


def _effective_split_mode(checked: Sequence[Mapping[str, Any]], group_by: str) -> str:
    if group_by not in SPLIT_MODES:
        raise ValueError(f"group_by must be one of {SPLIT_MODES}")
    tagged = sum(1 for s in checked if s.get("group") is not None)
    if group_by == "group" and tagged != len(checked):
        raise ValueError(
            f"group_by='group' needs a group tag on every structure; {len(checked) - tagged} of {len(checked)} have none. "
            "Add group=<id> (or trajectory=<id>) to each frame's comment line in the extended XYZ, or use 'composition'"
        )
    if group_by == "auto":
        return "group" if tagged == len(checked) else "composition"
    return group_by


def split_group_key(structure: Mapping[str, Any], mode: str) -> str:
    """The unit a split keeps together: the structure's group tag in 'group' mode, else its composition."""
    return str(structure["group"]) if mode == "group" else composition_key(structure)


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 0,
    group_by: str = "stratified",
) -> dict[str, list[dict[str, Any]]]:
    """Split a labelled dataset into train / val / test.

    `group_by="stratified"` shuffles within each composition and cuts the fractions there, so every split sees
    every composition: right only when every structure is an independent draw (the generated sample).
    `"group"` (every record carries a `group` tag, e.g. a trajectory id), `"composition"` (one group per formula)
    and `"auto"` (group tags when every record has one, else composition) assign whole groups to one split, so
    near-duplicate frames never straddle train and test; the fractions then count groups, not structures.
    Every split must reach MIN_SPLIT_SIZES, and every element of the validation and test splits must occur in
    training (the composition baseline and the E0 calibration are fitted there); otherwise a ValueError names
    the rule that failed and what to change."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, min_records=4)["records"]
    mode = _effective_split_mode(checked, group_by)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    if mode == "stratified":
        by_composition: dict[str, list[dict[str, Any]]] = {}
        for structure in checked:
            by_composition.setdefault(structure["composition"], []).append(structure)
        for composition in sorted(by_composition):
            group = list(by_composition[composition])
            rng.shuffle(group)
            n_test = max(1, round(len(group) * test_fraction))
            n_val = round(len(group) * val_fraction)
            splits["test"].extend(group[:n_test])
            splits["val"].extend(group[n_test : n_test + n_val])
            splits["train"].extend(group[n_test + n_val :])
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for structure in checked:
            groups.setdefault(split_group_key(structure, mode), []).append(structure)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError(
                f"a {mode} split needs at least 3 groups (one each for train, validation and test); found "
                f"{len(keys)}: {keys}. Supply more trajectories or compositions, tag frames with group=<id>, or, only "
                "if every structure is an independent draw, use 'stratified'"
            )
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_fraction)) if val_fraction > 0 else 0
        n_test = min(max(1, round(len(keys) * test_fraction)), len(keys) - n_val - 1)
        for key in keys[:n_test]:
            splits["test"].extend(groups[key])
        for key in keys[n_test : n_test + n_val]:
            splits["val"].extend(groups[key])
        for key in keys[n_test + n_val :]:
            splits["train"].extend(groups[key])
    for name, minimum in MIN_SPLIT_SIZES.items():
        if len(splits[name]) < minimum:
            what = {"train": "training", "val": "validation", "test": "test"}[name]
            raise ValueError(
                f"the {what} split has {len(splits[name])} structure(s); at least {minimum} are needed "
                f"(splits {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} with {mode} grouping). "
                "Supply more structures or groups, or change VAL_FRACTION / TEST_FRACTION"
            )
    train_elements = {s for record in splits["train"] for s in record["symbols"]}
    for name in ("val", "test"):
        missing = sorted({s for record in splits[name] for s in record["symbols"]} - train_elements)
        if missing:
            raise ValueError(
                f"elements {missing} occur in the {name} split but not in training, so the composition baseline and "
                "the E0 calibration cannot cover them; change SEED, add structures containing them to more groups, "
                "or regroup the data"
            )
    return splits


def split_report(splits: Mapping[str, Sequence[Mapping[str, Any]]], *, group_by: str = "stratified") -> dict[str, Any]:
    """Which groups landed in which split, and any group found in more than one split (expected only for
    'stratified', where compositions are shared by design)."""
    everything = [s for name in ("train", "val", "test") for s in splits[name]]
    mode = _effective_split_mode(everything, group_by)
    key_mode = "group" if mode == "group" else "composition"
    members = {name: sorted({split_group_key(s, key_mode) for s in splits[name]}) for name in ("train", "val", "test")}
    train, val, test = (set(members[n]) for n in ("train", "val", "test"))
    return {
        "mode": mode,
        "unit": "group tag" if mode == "group" else "composition",
        "sizes": {name: len(splits[name]) for name in ("train", "val", "test")},
        "groups": members,
        "groups_in_more_than_one_split": sorted((train & (val | test)) | (val & test)),
    }


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read an extended-XYZ file (`ase.io.read`, all frames) into structure mappings with labels from
    `energy` / `forces` (info/arrays or the attached single-point results). A per-frame `group=<id>` (or
    `trajectory=<id>`) on the comment line becomes the record's `group`, which `split_dataset` keeps together."""
    import ase.io

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    if file_path.suffix.lower() not in (".xyz", ".extxyz"):
        raise ValueError("BYOD datasets must be extended XYZ (.xyz / .extxyz)")
    frames = ase.io.read(str(file_path), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    records = []
    for index, atoms in enumerate(frames):
        if len(atoms) > MAX_ATOMS_PER_STRUCTURE:
            raise ValueError(f"frame {index}: {len(atoms)} atoms exceed {MAX_ATOMS_PER_STRUCTURE}")
        record = from_atoms(atoms, name=atoms.info.get("name") or f"{file_path.stem}-{index:04d}")
        for key in GROUP_INFO_KEYS:
            if atoms.info.get(key) is not None:
                record["group"] = str(atoms.info[key])
                break
        records.append(record)
    return records


def write_dataset_xyz(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write structures (with energy/forces when present) as extended XYZ, the shape BYOD expects."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = []
    for record in records:
        atoms = to_atoms(record)
        if record.get("group") is not None:
            atoms.info["group"] = str(record["group"])  # kept so a re-read file splits the same way
        atoms.info.pop("energy", None)
        forces = atoms.arrays.pop("forces", None)
        if record.get("energy") is not None and forces is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=float(record["energy"]), forces=forces)
        frames.append(atoms)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(out), frames, format="extxyz")
    return out
