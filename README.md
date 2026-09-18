# MACE Materials Pipeline

DIMER-oriented pipeline for **MACE-MP-0b2 small** (`mace-foundations/mace-mp-0`, the universal machine-learned interatomic potential from Batatia et al.), pinned to an immutable Hugging Face revision. The repository exposes energy, per-atom energy, force and stress prediction for atomic structures, a labelled-dataset contract with explicit ceilings, per-element reference-energy calibration, a bounded fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `mace-foundations/mace-mp-0`
- Revision: `e291ace2bfae073c3ebc7ae2f9479a525989baa7`
- Source asset: `mace-mp-0b2-small.model` (67,622,684 bytes, SHA-256 `d5773bf9…`), byte-identical to the upstream GitHub release asset `ACEsuit/mace-mp` → `mace_mp_0b2/mace-small-density-agnesi-stress.model` (2024-11-12)
- Upstream weight license: MIT
- Upstream task: universal interatomic potential trained on Materials Project PBE/PBE+U trajectories (89 elements); this repository uses it for single-point energies, forces and stresses and for fine-tuning to local reference data
- Runtime: `mace-torch==0.3.16` + `e3nn==0.4.4` + `torch==2.14.0` + `ase==3.29.0` — the model class comes from PyPI, **no Hub-hosted code and no served pickle** (see below)
- Repository adaptation: **E2E** (E0 calibration plus bounded fine-tuning of the readouts, reference energies, scale/shift and the last *n* interaction blocks, with a portable safetensors adapter)

## Two things to know before you start

**The upstream asset is a pickle, and it is never served.** `mace-mp-0b2-small.model` is a pickled torch module: unpickling it imports 61 torch / e3nn / mace globals, `exec`s 10 fx-generated source strings and compiles 30 TorchScript sources from 18 nested archives. This repository treats that as executable serialization (fleet asset specification §11): `audit_model_file()` disassembles the file statically without executing it and refuses anything outside the torch / e3nn / mace allow-lists (audit digest pinned); `convert_model()` unpickles the digest-verified, audit-clean file **once** and writes a code-free pair — `mace-mp-0b2-small.config.json` (3,279 bytes, committed) + `mace-mp-0b2-small.safetensors` (67,400,566 bytes, 86 tensors, git-ignored) — whose digests are pinned in `pipeline.py`; `from_pretrained()` rebuilds `ScaleShiftMACE` from the installed `mace-torch` and loads that pair with `strict=True`. The conversion is deterministic and measured: 80/80 carried tensors bit-identical, energies within 2.8×10⁻¹⁴ eV and forces within 3.6×10⁻¹⁵ eV/Å of the pickled model and of upstream's own `MACECalculator`. `docs/WEIGHTS.md` records the nine-buffer layout difference between the pickle and the current `mace-torch`.

**The tutorial's reference labels are EMT, not DFT.** The sample dataset is 48 rattled, strained fcc supercells of Cu, Al and a random Cu₁₆Al₁₆ alloy labelled by ASE's effective-medium-theory potential — a real, deterministic potential for these metals and a different level of theory from the PBE data MACE-MP-0 learned, which is what makes the fine-tuning story honest (the frozen model predicts forces well and energies several eV/atom off; calibration and adaptation close the gap on held-out structures) — and also a toy, which is why no number here says anything about DFT accuracy.

## Quick start

```python
from mace_materials_pipeline import (
    MaceMaterialsPipeline, composition_baseline, generate_sample_dataset, split_dataset, zero_force_baseline,
)

pipe = MaceMaterialsPipeline.from_pretrained()   # verifies the snapshot, audits + converts the pickle once, loads safetensors strictly
records = generate_sample_dataset()              # 48 EMT-labelled fcc supercells (Cu32, Al32, Cu16Al16)
result = pipe.predict(records[:2])
print(result["results"][0]["energy_per_atom"], result["results"][0]["forces"][0], result["results"][0]["stress"][0])

splits = split_dataset(records)
print(composition_baseline(splits["train"], splits["test"])["energy_mae_per_atom"])   # blind to geometry
print(zero_force_baseline(splits["test"])["force_mae"])
print(pipe.evaluate(splits["test"]))             # the frozen foundation model: forces good, energies offset
pipe.adapt(splits["train"], splits["val"])       # E0 calibration + bounded Adam fine-tuning (last block)
print(pipe.evaluate(splits["test"]))
pipe.save_artifact("outputs/adapter")
```

`predict()` and `evaluate()` take 1..32 structures (`MAX_STRUCTURES_PER_CALL`), each a `{symbols, positions, cell, pbc}` mapping in ångström with 1..512 atoms (`MAX_ATOMS_PER_STRUCTURE`) from the 89 elements MACE-MP-0 supports, at most 4,096 atoms per call, no two atoms closer than 0.5 Å (periodic images included), and a finite 3×3 cell whenever any periodic flag is set. `evaluate()`, `calibrate_e0()` and `adapt()` additionally need `energy` (eV) and `forces` (eV/Å) on every structure. Validation is structural: nothing checks that a structure is chemically sensible.

## Weights layout

```
weights/mace-mp-0b2-small/   README.md  mace-mp-0b2-small.model  dimer-base-manifest.json
                             mace-mp-0b2-small.config.json  (committed, converted)
                             mace-mp-0b2-small.safetensors  (git-ignored, converted)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of both manifest entries and of the converted pair when present), converts the source when the pair is absent, and refuses on the first mismatch. With `require_source=False` a digest-verified converted pair is accepted without the pickle — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audit, the conversion and the DIMER hosting notes; the source pickle and the safetensors file are git-ignored.

## Sample data

`generate_sample_dataset(seed=42)` builds 16 structures each of Cu₃₂, Al₃₂ and Cu₁₆Al₁₆ — 2×2×2 fcc supercells with isotropic strain drawn from ±3 % and Gaussian rattles of σ = 0.05, 0.10 or 0.15 Å — and labels them with ASE's EMT calculator at run time (no download, no committed data). Every structure of one composition has the same atoms, so the composition baseline (one energy per element, fitted on the training split) cannot see the displacements at all; `split_dataset` stratifies by composition. `write_dataset_xyz` / `load_byod_dataset` round-trip extended XYZ, the BYOD format.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the tensors the adaptation could change — readouts, reference energies, scale/shift and the unfrozen blocks; about 16 MB with the default one block, 20 KB readouts-only) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digests, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `MaceMaterialsPipeline.from_artifact(dir)` re-verifies the base, checks the manifest and digest before deserialising, and overlays the tensors onto a freshly rebuilt base.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles, temporary manifests and plain-Python structures, never the weights; `tests/test_model_backed.py` runs only where `mace-torch` and the converted weights are present (physics checks, a one-epoch adaptation, artifact round trip) and is skipped in CI.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mace-materials-pipeline/blob/main/tutorials/mace_materials_colab.ipynb)

`tutorials/mace_materials_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It downloads the pinned pickle, audits and converts it in the runtime (the report is printed before the model loads), and then runs the sample path: validation, split, physics checks on the frozen model, baselines, calibration + fine-tuning, held-out evaluation, inference on new structures, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the notebook source passes all static checks and one local CPU pre-flight execution of the committed blob is recorded; a clean run in a supported hosted runtime is still required (see `STATUS.md` and `docs/release-verification.md`).

## Licensing

- Upstream weights: MIT (`mace-foundations/mace-mp-0`; the `ACEsuit/mace` code and the `mace_mp_0b2` release assets are MIT), staged from the pinned revision and converted, not modified, into the served pair.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
