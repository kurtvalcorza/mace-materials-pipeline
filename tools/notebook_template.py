"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E interatomic-potential workflow: the pinned MACE-MP-0b2 small source
asset is digest-verified, statically audited and converted once into a code-free safetensors pair,
an EMT-labelled dataset of rattled fcc supercells is generated in code, validated and split, the
frozen foundation model is checked for equivariance and force consistency, its zero-shot errors are
measured against two trivial baselines, a bounded fine-tuning runs in the kernel, held-out energy and
force errors are reported, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "mace-materials-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/mace_materials_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-mace--foundations%2Fmace--mp--0-ffcc4d?style=flat",
        "https://huggingface.co/mace-foundations/mace-mp-0",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ACEsuit%2Fmace-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ACEsuit/mace",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2401.00096-b31b1b.svg", "https://arxiv.org/abs/2401.00096"),
]

TEMPLATE = {
    "package": "mace_materials_pipeline",
    "repo_name": REPO,
    "stem": "mace_materials",
    "notebook_name": "mace_materials_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies (torch, mace-torch, e3nn, ase, "
        "numpy, safetensors, huggingface-hub), stages and digest-verifies the pinned MACE-MP-0b2 small source asset (67.6 MB "
        "pickled module from the Hub), statically audits every global and generated source string that pickle would execute "
        "and refuses anything outside the torch / e3nn / mace allow-lists, unpickles it exactly once to write a code-free "
        "JSON config + safetensors pair whose digests are pinned in the module, rebuilds the model from the installed "
        "mace-torch and loads it strictly, generates a deterministic 48-structure dataset in code (rattled, strained fcc "
        "Cu, Al and Cu₁₆Al₁₆ supercells labelled by ASE's EMT potential — no download), validates the structures and the "
        "dataset contract, splits them by composition into train/validation/test sets, checks the frozen model for rotation "
        "equivariance, permutation invariance and analytic-versus-finite-difference forces, measures its zero-shot energy and "
        "force errors against a composition baseline and a zero-force baseline, calibrates the per-element reference "
        "energies and runs a bounded Adam fine-tuning of the readouts plus the last interaction block, evaluates energy and "
        "force errors on the held-out test split, predicts three freshly generated structures, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The "
        "default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about two minutes of model time after the "
        "download."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled structures as an extended-XYZ file (one frame per structure, a `Lattice` and `pbc` when periodic, an `energy` "
        "in eV and a `forces` array in eV/Å per frame). They pass through the same validation, composition-stratified split, "
        "baselines, calibration, adaptation, held-out evaluation, inference, artifact export and reload-parity cells as the "
        "generated sample. The expected schema, the element support and the ceilings are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "MaceMaterialsPipeline",
    "model_load": "MaceMaterialsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "mace-mp-0b2-small",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "mace", "e3nn", "ase"],
    "title": "MACE-MP-0b2 small — DIMER E2E interatomic-potential fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "energy, force and stress prediction for atomic structures and bounded fine-tuning of the foundation interatomic potential",
    "intro": (
        "MACE-MP-0 is a universal machine-learned interatomic potential (Batatia et al., 2023): an equivariant message-passing "
        "model trained on Materials Project PBE/PBE+U trajectories that predicts a total energy, per-atom energies, forces and "
        "stress for structures of 89 elements. The *0b2 small* checkpoint is the 8.2 M-parameter, two-interaction, 128x0e "
        "variant with Agnesi radial transform and ZBL pair repulsion, released November 2024. Forces are the analytic gradient "
        "of the energy, so the model needs autograd even at inference — this notebook checks that the forces it returns really "
        "are that gradient.\n\n"
        "The upstream artifact is a **pickled torch module** (`.model`), which the fleet asset specification treats as executable "
        "serialization. So this pipeline never serves it: Section 3 downloads the pinned file, verifies its digest, disassembles "
        "it statically (every global it would import, every fx-generated source it would `exec`, every TorchScript source it "
        "would compile), refuses anything outside the torch / e3nn / mace allow-lists, unpickles it exactly once, and writes a "
        "code-free JSON config + safetensors pair whose digests are pinned in the carried module. The model you run is rebuilt "
        "from the installed `mace-torch` and loaded with `strict=True`; the pickle is provenance, not runtime.\n\n"
        "The tutorial dataset is generated in code and labelled with **ASE's effective-medium-theory potential (EMT)** — a real, "
        "deterministic interatomic potential for Cu and Al, but a different level of theory from the PBE data MACE-MP-0 was "
        "trained on. That is exactly the fine-tuning situation: the frozen model already predicts sensible forces, its energies "
        "sit on a different reference and its curvature differs, and the adaptation has to close the gap on held-out structures. "
        "Every structure of one composition has the same atoms, so a per-element energy regression (the composition baseline) "
        "cannot see the displacements at all."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify an "
        "immutable source asset, read a static audit of a pickled checkpoint and see it converted into a code-free serving "
        "pair; validate atomic structures and split a labelled dataset by composition; check a foundation potential for "
        "equivariance, permutation invariance and gradient-consistent forces; measure zero-shot energy and force errors against "
        "two trivial baselines; calibrate reference energies and run a bounded fine-tuning with explicit hyperparameters; "
        "evaluate per-atom energy and force errors on an independent test split; predict new structures; and export a "
        "safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "molecular-dynamics driving, geometry optimisation, phonons, the published MACE-MP-0 benchmarks, the medium and large "
        "checkpoints, multi-head replay fine-tuning against the Materials Project data, LAMMPS or CUDA-equivariance "
        "deployment, and any claim that an EMT-labelled fcc supercell stands in for a DFT dataset. The repository exposes none "
        "of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — a 32-atom cell is scored in under 0.1 s and the default fine-tuning takes about a minute — and CUDA is used automatically when present. The model runs in float64, as upstream ships it.",
        "- **Knowledge:** what an interatomic potential, a periodic cell and a force are; why energies are compared per atom; and what MAE and RMSE mean.",
        "- **Executable serialization handled explicitly:** the pinned `.model` file is a pickle. It is digest-verified, statically audited against allow-lists (the audit digest is pinned) and unpickled **once** to produce the safetensors pair the model is actually loaded from. No Hub-hosted Python module is imported; `mace-torch` is installed from PyPI at a pinned version.",
        "- **Data contract:** structures are `{{symbols, positions, cell, pbc}}` in Å with optional `energy` (eV) and `forces` (eV/Å); elements limited to the 89 MACE-MP-0 supports (Z 1–83, 89–94); 1..512 atoms per structure, at most 32 structures and 4,096 atoms per call; no two atoms closer than 0.5 Å (periodic images included); cell vectors under 200 Å and periodic cell heights of at least 1 Å; a dataset needs at least 8 labelled structures with unique names. BYOD accepts extended XYZ.",
        "- **Validation is structural, not chemical:** nothing checks that a structure is charge-neutral, near equilibrium or physically meaningful — a random cloud of supported atoms is scored without complaint, and EMT labels are only meaningful for the metals it parameterises.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — proprietary alloy compositions or unpublished DFT datasets are exactly that. The default path uploads nothing.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample dataset, validation and split\n\n"
                "The default dataset is generated in code with a fixed seed: 16 structures each of Cu₃₂, Al₃₂ and a random "
                "substitutional Cu₁₆Al₁₆ alloy — 2×2×2 fcc supercells with an isotropic strain drawn from ±3 % and Gaussian "
                "rattles of σ = 0.05, 0.10 or 0.15 Å — labelled with energies and forces from ASE's EMT calculator. "
                "`validate_dataset` checks every structure (element support, distances, cell/pbc, label shapes) and the dataset "
                "contract before any model runs; `split_dataset` shuffles within each composition and cuts 20 % validation / "
                "25 % test, so every split sees every composition.\n\n"
                "Look for: 48 structures, three compositions of 16, 1,536 atoms, splits 27/9/12, EMT energies of a few tenths "
                "of an eV per atom, and a written `outputs/{stem}_sample_dataset.xyz` in the extended-XYZ shape BYOD expects. "
                "Four refusal probes follow — an unsupported element, overlapping atoms, periodicity without a cell, and a "
                "malformed force array — each rejected before `torch` does anything."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset(seed=SEED)\n"
                "    data_source = f'generated fcc supercells labelled by {{SAMPLE_LABEL_SOURCE}} (seed {{SEED}}, {{SAMPLE_SIZE}} structures)'\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['val'], splits['test']\n"
                "write_dataset_xyz(records, 'outputs/{stem}_sample_dataset.xyz')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'n_atoms': dataset_manifest['n_atoms'], 'compositions': dataset_manifest['compositions']}})\n"
                "print({{'elements': dataset_manifest['elements'], 'max_atoms': dataset_manifest['max_atoms'], 'labelled': dataset_manifest['labelled'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'energy_per_atom_range_eV': (round(min(r['energy'] / len(r['symbols']) for r in records), 4), round(max(r['energy'] / len(r['symbols']) for r in records), 4))}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})\n"
                "print({{'example': {{k: (v if not isinstance(v, list) else f'list[{{len(v)}}]') for k, v in records[0].items()}}}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'unsupported element': {{'symbols': ['Po', 'Cu'], 'positions': [[0, 0, 0], [2.5, 0, 0]]}},\n"
                "    'overlapping atoms': {{'symbols': ['Cu', 'Cu'], 'positions': [[0, 0, 0], [0.3, 0, 0]]}},\n"
                "    'pbc without a cell': {{'symbols': ['Cu'], 'positions': [[0, 0, 0]], 'pbc': True}},\n"
                "    'malformed forces': {{**records[0], 'forces': [[0.0, 0.0]] * len(records[0]['symbols'])}},\n"
                "}}\n"
                "for name, structure in probes.items():\n"
                "    try:\n"
                "        validate_inputs([structure])\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Zero-shot prediction and physics checks on the frozen model\n\n"
                "`pipe.predict` returns, per structure, the total energy (eV), the per-atom energy, per-atom energy "
                "contributions, forces (eV/Å) and — for fully periodic cells — the stress tensor (eV/Å³). The model is the "
                "pinned foundation potential, untouched.\n\n"
                "Five checks say whether the rebuilt model behaves like an interatomic potential should. **Rotation:** rotating "
                "the cell and every atom leaves the energy unchanged and rotates the forces with it (equivariance). "
                "**Translation** and **permutation** of atoms change nothing. **Gradient consistency:** the force on one atom "
                "equals the central finite difference of the energy along that coordinate, to ~1e-8 eV/Å — the forces are the "
                "analytic gradient, not a separate head. **Extensivity:** a 2×1×1 supercell has twice the energy. Look for "
                "differences at the 1e-13 level for the symmetry checks and 1e-8 for the finite difference; a rebuilt model with "
                "a wrong config would fail these before any accuracy number."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n\n"
                "s0 = test_records[0]\n"
                "t0 = time.perf_counter()\n"
                "prediction = pipe.predict(test_records[:8])\n"
                "print({{'structures': 8, 'seconds': round(time.perf_counter() - t0, 2), 'units': prediction['units']}})\n"
                "base = prediction['results'][0]\n"
                "print({{'name': base['name'], 'energy_eV': round(base['energy'], 4), 'energy_per_atom_eV': round(base['energy_per_atom'], 4), 'reference_eV': round(s0['energy'], 4), 'max_force_eV_A': round(float(np.abs(base['forces']).max()), 4), 'stress_diag_eV_A3': [round(base['stress'][i][i], 5) for i in range(3)]}})\n\n"
                "rng = np.random.default_rng(0)\n"
                "q, _ = np.linalg.qr(rng.normal(size=(3, 3)))\n"
                "if np.linalg.det(q) < 0:\n"
                "    q[:, 0] *= -1\n"
                "rotated = {{**s0, 'positions': (np.array(s0['positions']) @ q.T).tolist(), 'cell': (np.array(s0['cell']) @ q.T).tolist()}}\n"
                "rot = pipe.predict([rotated])['results'][0]\n"
                "translated = {{**s0, 'positions': (np.array(s0['positions']) + 1.234).tolist()}}\n"
                "tr = pipe.predict([translated])['results'][0]\n"
                "perm = rng.permutation(len(s0['symbols']))\n"
                "permuted = {{**s0, 'symbols': [s0['symbols'][i] for i in perm], 'positions': [s0['positions'][i] for i in perm]}}\n"
                "pm = pipe.predict([permuted])['results'][0]\n"
                "h = 1e-4\n"
                "plus, minus = np.array(s0['positions']), np.array(s0['positions'])\n"
                "plus[3, 0] += h\n"
                "minus[3, 0] -= h\n"
                "e_plus = pipe.predict([{{**s0, 'positions': plus.tolist()}}])['results'][0]['energy']\n"
                "e_minus = pipe.predict([{{**s0, 'positions': minus.tolist()}}])['results'][0]['energy']\n"
                "fd_force = -(e_plus - e_minus) / (2 * h)\n"
                "supercell = pipe.predict([from_atoms(to_atoms(s0) * (2, 1, 1))])['results'][0]\n"
                "physics = {{\n"
                "    'rotation_energy_diff': abs(rot['energy'] - base['energy']),\n"
                "    'rotation_force_diff': float(np.abs(np.array(rot['forces']) - np.array(base['forces']) @ q.T).max()),\n"
                "    'translation_energy_diff': abs(tr['energy'] - base['energy']),\n"
                "    'permutation_energy_diff': abs(pm['energy'] - base['energy']),\n"
                "    'permutation_force_diff': float(np.abs(np.array(pm['forces']) - np.array(base['forces'])[perm]).max()),\n"
                "    'finite_difference_force': fd_force,\n"
                "    'analytic_force': base['forces'][3][0],\n"
                "    'gradient_consistency_diff': abs(fd_force - base['forces'][3][0]),\n"
                "    'extensivity_diff_per_atom': abs(supercell['energy'] - 2 * base['energy']) / supercell['n_atoms'],\n"
                "    'batch_vs_single_energy_diff': abs(pipe.predict([s0])['results'][0]['energy'] - base['energy']),\n"
                "}}\n"
                "for key, value in physics.items():\n"
                "    print({{key: f'{{value:.3e}}'}})\n"
                "assert physics['rotation_energy_diff'] < 1e-8 and physics['rotation_force_diff'] < 1e-8\n"
                "assert physics['permutation_energy_diff'] < 1e-8 and physics['translation_energy_diff'] < 1e-8\n"
                "assert physics['gradient_consistency_diff'] < 1e-6\n"
                "assert physics['extensivity_diff_per_atom'] < 1e-8"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the zero-shot error\n\n"
                "Three numbers frame everything that follows. The **composition baseline** fits one energy per element on the "
                "training split (the classical E0 regression) and predicts zero forces: it is blind to geometry by construction, "
                "so its energy error is the within-composition spread of the labels and its force error is the mean absolute "
                "reference force. The **zero-force baseline** is that force number alone. The **frozen foundation model** is "
                "evaluated as is: expect a force MAE well below the zero baseline — MACE-MP-0 already knows how metals push on "
                "each other — but an energy error of several eV per atom, because PBE total energies and EMT energies sit on "
                "different absolute references. That offset is the first thing adaptation removes.\n\n"
                "Energies are per atom (eV/atom); forces are per Cartesian component (eV/Å)."
            ),
            "code": (
                "baseline_composition = composition_baseline(train_records, test_records)\n"
                "baseline_zero_force = zero_force_baseline(test_records)\n"
                "t0 = time.perf_counter()\n"
                "zero_shot_test = pipe.evaluate(test_records)\n"
                "zero_shot_seconds = round(time.perf_counter() - t0, 2)\n"
                "print({{'composition_baseline': {{'energy_mae_per_atom': round(baseline_composition['energy_mae_per_atom'], 4), 'force_mae': round(baseline_composition['force_mae'], 4), 'e0_eV': {{k: round(v, 4) for k, v in baseline_composition['e0_ev'].items()}}}}}})\n"
                "print({{'zero_force_baseline': {{'force_mae': round(baseline_zero_force['force_mae'], 4), 'force_rmse': round(baseline_zero_force['force_rmse'], 4)}}}})\n"
                "print({{'frozen_foundation_model': {{k: round(v, 4) for k, v in zero_shot_test.items() if isinstance(v, float)}}, 'seconds': zero_shot_seconds}})\n"
                "assert zero_shot_test['force_mae'] < baseline_zero_force['force_mae']"
            ),
        },
        {
            "md": (
                "## 7. Calibrate and fine-tune\n\n"
                "`pipe.adapt` does two things in order. First it **calibrates the per-element reference energies**: a least-squares "
                "fit of one offset per element between the training labels and the frozen predictions, added to the model's "
                "atomic energies — two numbers on a Cu/Al dataset, and the cheapest possible move to a new level of theory. That "
                "calibrated frozen model is recorded as epoch 0 so every later number is comparable to it. Then it trains the "
                "readouts, the reference energies, the scale/shift block and — with `TRAINABLE_BLOCKS = 1` — the last "
                "interaction and product blocks: 1.92 M of 8.22 M parameters, Adam at a fixed learning rate, loss = 100 × "
                "MSE(per-atom energy) + 10 × MSE(force components), batches of 4, no scheduler, seeded shuffling. The epoch "
                "with the lowest validation loss is kept.\n\n"
                "Watch the validation energy MAE fall from ~0.1 eV/atom (the calibrated frozen model) to below 0.01 and the force "
                "MAE roughly halve over eight epochs, about a minute on CPU. The counter-examples are worth running once: "
                "`TRAINABLE_BLOCKS = 0` (readouts only, 2,192 parameters) fixes energies slowly and barely moves forces; "
                "`TRAINABLE_BLOCKS = 2` trains everything and is slower without being better on this small set."
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-3  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_energy_mae_per_atom'] = round(entry['val']['energy_mae_per_atom'], 5)\n"
                "        row['val_force_mae'] = round(entry['val']['force_mae'], 5)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=TRAINABLE_BLOCKS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'seconds': adapt_seconds}})\n"
                "print({{'e0_calibration_eV': dict(zip(adapt_result['calibration']['elements'], [round(s, 4) for s in adapt_result['calibration']['shifts_ev']]))}})\n"
                "print({{'trainable_prefixes': adapt_result['trainable_prefixes']}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for calibration, training or epoch selection. The adapted numbers are read "
                "against the composition baseline (what a model that cannot see geometry achieves), the zero-force baseline and "
                "the frozen foundation model from Section 6. Look for an energy MAE an order of magnitude below the composition "
                "baseline and a force MAE well below the frozen model's. Twelve test structures from one seeded split give no "
                "dispersion estimate — the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark."
            ),
            "code": (
                "test_metrics = pipe.evaluate(test_records)\n"
                "val_metrics = pipe.evaluate(val_records)\n"
                "print({{'test_adapted': {{k: round(v, 5) for k, v in test_metrics.items() if isinstance(v, float)}}, 'n_structures': test_metrics['n_structures']}})\n"
                "comparison = {{\n"
                "    'energy_mae_per_atom': {{'composition_baseline': round(baseline_composition['energy_mae_per_atom'], 5), 'frozen_model': round(zero_shot_test['energy_mae_per_atom'], 5), 'adapted': round(test_metrics['energy_mae_per_atom'], 5)}},\n"
                "    'force_mae': {{'zero_force_baseline': round(baseline_zero_force['force_mae'], 5), 'frozen_model': round(zero_shot_test['force_mae'], 5), 'adapted': round(test_metrics['force_mae'], 5)}},\n"
                "}}\n"
                "for metric, values in comparison.items():\n"
                "    print({{metric: values}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'composition': baseline_composition, 'zero_force': baseline_zero_force, 'frozen_model': zero_shot_test}},\n"
                "    'physics_checks': physics,\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'history'}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert test_metrics['energy_mae_per_atom'] < baseline_composition['energy_mae_per_atom']\n"
                "assert test_metrics['force_mae'] < zero_shot_test['force_mae']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Inference on new structures, artifact export and fresh reload\n\n"
                "Three structures are generated with a different seed — one per composition — and labelled with EMT so the "
                "adapted model's errors on them can be printed alongside its predictions; they were never part of any split. "
                "The stress tensor is returned because the cells are fully periodic.\n\n"
                "`pipe.save_artifact` writes only the tensors the adaptation could change (readouts, reference energies, "
                "scale/shift and the unfrozen blocks) as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digests of the converted base pair, the tensor names, the file size "
                "and SHA-256, the training configuration and the epoch history (OUT8). `MaceMaterialsPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest and digest **before** deserialising, and overlays "
                "the tensors onto a freshly rebuilt base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical energies and forces (VER4)."
            ),
            "code": (
                "import random\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_records = test_records[:3]\n"
                "    new_source = 'first three BYOD test-split structures'\n"
                "else:\n"
                "    new_rng = random.Random(7)\n"
                "    new_records = []\n"
                "    for composition in SAMPLE_COMPOSITIONS:\n"
                "        structure = build_sample_structure(composition, strain=new_rng.uniform(-0.03, 0.03), sigma=0.10, rng=new_rng)\n"
                "        structure['energy'], structure['forces'] = _emt_energy_forces(to_atoms(structure))\n"
                "        new_records.append(structure)\n"
                "    new_source = 'freshly generated fcc supercells (seed 7), one per composition'\n"
                "new_prediction = pipe.predict(new_records)\n"
                "print({{'new_source': new_source, 'adapted': new_prediction['model']['adapted']}})\n"
                "for record, result in zip(new_records, new_prediction['results']):\n"
                "    ref_forces = np.array(record['forces'])\n"
                "    print({{'name': result['name'], 'n_atoms': result['n_atoms'], 'energy_per_atom_eV': round(result['energy_per_atom'], 4), 'reference_per_atom_eV': round(record['energy'] / result['n_atoms'], 4), 'force_mae_eV_A': round(float(np.abs(np.array(result['forces']) - ref_forces).mean()), 4), 'pressure_eV_A3': round(-sum(result['stress'][i][i] for i in range(3)) / 3, 5)}})\n"
                "new_metrics = pipe.evaluate(new_records)\n"
                "print({{'new_structures': {{k: round(v, 5) for k, v in new_metrics.items() if isinstance(v, float)}}, 'note': 'sanity check on generated structures, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'source': new_source, 'units': new_prediction['units'], 'results': [{{k: v for k, v in r.items() if k != 'node_energies'}} for r in new_prediction['results']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = MaceMaterialsPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.predict(test_records[:4])['results']\n"
                "after = reloaded.predict(test_records[:4])['results']\n"
                "parity = {{\n"
                "    'max_abs_energy_diff': max(abs(a['energy'] - b['energy']) for a, b in zip(before, after)),\n"
                "    'max_abs_force_diff': max(float(np.abs(np.array(a['forces']) - np.array(b['forces'])).max()) for a, b in zip(before, after)),\n"
                "}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['max_abs_energy_diff'] < 1e-9 and parity['max_abs_force_diff'] < 1e-9\n\n"
                "import platform\n\n"
                "import safetensors\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': next(e for e in MANIFEST['files'] if e['path'] == SOURCE_MODEL_NAME),\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'mace': mace.__version__, 'e3nn': e3nn.__version__, 'ase': ase.__version__, 'safetensors': safetensors.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'test_metrics': test_metrics,\n"
                "    'comparison': comparison,\n"
                "    'new_structures': new_metrics,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen foundation potential already predicts EMT forces to roughly a fifth of an eV/Å without ever having seen "
        "EMT — that is what a universal potential trained on Materials Project trajectories carries into a new system — but its "
        "energies sit several eV per atom away, because PBE and EMT put their zero in different places. Two per-element offsets "
        "remove that; a bounded fine-tuning of the readouts and the last interaction block then brings the held-out energy error "
        "an order of magnitude below the composition baseline and roughly halves the force error, in about a minute on CPU. That "
        "is the claim: the adaptation contract moves a foundation interatomic potential onto a different level of theory from a "
        "few dozen labelled structures, and the physics checks show the rebuilt, converted model is still an equivariant "
        "potential whose forces are the gradient of its energy.\n\n"
        "The test split has 12 generated supercells of two elements, the metrics come from one seeded split with no dispersion "
        "estimate, and EMT is a toy level of theory chosen because it runs in the notebook. So a small error here says the "
        "contract works, not that the adapted model reproduces DFT for Cu–Al, transfers to other compositions, defects or "
        "surfaces, or that it is stable in molecular dynamics — none of which this repository exercises. Fine-tuning a "
        "foundation potential on a narrow dataset can also erode its behaviour elsewhere; upstream mitigates that with "
        "multi-head replay against the original training data, which is out of scope here.\n\n"
        "Three things to carry to real data. **Reference consistency:** every label must come from one code, one functional and "
        "one set of settings; mixing references is the most common way to get an unlearnable dataset. **Splits:** structures "
        "from one trajectory or one relaxation are near-duplicates — split by trajectory or by composition, never at random over "
        "frames. **Baselines first:** if the composition baseline is already good, your labels vary with stoichiometry more than "
        "with geometry, and the force error is the number to read.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned source asset, audit and convert a pickled checkpoint into a code-free "
        "serving pair without executing anything outside the audited allow-lists, rebuild the model from the installed library, "
        "validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate against two trivial baselines and the "
        "frozen model on an independent split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or physical validity beyond the checks "
        "shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_BLOCKS = 0` to train the readouts alone "
        "and watch the forces stay put; set `TRAINABLE_BLOCKS = 2` to fine-tune everything and compare the time; raise "
        "`EPOCHS`; or bring your own extended-XYZ dataset through BYOD and read the composition baseline before the adapted "
        "number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/mace-foundations/mace-mp-0 (revision `{MODEL_REVISION}`)\n"
        "- Upstream release asset (byte-identical): https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0b2\n"
        "- Batatia et al., *A foundation model for atomistic materials chemistry*, arXiv:2401.00096 (2023): https://arxiv.org/abs/2401.00096\n"
        "- Batatia et al., *MACE: Higher order equivariant message passing neural networks for fast and accurate force fields*, NeurIPS 2022: https://arxiv.org/abs/2206.07697\n"
        "- ASE EMT calculator: https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
