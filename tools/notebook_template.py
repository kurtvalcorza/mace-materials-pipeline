"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, generator /2.1).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E interatomic-potential workflow: the pinned MACE-MP-0b2 small source
asset is digest-verified, statically audited and converted once into a code-free safetensors pair,
an EMT-labelled dataset of rattled fcc supercells is generated in code, validated and split, the
frozen foundation model is checked for equivariance and force consistency, its zero-shot errors are
measured against two trivial baselines, a bounded fine-tuning runs in the kernel, held-out energy and
force errors are reported, the adapter is exported and reloaded, and a change-one-thing activity
fine-tunes a second, separate pipeline with a different trainable scope.
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
    "isolated_runtime": True,
    # MMC-M1: the fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # generator /2.1 = gliner-ner-pipeline fe3d5ba tools/build_notebook.py, plus the `lock_no_deps` key below). The kernel's
    # preloaded NumPy / CUDA bindings are never replaced, so Run all needs no restart. The lock is compiled with
    # `uv pip compile pyproject.toml -c <versions of ast-audio-classification-pipeline's T4-passed
    # tutorials/requirements-colab.lock.txt @ 16eee39, without huggingface-hub> --override tutorials/requirements-colab-overrides.txt
    # --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all:
    # -o tutorials/requirements-colab.lock.txt`: 73 packages; the 39 it shares with ast's lock (torch 2.14.0, torchvision 0.29.0,
    # numpy 2.5.3, safetensors 0.8.0, the CUDA 13 wheels, ...) are at ast's versions, huggingface-hub is this repository's own pin
    # 1.32.0 (ast pins 0.36.2), and mace-torch 0.3.16 brings e3nn, matscipy, scipy, pandas, matplotlib, h5py, lmdb, torchmetrics
    # and the rest as wheels. Its one sdist-only dependency, python-hostlist, is excluded (see `lock_no_deps`).
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "lock_no_deps": {
        "python-hostlist": "sdist only; imported by mace/tools/slurm_distributed.py inside try/except ImportError, for SLURM jobs only",
    },
    "infrastructure_labels": True,
    "collapse_model_cell": False,  # the model cell prints the pickle audit and conversion, which Section 3 asks the learner to read
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated Python 3.12.12 environment from the hash-locked "
        "pins (torch, mace-torch, e3nn, ase, numpy, safetensors, huggingface-hub and their dependencies; the kernel's own "
        "packages are left alone, so no restart is needed), stages and digest-verifies the pinned MACE-MP-0b2 small source "
        "asset (67.6 MB pickled module from the Hub), statically audits every global and generated source string that pickle "
        "would execute and refuses anything outside the torch / e3nn / mace allow-lists, unpickles it exactly once to write a "
        "code-free JSON config + safetensors pair whose digests are pinned in the module, rebuilds the model from the installed "
        "mace-torch and loads it strictly, generates a deterministic 48-structure dataset in code (rattled, strained fcc Cu, Al "
        "and Cu₁₆Al₁₆ supercells labelled by ASE's EMT potential — no download), validates the structures and the dataset "
        "contract, splits them into train/validation/test sets (stratified by composition, since every generated structure is "
        "an independent draw), checks the frozen model for rotation equivariance, permutation invariance and "
        "analytic-versus-finite-difference forces, measures its zero-shot energy and force errors against a composition "
        "baseline and a zero-force baseline, calibrates the per-element reference energies and runs a bounded Adam fine-tuning "
        "of the readouts plus the last interaction block, evaluates energy and force errors on the held-out test split, "
        "predicts three freshly generated structures, exports the adapter as safetensors with a manifest, reloads that artifact "
        "into a fresh pipeline to verify prediction parity, and runs the Section 10 activity (a second, separate fine-tuning "
        "of the readouts only, also exported and reloaded). The default path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Building the isolated "
        "environment takes a few minutes; after it, the model work takes about four minutes on CPU and well under a minute on "
        "a GPU."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose **Runtime → Run "
        "after** (it re-runs Section 4 and every later cell). Supply one extended-XYZ file of labelled structures (one frame per "
        "structure, a `Lattice` and `pbc` when periodic, an `energy` in eV and a `forces` array in eV/Å per frame, and "
        "optionally `group=<id>` or `trajectory=<id>` on each frame's comment line) through the upload dialog on Colab, or by "
        "path in `BYOD_PATH` on any runtime (a path, when set, is used instead of the dialog). By default (`BYOD_SPLIT = "
        "'auto'`) whole groups — your trajectory tags, or else whole compositions — go to one split, so near-duplicate frames "
        "never straddle train and test; the grouping is printed. Sections 5–7 return to the pinned foundation weights before "
        "they run, so the frozen-model rows and the adaptation start from the foundation model, not from the sample "
        "adaptation. BYOD outputs are written under `outputs/mace_materials_byod_*` and never overwrite the sample's. The "
        "expected schema, the size limits and the element support are stated in the Prerequisites and in Section 4, and the "
        "file stays inside this runtime. BYOD is optional and never part of the default path."
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
        "cannot see the displacements at all.\n\n"
        "**Who this is for.** A learner who knows basic Python and has used Colab or Jupyter, and wants to see how a pretrained "
        "interatomic potential is checked, measured against baselines and fine-tuned to a different level of theory. You "
        "should know what an atom's position, a periodic cell, an energy and a force are; no experience with MACE, "
        "equivariant networks or fine-tuning is assumed. Per-atom energies, MAE, equivariance, E0 calibration, adapters and "
        "the other terms are explained where they are first used and again in the **Glossary** at the end. A CPU runtime is "
        "enough.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference | Adaptation |\n"
        "|---|---|---|\n"
        "| Input | 1 to 32 structures per call, each `{symbols, positions, cell, pbc}` in Å (1 to 512 atoms, at most 4,096 atoms per call) | at least 8 labelled structures (an `energy` in eV and `forces` in eV/Å each), split into train / validation / test |\n"
        "| Model | MACE-MP-0b2 small rebuilt from the converted pair, float64, forces as the gradient of the energy | the same model after a per-element reference-energy (E0) calibration, with the readouts, E0, scale/shift and the last interaction block trained by Adam |\n"
        "| Output | per structure: total energy (eV), energy per atom, forces (eV/Å), stress (eV/Å³) for periodic cells — point predictions with **no uncertainty and no out-of-domain flag** | per-atom energy and force errors on the test split beside two baselines and the frozen model, and a safetensors adapter with a manifest |\n\n"
        "**How to use this notebook.** Choose **Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated "
        "environment, the carried code and the model verification — and can be run without study; the code of Sections 1 and 2 "
        "is collapsed. Section 3's output is worth reading: it prints the pickle audit and the conversion. The learning path "
        "starts in Section 4. Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the "
        "recorded path. Each stage asks you to **predict** before it runs, and the next cell opens with **What to notice** and a "
        "collapsible **Check your reasoning** block with a worked answer from this repository's CPU run of the same stages "
        "(GPU numbers can differ in the last decimals). Section 10 is a **change-one-thing activity**. **Troubleshooting**, a "
        "**Glossary** and a **Conclusion** template are at the end. Your notes are optional and are not required submissions.\n\n"
        "**Roadmap:** 4 generate, validate and split the structures → 5 physics checks on the frozen model → 6 baselines and the "
        "zero-shot error → 7 calibrate and fine-tune → 8 held-out evaluation → 9 new structures, adapter export and reload → "
        "10 **change one thing: which blocks train** → conclude. Core concepts are Sections 5 and 6 (what the model returns and "
        "what a good error is); evaluation practice is Sections 4, 6 and 8 (split, baselines, held-out errors); engineering "
        "and reproducibility are Sections 1–3 and 9."
    ),
    "learning_objectives": (
        "inspect the carried pipeline, dataset and metrics modules; stage and digest-verify an immutable source asset, read a "
        "static audit of a pickled checkpoint and see it converted into a code-free serving pair; validate atomic structures and "
        "split a labelled dataset so that related structures stay on one side; check a foundation potential for equivariance, "
        "permutation invariance and gradient-consistent forces; measure zero-shot energy and force errors against two trivial "
        "baselines; calibrate reference energies and run a bounded fine-tuning with explicit hyperparameters; evaluate "
        "per-atom energy and force errors on an independent test split; predict new structures; export a safetensors adapter "
        "that reloads against the pinned base with verified parity; and compare two trainable scopes from the same starting "
        "point."
    ),
    "exclusions": (
        "molecular-dynamics driving, geometry optimisation, phonons, the published MACE-MP-0 benchmarks, the medium and large "
        "checkpoints, multi-head replay fine-tuning against the Materials Project data, LAMMPS or CUDA-equivariance "
        "deployment, per-prediction uncertainty or out-of-domain detection, and any claim that an EMT-labelled fcc supercell "
        "stands in for a DFT dataset. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh Linux x86_64 runtime (Google Colab, Kaggle or Linux Jupyter); the notebook builds its own Python 3.12.12 environment, so the kernel's Python version does not matter. CPU is enough — a 32-atom cell is scored in under 0.1 s and each default fine-tuning takes about a minute — and CUDA is used automatically when present. The model runs in float64, as upstream ships it.",
        "- **Knowledge:** what an interatomic potential, a periodic cell and a force are; why energies are compared per atom; and what MAE and RMSE mean (all repeated in the Glossary).",
        "- **Executable serialization handled explicitly:** the pinned `.model` file is a pickle. It is digest-verified, statically audited against allow-lists (the audit digest is pinned) and unpickled **once** to produce the safetensors pair the model is actually loaded from. No Hub-hosted Python module is imported; `mace-torch` is installed from PyPI at a pinned version. One process-wide side effect to know: importing `mace` (Section 1) sets the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which `e3nn` needs to load its own constants under current torch. For the rest of the session every `torch.load` call that does not pass `weights_only` explicitly unpickles arbitrary objects (`weights_only=False`) — including any you write yourself. This notebook loads its own files through safetensors only and unpickles the source asset exactly once, after the audit; if you load other checkpoints in this runtime, pass `weights_only=True` yourself.",
        "- **Data contract:** structures are `{symbols, positions, cell, pbc}` in Å with optional `energy` (eV) and `forces` (eV/Å); elements limited to the 89 MACE-MP-0 supports (Z 1–83, 89–94); 1..512 atoms per structure, at most 32 structures and 4,096 atoms per prediction call (evaluation and adaptation process longer lists in chunks of that size); no two atoms closer than 0.5 Å (periodic images included); cell vectors under 200 Å and periodic cell heights of at least 1 Å. A dataset needs 8 to 1,000 labelled structures with unique names, and its split must leave at least 4 training, 1 validation and 1 test structure, with every validation and test element present in training; a grouped split also needs at least 3 groups. Section 4 refuses a dataset that breaks any of these, naming the rule. BYOD accepts extended XYZ.",
        "- **Validation is structural, not chemical:** nothing checks that a structure is charge-neutral, near equilibrium or physically meaningful — a random cloud of supported atoms is scored without complaint, and EMT labels are only meaningful for the metals it parameterises.",
        "- **No uncertainty, no domain flag:** every energy, force and stress is a point prediction. Nothing in this notebook estimates its uncertainty or flags a structure far from the training data; an unfamiliar chemistry is scored with the same confidence as a familiar one.",
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
                "contract before any model runs.\n\n"
                "**How the split keeps structures independent.** Frames of one molecular-dynamics trajectory or one relaxation "
                "are near-duplicates: if some land in training and their neighbours in the test split, the test error measures "
                "memory, not generalisation. `split_dataset` therefore has two kinds of mode. The generated sample uses "
                "`'stratified'`: every structure is its own independent draw (its own strain, rattle and alloy arrangement), so "
                "it shuffles within each composition and cuts 20 % validation / 25 % test, and every split sees every "
                "composition. BYOD data uses `BYOD_SPLIT` (default `'auto'`): whole groups go to one split — the `group=<id>` "
                "or `trajectory=<id>` tags on your frames when every frame has one, otherwise whole compositions — and the "
                "fractions then count groups, not structures. The cell prints which groups landed where and asserts that no "
                "group sits in two splits. Use `'stratified'` for your own data only if every structure is an independent "
                "draw.\n\n"
                "**Predict before running:** with 48 structures, three compositions and the 20 % / 25 % fractions, how many "
                "structures will each split hold, and will any composition be missing from the test split?\n\n"
                "Four refusal probes follow the split — an unsupported element, overlapping atoms, periodicity without a cell "
                "and a malformed force array — each rejected before `torch` does anything."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_SPLIT = 'auto'  # @param [\"auto\", \"group\", \"composition\", \"stratified\"]\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# BYOD outputs carry their own names, so a BYOD run never overwrites (or is mistaken for) the sample's.\n"
                "OUTPUTS = {{\n"
                "    'dataset': 'outputs/{stem}_byod_dataset.xyz' if USE_BYOD else 'outputs/{stem}_sample_dataset.xyz',\n"
                "    'report': 'outputs/{stem}_byod_evaluation_report.json' if USE_BYOD else 'outputs/{stem}_evaluation_report.json',\n"
                "    'predictions': 'outputs/{stem}_byod_predictions.json' if USE_BYOD else 'outputs/{stem}_predictions.json',\n"
                "    'adapter': 'outputs/{stem}_byod_adapter' if USE_BYOD else 'outputs/{stem}_adapter',\n"
                "    'result': 'outputs/{stem}_byod_result.json' if USE_BYOD else 'outputs/{stem}_result.json',\n"
                "    'activity_adapter': 'outputs/{stem}_byod_activity_adapter' if USE_BYOD else 'outputs/{stem}_activity_adapter',\n"
                "}}\n"
                "if USE_BYOD:\n"
                "    print({{'expected_byod_input': 'one extended-XYZ file: per frame a Lattice and pbc when periodic, energy (eV), forces (eV/A), optional group=<id> or trajectory=<id>', 'records': f'{{MIN_RECORDS}}..{{MAX_DATASET_RECORDS}}', 'split': BYOD_SPLIT, 'source': 'BYOD_PATH' if BYOD_PATH else 'Colab upload dialog'}})\n"
                "    if BYOD_PATH:\n"
                "        byod_path = Path(BYOD_PATH)\n"
                "        if not byod_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file in this runtime: give the path of one extended-XYZ file and run this cell again')\n"
                "        data_source = f'BYOD file {{byod_path.name}} (BYOD_PATH)'\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD = True needs a file: this runtime has no Colab upload dialog, so set BYOD_PATH to the path of one extended-XYZ file and run this cell again') from None\n"
                "        uploaded = files.upload()\n"
                "        if not uploaded:\n"
                "            raise ValueError('no file uploaded; run this cell again and choose one extended-XYZ file')\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'{{len(uploaded)}} files uploaded; run this cell again and choose exactly one extended-XYZ file')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / Path(file_name).name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "        data_source = f'BYOD upload {{byod_path.name}}'\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    split_mode = BYOD_SPLIT\n"
                "else:\n"
                "    records = generate_sample_dataset(seed=SEED)\n"
                "    data_source = f'generated fcc supercells labelled by {{SAMPLE_LABEL_SOURCE}} (seed {{SEED}}, {{SAMPLE_SIZE}} structures)'\n"
                "    split_mode = 'stratified'  # every generated structure is an independent draw (own strain, rattle and alloy arrangement)\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED, group_by=split_mode)\n"
                "train_records, val_records, test_records = splits['train'], splits['val'], splits['test']\n"
                "split_summary = split_report(splits, group_by=split_mode)\n"
                "write_dataset_xyz(records, OUTPUTS['dataset'])\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'n_atoms': dataset_manifest['n_atoms'], 'compositions': dataset_manifest['compositions']}})\n"
                "print({{'elements': dataset_manifest['elements'], 'max_atoms': dataset_manifest['max_atoms'], 'labelled': dataset_manifest['labelled'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'energy_per_atom_range_eV': (round(min(r['energy'] / len(r['symbols']) for r in records), 4), round(max(r['energy'] / len(r['symbols']) for r in records), 4))}})\n"
                "print({{'split_mode': split_summary['mode'], 'kept_together': split_summary['unit'], 'sizes': split_summary['sizes']}})\n"
                "for name in ('train', 'val', 'test'):\n"
                "    print({{name: split_summary['groups'][name]}})\n"
                "if split_summary['mode'] != 'stratified':\n"
                "    assert not split_summary['groups_in_more_than_one_split'], split_summary['groups_in_more_than_one_split']\n"
                "print({{'written': OUTPUTS['dataset'], 'example': {{k: (v if not isinstance(v, list) else f'list[{{len(v)}}]') for k, v in records[0].items()}}}})\n\n"
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
                "**What to notice:** 48 structures, three compositions of 16, 1,536 atoms, `split_mode` `stratified` with sizes "
                "27/9/12, every composition in every split, EMT energies of a few tenths of an eV per atom, the written "
                "`outputs/mace_materials_sample_dataset.xyz`, and four refusals, each naming the rule.\n\n"
                "<details><summary>Check your reasoning</summary>Each composition has 16 structures: 25 % is 4 for the test "
                "split, 20 % rounds to 3 for validation, and 9 stay in training, so the splits hold 3 × 9 = 27, 3 × 3 = 9 and 3 × "
                "4 = 12 structures and every composition is in every split. That is right for these structures because each is "
                "an independent draw. Had they been 48 frames of three trajectories, the same split would put neighbouring frames "
                "on both sides; with `BYOD_SPLIT = 'auto'` the same file without group tags would instead be split by "
                "composition, one whole composition per split — and then refused, because the validation and test elements must "
                "also appear in training (Cu₃₂ alone in training cannot cover Al).</details>"
            ),
        },
        {
            "md": (
                "## 5. Zero-shot prediction and physics checks on the frozen model\n\n"
                "`pipe.predict` returns, per structure, the total energy (eV), the per-atom energy, per-atom energy "
                "contributions, forces (eV/Å) and — for fully periodic cells — the stress tensor (eV/Å³). The model is the "
                "pinned foundation potential, untouched: if this cell runs after an adaptation (a BYOD re-run, for example), it "
                "first returns the pipeline to the pinned base weights with `pipe.reset_to_pretrained()` and says so.\n\n"
                "Five checks say whether the rebuilt model behaves like an interatomic potential should. **Rotation:** rotating "
                "the cell and every atom leaves the energy unchanged and rotates the forces with it (equivariance). "
                "**Translation** and **permutation** of atoms change nothing. **Gradient consistency:** the force on one atom "
                "equals the central finite difference of the energy along that coordinate, to ~1e-8 eV/Å — the forces are the "
                "analytic gradient, not a separate head. **Extensivity:** a 2×1×1 supercell has twice the energy. A rebuilt "
                "model with a wrong config would fail these before any accuracy number.\n\n"
                "**Predict before running:** which of the five differences do you expect to be at the level of floating-point "
                "round-off (around 1e-13), and which one larger? Why?"
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n\n"
                "if pipe.adapter is not None:\n"
                "    print({{'reset_to_pretrained': pipe.reset_to_pretrained()}})\n"
                "s0 = test_records[0]\n"
                "t0 = time.perf_counter()\n"
                "prediction = pipe.predict(test_records[:8])\n"
                "print({{'structures': len(prediction['results']), 'seconds': round(time.perf_counter() - t0, 2), 'units': prediction['units']}})\n"
                "base = prediction['results'][0]\n"
                "stress_diag = [round(base['stress'][i][i], 5) for i in range(3)] if base['stress'] is not None else None\n"
                "print({{'name': base['name'], 'energy_eV': round(base['energy'], 4), 'energy_per_atom_eV': round(base['energy_per_atom'], 4), 'reference_eV': round(s0['energy'], 4), 'max_force_eV_A': round(float(np.abs(base['forces']).max()), 4), 'stress_diag_eV_A3': stress_diag}})\n\n"
                "rng = np.random.default_rng(0)\n"
                "q, _ = np.linalg.qr(rng.normal(size=(3, 3)))\n"
                "if np.linalg.det(q) < 0:\n"
                "    q[:, 0] *= -1\n"
                "rotated = {{**s0, 'positions': (np.array(s0['positions']) @ q.T).tolist(), 'cell': None if s0['cell'] is None else (np.array(s0['cell']) @ q.T).tolist()}}\n"
                "rot = pipe.predict([rotated])['results'][0]\n"
                "translated = {{**s0, 'positions': (np.array(s0['positions']) + 1.234).tolist()}}\n"
                "tr = pipe.predict([translated])['results'][0]\n"
                "perm = rng.permutation(len(s0['symbols']))\n"
                "permuted = {{**s0, 'symbols': [s0['symbols'][i] for i in perm], 'positions': [s0['positions'][i] for i in perm]}}\n"
                "pm = pipe.predict([permuted])['results'][0]\n"
                "h = 1e-4\n"
                "plus, minus = np.array(s0['positions']), np.array(s0['positions'])\n"
                "atom = min(3, len(s0['symbols']) - 1)\n"
                "plus[atom, 0] += h\n"
                "minus[atom, 0] -= h\n"
                "e_plus = pipe.predict([{{**s0, 'positions': plus.tolist()}}])['results'][0]['energy']\n"
                "e_minus = pipe.predict([{{**s0, 'positions': minus.tolist()}}])['results'][0]['energy']\n"
                "fd_force = -(e_plus - e_minus) / (2 * h)\n"
                "periodic = s0['cell'] is not None and all(s0['pbc'])  # the supercell check needs a fully periodic cell\n"
                "supercell = pipe.predict([from_atoms(to_atoms(s0) * (2, 1, 1))])['results'][0] if periodic else None\n"
                "physics = {{\n"
                "    'rotation_energy_diff': abs(rot['energy'] - base['energy']),\n"
                "    'rotation_force_diff': float(np.abs(np.array(rot['forces']) - np.array(base['forces']) @ q.T).max()),\n"
                "    'translation_energy_diff': abs(tr['energy'] - base['energy']),\n"
                "    'permutation_energy_diff': abs(pm['energy'] - base['energy']),\n"
                "    'permutation_force_diff': float(np.abs(np.array(pm['forces']) - np.array(base['forces'])[perm]).max()),\n"
                "    'finite_difference_force': fd_force,\n"
                "    'analytic_force': base['forces'][atom][0],\n"
                "    'gradient_consistency_diff': abs(fd_force - base['forces'][atom][0]),\n"
                "    'extensivity_diff_per_atom': abs(supercell['energy'] - 2 * base['energy']) / supercell['n_atoms'] if periodic else float('nan'),\n"
                "    'batch_vs_single_energy_diff': abs(pipe.predict([s0])['results'][0]['energy'] - base['energy']),\n"
                "}}\n"
                "for key, value in physics.items():\n"
                "    print({{key: f'{{value:.3e}}'}})\n"
                "assert physics['rotation_energy_diff'] < 1e-8 and physics['rotation_force_diff'] < 1e-8\n"
                "assert physics['permutation_energy_diff'] < 1e-8 and physics['translation_energy_diff'] < 1e-8\n"
                "assert physics['gradient_consistency_diff'] < 1e-6\n"
                "assert not periodic or physics['extensivity_diff_per_atom'] < 1e-8"
            ),
        },
        {
            "md": (
                "**What to notice:** the timing line, one structure's energy beside its EMT reference (several eV per atom apart "
                "— Section 6 explains why), and ten differences, all small enough for the four assertions to pass.\n\n"
                "<details><summary>Check your reasoning</summary>The rotation, translation, permutation, extensivity and "
                "batch-versus-single differences come out at the round-off level of float64 arithmetic (0 to about 1e-13) "
                "because the model is built to respect those symmetries exactly. The finite-difference check is the larger one, "
                "about 1e-8 eV/Å in this repository's CPU check: a central difference with step h = 1e-4 Å has a truncation error "
                "of order h², and the energies it subtracts are total energies of hundreds of eV, so cancellation costs digits. "
                "Agreement to 1e-8 against forces of order 0.1–1 eV/Å is what \"the forces are the gradient of the energy\" "
                "looks like numerically.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the zero-shot error\n\n"
                "Three numbers frame everything that follows. The **composition baseline** fits one energy per element on the "
                "training split (the classical E0 regression) and predicts zero forces: it is blind to geometry by construction, "
                "so its energy error is the within-composition spread of the labels and its force error is the mean absolute "
                "reference force. The **zero-force baseline** is that force number alone. The **frozen foundation model** is "
                "evaluated as is (this cell, too, returns to the pinned base weights first if the pipeline was adapted). "
                "Energies are per atom (eV/atom); forces are per Cartesian component (eV/Å).\n\n"
                "**Predict before running:** the frozen model was trained on PBE (DFT) energies and is scored here against EMT "
                "labels. Which will it get closer to the reference — the forces or the energies? Will it beat the composition "
                "baseline on energy?"
            ),
            "code": (
                "if pipe.adapter is not None:\n"
                "    print({{'reset_to_pretrained': pipe.reset_to_pretrained()}})\n"
                "baseline_composition = composition_baseline(train_records, test_records)\n"
                "baseline_zero_force = zero_force_baseline(test_records)\n"
                "t0 = time.perf_counter()\n"
                "zero_shot_test = pipe.evaluate(test_records)\n"
                "zero_shot_seconds = round(time.perf_counter() - t0, 2)\n"
                "print({{'composition_baseline': {{'energy_mae_per_atom': round(baseline_composition['energy_mae_per_atom'], 4), 'force_mae': round(baseline_composition['force_mae'], 4), 'e0_eV': {{k: round(v, 4) for k, v in baseline_composition['e0_ev'].items()}}}}}})\n"
                "print({{'zero_force_baseline': {{'force_mae': round(baseline_zero_force['force_mae'], 4), 'force_rmse': round(baseline_zero_force['force_rmse'], 4)}}}})\n"
                "print({{'frozen_foundation_model': {{k: round(v, 4) for k, v in zero_shot_test.items() if isinstance(v, float)}}, 'seconds': zero_shot_seconds}})\n"
                "if not USE_BYOD:  # the sample's recorded result; on your data a frozen model that loses to zero forces is a finding\n"
                "    assert zero_shot_test['force_mae'] < baseline_zero_force['force_mae']\n"
                "elif zero_shot_test['force_mae'] >= baseline_zero_force['force_mae']:\n"
                "    print('On this data the frozen model predicts forces no better than zero: expect adaptation to have a lot to do.')"
            ),
        },
        {
            "md": (
                "**What to notice:** the frozen model's force MAE well below the zero-force baseline, and its energy MAE far "
                "above the composition baseline's.\n\n"
                "<details><summary>Check your reasoning</summary>Forces. In this repository's CPU check the frozen model's force "
                "MAE was 0.2079 eV/Å against 0.7376 eV/Å for predicting zero — it already knows how metal atoms push on each "
                "other. Its energy MAE was 3.98 eV/atom against 0.0929 eV/atom for the composition baseline: PBE total energies "
                "and EMT energies sit on different absolute references (EMT's zero is close to the isolated-atom reference, PBE's "
                "is not), so every structure is off by roughly the same few eV per atom. A per-element offset removes most of "
                "that — which is exactly the first step of Section 7. The forces are unaffected by such an offset, because a "
                "constant energy per atom has zero gradient.</details>"
            ),
        },
        {
            "md": (
                "## 7. Calibrate and fine-tune\n\n"
                "The cell first returns the pipeline to the pinned foundation weights (`pipe.reset_to_pretrained()`), so every "
                "run of this cell — the default one, a re-run after you change a field, or a BYOD run — starts from the same "
                "model; `pipe.adapt` refuses to train a pipeline that is already adapted. `pipe.adapt` then does two things in "
                "order. First it **calibrates the per-element reference energies**: a least-squares fit of one offset per element "
                "between the training labels and the frozen predictions, added to the model's atomic energies — two numbers on a "
                "Cu/Al dataset, and the cheapest possible move to a new level of theory. That calibrated frozen model is recorded "
                "as epoch 0 so every later number is comparable to it. Then it trains the readouts, the reference energies, the "
                "scale/shift block and — with `TRAINABLE_BLOCKS = 1` — the last interaction and product blocks: 1.92 M of 8.22 M "
                "parameters, Adam at a fixed learning rate, loss = 100 × MSE(per-atom energy) + 10 × MSE(force components), "
                "batches of 4, no scheduler, seeded shuffling. The epoch with the lowest validation loss is kept.\n\n"
                "To try another setting here, change the field, select this cell and choose **Runtime → Run after**: Sections "
                "7–10 run again from the foundation model and overwrite the outputs. Section 10 does the same comparison without "
                "touching the main run.\n\n"
                "**Predict before running:** how far will the validation energy MAE fall from epoch 0, and will the force MAE "
                "fall by the same factor?"
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
                "print({{'reset_to_pretrained': pipe.reset_to_pretrained()}})\n"
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
                "**What to notice:** epoch 0 labelled `frozen model after E0 calibration`, the two calibration shifts (several "
                "eV each), the validation energy MAE falling by about an order of magnitude, and the force MAE falling by a "
                "smaller factor.\n\n"
                "<details><summary>Check your reasoning</summary>In this repository's CPU check epoch 0 (the calibrated frozen "
                "model) scored 0.0985 eV/atom and 0.1967 eV/Å on validation, after shifts of Al +3.768 and Cu +4.176 eV; "
                "the kept epoch (epoch 8) scored 0.0076 eV/atom and 0.0762 eV/Å, with 1,919,128 of 8,221,984 parameters "
                "trained in about 66 s. The energy falls further because two offsets plus a retrained readout can absorb most of "
                "the systematic PBE-versus-EMT difference, while the forces depend on the curvature of the energy surface, which "
                "only the trained interaction block can change. The validation energy MAE does not have to fall every epoch: "
                "selection uses the weighted loss (energy and forces together), so a kept epoch can have a slightly worse energy "
                "MAE than its neighbour.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for calibration, training or epoch selection. The adapted numbers are read "
                "against the composition baseline (what a model that cannot see geometry achieves), the zero-force baseline and "
                "the frozen foundation model from Section 6. Twelve test structures from one seeded split give no dispersion "
                "estimate — the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark.\n\n"
                "**Predict before running:** will the adapted energy MAE beat the composition baseline, and by how much?"
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
                "    'split': split_summary,\n"
                "    'baselines': {{'composition': baseline_composition, 'zero_force': baseline_zero_force, 'frozen_model': zero_shot_test}},\n"
                "    'physics_checks': physics,\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'history'}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'uncertainty': 'none: point predictions with no per-prediction uncertainty and no out-of-domain flag',\n"
                "}}\n"
                "with open(OUTPUTS['report'], 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "beats = {{'energy_beats_composition_baseline': test_metrics['energy_mae_per_atom'] < baseline_composition['energy_mae_per_atom'], 'force_beats_frozen_model': test_metrics['force_mae'] < zero_shot_test['force_mae']}}\n"
                "print({{'reading': beats, 'report': OUTPUTS['report']}})\n"
                "if USE_BYOD:\n"
                "    # Your data decides; a baseline that wins is a result to report, not an error (see Interpretation, 'Baselines first').\n"
                "    if not all(beats.values()):\n"
                "        print('On this data the adapted model does not beat every reference above: read the comparison before using the adapter.')\n"
                "else:\n"
                "    # The generated sample's recorded result; a failure here means the run differs from the recorded one.\n"
                "    assert test_metrics['energy_mae_per_atom'] < baseline_composition['energy_mae_per_atom']\n"
                "    assert test_metrics['force_mae'] < zero_shot_test['force_mae']"
            ),
        },
        {
            "md": (
                "**What to notice:** the comparison table — adapted energy MAE an order of magnitude below the composition "
                "baseline, adapted force MAE well below the frozen model's.\n\n"
                "<details><summary>Check your reasoning</summary>In this repository's CPU check (and in the earlier recorded "
                "Kaggle T4 run of the same default path) the test energy MAE was 0.00725 eV/atom adapted against 0.09291 for the "
                "composition baseline and 3.98 for the frozen model, and the force MAE 0.07803 eV/Å adapted against 0.2079 "
                "frozen and 0.7376 for zero forces. Twelve structures from one seeded split of one toy level of theory: the "
                "numbers show that the calibration and fine-tuning contract works on held-out structures of the same kind, not how "
                "well the model would do on DFT data or on other compositions.</details>"
            ),
        },
        {
            "md": (
                "## 9. Inference on new structures, artifact export and fresh reload\n\n"
                "For the sample, three structures are generated with a different seed — one per composition — and labelled with "
                "EMT so the adapted model's errors on them can be printed alongside its predictions; they were never part of any "
                "split. For BYOD there is no labeller in the notebook, so the new structures are three test-split structures with "
                "a fresh 0.03 Å rattle: new, unscored geometries, predicted without reference labels. The stress tensor is "
                "returned for fully periodic cells. Each prediction is a point value: there is no per-prediction uncertainty and "
                "no flag for a structure unlike the training data.\n\n"
                "`pipe.save_artifact` writes only the tensors the adaptation could change (readouts, reference energies, "
                "scale/shift and the unfrozen blocks) as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digests of the converted base pair, the tensor names, the file size "
                "and SHA-256, the training configuration and the epoch history (OUT8). The export goes to a staging folder first. "
                "`MaceMaterialsPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and digest "
                "**before** deserialising, and overlays the tensors onto a freshly rebuilt base — a new object from files, not "
                "the in-memory model (VER2). Only when the reloaded model reproduces the in-memory energies and forces (VER4) does "
                "the staged adapter replace the previous one, so a failed check never deletes a good adapter.\n\n"
                "**Predict before running:** will the reloaded model's energies equal the in-memory model's exactly, or only to "
                "about 1e-6?"
            ),
            "code": (
                "import random\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_rng = np.random.default_rng(7)\n"
                "    new_records = []\n"
                "    for record in test_records[:3]:\n"
                "        positions = np.array(record['positions']) + new_rng.normal(0.0, 0.03, size=(len(record['symbols']), 3))\n"
                "        new_records.append({{'name': record['name'] + '-new', 'symbols': record['symbols'], 'positions': positions.tolist(), 'cell': record['cell'], 'pbc': record['pbc']}})\n"
                "    new_source = 'three new geometries: BYOD test-split structures with a fresh 0.03 A rattle (seed 7), unlabelled'\n"
                "else:\n"
                "    new_rng = random.Random(7)\n"
                "    new_records = []\n"
                "    for composition in SAMPLE_COMPOSITIONS:\n"
                "        structure = build_sample_structure(composition, strain=new_rng.uniform(-0.03, 0.03), sigma=0.10, rng=new_rng)\n"
                "        structure['energy'], structure['forces'] = _emt_energy_forces(to_atoms(structure))\n"
                "        new_records.append(structure)\n"
                "    new_source = 'freshly generated fcc supercells (seed 7), one per composition'\n"
                "validate_inputs(new_records)\n"
                "new_prediction = pipe.predict(new_records)\n"
                "labelled_new = all(r.get('energy') is not None and r.get('forces') is not None for r in new_records)\n"
                "print({{'new_source': new_source, 'adapted': new_prediction['model']['adapted'], 'uncertainty': 'none (point predictions, no out-of-domain flag)'}})\n"
                "for record, result in zip(new_records, new_prediction['results']):\n"
                "    row = {{'name': result['name'], 'n_atoms': result['n_atoms'], 'energy_per_atom_eV': round(result['energy_per_atom'], 4)}}\n"
                "    if result['stress'] is not None:\n"
                "        row['pressure_eV_A3'] = round(-sum(result['stress'][i][i] for i in range(3)) / 3, 5)\n"
                "    if labelled_new:\n"
                "        row['reference_per_atom_eV'] = round(record['energy'] / result['n_atoms'], 4)\n"
                "        row['force_mae_eV_A'] = round(float(np.abs(np.array(result['forces']) - np.array(record['forces'])).mean()), 4)\n"
                "    print(row)\n"
                "new_metrics = pipe.evaluate(new_records) if labelled_new else None\n"
                "if new_metrics is not None:\n"
                "    print({{'new_structures': {{k: round(v, 5) for k, v in new_metrics.items() if isinstance(v, float)}}, 'note': 'sanity check on generated structures, not an evaluation'}})\n\n"
                "with open(OUTPUTS['predictions'], 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'source': new_source, 'units': new_prediction['units'], 'uncertainty': None, 'results': [{{k: v for k, v in r.items() if k != 'node_energies'}} for r in new_prediction['results']]}}, f, indent=2)\n\n"
                "def export_and_reload(model_pipe, target, metadata):\n"
                "    # Export to a staging folder, reload it into a fresh pipeline, compare, and only then replace `target`.\n"
                "    target = Path(target)\n"
                "    staging = target.with_name(target.name + '.staging')\n"
                "    shutil.rmtree(staging, ignore_errors=True)\n"
                "    model_pipe.save_artifact(staging, metadata=metadata)\n"
                "    fresh = MaceMaterialsPipeline.from_artifact(staging, weights_dir=WEIGHTS_DIR, device=model_pipe.device)\n"
                "    before = model_pipe.predict(test_records[:4])['results']\n"
                "    after = fresh.predict(test_records[:4])['results']\n"
                "    check = {{\n"
                "        'max_abs_energy_diff': max(abs(a['energy'] - b['energy']) for a, b in zip(before, after)),\n"
                "        'max_abs_force_diff': max(float(np.abs(np.array(a['forces']) - np.array(b['forces'])).max()) for a, b in zip(before, after)),\n"
                "    }}\n"
                "    if not (check['max_abs_energy_diff'] < 1e-9 and check['max_abs_force_diff'] < 1e-9):\n"
                "        raise AssertionError(f'reload parity failed {{check}}; the previous {{target}} was left in place and the export is in {{staging}}')\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    staging.rename(target)\n"
                "    return fresh, check, json.loads((target / 'manifest.json').read_text(encoding='utf-8'))\n\n"
                "artifact_dir = Path(OUTPUTS['adapter'])\n"
                "reloaded, parity, artifact_manifest = export_and_reload(pipe, artifact_dir, {{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n"
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
                "        'torch_force_no_weights_only_load': os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'),\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'mace': mace.__version__, 'e3nn': e3nn.__version__, 'ase': ase.__version__, 'safetensors': safetensors.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'split': split_summary,\n"
                "    'test_metrics': test_metrics,\n"
                "    'comparison': comparison,\n"
                "    'new_structures': {{'source': new_source, 'metrics': new_metrics}},\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "    'uncertainty': 'none: point predictions with no per-prediction uncertainty and no out-of-domain flag',\n"
                "}}\n"
                "with open(OUTPUTS['result'], 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                "**What to notice:** three new structures with their predictions (and, for the sample, their EMT errors), the "
                "adapter's tensor count and size, and reload-parity differences that are exactly zero or at round-off.\n\n"
                "<details><summary>Check your reasoning</summary>Exactly, on one device. The reloaded model is the same "
                "digest-verified base files plus the same adapter tensors (checked against the manifest's SHA-256 before "
                "loading), and float64 inference in evaluation mode is deterministic on one CPU, so this repository's CPU check "
                "printed 0.0 for both differences (the recorded Kaggle T4 run showed about 1e-14, GPU summation order). The "
                "adapter held 39 tensors in 16,155,811 bytes; on the three new sample structures the "
                "adapted model's errors were 0.0087 eV/atom and 0.0443 eV/Å. A parity failure would mean the adapter did not "
                "capture every change the adaptation made — which is why `adapt` refuses an already adapted pipeline.</details>"
            ),
        },
        {
            "md": (
                "## 10. Your turn — change one thing: which blocks train\n\n"
                "This activity fine-tunes a **second, separate pipeline** (`activity_pipe`, loaded from the same verified base "
                "files) with a different trainable scope, so the main model, its outputs and its adapter stay as they are. It uses "
                "the same splits and the Section 7 fields `EPOCHS`, `LEARNING_RATE` and `BATCH_SIZE`; only "
                "`ACTIVITY_TRAINABLE_BLOCKS` differs. On **Run all** it trains the readouts alone (`0`: 2,192 parameters), exports "
                "that adapter to `outputs/mace_materials_activity_adapter` and checks its reload parity too.\n\n"
                "**Predict → Change one thing → Run → Observe → Explain**\n\n"
                "1. **Predict.** Training only the readouts (2,192 parameters) instead of the last interaction block as well "
                "(1.92 M): will epoch 0 differ between the two runs? Which test error — energy or force — will suffer more?\n"
                "2. **Change one thing.** Leave `ACTIVITY_TRAINABLE_BLOCKS = 0` for the first pass; afterwards set it to `2` "
                "(everything trains).\n"
                "3. **Run.** Run this cell (on **Run all** it has already run once with `0`).\n"
                "4. **Observe.** The table prints the Section 7 run and every activity run of this session: trainable "
                "parameters, epoch-0 validation errors, the kept epoch, the test errors and the time.\n"
                "5. **Explain.** Why is epoch 0 the same in every row? Why does the force error barely move when only the "
                "readouts train? Is `2` worth its extra time on this small set?\n\n"
                "**Optional experiments** (each re-run from the cell named): a different `SEED` in Section 4 (Run after from "
                "Section 4) changes which structures are held out; more `EPOCHS` or another `LEARNING_RATE` in Section 7 (Run "
                "after from Section 7); your own extended-XYZ data through BYOD (Section 4) — read the composition baseline "
                "before the adapted number."
            ),
            "code": (
                "ACTIVITY_TRAINABLE_BLOCKS = 0  # @param [0, 1, 2]\n\n"
                "activity_pipe = MaceMaterialsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "t0 = time.perf_counter()\n"
                "activity_result = activity_pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=int(ACTIVITY_TRAINABLE_BLOCKS))\n"
                "activity_seconds = round(time.perf_counter() - t0, 1)\n"
                "activity_test = activity_pipe.evaluate(test_records)\n"
                "_, activity_parity, _ = export_and_reload(activity_pipe, OUTPUTS['activity_adapter'], {{'tutorial': '{stem}', 'activity': 'Section 10', 'data_source': data_source}})\n\n"
                "def activity_row(label, result, test, seconds):\n"
                "    epoch0 = result['history'][0]['val']\n"
                "    return {{'run': label, 'trainable_blocks': result['trainable_blocks'], 'trainable_parameters': result['n_trainable'], 'epoch0_val_energy_mae': round(epoch0['energy_mae_per_atom'], 5), 'epoch0_val_force_mae': round(epoch0['force_mae'], 5), 'best_epoch': result['best_epoch'], 'test_energy_mae': round(test['energy_mae_per_atom'], 5), 'test_force_mae': round(test['force_mae'], 5), 'seconds': seconds}}\n\n"
                "activity_history = globals().get('activity_history', [])\n"
                "activity_history.append(activity_row(f'activity {{len(activity_history) + 1}}', activity_result, activity_test, activity_seconds))\n"
                "print(activity_row('Section 7', adapt_result, test_metrics, adapt_seconds))\n"
                "for row in activity_history:\n"
                "    print(row)\n"
                "start_diff = abs(activity_result['history'][0]['val']['energy_mae_per_atom'] - adapt_result['history'][0]['val']['energy_mae_per_atom'])\n"
                "print({{'epoch0_difference_eV_per_atom': start_diff, 'activity_reload_parity': activity_parity}})\n"
                "assert start_diff < 1e-6, 'both runs must start from the same calibrated frozen model'\n"
                "del activity_pipe  # the main pipeline `pipe` is unchanged"
            ),
        },
        {
            "md": (
                "**What to notice:** the same epoch-0 validation errors in every row, the activity's reload parity at zero, and "
                "the test errors of the readouts-only run beside the Section 7 run.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 is the E0-calibrated frozen model, and both runs start "
                "from the same verified base weights and the same training split, so it is identical (0.0985 eV/atom, 0.1967 "
                "eV/Å in this repository's CPU check; the printed difference is 0). With only the 2,192 readout, E0 and scale/shift "
                "parameters training, the test errors were 0.0904 eV/atom and 0.2075 eV/Å (one to two minutes of training on CPU), against "
                "0.00725 and 0.07803 for the Section 7 run: the readouts can rescale the energy, but the forces depend on the "
                "geometry-dependent features of the interaction blocks, which stay frozen. Training everything (`2`, all 8.2 M parameters) costs more time per epoch; "
                "whether it is better on 27 training structures is what your extra row shows — with one seeded split, a small "
                "difference either way is not evidence.</details>"
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
        "**No uncertainty.** Every energy, force and stress this notebook prints is a point prediction: there is no "
        "per-prediction uncertainty, no committee or ensemble, and no out-of-domain flag. A structure unlike anything in the "
        "training data — another chemistry, a surface, a molecule in vacuum — is scored with the same silent confidence as one "
        "like the test split, so checking applicability is the user's job.\n\n"
        "Three things to carry to real data. **Reference consistency:** every label must come from one code, one functional and "
        "one set of settings; mixing references is the most common way to get an unlearnable dataset. **Splits:** structures "
        "from one trajectory or one relaxation are near-duplicates — split by trajectory or by composition, never at random over "
        "frames; tag your frames with `group=<id>` and keep `BYOD_SPLIT = 'auto'`. **Baselines first:** if the composition "
        "baseline is already good, your labels vary with stoichiometry more than with geometry, and the force error is the "
        "number to read.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned source asset, audit and convert a pickled checkpoint into a code-free "
        "serving pair without executing anything outside the audited allow-lists, rebuild the model from the installed library, "
        "validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate against two trivial baselines and the "
        "frozen model on an independent split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or physical validity beyond the checks "
        "shown.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment uses manylinux wheels; use Google Colab, Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails while downloading.** PyPI or the managed-Python download was interrupted: run the cell again (finished parts are reused). A size/SHA-256 refusal of the `uv` wheel that repeats means the download is being altered.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose **Run all**.\n"
        "- **Section 3 fails with a digest or size mismatch.** A snapshot file was truncated or altered: delete the `weights/` folder in the runtime and run Section 3 again.\n"
        "- **`this pipeline is already adapted`.** You called `pipe.adapt` yourself after an adaptation; call `pipe.reset_to_pretrained()` first (Section 7 does this for you).\n"
        "- **BYOD errors.** Each names the failed rule: no file or more than one uploaded; no upload dialog outside Colab (set `BYOD_PATH`); a file that is not `.xyz`/`.extxyz`; a frame without `energy` and `forces`; fewer than 8 or more than 1,000 structures; fewer than 3 groups for a grouped split; a split with fewer than 4 training, 1 validation or 1 test structure; or a validation/test element missing from training. Fix the file or the field (`BYOD_SPLIT`, `VAL_FRACTION`, `TEST_FRACTION`, `SEED`) and **Run after** from Section 4.\n"
        "- **Reload parity failed.** The adapter in the staging folder does not reproduce the model; the previous adapter is left in place. Re-run from Section 7.\n\n"
        "## Glossary\n\n"
        "- **Interatomic potential:** a function from atomic positions (and the cell) to a total energy; forces are minus its gradient.\n"
        "- **Per-atom energy (eV/atom):** the total energy divided by the number of atoms, so structures of different sizes compare.\n"
        "- **MAE / RMSE:** mean absolute error / root-mean-square error over structures (energy) or force components.\n"
        "- **Equivariance:** rotating the input rotates the forces the same way and leaves the energy unchanged.\n"
        "- **Level of theory:** the method that produced the labels (here EMT; MACE-MP-0 was trained on PBE DFT).\n"
        "- **E0 calibration:** fitting one reference energy per element so the model's energy zero matches the labels'.\n"
        "- **Composition baseline:** a per-element energy regression that cannot see geometry and predicts zero forces.\n"
        "- **Group split:** a split that keeps related structures (one trajectory, one composition) on one side.\n"
        "- **Adapter:** the changed tensors saved separately from the base model and overlaid on it at load time.\n"
        "- **Reload parity:** the reloaded model reproducing the in-memory model's predictions.\n"
        "- **Out-of-domain:** a structure unlike the training data; this notebook does not detect it.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional; not a required submission. Complete these sentences from your own run:\n\n"
        "1. The frozen model's force error was ___ against ___ for zero forces, and its energy error was large because ___.\n"
        "2. After calibration and fine-tuning the test energy MAE moved from ___ to ___ eV/atom.\n"
        "3. Training only the readouts changed the force error from ___ to ___, because ___.\n"
        "4. Before trusting an adapted potential on my own data, I would check ___.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/mace-materials-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/mace-foundations/mace-mp-0 (revision `{MODEL_REVISION}`)\n"
        "- Upstream release asset (byte-identical): https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0b2\n"
        "- Batatia et al., *A foundation model for atomistic materials chemistry*, arXiv:2401.00096 (2023): https://arxiv.org/abs/2401.00096\n"
        "- Batatia et al., *MACE: Higher order equivariant message passing neural networks for fast and accurate force fields*, NeurIPS 2022: https://arxiv.org/abs/2206.07697\n"
        "- ASE EMT calculator: https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
