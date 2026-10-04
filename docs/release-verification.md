# Release verification

`tutorials/mace_materials_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated uv install
  (pinned `uv` wheel checked by size and SHA-256, managed CPython 3.12.12, the carried hash lock installed with
  `--require-hashes --only-binary :all: --no-deps`) and the router that sends every later cell to that environment;
  `NOTEBOOK_SOURCE` recorded in exports; the guided layer (who it is for, Input → Model → Output, roadmap, predictions,
  worked answers, the Section 10 activity, troubleshooting, glossary) and no stale restart or install text;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `MaceMaterialsPipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the static audit and the
  conversion are printed before the model loads, `generate_sample_dataset`, `validate_dataset`, `split_dataset`,
  `write_dataset_xyz`, `validate_inputs` with the refusal probes, `pipe.predict` with the rotation, translation,
  permutation, finite-difference and extensivity assertions, `composition_baseline`, `zero_force_baseline`,
  `pipe.evaluate` on the frozen model and on both the validation and the test split after adaptation with the
  baseline assertions, `pipe.adapt` with its explicit hyperparameters and the printed E0 calibration,
  `pipe.save_artifact`, `MaceMaterialsPipeline.from_artifact` and the reload-parity assertion), the five expected
  `outputs/` paths, the learner-facing statements (the pickle is audited and unpickled once into a code-free pair,
  forces are the analytic gradient, the composition baseline is blind to geometry, EMT is a different level of
  theory, multi-head replay is out of scope, split by trajectory or composition) and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `mace` / `MACECalculator` use,
  `torch.load(`, `weights_only=False` or `pickle.load` **outside the carried module cells**,
  `trust_remote_code=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; crafted pickles, temporary manifests and
plain-Python structures, no weights and no model library — `tests/test_model_backed.py` is skipped there). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/mace-mp-0b2-small/` (the standalone path writes the manifest itself, stages both listed files
   from the Hub, audits and converts the pickle, so the directory may not be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`, `SEED = 42`, `EPOCHS = 8`,
   `LEARNING_RATE = 1e-3`, `BATCH_SIZE = 4`, `TRAINABLE_BLOCKS = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `mace-torch==0.3.16`, `e3nn==0.4.4`, `ase==3.29.0`, `numpy==2.5.3`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0`, installed by the isolated-environment cell from the carried hash
   lock (`tutorials/requirements-colab.lock.txt`, 73 packages) into managed CPython 3.12.12, with **no restart** and
   no error output anywhere in the run (record "1 pass, 0 restarts");
5. verify every default-path stage completes:
   - the isolated environment built from the carried lock with no GitHub access, and every later cell routed to it;
   - the three carried module cells execute (defining `MaceMaterialsPipeline`, `audit_model_file`,
     `convert_model`, `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`,
     `validate_inputs`, `to_atoms`, `from_atoms`, `generate_sample_dataset`, `build_sample_structure`,
     `validate_dataset`, `split_dataset`, `write_dataset_xyz`, `load_byod_dataset`, `regression_metrics`,
     `composition_baseline`, `zero_force_baseline`) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 2 entries fetched from `mace-foundations/mace-mp-0` at the immutable revision and
     `verify_snapshot` reporting 2 verified files;
   - the model cell printing the **pickle audit** (18 nested archives, 28 fx sources, 10 import blocks, 30
     TorchScript sources, 0 violations, audit SHA-256 `9bb150f1…`) and the **conversion record** (config
     `130b6411…` 3,279 bytes, safetensors `2ed99065…` 67,400,566 bytes, 3 dropped keys, 6 added flags), then
     the load report on the chosen device with source "converted from the manifest-verified source pickle";
   - the dataset manifest with 48 records, compositions `Cu32`/`Al32`/`Al16Cu16` at 16 each, 1,536 atoms, elements
     `['Al', 'Cu']`, digest `d2409300…`, the splits 27 / 9 / 12, the written `outputs/mace_materials_sample_dataset.xyz`,
     and four refusals (unsupported element, overlapping atoms, pbc without a cell, malformed forces);
   - `pipe.predict` on 8 test structures and the physics checks: rotation, translation and permutation differences at
     the 1e-13 level, finite-difference-versus-analytic force below 1e-6 eV/Å, extensivity below 1e-8 eV/atom (the
     cell asserts all of them);
   - the composition baseline (on the sample: ≈ 0.09 eV/atom), the zero-force baseline (≈ 0.74 eV/Å) and the
     frozen model's error (≈ 3.98 eV/atom, ≈ 0.21 eV/Å; the cell asserts the frozen model beats the zero-force
     baseline);
   - `pipe.adapt` printing epoch 0 as the E0-calibrated frozen model, 1,919,128 trainable of 8,221,984 parameters,
     the Al/Cu calibration shifts (≈ +3.8 / +4.2 eV), and an eight-epoch history with per-epoch validation metrics;
   - `pipe.evaluate` on the test split with the comparison table and `outputs/mace_materials_evaluation_report.json`
     written (the cell asserts the adapted energy MAE is below the composition baseline and the adapted force MAE
     below the frozen model's);
   - three freshly generated structures predicted with stress tensors and `outputs/mace_materials_predictions.json`
     written;
   - `pipe.save_artifact` writing `outputs/mace_materials_adapter/{adapter.safetensors,manifest.json}` (39 tensors,
     about 16 MB), and `MaceMaterialsPipeline.from_artifact` reloading it with identical energies and forces (the
     cell asserts both below 1e-9);
   - `outputs/mace_materials_result.json` written with the model identity, the comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache and the weights directory were clean, outcome, produced outputs, the
   observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD` deviation in the
   tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `mace_materials_colab.ipynb` (`E2E`) | `cc91bbd` / `1044b4aa` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-mace-materials` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda:0`) | **Completed only after a manual restart; not one-pass evidence** — completed only after a manual restart — pass 1 stopped in the install cell at its restart guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3), pass 2 after an executor restart ran 11/11 code cells ok; 9 files, 135 MB staged from the Hub into a clean cache (the `.model` source fetched and converted in the notebook); comparison {energy_mae_per_atom: {composition_baseline: 0.09291, frozen_model: 3.981, adapted: 0.00725}, force_mae: {zero_force_baseline: 0.7376, frozen_model: 0.2079, adapted: 0.07803}}; reload parity {max_abs_energy_diff: 1.421e-14, max_abs_force_diff: 7.772e-15}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-mace-materials/v3/evidence/` in the workspace |
| `mace_materials_colab.ipynb` | review-fix head of PR #9 / `6df363ea` | 2026-10-05 | Local pre-flight harness (Windows, CPython 3.12, CPU float64, `torch 2.14.0+cpu`, the notebook's pins in a private venv **without `python-hostlist`**, `DIMER_NOTEBOOK_CI_PREINSTALLED=1` so the isolated-environment cells were skipped) | PASS — pre-flight only, **not** promotion evidence; the isolated uv install itself was only dry-run resolved for manylinux x86_64 |
| `mace_materials_colab.ipynb` | `9afc026` / `1b9359b3` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/mace_materials_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/mace_materials_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-05 | review-fix head of PR #9 / `6df363ea` | Local pre-flight harness (Windows, CPython 3.12, CPU float64, `torch 2.14.0+cpu`, `mace-torch 0.3.16` without `python-hostlist`; the `.model` source pre-staged, the notebook wrote the manifest, verified, audited and converted it) | Default path (Sections 3–10, including the Section 10 activity), then the documented experiment `TRAINABLE_BLOCKS = 0` with Sections 7–9 re-run; separately a BYOD re-run of the sample XYZ after a default run, and the BYOD contract probes (grouped trajectories, one structure per composition, 4 compositions × 2 end to end, 201 structures through Sections 4, 6 and 7, missing path, no Colab, unlabelled frames) | 187.6 s (journey A) | **PASS, pre-flight only** — splits 27/9/12; test comparison {composition 0.09291, frozen 3.98075, adapted 0.00725 eV/atom; zero-force 0.7376, frozen 0.20787, adapted 0.07803 eV/Å}, identical to the 2026-09-19 Kaggle numbers; reload parity 0.0 / 0.0; activity (readouts only) test 0.0904 / 0.2075 with the same epoch 0 and parity 0.0 / 0.0; `TRAINABLE_BLOCKS = 0` re-run: epoch 0 identical to the default run's (0.0985 eV/atom, difference 0.0), 12-tensor adapter, reload parity 0.0 / 0.0; BYOD re-run Section 6 frozen 3.98075 / 0.20787 (the true frozen model). Not a hosted run, not a Run all through the isolated environment |
| 2026-09-19 | `cc91bbd` / `1044b4aa` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-mace-materials` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 243.3 s | **Completed only after a manual restart; not one-pass evidence** — completed only after a manual restart — pass 1 stopped in the install cell at its restart guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3), pass 2 after an executor restart ran 11/11 code cells ok; 9 files, 135 MB staged from the Hub into a clean cache (the `.model` source fetched and converted in the notebook); comparison {energy_mae_per_atom: {composition_baseline: 0.09291, frozen_model: 3.981, adapted: 0.00725}, force_mae: {zero_force_baseline: 0.7376, frozen_model: 0.2079, adapted: 0.07803}}; reload parity {max_abs_energy_diff: 1.421e-14, max_abs_force_diff: 7.772e-15}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-mace-materials/v3/evidence/` in the workspace |
| 2026-09-18 | `9afc026` / `1b9359b3` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float64, `torch 2.14.0+cpu`, `mace-torch 0.3.16`, `e3nn 0.4.4`, `ase 3.29.0`) | Default sample path (stage → verify → **static audit + conversion in the notebook** → strict rebuild → generate + validate → split → refusal probes → predict + physics checks → baselines + frozen evaluation → calibrate + adapt → evaluate → predict new → export → reload); the two Hub files were pre-staged, so `stage_missing_files` fetched 0 of 2 entries, `verify_snapshot` verified 2, and `convert_model` produced the pinned digests (config `130b6411…`, safetensors `2ed99065…`) | 85.8 s | **PASSED** — 11/11 code cells; audit 0 violations, digest `9bb150f1…`; physics checks 0.0 / 2.8×10⁻¹⁴ / 0.0 / 0.0 / 5.2×10⁻¹⁵ / 2.0×10⁻⁸ / 1.3×10⁻¹⁵ / 0.0; test (n = 12): composition baseline 0.0929 eV/atom, zero-force 0.7376 eV/Å, frozen 3.9807 / 0.2079, adapted **0.00725 eV/atom / 0.0780 eV/Å** after 8 epochs (68.5 s, 1,919,128 params, calibration Al +3.768 / Cu +4.176 eV); new structures 0.0087 / 0.0443; adapter 16,155,811 B (39 tensors); reload parity 0.0 / 0.0. Pre-flight; hosted clean-runtime run still required |

## Current status

**Candidate.** The `E2E` notebook blob `1044b4aa` (committed at `cc91bbd`) ran in a clean Kaggle Tesla T4 runtime on 2026-09-19 with no repository checkout, but it completed only after a manual restart — pass 1 stopped in the install cell at its restart guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3), pass 2 after an executor restart ran 11/11 code cells ok (243.3 s, 9 files, 135 MB fetched from the Hub and digest-verified inside the notebook). A run that needs a restart is not the one-pass Run-all evidence REL10/REL11 require, and that blob has since been replaced: the review fixes (PR #9) move the install into an isolated uv environment, reset the model before every frozen-model and adaptation section, group BYOD splits, and add the guided layer and the Section 10 activity. The local CPU pre-flight of the new notebook below is not promotion evidence; a one-pass hosted run (Colab, or Kaggle T4) of the current blob covering the default path, recorded here as "1 pass, 0 restarts", is required before the status can change. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
