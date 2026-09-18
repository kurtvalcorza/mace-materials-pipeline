# Release verification

`tutorials/mace_materials_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
   (= `pyproject.toml`): `torch==2.14.0`, `mace-torch==0.3.16`, `e3nn==0.4.4`, `ase==3.29.0`, `numpy==2.5.3`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0` (an interpreter restart after the install is expected where
   the runtime's preinstalled torch differs from the pin);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
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
| `mace_materials_colab.ipynb` | `9afc026` / `1b9359b3` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/mace_materials_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/mace_materials_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `9afc026` / `1b9359b3` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float64, `torch 2.14.0+cpu`, `mace-torch 0.3.16`, `e3nn 0.4.4`, `ase 3.29.0`) | Default sample path (stage → verify → **static audit + conversion in the notebook** → strict rebuild → generate + validate → split → refusal probes → predict + physics checks → baselines + frozen evaluation → calibrate + adapt → evaluate → predict new → export → reload); the two Hub files were pre-staged, so `stage_missing_files` fetched 0 of 2 entries, `verify_snapshot` verified 2, and `convert_model` produced the pinned digests (config `130b6411…`, safetensors `2ed99065…`) | 85.8 s | **PASSED** — 11/11 code cells; audit 0 violations, digest `9bb150f1…`; physics checks 0.0 / 2.8×10⁻¹⁴ / 0.0 / 0.0 / 5.2×10⁻¹⁵ / 2.0×10⁻⁸ / 1.3×10⁻¹⁵ / 0.0; test (n = 12): composition baseline 0.0929 eV/atom, zero-force 0.7376 eV/Å, frozen 3.9807 / 0.2079, adapted **0.00725 eV/atom / 0.0780 eV/Å** after 8 epochs (68.5 s, 1,919,128 params, calibration Al +3.768 / Cu +4.176 eV); new structures 0.0087 / 0.0443; adapter 16,155,811 B (39 tensors); reload parity 0.0 / 0.0. Pre-flight; hosted clean-runtime run still required |

## Current status

The notebook source is complete and passes all static checks, including the generator parity checks (`--check` OK).
A local pre-flight execution of the committed blob completed the whole default path on CPU — including the static
audit and the conversion of the downloaded pickle inside the notebook — which catches defects but is **not** a
supported runtime under REL1/REL10, and it ran with the two Hub files pre-staged, so the 67.6 MB Hub download has not
been exercised end to end by the notebook; the hosted run must cover it. The repository stays at **Candidate** until a
Colab or fresh-container run of the exact release revision is recorded above.
