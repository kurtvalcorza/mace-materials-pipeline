# Weight provenance, the pickle audit, the conversion and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`. Its weight entry is a pickled torch module, which this pipeline audits and converts but never serves.

## MACE-MP-0b2 small weights

- Upstream: `mace-foundations/mace-mp-0`
- Immutable revision: `e291ace2bfae073c3ebc7ae2f9479a525989baa7`
- Source format: pickled `ScaleShiftMACE` module in a torch zip archive (`mace-mp-0b2-small.model`, 67,622,684 bytes, 87 zip entries, 83 state tensors, 8,221,984 float64 parameters)
- Upstream weight license: MIT (`license: mit` in the pinned upstream README front matter and in the Hub repository metadata; the `ACEsuit/mace` code and the `mace-mp` release assets are MIT)
- Local layout: `weights/mace-mp-0b2-small/` holds the 2 manifest entries (`mace-mp-0b2-small.model`, upstream `README.md`; 67,622,708 bytes total) with byte size and SHA-256 for each, plus the converted pair described below. `verify_snapshot()` in `src/mace_materials_pipeline/pipeline.py` checks the manifest entries, asserts the source digest against the package constant, and checks the converted pair against its pinned digests when present.
- Cross-checks: the manifest's source digest `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9` equals the Hub `X-Linked-ETag` of the file at the pinned revision **and** the SHA-256 of the upstream GitHub release asset `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model` (release published 2024-11-12; downloaded and hashed at build time). The Hub repository (created 2025-05-22) is a mirror of that release.

## What the pickle would execute, and how it is audited

A `.model` file is executable serialization under the fleet asset specification (§11). Statically disassembled with `pickletools` — outer archive plus the 18 nested TorchScript archives that e3nn's code-generation mixin embeds as latin-1 strings — unpickling it would:

- import **61 globals**: `torch` (storages, `_rebuild_tensor_v2`, `_rebuild_parameter`, `nn.functional.silu`, `ModuleList`, `ParameterList`, `fx._symbolic_trace.Tracer`, `fx.graph_module.reduce_graph_module`, `jit._pickle.restore_type_tag`), 13 `e3nn` classes (irreps, `Linear`, `TensorProduct`, `FullyConnectedTensorProduct`, `SphericalHarmonics`, `FullyConnectedNet`, `Activation`, …), 17 `mace.modules` classes (`ScaleShiftMACE`, the interaction, product, readout, radial, cutoff, ZBL and symmetric-contraction blocks), `collections.OrderedDict`, `_codecs.encode`, `__builtin__.set`, and 18 TorchScript-mangled `__torch__.torch.fx.graph_module.___torch_mangle_N` names;
- `exec` **10 fx-generated source strings** (tensor-product forwards: reshapes, `einsum`, `tensordot`, `cat`, `_assert`) together with **10 import blocks** consisting of exactly `import torch`, `from math import inf`, `from math import nan`, `from torch import device`, `import torch.fx._pytree as fx_pytree`, `import torch.utils._pytree as pytree` and `NoneType = type(None)` — this is `torch.fx.graph_module.reduce_graph_module`'s mechanism;
- compile **30 TorchScript sources** (the same tensor-product graphs, plus `torch.functional.tensordot`) — `torch.jit.load` on the nested archives.

`audit_model_file()` collects all of that without executing anything, refuses any global whose root is not `torch`, `e3nn`, `mace`, `collections`, `_codecs` or `__torch__` (with `__builtin__.set` allowed explicitly), any import line outside the seven above, any forbidden token (`os`, `subprocess`, `exec`, `eval`, `compile`, `open`, `__import__`, `socket`, `sys`, `builtins`, `importlib`, `pickle`, `shutil`, `requests`, `urllib`, `globals`, `locals`, `breakpoint`, `input`, `ctypes`, `signal`, `threading`, `multiprocessing`) in any generated or TorchScript source, and any bare-name call in fx-generated code. The audit of the pinned file reports 0 violations and the SHA-256 `9bb150f1ecc9212e594313ba0ef9aa4884b7634dfc8a11da8cbe0c9ebf70d1e5` over its sorted globals and every source string; `convert_model()` refuses a file whose audit digest differs. Tests craft pickles with `os.system`, `builtins.eval` (STACK_GLOBAL), a `subprocess` call in generated code, `import os` in an import block and `os.getcwd()` in a nested TorchScript source and assert each is refused.

An allow-list bounds what the unpickler can name; it is not a proof of safety for arbitrary constructions over the allowed classes. The digest pin ties the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs in this order: size check → SHA-256 check against `SOURCE_MODEL_SHA256` → static audit and audit-digest check → the single `torch.load(weights_only=False)` → model-class check → `mace.tools.scripts_utils.extract_config_mace_model` → JSON config (classes and callables by name, irreps as strings, arrays as lists, sorted keys) asserted against the package constants → a fresh `ScaleShiftMACE(**config)` from the installed `mace-torch` loaded with the pickled tensors → bit-identity check of every carried tensor → safetensors.

The pickle's state dict and `mace-torch==0.3.16`'s layout differ in nine buffers, and `convert_model` accepts exactly this difference and nothing else:

- **dropped (3):** `pair_repulsion_fn.cutoff.p`, `pair_repulsion_fn.cutoff.r_max`, `pair_repulsion_fn.r_max` (all 5.0) — the ZBL block of the mace version that produced the pickle carried its own cutoff; the current `ZBLBasis` derives the cutoff radius from the covalent radii it also carries (`pair_repulsion_fn.covalent_radii`, identical) with the same polynomial order (`pair_repulsion_fn.p` = 5, identical), and its forward does not read the dropped buffers;
- **added (6):** `products.{0,1}.symmetric_contractions.contractions.0.weights_{0,1,max}_zeroed` — boolean path-pruning flags the current `Contraction` constructor registers from its U-matrices (all `False` here, i.e. no path is pruned); they are not read by the forward.

Both models therefore run the same 0.3.16 forward over the same 80 tensors. Measured at build time on a rattled Cu₃₂ cell, a rattled Al₃₂ cell and an H₂O molecule (float64, CPU): energies −127.705045, −117.066553, −14.169908 eV from both; max |Δ| 2.8×10⁻¹⁴ eV (energy), 3.6×10⁻¹⁵ eV/Å (forces), 2.3×10⁻¹⁶ eV/Å³ (stress), 1.8×10⁻¹⁵ eV (per-atom energies); upstream's `MACECalculator(model_paths=<pickle>)` agrees with the batched energies to 2.8×10⁻¹⁴ eV. The comparison is under one `mace-torch` version on both sides; the pickle's behaviour under the version that produced it was not measured.

Serving pair (both identities recorded, `derived_from_sha256` = `d5773bf9…`):

| File | Bytes | SHA-256 | In Git |
|---|---|---|---|
| `mace-mp-0b2-small.config.json` | 3,279 | `130b641179c5da6dbf6ec439b9c20661e3d60512007f6f2d42669d4f87fc9893` | yes |
| `mace-mp-0b2-small.safetensors` | 67,400,566 | `2ed99065c4decf21613b7038dde6bda2b694b2f414012c0ca743f7ff7f86fe93` | no (regenerated) |

The conversion is deterministic: the digests were reproduced on every run at build time and by the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks sizes and digests; `_config_from_json()` asserts `model_class`, `r_max`, `num_interactions`, `hidden_irreps`, `num_elements`, `correlation`, `max_ell`, `dtype`, `heads` and the 89 atomic numbers against the package constants before any model library is imported.

## Runtime facts

- The model is float64 as shipped; `build_model()` sets `torch.set_default_dtype(torch.float64)` before construction so every buffer the constructor creates matches the pickled ones.
- Importing `mace` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the whole process (it lets `e3nn==0.4.4` load its Wigner constants under torch ≥ 2.6). This pipeline loads its own artifacts through safetensors only, but any other `torch.load` in the same process is affected; `build_model` imports `mace` before `e3nn` for that reason.
- Forces are `−∂E/∂x` by autograd, so `predict` runs the forward with gradients enabled and detaches afterwards; `torch.no_grad()` would zero the forces.
- `mace-torch` pulls `e3nn==0.4.4`, `opt_einsum`, `matscipy`, `torch-ema`, `torchmetrics`, `h5py`, `lmdb`, `orjson`, `pandas`, `matplotlib`, `GitPython`, `prettytable`, `python-hostlist`, `configargparse`, `pyYAML` and `tqdm` as its own dependencies; the pipeline uses `mace.modules`, `mace.data`, `mace.tools` and `mace.tools.scripts_utils.extract_config_mace_model` only.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `mace-mp-0b2-medium.model` (79,462,798 bytes), `mace-mp-0b2-large.model` (98,544,548 bytes), `mace-mp-0b3-medium.model` (79,472,952 bytes) and `.gitattributes` (1,519 bytes). None is listed in the manifest; the medium and large 0b2 checkpoints would go through the same audit + conversion path if a row for them were built.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence and copyright notice. DIMER may host the converted pair in its model store under those terms; the pair is derived from, and recorded beside, the unmodified upstream asset.
- Upload set: `mace-mp-0b2-small.safetensors` + `mace-mp-0b2-small.config.json`. **The `.model` pickle must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class comes from `mace-torch==0.3.16` on PyPI, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified pair without the manifest or the pickle.
- Serving shape: single-point prediction needs the 67 MB pair and about 1 s to construct on CPU (float64; 32 atoms in ~80 ms); an adapted profile needs the pair plus a 16 MB adapter (default) or a 20 KB readouts-only adapter.
- The open review item: whether the one-time audited unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER bar, or whether DIMER hosts only a pair converted and verified once by the maintainer. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest — this matters here because `mace-mp-0b2-small.config.json` is committed.
