"""MACE-MP-0b2 small (`mace-foundations/mace-mp-0`) DIMER pipeline: verified snapshot, energy / force /
stress prediction for atomic structures, and bounded fine-tuning of the foundation interatomic potential
to a user's reference data with a portable adapter.

MACE-MP-0 is a universal machine-learned interatomic potential (Batatia et al., 2023): an equivariant
message-passing model trained on the Materials Project PBE/PBE+U trajectories that predicts a total
energy, per-atom energies, forces and stress for any structure of the 89 supported elements. The "0b2
small" checkpoint is the 128x0e, two-interaction, 8.2 M-parameter variant with the Agnesi radial
transform and ZBL pair repulsion, published November 2024.

The upstream distribution format is a **pickled torch module** (`.model`). Under the fleet asset
specification (§11) that is executable serialization, not data, so this package never loads it as a
runtime artifact. Instead:

* `audit_model_file` statically disassembles the pickle (outer archive and the TorchScript archives
  e3nn embeds in it) without executing anything, lists every global it would import and every
  generated source string it would `exec`, and refuses anything outside the torch / e3nn / mace
  allow-lists. The audit result is digest-pinned in `PICKLE_AUDIT_SHA256`.
* `convert_model` unpickles the digest-verified, audit-clean source **once**, extracts the
  constructor arguments and the state dict, and writes a code-free pair: a JSON config plus a
  safetensors file. Their digests are pinned in `CONVERTED_SHA256`; the conversion is deterministic.
* `MaceMaterialsPipeline.from_pretrained` rebuilds `ScaleShiftMACE(**config)` from the **installed**
  `mace-torch` package and loads the safetensors with `strict=True`. The DIMER-hosted artifact is the
  converted pair; the `.model` file is the immutable provenance, not the served asset.

Behaviour is preserved: on three probe structures (rattled Cu and Al supercells, an H2O molecule) the
rebuilt model and the pickled model differ by at most 2.8e-14 eV in energy and 3.6e-15 eV/Å in
forces, and the upstream `MACECalculator` agrees to the same precision (docs/WEIGHTS.md).

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input
validation run (and can refuse) before `torch`, `e3nn` or `mace` are imported (fleet RTM-001). `ase`
and `numpy` are used for structure handling and are imported freely.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import pickletools
import re
import time
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def _float64_default() -> Any:
    """Run a block with torch's default dtype set to float64 (the model's dtype) and hand the caller's default
    back afterwards, even on failure. Loading this pipeline must not change tensor creation elsewhere in the
    hosting process, while every tensor mace-torch builds for us (model, graph data) must be float64."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)

MODEL_ID = "mace-foundations/mace-mp-0"
MODEL_REVISION = "e291ace2bfae073c3ebc7ae2f9479a525989baa7"
MODEL_LICENSE = "mit"
MODEL_KEY = "mace-mp-0b2-small"
ARTIFACT_FORMAT = "org.valcorza.mace-materials.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (pickled torch module). Byte-identical to the upstream release asset
# ACEsuit/mace-mp `mace_mp_0b2/mace-small-density-agnesi-stress.model` (2024-11-12); see docs/WEIGHTS.md.
SOURCE_MODEL_NAME = "mace-mp-0b2-small.model"
SOURCE_MODEL_BYTES = 67_622_684
SOURCE_MODEL_SHA256 = "d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9"
# Code-free serving pair produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_CONFIG_NAME = "mace-mp-0b2-small.config.json"
CONVERTED_WEIGHTS_NAME = "mace-mp-0b2-small.safetensors"
CONVERTED_SHA256 = {
    CONVERTED_CONFIG_NAME: "130b641179c5da6dbf6ec439b9c20661e3d60512007f6f2d42669d4f87fc9893",
    CONVERTED_WEIGHTS_NAME: "2ed99065c4decf21613b7038dde6bda2b694b2f414012c0ca743f7ff7f86fe93",
}
CONVERTED_BYTES = {CONVERTED_CONFIG_NAME: 3_279, CONVERTED_WEIGHTS_NAME: 67_400_566}
# Digest of the static pickle audit (sorted globals + every generated source string, see audit_model_file).
PICKLE_AUDIT_SHA256 = "9bb150f1ecc9212e594313ba0ef9aa4884b7634dfc8a11da8cbe0c9ebf70d1e5"

# Architecture facts asserted against the converted config before the model library is imported.
MODEL_CLASS = "ScaleShiftMACE"
R_MAX = 5.0
NUM_INTERACTIONS = 2
HIDDEN_IRREPS = "128x0e"
NUM_ELEMENTS = 89
CORRELATION = 3
MAX_ELL = 3
SUPPORTED_ATOMIC_NUMBERS: tuple[int, ...] = tuple(range(1, 84)) + tuple(range(89, 95))  # H..Bi, Ac..Pu
PARAMETER_COUNT = 8_221_984
STATE_TENSORS = 86  # 80 weight / buffer tensors from the pickle + 6 path-pruning flags of mace-torch 0.3.16

# Operational ceilings (structural validation, before any model import).
MAX_STRUCTURES_PER_CALL = 32
MAX_ATOMS_PER_STRUCTURE = 512
MAX_ATOMS_PER_CALL = 4_096
MIN_INTERATOMIC_DISTANCE = 0.5  # Å; closer than any chemistry, the ZBL core would dominate
MAX_CELL_LENGTH = 200.0  # Å
MIN_CELL_HEIGHT = 1.0  # Å for each periodic direction
MAX_ABS_COORDINATE = 10_000.0  # Å
MAX_ABS_ENERGY_PER_ATOM = 1_000.0  # eV
MAX_ABS_FORCE = 1_000.0  # eV/Å

_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")

# Static pickle-audit allow-lists (asset spec §11): what a MACE `.model` file may legitimately reference.
PICKLE_ALLOWED_MODULE_ROOTS = ("torch", "e3nn", "mace", "collections", "_codecs", "__torch__")
PICKLE_ALLOWED_EXACT_GLOBALS = frozenset({"__builtin__.set"})
# Every line torch.fx's `reduce_graph_module` would exec from the pickled import block, verbatim.
GENERATED_CODE_ALLOWED_IMPORT_LINES = frozenset(
    {
        "import torch",
        "from math import inf",
        "from math import nan",
        "from torch import device",
        "import torch.fx._pytree as fx_pytree",
        "import torch.utils._pytree as pytree",
        "NoneType = type(None)",
    }
)
GENERATED_CODE_FORBIDDEN = re.compile(
    r"\b(os|subprocess|exec|eval|compile|open|__import__|socket|sys|builtins|importlib|pickle|shutil|"
    r"requests|urllib|globals|locals|breakpoint|input|ctypes|signal|threading|multiprocessing)\b"
)
GENERATED_CODE_CALL_PREFIXES = ("torch.", "fx_pytree.", "pytree.", "math.", "device(")


# --------------------------------------------------------------------------------------------------
# snapshot manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_MODEL_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_MODEL_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_MODEL_NAME and (digest != SOURCE_MODEL_SHA256 or size != SOURCE_MODEL_BYTES):
            raise ValueError(f"{SOURCE_MODEL_NAME}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the code-free serving pair (config JSON + safetensors) against the pinned digests."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    report: dict[str, Any] = {"files": []}
    for name, expected in CONVERTED_SHA256.items():
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"converted file missing: {file_path}")
        size = file_path.stat().st_size
        if size != CONVERTED_BYTES[name]:
            raise ValueError(f"{name}: size {size} != pinned {CONVERTED_BYTES[name]}")
        digest = _sha256_file(file_path)
        if digest != expected:
            raise ValueError(f"{name}: sha256 {digest} != pinned {expected}")
        report["files"].append({"path": name, "bytes": size, "sha256": digest})
    return report


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and,
    when the converted serving pair is present, that pair against the pinned digests."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = all((root / name).is_file() for name in CONVERTED_SHA256)
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and the
    3 KB converted config but git-ignores the 67.6 MB `.model` source and the safetensors it converts
    to). `verify_snapshot` still runs after; conversion happens in `from_pretrained`."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _audit_pickle_bytes(data: bytes, report: dict[str, Any]) -> None:
    """Walk one pickle stream with `pickletools.genops` (no execution) and record what it references."""
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            report["globals"][key] = report["globals"].get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            report["globals"][key] = report["globals"].get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
            if isinstance(arg, str):
                if arg.startswith("PK\x03\x04"):
                    report["nested_archives"] += 1
                    _audit_archive_bytes(arg.encode("latin-1"), report)
                elif "def forward" in arg:
                    report["code_strings"].append(arg)
                elif re.search(r"^(?:import|from)\s", arg, re.M):
                    report["import_blocks"].append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)


def _audit_archive_bytes(blob: bytes, report: dict[str, Any]) -> None:
    archive = zipfile.ZipFile(io.BytesIO(blob))
    for name in archive.namelist():
        if name.endswith(".pkl"):
            _audit_pickle_bytes(archive.read(name), report)
        elif name.endswith(".py"):
            report["torchscript_sources"].append(archive.read(name).decode("utf-8"))


def audit_model_file(path: str | Path | None = None) -> dict[str, Any]:
    """Statically audit the pickled `.model` file without executing it.

    Disassembles the outer torch archive and every nested TorchScript archive with `pickletools`,
    collects every GLOBAL the unpickler would import, every fx-generated source string it would
    `exec`, and every TorchScript source it would compile, then checks them against the allow-lists.
    Raises `ValueError` on any violation. Returns the audit report with its digest."""
    file_path = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR / SOURCE_MODEL_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"source model not found: {file_path}")
    report: dict[str, Any] = {
        "file": file_path.name,
        "globals": {},
        "nested_archives": 0,
        "code_strings": [],
        "import_blocks": [],
        "torchscript_sources": [],
    }
    _audit_archive_bytes(file_path.read_bytes(), report)
    violations: list[str] = []
    for name in sorted(report["globals"]):
        root = name.split(".")[0]
        if name in PICKLE_ALLOWED_EXACT_GLOBALS or root in PICKLE_ALLOWED_MODULE_ROOTS:
            continue
        violations.append(f"global outside allow-list: {name}")
    import_lines: set[str] = set()
    for block in report["import_blocks"]:
        import_lines.update(line.strip() for line in block.splitlines() if line.strip())
    for code in report["code_strings"]:
        for match in re.finditer(r"^\s*((?:from|import)\s[^\n]+)", code, re.M):
            import_lines.add(match.group(1).strip())
    for line in sorted(import_lines - GENERATED_CODE_ALLOWED_IMPORT_LINES):
        violations.append(f"generated code import line outside allow-list: {line}")
    for kind, sources in (
        ("fx", report["code_strings"]),
        ("import-block", report["import_blocks"]),
        ("torchscript", report["torchscript_sources"]),
    ):
        for src in sources:
            hit = GENERATED_CODE_FORBIDDEN.search(src)
            if hit:
                violations.append(f"forbidden token {hit.group(0)!r} in {kind} source")
                break
    for code in report["code_strings"]:
        for match in re.finditer(r"(?<![\w.])([A-Za-z_][\w.]*)\(", code):
            callee = match.group(1)
            if callee in ("forward", "slice", "int", "float", "tuple", "list", "range", "len", "getattr_1"):
                continue
            if not callee.startswith(GENERATED_CODE_CALL_PREFIXES) and "." not in callee:
                violations.append(f"generated code calls a bare name: {callee}")
                break
    summary = {
        "globals": sorted(report["globals"]),
        "nested_archives": report["nested_archives"],
        "fx_code_strings": len(report["code_strings"]),
        "import_blocks": len(report["import_blocks"]),
        "torchscript_sources": len(report["torchscript_sources"]),
        "generated_code_import_lines": sorted(import_lines),
        "violations": violations,
    }
    digest = hashlib.sha256()
    digest.update("\n".join(summary["globals"]).encode("utf-8"))
    digest.update(b"\n--fx--\n" + "\n---\n".join(sorted(report["code_strings"])).encode("utf-8"))
    digest.update(b"\n--imports--\n" + "\n---\n".join(sorted(report["import_blocks"])).encode("utf-8"))
    digest.update(b"\n--ts--\n" + "\n---\n".join(sorted(report["torchscript_sources"])).encode("utf-8"))
    summary["audit_sha256"] = digest.hexdigest()
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed: {violations}")
    return summary


def _config_from_json(config: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the converted config names the pinned architecture (before any model import)."""
    expected = {
        "model_class": MODEL_CLASS,
        "r_max": R_MAX,
        "num_interactions": NUM_INTERACTIONS,
        "hidden_irreps": HIDDEN_IRREPS,
        "num_elements": NUM_ELEMENTS,
        "correlation": CORRELATION,
        "max_ell": MAX_ELL,
        "dtype": "float64",
        "heads": ["default"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"converted config {key}={config.get(key)!r} disagrees with the package constant {value!r}")
    if tuple(config.get("atomic_numbers", ())) != SUPPORTED_ATOMIC_NUMBERS:
        raise ValueError("converted config atomic_numbers disagree with SUPPORTED_ATOMIC_NUMBERS")
    if len(config.get("atomic_energies", ())) != NUM_ELEMENTS:
        raise ValueError("converted config atomic_energies length != NUM_ELEMENTS")
    return dict(config)


def _json_ready(value: Any) -> Any:
    """Serialise `extract_config_mace_model` output: classes and callables by name, irreps as strings."""
    import numpy as np
    import torch
    from e3nn import o3

    if isinstance(value, type):
        return value.__name__
    if isinstance(value, o3.Irreps):
        return str(value)
    if callable(value):
        return value.__name__
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(x) if isinstance(x, np.integer) else x for x in value]
    return value


def build_model(config: Mapping[str, Any]) -> Any:
    """Instantiate `ScaleShiftMACE` from the converted JSON config using the installed mace-torch."""
    import mace  # noqa: F401  (import order: mace sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before e3nn loads its constants)
    import numpy as np
    import torch
    from e3nn import o3
    from mace.modules import ScaleShiftMACE, blocks

    # The model is float64: construct it under that default and hand the caller's default back afterwards.
    with _float64_default():
        kwargs = {k: v for k, v in config.items() if k not in ("model_class", "dtype")}
        kwargs["interaction_cls"] = getattr(blocks, kwargs["interaction_cls"])
        kwargs["interaction_cls_first"] = getattr(blocks, kwargs["interaction_cls_first"])
        kwargs["readout_cls"] = getattr(blocks, kwargs["readout_cls"])
        kwargs["hidden_irreps"] = o3.Irreps(kwargs["hidden_irreps"])
        kwargs["MLP_irreps"] = o3.Irreps(kwargs["MLP_irreps"])
        kwargs["gate"] = {"silu": torch.nn.functional.silu}[kwargs["gate"]]
        kwargs["atomic_energies"] = np.array(kwargs["atomic_energies"], dtype=float)
        kwargs["atomic_inter_scale"] = float(kwargs["atomic_inter_scale"])
        kwargs["atomic_inter_shift"] = float(kwargs["atomic_inter_shift"])
        if kwargs.get("edge_irreps") is not None:
            kwargs["edge_irreps"] = o3.Irreps(kwargs["edge_irreps"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ScaleShiftMACE(**kwargs)


# State-dict keys of the pickled checkpoint that mace-torch 0.3.16 no longer registers (the ZBL block now
# derives its cutoff from covalent radii); their values (p=5, r_max=5.0) are not used by the current forward.
STALE_SOURCE_KEYS = ("pair_repulsion_fn.cutoff.p", "pair_repulsion_fn.cutoff.r_max", "pair_repulsion_fn.r_max")


def convert_model(path: str | Path | None = None, *, audit: bool = True) -> dict[str, Any]:
    """Convert the digest-verified `.model` pickle into the code-free serving pair, deterministically.

    Order of operations: digest check → static audit (refuses on any allow-list violation or audit
    digest drift) → single unpickle (`torch.load(weights_only=False)`, the only place this package
    executes anything from the file) → `extract_config_mace_model` → JSON config + safetensors state
    dict of a freshly constructed model that loaded the pickled tensors. Returns the digests."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = root / SOURCE_MODEL_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source model not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_MODEL_BYTES:
        raise ValueError(f"{SOURCE_MODEL_NAME}: size {size} != pinned {SOURCE_MODEL_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_MODEL_SHA256:
        raise ValueError(f"{SOURCE_MODEL_NAME}: sha256 {digest} != pinned {SOURCE_MODEL_SHA256}")
    if audit:
        summary = audit_model_file(source)
        if summary["audit_sha256"] != PICKLE_AUDIT_SHA256:
            raise ValueError(
                f"{SOURCE_MODEL_NAME}: pickle audit digest {summary['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}"
            )
    import mace  # noqa: F401
    import torch
    from mace.tools.scripts_utils import extract_config_mace_model
    from safetensors.torch import save_file

    # Conversion runs under a float64 default and restores the caller's default even on failure.
    with _float64_default():
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pickled = torch.load(source, map_location="cpu", weights_only=False)
        if type(pickled).__name__ != MODEL_CLASS:
            raise ValueError(f"{SOURCE_MODEL_NAME} unpickled to {type(pickled).__name__}, expected {MODEL_CLASS}")
        config = {k: _json_ready(v) for k, v in extract_config_mace_model(pickled).items()}
        config["model_class"] = type(pickled).__name__
        config["dtype"] = str(next(pickled.parameters()).dtype).replace("torch.", "")
        _config_from_json(config)
        text = json.dumps(config, indent=2, sort_keys=True) + "\n"
        (root / CONVERTED_CONFIG_NAME).write_bytes(text.encode("utf-8"))

        source_state = {k: v.contiguous() for k, v in pickled.state_dict().items()}
        model = build_model(config)
        result = model.load_state_dict({k: v for k, v in source_state.items() if k not in STALE_SOURCE_KEYS}, strict=False)
        if result.unexpected_keys or any(not k.endswith("_zeroed") for k in result.missing_keys):
            raise ValueError(
                f"state-dict layout drift between the pickle and mace-torch: "
                f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
        canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
        if len(canonical) != STATE_TENSORS:
            raise ValueError(f"converted state dict has {len(canonical)} tensors, expected {STATE_TENSORS}")
        for key, tensor in source_state.items():
            if key not in STALE_SOURCE_KEYS and not torch.equal(tensor, canonical[key]):
                raise ValueError(f"converted tensor {key} is not bit-identical to the pickled one")
        save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
        report = verify_converted(root)
        return {
            "source": {"path": SOURCE_MODEL_NAME, "bytes": size, "sha256": digest},
            "converted": report["files"],
            "dropped_keys": list(STALE_SOURCE_KEYS),
            "added_flag_keys": sorted(result.missing_keys),
            "seconds": round(time.perf_counter() - started, 2),
        }


# --------------------------------------------------------------------------------------------------
# structures and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "1..MAX_STRUCTURES_PER_CALL structures, each {symbols, positions, cell, pbc} (Å), "
        "optionally energy (eV) and forces (eV/Å)"
    ),
    "structures": [1, MAX_STRUCTURES_PER_CALL],
    "atoms_per_structure": [1, MAX_ATOMS_PER_STRUCTURE],
    "atoms_per_call": [1, MAX_ATOMS_PER_CALL],
    "elements": "the 89 elements of MACE-MP-0 (Z = 1..83 and 89..94)",
    "min_interatomic_distance_angstrom": MIN_INTERATOMIC_DISTANCE,
    "cell": f"3x3 lattice vectors in Å, each shorter than {MAX_CELL_LENGTH} Å; a cell is required whenever any pbc flag is true",
    "validation": (
        "element support, finite coordinates, cell/pbc consistency and interatomic-distance and size ceilings only. "
        "Nothing checks that a structure is chemically sensible, charge-neutral or near equilibrium; a random cloud "
        "of supported atoms is scored without complaint"
    ),
    "units": "energies in eV, forces in eV/Å, stress in eV/Å³ (ASE conventions)",
}


def _atomic_symbols() -> dict[str, int]:
    from ase.data import atomic_numbers

    return dict(atomic_numbers)


def _check_structure(structure: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Return a normalised copy of one structure or raise ValueError describing the first defect."""
    import numpy as np

    label = f"structure[{index}]"
    if not isinstance(structure, Mapping):
        raise ValueError(f"{label} must be a mapping with symbols/positions/cell/pbc, got {type(structure).__name__}")
    symbols = structure.get("symbols")
    positions = structure.get("positions")
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise ValueError(f"{label}: symbols must be a non-empty list of element symbols")
    n_atoms = len(symbols)
    if n_atoms > MAX_ATOMS_PER_STRUCTURE:
        raise ValueError(f"{label}: {n_atoms} atoms exceed the ceiling of {MAX_ATOMS_PER_STRUCTURE}")
    table = _atomic_symbols()
    numbers = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not _SYMBOL_RE.match(symbol) or symbol not in table:
            raise ValueError(f"{label}: {symbol!r} is not an element symbol")
        z = table[symbol]
        if z not in SUPPORTED_ATOMIC_NUMBERS:
            raise ValueError(f"{label}: element {symbol} (Z={z}) is outside the 89 elements MACE-MP-0 supports")
        numbers.append(z)
    try:
        pos = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: positions must be an (n_atoms, 3) array of numbers") from exc
    if pos.shape != (n_atoms, 3):
        raise ValueError(f"{label}: positions shape {pos.shape} != ({n_atoms}, 3)")
    if not np.all(np.isfinite(pos)) or np.abs(pos).max() > MAX_ABS_COORDINATE:
        raise ValueError(f"{label}: positions must be finite and within ±{MAX_ABS_COORDINATE} Å")
    pbc_raw = structure.get("pbc", False)
    if isinstance(pbc_raw, bool):
        pbc = [pbc_raw] * 3
    else:
        pbc = list(pbc_raw) if isinstance(pbc_raw, (list, tuple)) else None
        if pbc is None or len(pbc) != 3 or not all(isinstance(flag, (bool, int)) for flag in pbc):
            raise ValueError(f"{label}: pbc must be a bool or a list of three bools")
        pbc = [bool(flag) for flag in pbc]
    cell_raw = structure.get("cell")
    cell = None
    if cell_raw is not None:
        try:
            cell = np.asarray(cell_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: cell must be a 3x3 array of numbers") from exc
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError(f"{label}: cell must be a finite 3x3 array")
        lengths = np.linalg.norm(cell, axis=1)
        if lengths.max() > MAX_CELL_LENGTH:
            raise ValueError(f"{label}: cell vector of {lengths.max():.1f} Å exceeds {MAX_CELL_LENGTH} Å")
        volume = abs(float(np.linalg.det(cell)))
        if any(pbc):
            if volume <= 0.0:
                raise ValueError(f"{label}: periodic cell is singular")
            for axis in range(3):
                if pbc[axis]:
                    cross = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
                    height = volume / np.linalg.norm(cross)
                    if height < MIN_CELL_HEIGHT:
                        raise ValueError(
                            f"{label}: periodic cell height {height:.3f} Å along axis {axis} is below {MIN_CELL_HEIGHT} Å"
                        )
    elif any(pbc):
        raise ValueError(f"{label}: pbc requests periodicity but no cell is given")
    from ase.geometry import get_distances

    if n_atoms > 1:
        _, dists = get_distances(pos, cell=cell if any(pbc) else None, pbc=pbc if any(pbc) else None)
        np.fill_diagonal(dists, np.inf)
        d_min = float(dists.min())
        if d_min < MIN_INTERATOMIC_DISTANCE:
            raise ValueError(f"{label}: two atoms are {d_min:.3f} Å apart, below the {MIN_INTERATOMIC_DISTANCE} Å floor")
    else:
        d_min = math.inf
    energy = structure.get("energy")
    forces_raw = structure.get("forces")
    if energy is not None:
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
            raise ValueError(f"{label}: energy must be a finite number in eV")
        if abs(float(energy)) / n_atoms > MAX_ABS_ENERGY_PER_ATOM:
            raise ValueError(f"{label}: |energy| per atom exceeds {MAX_ABS_ENERGY_PER_ATOM} eV")
    forces = None
    if forces_raw is not None:
        try:
            forces = np.asarray(forces_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: forces must be an (n_atoms, 3) array of numbers") from exc
        if forces.shape != (n_atoms, 3) or not np.all(np.isfinite(forces)) or np.abs(forces).max() > MAX_ABS_FORCE:
            raise ValueError(f"{label}: forces must be a finite ({n_atoms}, 3) array within ±{MAX_ABS_FORCE} eV/Å")
    name = structure.get("name")
    if name is not None and (not isinstance(name, str) or len(name) > 200):
        raise ValueError(f"{label}: name must be a string of at most 200 characters")
    return {
        "name": name if name is not None else f"structure-{index}",
        "symbols": [str(s) for s in symbols],
        "numbers": numbers,
        "positions": pos.tolist(),
        "cell": cell.tolist() if cell is not None else None,
        "pbc": pbc,
        "energy": float(energy) if energy is not None else None,
        "forces": forces.tolist() if forces is not None else None,
        "n_atoms": n_atoms,
        "min_distance": d_min,
        "periodic": any(pbc),
    }


def _check_structures(structures: Any) -> list[dict[str, Any]]:
    if isinstance(structures, Mapping) or not isinstance(structures, Sequence) or isinstance(structures, (str, bytes)):
        raise ValueError("structures must be a list of structure mappings")
    if not structures:
        raise ValueError("at least one structure is required")
    if len(structures) > MAX_STRUCTURES_PER_CALL:
        raise ValueError(f"{len(structures)} structures exceed the ceiling of {MAX_STRUCTURES_PER_CALL} per call")
    checked = [_check_structure(s, i) for i, s in enumerate(structures)]
    total = sum(s["n_atoms"] for s in checked)
    if total > MAX_ATOMS_PER_CALL:
        raise ValueError(f"{total} atoms in one call exceed the ceiling of {MAX_ATOMS_PER_CALL}")
    return checked


def structure_digest(structure: Mapping[str, Any]) -> str:
    """SHA-256 of the geometry (symbols, positions rounded to 1e-6 Å, cell, pbc); labels excluded."""
    checked = _check_structure(structure, 0)
    payload = {
        "symbols": checked["symbols"],
        "positions": [[round(x, 6) for x in row] for row in checked["positions"]],
        "cell": [[round(x, 6) for x in row] for row in checked["cell"]] if checked["cell"] else None,
        "pbc": checked["pbc"],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_inputs(structures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Structural validation only; raises ValueError before any model library is imported."""
    checked = _check_structures(structures)
    return {
        "n_structures": len(checked),
        "n_atoms": sum(s["n_atoms"] for s in checked),
        "elements": sorted({s for c in checked for s in c["symbols"]}),
        "periodic": [c["periodic"] for c in checked],
        "min_distance": min(c["min_distance"] for c in checked),
        "labelled": sum(1 for c in checked if c["energy"] is not None and c["forces"] is not None),
    }


def to_atoms(structure: Mapping[str, Any]) -> Any:
    """ase.Atoms for one validated structure (labels attached as `info['energy']` / `arrays['forces']`)."""
    import numpy as np
    from ase import Atoms

    checked = _check_structure(structure, 0)
    atoms = Atoms(symbols=checked["symbols"], positions=np.array(checked["positions"]), pbc=checked["pbc"])
    if checked["cell"] is not None:
        atoms.set_cell(np.array(checked["cell"]))
    if checked["energy"] is not None:
        atoms.info["energy"] = checked["energy"]
    if checked["forces"] is not None:
        atoms.arrays["forces"] = np.array(checked["forces"])
    atoms.info["name"] = checked["name"]
    return atoms


def from_atoms(atoms: Any, *, name: str | None = None) -> dict[str, Any]:
    """Structure mapping from ase.Atoms; picks up energy/forces from `info`/`arrays` or a SinglePointCalculator."""
    import numpy as np

    energy = atoms.info.get("energy")
    forces = atoms.arrays.get("forces")
    calc = getattr(atoms, "calc", None)
    if calc is not None and getattr(calc, "results", None):
        energy = calc.results.get("energy", energy)
        forces = calc.results.get("forces", forces)
    cell = np.array(atoms.get_cell())
    has_cell = bool(np.any(cell != 0.0))
    return {
        "name": name or atoms.info.get("name"),
        "symbols": list(atoms.get_chemical_symbols()),
        "positions": atoms.get_positions().tolist(),
        "cell": cell.tolist() if has_cell else None,
        "pbc": [bool(x) for x in atoms.get_pbc()],
        "energy": float(energy) if energy is not None else None,
        "forces": np.asarray(forces, dtype=float).tolist() if forces is not None else None,
    }


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------

TRAINABLE_ALWAYS = ("readouts.", "atomic_energies_fn.", "scale_shift.")
ENERGY_WEIGHT = 100.0  # on the per-atom energy MSE (eV/atom)²
FORCES_WEIGHT = 10.0  # on the force-component MSE (eV/Å)²


def _trainable_prefixes(trainable_blocks: int) -> tuple[str, ...]:
    if not isinstance(trainable_blocks, int) or not 0 <= trainable_blocks <= NUM_INTERACTIONS:
        raise ValueError(f"trainable_blocks must be an int in 0..{NUM_INTERACTIONS}")
    prefixes = list(TRAINABLE_ALWAYS)
    for k in range(NUM_INTERACTIONS - trainable_blocks, NUM_INTERACTIONS):
        prefixes += [f"interactions.{k}.", f"products.{k}."]
    return tuple(prefixes)


@dataclass
class MaceMaterialsPipeline:
    """Energy / force / stress prediction and bounded fine-tuning on top of the verified MACE-MP-0b2 small model."""

    model: Any
    config: dict[str, Any]
    device: str
    weights_dir: Path
    source: str
    adapter: dict[str, Any] | None = None
    _z_table: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str = "cpu",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> MaceMaterialsPipeline:
        """Verify, convert if needed, rebuild and strictly load. With `require_source=False` the pickled
        source may be absent (the DIMER-hosted case) as long as the converted pair verifies. `report`
        receives the static-audit summary and the conversion record when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                if report is not None:
                    audit = audit_model_file(root / SOURCE_MODEL_NAME)
                    report(
                        {
                            "pickle_audit": {k: v for k, v in audit.items() if k != "globals"},
                            "global_count": len(audit["globals"]),
                        }
                    )
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted pair already present and digest-verified"})
            source = "converted from the manifest-verified source pickle"
        else:
            # Converted-only deployment: only the code-free pair is verified, whether or not the committed
            # source manifest happens to sit beside it; the pickle is never required or fetched here.
            verify_converted(root)
            source = "converted pair, pinned digests (source pickle not required)"
        config = _config_from_json(json.loads((root / CONVERTED_CONFIG_NAME).read_text(encoding="utf-8")))
        import torch
        from safetensors.torch import load_file

        model = build_model(config)
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        model.load_state_dict(state, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(device)).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        from mace.tools import AtomicNumberTable

        return cls(
            model=model,
            config=config,
            device=device,
            weights_dir=root,
            source=source,
            _z_table=AtomicNumberTable(list(SUPPORTED_ATOMIC_NUMBERS)),
        )

    # ---- batching -------------------------------------------------------------------------------------

    def _dataset(self, checked: Sequence[Mapping[str, Any]], *, labels: bool) -> list[Any]:
        from mace.data import AtomicData, config_from_atoms
        from mace.data.utils import KeySpecification

        spec = KeySpecification(info_keys={"energy": "energy"}, arrays_keys={"forces": "forces"})
        data = []
        with _float64_default():  # graph tensors follow the default dtype; the model is float64
            for structure in checked:
                atoms = to_atoms(structure)
                if labels and (structure["energy"] is None or structure["forces"] is None):
                    raise ValueError(f"{structure['name']}: energy and forces are required for this operation")
                conf = config_from_atoms(atoms, key_specification=spec)
                data.append(AtomicData.from_config(conf, z_table=self._z_table, cutoff=R_MAX))
        return data

    def _loader(self, data: Sequence[Any], batch_size: int, shuffle: bool, seed: int = 0) -> Any:
        import torch
        from mace.tools import torch_geometric

        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return torch_geometric.dataloader.DataLoader(
            list(data), batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=generator
        )

    def _forward(self, model: Any, batch: Any, *, training: bool, stress: bool) -> dict[str, Any]:
        import torch

        batch = batch.to(self.device)  # graph tensors are built on the CPU; the model may live on CUDA
        data = batch.to_dict()
        # Forces are -dE/dx, so the forward needs autograd even at inference; no_grad would zero them.
        out = model(data, training=training, compute_force=True, compute_stress=stress)
        if not training:
            out = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        return out

    # ---- inference ------------------------------------------------------------------------------------

    def predict(self, structures: Sequence[Mapping[str, Any]], *, batch_size: int = 8) -> dict[str, Any]:
        """Total energy (eV), per-atom energies, forces (eV/Å) and, for fully periodic structures, stress (eV/Å³)."""
        checked = _check_structures(structures)
        started = time.perf_counter()
        data = self._dataset(checked, labels=False)
        model = self.model
        # Stress is a batch-level switch in the MACE forward, so fully periodic structures (stress-eligible)
        # and everything else are batched separately; results are put back in the caller's order.
        eligible = [i for i, s in enumerate(checked) if s["periodic"] and all(s["pbc"])]
        others = [i for i in range(len(checked)) if i not in set(eligible)]
        results: list[dict[str, Any] | None] = [None] * len(checked)
        for indices, stress in ((eligible, True), (others, False)):
            if not indices:
                continue
            position = 0
            for batch in self._loader([data[i] for i in indices], batch_size, shuffle=False):
                out = self._forward(model, batch, training=False, stress=stress)
                ptr = batch.ptr.tolist()
                energies = out["energy"].cpu().tolist()
                node_energy = out["node_energy"].cpu()
                forces = out["forces"].cpu()
                stresses = out.get("stress")
                for i in range(len(ptr) - 1):
                    lo, hi = ptr[i], ptr[i + 1]
                    results[indices[position]] = {
                        "energy": float(energies[i]),
                        "energy_per_atom": float(energies[i]) / (hi - lo),
                        "node_energies": node_energy[lo:hi].tolist(),
                        "forces": forces[lo:hi].tolist(),
                        "stress": stresses[i].cpu().tolist() if (stress and stresses is not None) else None,
                    }
                    position += 1
        if any(r is None for r in results):
            raise RuntimeError("prediction did not cover every structure")
        for structure, result in zip(checked, results, strict=True):
            result["name"] = structure["name"]
            result["n_atoms"] = structure["n_atoms"]
            result["periodic"] = structure["periodic"]
            if not structure["periodic"] or not all(structure["pbc"]):
                result["stress"] = None
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "units": {"energy": "eV", "forces": "eV/Å", "stress": "eV/Å³"},
            "results": results,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, structures: Sequence[Mapping[str, Any]], *, batch_size: int = 8) -> dict[str, Any]:
        """Energy and force errors against the reference labels carried by the structures."""
        from .metrics import regression_metrics

        checked = _check_structures(structures)
        if any(s["energy"] is None or s["forces"] is None for s in checked):
            raise ValueError("every structure needs energy and forces for evaluation")
        prediction = self.predict(checked, batch_size=batch_size)
        return regression_metrics(checked, prediction["results"])

    # ---- adaptation -----------------------------------------------------------------------------------

    def calibrate_e0(self, structures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Least-squares per-element energy offsets between the reference labels and the current model,
        added to the model's atomic reference energies. Two-parameter change on a Cu/Al dataset; the
        cheapest possible adaptation to a different level of theory. Returns the shifts in eV."""
        import numpy as np
        import torch

        checked = _check_structures(structures)
        if any(s["energy"] is None for s in checked):
            raise ValueError("every structure needs an energy for E0 calibration")
        prediction = self.predict(checked)
        elements = sorted({s for c in checked for s in c["symbols"]})
        counts = np.array([[c["symbols"].count(el) for el in elements] for c in checked], dtype=float)
        residual = np.array([c["energy"] - r["energy"] for c, r in zip(checked, prediction["results"], strict=True)])
        shifts, *_ = np.linalg.lstsq(counts, residual, rcond=None)
        table = _atomic_symbols()
        with torch.no_grad():
            energies = self.model.atomic_energies_fn.atomic_energies
            for element, shift in zip(elements, shifts, strict=True):
                energies[self._z_table.z_to_index(table[element])] += float(shift)
        return {"elements": elements, "shifts_ev": [float(s) for s in shifts], "n_structures": len(checked)}

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 1e-3,
        batch_size: int = 4,
        trainable_blocks: int = 1,
        energy_weight: float = ENERGY_WEIGHT,
        forces_weight: float = FORCES_WEIGHT,
        calibrate: bool = True,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning to the reference energies and forces of `train`.

        Always trains the readouts, the per-element reference energies and the scale/shift block;
        `trainable_blocks` (0..2) additionally unfreezes that many trailing interaction + product
        blocks (1 = the last block, 1.92 M of 8.22 M parameters). Loss = energy_weight × MSE(per-atom
        energy) + forces_weight × MSE(force components), Adam, fixed learning rate, no scheduler. The
        epoch with the lowest validation loss is kept; epoch 0 records the (E0-calibrated) frozen model
        so every number is comparable to the starting point."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 0.1):
            raise ValueError("lr must be in (0, 0.1]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_STRUCTURES_PER_CALL:
            raise ValueError(f"batch_size must be an int in 1..{MAX_STRUCTURES_PER_CALL}")
        prefixes = _trainable_prefixes(trainable_blocks)
        train_checked = validate_dataset(train, min_records=4)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val else []
        import torch

        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.model
        pre_state = copy.deepcopy(model.state_dict())  # restored if anything below raises
        calibration = None
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith(prefixes))
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.Adam(params, lr=lr)
        train_data = self._dataset(train_checked, labels=True)
        val_data = self._dataset(val_checked, labels=True) if val_checked else []

        def weighted_loss(out: Mapping[str, Any], batch: Any) -> Any:
            device = out["energy"].device
            n_atoms = batch.ptr.to(device).diff().to(out["energy"].dtype)
            e_loss = torch.mean(((out["energy"] - batch.energy.to(device)) / n_atoms) ** 2)
            f_loss = torch.mean((out["forces"] - batch.forces.to(device)) ** 2)
            return energy_weight * e_loss + forces_weight * f_loss

        def val_loss() -> float | None:
            if not val_data:
                return None
            model.eval()
            total, count = 0.0, 0
            for batch in self._loader(val_data, batch_size, shuffle=False):
                out = self._forward(model, batch, training=False, stress=False)
                total += float(weighted_loss(out, batch)) * batch.num_graphs
                count += batch.num_graphs
            return total / count

        try:
            calibration = self.calibrate_e0(train_checked) if calibrate else None
            history: list[dict[str, Any]] = []
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = 0
            entry: dict[str, Any] = {
                "epoch": 0,
                "train_loss": None,
                "val_loss": val_loss(),
                "note": "frozen model after E0 calibration" if calibrate else "frozen model",
            }
            if val_checked:
                entry["val"] = self.evaluate(val_checked)
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            if progress:
                progress(entry)
            for epoch in range(1, epochs + 1):
                model.train()
                total, count = 0.0, 0
                for batch in self._loader(train_data, batch_size, shuffle=True, seed=seed + epoch):
                    out = self._forward(model, batch, training=True, stress=False)
                    loss = weighted_loss(out, batch)
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * batch.num_graphs
                    count += batch.num_graphs
                model.eval()
                entry = {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss()}
                if val_checked:
                    entry["val"] = self.evaluate(val_checked)
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in calibration, training, validation or the progress callback
            # leaves the model exactly as it was before adapt(), frozen, with no adapter attached.
            model.load_state_dict(pre_state, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        model.load_state_dict(best_state, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_prefixes": list(prefixes),
            "trainable_blocks": trainable_blocks,
            "n_trainable": n_trainable,
            "n_total": n_total,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "calibration": calibration,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors (only those the adaptation could change) as safetensors plus a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefixes = tuple(self.adapter["trainable_prefixes"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k.startswith(prefixes)}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "converted_sha256": dict(CONVERTED_SHA256),
            },
            "adapter": {k: v for k, v in self.adapter.items() if k != "history"},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> tuple[Path, list[str]]:
        """Validate an adapter manifest before anything is deserialised: format and version, the pinned base,
        exactly one weights entry named `adapter.safetensors` inside the artifact directory, an in-range
        `trainable_blocks`, and a tensor list equal to the exact set that scope implies for this base."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records different converted-base digests")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        if not isinstance(adapter, Mapping):
            raise ValueError("artifact manifest has no adapter record")
        trainable_blocks = adapter.get("trainable_blocks")
        if isinstance(trainable_blocks, bool) or not isinstance(trainable_blocks, int):
            raise ValueError("artifact adapter.trainable_blocks must be an int")
        prefixes = _trainable_prefixes(trainable_blocks)  # range-checked there
        expected = sorted(k for k in self.model.state_dict() if k.startswith(prefixes))
        if not isinstance(manifest.get("tensors"), list) or sorted(manifest["tensors"]) != expected:
            raise ValueError(
                f"artifact tensor list does not match the {len(expected)} tensors that "
                f"trainable_blocks={trainable_blocks} may change on this base"
            )
        return weights_path, expected

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the tensors the scope allows."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, expected = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        digest = _sha256_file(weights_path)
        if digest != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, base has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str = "cpu",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> MaceMaterialsPipeline:
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
