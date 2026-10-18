"""Local paths, repository source selection and faithful result serialization."""

import csv
from copy import deepcopy
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import warnings

BASE = Path(__file__).resolve().parents[1]
AUDIT = BASE.parent
REPO = BASE.parents[2]
SOURCE = REPO
if not (SOURCE / "turbodash/__init__.py").is_file():
    raise RuntimeError("The repository turbodash package is missing; keep this audit inside improving_turbodash.")
os.environ["MPLCONFIGDIR"] = str(BASE / "data/.cache/matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import numpy as np


def imported_sources():
    """Fail if an installed copy shadows the intended repository functions."""
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name != "turbodash" and not name.startswith("turbodash."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(SOURCE.resolve()):
            raise RuntimeError(f"External model import: {name} from {path}")
        paths[name] = dict(path=str(path), relative_path=path.relative_to(REPO).as_posix(), sha256=sha256(path))
    return paths


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def output_path(path):
    return Path(path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt" if path.suffix == ".gz" else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_digest(path):
    """Hash parsed numbers/statuses, independent of gzip headers and float spelling."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            normalized = {}
            for key, value in row.items():
                try:
                    normalized[key] = {"number": float(value).hex()}
                except (ValueError, TypeError):
                    normalized[key] = {"text": value}
            rows.append(normalized)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return dict(rows=len(rows), sha256=hashlib.sha256(payload).hexdigest())


def modules(version):
    imported_sources()
    package = "turbodash.loss_models_new" if version == "candidate" else "turbodash.loss_models"
    result = {key: importlib.import_module(package + "." + name) for key, name in {
        "benner": "loss_model_benner",
        "ko": "loss_model_kacker_okapuu",
        "mo": "loss_model_moustapha",
        "conversion": "loss_coefficient_conversion",
        "dispatch": "loss_model",
    }.items()} | {"math": importlib.import_module("turbodash.math")}
    imported_sources()
    return result


def evaluate(call):
    result = {}
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        try:
            result = {k: float(v) for k, v in call().items()}
            status = "ok" if all(np.isfinite(v) for v in result.values()) else "nonfinite"
            error = ""
        except Exception as exc:
            status, error = "exception", type(exc).__name__ + ": " + str(exc)
    result.update(status=status, error=error, warnings=" | ".join(dict.fromkeys(str(w.message) for w in caught)))
    return result


def array_case(case):
    case = deepcopy(case)
    for section in ("flow", "geometry"):
        case[section] = {k: np.float64(v) if isinstance(v, (float, int)) else v for k, v in case[section].items()}
    return case
