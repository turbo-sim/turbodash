"""Evaluate the selected thesis example once and copy its geometry into Turbodash.

Run with an environment containing the thesis meanline dependencies and an
editable Turbodash install. This script never optimizes or edits the thesis.
The emitted configuration contains explicitly documented geometric assumptions.

    python development/impeller/export_meanline_example.py --thesis-root ../MastersThesis
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
OUTPUT_NAMES = (
    "source_meanline_result.json", "source_snapshot.yaml",
    "source_provenance.json", "reversible_impeller.yaml",
)
SOURCE_FILES = (
    "meanline/examples/example_reversible.yaml",
    "meanline/reversible_machine/configuration.py",
    "meanline/reversible_machine/design.py",
    "meanline/centrifugal_compressor/configuration.py",
    "meanline/centrifugal_compressor/design.py",
    "meanline/centrifugal_compressor/geometry.py",
    "meanline/centrifugal_compressor/fluids.py",
    "meanline/requirements-meanline.txt",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """Preserve JSON-compatible output values; inactive nonfinite values become null."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported meanline output value: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path, default=HERE.parents[2] / "MastersThesis")
    parser.add_argument("--output-dir", type=Path, default=HERE / "examples")
    parser.add_argument("--overwrite", action="store_true", help="Replace the four generated example files.")
    args = parser.parse_args()
    thesis = args.thesis_root.resolve()
    output_dir = args.output_dir.resolve()
    case_path = thesis / SOURCE_FILES[0]
    if not case_path.is_file():
        parser.error(f"Missing selected case: {case_path}")
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error("Refusing to overwrite existing files without --overwrite: " + ", ".join(existing))

    # Read-only use of the source checkout: suppress Python's bytecode cache writes.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(thesis / "meanline"))
    from reversible_machine.configuration import load_case
    from reversible_machine.design import evaluate_design
    from turbodash.geometry.impeller.meanline import from_meanline_geometry

    case = load_case(case_path)
    variables = {key: float(spec["value"]) for key, spec in case["design_variables"].items()}
    result = evaluate_design(variables, case)
    reported = {
        key: result[key] for key in (
            "flag", "objective", "charge_speed", "discharge_speed", "impeller_diameter",
            "compressor_eta_tt", "compressor_eta_ts", "compressor_beta_tt",
            "turbine_eta_tt", "turbine_eta_ts", "turbine_beta_tt",
            "rotor_margin", "vaned_margin", "constraints",
        )
    }
    reported["machines"] = {
        machine: {
            "flag": result[machine]["data"]["flag"],
            "partial_geometry": result[machine]["partial_geometry"],
            "geometry": result[machine]["geometry"],
        }
        for machine in ("compressor", "turbine")
    }
    raw = {
        "description": "Actual evaluate_design output subset at the committed YAML's starting values; no optimization.",
        "serialization": "NumPy values converted to ordinary numbers; inactive nonfinite values, if any, encoded as null. Flow-state objects omitted.",
        "design_variables": variables,
        "result": plain(reported),
    }
    violations = []
    for variable, relation, target, actual in result["constraints"]:
        difference = float(actual) - float(target)
        residual = {
            "=": abs(difference),
            ">": max(0.0, -difference),
            "<": max(0.0, difference),
        }[relation]
        if not math.isfinite(float(actual)) or residual > 1e-6:
            violations.append({
                "variable": variable, "relation": relation,
                "target": float(target), "actual": float(actual),
            })
    git_head = subprocess.check_output(["git", "-C", str(thesis), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(thesis), "status", "--short"], text=True).strip()
    provenance = {
        "source_repository": "MastersThesis",
        "source_case": SOURCE_FILES[0],
        "source_commit": git_head,
        "source_worktree_clean": not bool(git_status),
        "source_worktree_status": git_status,
        "source_sha256": {name: sha256(thesis / name) for name in SOURCE_FILES},
        "evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "reversible_machine.design.evaluate_design at each design_variables.*.value",
        "optimized": False,
        "adopted_thesis_design": False,
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "CoolProp", "jaxprop", "PyYAML")},
        },
        "reproduce_from_turbodash": "python development/impeller/export_meanline_example.py --thesis-root ../MastersThesis --output-dir development/impeller/examples --overwrite",
        "evaluation_status": {
            "flag": int(result["flag"]),
            "compressor_flag": int(result["compressor"]["data"]["flag"]),
            "turbine_flag": int(result["turbine"]["data"]["flag"]),
            "finite_performance": bool(all(math.isfinite(float(result[key])) for key in ("compressor_eta_tt", "turbine_eta_tt", "compressor_beta_tt", "turbine_beta_tt"))),
            "constraint_targets_satisfied": not bool(violations),
            "constraint_screen_absolute_tolerance": 1e-6,
            "violated_constraints": plain(violations),
            "note": "This is a starting design; the simple absolute constraint screen is in each reported quantity's units, not an optimizer acceptance check.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "source_meanline_result.json"
    result_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    provenance["result_file"] = result_path.name
    provenance["result_sha256"] = sha256(result_path)
    snapshot = {
        "description": "Completed compressor geometry copied from source_meanline_result.json; metres and degrees.",
        "source_result_sha256": provenance["result_sha256"],
        "geometry": raw["result"]["machines"]["compressor"]["geometry"],
    }
    (output_dir / "source_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    (output_dir / "source_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config = from_meanline_geometry(snapshot["geometry"], provenance=provenance)
    config["name"] = "reversible_example_shrouded_impeller"
    (output_dir / "reversible_impeller.yaml").write_text(
        "# Editable geometry seed from the selected reversible meanline example.\n"
        "# Source output, geometry overrides and assumed shaping controls are recorded below.\n"
        + yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    print(f"Wrote {len(OUTPUT_NAMES)} files to {output_dir}")
    print(f"Source {git_head}, clean={not bool(git_status)}, evaluator flag={result['flag']}")
    print("Starting design only; shrouded geometry override has not been re-evaluated aerodynamically.")


if __name__ == "__main__":
    main()
