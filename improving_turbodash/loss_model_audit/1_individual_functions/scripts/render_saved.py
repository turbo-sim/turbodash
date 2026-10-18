"""Redraw all audit figures from saved values/derivatives; no model evaluation."""

import csv
import gzip
import json
from time import perf_counter

from common import BASE, imported_sources, sha256, write_json
import numpy as np
from plotting import AXES, axis_label, plot_group


def saved_arrays(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [key for key in rows[0] if key not in ("version", "x", "base_grid", "status", "error", "warnings")]
    result = {}
    for version in dict.fromkeys(row["version"] for row in rows):
        selected = [row for row in rows if row["version"] == version]
        def array(key):
            return np.array([float(row[key]) if row[key] else np.nan for row in selected])
        result[version] = dict(x=array("x"), values={key: array(key) for key in keys if not key.startswith("d_")}, derivatives={key: array(key) for key in keys if key.startswith("d_")})
    return result


def render_validation():
    with (BASE / "data/validation/samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    arrays = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    study = dict(lo=-2, hi=2, xlabel="Known-function coordinate x", kind="helper")
    values = {key: value for key, value in arrays.items() if key != "x" and not key.startswith("d_")}
    derivatives = {key: value for key, value in arrays.items() if key.startswith("d_")}
    curves = [("known-function audit stencil", study, {"validation": dict(x=arrays["x"], values=values, derivatives=derivatives)})]
    for group, metrics in [("near_constant_validation", ["linear", "tiny_linear_gradient"]), ("features_and_gaps_validation", ["genuine_kink", "genuine_jump", "explicit_gap"])]:
        plot_group(group, curves, metrics, directory=BASE / "figures")


def main():
    started = perf_counter()
    sweeps = json.loads((BASE / "data/sweep_manifest.json").read_text(encoding="utf-8"))
    plots = json.loads((BASE / "data/plot_manifest.json").read_text(encoding="utf-8"))
    numeric_paths = [BASE / study["data_path"] for study in sweeps] + [BASE / "data/validation/samples.csv"]
    from profile_final_review import DATA as FINAL_DATA, TABLES as FINAL_TABLES
    numeric_paths.extend(FINAL_DATA / (name+".csv") for name in FINAL_TABLES)
    from secondary_review import DATA as SECONDARY_DATA, TABLES as SECONDARY_TABLES
    numeric_paths.extend(SECONDARY_DATA / (name+".csv") for name in SECONDARY_TABLES)
    from clearance_review import DATA as CLEARANCE_DATA, TABLES as CLEARANCE_TABLES
    numeric_paths.extend(CLEARANCE_DATA / (name+".csv") for name in CLEARANCE_TABLES)
    from trailing_review import DATA as TRAILING_DATA, TABLES as TRAILING_TABLES
    numeric_paths.extend(TRAILING_DATA / (name+".csv") for name in TRAILING_TABLES)
    from ko_safeguards_review import DATA as SAFEGUARD_DATA, TABLES as SAFEGUARD_TABLES
    numeric_paths.extend(SAFEGUARD_DATA / (name+".csv") for name in SAFEGUARD_TABLES)
    before = {path.relative_to(BASE).as_posix(): sha256(path) for path in numeric_paths}
    labels = {"incidence": "Inlet flow minus fixed inlet metal angle (deg)", "exit_angle": "Signed exit angle in the fixed design outlet direction (deg)", "raw_penetration": "Raw penetration / full height", "chi": "Benner incidence parameter chi", "delta_phi2": "Kinetic-energy loss increment delta_phi^2"}
    for number, plot in enumerate(plots, 1):
        curves = []
        for source in sweeps:
            if source["group"] != plot["group"]:
                continue
            study = dict(source)
            study["xlabel"] = labels.get(study.get("variable"), AXES.get(study.get("variable"), axis_label(study["group"])))
            curves.append((study["family"], study, saved_arrays(BASE / study["data_path"])))
        metrics = [key for key in next(iter(curves[0][2].values()))["values"] if key != "floor_margin"]
        if plot["group"].startswith("ko_profile_"):
            metrics = ["Y_profile"]
        plot_group(plot["group"], curves, metrics, curves[0][1]["log"], (BASE / plot["path"]).parent)
        if number % 20 == 0 or number == len(plots):
            print(f"Rendered {number}/{len(plots)} comparison figures", flush=True)
    render_validation()
    from reynolds_review import render as render_reynolds
    from profile_review import render as render_profiles
    render_reynolds()
    render_profiles()
    from profile_interpolation_review import render as render_interpolation
    render_interpolation()
    from profile_terms import render as render_terms
    terms = render_terms()
    from profile_final_review import render as render_final
    final = render_final()
    from secondary_review import render as render_secondary
    secondary = render_secondary()
    from clearance_review import render as render_clearance
    clearance = render_clearance()
    from trailing_review import render as render_trailing
    trailing = render_trailing()
    from ko_safeguards_review import render as render_safeguards
    safeguards = render_safeguards()
    supplemental_count = 4 + 1 + 4 + terms["figures"] + final["figures"]
    after = {path.relative_to(BASE).as_posix(): sha256(path) for path in numeric_paths}
    if before != after:
        raise AssertionError("Saved numerical inputs changed during rendering")
    write_json(BASE / "data/provenance/render_manifest.json", dict(
        comparison_figures=len(plots), validation_figures=2, supplemental_profile_figures=supplemental_count,
        supplemental_secondary_figures=secondary["figures"], render_seconds=perf_counter() - started,
        supplemental_clearance_figures=clearance["figures"],
        supplemental_trailing_figures=trailing["figures"],
        supplemental_safeguard_figures=safeguards["figures"],
        saved_values_and_derivatives_unchanged=True, input_data_sha256=before,
        model_calls=0, derivative_recalculations=0,
        script_sha256={name: sha256(BASE / "scripts" / name) for name in ("render_saved.py", "plotting.py", "reynolds_review.py", "profile_review.py", "profile_interpolation_review.py", "profile_terms.py", "shock_review.py", "profile_final_review.py", "secondary_review.py", "clearance_review.py", "trailing_review.py", "ko_safeguards_review.py")},
        imported_model_modules=imported_sources(),
        note="Existing imports can be present when invoked after run.py; the renderer calls no loss function.",
    ))
    from verify import verify
    verify(require_figures=True)
    print(f"Rendered {len(plots)+2+supplemental_count+secondary['figures']+clearance['figures']+trailing['figures']+safeguards['figures']} figures; saved numerical inputs unchanged.", flush=True)


if __name__ == "__main__":
    main()
