"""Check local navigation, paired grids and exact parsed-number reproduction."""

import csv
import gzip
import json
from pathlib import Path
import re
from urllib.parse import unquote

from common import BASE, numeric_digest, write_json
import numpy as np


def verify(require_figures=True):
    from profile_terms import verify_saved
    verify_saved()
    from profile_interpolation_review import verify_saved as verify_interpolation
    verify_interpolation()
    from profile_final_review import verify_saved as verify_final
    verify_final()
    from secondary_review import verify_saved as verify_secondary
    verify_secondary()
    from clearance_review import verify_saved as verify_clearance
    verify_clearance()
    from trailing_review import verify_saved as verify_trailing
    verify_trailing()
    references = json.loads((BASE / "data/reference_digests.json").read_text(encoding="utf-8"))
    differences = []
    for relative, expected in references["files"].items():
        actual = numeric_digest(BASE / relative)
        if actual != expected:
            differences.append(relative)
    sweeps = json.loads((BASE / "data/sweep_manifest.json").read_text(encoding="utf-8"))
    assert len(sweeps) == 78
    for study in sweeps:
        with gzip.open(BASE / study["data_path"], "rt", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        versions = {version: [row for row in rows if row["version"] == version] for version in ("baseline", "candidate")}
        np.testing.assert_array_equal([float(row["x"]) for row in versions["baseline"]], [float(row["x"]) for row in versions["candidate"]])
        assert study["base_points"] == 200
        for version in versions.values():
            assert sum(row["base_grid"].lower() == "true" for row in version) == 200
    supplemental = []
    for filename, grouping, coordinate in (
        ("data/reynolds_review/samples.csv", (), "Re"),
        ("data/profile_review/pitch_families.csv", ("family", "angle_deg"), "pitch_chord"),
        ("data/profile_review/angle_transitions.csv", ("family", "pitch_chord"), "angle_deg"),
    ):
        with (BASE / filename).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups = {tuple(row[k] for k in grouping) for row in rows}
        for group in groups:
            selected = [r for r in rows if tuple(r[k] for k in grouping) == group]
            grids = {v: [float(r[coordinate]) for r in selected if r["version"] == v]
                     for v in ("baseline", "candidate")}
            np.testing.assert_array_equal(grids["baseline"], grids["candidate"])
            assert len(grids["baseline"]) >= 200
            for version in ("baseline", "candidate"):
                assert sum(r["base_grid"].lower() in ("true", "1") for r in selected if r["version"] == version) == 200
        supplemental.append(dict(file=filename, paired_sweeps=len(groups), rows=len(rows)))
    broken = []
    checked = 0
    for document in BASE.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or "://" in target:
                continue
            resolved = (document.parent / target).resolve()
            if resolved == (BASE / "data/validation/reproduction.json").resolve():
                # This very invocation creates the verification result below.
                checked += 1
                continue
            if not require_figures and resolved == (BASE / "data/provenance/render_manifest.json").resolve():
                # A calculation-only run may precede the first saved-data redraw.
                checked += 1
                continue
            if not resolved.is_relative_to(BASE.parents[2].resolve()):
                broken.append(f"Outside repository: {document.name}: {target}")
            elif require_figures or resolved.suffix != ".png":
                if not resolved.exists():
                    broken.append(f"Missing: {document.name}: {target}")
            checked += 1
    result = dict(reference_files=len(references["files"]), numeric_reference_differences=differences, paired_sweeps=len(sweeps), supplemental=supplemental, base_points=200, checked_markdown_links=checked, broken_links=broken, figures_checked=require_figures, note="Digests use parsed float.hex values and text/status fields, independent of compressed-file timestamps. References include the reviewed wider K1/K2 blends and unchanged rational Kp continuation. Preceding results remain in data/history/before_wider_compressibility_blends.zip.")
    write_json(BASE / "data/validation/reproduction.json", result)
    assert not differences, f"Numerical results differ from the frozen report reference: {differences}"
    assert not broken, "\n".join(broken)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
