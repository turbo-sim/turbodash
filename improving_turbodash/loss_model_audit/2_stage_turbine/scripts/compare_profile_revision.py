"""Compare recalculated turbines with the preceding saved audit."""
from collections import Counter
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
ARCHIVE = HERE / "data/history/before_reaction_range_extension.zip"


def compare():
    saved = json.loads(ARCHIVE.with_suffix(".json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(ARCHIVE) == saved["archive_sha256"]
    protected = {n: h for n, h in saved["production_sources"].items()
                 if n.startswith("turbodash/loss_models/") or n == "turbodash/math.py"}
    assert all(digest(REPO / n) == h for n, h in protected.items())
    totals = {v: dict(rows_compared=0, changed_rows=0, changed_fields=Counter(),
                     max_abs_numeric_differences={}) for v in ("previous", "revised")}
    studies = []
    design_changes = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        old_designs = json.loads(archive.read("data/reference_results.json"))
        new_designs = json.loads((HERE / "data/reference_results.json").read_text(encoding="utf-8"))
        for case in ("0", "80"):
            before = old_designs[case]["revised"]["overall_performance"]
            after = new_designs[case]["revised"]["overall_performance"]
            design_changes.append(dict(case=case, model="benner_new",
                                       eta_ts_before=before["efficiency_ts"], eta_ts_after=after["efficiency_ts"],
                                       eta_ts_change_pp=100*(after["efficiency_ts"]-before["efficiency_ts"]),
                                       eta_tt_change_pp=100*(after["efficiency_tt"]-before["efficiency_tt"])))
        for name in sorted(n for n in archive.namelist() if n.startswith("data/eps") and n.endswith(".csv")):
            old = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))
            with (HERE / name).open(newline="", encoding="utf-8") as stream:
                new = list(csv.DictReader(stream))
            by_coordinate = {(r["version"], r["x"]): r for r in new}
            old_coordinates = {(r["version"], r["x"]) for r in old}
            assert len(by_coordinate) == len(new), (name, "duplicate coordinates")
            assert old_coordinates <= by_coordinate.keys(), (name, "a preserved coordinate was removed")
            added_rows = len(new)-len(old)
            assert added_rows == 0 or name.endswith("_reaction.csv"), (name, "unexpected grid extension")
            changes = Counter()
            for before in old:
                after = by_coordinate[before["version"], before["x"]]
                assert before.keys() == after.keys()
                assert all(before[k] == str(after[k]) for k in ("case", "study", "variable", "version", "x", "base_grid"))
                version = before["version"]
                tally = totals[version]
                tally["rows_compared"] += 1
                fields = [k for k in before if k != "api_seconds" and before[k] != after[k]]
                if "review_flags" in fields:
                    # Efficiency changes may cross the plotting/review range.
                    # Preserve every independent physical-domain flag, and
                    # reconstruct the one efficiency flag from each result.
                    efficiency_flag = "efficiency outside [0,1]"
                    flag_sets = [set(r["review_flags"].split(" | ")) for r in (before, after)]
                    assert flag_sets[0] - {efficiency_flag} == flag_sets[1] - {efficiency_flag}
                    for result, flags in zip((before, after), flag_sets):
                        outside = not (0 <= float(result["efficiency_ts"]) <= 1 and
                                       0 <= float(result["efficiency_tt"]) <= 1)
                        assert (efficiency_flag in flags) == outside
                if fields:
                    tally["changed_rows"] += 1
                    changes[version] += 1
                for key in fields:
                    tally["changed_fields"][key] += 1
                    try:
                        delta = abs(float(after[key]) - float(before[key]))
                    except ValueError:
                        continue
                    if delta > tally["max_abs_numeric_differences"].get(key, 0.):
                        tally["max_abs_numeric_differences"][key] = delta
            studies.append(dict(csv=name, preserved_rows=len(old), added_rows=added_rows, changed_rows=dict(changes)))
    assert totals["previous"]["changed_rows"] == 0, "Protected previous-model outputs changed"
    assert totals["revised"]["changed_rows"] == 0, "An existing candidate coordinate changed during an audit-only range extension"
    allowed_changes = {
        "efficiency_ts", "efficiency_tt", "power_actual", "stage_efficiency_ts", "stage_efficiency_tt",
        "driver_row_loss_fraction", "energy_balance_error", "review_flags",
        *(f"{row}_{field}" for row in ("stator", "rotor") for field in (
            "Y_total", "Y_profile", "driver_loss_fraction_total", "driver_loss_fraction_profile", "component_sum_error")),
    }
    assert set(totals["revised"]["changed_fields"]) <= allowed_changes, "An unrelated flow/geometry/component diagnostic changed"
    result = dict(comparison="Reaction range extended to -0.1..0.9 and refined near feasibility/acoustic boundaries; every preceding baseline and candidate coordinate preserved",
                  archive=ARCHIVE.relative_to(HERE).as_posix(), archive_sha256=digest(ARCHIVE),
                  protected_source_files=len(protected), previous_model_unchanged=True,
                  selected_models=dict(previous="benner", revised="benner_new"),
                  csv_files_compared=len(studies), excluded_fields=["api_seconds"], versions=totals, studies=studies,
                  flow_geometry_statuses_and_other_components_unchanged=True,
                  changed_review_flags_reconstructed_from_efficiencies=True,
                  all_preceding_coordinates_retained=True, added_rows=sum(r["added_rows"] for r in studies),
                  saved_candidate_design_changes=design_changes)
    (HERE / "data/profile_revision_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    parity_path = HERE / "data/v2_parity.json"
    parity = json.loads(parity_path.read_text())
    parity.update(comparison="Baseline retains historical v2 parity; current Benner candidate uses analytical AM reference fits",
                  current_candidate_change_details="profile_revision_comparison.json",
                  candidate_historical_parity_is_not_current_acceptance=True)
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = compare()
    print(json.dumps({k: v for k, v in result.items() if k != "studies"}, indent=2))
