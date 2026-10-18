"""Compare current stage CSVs with the preserved pre-Reynolds-revision audit."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parents[1]
ARCHIVE = HERE / "data" / "history" / "before_wider_reynolds.zip"
IGNORED_FIELDS = {"api_seconds"}
PERFORMANCE_FIELDS = (
    "efficiency_ts", "efficiency_tt", "driver_row_loss_fraction",
    "stator_Y_total", "rotor_Y_total", "stator_Y_profile", "rotor_Y_profile",
)


def read_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def compare():
    manifest = json.loads(ARCHIVE.with_suffix(".json").read_text(encoding="utf-8"))
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == manifest["archive_sha256"], "Historical archive hash changed"
    totals = {version: {"rows_compared": 0, "changed_rows": 0, "changed_fields": Counter(), "max_abs_numeric_differences": {}} for version in ("previous", "revised")}
    studies = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        historical_parity = json.loads(archive.read("data/v2_parity.json"))
        names = sorted(name for name in archive.namelist() if name.startswith("data/eps") and name.endswith(".csv"))
        for name in names:
            historical = read_rows(archive.read(name))
            current = read_rows((HERE / name).read_bytes())
            assert len(historical) == len(current), (name, "grid length changed")
            stats = {version: {"points": 0, "changed_rows": 0, "historical_status_counts": Counter(), "current_status_counts": Counter(), "max_abs_performance_differences": {key: 0.0 for key in PERFORMANCE_FIELDS}} for version in totals}
            for old, new in zip(historical, current):
                assert all(old[key] == new[key] for key in ("case", "study", "variable", "version", "x", "base_grid")), (name, "coordinate/order changed")
                assert old.keys() == new.keys(), (name, "CSV schema changed")
                version = old["version"]
                tally, local = totals[version], stats[version]
                tally["rows_compared"] += 1
                local["points"] += 1
                local["historical_status_counts"][old["status"]] += 1
                local["current_status_counts"][new["status"]] += 1
                changes = []
                for key in old:
                    if key in IGNORED_FIELDS or old[key] == new[key]:
                        continue
                    changes.append(key)
                    tally["changed_fields"][key] += 1
                    try:
                        delta = abs(float(new[key]) - float(old[key]))
                    except ValueError:
                        continue
                    if delta > tally["max_abs_numeric_differences"].get(key, 0.0):
                        tally["max_abs_numeric_differences"][key] = delta
                    if key in PERFORMANCE_FIELDS:
                        local["max_abs_performance_differences"][key] = max(local["max_abs_performance_differences"][key], delta)
                if changes:
                    tally["changed_rows"] += 1
                    local["changed_rows"] += 1
            studies.append({"csv": Path(name).name, "case": historical[0]["case"], "study": historical[0]["study"], "versions": stats})
    assert totals["previous"]["changed_rows"] == 0, "The preserved previous-model results changed"
    data = {
        "comparison": "Current stage results versus the preserved v3 results before widening the revised Reynolds transition",
        "historical_archive": ARCHIVE.relative_to(HERE).as_posix(),
        "archive_sha256": manifest["archive_sha256"],
        "historical_file_hashes_verified_at_preservation": manifest["all_archived_file_hashes_verified"],
        "excluded_fields": sorted(IGNORED_FIELDS),
        "csv_files_compared": len(names),
        "previous_model_unchanged": True,
        "revised_model_changes_are_measured_not_required_to_be_zero": True,
        "versions": totals,
        "studies": studies,
    }
    (HERE / "data/reynolds_revision_comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    original_names = set(historical_parity["csv_names"])
    original = [study for study in studies if study["csv"] in original_names]
    parity = {
        "comparison": "Current previous-model results retain v2 parity; current revised-model results intentionally follow the widened Reynolds transition",
        "historical_exact_parity_evidence": "data/history/before_wider_reynolds.zip contains the original data/v2_parity.json, source hashes and CSVs",
        "historical_original_sweeps_exact_to_v2": historical_parity["original_numerical_fields_statuses_errors_warnings_exact"],
        "original_csv_files": len(original),
        "previous_model_rows_compared": sum(study["versions"]["previous"]["points"] for study in original),
        "previous_model_exact_to_preserved_v3_and_hence_v2": all(study["versions"]["previous"]["changed_rows"] == 0 for study in original),
        "revised_model_changed_rows_in_original_sweeps": sum(study["versions"]["revised"]["changed_rows"] for study in original),
        "current_candidate_change_details": "reynolds_revision_comparison.json",
        "current_total_csv_files": len(studies),
        "current_total_plots": len(list((HERE / "plots").glob("*.png"))),
        "no_current_candidate_exact_parity_assumed": True,
    }
    (HERE / "data/v2_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    result = compare()
    print(json.dumps({"csv_files": result["csv_files_compared"], "versions": result["versions"]}, indent=2))
