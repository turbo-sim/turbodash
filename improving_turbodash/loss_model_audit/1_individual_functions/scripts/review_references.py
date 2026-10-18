"""Accept the reviewed wider K1/K2 blends after validation."""
import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from common import BASE, REPO, numeric_digest, sha256, write_json
from compare_profile_revision import compare

EXPECTED_CHANGED = {
    "data/provenance/fundamental_curvature_intervals.csv",
    "data/provenance/fundamental_join_intervals.csv",
    "data/sweeps/fundamental_compressibility_mach.csv.gz",
    "data/sweeps/fundamental_hub_interpolation.csv.gz",
    "data/sweeps/fundamental_ko_profile_hub_tip__stator.csv.gz",
    "data/sweeps/fundamental_ko_profile_ma_rel_in__stator.csv.gz",
    "data/sweeps/fundamental_ko_profile_ma_rel_out__stator.csv.gz",
    "data/sweeps/fundamental_ko_secondary_ma_rel_out__stator.csv.gz",
}
RETIRED = set()
ADDED = set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args()
    review = compare()
    assert set(review["changed_reference_tables"]) == EXPECTED_CHANGED, review["changed_reference_tables"]
    integration = json.loads((BASE / "data/validation/analytical_integration.json").read_text(encoding="utf-8"))
    interpolation = json.loads((BASE / "data/validation/profile_interpolation.json").read_text(encoding="utf-8"))
    assert integration["passed"] and interpolation["passed"]
    assert all(sha256(REPO / n) == digest for n, digest in integration["source_hashes"].items())
    from profile_terms import verify_saved
    terms = verify_saved()
    assert terms["compressibility_range"] == [0, 2.0]
    assert len(terms["compressibility_checks"]) == 8
    assert len(terms["raw_compressibility_checks"]) == 6
    assert terms["raw_compressibility_kink"] == .2 and terms["raw_compressibility_zero_mach_limit"] == 1.
    assert any(c["negative_samples"] > 0 for c in terms["raw_compressibility_checks"])
    assert sum(c["undefined_derivative_samples"] for c in terms["raw_compressibility_checks"]) == 5
    assert terms["k1_low_transition"] == [.15, .25]
    assert terms["k1_tail_start"] == .9 and terms["k2_saturation_start"] == .8
    assert terms["kp_transition_start"] == .6 and terms["kp_asymptote"] == .4
    assert terms["kp_physical_minimum"] == .4+.2/13 and terms["kp_minimum_slope"] == 7/169
    evidence = {}
    for name in ("analytical_integration_tests.xml", "repository_model_tests.xml"):
        path = BASE / "data/validation" / name
        suites = ET.parse(path).getroot().findall("testsuite")
        assert suites and all(int(s.get("errors", "0")) == int(s.get("failures", "0")) == 0 for s in suites)
        evidence[name] = dict(tests=sum(int(s.get("tests", "0")) for s in suites), sha256=sha256(path))
    with zipfile.ZipFile(BASE / review["archive"]) as archive:
        prior = json.loads(archive.read("data/reference_digests.json"))
    files = {n: numeric_digest(BASE / n) for n in (set(prior["files"])-RETIRED)|ADDED}
    record = dict(accepted=args.accept, revision="Widen K1 low join to 0.15-0.25, high tail to 0.9 and K2 saturation to 0.8",
                  archive=review["archive"], archive_sha256=review["archive_sha256"],
                  protected_sources=review["protected_source_files"],
                  baseline_values_and_statuses_unchanged=True,
                  changed_tables=[dict(path=n, before=prior["files"][n], after=files[n]) for n in sorted(EXPECTED_CHANGED)],
                  existing_shock_refresh_tables=["data/sweeps/fundamental_hub_interpolation.csv.gz",
                                                "data/sweeps/fundamental_ko_profile_hub_tip__stator.csv.gz"],
                  retired_tables=sorted(RETIRED), added_tables=sorted(ADDED),
                  tests=evidence, physical_validation=False,
                  notes="Only the new KO module changes. AST comparison with the archived source identifies get_compressible_correction_factors as the only changed function. Six reference-table changes concern compressibility and its join probes; two hub-related tables refresh older data against the shock helpers already present in the archived source. Tests verify the wider K1/K2 joins, endpoint values/slopes/curvatures, monotonicity and absence of slope overshoot in the K1 low join, plus the unchanged rational Kp continuation. All baseline sweep values and interpolation/thickness reference tables are unchanged. Focused compressibility and continuation tables are checked by current-source hashes and independent real differences. The raw comparison includes unsmoothed K1 with linear high-Mach extrapolation, unbounded K2 and no Kp limiting; it is diagnostic, not additional calibration.")
    write_json(BASE / "data/validation/reference_review.json", record)
    if args.accept:
        write_json(BASE / "data/reference_digests.json", dict(
            description="Reviewed wider K1/K2 blends, with unchanged rational Kp continuation. Ordinary runs never refresh references.",
            review="data/validation/reference_review.json", files=dict(sorted(files.items()))))
    print(json.dumps(dict(accepted=args.accept, changed=len(EXPECTED_CHANGED), added=len(ADDED),
                         retired=len(RETIRED), reference_tables=len(files)), indent=2))


if __name__ == "__main__":
    main()
