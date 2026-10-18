"""Recompute the 78 individual-function comparisons and local probes."""

from time import perf_counter

START = perf_counter()
import argparse
import platform
import sys

from common import BASE, REPO, imported_sources, modules, sha256, write_csv, write_json
from engine import run_study
from fundamental import fundamental_studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Recompute numerical evidence; use render_saved.py to redraw later.")
    args = parser.parse_args()
    versions = {"candidate": modules("candidate")}
    summaries, comparisons, plots = [], [], []
    groups = set()
    import_seconds = perf_counter() - START
    started = perf_counter()
    for number, study in enumerate(fundamental_studies(), 1):
        _, summary, differences = run_study(study, versions, BASE / "data/sweeps", points=200)
        summaries.append(summary)
        comparisons.extend(differences)
        if study["group"] not in groups and study["group"] not in ("ko_profile_Re_out", "nozzle_angle"):
            groups.add(study["group"])
            plots.append(dict(group=study["group"], mechanism=study["mechanism"], path=f"figures/{study['group'].lower()}.png"))
        if number % 20 == 0 or number == 78:
            print(f"Evaluated {number}/78 individual-function sweeps", flush=True)
    calculation_seconds = perf_counter() - started
    write_json(BASE / "data/sweep_manifest.json", summaries)
    write_json(BASE / "data/plot_manifest.json", plots)
    from fundamental_probes import run_probes
    from validate_derivatives import main as validate_derivatives
    from review_nozzle_patch import main as review_nozzle
    from reynolds_review import calculate as calculate_reynolds
    from profile_review import calculate as calculate_profiles

    started = perf_counter()
    run_probes(versions)
    validate_derivatives(render=False)
    review_nozzle()
    calculate_reynolds()
    calculate_profiles()
    from analytical_profile_review import calculate as calculate_integration
    calculate_integration()
    from profile_interpolation_review import calculate as calculate_interpolation
    calculate_interpolation()
    from profile_terms import calculate as calculate_terms
    calculate_terms()
    from profile_final_review import calculate as calculate_final
    calculate_final()
    from secondary_review import calculate as calculate_secondary
    calculate_secondary()
    from clearance_review import calculate as calculate_clearance
    calculate_clearance()
    from trailing_review import calculate as calculate_trailing
    calculate_trailing()
    from ko_safeguards_review import calculate as calculate_safeguards
    calculate_safeguards()
    probe_seconds = perf_counter() - started
    write_json(BASE / "data/provenance/run_summary.json", dict(
        level="fundamental", initial_points=200, studies=len(summaries), comparison_figures=len(plots),
        status_counts={version: {status: sum(row["status_counts"][version][status] for row in summaries) for status in ("ok", "nonfinite", "exception")} for version in versions},
        implementation="current production source",
    ))
    write_json(BASE / "data/provenance/source_manifest.json", dict(
        note="Paths relative to the repository; actual imported paths also recorded. Model functions are imported directly from repository turbodash, without audit source copies.",
        files={path.relative_to(REPO).as_posix(): sha256(path) for path in [*sorted((REPO / "turbodash/loss_models_new").glob("*.py")), REPO / "turbodash/math.py", *sorted((BASE / "scripts").glob("*.py")), *sorted((BASE / "tests").glob("*.py"))]},
        imported_modules=imported_sources(),
    ))
    write_json(BASE / "data/provenance/timings.json", dict(
        import_seconds=import_seconds, calculation_and_csv_seconds=calculation_seconds, probe_and_validation_seconds=probe_seconds,
        function_seconds_by_version={version: sum(row["function_seconds_by_version"][version] for row in summaries) for version in versions},
        calculation_wall_seconds=perf_counter() - START, executable=sys.executable, python=platform.python_version(),
        note="200 base points plus shared refinement; no complete row/stage calls. Plotting is measured separately by render_saved.py.",
    ))
    from verify import verify
    result = verify(require_figures=False)
    print(f"Verified current-source data: {result['sweeps']} sweeps", flush=True)
    if not args.no_plots:
        from render_saved import main as render
        render()
    print(f"Results: {BASE}", flush=True)


if __name__ == "__main__":
    main()
