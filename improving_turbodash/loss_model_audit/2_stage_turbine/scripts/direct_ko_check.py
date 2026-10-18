"""Focused full-API KO check, kept separate from the main Benner comparison."""
from collections import Counter
from copy import deepcopy
import numpy as np
import yaml
from run_sweeps import HERE, configuration, evaluate, extract, csv_write, json_write


def calculate():
    rows = []
    for case in ("0", "80"):
        base = yaml.safe_load((HERE / "inputs" / f"R245fa_turbine_{case}_final.yaml").read_text())
        original = deepcopy(base)
        for variable, lo, hi in (("blade_velocity_ratio", .5, .9), ("stator_exit_angle", 40., 80.), ("solidity_multiplier", .6, 1.6)):
            for version, model in (("previous", "kacker_okapuu"), ("revised", "kacker_okapuu_new")):
                for x in np.linspace(lo, hi, 11):
                    cfg = configuration(base, variable, x, None, version)
                    cfg["inputs"]["loss_model"] = model
                    snapshot = deepcopy(cfg)
                    metrics, _ = evaluate(cfg)
                    assert cfg == snapshot
                    if metrics["status"] == "ok":
                        assert abs(metrics["energy_balance_error"]) < 1e-9
                        for row in ("stator", "rotor"):
                            assert abs(metrics[row + "_component_sum_error"]) < 1e-10
                    rows.append(dict(case=case, variable=variable, x=x, version=version, model=model, **metrics))
        assert base == original
    csv_write(HERE / "data/direct_ko_samples.csv", rows)
    differences = []
    for case in ("0", "80"):
        for variable in sorted({r["variable"] for r in rows}):
            groups = {v: [r for r in rows if r["case"] == case and r["variable"] == variable and r["version"] == v]
                      for v in ("previous", "revised")}
            pairs = [(a, b) for a, b in zip(groups["previous"], groups["revised"]) if a["status"] == b["status"] == "ok"]
            assert pairs, (case, variable, "No finite direct-KO pairs; this is not a successful check")
            assert [r["x"] for r in groups["previous"]] == [r["x"] for r in groups["revised"]]
            differences.append(dict(case=case, variable=variable, finite_pairs=len(pairs),
                                    max_abs_eta_ts_difference=max((abs(b["efficiency_ts"] - a["efficiency_ts"]) for a, b in pairs), default=None)))
    result = dict(selected_models=["kacker_okapuu", "kacker_okapuu_new"], full_api_evaluations=len(rows),
                  status_counts={v: dict(Counter(r["status"] for r in rows if r["version"] == v)) for v in ("previous", "revised")},
                  paired_sensitivities=differences, input_immutability=True, finite_energy_and_component_sums_checked=True,
                  finite_evaluations_checked=sum(r["status"] == "ok" for r in rows),
                  interpretation="Focused direct-KO software sensitivity check; the main 26 sweeps remain Benner. Core normalization and two-phase/supersonic limitations apply.")
    json_write(HERE / "data/direct_ko_checks.json", result)
    return result


if __name__ == "__main__":
    print(calculate())
