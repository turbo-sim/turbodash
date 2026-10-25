"""Run the basic demo through its CLI and verify native PySolverView outputs."""

from copy import deepcopy
from pathlib import Path
import runpy
import sys

import numpy as np
import pysolver_view as psv
import pytest
import yaml

from turbodash import optimization
from turbodash.core import turbine


def test_basic_demo_uses_one_native_convergence_plot_and_complete_reports(
    tmp_path, monkeypatch, capsys
):
    """The exact script replaces retired plots with the frontend's own output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = {
        "inputs": {"design": {"x": 0.2}},
        "optimization": {
            "design_variables": {
                "design.x": {"lower_value": 0.0, "upper_value": 1.0},
            },
            "objective_function": {
                "variable": "overall_performance.efficiency_ts", "type": "maximize",
            },
            "constraints": [
                {"variable": "overall_performance.x", "type": "<=", "value": 0.8},
            ],
            "solver_options": {
                "print_convergence": False,
                "first_step_fraction": 1.0,
                "step_growth_factor": None,
                "ftol": 1e-10,
            },
        },
    }
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    retired = [output / f"physical_convergence_history.{extension}"
               for extension in ("png", "svg")]
    for path in retired:
        path.write_bytes(b"Obsolete physical convergence plot")
    unrelated = output / "keep_this_report.txt"
    unrelated.write_text("Unrelated existing output", encoding="utf-8")

    def evaluate(config):
        """Supply a cheap physical turbine model with an interior known optimum."""
        x = config["inputs"]["design"]["x"]
        return {
            "inputs": deepcopy(config["inputs"]),
            "overall_performance": {"efficiency_ts": 0.9 - (x - 0.6) ** 2, "x": x},
            "stages_performance": [{"geometry_meanline": {"stator": {}, "rotor": {}}}],
        }

    figures, physical_figures, native_calls = [], [], []
    make_figure = plt.figure
    native_plot = psv.OptimizationSolver.plot_convergence_history

    def record_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        if all(figure is not previous for previous in figures):
            figures.append(figure)
        return figure

    def plot_turbine(results, title=None, figure=None):
        """Substitute only geometry rendering; optimization plotting stays real."""
        assert figure is None
        figure, axis = plt.subplots(figsize=(2, 1))
        axis.set_title(title)
        axis.plot([results["overall_performance"]["efficiency_ts"]])
        physical_figures.append(figure)
        return figure

    def record_native_plot(optimizer, *args, **kwargs):
        figure = native_plot(optimizer, *args, **kwargs)
        native_calls.append((optimizer, figure, kwargs))
        return figure

    script = Path(__file__).resolve().parents[1] / "demos/optimization/optimize_turbine.py"
    monkeypatch.setattr(turbine, "compute_turbine_performance", evaluate)
    monkeypatch.setattr(optimization, "plot_turbine", plot_turbine)
    monkeypatch.setattr(plt, "figure", record_figure)
    monkeypatch.setattr(psv.OptimizationSolver, "plot_convergence_history", record_native_plot)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", [str(script), "--case", str(case_path), "--output", str(output)])
    try:
        with pytest.raises(SystemExit) as exited:
            runpy.run_path(str(script), run_name="__main__")
        assert exited.value.code == 0

        assert len(physical_figures) == 2
        assert len(native_calls) == 1
        assert len(figures) == 3
        optimizer, convergence, arguments = native_calls[0]
        assert isinstance(optimizer, psv.OptimizationSolver)
        assert all(convergence is not figure for figure in physical_figures)
        assert arguments["savefile"] and not arguments["showfig"]
        assert arguments["filename"] == "convergence_history"
        np.testing.assert_allclose(
            convergence.axes[0].lines[0].get_ydata(),
            optimizer.convergence_history["objective_value"],
        )
        assert np.all(np.asarray(convergence.axes[0].lines[0].get_ydata()) < 0)

        for basename in ("initial_turbine", "optimized_turbine", "convergence_history"):
            for extension in ("png", "svg"):
                assert (output / f"{basename}.{extension}").stat().st_size > 0
        assert not any(path.exists() for path in retired)
        assert unrelated.read_text(encoding="utf-8") == "Unrelated existing output"
        for name in ("input_case.yaml", "initial_case.yaml", "optimized_case.yaml"):
            assert yaml.safe_load((output / name).read_text(encoding="utf-8"))
        saved = yaml.safe_load((output / "optimization_result.yaml").read_text(encoding="utf-8"))
        assert saved["success"]
        assert saved["provenance"]["solver"] == "pysolver_view.OptimizationSolver"
        assert saved["design_variables"]["design.x"] == pytest.approx(0.6, abs=1e-6)

        history = (output / "convergence_history.txt").read_text(encoding="utf-8")
        assert "Starting optimization process" in history and "Success: True" in history
        report = (output / "optimization_report.txt").read_text(encoding="utf-8")
        for title in (
            "Optimization variables report (physical values)",
            "Optimization variables report (normalized values)",
            "Optimization constraints report",
            "Karush-Kuhn-Tucker (KKT) conditions",
            "Lagrange multipliers report",
        ):
            assert title in report
        assert report in capsys.readouterr().out
    finally:
        for figure in figures:
            plt.close(figure)
