"""Sensitivity of the shared blade kernel to independent shoulder fullness."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from parametrization_blades.blade_geometry import analysis as geometry
from parametrization_blades.blade_geometry.thickness import denton_body

VALUES = (-.5, -.25, 0., .25, .5)
COLORS = ("#256995", "#5998ac", "#424b51", "#c28e41", "#b95532")


def paired_plot(family, baseline, blades):
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=.08, hspace=.035)
    xlim, blade_ylim, thickness_ylim = geometry.common_limits(baseline, blades)
    for row, (blade, color) in enumerate(zip(blades, COLORS)):
        f_le, f_te = blade["fullness"], blade["trailing_fullness"]
        label = f"LE fullness {f_le:+.2f} | TE fullness {f_te:+.2f}"
        if f_le == f_te == 0:
            label += " | Original Denton"
        axes[row, 0].set_title(label, loc="left", fontsize=11)
        maximum = float(np.max(blade["plus_offset"]+blade["minus_offset"]))
        axes[row, 1].set_title(f"Final blade | maximum thickness {maximum:.3f} mm", loc="left", fontsize=11)
        for col, ax in enumerate(axes[row]):
            flattened = col == 0
            original = geometry.outline(baseline, flattened)
            modified = geometry.outline(blade, flattened)
            ax.fill(modified[:, 0], modified[:, 1], color=color, alpha=.13)
            ax.plot(original[:, 0], original[:, 1], color="#8d989f", ls="--", lw=1.3)
            ax.plot(modified[:, 0], modified[:, 1], color=color, lw=1.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(thickness_ylim if flattened else blade_ylim)
            ax.grid(alpha=.16)
        axes[row, 0].axhline(0, color=".55", lw=.7, ls=":")
        axes[row, 0].set(xlabel="Camber arclength [mm]", ylabel="Side offset [mm]")
        axes[row, 1].plot(blade["camber"][:, 0], blade["camber"][:, 1], color=".55", lw=.7, ls=":")
        axes[row, 1].set(xlabel="Meridional distance [mm]", ylabel="Circumferential distance [mm]")
    description = "Equal LE and TE fullness" if family == "impulse" else "Vary TE fullness; LE fullness fixed at zero"
    fig.suptitle(f"{family.capitalize()} blade | {description}\nUnrolled thickness on the left; final blade on the right", fontsize=13)
    fig.legend(handles=[Line2D([], [], color="#8d989f", ls="--", label="Original Denton"),
                        Line2D([], [], color=COLORS[0], label="Rational modification (row color)")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=10)
    fig.canvas.draw()
    scales = []
    for ax in axes.flat:
        p = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        dx, dy = p[1, 0]-p[0, 0], p[2, 1]-p[0, 1]
        assert abs(dx/dy-1) < 1e-10
        scales.append(float(dx))
    assert np.ptp(scales) < 1e-8
    for extension in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"{family}_sensitivity.{extension}", dpi=180)
    plt.close(fig)
    return dict(family=family, equal_aspect=True, common_pixels_per_mm=scales[0])


def main():
    (HERE/"figures").mkdir(exist_ok=True)
    (HERE/"results").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "svg.fonttype":"none", "path.simplify":False, "savefig.facecolor":"white"})
    report = dict(law="Shared blade kernel: rational Denton law with independent reference fullness inputs",
                  input="f_LE and f_TE, not the internal a coefficients", values=VALUES,
                  base_points=601, baselines={}, cases=[], figures=[])
    for family in ("impulse", "reaction"):
        baseline = geometry.baseline(family, count=601)
        report["baselines"][family] = dict(parameters=baseline["parameters"],
                                           metrics=geometry.audit(baseline, count=16001))
        blades = []
        for value in VALUES:
            leading = value if family == "impulse" else 0.
            blade = geometry.with_fullness(baseline, leading, value)
            result = geometry.audit(blade, count=16001)
            assert result["positive_thickness"] and result["minimum_normal_map_margin"] > 0
            assert result["thickness_slope_sign_changes"] == 1
            assert not result["self_intersection"] and not result["neighbor_intersection"]
            th = baseline["parameters"]["thickness"]
            stations = np.array([0., th["maximum_thickness_location"], 1.])
            expected = np.array([th["leading_edge_thickness_ratio"], th["maximum_thickness_ratio"], th["trailing_edge_thickness_ratio"]])
            actual = denton_body(stations, blade["parameters"]["thickness"])
            assert np.max(np.abs(actual-expected)) < 1e-14
            if leading == value == 0:
                error = max(float(np.max(np.abs(blade[side]-baseline[side]))) for side in ("plus", "minus"))
                assert error < 1e-12
                result["neutral_coordinate_error_mm"] = error
            if family == "impulse":
                xx = np.linspace(0, 1, 4001)
                error = np.max(np.abs(denton_body(xx, blade["parameters"]["thickness"])
                                     -denton_body(1-xx, blade["parameters"]["thickness"])))
                assert error < 1e-14
            report["cases"].append(result)
            blades.append(blade)
        report["figures"].append(paired_plot(family, baseline, blades))
        print(f"Completed {family}: five cases, neutral recovery and geometry checks passed.", flush=True)
    (HERE/"results/sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([dict(family=r["family"], f_LE=r["leading_fullness"], f_TE=r["trailing_fullness"],
                           maximum_thickness_mm=r["maximum_paired_thickness_mm"],
                           normal_map_margin=r["minimum_normal_map_margin"])
                      for r in report["cases"]], indent=2))


if __name__ == "__main__":
    main()
