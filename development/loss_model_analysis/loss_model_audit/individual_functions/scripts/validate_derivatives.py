"""Render and record known-function checks without running any loss model."""

from common import BASE, write_csv, write_json
from engine import derivative, local_join, merge_grid
from plotting import plot_group
import numpy as np


def main(render=True):
    initial = np.linspace(-2, 2, 200)
    coordinates = set(initial) | {0.0, 1.0}
    for anchor in (0.0, 1.0):
        for exponent in range(2, 16):
            coordinates.update((anchor - 10.0**-exponent, anchor + 10.0**-exponent))
    x = merge_grid(coordinates, initial)
    values = dict(
        linear=2.0 + 3.0 * x,
        tiny_linear_gradient=2.0 + 1e-6 * x,
        genuine_kink=np.where(x < 0, -2.0 * x, 3.0 * x),
        genuine_jump=(x >= 0).astype(float),
        explicit_gap=np.where(
            x < -0.25, 2 * x, np.where(x > 0.25, 100 + 7 * x, np.nan)
        ),
    )
    derivatives = {"d_" + key + "_dx": derivative(x, y) for key, y in values.items()}
    study = dict(lo=-2, hi=2, xlabel="Known-function coordinate x", kind="helper")
    arrays = {"validation": dict(x=x, values=values, derivatives=derivatives)}
    curves = [("known-function audit stencil", study, arrays)]
    plot_dir = BASE / "figures"
    if render:
        plot_group(
            "near_constant_validation",
            curves,
            ["linear", "tiny_linear_gradient"],
            directory=plot_dir,
        )
        plot_group(
            "features_and_gaps_validation",
            curves,
            ["genuine_kink", "genuine_jump", "explicit_gap"],
            directory=plot_dir,
        )
    write_csv(
        BASE / "data/validation/samples.csv",
        [
            dict(
                x=z,
                **{key: y[i] for key, y in values.items()},
                **{key: y[i] for key, y in derivatives.items()}
            )
            for i, z in enumerate(x)
        ],
    )
    functions = {
        "linear": lambda z: 2.0 + 3.0 * z,
        "genuine_kink": lambda z: np.where(z < 0, -2.0 * z, 3.0 * z),
        "genuine_jump": lambda z: np.float64(z >= 0),
    }
    intervals, findings = [], []
    for name, function in functions.items():
        rows, results = local_join(
            name,
            lambda z, mods: dict(value=function(z)),
            0.0,
            {"known_function": None},
            "value",
        )
        intervals.extend(rows)
        findings.extend(results)
    write_csv(BASE / "data/validation/shrinking_probes.csv", intervals)
    write_json(
        BASE / "data/validation/summary.json",
        dict(
            value_samples=len(x),
            base_samples=len(initial),
            spacing_ratio=float(np.max(np.diff(x)) / np.min(np.diff(x))),
            retained_derivative_samples=int(
                np.isfinite(derivatives["d_linear_dx"]).sum()
            ),
            maximum_linear_gradient_error=float(
                np.nanmax(np.abs(derivatives["d_linear_dx"] - 3.0))
            ),
            maximum_tiny_linear_gradient_error=float(
                np.nanmax(np.abs(derivatives["d_tiny_linear_gradient_dx"] - 1e-6))
            ),
            shrinking_probe_findings=findings,
        ),
    )


if __name__ == "__main__":
    main()
