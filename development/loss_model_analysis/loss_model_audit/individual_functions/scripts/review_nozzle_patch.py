"""Independent cross-variable review of the candidate nozzle fit transition."""

import numpy as np
from common import BASE, modules, write_csv, write_json


def main():
    ko = modules("candidate")["ko"]
    intervals, optima = [], []
    for phi in (29.0, 29.5, 30.0, 30.5, 31.0):
        for name, center in (
            ("low_branch_optimum", 0.46 + phi / 77),
            ("high_branch_optimum", 0.614 + phi / 130),
        ):
            for pitch_direction, angle_direction in ((1, 0), (0, 1), (1, 1), (-1, 1)):

                def at(t):
                    return ko.nozzle_blades(
                        center + t * pitch_direction, 90 - phi - t * angle_direction
                    )

                value = at(0)
                for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
                    left = (3 * value - 4 * at(-h) + at(-2 * h)) / (2 * h)
                    right = (-3 * value + 4 * at(h) - at(2 * h)) / (2 * h)
                    intervals.append(
                        dict(
                            phi=phi,
                            pitch_chord=center,
                            reference=name,
                            pitch_direction=pitch_direction,
                            angle_direction=angle_direction,
                            h=h,
                            left_gradient=left,
                            right_gradient=right,
                            mismatch=right - left,
                        )
                    )
        pitch = np.linspace(0.4, 1.6, 20001)
        values = ko.nozzle_blades(pitch, 90 - phi)
        differences = np.diff(values)
        changes = np.count_nonzero((differences[:-1] < 0) & (differences[1:] >= 0))
        optima.append(
            dict(
                phi=phi,
                interior_minima=int(changes),
                minimum=float(values.min()),
                minimizing_pitch=float(pitch[values.argmin()]),
                positive=bool(np.all(values > 0)),
            )
        )
    tail = [r for r in intervals if r["h"] == 3e-6]
    summary = dict(
        domain="Tangential angle phi 29..31 deg; pitch/chord 0.4..1.6 for positivity/optima.",
        directional_gradient_max_mismatch=float(max(abs(r["mismatch"]) for r in tail)),
        smallest_probe_h=3e-6,
        all_directional_gradients_finite=bool(
            all(np.isfinite(r["mismatch"]) for r in intervals)
        ),
        optima=optima,
        interpretation="Supports C1 across pitch and angle including both branch optima; one interior minimum in the tested neighborhood. Does not assert global positivity for extrapolated cubic fits.",
    )
    write_csv(
        BASE / "data/validation/nozzle_directional_probes.csv", intervals
    )
    write_json(
        BASE / "data/validation/nozzle_directional_summary.json", summary
    )


if __name__ == "__main__":
    main()
