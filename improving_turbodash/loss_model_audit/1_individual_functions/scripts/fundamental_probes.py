"""Local shrinking probes of components/helpers; never call compute_losses."""

import numpy as np
from common import BASE, modules, evaluate, write_csv, write_json
from cases import manufactured, perturb
from fundamental import component_values, profile_values
from engine import bisection, local_join


def probe_definitions(versions):
    case = manufactured()
    probes = []

    def add(name, x, fun, field, regularity="C1"):
        probes.append((name, x, fun, field, regularity))

    for re in (2e5, 1e6):
        factor = versions["candidate"]["ko"].REYNOLDS_TRANSITION_FACTOR
        for location in (re / factor, re, re * factor):
            add(
                f"profile_Re_{location:g}",
                location,
                lambda x, m: profile_values(perturb(case, "Re_out", x), m),
                "Y_profile",
                "C2",
            )
    for mechanism in ("ko_secondary", "benner_secondary"):
        joins = (1.5, 2.0, 2.5) if mechanism == "ko_secondary" else (1.8, 2.0, 2.2)
        for location in joins:
            add(
                f"{mechanism}_AR{location}",
                location,
                lambda x, m, mechanism=mechanism: component_values(
                    mechanism, perturb(case, "aspect_ratio", x), m
                ),
                "Y_secondary",
                "C2",
            )
    for angle in (59.0, 60.0, 61.0, 62.0, 63.0, 64.0):
        for name in ("nozzle_blades", "impulse_blades"):
            add(
                f"{name}_{angle}",
                angle,
                lambda x, m, name=name: {"Y": getattr(m["ko"], name)(0.8, x)},
                "Y",
                "C2",
            )
    for location in (-0.1, 0.0, 5.0):
        add(
            f"incidence_chi_{location}",
            location,
            lambda x, m: {
                "raw": m["benner"].get_incidence_profile_loss_increment(
                    x, loss_limit=None
                )
            },
            "raw",
            "C1 retained" if location == 5 else "C2",
        )
    for location in (0.45, 0.55):
        add(
            f"penetration_cap_{location}",
            location,
            lambda x, m: {"Y": m["benner"].limit_penetration_depth(x)},
            "Y",
            "C2 retained",
        )
    for location in (0.15, 0.2, 0.25, 0.9, 1.0):
        add(
            f"compressibility_{location}",
            location,
            lambda x, m: {"K1": m["ko"].get_compressible_correction_factors(0.5, x)[2]},
            "K1",
            "C2",
        )
    for location in (.8, 1.):
        add(
            f"compressibility_ratio_{location}", location,
            lambda x, m: {"K2": m["ko"].get_compressible_correction_factors(x, 1.)[1]},
            "K2", "C2",
        )
    root = bisection(
        lambda x: profile_values(perturb(case, "pitch_chord", x), versions["candidate"])[
            "floor_margin"
        ],
        1.3,
        2.0,
    )
    if root is not None:
        add(
            "profile_floor",
            root,
            lambda x, m: profile_values(perturb(case, "pitch_chord", x), m),
            "Y_profile",
            "C2",
        )
    for location in (0.1, 0.2, 0.3, 0.4):
        add(
            f"trailing_{location}",
            location,
            lambda x, m: component_values(
                "ko_trailing", perturb(case, "trailing_ratio", x), m
            ),
            "Y_trailing",
        )
    for location in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        add(
            f"hub_{location}",
            location,
            lambda x, m: {"ratio": m["ko"].get_hub_to_mean_mach_ratio(x, 1)},
            "ratio",
        )
    # Unchanged physical-reference questions must remain visible in the result.
    add(
        "moustapha_incidence_reference",
        0.0,
        lambda x, m: component_values(
            "moustapha_incidence", perturb(case, "incidence", x), m
        ),
        "delta_phi2_raw",
        "deferred",
    )
    add(
        "moustapha_secondary_reference",
        0.0,
        lambda x, m: component_values(
            "moustapha_secondary", perturb(case, "incidence", x), m
        ),
        "secondary_factor",
        "C1 retained",
    )
    return probes


def run_probes(versions=None):
    versions = versions or {"candidate": modules("candidate")}
    records, findings, curvature = [], [], []
    for name, location, fun, field, claim in probe_definitions(versions):
        rows, result = local_join(name, fun, location, versions, field)
        records.extend(rows)
        findings.extend(dict(r, intended_regularity=claim) for r in result)
        # Cubic one-sided extrapolation, translated ordinates to reduce cancellation.
        # Seven h scales retained; smallest h never alone establishes a C2 claim.
        scale = max(1.0, abs(location))
        for version, mods in versions.items():
            for exponent in range(2, 8):
                h = scale * 10.0 ** (-exponent)
                sides = []
                for sign in (-1, 1):
                    u = sign * np.arange(1.0, 5.0)
                    y = np.array(
                        [
                            evaluate(lambda z=z: fun(np.float64(z), mods)).get(
                                field, np.nan
                            )
                            for z in location + h * u
                        ]
                    )
                    if np.all(np.isfinite(y)):
                        c = np.polynomial.polynomial.polyfit(u, y - y[0], 3)
                        sides.append((y[0] + c[0], c[1] / h, 2 * c[2] / h**2))
                    else:
                        sides.append((np.nan,) * 3)
                left, right = np.array(sides)
                curvature.append(
                    dict(
                        feature=name,
                        version=version,
                        x=location,
                        h=h,
                        value_difference=right[0] - left[0],
                        d1_left=left[1],
                        d1_right=right[1],
                        d2_left=left[2],
                        d2_right=right[2],
                        d2_difference=right[2] - left[2],
                        intended_regularity=claim,
                    )
                )
    write_csv(BASE / "data/provenance/fundamental_join_intervals.csv", records)
    write_json(BASE / "data/provenance/fundamental_join_findings.json", findings)
    write_csv(BASE / "data/provenance/fundamental_curvature_intervals.csv", curvature)
    return findings


if __name__ == "__main__":
    run_probes()
