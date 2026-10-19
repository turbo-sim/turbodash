"""200-point initial grids, shared refinement, gap-aware derivatives and joins."""

import numpy as np
from time import perf_counter
from common import BASE, evaluate, write_csv
from differentiation import POLICY, derivative, finite_runs


def merge_grid(refined, initial):
    """Retain every distinct value coordinate, including near-ULP samples.

    Only exact coordinate duplicates are coalesced. Derivative stencil
    selection is separate and cannot discard a function evaluation.
    """
    return np.asarray(sorted(set(refined) | set(initial)))


def bisection(fun, a, b, iterations=48):
    fa, fb = fun(a), fun(b)
    if not np.isfinite(fa * fb) or fa * fb >= 0:
        return None
    for _ in range(iterations):
        mid = (a + b) / 2
        fm = fun(mid)
        if not np.isfinite(fm):
            return None
        if fa * fm <= 0:
            b = mid
        else:
            a = mid
            fa = fm
    return (a + b) / 2


def run_study(study, versions, output, points=200):
    started = perf_counter()
    function_seconds = {version: 0.0 for version in versions}
    initial = (
        np.geomspace(study["lo"], study["hi"], points)
        if study["log"]
        else np.linspace(study["lo"], study["hi"], points)
    )
    cache = {version: {} for version in versions}

    def at(x, version):
        x = float(x)
        if x not in cache[version]:
            call_start = perf_counter()
            cache[version][x] = evaluate(
                lambda: study["evaluate"](np.float64(x), versions[version])
            )
            function_seconds[version] += perf_counter() - call_start
        return cache[version][x]

    for v in versions:
        for x in initial:
            at(x, v)
    refined = set(initial)
    feature_locations = []
    anchors = [a for a in study["anchors"] if study["lo"] <= a <= study["hi"]]
    # Refine at genuine branch equations as well as nominal axis thresholds.
    for version in versions:
        rows = [at(x, version) for x in initial]
        for field, target in (
            ("floor_margin", 0),
            ("penetration_raw_ratio", 0.45),
            ("penetration_raw_ratio", 0.55),
            ("incidence_chi", 0),
            ("incidence_chi", 5),
            ("trailing_ratio_preprocessed", 0.2),
            ("hub_ratio_preprocessed", 0.6),
            ("hub_ratio_preprocessed", 0.7),
            ("hub_ratio_preprocessed", 0.8),
            ("hub_ratio_preprocessed", 0.9),
        ):
            if field not in rows[0]:
                continue
            values = np.array([r.get(field, np.nan) - target for r in rows])
            for i in np.flatnonzero(values[:-1] * values[1:] < 0):
                root = bisection(
                    lambda z: at(z, version).get(field, np.nan) - target,
                    initial[i],
                    initial[i + 1],
                )
                if root is not None:
                    anchors.append(root)
                    feature_locations.append(
                        dict(field=field, target=target, x=root, version=version)
                    )
        # Curvature and invalid/valid interfaces are suspicion flags, not diagnoses.
        metrics = (
            ("loss_total", "loss_profile", "loss_secondary", "loss_incidence")
            if study["kind"] not in ("helper", "fundamental")
            else tuple(k for k, v in rows[0].items() if isinstance(v, float))
        )
        for field in metrics:
            y = np.array([r.get(field, np.nan) for r in rows])
            finite = np.isfinite(y)
            for i in np.flatnonzero(finite[:-1] != finite[1:]):
                refined.update(np.linspace(initial[i], initial[i + 1], 11))
            if np.count_nonzero(finite) < 4:
                continue
            d = derivative(initial, y)
            curvature = np.abs(derivative(initial, d))
            candidates = np.argsort(np.nan_to_num(curvature, nan=-1, posinf=-1))[-3:]
            for i in candidates:
                if i > 0 and i < len(initial) - 1:
                    refined.update(np.linspace(initial[i - 1], initial[i + 1], 11))
    for anchor in anchors:
        scale = max(abs(anchor), 1.0)
        refined.add(float(anchor))
        for sign in (-1, 1):
            for magnitude in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                point = anchor + sign * scale * magnitude
                if study["lo"] <= point <= study["hi"]:
                    refined.add(float(point))
    # Root finding is also evidence. Retain every evaluated location and
    # evaluate the union for every model so comparison grids stay identical.
    for version_cache in cache.values():
        refined.update(version_cache)
    x = merge_grid(refined, initial)
    records = []
    arrays = {}
    parity = []
    for version in versions:
        rows = [at(z, version) for z in x]
        metrics = sorted(
            set(k for r in rows for k, v in r.items() if isinstance(v, float))
        )
        values = {k: np.array([r.get(k, np.nan) for r in rows]) for k in metrics}
        slopes = {f"d_{k}_dx": derivative(x, y) for k, y in values.items()}
        arrays[version] = dict(
            x=x, values=values, derivatives=slopes, status=[r["status"] for r in rows]
        )
        for i, (z, r) in enumerate(zip(x, rows)):
            records.append(
                dict(
                    version=version,
                    x=z,
                    base_grid=bool(np.any(initial == z)),
                    **r,
                    **{key: val[i] for key, val in slopes.items()},
                )
            )
    if "baseline" in arrays and "candidate" in arrays:
        a, b = arrays["baseline"], arrays["candidate"]
        for field in sorted(set(a["values"]) | set(b["values"])):
            av = a["values"].get(field, np.full(x.shape, np.nan))
            bv = b["values"].get(field, np.full(x.shape, np.nan))
            finite = np.isfinite(av) & np.isfinite(bv)
            parity.append(
                dict(
                    study=study["id"],
                    field=field,
                    finite_pairs=int(finite.sum()),
                    max_absolute_difference=(
                        float(np.max(np.abs(av[finite] - bv[finite])))
                        if finite.any()
                        else 0.0
                    ),
                    nonfinite_mask_equal=bool(
                        np.array_equal(np.isnan(av), np.isnan(bv))
                        and np.array_equal(np.isposinf(av), np.isposinf(bv))
                        and np.array_equal(np.isneginf(av), np.isneginf(bv))
                    ),
                    statuses_equal=a["status"] == b["status"],
                )
            )
    write_csv(output / (study["id"].lower() + ".csv.gz"), records)
    summary = dict(
        id=study["id"],
        group=study["group"],
        family=study["family"],
        kind=study["kind"],
        rule=study["rule"],
        base_points=len(initial),
        refined_points=len(x),
        lo=study["lo"],
        hi=study["hi"],
        log=study["log"],
        data_path=(output / (study["id"].lower() + ".csv.gz")).relative_to(BASE).as_posix(),
        level=study.get("level", "combined"),
        category=study.get("category", "diagnostics"),
        mechanism=study.get("mechanism"),
        variable=study.get("variable"),
        function_seconds_by_version=function_seconds,
        elapsed_seconds=perf_counter() - started,
        found_branches=feature_locations,
        derivative_policy=POLICY,
        derivative_coordinate="x (including logarithmic display axes)",
        derivative_sample_counts={
            version: {
                field: dict(
                    retained=int(
                        np.isfinite(
                            arrays[version]["derivatives"]["d_" + field + "_dx"]
                        ).sum()
                    ),
                    finite_values=int(np.isfinite(values).sum()),
                )
                for field, values in arrays[version]["values"].items()
            }
            for version in versions
        },
        status_counts={
            v: {
                s: sum(r["status"] == s for r in cache[v].values())
                for s in ("ok", "nonfinite", "exception")
            }
            for v in versions
        },
    )
    return arrays, summary, parity


def local_join(name, fun, location, versions, field):
    """One-sided extrapolated values/slopes at seven shrinking scaled intervals."""
    rows = []
    results = []
    scale = max(1.0, abs(location))
    for version, mods in versions.items():
        measured = []
        for exponent in range(2, 9):
            h = scale * 10.0 ** (-exponent)
            samples = [
                evaluate(lambda z=z: fun(np.float64(z), mods))
                for z in (
                    location - 2 * h,
                    location - h,
                    location,
                    location + h,
                    location + 2 * h,
                )
            ]
            vals = [r.get(field, np.nan) for r in samples]
            ll, l, c, r, rr = vals
            left_limit = 2 * l - ll
            right_limit = 2 * r - rr
            jump = right_limit - left_limit
            dl = (l - ll) / h
            dr = (rr - r) / h
            record = dict(
                feature=name,
                version=version,
                x=location,
                h=h,
                field=field,
                left_limit=left_limit,
                right_limit=right_limit,
                at_join=c,
                jump_estimate=jump,
                left_derivative=dl,
                right_derivative=dr,
                scaled_slope_change=scale * (dr - dl),
                finite=bool(np.all(np.isfinite(vals))),
                statuses=";".join(r["status"] for r in samples),
            )
            rows.append(record)
            measured.append(record)
        tail = measured[-3:]
        last = tail[-1]
        if not all(r["finite"] for r in tail):
            classification = "singular / invalid-domain boundary"
        else:
            yscale = max(1.0, abs(last["left_limit"]), abs(last["right_limit"]))
            jump_tol = 1e-8 * yscale
            slope_tol = 1e-4 * max(
                1.0,
                abs(scale * last["left_derivative"]),
                abs(scale * last["right_derivative"]),
            )
            jumps = np.array([r["jump_estimate"] for r in tail])
            slopes = np.array([r["scaled_slope_change"] for r in tail])
            if np.min(abs(jumps)) > jump_tol and np.ptp(jumps) < 0.1 * abs(jumps[-1]):
                classification = "finite value jump"
            elif abs(slopes[-1]) > slope_tol and np.ptp(slopes) < 0.1 * abs(slopes[-1]):
                classification = "derivative kink"
            elif abs(slopes[-1]) > slope_tol:
                classification = (
                    "nonconverged steep / singular derivative; inspect intervals"
                )
            else:
                classification = (
                    "smooth at resolved tolerance (not proof of global validity)"
                )
        results.append(
            dict(
                feature=name,
                version=version,
                x=location,
                field=field,
                classification=classification,
                jump=last["jump_estimate"],
                left_derivative=last["left_derivative"],
                right_derivative=last["right_derivative"],
                final_h=last["h"],
                tolerance="jump 1e-8*max(1,|Y|); scaled slope 1e-4*max(1,|dY/dscaled_x|); last 3 intervals for convergence",
            )
        )
    return rows, results
