"""Version-specific bridge to SciPy's compiled SLSQP reverse communication.

State/workspace setup is adapted from scipy/optimize/_slsqp_py.py v1.17.1;
see SCIPY_LICENSE.txt. This is the only module that uses SciPy private APIs.
No installed files or process-global functions are modified.
"""

import numpy as np
import scipy

SUPPORTED_SCIPY = "1.17.1"


class Backend:
    def __init__(self, x, lower, upper, meq, mineq, *, ftol, maxiter):
        if scipy.__version__ != SUPPORTED_SCIPY:
            raise RuntimeError(
                f"The Turbodash SLSQP driver requires SciPy {SUPPORTED_SCIPY}; "
                f"found {scipy.__version__}. Validate the private interface before upgrading."
            )
        from scipy.optimize._slsqplib import slsqp

        self.call = slsqp
        n, m = len(x), meq + mineq
        self.state = dict.fromkeys(
            ("alpha", "f0", "gs", "h1", "h2", "h3", "h4", "t", "t0"), 0.0
        )
        self.state.update(
            acc=ftol,
            tol=10 * ftol,
            exact=0,
            inconsistent=0,
            reset=0,
            iter=0,
            itermax=maxiter,
            line=0,
            mode=0,
            n=n,
            m=m,
            meq=meq,
        )
        size = (
            n * (n + 1) // 2
            + 3 * m * n
            - (m + 5 * n + 7) * meq
            + 9 * m
            + 8 * n * n
            + 35 * n
            + meq * meq
            + 28
        )
        if mineq == 0:
            size += 2 * n * (n + 1)
        self.buffer = np.zeros(max(size, 1), dtype=np.float64)
        self.indices = np.zeros(max(m + 2 * n + 2, 1), dtype=np.int32)
        self.multipliers = np.zeros(max(m + 2 * n + 2, 1), dtype=np.float64)
        self.normals = np.zeros((max(m, 1), n), dtype=np.float64, order="F")
        self.constraints = np.zeros(max(m, 1), dtype=np.float64)
        self.lower = np.where(np.isfinite(lower), lower, np.nan)
        self.upper = np.where(np.isfinite(upper), upper, np.nan)
        self.bounds = lower, upper
        self.x = x.copy()
        offset = n * (n + 1) // 2
        self.base = self.buffer[offset : offset + n]
        self.step = self.buffer[offset + n + m : offset + 2 * n + m]

    def advance(self, evaluation, jacobian):
        m = self.state["m"]
        if m:
            self.constraints[:m] = np.r_[evaluation.equalities, evaluation.inequalities]
            self.normals[:m] = jacobian[1:]
        self.call(
            self.state,
            evaluation.objective,
            jacobian[0].copy(),
            self.normals,
            self.constraints,
            self.x,
            self.multipliers,
            self.lower,
            self.upper,
            self.buffer,
            self.indices,
        )
        # The native implementation clips line-search requests to the bounds.
        # Verify rather than silently evaluate a different point.
        if not np.all(np.isfinite(self.x)):
            raise RuntimeError("SLSQP returned nonfinite coordinates.")
        return self.state["mode"]

    def shorten_trial(self, fraction):
        """Shorten the actual stored step; preserve native merit/BFGS bookkeeping."""
        if self.state["mode"] != 1:
            raise RuntimeError("Can only shorten a pending SLSQP function request.")
        expected = np.clip(self.base + self.step, *self.bounds)
        if not np.allclose(self.x, expected, rtol=1e-13, atol=1e-13):
            raise RuntimeError(
                "Unexpected SLSQP workspace layout; refusing to modify it."
            )
        self.step *= fraction
        self.state["h3"] *= fraction
        # alpha is the last incremental factor, not a cumulative step fraction.
        self.state["alpha"] *= fraction
        self.x[:] = np.clip(self.base + self.step, *self.bounds)
