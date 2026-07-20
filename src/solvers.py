"""
Solvers for the n-atomic q-Bass reduced system f(h) = x*.

The system f(h) = x* encodes the Bass martingale fixed-point equations in
terms of the increment vector h.  This module provides single-start and
multi-start root-finding, solution filtering, and a uniqueness diagnostic.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, root


def solve_qbass_system(
    f_map: Callable[[np.ndarray], np.ndarray],
    x_star: np.ndarray,
    h_init: np.ndarray,
    h_bounds: Tuple[np.ndarray, np.ndarray],
    tol: float = 1e-8,
    method: str = 'lm',
) -> Tuple[Optional[np.ndarray], float, bool]:
    """Solve f(h) = x_star from a single initial point using scipy root-finders.

    Attempts ``scipy.optimize.root`` first with the chosen method.  If that
    does not converge, falls back to ``scipy.optimize.least_squares`` with
    explicit box constraints given by *h_bounds*.

    Parameters
    ----------
    f_map : callable
        The q-Bass map f: R^{n-1} → R^{n-1}.
    x_star : np.ndarray
        Target vector of length n-1.
    h_init : np.ndarray
        Initial guess for the increment vector, length n-1.
    h_bounds : (np.ndarray, np.ndarray)
        Pair (h_lo, h_hi) of lower and upper bound arrays, each of length n-1.
    tol : float, optional
        Convergence tolerance passed to the solver.  Default 1e-8.
    method : str, optional
        Method for ``scipy.optimize.root`` (e.g. ``'lm'``, ``'hybr'``).
        Default ``'lm'``.

    Returns
    -------
    h_solution : np.ndarray or None
        Solution vector clipped to bounds, or ``None`` on failure.
    residual_norm : float
        L2 norm of f(h_solution) - x_star, or ``np.inf`` on failure.
    success : bool
        ``True`` when residual_norm < 1e-6.
    """
    x_star = np.asarray(x_star, dtype=float)
    h_init = np.asarray(h_init, dtype=float)
    lo, hi = np.asarray(h_bounds[0], dtype=float), np.asarray(h_bounds[1], dtype=float)

    def residual(h: np.ndarray) -> np.ndarray:
        h_c = np.clip(h, lo, hi)
        return np.asarray(f_map(h_c), dtype=float) - x_star

    try:
        # 'lm' uses 'maxiter'; 'hybr' and others use 'maxfev'
        _opts = {'maxiter': 2000} if method == 'lm' else {'maxfev': 2000}
        result = root(residual, h_init, method=method, tol=tol, options=_opts)
        if result.success:
            h_sol = np.clip(result.x, lo, hi)
            res = float(np.linalg.norm(np.asarray(f_map(h_sol), dtype=float) - x_star))
            if res < 1e-6:
                return h_sol, res, True
            # root declared success but actual residual is still too large;
            # warm-start least_squares from the current iterate
            h_init = h_sol

        # Fallback: bounded least-squares (handles degenerate Jacobians better)
        result2 = least_squares(
            residual, h_init, bounds=(lo, hi),
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000,
        )
        h_sol = result2.x
        res = float(np.linalg.norm(np.asarray(f_map(h_sol), dtype=float) - x_star))
        return h_sol, res, res < 1e-6

    except Exception:
        return None, np.inf, False


def multi_start_solve(
    f_map: Callable[[np.ndarray], np.ndarray],
    x_star: np.ndarray,
    h_bounds: Tuple[np.ndarray, np.ndarray],
    n_starts: int = 20,
    tol: float = 1e-8,
    seed: int = 42,
) -> List[Tuple[np.ndarray, float, bool]]:
    """Solve f(h) = x_star from many initial conditions.

    For a 2-component system (n-1 = 2) the starts are placed on a regular
    grid; for higher-dimensional systems random uniform draws are used.
    All starts are supplemented to reach exactly *n_starts* attempts.

    Parameters
    ----------
    f_map : callable
        The q-Bass map f: R^{n-1} → R^{n-1}.
    x_star : np.ndarray
        Target vector of length n-1.
    h_bounds : (np.ndarray, np.ndarray)
        Box bounds (h_lo, h_hi), each of length n-1.
    n_starts : int, optional
        Total number of starting points.  Default 20.
    tol : float, optional
        Tolerance forwarded to :func:`solve_qbass_system`.  Default 1e-8.
    seed : int, optional
        RNG seed for reproducibility.  Default 42.

    Returns
    -------
    list of (h_solution, residual_norm, success)
        One tuple per starting point that did not raise an exception.
    """
    rng = np.random.default_rng(seed)
    n = len(x_star)
    lo, hi = np.asarray(h_bounds[0], dtype=float), np.asarray(h_bounds[1], dtype=float)
    results: List[Tuple[np.ndarray, float, bool]] = []

    if n == 2:
        # 2-component system: use a regular grid over the parameter square
        side = max(2, int(np.sqrt(n_starts)))
        h1_vals = np.linspace(lo[0] + 0.01, hi[0] - 0.01, side)
        h2_vals = np.linspace(lo[1] + 0.01, hi[1] - 0.01, side)
        for h1 in h1_vals:
            for h2 in h2_vals:
                sol, res, ok = solve_qbass_system(
                    f_map, x_star, np.array([h1, h2]), h_bounds, tol
                )
                if sol is not None:
                    results.append((sol, res, ok))
    else:
        # General case: random uniform starts
        for _ in range(n_starts):
            h_init = rng.uniform(lo + 0.01, hi - 0.01)
            sol, res, ok = solve_qbass_system(f_map, x_star, h_init, h_bounds, tol)
            if sol is not None:
                results.append((sol, res, ok))

    return results


def filter_solutions(
    results: List[Tuple[np.ndarray, float, bool]],
    residual_threshold: float = 1e-6,
) -> List[np.ndarray]:
    """Return solutions whose residual norm is below *residual_threshold*.

    Parameters
    ----------
    results : list of (np.ndarray, float, bool)
        Output of :func:`multi_start_solve` or repeated calls to
        :func:`solve_qbass_system`.
    residual_threshold : float, optional
        Maximum accepted L2 residual norm.  Default 1e-6.

    Returns
    -------
    list of np.ndarray
        Filtered h-solution vectors.
    """
    return [h for h, res, _ok in results if res < residual_threshold]


def check_uniqueness(
    solutions: List[np.ndarray],
    translation_normalize: bool = True,
) -> Tuple[float, float]:
    """Measure pairwise spread among a collection of h-solutions.

    Parameters
    ----------
    solutions : list of np.ndarray
        Collection of candidate solution vectors.
    translation_normalize : bool, optional
        Unused; kept for API compatibility.  When ``True`` the solutions are
        compared as-is (the translation degree of freedom is already removed
        by fixing y_1 = 0 in the reduced system).  Default ``True``.

    Returns
    -------
    max_L2_discrepancy : float
        Maximum pairwise L2 distance.
    max_Linf_discrepancy : float
        Maximum pairwise L-infinity distance.
    """
    if len(solutions) <= 1:
        return 0.0, 0.0

    h_arr = [np.asarray(s, dtype=float) for s in solutions]
    max_l2, max_linf = 0.0, 0.0
    for i in range(len(h_arr)):
        for j in range(i + 1, len(h_arr)):
            diff = h_arr[i] - h_arr[j]
            max_l2 = max(max_l2, float(np.linalg.norm(diff)))
            max_linf = max(max_linf, float(np.max(np.abs(diff))))
    return max_l2, max_linf
