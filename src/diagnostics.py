"""
Diagnostic functions for validating q-Bass martingale solutions.

Provides checks for convex order, irreducibility, mean equality, monotonicity
of S_alpha, residual computation, pushforward verification, and numerical
Hessian/Jacobian estimation.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def check_convex_order(
    U_mu_fn: Callable[[np.ndarray], np.ndarray],
    U_nu_fn: Callable[[np.ndarray], np.ndarray],
    p_grid: np.ndarray,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Check whether mu <=_cx nu in convex order via integrated quantiles.

    The convex order mu <=_cx nu is equivalent to
        U_nu(p) >= U_mu(p)  for all p in [0, 1],
    where U_F(p) = integral_0^p Q_F(u) du is the integrated quantile of F.
    A gap of at least ``-tol`` is accepted as numerically valid.

    Parameters
    ----------
    U_mu_fn : callable
        Integrated quantile of mu; accepts and returns np.ndarray.
    U_nu_fn : callable
        Integrated quantile of nu; accepts and returns np.ndarray.
    p_grid : np.ndarray
        Probability grid in [0, 1] at which to evaluate the functions.
    tol : float, optional
        Numerical tolerance; gap >= -tol is treated as convex order.
        Default 1e-8.

    Returns
    -------
    dict with keys:
        ``'min_gap'`` (float)
            Minimum of U_mu(p) - U_nu(p) over p_grid.
        ``'is_convex_order'`` (bool)
            ``True`` when min_gap >= -tol.
        ``'gap'`` (np.ndarray)
            Full array U_mu(p) - U_nu(p).
    """
    p_grid = np.asarray(p_grid, dtype=float)
    U_mu = np.asarray(U_mu_fn(p_grid), dtype=float)
    U_nu = np.asarray(U_nu_fn(p_grid), dtype=float)
    # gap = U_mu - U_nu >= 0  iff  mu <=_cx nu  (standard convex order)
    gap = U_mu - U_nu
    min_gap = float(np.min(gap))
    return {
        'min_gap': min_gap,
        'is_convex_order': min_gap >= -tol,
        'gap': gap,
    }


def check_irreducibility(
    U_mu_fn: Callable[[np.ndarray], np.ndarray],
    U_nu_fn: Callable[[np.ndarray], np.ndarray],
    p_grid: np.ndarray,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Check whether the convex order mu <=_cx nu is strict (irreducible).

    Irreducibility requires U_mu(p) > U_nu(p) strictly for all p in the
    interior of [0, 1], i.e. min_gap > tol where gap = U_mu - U_nu.

    Parameters
    ----------
    U_mu_fn : callable
        Integrated quantile of mu.
    U_nu_fn : callable
        Integrated quantile of nu.
    p_grid : np.ndarray
        Probability grid in [0, 1].
    tol : float, optional
        Strict gap threshold.  Default 1e-10.

    Returns
    -------
    dict with keys:
        ``'min_gap'`` (float)
            Minimum of U_mu(p) - U_nu(p) over p_grid.
        ``'is_irreducible'`` (bool)
            ``True`` when min_gap > tol.
        ``'gap'`` (np.ndarray)
            Full array U_mu(p) - U_nu(p).
    """
    p_grid = np.asarray(p_grid, dtype=float)
    U_mu = np.asarray(U_mu_fn(p_grid), dtype=float)
    U_nu = np.asarray(U_nu_fn(p_grid), dtype=float)
    gap = U_mu - U_nu
    min_gap = float(np.min(gap))
    return {
        'min_gap': min_gap,
        'is_irreducible': min_gap > tol,
        'gap': gap,
    }


def check_mean_equality(
    dist1: Any,
    dist2: Any,
    tol: float = 1e-10,
) -> bool:
    """Check whether two distributions have the same mean.

    Both objects must expose a ``.mean()`` method.

    Parameters
    ----------
    dist1 : distribution object
        First distribution with ``.mean()`` method.
    dist2 : distribution object
        Second distribution with ``.mean()`` method.
    tol : float, optional
        Absolute tolerance for mean difference.  Default 1e-10.

    Returns
    -------
    bool
        ``True`` when |dist1.mean() - dist2.mean()| < tol.
    """
    return bool(abs(dist1.mean() - dist2.mean()) < tol)


def check_monotonicity(
    S_vals: np.ndarray,
    y_vals: np.ndarray,
    tol: float = 1e-10,
) -> bool:
    """Check that S_alpha is non-decreasing on the atom locations y_vals.

    Parameters
    ----------
    S_vals : np.ndarray
        Values S_alpha(y_i) at the atom locations, shape (n,).
    y_vals : np.ndarray
        Atom locations (unused in computation; included for API clarity).
    tol : float, optional
        Tolerance for numerical non-decrease.  Default 1e-10.

    Returns
    -------
    bool
        ``True`` when every successive difference S_vals[i+1] - S_vals[i]
        is at least -tol.
    """
    S_vals = np.asarray(S_vals, dtype=float)
    if len(S_vals) < 2:
        return True
    return bool(np.all(np.diff(S_vals) >= -tol))


def compute_residuals(
    f_map: Callable[[np.ndarray], np.ndarray],
    h: np.ndarray,
    x_star: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate and summarise the residual of the q-Bass system at h.

    Parameters
    ----------
    f_map : callable
        The q-Bass map f: R^{n-1} → R^{n-1}.
    h : np.ndarray
        Current increment vector, length n-1.
    x_star : np.ndarray
        Target vector, length n-1.

    Returns
    -------
    dict with keys:
        ``'vector'`` (np.ndarray)
            Component-wise residual f(h) - x_star.
        ``'norm'`` (float)
            L2 norm of the residual vector.
        ``'max_abs'`` (float)
            L-infinity norm (maximum absolute component).
    """
    h = np.asarray(h, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    f_vals = np.asarray(f_map(h), dtype=float)
    vec = f_vals - x_star
    return {
        'vector': vec,
        'norm': float(np.linalg.norm(vec)),
        'max_abs': float(np.max(np.abs(vec))),
    }


def verify_pushforward_nu(
    T_alpha_fn: Callable[[np.ndarray], np.ndarray],
    alpha_q_cdf: Any,
    Q_nu: Callable[[float], float],
    p_grid: np.ndarray,
) -> float:
    """Verify that T_alpha pushes alpha*q forward to nu.

    By construction T_alpha = Q_nu ∘ F_{alpha*q}, so T_alpha # (alpha*q) = nu.
    This is equivalent to the identity
        Q_nu(p) = T_alpha(Q_{alpha*q}(p))   for all p in (0, 1).

    The function evaluates both sides on *p_grid* and returns the maximum
    absolute discrepancy.

    Parameters
    ----------
    T_alpha_fn : callable
        Transport map T_alpha; accepts np.ndarray and returns np.ndarray.
    alpha_q_cdf : distribution object
        Distribution object for alpha*q with a ``.quantile`` method.
    Q_nu : callable
        Quantile function of nu; scalar float → float.
    p_grid : np.ndarray
        Probability grid in (0, 1) at which to check the identity.

    Returns
    -------
    float
        Maximum absolute discrepancy max_p |Q_nu(p) - T_alpha(Q_{alpha*q}(p))|.
    """
    p_grid = np.asarray(p_grid, dtype=float)
    # Quantiles of alpha*q at each p
    aq_quantiles = np.asarray(alpha_q_cdf.quantile(p_grid), dtype=float)
    # Apply T_alpha to those quantiles
    T_at_quantiles = np.asarray(T_alpha_fn(aq_quantiles), dtype=float)
    # Reference: Q_nu evaluated at the same p values
    Q_nu_vals = np.array([Q_nu(float(p)) for p in p_grid])
    return float(np.max(np.abs(T_at_quantiles - Q_nu_vals)))


def compute_hessian_numerically(
    f_map: Callable[[np.ndarray], np.ndarray],
    h: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Approximate the Jacobian (or Hessian) of f_map at h by central differences.

    For the q-Bass map f: R^{n-1} → R^{n-1}, this returns the (n-1)×(n-1)
    Jacobian matrix J with J[i, j] = df_i/dh_j, approximated by
        J[:, j] ≈ (f(h + eps e_j) - f(h - eps e_j)) / (2 eps).

    When f_map is the gradient of a scalar potential V, the Jacobian equals
    the Hessian of V.

    Parameters
    ----------
    f_map : callable
        Vector-valued function f: R^{n-1} → R^{n-1} (or R^m for general m).
    h : np.ndarray
        Point at which to approximate the Jacobian, length n-1.
    eps : float, optional
        Finite-difference step size.  Default 1e-5.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (m, n-1).
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    f0 = np.asarray(f_map(h), dtype=float)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        h_p = h.copy()
        h_p[j] += eps
        h_m = h.copy()
        h_m[j] -= eps
        J[:, j] = (
            np.asarray(f_map(h_p), dtype=float)
            - np.asarray(f_map(h_m), dtype=float)
        ) / (2.0 * eps)
    return J
