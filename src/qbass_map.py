"""
q-Bass martingale map implementation.

Implements the n-atomic q-Bass map (paper Section 2, Eq. 2.2) and its
auxiliary functions.  No standard library provides this construction;
the implementation follows the paper directly.

Mathematical setup
------------------
Given mu = sum_j p_j * delta_{x_j} (Bass measure) and nu (target), we seek
alpha = sum_j p_j * delta_{y_j} such that the Brenier map T_alpha from alpha*q
to nu satisfies the Bass condition.  Writing y_j = s_{j-1} with
s_0 = 0, s_j = h_1 + ... + h_j, the reduced system becomes f(h) = x* where

    g(h, z)      = sum_j p_j * F_q(z - s_{j-1})       [CDF of alpha*q at z]
    f_i(h)       = int Q_nu(g(h,z)) [sum_{j=1}^i p_j rho_q(z-s_{j-1})] dz
    x_i*         = U_mu(P_i) = sum_{j=1}^i p_j x_j    [partial means of mu]

    F_{alpha*q}(x) = sum_j p_j F_q(x - y_j)
    T_alpha(x)     = Q_nu(F_{alpha*q}(x))
    S_alpha(y_i)   = int T_alpha(y_i + z) rho_q(z) dz
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Offset-vector construction
# ---------------------------------------------------------------------------

def build_h_to_s(h: np.ndarray) -> np.ndarray:
    """Convert an increment vector h to an offset vector s.

    Given h = [h_1, h_2, ..., h_{n-1}] (n-1 positive increments between
    consecutive shifted copies of q), the offset vector is:
        s_0 = 0,  s_j = h_1 + h_2 + ... + h_j,  j = 1, ..., n-1.

    Parameters
    ----------
    h : np.ndarray
        Increment vector of length n-1.  Each entry should be positive.

    Returns
    -------
    np.ndarray
        Offset vector s of length n: s[0] = 0, s[j] = sum_{k=1}^j h_k.
    """
    return np.concatenate([[0.0], np.cumsum(h)])


# ---------------------------------------------------------------------------
# g-function  (CDF of the mixture alpha*q)
# ---------------------------------------------------------------------------

def _g_vec(
    z_grid: np.ndarray,
    s: np.ndarray,
    p_weights: np.ndarray,
    F_q: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Vectorised CDF of the mixture alpha*q on a grid.

    Computes g(z) = sum_j p_j * F_q(z - s_j) for every z in z_grid
    using one array call to F_q per atom.

    Parameters
    ----------
    z_grid : np.ndarray
        1-D evaluation grid, shape (N,).
    s : np.ndarray
        Offset vector returned by :func:`build_h_to_s`, shape (n,).
    p_weights : np.ndarray
        Mixture weights, shape (n,).
    F_q : callable
        CDF of q; accepts and returns np.ndarray.

    Returns
    -------
    np.ndarray
        g values on z_grid, shape (N,), clipped to [1e-12, 1-1e-12].
    """
    g = np.zeros(len(z_grid), dtype=float)
    for j in range(len(p_weights)):
        g += p_weights[j] * np.asarray(F_q(z_grid - s[j]), dtype=float)
    return np.clip(g, 1e-12, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# f_i component of the q-Bass map
# ---------------------------------------------------------------------------

def f_i_component(
    i: int,
    h: np.ndarray,
    p_weights: np.ndarray,
    F_q: Callable[[float], float],
    rho_q: Callable[[float], float],
    Q_nu: Callable[[float], float],
    q_support: Tuple[float, float],
    n_quad: int = 2000,
) -> float:
    """Evaluate the i-th component f_i(h) of the q-Bass map.

    The component is defined by:
        f_i(h) = integral Q_nu(g(h, z))
                          * [sum_{j=1}^{i} p_j * rho_q(z - s_{j-1})] dz

    integrated over the union of all shifted supports
    [q_a + s_j, q_b + s_j], j = 0, ..., n-1.

    Parameters
    ----------
    i : int
        Component index (1-indexed, 1 <= i <= n-1).
    h : np.ndarray
        Increment vector of length n-1.
    p_weights : np.ndarray
        Mixture weights of length n.
    F_q : callable
        CDF of the reference distribution q; scalar → scalar.
    rho_q : callable
        Density of the reference distribution q; scalar → scalar.
    Q_nu : callable
        Quantile function of the target distribution nu;
        scalar in (0, 1) → real.
    q_support : (float, float)
        Support interval [q_a, q_b] of the reference distribution q.
    n_quad : int, optional
        Number of trapezoidal quadrature points.  Default 2000.

    Returns
    -------
    float
        Value of f_i(h).
    """
    s = build_h_to_s(h)
    q_bounds = np.asarray(q_support, dtype=float).ravel()

    # Identify the actual support sub-intervals of q (skip gaps)
    support_ivls: list = []  # (q_lo, q_hi, rho_val)
    for k in range(len(q_bounds) - 1):
        lo_k, hi_k = q_bounds[k], q_bounds[k + 1]
        rho_mid = float(np.asarray(rho_q(0.5 * (lo_k + hi_k)))[()])
        if rho_mid > 0.0:
            support_ivls.append((lo_k, hi_k, rho_mid))

    if not support_ivls:
        return 0.0

    # Kink positions of Q_nu(g(z)): z = s_m + q_bnd for every atom m and bound
    all_kinks = (s[:, None] + q_bounds[None, :]).ravel()

    # Decompose: f_i = sum_{j<i} sum_k  p_j * int_{s_j+q_lo_k}^{s_j+q_hi_k} Q_nu(g)*rho_q(z-s_j) dz
    # Integrating each (j,k) pair separately ensures rho_q has no cross-interval
    # jump discontinuities within each integration domain.
    total_segs = sum(
        max(1, len(np.unique(all_kinks[(all_kinks >= s[j] + q_lo) & (all_kinks <= s[j] + q_hi)])) + 1)
        for j in range(i) for q_lo, q_hi, _ in support_ivls
    )
    pts = max(4, n_quad // max(1, total_segs))

    _rho_q_vec = np.vectorize(rho_q)  # safe for both scalar and array callables

    total = 0.0
    for j in range(i):
        sj = s[j]
        for q_lo, q_hi, _rho_mid in support_ivls:
            int_lo, int_hi = sj + q_lo, sj + q_hi
            kinks_in = all_kinks[(all_kinks >= int_lo) & (all_kinks <= int_hi)]
            bp = np.unique(np.concatenate([[int_lo, int_hi], kinks_in]))
            parts = [np.linspace(bp[m], bp[m + 1], pts) for m in range(len(bp) - 1)]
            z_grid = np.unique(np.concatenate(parts))
            g_vals = _g_vec(z_grid, s, p_weights, F_q)
            Q_vals = np.asarray(Q_nu(g_vals), dtype=float)
            rho_vals = np.asarray(_rho_q_vec(z_grid - sj), dtype=float)
            total += p_weights[j] * float(np.trapz(Q_vals * rho_vals, z_grid))

    return total


# ---------------------------------------------------------------------------
# Full f-map
# ---------------------------------------------------------------------------

def f_map(
    h: np.ndarray,
    p_weights: np.ndarray,
    F_q: Callable[[float], float],
    rho_q: Callable[[float], float],
    Q_nu: Callable[[float], float],
    q_support: Tuple[float, float],
    n_quad: int = 2000,
) -> np.ndarray:
    """Evaluate the n-atomic q-Bass map f(h) = [f_1(h), ..., f_{n-1}(h)].

    The solution h* of f(h) = x* yields the atom spacings of the Bass
    martingale coupling between the reference mixture alpha * q and the
    target nu.

    Parameters
    ----------
    h : np.ndarray
        Increment vector of length n-1.
    p_weights : np.ndarray
        Mixture weights of length n.
    F_q : callable
        CDF of the reference distribution q.
    rho_q : callable
        Density of the reference distribution q.
    Q_nu : callable
        Quantile function of the target distribution nu.
    q_support : (float, float)
        Support interval [q_a, q_b] of q.
    n_quad : int, optional
        Trapezoidal quadrature points per component.  Default 2000.

    Returns
    -------
    np.ndarray
        Vector of length n-1 with entries f_1(h), ..., f_{n-1}(h).
    """
    n = len(p_weights)
    result = np.zeros(n - 1)
    for i in range(1, n):
        result[i - 1] = f_i_component(
            i, h, p_weights, F_q, rho_q, Q_nu, q_support, n_quad
        )
    return result


# ---------------------------------------------------------------------------
# Target vector x*
# ---------------------------------------------------------------------------

def compute_x_star(atoms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the target vector x* = [x_1*, ..., x_{n-1}*].

    The i-th entry is the partial weighted sum
        x_i* = U_mu(P_i) = sum_{j=1}^i p_j x_j,
    where P_i = sum_{j=1}^i p_j is the i-th cumulative weight.

    These partial means are the values the q-Bass map must match:
    f(h*) = x*.

    Parameters
    ----------
    atoms : np.ndarray
        Sorted atom locations of mu; shape (n,).
    weights : np.ndarray
        Probability masses of mu; shape (n,).

    Returns
    -------
    np.ndarray
        Target vector of length n-1.
    """
    n = len(atoms)
    x_star = np.zeros(n - 1)
    for i in range(1, n):
        x_star[i - 1] = float(np.sum(weights[:i] * atoms[:i]))
    return x_star


# ---------------------------------------------------------------------------
# Transport map T_alpha
# ---------------------------------------------------------------------------

def compute_T_alpha(
    x: np.ndarray,
    y: np.ndarray,
    p_weights: np.ndarray,
    F_q: Callable[[float], float],
    Q_nu: Callable[[float], float],
) -> np.ndarray:
    """Evaluate the Bass transport map T_alpha(x) = Q_nu(F_{alpha*q}(x)).

    The mixture CDF is
        F_{alpha*q}(x) = sum_j p_j * F_q(x - y_j),
    and T_alpha := Q_nu composed with F_{alpha*q}.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate T_alpha; shape (m,).
    y : np.ndarray
        Shift centres (atom locations) of the mixture; shape (n,).
    p_weights : np.ndarray
        Mixture weights; shape (n,).
    F_q : callable
        CDF of the reference distribution q; scalar → scalar.
    Q_nu : callable
        Quantile of the target distribution nu; scalar in (0, 1) → real.

    Returns
    -------
    np.ndarray
        T_alpha values at each point in x; shape (m,).
    """
    x = np.asarray(x, dtype=float)
    F_vals = np.zeros(len(x), dtype=float)
    for j, yj in enumerate(y):
        F_vals += p_weights[j] * np.asarray(F_q(x - yj), dtype=float)
    F_vals = np.clip(F_vals, 1e-12, 1.0 - 1e-12)
    return np.asarray(Q_nu(F_vals), dtype=float)


# ---------------------------------------------------------------------------
# Convolution S_alpha
# ---------------------------------------------------------------------------

def compute_S_alpha_at_yi(
    yi: float,
    y: np.ndarray,
    p_weights: np.ndarray,
    F_q: Callable[[float], float],
    rho_q: Callable[[float], float],
    Q_nu: Callable[[float], float],
    q_support: Tuple[float, float],
    n_quad: int = 2000,
) -> float:
    """Evaluate S_alpha(y_i) = integral T_alpha(y_i + z) * rho_q(z) dz.

    This convolves the transport map T_alpha with the reference density
    rho_q at the point y_i.  At the solution, S_alpha(y_i) equals the
    conditional mean of the Bass martingale at atom y_i.

    Parameters
    ----------
    yi : float
        The atom location at which to evaluate S_alpha.
    y : np.ndarray
        All atom locations of the mixture; shape (n,).
    p_weights : np.ndarray
        Mixture weights; shape (n,).
    F_q : callable
        CDF of q; scalar → scalar.
    rho_q : callable
        Density of q; scalar → scalar.
    Q_nu : callable
        Quantile of nu; scalar in (0, 1) → real.
    q_support : (float, float)
        Support interval [q_a, q_b] of q.
    n_quad : int, optional
        Trapezoidal quadrature points.  Default 2000.

    Returns
    -------
    float
        Value of S_alpha(y_i).
    """
    q_bounds = np.asarray(q_support, dtype=float).ravel()

    # Identify support sub-intervals of q
    support_ivls: list = []  # (q_lo, q_hi, rho_val)
    for k in range(len(q_bounds) - 1):
        lo_k, hi_k = q_bounds[k], q_bounds[k + 1]
        rho_mid = float(np.asarray(rho_q(0.5 * (lo_k + hi_k)))[()])
        if rho_mid > 0.0:
            support_ivls.append((lo_k, hi_k, rho_mid))

    if not support_ivls:
        return 0.0

    # Kink positions of T_alpha in z-space: z = y_m - yi + q_bnd for all m, bnd
    all_kinks_z = ((np.asarray(y) - yi)[:, None] + q_bounds[None, :]).ravel()

    total_segs = sum(
        max(1, len(np.unique(all_kinks_z[(all_kinks_z >= q_lo) & (all_kinks_z <= q_hi)])) + 1)
        for q_lo, q_hi, _ in support_ivls
    )
    pts = max(4, n_quad // max(1, total_segs))

    total = 0.0
    for q_lo, q_hi, rho_val in support_ivls:
        kinks_in = all_kinks_z[(all_kinks_z >= q_lo) & (all_kinks_z <= q_hi)]
        bp = np.unique(np.concatenate([[q_lo, q_hi], kinks_in]))
        parts = [np.linspace(bp[m], bp[m + 1], pts) for m in range(len(bp) - 1)]
        z_grid = np.unique(np.concatenate(parts))
        x_vals = yi + z_grid
        T_vals = compute_T_alpha(x_vals, y, p_weights, F_q, Q_nu)
        total += rho_val * float(np.trapz(T_vals, z_grid))

    return total
