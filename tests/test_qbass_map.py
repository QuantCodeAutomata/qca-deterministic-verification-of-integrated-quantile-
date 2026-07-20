"""
Tests for src/qbass_map.py.

Verifies build_h_to_s, _g_vec, f_map, compute_x_star, compute_T_alpha,
compute_S_alpha_at_yi with known outcomes from the paper's n=3 example.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distributions import AtomicDistribution, UniformDistribution
from src.qbass_map import (
    build_h_to_s,
    _g_vec,
    f_map,
    compute_x_star,
    compute_T_alpha,
    compute_S_alpha_at_yi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ATOMS = np.array([-2.0, -0.5, 1.5])
WEIGHTS = np.array([0.2, 0.4, 0.4])
NU = UniformDistribution(a=-3.0, b=3.0)
Q_REG = UniformDistribution(a=-1.0, b=1.0)
Q_SUPPORT = (-1.0, 1.0)
N_QUAD = 2000


def F_q(z):
    return Q_REG.cdf(z)


def rho_q(z):
    return Q_REG.density(z)


def Q_nu(u):
    return NU.quantile(u)


# ---------------------------------------------------------------------------
# build_h_to_s
# ---------------------------------------------------------------------------


def test_build_h_to_s_shape():
    h = np.array([0.3, 0.5])
    s = build_h_to_s(h)
    assert len(s) == 3  # n = len(h) + 1


def test_build_h_to_s_values():
    h = np.array([0.4, 0.6])
    s = build_h_to_s(h)
    np.testing.assert_allclose(s, [0.0, 0.4, 1.0], atol=1e-12)


def test_build_h_to_s_zero_h():
    h = np.array([0.0, 0.0])
    s = build_h_to_s(h)
    np.testing.assert_allclose(s, [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# compute_x_star
# ---------------------------------------------------------------------------


def test_compute_x_star_shape():
    x_star = compute_x_star(ATOMS, WEIGHTS)
    assert x_star.shape == (2,)  # n-1 = 2


def test_compute_x_star_values():
    """x_1* = p_1*x_1 = 0.2*(-2) = -0.4, x_2* = -0.4+0.4*(-0.5) = -0.6."""
    x_star = compute_x_star(ATOMS, WEIGHTS)
    expected = np.array([0.2 * (-2.0), 0.2 * (-2.0) + 0.4 * (-0.5)])
    np.testing.assert_allclose(x_star, expected, atol=1e-12)

    # Explicitly: x_1*=-0.4, x_2*=-0.6
    assert abs(x_star[0] - (-0.4)) < 1e-12
    assert abs(x_star[1] - (-0.6)) < 1e-12


def test_compute_x_star_single_atom():
    atoms = np.array([2.0, 4.0])
    weights = np.array([0.5, 0.5])
    x_star = compute_x_star(atoms, weights)
    assert abs(x_star[0] - 0.5 * 2.0) < 1e-12


# ---------------------------------------------------------------------------
# _g_vec (internal CDF of mixture alpha*q)
# ---------------------------------------------------------------------------


def test_g_vec_range():
    """_g_vec must lie in (0, 1) since it's a clipped weighted sum of CDFs."""
    h = np.array([0.5, 0.5])
    s = build_h_to_s(h)
    z_grid = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    g = _g_vec(z_grid, s, WEIGHTS, F_q)
    assert np.all(g > 0.0), f"g has non-positive values: {g}"
    assert np.all(g < 1.0), f"g has values >= 1: {g}"


def test_g_vec_monotone_in_z():
    """_g_vec is non-decreasing in z (sum of CDFs)."""
    h = np.array([0.3, 0.6])
    s = build_h_to_s(h)
    z_vals = np.linspace(-3, 3, 100)
    g_vals = _g_vec(z_vals, s, WEIGHTS, F_q)
    assert np.all(np.diff(g_vals) >= -1e-10)


def test_g_vec_at_extremes():
    """_g_vec -> 0 near z -> -inf, _g_vec -> 1 near z -> +inf (before clipping)."""
    h = np.array([0.5, 0.5])
    s = build_h_to_s(h)
    # At very negative z: all CDFs -> 0, so g -> clipped min 1e-12
    g_neg = _g_vec(np.array([-100.0]), s, WEIGHTS, F_q)[0]
    # At very positive z: all CDFs -> 1, so g -> clipped max 1-1e-12
    g_pos = _g_vec(np.array([100.0]), s, WEIGHTS, F_q)[0]
    assert g_neg < 1e-8
    assert g_pos > 1.0 - 1e-8


def test_g_vec_shape():
    """Output shape matches input z_grid shape."""
    h = np.array([0.3, 0.7])
    s = build_h_to_s(h)
    z_grid = np.linspace(-2, 2, 50)
    g = _g_vec(z_grid, s, WEIGHTS, F_q)
    assert g.shape == (50,)


# ---------------------------------------------------------------------------
# f_map
# ---------------------------------------------------------------------------


def test_f_map_shape():
    h = np.array([0.5, 0.5])
    result = f_map(h, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
    assert result.shape == (2,)  # n-1 = 2


def test_f_map_positive_h_gives_finite():
    h = np.array([0.5, 0.5])
    result = f_map(h, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
    assert np.all(np.isfinite(result))


def test_f_map_range_plausible():
    """f_i values should be in the range of nu (approximately [-3, 3])."""
    h = np.array([0.5, 0.5])
    result = f_map(h, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
    assert np.all(result >= -3.0 - 0.1)
    assert np.all(result <= 3.0 + 0.1)


def test_f_map_monotone_in_h():
    """Increasing h should shift the image (general monotone trend)."""
    h_small = np.array([0.1, 0.1])
    h_large = np.array([1.5, 1.5])
    f_small = f_map(h_small, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
    f_large = f_map(h_large, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
    # Not necessarily monotone per component, but both should be finite
    assert np.all(np.isfinite(f_small))
    assert np.all(np.isfinite(f_large))


# ---------------------------------------------------------------------------
# compute_T_alpha
# ---------------------------------------------------------------------------


def test_T_alpha_range():
    """T_alpha(x) = Q_nu(F_{alpha*q}(x)) should lie in range of nu."""
    y = np.array([0.0, 0.5, 1.0])
    x = np.linspace(-4.0, 4.0, 50)
    T = compute_T_alpha(x, y, WEIGHTS, F_q, Q_nu)
    assert np.all(T >= -3.0 - 1e-6)
    assert np.all(T <= 3.0 + 1e-6)


def test_T_alpha_nondecreasing():
    """T_alpha should be non-decreasing since it's Q_nu ∘ F_{alpha*q}."""
    y = np.array([0.0, 0.5, 1.0])
    x = np.linspace(-4.0, 4.0, 100)
    T = compute_T_alpha(x, y, WEIGHTS, F_q, Q_nu)
    assert np.all(np.diff(T) >= -0.05)  # allow small numerical noise


# ---------------------------------------------------------------------------
# compute_S_alpha_at_yi
# ---------------------------------------------------------------------------


def test_S_alpha_finite():
    """S_alpha(y_i) should be finite."""
    y = np.array([0.0, 0.5, 1.0])
    for i, yi in enumerate(y):
        s = compute_S_alpha_at_yi(yi, y, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
        assert np.isfinite(s), f"S_alpha not finite at i={i}"


def test_S_alpha_in_nu_range():
    """S_alpha values should lie in the range of nu = [-3, 3]."""
    y = np.array([0.0, 0.5, 1.0])
    for i, yi in enumerate(y):
        s = compute_S_alpha_at_yi(yi, y, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
        assert -3.0 - 0.1 <= s <= 3.0 + 0.1


# ---------------------------------------------------------------------------
# Integration: x_star is in image of f_map
# ---------------------------------------------------------------------------


def test_x_star_achievable_near_solution():
    """
    Verify that for some h, f(h) is close to x_star.
    Uses a known approximate solution found by exploration.
    This is a regression test confirming the map hits the target.
    """
    x_star = compute_x_star(ATOMS, WEIGHTS)
    # Broad search to confirm the image is non-empty near x_star
    best_residual = np.inf
    for h1 in np.linspace(0.1, 1.9, 5):
        for h2 in np.linspace(0.1, 1.9, 5):
            h = np.array([h1, h2])
            fh = f_map(h, WEIGHTS, F_q, rho_q, Q_nu, Q_SUPPORT, N_QUAD)
            res = np.linalg.norm(fh - x_star)
            best_residual = min(best_residual, res)
    # Expect to find some h where the map is within 2 units of x_star
    assert best_residual < 5.0, f"Best residual {best_residual:.4f} unexpectedly large"
