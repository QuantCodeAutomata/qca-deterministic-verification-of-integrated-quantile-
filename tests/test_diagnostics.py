"""
Tests for src/diagnostics.py.

Verifies check_convex_order, check_irreducibility, check_mean_equality,
check_monotonicity, compute_residuals, and compute_hessian_numerically.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distributions import AtomicDistribution, UniformDistribution, UniformUnionDistribution
from src.diagnostics import (
    check_convex_order,
    check_irreducibility,
    check_mean_equality,
    check_monotonicity,
    compute_residuals,
    compute_hessian_numerically,
)


# ---------------------------------------------------------------------------
# check_convex_order
# ---------------------------------------------------------------------------


class TestCheckConvexOrder:
    def setup_method(self):
        self.p_grid = np.linspace(1e-6, 1 - 1e-6, 5000)

    def test_case_a_convex_order_holds(self):
        """Case A: U_mu >= U_nu everywhere."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        result = check_convex_order(mu.integrated_quantile, nu.integrated_quantile, self.p_grid)
        assert result["is_convex_order"] is True
        assert result["min_gap"] > 0

    def test_case_b_convex_order_holds(self):
        """Case B: U_mu >= U_nu everywhere (convex order)."""
        mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        nu = UniformDistribution(a=-3.0, b=3.0)
        result = check_convex_order(mu.integrated_quantile, nu.integrated_quantile, self.p_grid)
        assert result["is_convex_order"] is True

    def test_reverse_order_fails(self):
        """If mu and nu are swapped, convex order should fail."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        # Swap: check nu <=_cx mu (should be False)
        result = check_convex_order(nu.integrated_quantile, mu.integrated_quantile, self.p_grid)
        assert result["is_convex_order"] is False

    def test_gap_array_shape(self):
        mu = AtomicDistribution(np.array([0.0]), np.array([1.0]))
        nu = UniformDistribution(-1.0, 1.0)
        result = check_convex_order(mu.integrated_quantile, nu.integrated_quantile, self.p_grid)
        assert result["gap"].shape == self.p_grid.shape

    def test_identical_distributions(self):
        """Identical distributions: min_gap ≈ 0."""
        nu = UniformDistribution(a=-3.0, b=3.0)
        result = check_convex_order(nu.integrated_quantile, nu.integrated_quantile, self.p_grid)
        assert abs(result["min_gap"]) < 1e-10
        assert result["is_convex_order"] is True


# ---------------------------------------------------------------------------
# check_irreducibility
# ---------------------------------------------------------------------------


class TestCheckIrreducibility:
    def setup_method(self):
        self.p_grid = np.linspace(1e-6, 1 - 1e-6, 5000)

    def test_case_a_is_irreducible(self):
        """Case A should show strict domination (irreducible)."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        result = check_irreducibility(mu.integrated_quantile, nu.integrated_quantile, self.p_grid)
        assert result["is_irreducible"] is True
        assert result["min_gap"] > 1e-10

    def test_case_b_not_irreducible(self):
        """Case B should NOT be irreducible (contact at p=0.5).

        The gap U_mu - U_nu touches zero exactly at p = 0.5 (the midpoint
        between the two intervals).  We must include that point in the grid;
        otherwise a dense-but-uniform linspace misses it by ~O(grid_step).
        """
        mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        nu = UniformDistribution(a=-3.0, b=3.0)
        # Include the exact contact point p=0.5 in the evaluation grid.
        grid_with_contact = np.sort(
            np.concatenate([self.p_grid, [0.5]])
        )
        result = check_irreducibility(
            mu.integrated_quantile, nu.integrated_quantile, grid_with_contact
        )
        assert result["is_irreducible"] is False
        # Minimum gap should be ≈ 0 (contact at p=0.5)
        assert result["min_gap"] < 1e-8


# ---------------------------------------------------------------------------
# check_mean_equality
# ---------------------------------------------------------------------------


def test_check_mean_equality_same_mean():
    mu = AtomicDistribution(np.array([-2.0, -0.5, 1.5]), np.array([0.2, 0.4, 0.4]))
    nu = UniformDistribution(-3.0, 3.0)
    assert check_mean_equality(mu, nu) is True


def test_check_mean_equality_different_means():
    mu = AtomicDistribution(np.array([1.0]), np.array([1.0]))
    nu = UniformDistribution(-3.0, 3.0)
    assert check_mean_equality(mu, nu) is False


# ---------------------------------------------------------------------------
# check_monotonicity
# ---------------------------------------------------------------------------


def test_check_monotonicity_increasing():
    S_vals = np.array([-2.0, -0.5, 1.5])
    y_vals = np.array([0.0, 0.5, 1.0])
    assert check_monotonicity(S_vals, y_vals) is True


def test_check_monotonicity_constant():
    S_vals = np.array([1.0, 1.0, 1.0])
    y_vals = np.array([0.0, 1.0, 2.0])
    assert check_monotonicity(S_vals, y_vals) is True


def test_check_monotonicity_decreasing():
    S_vals = np.array([1.5, 0.0, -2.0])
    y_vals = np.array([0.0, 0.5, 1.0])
    assert check_monotonicity(S_vals, y_vals) is False


def test_check_monotonicity_single_value():
    assert check_monotonicity(np.array([3.0]), np.array([0.0])) is True


def test_check_monotonicity_two_values():
    assert check_monotonicity(np.array([1.0, 2.0]), np.array([0.0, 1.0])) is True
    assert check_monotonicity(np.array([2.0, 1.0]), np.array([0.0, 1.0])) is False


# ---------------------------------------------------------------------------
# compute_residuals
# ---------------------------------------------------------------------------


def test_compute_residuals_exact_zero():
    """If h gives f(h) = x_star exactly, residual is zero."""
    def f_map(h):
        return h * 2.0  # simple f(h) = 2h

    h = np.array([1.0, 2.0])
    x_star = np.array([2.0, 4.0])
    res = compute_residuals(f_map, h, x_star)
    assert abs(res["norm"]) < 1e-12
    assert abs(res["max_abs"]) < 1e-12
    np.testing.assert_allclose(res["vector"], 0.0, atol=1e-12)


def test_compute_residuals_known_value():
    def f_map(h):
        return np.array([1.0, 2.0])

    h = np.array([0.0, 0.0])
    x_star = np.array([0.0, 0.0])
    res = compute_residuals(f_map, h, x_star)
    expected_norm = np.sqrt(1.0 ** 2 + 2.0 ** 2)
    assert abs(res["norm"] - expected_norm) < 1e-12


# ---------------------------------------------------------------------------
# compute_hessian_numerically
# ---------------------------------------------------------------------------


def test_hessian_numerical_diagonal():
    """For f(h) = [h1^2, h2^2], Jacobian at [1,1] should be diag([2,2])."""
    def f_map(h):
        return h ** 2

    h = np.array([1.0, 1.0])
    J = compute_hessian_numerically(f_map, h, eps=1e-5)
    expected = np.diag([2.0, 2.0])
    np.testing.assert_allclose(J, expected, atol=1e-4)


def test_hessian_numerical_linear():
    """For f(h) = A @ h, Jacobian should be A (constant)."""
    A = np.array([[2.0, 0.5], [0.3, 1.5]])

    def f_map(h):
        return A @ h

    h = np.array([1.0, 2.0])
    J = compute_hessian_numerically(f_map, h, eps=1e-5)
    np.testing.assert_allclose(J, A, atol=1e-6)


def test_hessian_shape():
    def f_map(h):
        return h * 2

    h = np.array([1.0, 2.0, 3.0])
    J = compute_hessian_numerically(f_map, h)
    assert J.shape == (3, 3)
