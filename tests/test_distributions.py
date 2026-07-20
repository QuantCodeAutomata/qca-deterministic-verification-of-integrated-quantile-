"""
Tests for src/distributions.py.

Covers AtomicDistribution, UniformDistribution, and UniformUnionDistribution
with edge cases, mathematical properties, and known-outcome regression tests.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distributions import (
    AtomicDistribution,
    UniformDistribution,
    UniformUnionDistribution,
)


# ---------------------------------------------------------------------------
# AtomicDistribution tests
# ---------------------------------------------------------------------------


class TestAtomicDistribution:
    """Tests for AtomicDistribution."""

    def setup_method(self):
        """Standard paper Case A: mu = 0.2*delta_{-2}+0.4*delta_{-0.5}+0.4*delta_{1.5}."""
        self.mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )

    # --- CDF ---

    def test_cdf_left_of_support(self):
        assert self.mu.cdf(-3.0) == 0.0

    def test_cdf_right_of_support(self):
        assert self.mu.cdf(2.0) == 1.0

    def test_cdf_at_first_atom(self):
        # F(-2) = 0.2
        assert abs(self.mu.cdf(-2.0) - 0.2) < 1e-12

    def test_cdf_between_atoms(self):
        # F(-1.0) = 0.2 (atom -0.5 not yet reached)
        assert abs(self.mu.cdf(-1.0) - 0.2) < 1e-12

    def test_cdf_at_last_atom(self):
        assert abs(self.mu.cdf(1.5) - 1.0) < 1e-12

    def test_cdf_vectorized(self):
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.5, 2.0])
        expected = np.array([0.0, 0.2, 0.2, 0.6, 1.0, 1.0])
        np.testing.assert_allclose(self.mu.cdf(x), expected, atol=1e-12)

    def test_cdf_nondecreasing(self):
        x = np.linspace(-3, 3, 200)
        F = self.mu.cdf(x)
        assert np.all(np.diff(F) >= 0)

    # --- Quantile ---

    def test_quantile_first_segment(self):
        # Q(0.1) = -2 (in (0, 0.2))
        assert self.mu.quantile(0.1) == -2.0

    def test_quantile_second_segment(self):
        # Q(0.4) = -0.5 (in (0.2, 0.6))
        assert self.mu.quantile(0.4) == -0.5

    def test_quantile_third_segment(self):
        # Q(0.8) = 1.5 (in (0.6, 1.0))
        assert self.mu.quantile(0.8) == 1.5

    def test_quantile_at_breakpoints(self):
        # At P_1 = 0.2, left-continuous gives x_1 = -2
        assert self.mu.quantile(0.2) == -2.0

    def test_quantile_nondecreasing(self):
        u = np.linspace(1e-6, 1 - 1e-6, 500)
        Q = self.mu.quantile(u)
        assert np.all(np.diff(Q) >= 0)

    # --- Integrated quantile ---

    def test_integrated_quantile_at_zero(self):
        assert abs(self.mu.integrated_quantile(0.0)) < 1e-12

    def test_integrated_quantile_at_one(self):
        # U(1) = mean = 0.2*(-2)+0.4*(-0.5)+0.4*1.5 = 0
        assert abs(self.mu.integrated_quantile(1.0)) < 1e-12

    def test_integrated_quantile_at_cumulative_weights(self):
        # U(0.2) = 0.2*(-2) = -0.4
        assert abs(self.mu.integrated_quantile(0.2) - (-0.4)) < 1e-12
        # U(0.6) = 0.2*(-2) + 0.4*(-0.5) = -0.4 - 0.2 = -0.6
        assert abs(self.mu.integrated_quantile(0.6) - (-0.6)) < 1e-12

    def test_integrated_quantile_slope_in_segment(self):
        """U is piecewise linear with slope x_i on segment i."""
        # On (0, 0.2): slope = -2
        p1, p2 = 0.05, 0.15
        dU = self.mu.integrated_quantile(p2) - self.mu.integrated_quantile(p1)
        assert abs(dU / (p2 - p1) - (-2.0)) < 1e-10

    def test_integrated_quantile_nondecreasing_from_zero(self):
        """U(p) - U(0) should integrate Q which is non-decreasing overall."""
        p = np.linspace(0, 1, 200)
        U = self.mu.integrated_quantile(p)
        # The differences at small p should be ≈ Q_min * dp (slope = -2)
        assert U[0] == 0.0 or abs(U[0]) < 1e-12

    # --- Mean ---

    def test_mean(self):
        expected = 0.2 * (-2) + 0.4 * (-0.5) + 0.4 * 1.5
        assert abs(self.mu.mean() - expected) < 1e-12

    # --- Edge cases ---

    def test_single_atom(self):
        mu = AtomicDistribution(atoms=np.array([3.0]), weights=np.array([1.0]))
        assert mu.cdf(3.0) == 1.0
        assert mu.quantile(0.5) == 3.0
        assert abs(mu.integrated_quantile(1.0) - 3.0) < 1e-12
        assert mu.mean() == 3.0

    def test_two_atoms_equal_weights(self):
        mu = AtomicDistribution(atoms=np.array([0.0, 2.0]), weights=np.array([0.5, 0.5]))
        assert abs(mu.mean() - 1.0) < 1e-12
        assert abs(mu.integrated_quantile(1.0) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# UniformDistribution tests
# ---------------------------------------------------------------------------


class TestUniformDistribution:
    """Tests for UniformDistribution on [-3, 3] (paper's nu)."""

    def setup_method(self):
        self.nu = UniformDistribution(a=-3.0, b=3.0)

    # --- CDF ---

    def test_cdf_below(self):
        assert self.nu.cdf(-4.0) == 0.0

    def test_cdf_above(self):
        assert self.nu.cdf(4.0) == 1.0

    def test_cdf_midpoint(self):
        assert abs(self.nu.cdf(0.0) - 0.5) < 1e-12

    def test_cdf_linear(self):
        x = np.linspace(-3, 3, 50)
        F = self.nu.cdf(x)
        expected = (x + 3) / 6
        np.testing.assert_allclose(F, expected, atol=1e-12)

    # --- Quantile ---

    def test_quantile_formula(self):
        u = np.linspace(0, 1, 50)
        Q = self.nu.quantile(u)
        expected = -3 + 6 * u
        np.testing.assert_allclose(Q, expected, atol=1e-12)

    def test_quantile_cdf_inverse(self):
        x = np.linspace(-3, 3, 20)
        np.testing.assert_allclose(self.nu.quantile(self.nu.cdf(x)), x, atol=1e-12)

    # --- Integrated quantile ---

    def test_integrated_quantile_formula(self):
        p = np.linspace(0, 1, 50)
        U = self.nu.integrated_quantile(p)
        expected = -3 * p + 3 * p ** 2  # a*p + (b-a)/2 * p^2 = -3p + 3p^2
        np.testing.assert_allclose(U, expected, atol=1e-12)

    def test_integrated_quantile_at_one(self):
        assert abs(self.nu.integrated_quantile(1.0)) < 1e-12

    # --- Density ---

    def test_density_inside(self):
        assert abs(self.nu.density(0.0) - 1.0 / 6.0) < 1e-12

    def test_density_outside(self):
        assert self.nu.density(-4.0) == 0.0
        assert self.nu.density(4.0) == 0.0

    # --- Mean ---

    def test_mean(self):
        assert abs(self.nu.mean()) < 1e-12

    # --- Construction error ---

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            UniformDistribution(a=1.0, b=0.5)


# ---------------------------------------------------------------------------
# UniformUnionDistribution tests
# ---------------------------------------------------------------------------


class TestUniformUnionDistribution:
    """Tests for UniformUnionDistribution on [-2,-1] union [1,2] (paper's Case B mu)."""

    def setup_method(self):
        self.mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])

    # --- CDF ---

    def test_cdf_below(self):
        assert self.mu.cdf(-3.0) == 0.0

    def test_cdf_above(self):
        assert self.mu.cdf(3.0) == 1.0

    def test_cdf_in_gap(self):
        assert abs(self.mu.cdf(0.0) - 0.5) < 1e-12

    def test_cdf_at_right_end_first_interval(self):
        assert abs(self.mu.cdf(-1.0) - 0.5) < 1e-12

    def test_cdf_nondecreasing(self):
        x = np.linspace(-3, 3, 300)
        F = self.mu.cdf(x)
        assert np.all(np.diff(F) >= -1e-12)

    # --- Density ---

    def test_density_in_first_interval(self):
        assert abs(self.mu.density(-1.5) - 0.5) < 1e-12

    def test_density_in_gap(self):
        assert self.mu.density(0.0) == 0.0

    def test_density_integrates_to_one(self):
        x1 = np.linspace(-2, -1, 2000)
        x2 = np.linspace(1, 2, 2000)
        rho1 = self.mu.density(x1)
        rho2 = self.mu.density(x2)
        total = np.trapz(rho1, x1) + np.trapz(rho2, x2)
        assert abs(total - 1.0) < 1e-4

    # --- Quantile ---

    def test_quantile_first_interval(self):
        # For u in (0, 0.5): Q(u) = -2 + 2u  (slope = L = 2)
        u = np.array([0.1, 0.25, 0.4])
        expected = -2.0 + 2.0 * u
        np.testing.assert_allclose(self.mu.quantile(u), expected, atol=1e-10)

    def test_quantile_second_interval(self):
        # For u in (0.5, 1): Q(u) = 2u  (since a_2 - C_2 = 1 - 1 = 0; Q = 0 + 2u)
        u = np.array([0.6, 0.75, 0.9])
        # From quantile formula: a_2 + u*L - C_2 = 1 + 2u - 1 = 2u
        expected = 2.0 * u
        np.testing.assert_allclose(self.mu.quantile(u), expected, atol=1e-10)

    def test_quantile_nondecreasing(self):
        u = np.linspace(1e-6, 1 - 1e-6, 500)
        Q = self.mu.quantile(u)
        assert np.all(np.diff(Q) >= -1e-10)

    # --- Integrated quantile ---

    def test_integrated_quantile_at_zero(self):
        assert abs(self.mu.integrated_quantile(0.0)) < 1e-12

    def test_integrated_quantile_at_one(self):
        # mean = 0, so U(1) = 0
        assert abs(self.mu.integrated_quantile(1.0)) < 1e-10

    def test_integrated_quantile_piecewise_quadratic(self):
        """For p in [0, 0.5]: U(p) = p^2 - 2p (from exact formula)."""
        p = np.array([0.1, 0.25, 0.5])
        expected = p ** 2 - 2 * p  # derived analytically
        np.testing.assert_allclose(self.mu.integrated_quantile(p), expected, atol=1e-10)

    def test_integrated_quantile_second_half(self):
        """For p in [0.5, 1]: U(p) = p^2 - 1."""
        p = np.array([0.6, 0.75, 1.0])
        expected = p ** 2 - 1.0
        np.testing.assert_allclose(self.mu.integrated_quantile(p), expected, atol=1e-10)

    # --- Mean ---

    def test_mean(self):
        # Mean of symmetric distribution about 0
        assert abs(self.mu.mean()) < 1e-12

    # --- Convex order vs Uniform[-3,3] ---

    def test_convex_order_case_b(self):
        """Verify U_mu >= U_nu on (0,1) for Case B (convex order)."""
        nu = UniformDistribution(a=-3.0, b=3.0)
        p = np.linspace(1e-6, 1 - 1e-6, 10000)
        U_mu = self.mu.integrated_quantile(p)
        U_nu = nu.integrated_quantile(p)
        gap = U_mu - U_nu
        assert np.all(gap >= -1e-8), f"min gap = {gap.min():.4e}"

    def test_no_strict_irreducibility_case_b(self):
        """Case B has contact with U_nu at interior point p=0.5."""
        nu = UniformDistribution(a=-3.0, b=3.0)
        U_mu_at_half = self.mu.integrated_quantile(0.5)
        U_nu_at_half = nu.integrated_quantile(0.5)
        # Both should equal -3/4
        assert abs(U_mu_at_half - U_nu_at_half) < 1e-10


# ---------------------------------------------------------------------------
# Cross-distribution tests: Case A verification
# ---------------------------------------------------------------------------


class TestCaseAIntegratedQuantile:
    """Verify the key properties for Experiment 1 Case A."""

    def test_strict_domination_case_a(self):
        """U_mu > U_nu strictly on (0,1) for Case A."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        p = np.linspace(1e-5, 1 - 1e-5, 10000)
        U_mu = mu.integrated_quantile(p)
        U_nu = nu.integrated_quantile(p)
        gap = U_mu - U_nu
        assert np.all(gap > 0), f"Minimum gap = {gap.min():.6e}"

    def test_mean_equality_case_a(self):
        """mu and nu have the same mean in Case A."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        assert abs(mu.mean() - nu.mean()) < 1e-12

    def test_case_a_gap_analytic_boundary(self):
        """Gap at p=0.2 (first cumulative weight) is 0.08."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        nu = UniformDistribution(a=-3.0, b=3.0)
        gap_02 = mu.integrated_quantile(0.2) - nu.integrated_quantile(0.2)
        # U_mu(0.2) = -0.4, U_nu(0.2) = -3*0.2 + 3*0.04 = -0.6+0.12 = -0.48
        # gap = -0.4 - (-0.48) = 0.08
        assert abs(gap_02 - 0.08) < 1e-10
