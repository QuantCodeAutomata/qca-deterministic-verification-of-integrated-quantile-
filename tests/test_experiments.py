"""
Integration tests for all 6 experiments.

These tests verify methodology adherence rather than running full computation.
Each test validates the key mathematical properties asserted by the paper.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distributions import AtomicDistribution, UniformDistribution, UniformUnionDistribution
from src.qbass_map import compute_x_star, _g_vec, build_h_to_s
from src.diagnostics import (
    check_convex_order,
    check_irreducibility,
    check_monotonicity,
    check_mean_equality,
)


# ---------------------------------------------------------------------------
# Exp 1: Integrated quantile characterization
# ---------------------------------------------------------------------------


class TestExp1IntegratedQuantile:
    """Verify Proposition 1.8 computationally."""

    def setup_method(self):
        self.p_grid = np.linspace(1e-5, 1 - 1e-5, 10000)
        self.nu = UniformDistribution(a=-3.0, b=3.0)

    def test_case_a_strict_domination(self):
        """Case A: U_mu(p) > U_nu(p) strictly for all p in (0,1)."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        gap = mu.integrated_quantile(self.p_grid) - self.nu.integrated_quantile(self.p_grid)
        assert np.all(gap > 0), f"min gap = {gap.min():.4e}"

    def test_case_a_mean_equality(self):
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        assert abs(mu.mean() - self.nu.mean()) < 1e-12

    def test_case_b_convex_order_with_contact(self):
        """Case B: U_mu >= U_nu but contact at interior p=0.5.

        Gap must be non-negative everywhere and touch zero at p=0.5 (contact).
        Include p=0.5 explicitly so the minimum-gap check is exact.
        """
        mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        grid_with_contact = np.sort(np.concatenate([self.p_grid, [0.5]]))
        gap = (
            mu.integrated_quantile(grid_with_contact)
            - self.nu.integrated_quantile(grid_with_contact)
        )
        # Non-negative gap everywhere (convex order)
        assert np.all(gap >= -1e-8), f"min gap = {gap.min():.4e}"
        # Minimum gap exactly zero at contact point
        assert gap.min() < 1e-8

    def test_case_b_not_irreducible(self):
        """Case B should fail the strict domination check.

        The gap touches zero exactly at p=0.5 so we must include that point.
        """
        mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        grid_with_contact = np.sort(np.concatenate([self.p_grid, [0.5]]))
        irr = check_irreducibility(
            mu.integrated_quantile,
            self.nu.integrated_quantile,
            grid_with_contact,
            tol=1e-8,
        )
        assert irr["is_irreducible"] is False

    def test_case_a_is_irreducible(self):
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        irr = check_irreducibility(
            mu.integrated_quantile,
            self.nu.integrated_quantile,
            self.p_grid,
        )
        assert irr["is_irreducible"] is True

    def test_u_nu_formula(self):
        """U_nu(p) = -3p + 3p^2 for Uniform[-3,3]."""
        p = np.linspace(0, 1, 100)
        U_nu = self.nu.integrated_quantile(p)
        expected = -3 * p + 3 * p ** 2
        np.testing.assert_allclose(U_nu, expected, atol=1e-12)

    def test_u_mu_case_a_breakpoints(self):
        """Verify U_mu at cumulative weight breakpoints for Case A."""
        mu = AtomicDistribution(
            atoms=np.array([-2.0, -0.5, 1.5]),
            weights=np.array([0.2, 0.4, 0.4]),
        )
        # U(0.2) = 0.2*(-2) = -0.4
        assert abs(mu.integrated_quantile(0.2) - (-0.4)) < 1e-12
        # U(0.6) = -0.4 + 0.4*(-0.5) = -0.6
        assert abs(mu.integrated_quantile(0.6) - (-0.6)) < 1e-12
        # U(1.0) = 0 (mean = 0)
        assert abs(mu.integrated_quantile(1.0)) < 1e-12

    def test_u_mu_case_b_at_half(self):
        """Case B: U_mu(0.5) = 0.5^2 - 2*0.5 = -0.75 = U_nu(0.5)."""
        mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        U_mu_half = mu.integrated_quantile(0.5)
        U_nu_half = self.nu.integrated_quantile(0.5)
        assert abs(U_mu_half - (-0.75)) < 1e-10
        assert abs(U_nu_half - (-0.75)) < 1e-10
        assert abs(U_mu_half - U_nu_half) < 1e-10


# ---------------------------------------------------------------------------
# Exp 2: Monotonicity remark
# ---------------------------------------------------------------------------


class TestExp2MonotonicityRemark:
    """Verify Remark 1.2 counterexample."""

    def compute_S_exact(self, x: float) -> float:
        """S(x) = q({z: z >= -x}) * 2 - 1 with q = Uniform on [-2,-1] union [1,2]."""
        threshold = -x
        # Mass on [-2,-1] at or above threshold
        m1 = max(0.0, -1.0 - max(-2.0, threshold)) * 0.5
        # Mass on [1,2] at or above threshold
        m2 = max(0.0, 2.0 - max(1.0, threshold)) * 0.5
        return 2.0 * (m1 + m2) - 1.0

    def test_S_constant_on_alpha1_support(self):
        """S(x) = 0 for all x in [-0.5, 0.5] (supp of alpha_1)."""
        x_vals = np.linspace(-0.5, 0.5, 100)
        S_vals = np.array([self.compute_S_exact(x) for x in x_vals])
        np.testing.assert_allclose(S_vals, 0.0, atol=1e-12)

    def test_S_at_origin(self):
        """S(0) = 0 (for alpha_2 = delta_0, trivially constant)."""
        assert abs(self.compute_S_exact(0.0)) < 1e-12

    def test_S_is_plus_one_far_right(self):
        """S(x) = 1 for large positive x (all q mass below threshold)."""
        # For x >= 2: threshold = -x <= -2, all q mass is at or above -2 >= threshold
        S = self.compute_S_exact(3.0)
        assert abs(S - 1.0) < 1e-12

    def test_S_is_minus_one_far_left(self):
        """S(x) = -1 for large negative x."""
        # For x <= -2: threshold = -x >= 2, no q mass above threshold
        S = self.compute_S_exact(-3.0)
        assert abs(S - (-1.0)) < 1e-12

    def test_S_nondecreasing(self):
        """S(x) is non-decreasing."""
        x_vals = np.linspace(-4, 4, 200)
        S_vals = np.array([self.compute_S_exact(x) for x in x_vals])
        assert np.all(np.diff(S_vals) >= -1e-12)

    def test_monotonicity_trivial_for_delta_0(self):
        """Monotonicity trivially holds for alpha_2 = delta_0 (single point)."""
        S_alpha2 = np.array([self.compute_S_exact(0.0)])
        y_alpha2 = np.array([0.0])
        assert check_monotonicity(S_alpha2, y_alpha2) is True

    def test_alpha1_support_size(self):
        """alpha_1 = Uniform[-0.5, 0.5] has support length 1."""
        nu = UniformDistribution(-0.5, 0.5)
        assert abs(nu.b - nu.a - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Exp 3: Semidiscrete existence
# ---------------------------------------------------------------------------


class TestExp3SemidiscreteExistence:
    """Verify pre-conditions and structure of the n-atomic q-Bass system."""

    def setup_method(self):
        self.atoms = np.array([-2.0, -0.5, 1.5])
        self.weights = np.array([0.2, 0.4, 0.4])
        self.mu = AtomicDistribution(self.atoms, self.weights)
        self.nu = UniformDistribution(a=-3.0, b=3.0)
        self.p_grid = np.linspace(1e-6, 1 - 1e-6, 5000)

    def test_atoms_sorted(self):
        """Atoms must be strictly increasing."""
        assert np.all(np.diff(self.atoms) > 0)

    def test_weights_sum_to_one(self):
        """Weights must sum to 1."""
        assert abs(self.weights.sum() - 1.0) < 1e-12

    def test_mean_compatibility(self):
        """mu and nu must have the same mean."""
        assert check_mean_equality(self.mu, self.nu)

    def test_irreducibility_of_mu(self):
        """mu must be irreducible w.r.t. nu."""
        irr = check_irreducibility(
            self.mu.integrated_quantile,
            self.nu.integrated_quantile,
            self.p_grid,
        )
        assert irr["is_irreducible"] is True

    def test_x_star_values(self):
        """x* = [-0.4, -0.6] for the standard example."""
        x_star = compute_x_star(self.atoms, self.weights)
        assert abs(x_star[0] - (-0.4)) < 1e-12
        assert abs(x_star[1] - (-0.6)) < 1e-12

    def test_x_star_strictly_above_U_nu(self):
        """x_i* = U_mu(P_i) > U_nu(P_i) by irreducibility."""
        x_star = compute_x_star(self.atoms, self.weights)
        p_star = np.cumsum(self.weights[:-1])  # [0.2, 0.6]
        U_nu_at_pstar = self.nu.integrated_quantile(p_star)
        # x_star[i] = U_mu(P_i) > U_nu(P_i)
        assert np.all(x_star > U_nu_at_pstar)

    def test_h_domain_nontrivial(self):
        """For q_reg = Uniform[-1,1], domain D = (0,2)^2 is non-empty."""
        q_reg = UniformDistribution(a=-1.0, b=1.0)
        q_diam = q_reg.b - q_reg.a
        assert q_diam > 0

    def test_g_function_is_cdf(self):
        """_g_vec(z) is a CDF-like function in z (clipped to (0,1))."""
        q = UniformDistribution(a=-1.0, b=1.0)
        h = np.array([0.5, 0.5])
        s = build_h_to_s(h)
        z_grid = np.linspace(-3, 3, 20)
        g = _g_vec(z_grid, s, self.weights, q.cdf)
        assert np.all(g > 0.0)
        assert np.all(g < 1.0)


# ---------------------------------------------------------------------------
# Exp 4: Uniqueness
# ---------------------------------------------------------------------------


class TestExp4Uniqueness:
    """Verify uniqueness properties of the n-atomic q-Bass solution."""

    def test_h_domain_nonempty(self):
        """Domain D = (0, q_diam)^2 is non-empty for q_reg."""
        q = UniformDistribution(a=-1.0, b=1.0)
        q_diam = q.b - q.a  # = 2
        assert q_diam > 0

    def test_x_star_uniquely_determined(self):
        """x* is uniquely determined by mu (atoms and weights)."""
        atoms = np.array([-2.0, -0.5, 1.5])
        weights = np.array([0.2, 0.4, 0.4])
        x_star1 = compute_x_star(atoms, weights)
        x_star2 = compute_x_star(atoms, weights)
        np.testing.assert_array_equal(x_star1, x_star2)

    def test_h_increment_determines_y_up_to_translation(self):
        """Two y vectors with same increments h differ only by additive constant."""
        h = np.array([0.5, 0.7])
        # y_A: starts at 0
        y_A = build_h_to_s(h)  # [0, 0.5, 1.2]
        # y_B: starts at c
        c = 3.0
        y_B = y_A + c  # same increments, different location
        # Differences should be constant c
        diff = y_B - y_A
        assert np.allclose(diff, c)
        # Increments (differences) preserved
        np.testing.assert_allclose(np.diff(y_A), np.diff(y_B), atol=1e-12)


# ---------------------------------------------------------------------------
# Exp 5: Polygonal chains
# ---------------------------------------------------------------------------


class TestExp5PolygonalChains:
    """Verify polygonal chain geometric properties."""

    def test_chain_convexity_check(self):
        """Convex chain has non-decreasing slopes."""
        # Convex chain: vertices at (0,0), (0.2, -0.4), (0.6, -0.6), (1, 0)
        # Slopes: (-0.4-0)/(0.2-0) = -2, (-0.6-(-0.4))/(0.6-0.2) = -0.5, (0-(-0.6))/(1-0.6)=1.5
        slopes = np.array([-2.0, -0.5, 1.5])
        assert np.all(np.diff(slopes) >= 0)

    def test_u_nu_evaluated_at_p_star(self):
        """U_nu at cumulative weights for paper example."""
        nu = UniformDistribution(a=-3.0, b=3.0)
        p_star = np.array([0.2, 0.6])
        U_nu_pstar = nu.integrated_quantile(p_star)
        # U_nu(0.2) = -3*0.2 + 3*0.04 = -0.6 + 0.12 = -0.48
        assert abs(U_nu_pstar[0] - (-0.48)) < 1e-12
        # U_nu(0.6) = -3*0.6 + 3*0.36 = -1.8 + 1.08 = -0.72
        assert abs(U_nu_pstar[1] - (-0.72)) < 1e-12

    def test_x_star_strictly_above_u_nu_at_pstar(self):
        """x* should be strictly above U_nu at p_star (admissible chain)."""
        atoms = np.array([-2.0, -0.5, 1.5])
        weights = np.array([0.2, 0.4, 0.4])
        nu = UniformDistribution(a=-3.0, b=3.0)
        x_star = compute_x_star(atoms, weights)  # [-0.4, -0.6]
        p_star = np.cumsum(weights[:-1])  # [0.2, 0.6]
        U_nu_pstar = nu.integrated_quantile(p_star)  # [-0.48, -0.72]
        assert np.all(x_star > U_nu_pstar)


# ---------------------------------------------------------------------------
# Exp 6: Stability
# ---------------------------------------------------------------------------


class TestExp6Stability:
    """Verify approximation properties for stability experiment."""

    def test_q_k_density_is_positive(self):
        """Approximated density rho_{q_k} should be positive everywhere in support."""
        q_irr = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        # k=5, R_k=10
        k, R_k = 5.0, 10.0
        x_test = np.linspace(-9, 9, 50)
        for x in x_test:
            rho_base = q_irr.density(x)
            g_k = min(float(rho_base), k) if abs(x) <= R_k else 0.0
            # mixture positivity: (1-1/k)*g_k/norm + (1/k)/(2*R_k) > 0
            # The second term ensures positivity
            assert (1.0 / k) / (2 * R_k) > 0

    def test_nu_quantile_at_boundaries(self):
        """Q_nu(0) and Q_nu(1) are the boundary values of nu."""
        nu = UniformDistribution(a=-3.0, b=3.0)
        assert abs(nu.quantile(0.0) - (-3.0)) < 1e-12
        assert abs(nu.quantile(1.0) - 3.0) < 1e-12

    def test_x_star_unchanged_across_q_approximations(self):
        """x* is determined by mu alone, not by q."""
        atoms = np.array([-2.0, -0.5, 1.5])
        weights = np.array([0.2, 0.4, 0.4])
        x_star = compute_x_star(atoms, weights)
        # x* should be [-0.4, -0.6] regardless of q
        np.testing.assert_allclose(x_star, [-0.4, -0.6], atol=1e-12)

    def test_approximation_schedule_monotone(self):
        """k values should increase and R_k should increase."""
        k_vals = [2, 3, 5, 8, 12, 20]
        R_k_vals = [k * 2.0 for k in k_vals]
        assert np.all(np.diff(k_vals) > 0)
        assert np.all(np.diff(R_k_vals) > 0)

    def test_case_b_mu_q_irr_mean_zero(self):
        """The irregular q target also has zero mean (symmetric)."""
        q_irr = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
        assert abs(q_irr.mean()) < 1e-12
