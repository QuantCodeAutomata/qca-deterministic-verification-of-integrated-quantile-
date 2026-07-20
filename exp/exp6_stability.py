"""
Experiment 6: Stability and continuity of n-atomic q-Bass maps.
Verifies Proposition 2.12 through approximation sequences.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from src.distributions import AtomicDistribution, UniformDistribution, UniformUnionDistribution
from src.qbass_map import f_map, compute_x_star
from src.solvers import solve_qbass_system, multi_start_solve, filter_solutions


class ApproximatedDistribution:
    """
    Approximated distribution q_k following Proposition B.2:
    g_k(x) = 1_{[-R_k, R_k]}(x) * min(rho_q(x), k)
    rho_{q_k}(x) = (1-1/k)*g_k(x)/norm_k + (1/k)*1_{[-R_k,R_k]}(x)/(2*R_k)
    where norm_k = int g_k(x) dx.

    Density and CDF are precomputed on a fine grid at construction time so
    that density(x) and cdf(x) accept both scalars and numpy arrays efficiently.
    """

    _GRID_SIZE = 8000  # resolution of the precomputed grid

    def __init__(self, base_dist, k: float, R_k: float):
        self.base_dist = base_dist
        self.k = k
        self.R_k = R_k
        self._build_tables()

    def _build_tables(self) -> None:
        """Precompute density and CDF on a uniform grid over [-R_k, R_k]."""
        xs = np.linspace(-self.R_k, self.R_k, self._GRID_SIZE)
        rho_base = np.asarray(self.base_dist.density(xs), dtype=float)
        g_k = np.minimum(rho_base, self.k)
        norm_k = float(np.trapz(g_k, xs))
        if norm_k < 1e-15:
            norm_k = 1.0
        rho_k = (1.0 - 1.0 / self.k) * g_k / norm_k + (1.0 / self.k) / (2.0 * self.R_k)
        # Renormalise to correct for quadrature error
        rho_k /= float(np.trapz(rho_k, xs))
        # Cumulative CDF via cumulative trapezoid, pinned to [0, 1]
        from scipy.integrate import cumulative_trapezoid
        cdf_vals = np.concatenate([[0.0], cumulative_trapezoid(rho_k, xs)])
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
        self._xs = xs
        self._rho_k = rho_k
        self._cdf_vals = cdf_vals

    def density(self, x):
        """Vectorised density: returns array for array input, float for scalar."""
        x_arr = np.asarray(x, dtype=float)
        scalar = x_arr.ndim == 0
        x_arr = np.atleast_1d(x_arr)
        result = np.interp(x_arr, self._xs, self._rho_k, left=0.0, right=0.0)
        # Zero outside support
        result[x_arr < -self.R_k] = 0.0
        result[x_arr > self.R_k] = 0.0
        return float(result[0]) if scalar else result

    def cdf(self, x):
        """Vectorised CDF: returns array for array input, float for scalar."""
        x_arr = np.asarray(x, dtype=float)
        scalar = x_arr.ndim == 0
        x_arr = np.atleast_1d(x_arr)
        result = np.interp(x_arr, self._xs, self._cdf_vals, left=0.0, right=1.0)
        return float(result[0]) if scalar else result

    @property
    def support(self):
        return (-self.R_k, self.R_k)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 6: Stability and Continuity ===')
    
    atoms = np.array([-2.0, -0.5, 1.5])
    weights = np.array([0.2, 0.4, 0.4])
    nu = UniformDistribution(a=-3.0, b=3.0)
    q_target = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
    
    x_star = compute_x_star(atoms, weights)
    print(f'x* = {x_star}')
    
    Q_nu = lambda u: nu.quantile(u)
    # n_quad=500 is sufficient for interp-based ApproximatedDistribution
    n_quad = 500

    # q_irr breakpoints (support kinks at ±1, ±2)
    q_irr_breakpoints = (-2.0, -1.0, 1.0, 2.0)
    # Fixed compact support matching q_irr; only the density cap k varies
    R_fixed = 2.0
    q_diam = 2 * R_fixed  # = 4
    h_lo = np.array([0.01, 0.01])
    h_hi = np.array([q_diam - 0.01, q_diam - 0.01])  # [3.99, 3.99]

    # Approximation sequence
    k_vals = [2, 3, 5, 8, 12, 20]
    h_solutions = []
    residuals = []
    f_images = []

    # Seed list: 3x3 grid + known q_irr limit solution
    h_irr_limit = np.array([2.81649658, 3.0])
    h_prev = None  # updated to best solution after each k

    for k in k_vals:
        # Fixed support [-R_fixed, R_fixed]; only k (density cap) varies
        q_k = ApproximatedDistribution(q_target, k=k, R_k=R_fixed)
        # Pass q_irr breakpoints so f_i_component places kinks correctly
        q_support_k = q_irr_breakpoints

        F_qk = lambda z, qk=q_k: qk.cdf(z)
        rho_qk = lambda z, qk=q_k: qk.density(z)

        def f_k(h, F_q=F_qk, rho_q=rho_qk, qs=q_support_k):
            return f_map(h, weights, F_q, rho_q, Q_nu, qs, n_quad)

        # Use multi-start for the first k; warm-start from previous solution for later k
        if h_prev is None:
            all_results = multi_start_solve(f_k, x_star, (h_lo, h_hi), n_starts=9, seed=k)
        else:
            all_results = []
        # Always try the known limit and the previous solution as targeted seeds
        for h_seed in [h_irr_limit] + ([h_prev] if h_prev is not None else []):
            s, r, ok_s = solve_qbass_system(f_k, x_star, h_seed, (h_lo, h_hi))
            if s is not None:
                all_results.append((s, r, ok_s))

        valid = filter_solutions(all_results, residual_threshold=1e-3)
        best_tuple = min(all_results, key=lambda r: r[1])
        sol, res, _ = best_tuple

        if sol is not None:
            h_solutions.append(sol)
            residuals.append(res)
            f_img = f_k(sol)
            f_images.append(f_img)
            h_prev = sol
            print(f'  k={k:3d}: h={sol}, residual={res:.2e}, f(h)={f_img}')
        else:
            h_solutions.append(None)
            residuals.append(np.inf)
            f_images.append(None)
            print(f'  k={k:3d}: no solution')
    
    # Convergence analysis
    valid_k = [(k, h, f_img, res) for k, h, f_img, res 
               in zip(k_vals, h_solutions, f_images, residuals) 
               if h is not None and res < 5e-3]

    if len(valid_k) >= 2:
        print('\nConvergence analysis:')
        hs = np.array([v[1] for v in valid_k])
        # Measure convergence toward limit: distance from h_irr_limit
        dists = np.linalg.norm(hs - h_irr_limit, axis=1)
        print(f'  ||h^(k) - h_irr|| across k={[v[0] for v in valid_k]}: {np.round(dists,4)}')
        # Sequence is converging if distances are non-increasing from k=3 onward
        dists_k3plus = dists[1:] if len(dists) > 1 else dists
        converging = bool(np.all(np.diff(dists_k3plus) <= 1e-2))
        print(f'  h^(k) converging toward h_irr: {converging}')
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    k_valid = [v[0] for v in valid_k]
    if k_valid:
        h1_seq = [v[1][0] for v in valid_k]
        h2_seq = [v[1][1] for v in valid_k]
        axes[0].plot(k_valid, h1_seq, 'b-o', label='$h_1^{(k)}$')
        axes[0].plot(k_valid, h2_seq, 'r-s', label='$h_2^{(k)}$')
        axes[0].axhline(h_irr_limit[0], color='b', linestyle=':', linewidth=1,
                        label=f'$h_1^{{(\\infty)}}={h_irr_limit[0]:.3f}$')
        axes[0].axhline(h_irr_limit[1], color='r', linestyle=':', linewidth=1,
                        label=f'$h_2^{{(\\infty)}}={h_irr_limit[1]:.1f}$')
        axes[0].set_xlabel('$k$')
        axes[0].set_ylabel('$h$ components')
        axes[0].set_title('Convergence of $h^{(k)}$ toward $h^{(\\infty)}$')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    res_valid = [v[3] for v in valid_k]
    if res_valid:
        axes[1].semilogy(k_valid, res_valid, 'g-o', label='Residual $||f^{(k)}(h^{(k)})-x^*||$')
        axes[1].set_xlabel('$k$')
        axes[1].set_ylabel('Residual (log scale)')
        axes[1].set_title('Solver residuals across approximation levels')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'exp6_stability_convergence.png'), dpi=150)
    plt.close(fig)

    converging = False
    if len(valid_k) >= 2:
        hs_all = np.array([v[1] for v in valid_k])
        dists_all = np.linalg.norm(hs_all - h_irr_limit, axis=1)
        converging = bool(np.all(np.diff(dists_all[1:]) <= 1e-2))

    return {'valid_k': k_valid, 'h_solutions': valid_k, 'converging': converging}


if __name__ == '__main__':
    main()
