"""
Experiment 4: Uniqueness up to translation of the Bass measure.
Verifies Theorem 1.12.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.distributions import AtomicDistribution, UniformDistribution
from src.qbass_map import f_map, compute_x_star
from src.solvers import multi_start_solve, filter_solutions, check_uniqueness
from src.diagnostics import check_irreducibility, compute_hessian_numerically


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 4: Uniqueness of Bass Measure ===')
    
    atoms = np.array([-2.0, -0.5, 1.5])
    weights = np.array([0.2, 0.4, 0.4])
    mu = AtomicDistribution(atoms=atoms, weights=weights)
    nu = UniformDistribution(a=-3.0, b=3.0)
    q = UniformDistribution(a=-1.0, b=1.0)
    q_support = (-1.0, 1.0)
    n_quad = 3000
    
    x_star = compute_x_star(atoms, weights)
    print(f'x* = {x_star}')
    
    F_q = lambda z: q.cdf(z)
    rho_q = lambda z: q.density(z)
    Q_nu = lambda u: nu.quantile(u)
    
    def f(h):
        return f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad)
    
    q_diam = 2.0  # Uniform[-1,1]
    h_lo = np.array([0.01, 0.01])
    h_hi = np.array([q_diam - 0.01, q_diam - 0.01])
    
    # Many starts
    n_starts = 25
    print(f'Running {n_starts}x{n_starts} grid of initial conditions...')
    all_results = multi_start_solve(f, x_star, (h_lo, h_hi), n_starts=n_starts, seed=0)
    valid = filter_solutions(all_results, residual_threshold=1e-4)
    
    print(f'Total runs: {len(all_results)}, valid solutions: {len(valid)}')
    
    if valid:
        l2_disc, linf_disc = check_uniqueness(valid)
        print(f'Max L2 discrepancy among valid h: {l2_disc:.2e}')
        print(f'Max Linf discrepancy among valid h: {linf_disc:.2e}')
        h_mean = np.mean(valid, axis=0)
        print(f'Mean h solution: {h_mean}')
        print(f'Unique solution confirmed: {linf_disc < 1e-3}')
        
        # Optional Hessian test at best solution
        best = min(all_results, key=lambda r: r[1])
        h_best = best[0]
        H = compute_hessian_numerically(f, h_best)
        eigenvalues = np.linalg.eigvalsh(H)
        print(f'Hessian eigenvalues at solution: {eigenvalues}')
        print(f'Negative definite (concave V): {np.all(eigenvalues < 0)}')
        
        # Plot distribution of solutions
        h_arr = np.array(valid)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(h_arr[:, 0], h_arr[:, 1], c='blue', alpha=0.7, s=50)
        axes[0].set_xlabel('$h_1$')
        axes[0].set_ylabel('$h_2$')
        axes[0].set_title('Valid Solutions in $(h_1, h_2)$ Space')
        axes[0].grid(True, alpha=0.3)
        
        residuals = [r for _, r, _ in all_results]
        axes[1].hist(np.log10(np.array(residuals) + 1e-16), bins=20, color='blue', alpha=0.7)
        axes[1].set_xlabel('$\\log_{10}$(residual)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'exp4_uniqueness.png'), dpi=150)
        plt.close(fig)
        
        return {
            'n_valid': len(valid),
            'l2_discrepancy': l2_disc,
            'linf_discrepancy': linf_disc,
            'h_mean': h_mean,
            'hessian_eigenvalues': eigenvalues,
            'unique_confirmed': linf_disc < 1e-3,
        }
    else:
        print('No valid solutions found.')
        return {'n_valid': 0}


if __name__ == '__main__':
    main()
