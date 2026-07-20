"""
Experiment 5: Geometric characterization of admissible polygonal chains.
Verifies Theorem 3.5 and Proposition 3.6.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.distributions import AtomicDistribution, UniformDistribution
from src.qbass_map import f_map, compute_x_star, build_h_to_s
from src.solvers import solve_qbass_system
from src.diagnostics import check_convex_order


def piecewise_linear_chain(p_breakpoints: np.ndarray, C_vals: np.ndarray, p_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate piecewise linear polygonal chain with vertices at
    (0,0), (p_breakpoints[0], C_vals[0]), ..., (p_breakpoints[-1], C_vals[-1]), (1,0).
    """
    # Build full vertex arrays
    p_verts = np.concatenate([[0.0], p_breakpoints, [1.0]])
    c_verts = np.concatenate([[0.0], C_vals, [0.0]])
    return np.interp(p_eval, p_verts, c_verts)


def is_convex_chain(p_breakpoints: np.ndarray, C_vals: np.ndarray) -> bool:
    """
    Check convexity of the polygonal chain.
    Chain is convex if slopes are non-decreasing.
    """
    p_verts = np.concatenate([[0.0], p_breakpoints, [1.0]])
    c_verts = np.concatenate([[0.0], C_vals, [0.0]])
    slopes = np.diff(c_verts) / np.diff(p_verts)
    return bool(np.all(np.diff(slopes) >= -1e-10))


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 5: Polygonal Chain Geometric Characterization ===')
    
    atoms = np.array([-2.0, -0.5, 1.5])
    weights = np.array([0.2, 0.4, 0.4])
    nu = UniformDistribution(a=-3.0, b=3.0)
    q = UniformDistribution(a=-1.0, b=1.0)
    q_support = (-1.0, 1.0)
    n_quad = 2000
    q_diam = 2.0
    
    p_star = np.cumsum(weights[:-1])  # [0.2, 0.6]
    
    F_q = lambda z: q.cdf(z)
    rho_q = lambda z: q.density(z)
    Q_nu = lambda u: nu.quantile(u)
    
    p_grid = np.linspace(1e-6, 1-1e-6, 2000)
    U_nu_vals = nu.integrated_quantile(p_grid)
    
    # Forward test: sample h values, compute f(h), check chain properties
    print('\nForward test: sampling h values...')
    h_samples = []
    chain_convex = []
    chain_dominates = []
    
    h_grid = np.linspace(0.05, q_diam - 0.05, 6)
    for h1 in h_grid:
        for h2 in h_grid:
            h = np.array([h1, h2])
            try:
                f_val = f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad)
                C_vals = f_val  # [C_1, C_2] at p_star
                convex = is_convex_chain(p_star, C_vals)
                phi = piecewise_linear_chain(p_star, C_vals, p_grid)
                dominates = bool(np.all(phi >= U_nu_vals - 1e-8))
                h_samples.append(h)
                chain_convex.append(convex)
                chain_dominates.append(dominates)
            except Exception:
                pass
    
    print(f'  Sampled {len(h_samples)} h values')
    print(f'  Chains convex: {sum(chain_convex)}/{len(chain_convex)}')
    print(f'  Chains dominate U_nu: {sum(chain_dominates)}/{len(chain_dominates)}')
    
    # Inverse test: given admissible chains, solve for h
    print('\nInverse test: solving f(h)=C for admissible chains...')
    h_lo = np.array([0.01, 0.01])
    h_hi = np.array([q_diam - 0.01, q_diam - 0.01])
    
    # Generate candidate chains above U_nu
    mu_test = AtomicDistribution(atoms=atoms, weights=weights)
    
    x_star = compute_x_star(atoms, weights)
    U_nu_at_pstar = nu.integrated_quantile(p_star)
    
    # Create several target chain vectors (strictly above U_nu at p_star)
    candidate_Cs = [
        x_star,  # The paper's example
        x_star * 1.05,  # Slightly higher
        x_star * 0.95 if np.all(x_star * 0.95 > U_nu_at_pstar) else x_star,
    ]
    
    for C_target in candidate_Cs:
        if np.any(C_target <= U_nu_at_pstar):
            continue
        if not is_convex_chain(p_star, C_target):
            continue
        def f_minus_C(h):
            return f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad) - C_target
        h_init = np.array([0.5, 0.5])
        sol, res, ok = solve_qbass_system(
            lambda h: f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad),
            C_target, h_init, (h_lo, h_hi)
        )
        print(f'  Chain C={C_target}: solved={ok}, residual={res:.2e}')
    
    # Boundary diagnostics
    print('\nBoundary diagnostics:')
    for h1_small in [0.02, 0.05]:
        h = np.array([h1_small, 0.5])
        f_val = f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad)
        U_nu_at_p = nu.integrated_quantile(p_star)
        print(f'  h1={h1_small:.2f}: f(h)={f_val}, U_nu(p*)={U_nu_at_p}')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_grid, U_nu_vals, 'r-', linewidth=2, label='$U_\\nu(p)$', zorder=5)
    
    # Plot some generated chains
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, min(10, len(h_samples))))
    for idx, (h, conv, dom) in enumerate(zip(h_samples[:10], chain_convex[:10], chain_dominates[:10])):
        f_val = f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad)
        phi = piecewise_linear_chain(p_star, f_val, p_grid)
        linestyle = '-' if (conv and dom) else '--'
        ax.plot(p_grid, phi, linestyle=linestyle, color=colors[idx], alpha=0.7, linewidth=1)
    
    ax.set_xlabel('$p$')
    ax.set_ylabel('Integrated quantile')
    ax.set_title('Theorem 3.5: Polygonal chains generated by $f(h)$ vs $U_\\nu$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'exp5_polygonal_chains.png'), dpi=150)
    plt.close(fig)
    
    return {
        'n_sampled': len(h_samples),
        'n_convex': sum(chain_convex),
        'n_dominates': sum(chain_dominates),
        'all_convex': all(chain_convex),
        'all_dominate': all(chain_dominates),
    }


if __name__ == '__main__':
    main()
