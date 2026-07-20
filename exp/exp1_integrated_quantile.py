"""
Experiment 1: Integrated-quantile characterization of convex order and irreducibility.
Replicates Figure 1 from the paper, verifying Proposition 1.8.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.distributions import AtomicDistribution, UniformDistribution, UniformUnionDistribution
from src.diagnostics import check_convex_order, check_irreducibility, check_mean_equality


def run_case_a(p_grid: np.ndarray, results_dir: str = 'results') -> dict:
    """
    Case A: mu = 0.2*delta_{-2} + 0.4*delta_{-0.5} + 0.4*delta_{1.5}, nu = Uniform[-3,3].
    Verifies strict domination (irreducibility).
    """
    # Define distributions
    mu = AtomicDistribution(atoms=np.array([-2.0, -0.5, 1.5]), 
                            weights=np.array([0.2, 0.4, 0.4]))
    nu = UniformDistribution(a=-3.0, b=3.0)
    
    # Compute integrated quantile functions on p_grid
    U_mu = mu.integrated_quantile(p_grid)
    U_nu = nu.integrated_quantile(p_grid)
    
    # Diagnostics
    gap = U_mu - U_nu
    min_gap = np.min(gap)
    
    co = check_convex_order(mu.integrated_quantile, nu.integrated_quantile, p_grid)
    irr = check_irreducibility(mu.integrated_quantile, nu.integrated_quantile, p_grid)
    mean_ok = check_mean_equality(mu, nu)
    
    # Plot
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_grid, U_mu, 'b-', linewidth=2, label=r'$U_\mu(p)$')
    ax.plot(p_grid, U_nu, 'r--', linewidth=2, label=r'$U_\nu(p)$')
    ax.fill_between(p_grid, U_nu, U_mu, alpha=0.2, color='blue', label=r'Gap $U_\mu - U_\nu$')
    ax.set_xlabel('p')
    ax.set_ylabel('Integrated quantile')
    ax.set_title('Case A: Irreducible example ($U_\\mu > U_\\nu$ strictly on $(0,1)$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'exp1_case_a_integrated_quantiles.png'), dpi=150)
    plt.close(fig)
    
    return {
        'case': 'A',
        'min_gap': min_gap,
        'is_convex_order': co['is_convex_order'],
        'is_irreducible': irr['is_irreducible'],
        'mean_equal': mean_ok,
        'mu_mean': mu.mean(),
        'nu_mean': nu.mean(),
        'gap_array': gap,
    }


def run_case_b(p_grid: np.ndarray, results_dir: str = 'results') -> dict:
    """
    Case B: mu = Uniform([-2,-1] union [1,2]), nu = Uniform[-3,3].
    Verifies convex order without irreducibility.
    """
    mu = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
    nu = UniformDistribution(a=-3.0, b=3.0)

    # Augment the grid with the known CDF breakpoints of mu (p=0.5 is where U_mu=U_nu)
    p_eval = np.sort(np.unique(np.concatenate([p_grid, [0.5]])))
    U_mu = mu.integrated_quantile(p_eval)
    U_nu = nu.integrated_quantile(p_eval)
    
    gap = U_mu - U_nu
    min_gap = np.min(gap)
    contact_points = p_eval[np.abs(gap) < 1e-8]
    
    co = check_convex_order(mu.integrated_quantile, nu.integrated_quantile, p_eval)
    irr = check_irreducibility(mu.integrated_quantile, nu.integrated_quantile, p_eval)
    mean_ok = check_mean_equality(mu, nu)
    
    # Plot
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_eval, U_mu, 'b-', linewidth=2, label=r'$U_\mu(p)$')
    ax.plot(p_eval, U_nu, 'r--', linewidth=2, label=r'$U_\nu(p)$')
    ax.fill_between(p_eval, U_nu, U_mu, alpha=0.2, color='blue', label=r'Gap $U_\mu - U_\nu$')
    if len(contact_points) > 0:
        ax.scatter(contact_points, 
                  [mu.integrated_quantile(cp) for cp in contact_points],
                  color='red', zorder=5, s=60, label='Contact points')
    ax.set_xlabel('p')
    ax.set_ylabel('Integrated quantile')
    ax.set_title('Case B: Convex order without irreducibility (contact at interior $p$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'exp1_case_b_integrated_quantiles.png'), dpi=150)
    plt.close(fig)
    
    return {
        'case': 'B',
        'min_gap': min_gap,
        'is_convex_order': co['is_convex_order'],
        'is_irreducible': irr['is_irreducible'],
        'mean_equal': mean_ok,
        'contact_points': contact_points,
        'gap_array': gap,
    }


def main():
    p_grid = np.linspace(1e-6, 1 - 1e-6, 10000)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 1: Integrated Quantile Characterization ===')
    
    res_a = run_case_a(p_grid, results_dir)
    print(f"Case A: min_gap={res_a['min_gap']:.6f}, convex_order={res_a['is_convex_order']}, irreducible={res_a['is_irreducible']}, mean_equal={res_a['mean_equal']}")
    
    res_b = run_case_b(p_grid, results_dir)
    print(f"Case B: min_gap={res_b['min_gap']:.6f}, convex_order={res_b['is_convex_order']}, irreducible={res_b['is_irreducible']}, contact_pts={res_b['contact_points'][:5]}")
    
    return res_a, res_b


if __name__ == '__main__':
    main()
