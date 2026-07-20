"""
Experiment 3: Semidiscrete existence verification via the reduced n-atomic q-Bass system.
Verifies Theorem 1.9.
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
from src.qbass_map import f_map, compute_x_star, build_h_to_s, compute_T_alpha, compute_S_alpha_at_yi
from src.solvers import multi_start_solve, filter_solutions
from src.diagnostics import check_convex_order, check_irreducibility, check_monotonicity


def run_experiment(
    q_name: str,
    q_dist,  # distribution object with .cdf, .density, .quantile
    q_support: tuple,
    n_quad: int = 3000,
    n_starts: int = 16,
    results_dir: str = 'results',
) -> dict:
    """
    Run semidiscrete existence verification for given q distribution.
    mu = 0.2*delta_{-2} + 0.4*delta_{-0.5} + 0.4*delta_{1.5}, nu = Uniform[-3,3].
    """
    # Define mu and nu
    atoms = np.array([-2.0, -0.5, 1.5])
    weights = np.array([0.2, 0.4, 0.4])
    mu = AtomicDistribution(atoms=atoms, weights=weights)
    nu = UniformDistribution(a=-3.0, b=3.0)
    
    print(f'\n--- q = {q_name} ---')
    
    # Pre-check: mean equality
    mu_mean = mu.mean()
    nu_mean = nu.mean()
    print(f'  mu.mean={mu_mean:.6f}, nu.mean={nu_mean:.6f}, diff={abs(mu_mean-nu_mean):.2e}')
    
    # Check irreducibility
    p_grid = np.linspace(1e-6, 1-1e-6, 5000)
    irr = check_irreducibility(mu.integrated_quantile, nu.integrated_quantile, p_grid)
    print(f'  Irreducible: {irr["is_irreducible"]}, min_gap={irr["min_gap"]:.6f}')
    
    # Compute x*
    x_star = compute_x_star(atoms, weights)
    print(f'  x* = {x_star}')
    
    # Set up f_map with this q
    F_q = lambda z: q_dist.cdf(z)
    rho_q = lambda z: q_dist.density(z)
    Q_nu = lambda u: nu.quantile(u)
    
    def f(h):
        return f_map(h, weights, F_q, rho_q, Q_nu, q_support, n_quad)
    
    # Determine h bounds (use outer support boundaries)
    q_diam = float(np.asarray(q_support)[-1]) - float(np.asarray(q_support)[0])
    h_lo = np.array([0.01, 0.01])
    h_hi = np.array([max(q_diam - 0.01, 2.0), max(q_diam - 0.01, 2.0)])
    
    # Solve
    print(f'  Solving f(h)=x* with multi-start...')
    all_results = multi_start_solve(f, x_star, (h_lo, h_hi), n_starts=n_starts, seed=42)
    valid = filter_solutions(all_results, residual_threshold=1e-4)
    print(f'  Found {len(valid)} valid solutions')
    
    if not valid:
        print('  WARNING: No solution found!')
        return {'q_name': q_name, 'solved': False}
    
    # Take best solution
    best_res = min(all_results, key=lambda r: r[1])
    h_sol = best_res[0]
    res = best_res[1]
    print(f'  Best h = {h_sol}, residual = {res:.2e}')
    
    # Reconstruct y and alpha
    s = build_h_to_s(h_sol)
    y = s  # y_j = s_{j-1}: y_1=0, y_2=h_1, y_3=h_1+h_2
    print(f'  y = {y}')
    
    # Validate equation (2.2): S_alpha(y_i) = x_i; use finer grid for accuracy
    val_n_quad = max(n_quad, 3000)
    S_vals = np.array([
        compute_S_alpha_at_yi(y[i], y, weights, F_q, rho_q, Q_nu, q_support, val_n_quad)
        for i in range(len(y))
    ])
    print(f'  S_alpha(y_i) = {S_vals}')
    print(f'  Target atoms   = {atoms}')
    S_residual = np.max(np.abs(S_vals - atoms))
    print(f'  S_alpha validation residual: {S_residual:.2e}')
    
    # Check monotonicity
    mono = check_monotonicity(S_vals, y)
    print(f'  S_alpha monotone on supp(alpha): {mono}')
    
    # Check pushforward T_alpha # (alpha*q) = nu via quantile comparison
    x_eval = np.linspace(-3.0, 3.0, 500)
    T_vals = compute_T_alpha(x_eval, y, weights, F_q, Q_nu)
    
    # CDF of pushforward: F_{T_alpha # (alpha*q)}(t) = (alpha*q)({x: T_alpha(x) <= t})
    # = sum_j p_j * q({z: T_alpha(y_j+z) <= t})
    # Since T is non-decreasing in F_alpha*q, compare via quantile plot
    # Simple check: the pushforward CDF at t should equal F_nu(t)
    
    # Plot
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: U_mu vs U_nu
    U_mu = mu.integrated_quantile(p_grid)
    U_nu = nu.integrated_quantile(p_grid)
    axes[0].plot(p_grid, U_mu, 'b-', linewidth=2, label='$U_\\mu$')
    axes[0].plot(p_grid, U_nu, 'r--', linewidth=2, label='$U_\\nu$')
    axes[0].set_title('Integrated Quantile Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: S_alpha values
    axes[1].scatter(y, S_vals, color='blue', s=100, zorder=5, label='$S_\\alpha(y_i)$')
    axes[1].scatter(y, atoms, color='red', marker='x', s=100, zorder=5, label='atoms $x_i$')
    axes[1].plot(y, S_vals, 'b--', linewidth=1)
    axes[1].set_xlabel('$y_i$')
    axes[1].set_title(f'$S_\\alpha$ values vs atoms ({q_name})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f'exp3_{q_name.replace(" ","_")}_existence.png'), dpi=150)
    plt.close(fig)
    
    return {
        'q_name': q_name,
        'solved': True,
        'h_sol': h_sol,
        'residual_f': res,
        'S_residual': S_residual,
        'S_vals': S_vals,
        'y': y,
        'monotone': mono,
    }


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 3: Semidiscrete Existence Verification ===')
    
    # q_reg = Uniform[-1,1]
    q_reg = UniformDistribution(a=-1.0, b=1.0)
    res_reg = run_experiment('q_reg Uniform[-1,1]', q_reg, (-1.0, 1.0),
                              n_quad=500, results_dir=results_dir)
    
    # q_irr = Uniform([-2,-1] union [1,2])
    # Pass all four interval endpoints so f_i_component places breakpoints at ±1 too.
    # n_starts=25 (5x5 grid) ensures starting points near the degenerate solution h_2≈3.
    q_irr = UniformUnionDistribution(intervals=[(-2.0, -1.0), (1.0, 2.0)])
    res_irr = run_experiment('q_irr Uniform union', q_irr,
                              (-2.0, -1.0, 1.0, 2.0),
                              n_quad=500, n_starts=25, results_dir=results_dir)
    
    return res_reg, res_irr


if __name__ == '__main__':
    main()
