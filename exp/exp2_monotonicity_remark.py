"""
Experiment 2: Monotonicity subtlety in Remark 1.2.
Verifies that q*T constant on supp(alpha_1) but trivially monotone for alpha_2=delta_0.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rho_q(z: float) -> float:
    """Density of q = Uniform([-2,-1] union [1,2]): 1/2 on each interval."""
    if -2.0 <= z <= -1.0 or 1.0 <= z <= 2.0:
        return 0.5
    return 0.0


def rho_q_vec(z: np.ndarray) -> np.ndarray:
    """Vectorized version of rho_q."""
    return np.where(((z >= -2) & (z <= -1)) | ((z >= 1) & (z <= 2)), 0.5, 0.0)


def T_map(x: float) -> float:
    """Transport map: T(x) = -1 if x<0, T(x)=1 if x>=0."""
    return 1.0 if x >= 0.0 else -1.0


def T_map_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized T map."""
    return np.where(x >= 0, 1.0, -1.0)


def compute_S_exact(x: float) -> float:
    """
    Compute S(x) = int T(x+z) rho_q(z) dz exactly via overlap formula.
    S(x) = 2*q({z: z >= -x}) - 1 since rho_q is a density.
    
    q({z: z >= -x}) = q([-2,-1] cap [-x,inf)) + q([1,2] cap [-x,inf))
    Each interval has density 1/2, total 1/2 mass each.
    """
    threshold = -x  # T(x+z) = 1 iff z >= threshold
    
    # Mass of q on [-2,-1] intersect [threshold, inf)
    m1 = max(0.0, -1.0 - max(-2.0, threshold)) * 0.5
    
    # Mass of q on [1,2] intersect [threshold, inf)
    m2 = max(0.0, 2.0 - max(1.0, threshold)) * 0.5
    
    return 2.0 * (m1 + m2) - 1.0


def compute_S_numerical(x: float, n_quad: int = 10000) -> float:
    """Compute S(x) by numerical quadrature as a check."""
    # Integrate over [-2,-1] and [1,2] separately (support is disconnected)
    z1 = np.linspace(-2.0, -1.0, n_quad // 2)
    z2 = np.linspace(1.0, 2.0, n_quad // 2)
    I1 = np.trapz(T_map_vec(x + z1) * rho_q_vec(z1), z1)
    I2 = np.trapz(T_map_vec(x + z2) * rho_q_vec(z2), z2)
    return float(I1 + I2)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print('=== Experiment 2: Monotonicity Remark 1.2 ===')
    
    # Evaluate S on a wide range
    x_grid = np.linspace(-3, 3, 1000)
    S_exact = np.array([compute_S_exact(x) for x in x_grid])
    S_numerical = np.array([compute_S_numerical(x) for x in x_grid])
    
    # Check agreement between exact and numerical
    max_diff = np.max(np.abs(S_exact - S_numerical))
    print(f'  Max |S_exact - S_numerical|: {max_diff:.2e}')
    
    # alpha_1 = Uniform[-1/2, 1/2]: evaluate S on support
    x_alpha1 = np.linspace(-0.5, 0.5, 1000)
    S_alpha1 = np.array([compute_S_exact(x) for x in x_alpha1])
    print(f'  S on supp(alpha_1): min={S_alpha1.min():.6f}, max={S_alpha1.max():.6f}, mean={S_alpha1.mean():.6f}')
    print(f'  S constant on supp(alpha_1): {np.allclose(S_alpha1, 0.0, atol=1e-10)}')
    
    # alpha_2 = delta_0: evaluate at x=0
    S_at_0 = compute_S_exact(0.0)
    print(f'  S(0) for alpha_2=delta_0: {S_at_0:.6f}')
    print(f'  Monotonicity trivially holds for alpha_2 (singleton support): True')
    
    # Plot S(x) function
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid, S_exact, 'b-', linewidth=2, label='$S(x) = (q \\star T)(x)$')
    ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax.axvspan(-0.5, 0.5, alpha=0.15, color='green', label='supp($\\alpha_1$)')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='supp($\\alpha_2=\\delta_0$)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$S(x)$')
    ax.set_title('Remark 1.2: $S(x) = (q \\star T)(x)$ — constant on supp($\\alpha_1$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'exp2_S_function.png'), dpi=150)
    plt.close(fig)
    
    results = {
        'S_constant_on_alpha1': bool(np.allclose(S_alpha1, 0.0, atol=1e-10)),
        'S_alpha1_max_abs': float(np.max(np.abs(S_alpha1))),
        'S_at_0': float(S_at_0),
        'exact_numerical_agreement': float(max_diff),
    }
    return results


if __name__ == '__main__':
    main()
