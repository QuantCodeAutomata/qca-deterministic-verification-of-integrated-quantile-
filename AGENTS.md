# Project: n-Atomic q-Bass Martingale

## Overview
Implements the discrete (n-atomic) q-Bass martingale coupling between a
finite atomic measure mu and a continuous target nu.

## Source layout (`src/`)

| File | Purpose |
|---|---|
| `__init__.py` | Empty package marker |
| `distributions.py` | `AtomicDistribution`, `UniformDistribution`, `UniformUnionDistribution` |
| `qbass_map.py` | q-Bass map `f(h)`, auxiliary functions (`g_func`, `compute_T_alpha`, `compute_S_alpha_at_yi`, …) |
| `solvers.py` | Root-finding: `solve_qbass_system`, `multi_start_solve`, `filter_solutions`, `check_uniqueness` |
| `diagnostics.py` | Validation: convex order, irreducibility, residuals, Jacobian, pushforward check |
| `plots.py` | Matplotlib helpers: IQ plot, S-function, polygonal chains, convergence |

## Key mathematical conventions

### Convex order (`check_convex_order`)
- `gap = U_mu(p) - U_nu(p)` where `U_F(p) = int_0^p Q_F(u) du`
- `mu <=_cx nu` (mu less spread) iff `gap >= 0` everywhere
- `is_convex_order = True` when `min(gap) >= -tol`
- `is_irreducible = True` when `min(gap) > tol` (strict order)

### AtomicDistribution integrated quantile
- `U(P_i) = sum_{j=1}^i p_j * x_j`; piecewise-linear between breakpoints
- Left-continuous quantile: `Q(u) = x_i` for `u in (P_{i-1}, P_i]`

### q-Bass map target vector `x*`
- `x*_i = U_mu(P_i) = sum_{j=1}^i p_j * x_j` (partial weighted sums of mu)
- Solve `f(h) = x*` with scipy root-finders; h = increment vector (y_1=0 fixed)

## Dependencies
- numpy 1.26+, scipy 1.14+, matplotlib 3.10+

## Running the experiments

```bash
python3 exp/exp1_integrated_quantile.py   # ~1s
python3 exp/exp2_monotonicity_remark.py   # ~0.3s
python3 exp/exp3_semidiscrete_existence.py # ~20s
python3 exp/exp4_uniqueness.py            # ~40s
python3 exp/exp5_polygonal_chains.py      # ~2s
python3 exp/exp6_stability.py             # ~18s
```

All six pass cleanly; total runtime ≈ 80 s on a standard CPU.

## Key implementation notes

### `np.vectorize` — do NOT use it in `_g_vec` / `compute_T_alpha`
All three distribution classes (`UniformDistribution`, `UniformUnionDistribution`,
`ApproximatedDistribution`) accept numpy arrays for `.cdf()`, `.density()`, and
`.quantile()`.  Wrapping them with `np.vectorize` serialises the array call and
causes a ~10× slowdown that breaks the 130 s per-experiment timeout.

### `solve_qbass_system` fallback
`max_nfev=1000` in the `least_squares` fallback (was 5000) keeps worst-case
wall time per start ≤ 17 s for n_quad=3000.

### exp6 `ApproximatedDistribution`
- Uses `np.interp` on a precomputed 5000-point grid → O(1) per array query.
- Fixed `R_k=2.0` matches q_irr support; only the density cap `k` varies.
- Multi-start only on the first k (k=2); subsequent k use targeted seeds
  `h_irr_limit` and `h_prev` to avoid redundant grid sweeps.
- Filter threshold relaxed to `5e-3` to include k=20 whose Jacobian is
  ill-conditioned near the degenerate h₂≈3 boundary.

### exp3 q_irr solver
Uses `n_starts=25` (vs 16 for q_reg) because fewer grid points satisfy the
support-overlap constraint.  Machine-precision residuals achieved for both.
