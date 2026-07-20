# q-Bass Martingale: Deterministic Verification of Integrated-Quantile Characterizations

A Python repository implementing deterministic, computational verification of theoretical results
from the q-Bass martingale literature, covering integrated-quantile convex order, irreducibility,
semidiscrete existence, uniqueness, polygonal-chain geometry, and stability under approximation.

---

## Overview

This project provides clean, exact implementations of the mathematical constructions in the
q-Bass martingale paper—specifically the verification of Proposition 1.8, Theorem 1.9,
Theorem 1.12, Theorem 3.5, and Proposition 2.12—using only numpy, scipy, and matplotlib.

There is **no dependence on market data** or stochastic simulation.  All experiments use
fully synthetic, analytically tractable probability distributions.

---

## Repository Structure

```
.
├── data/               # Synthetic distribution descriptors (no raw market files)
│   └── README.md
├── src/                # Core library modules
│   ├── distributions.py   # AtomicDistribution, UniformDistribution, UniformUnionDistribution
│   ├── qbass_map.py       # g(h,z), f_map, T_alpha, S_alpha, compute_x_star
│   ├── solvers.py         # Multi-start root-finding, continuation, uniqueness diagnostics
│   ├── diagnostics.py     # Convex-order, irreducibility, monotonicity, pushforward checks
│   └── plots.py           # Reusable plotting utilities
├── exp/                # Experiment scripts (one per paper result)
│   ├── exp1_integrated_quantile.py    # Proposition 1.8 — convex order & irreducibility
│   ├── exp2_monotonicity_remark.py    # Remark 1.2  — monotonicity subtlety
│   ├── exp3_semidiscrete_existence.py # Theorem 1.9  — semidiscrete existence
│   ├── exp4_uniqueness.py             # Theorem 1.12 — uniqueness up to translation
│   ├── exp5_polygonal_chains.py       # Theorem 3.5  — admissible polygonal chains
│   └── exp6_stability.py             # Proposition 2.12 — stability under approximation
├── tests/              # pytest test suite
│   ├── test_distributions.py
│   ├── test_diagnostics.py
│   ├── test_qbass_map.py
│   └── test_experiments.py
├── results/            # Generated plots, tables, and RESULTS.md
│   └── RESULTS.md
├── AGENTS.md           # Persistent agent memory
└── README.md
```

---

## Experiments

| ID   | Title                                              | Paper Result      |
|------|----------------------------------------------------|-------------------|
| exp1 | Integrated-quantile convex order & irreducibility  | Proposition 1.8   |
| exp2 | Monotonicity subtlety (Remark 1.2 counterexample)  | Remark 1.2        |
| exp3 | Semidiscrete n-atomic q-Bass existence             | Theorem 1.9       |
| exp4 | Uniqueness of Bass measure up to translation       | Theorem 1.12      |
| exp5 | Geometric characterization of admissible chains    | Theorem 3.5       |
| exp6 | Stability under approximation of nu and q          | Proposition 2.12  |

---

## Installation

```bash
pip install numpy scipy matplotlib
```

All other required packages (already listed in the environment) are used in tests only.

---

## Running Experiments

Run each experiment as a standalone script:

```bash
python exp/exp1_integrated_quantile.py
python exp/exp2_monotonicity_remark.py
python exp/exp3_semidiscrete_existence.py
python exp/exp4_uniqueness.py
python exp/exp5_polygonal_chains.py
python exp/exp6_stability.py
```

Or run all experiments in one shot:

```bash
for f in exp/exp*.py; do python "$f"; done
```

Plots and logs are written to `results/`.

---

## Running Tests

```bash
pytest tests/ -v
```

All 125+ tests cover unit behaviour (distributions, qbass_map, diagnostics) and
integration-level verification of each paper hypothesis.

---

## Key Mathematical Objects

| Symbol       | Description                                           |
|--------------|-------------------------------------------------------|
| `Q_xi(u)`    | Left-continuous quantile of xi: inf{x: F_xi(x)≥u}    |
| `U_xi(p)`    | Integrated quantile: ∫₀ᵖ Q_xi(u) du                  |
| `g(h,z)`     | CDF of mixture alpha*q at z                           |
| `f(h)`       | n-atomic q-Bass map; solution f(h)=x* gives alpha     |
| `T_alpha`    | Bass transport map = Q_nu ∘ F_{alpha*q}               |
| `S_alpha(y)` | Conditional mean: ∫ T_alpha(y+z) ρ_q(z) dz           |

---

## Results Summary

See `results/RESULTS.md` for a detailed quantitative summary of all experiment outcomes.

---

## License

MIT
