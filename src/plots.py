"""
Visualization functions for q-Bass martingale experiments.

All functions save the figure to ``results_dir/filename`` using
``tight_layout`` and return the ``matplotlib.figure.Figure`` object.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(results_dir: str) -> None:
    """Create results_dir if it does not exist."""
    os.makedirs(results_dir, exist_ok=True)


def plot_integrated_quantiles(
    p_grid: np.ndarray,
    U_mu_vals: np.ndarray,
    U_nu_vals: np.ndarray,
    title: str,
    filename: str,
    results_dir: str = 'results',
) -> plt.Figure:
    """Plot the integrated quantile functions of mu and nu with the gap shaded.

    The convex order mu <=_cx nu is visualised by plotting U_mu and U_nu on
    the same axes and filling the region between them.

    Parameters
    ----------
    p_grid : np.ndarray
        Probability grid in [0, 1], shape (N,).
    U_mu_vals : np.ndarray
        Integrated quantile U_mu evaluated at p_grid, shape (N,).
    U_nu_vals : np.ndarray
        Integrated quantile U_nu evaluated at p_grid, shape (N,).
    title : str
        Figure title.
    filename : str
        Output filename (e.g. ``'iq_comparison.png'``).
    results_dir : str, optional
        Directory in which to save the figure.  Created if absent.
        Default ``'results'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(p_grid, U_mu_vals, color='steelblue', linewidth=2.0, label=r'$U_\mu(p)$')
    ax.plot(p_grid, U_nu_vals, color='darkorange', linewidth=2.0, label=r'$U_\nu(p)$')

    # Shade gap U_nu - U_mu (should be >= 0 in convex order)
    ax.fill_between(
        p_grid, U_mu_vals, U_nu_vals,
        where=(U_nu_vals >= U_mu_vals),
        alpha=0.25, color='green', label='gap $U_\\nu - U_\\mu \\geq 0$',
    )
    ax.fill_between(
        p_grid, U_mu_vals, U_nu_vals,
        where=(U_nu_vals < U_mu_vals),
        alpha=0.25, color='red', label='gap $U_\\nu - U_\\mu < 0$',
    )

    ax.set_xlabel('$p$', fontsize=13)
    ax.set_ylabel('$U(p)$', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150)
    return fig


def plot_S_function(
    x_grid: np.ndarray,
    S_vals: np.ndarray,
    alpha_support: Tuple[float, float],
    title: str,
    filename: str,
    results_dir: str = 'results',
) -> plt.Figure:
    """Plot the convolution S(x) = integral T_alpha(x+z) rho_q(z) dz.

    A vertical band is drawn to indicate the support of alpha*q (or the
    support of T_alpha as a function of x).

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points, shape (N,).
    S_vals : np.ndarray
        Values S_alpha(x) at each point in x_grid, shape (N,).
    alpha_support : (float, float)
        Left and right endpoints of the region to highlight (the support
        of the mixture alpha*q in x-space).
    title : str
        Figure title.
    filename : str
        Output filename.
    results_dir : str, optional
        Output directory.  Default ``'results'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x_grid, S_vals, color='purple', linewidth=2.0, label=r'$S_\alpha(x)$')

    # Shade the indicated support region
    x_lo, x_hi = alpha_support
    ax.axvspan(x_lo, x_hi, alpha=0.12, color='purple', label='support of $\\alpha * q$')

    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel(r'$S_\alpha(x)$', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150)
    return fig


def plot_polygonal_chain(
    p_grid: np.ndarray,
    U_nu_vals: np.ndarray,
    chains_list: List[np.ndarray],
    chain_labels: List[str],
    title: str,
    filename: str,
    results_dir: str = 'results',
) -> plt.Figure:
    """Overlay polygonal chain approximations on the integrated quantile U_nu.

    Parameters
    ----------
    p_grid : np.ndarray
        Probability grid in [0, 1], shape (N,).
    U_nu_vals : np.ndarray
        Reference integrated quantile U_nu(p), shape (N,).
    chains_list : list of np.ndarray
        Each element is an array of shape (N,) giving one polygonal chain
        approximation evaluated at p_grid.
    chain_labels : list of str
        Legend labels, one per entry in chains_list.
    title : str
        Figure title.
    filename : str
        Output filename.
    results_dir : str, optional
        Output directory.  Default ``'results'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        p_grid, U_nu_vals,
        color='black', linewidth=2.5, label=r'$U_\nu(p)$', zorder=3,
    )

    cmap = plt.cm.tab10
    for k, (chain, label) in enumerate(zip(chains_list, chain_labels)):
        ax.plot(
            p_grid, chain,
            color=cmap(k % 10), linewidth=1.4,
            linestyle='--', label=label, zorder=2,
        )

    ax.set_xlabel('$p$', fontsize=13)
    ax.set_ylabel('$U(p)$', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150)
    return fig


def plot_convergence(
    k_vals: np.ndarray,
    h_sequence: np.ndarray,
    x_star_vals: np.ndarray,
    title: str,
    filename: str,
    results_dir: str = 'results',
) -> plt.Figure:
    """Plot the convergence of increment components h_k toward x*.

    Each component h_k[i] is plotted against iteration index k on a separate
    sub-panel, with a horizontal dashed line at the corresponding target
    value x*[i].

    Parameters
    ----------
    k_vals : np.ndarray
        Iteration indices, shape (K,).
    h_sequence : np.ndarray
        Increment vectors at each iteration, shape (K, n-1).  Each row is
        the h vector produced at step k_vals[k].
    x_star_vals : np.ndarray
        Target values for each component, shape (n-1,).
    title : str
        Overall figure title (set as ``suptitle``).
    filename : str
        Output filename.
    results_dir : str, optional
        Output directory.  Default ``'results'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_dir(results_dir)
    k_vals = np.asarray(k_vals)
    h_sequence = np.asarray(h_sequence)
    x_star_vals = np.asarray(x_star_vals)

    n_comp = h_sequence.shape[1] if h_sequence.ndim == 2 else 1
    if h_sequence.ndim == 1:
        h_sequence = h_sequence[:, np.newaxis]

    n_cols = min(n_comp, 3)
    n_rows = int(np.ceil(n_comp / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    cmap = plt.cm.tab10
    for i in range(n_comp):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        ax.plot(k_vals, h_sequence[:, i], color=cmap(i % 10), linewidth=1.8,
                label=f'$h_{i+1}$')
        if i < len(x_star_vals):
            ax.axhline(
                x_star_vals[i], color='black', linewidth=1.2,
                linestyle='--', label=f'$x^*_{i+1} = {x_star_vals[i]:.4f}$',
            )
        ax.set_xlabel('iteration $k$', fontsize=11)
        ax.set_ylabel(f'$h_{i+1}$', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    # Hide unused axes
    for idx in range(n_comp, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    return fig
