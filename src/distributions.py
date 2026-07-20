"""
Distribution classes for quantitative finance mathematical experiments.

Provides exact closed-form CDF, quantile, and integrated quantile functions
for atomic, uniform, and piecewise-uniform distributions.  All public methods
accept both Python scalars and NumPy arrays and return the same type.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[float, np.ndarray]


def _to_array(x: ArrayLike) -> Tuple[np.ndarray, bool]:
    """Return (atleast_1d float array, was_scalar_flag)."""
    x = np.asarray(x, dtype=float)
    return np.atleast_1d(x), x.ndim == 0


def _ret(arr: np.ndarray, scalar: bool) -> Union[float, np.ndarray]:
    """Squeeze to Python float when the original input was a scalar."""
    return float(arr[0]) if scalar else arr


# ---------------------------------------------------------------------------
# AtomicDistribution
# ---------------------------------------------------------------------------

class AtomicDistribution:
    """Finite discrete probability distribution supported on sorted atoms.

    The distribution is mu = sum_{i=1}^n p_i * delta_{x_i} with atoms
    x_1 < x_2 < ... < x_n and probability masses p_1, ..., p_n summing to 1.

    Parameters
    ----------
    atoms : array_like
        Sorted support points x_1 < x_2 < ... < x_n.
    weights : array_like
        Non-negative probability masses p_1, ..., p_n with sum equal to 1.

    Attributes
    ----------
    atoms : np.ndarray
        Sorted atom locations (shape ``(n,)``).
    weights : np.ndarray
        Probability masses (shape ``(n,)``).
    cumulative_weights : np.ndarray
        Partial sums P_i = sum_{j=1}^i p_j (shape ``(n,)``).
    """

    def __init__(self, atoms: np.ndarray, weights: np.ndarray) -> None:
        self.atoms: np.ndarray = np.asarray(atoms, dtype=float)
        self.weights: np.ndarray = np.asarray(weights, dtype=float)
        self.cumulative_weights: np.ndarray = np.cumsum(self.weights)  # [P_1, ..., P_n]

        # Extended breakpoints prepended with 0: [0, P_1, ..., P_n]
        self._P: np.ndarray = np.concatenate([[0.0], self.cumulative_weights])

        # Integrated-quantile values at probability breakpoints.
        # _U_vals[k] = sum_{j=0}^{k-1} weights[j] * atoms[j]   (0-based j)
        # so _U_vals[0] = 0, _U_vals[1] = p_1*x_1, ...
        self._U_vals: np.ndarray = np.concatenate(
            [[0.0], np.cumsum(self.weights * self.atoms)]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def cdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the right-continuous CDF F(x) = P(X <= x).

        The CDF is piecewise-constant, jumping by p_i at each atom x_i.

        Parameters
        ----------
        x : float or np.ndarray
            Points at which to evaluate F.

        Returns
        -------
        float or np.ndarray
            CDF values in [0, 1], same shape as input.
        """
        x, scalar = _to_array(x)
        # searchsorted(..., 'right') counts atoms strictly <= x
        idx = np.searchsorted(self.atoms, x, side='right')
        safe_idx = np.clip(idx - 1, 0, len(self.atoms) - 1)
        result = np.where(idx == 0, 0.0, self.cumulative_weights[safe_idx])
        return _ret(result, scalar)

    def quantile(self, u: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the left-continuous quantile Q(u) = inf{x : F(x) >= u}.

        Q is piecewise-constant:
            Q(u) = x_i  for u in (P_{i-1}, P_i],  i = 1, ..., n.

        Parameters
        ----------
        u : float or np.ndarray
            Probability levels in [0, 1].

        Returns
        -------
        float or np.ndarray
            Quantile values, same shape as input.
        """
        u, scalar = _to_array(u)
        # searchsorted(cumulative_weights, u, 'left') returns the smallest index k
        # such that cumulative_weights[k] >= u, i.e., the 0-based segment index
        # mapping u in (P_{k-1}, P_k] to atoms[k].
        idx = np.searchsorted(self.cumulative_weights, u, side='left')
        idx = np.clip(idx, 0, len(self.atoms) - 1)
        return _ret(self.atoms[idx], scalar)

    def integrated_quantile(self, p: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate U(p) = integral_0^p Q(u) du  (piecewise linear, exact).

        Breakpoint values:
            U(0)   = 0
            U(P_i) = sum_{j=1}^i p_j x_j

        Slope on (P_{i-1}, P_i) equals x_i, so between breakpoints:
            U(p) = U(P_{i-1}) + x_i * (p - P_{i-1}).

        Parameters
        ----------
        p : float or np.ndarray
            Upper integration limits in [0, 1].

        Returns
        -------
        float or np.ndarray
            U(p) values, same shape as input.
        """
        p, scalar = _to_array(p)
        idx = np.searchsorted(self.cumulative_weights, p, side='left')
        idx = np.clip(idx, 0, len(self.atoms) - 1)
        # Left probability breakpoint of the segment containing p
        p_left = self._P[idx]
        result = self._U_vals[idx] + self.atoms[idx] * (p - p_left)
        return _ret(result, scalar)

    def mean(self) -> float:
        """Return the mean E[X] = sum_i p_i x_i.

        Returns
        -------
        float
        """
        return float(np.dot(self.atoms, self.weights))


# ---------------------------------------------------------------------------
# UniformDistribution
# ---------------------------------------------------------------------------

class UniformDistribution:
    """Continuous uniform distribution on the closed interval [a, b].

    Parameters
    ----------
    a : float
        Left endpoint.
    b : float
        Right endpoint.  Must satisfy b > a.
    """

    def __init__(self, a: float, b: float) -> None:
        if b <= a:
            raise ValueError(f"Require b > a; got a={a}, b={b}.")
        self.a: float = float(a)
        self.b: float = float(b)
        self._width: float = self.b - self.a

    # ------------------------------------------------------------------

    def cdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the CDF F(x) = clip((x - a) / (b - a), 0, 1).

        Parameters
        ----------
        x : float or np.ndarray

        Returns
        -------
        float or np.ndarray
            Values in [0, 1], same shape as input.
        """
        x, scalar = _to_array(x)
        result = np.clip((x - self.a) / self._width, 0.0, 1.0)
        return _ret(result, scalar)

    def quantile(self, u: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the quantile Q(u) = a + (b - a) * u.

        Parameters
        ----------
        u : float or np.ndarray
            Probability levels in [0, 1].

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        u, scalar = _to_array(u)
        result = self.a + self._width * u
        return _ret(result, scalar)

    def integrated_quantile(self, p: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate U(p) = a * p + (b - a) * p^2 / 2  (exact closed form).

        Parameters
        ----------
        p : float or np.ndarray
            Upper integration limits in [0, 1].

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        p, scalar = _to_array(p)
        result = self.a * p + self._width * p ** 2 / 2.0
        return _ret(result, scalar)

    def density(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the density f(x) = 1 / (b - a) on [a, b], 0 elsewhere.

        Parameters
        ----------
        x : float or np.ndarray

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        x, scalar = _to_array(x)
        result = np.where((x >= self.a) & (x <= self.b), 1.0 / self._width, 0.0)
        return _ret(result, scalar)

    def mean(self) -> float:
        """Return the mean (a + b) / 2.

        Returns
        -------
        float
        """
        return (self.a + self.b) / 2.0


# ---------------------------------------------------------------------------
# UniformUnionDistribution
# ---------------------------------------------------------------------------

class UniformUnionDistribution:
    """Uniform distribution on the union of finitely many disjoint intervals.

    The density is the constant 1 / L on each sub-interval (a_i, b_i) and
    zero in the gaps, where L = sum_i (b_i - a_i) is the total length.

    Parameters
    ----------
    intervals : list of (float, float)
        Non-overlapping sub-intervals (a_i, b_i) sorted left to right with
        a_i < b_i and b_i <= a_{i+1}.

    Attributes
    ----------
    intervals : list of (float, float)
        The sub-intervals as stored.
    lengths : np.ndarray
        Lengths b_i - a_i of each sub-interval.
    L : float
        Total mass = sum of sub-interval lengths.
    """

    def __init__(self, intervals: List[Tuple[float, float]]) -> None:
        self.intervals: List[Tuple[float, float]] = [
            (float(a), float(b)) for a, b in intervals
        ]
        self.lengths: np.ndarray = np.array([b - a for a, b in self.intervals])
        self.L: float = float(np.sum(self.lengths))

        # _cum_lengths[i] = sum of lengths of intervals 0 ... i-1
        # so _cum_lengths[0] = 0,  _cum_lengths[n] = L
        self._cum_lengths: np.ndarray = np.concatenate(
            [[0.0], np.cumsum(self.lengths)]
        )
        # Probability breakpoints in [0, 1]
        self._u_breaks: np.ndarray = self._cum_lengths / self.L

        # Pre-compute U at each probability breakpoint.
        # On segment i: Q(u) = a_i + u*L - C_i  where C_i = _cum_lengths[i].
        # int_{u_l}^{u_r} Q(u) du = (a_i - C_i)*(u_r - u_l) + L/2*(u_r^2 - u_l^2)
        n = len(self.intervals)
        self._U_vals: np.ndarray = np.zeros(n + 1)
        for i in range(n):
            a_i = self.intervals[i][0]
            u_l, u_r = self._u_breaks[i], self._u_breaks[i + 1]
            C_i = self._cum_lengths[i]
            self._U_vals[i + 1] = (
                self._U_vals[i]
                + (a_i - C_i) * (u_r - u_l)
                + self.L / 2.0 * (u_r ** 2 - u_l ** 2)
            )

    # ------------------------------------------------------------------

    def cdf(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the CDF F(x).

        F is piecewise linear with slope 1/L inside each sub-interval and
        constant in every gap.

        Parameters
        ----------
        x : float or np.ndarray

        Returns
        -------
        float or np.ndarray
            Values in [0, 1], same shape as input.
        """
        x, scalar = _to_array(x)
        result = np.zeros(x.shape)
        for i, (a, b) in enumerate(self.intervals):
            mask_in = (x >= a) & (x <= b)
            result = np.where(
                mask_in,
                (self._cum_lengths[i] + x - a) / self.L,
                result,
            )
            # Every x strictly to the right of interval i gets at least C_{i+1}/L
            result = np.where(x > b, self._cum_lengths[i + 1] / self.L, result)
        result = np.clip(result, 0.0, 1.0)
        return _ret(result, scalar)

    def quantile(self, u: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the quantile Q(u).

        Q is piecewise linear:
            Q(u) = a_i + u * L - C_i   for u in (u_breaks[i], u_breaks[i+1]].

        Parameters
        ----------
        u : float or np.ndarray
            Probability levels in [0, 1].

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        u, scalar = _to_array(u)
        u_clipped = np.clip(u, 0.0, 1.0)
        # Segment index via searchsorted on the interior breakpoints
        idx = np.searchsorted(self._u_breaks[1:], u_clipped, side='left')
        idx = np.clip(idx, 0, len(self.intervals) - 1)
        a_vals = np.array([self.intervals[j][0] for j in idx])
        b_vals = np.array([self.intervals[j][1] for j in idx])
        C_vals = self._cum_lengths[idx]
        result = np.clip(a_vals + u_clipped * self.L - C_vals, a_vals, b_vals)
        return _ret(result, scalar)

    def integrated_quantile(self, p: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate U(p) = integral_0^p Q(u) du  (piecewise quadratic, exact).

        On each probability segment [u_breaks[i], u_breaks[i+1]]:
            U(p) = U(u_breaks[i])
                   + (a_i - C_i) * (p - u_breaks[i])
                   + L / 2 * (p^2 - u_breaks[i]^2).

        Parameters
        ----------
        p : float or np.ndarray
            Upper integration limits in [0, 1].

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        p, scalar = _to_array(p)
        p_clipped = np.clip(p, 0.0, 1.0)
        idx = np.searchsorted(self._u_breaks[1:], p_clipped, side='left')
        idx = np.clip(idx, 0, len(self.intervals) - 1)
        a_vals = np.array([self.intervals[j][0] for j in idx])
        C_vals = self._cum_lengths[idx]
        u_left = self._u_breaks[idx]
        U_left = self._U_vals[idx]
        result = (
            U_left
            + (a_vals - C_vals) * (p_clipped - u_left)
            + self.L / 2.0 * (p_clipped ** 2 - u_left ** 2)
        )
        return _ret(result, scalar)

    def density(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the density f(x) = 1 / L on each sub-interval, 0 in gaps.

        Parameters
        ----------
        x : float or np.ndarray

        Returns
        -------
        float or np.ndarray
            Same shape as input.
        """
        x, scalar = _to_array(x)
        result = np.zeros(x.shape)
        for a, b in self.intervals:
            result = np.where((x >= a) & (x <= b), 1.0 / self.L, result)
        return _ret(result, scalar)

    def mean(self) -> float:
        """Return the mean E[X] = sum_i (a_i + b_i) / 2 * L_i / L.

        Returns
        -------
        float
        """
        interval_means = np.array([(a + b) / 2.0 for a, b in self.intervals])
        return float(np.dot(interval_means, self.lengths / self.L))
