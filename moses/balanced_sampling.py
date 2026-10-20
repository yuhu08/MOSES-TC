"""Balanced sampling utilities (approximate cube method)."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy.linalg import null_space


def _fast_null_direction(H: np.ndarray, tol: float = 1e-8) -> Optional[np.ndarray]:
    """Compute a single null-space direction for an (m x (m+1)) matrix.

    For H with one more column than rows, we can avoid an SVD by fixing one
    component and solving a square system. Falls back to least squares if
    needed. Returns None if a stable direction is not found.
    """
    m, n_cols = H.shape
    if n_cols != m + 1:
        return None
    # Try a few pivot columns to avoid singular submatrices.
    for pivot in range(n_cols):
        A = np.delete(H, pivot, axis=1)
        b = -H[:, pivot]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        direction = np.insert(x, pivot, 1.0)
        if not np.all(np.isfinite(direction)):
            continue
        resid = H @ direction
        if np.linalg.norm(resid) <= tol * max(1.0, np.linalg.norm(H)):
            # Normalize for numerical stability.
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            return direction
    return None


def _qr_null_direction(H: np.ndarray) -> Optional[np.ndarray]:
    """Compute a null-space direction via QR on H.T.

    This is significantly faster than SVD and yields a stable, balanced direction
    that works well when many balance constraints are present.
    """
    m, n_cols = H.shape
    if n_cols != m + 1:
        return None
    try:
        Q, _ = np.linalg.qr(H.T, mode="complete")
    except Exception:
        return None
    direction = Q[:, -1]
    return direction


def _adjust_sample_size(selection: np.ndarray, target: int, rng: np.random.Generator, probs: np.ndarray) -> None:
    """Ensure the sample has exactly target elements."""
    current = int(selection.sum())
    if current == target:
        return
    if current < target:
        shortfall = target - current
        candidates = np.flatnonzero(~selection)
        if candidates.size == 0:
            return
        weights = probs[candidates]
        positive = weights > 0
        if not np.any(positive) or positive.sum() < shortfall:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        chosen = rng.choice(candidates, size=shortfall, replace=False, p=weights)
        selection[chosen] = True
    else:
        excess = current - target
        candidates = np.flatnonzero(selection)
        if candidates.size == 0:
            return
        weights = probs[candidates]
        positive = weights > 0
        if not np.any(positive) or positive.sum() < excess:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        chosen = rng.choice(candidates, size=excess, replace=False, p=weights)
        selection[chosen] = False


def _landing_phase(
    pi: np.ndarray,
    selection: np.ndarray,
    remaining: np.ndarray,
    X: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    probs: np.ndarray,
) -> None:
    """Brute-force landing phase for small fractional sets."""
    if remaining.size == 0:
        return
    max_enumerate = 16
    if remaining.size <= max_enumerate:
        base_vector = X @ selection.astype(np.float64)
        X_rem = X[:, remaining]
        best_loss = np.inf
        best_mask = None
        for mask in itertools.product([0, 1], repeat=remaining.size):
            mask_arr = np.asarray(mask, dtype=np.float64)
            candidate = base_vector + X_rem @ mask_arr
            loss = np.linalg.norm(candidate - target)
            if loss < best_loss:
                best_loss = loss
                best_mask = mask_arr
        if best_mask is not None:
            selection[remaining] = best_mask.astype(bool)
            return
    # Fallback: randomized rounding with probabilities
    selection[remaining] = rng.binomial(1, np.clip(pi[remaining], 0.0, 1.0)).astype(bool)
    target_count = int(round(probs.sum()))
    _adjust_sample_size(selection, target_count, rng, probs)


def balanced_sample(
    probabilities: np.ndarray,
    balance_matrix: Optional[np.ndarray],
    random_state: int = 0,
) -> np.ndarray:
    """
    Approximate cube method sampler.

    Parameters
    ----------
    probabilities : np.ndarray
        Inclusion probabilities (sum equals desired sample size).
    balance_matrix : np.ndarray or None
        Rows = features, columns = tracks.
    random_state : int
        RNG seed.

    Returns
    -------
    selection : np.ndarray (bool)
        Indicator array of sampled tracks.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    n = probs.size
    rng = np.random.default_rng(random_state)
    target_size = int(round(float(probs.sum())))
    pi = probs.copy()
    tol = 1e-12
    if balance_matrix is None:
        selection = rng.binomial(1, np.clip(pi, 0.0, 1.0)).astype(bool)
        _adjust_sample_size(selection, target_size, rng, probs)
        return selection

    X = np.vstack([balance_matrix, np.ones(n, dtype=np.float64)])
    target_vector = X @ probs
    m = X.shape[0]

    while True:
        fractional = np.where((pi > tol) & (pi < 1.0 - tol))[0]
        if fractional.size == 0 or fractional.size <= m:
            break
        chosen = rng.choice(fractional, size=m + 1, replace=False)
        H = X[:, chosen]
        # For small feature sets, the pivot-based direction is fastest and stable.
        # For large feature sets, QR gives a much better rounding direction at a
        # modest cost, avoiding expensive SVDs.
        if m <= 32:
            direction = _fast_null_direction(H, tol=1e-8)
            if direction is None:
                direction = _qr_null_direction(H)
        else:
            direction = _qr_null_direction(H)
        if direction is None:
            null = null_space(H)
            if null.size == 0:
                continue
            direction = null[:, 0]
        positive = direction > tol
        negative = direction < -tol
        if not positive.any() or not negative.any():
            continue
        step_plus = np.min((1.0 - pi[chosen][positive]) / direction[positive])
        step_minus = np.min(pi[chosen][negative] / (-direction[negative]))
        prob = step_minus / (step_minus + step_plus)
        if rng.random() < prob:
            pi[chosen] = np.clip(pi[chosen] + step_plus * direction, 0.0, 1.0)
        else:
            pi[chosen] = np.clip(pi[chosen] - step_minus * direction, 0.0, 1.0)

    selection = np.zeros(n, dtype=bool)
    selection[pi >= 1.0 - tol] = True
    fractional = np.where((pi > tol) & (pi < 1.0 - tol))[0]
    _landing_phase(pi, selection, fractional, X, target_vector, rng, probs)
    _adjust_sample_size(selection, target_size, rng, probs)
    return selection
