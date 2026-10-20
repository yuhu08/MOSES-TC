"""Capped conversion from calibrated weights to inclusion probabilities."""

from __future__ import annotations

import numpy as np

def scale_to_sum_with_cap(
    base: np.ndarray,
    target_sum: int,
    *,
    cap: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> np.ndarray:
    """Return pi = clip(c * base, 0, cap) with sum(pi) ~= target_sum.

    Uses a bisection search on the scaling factor c. If `target_sum` exceeds the
    number of positive entries in `base`, the requested sum is infeasible
    without assigning mass to zero-base tracks; in that case this returns the
    capped support indicator (sum < target).
    """
    base = np.asarray(base, dtype=np.float64)
    base = np.clip(base, 0.0, None)
    target = float(int(target_sum))
    if target_sum <= 0 or base.size == 0:
        return np.zeros_like(base, dtype=np.float64)
    support = int(np.sum(base > 0))
    if support == 0:
        return np.zeros_like(base, dtype=np.float64)
    if target >= float(support):
        return (base > 0).astype(np.float64) * float(cap)

    def capped_sum(scale: float) -> float:
        return float(np.sum(np.minimum(cap, scale * base)))

    low = 0.0
    high = 1.0
    while capped_sum(high) < target and high < 1e12:
        high *= 2.0

    for _ in range(int(max_iter)):
        mid = 0.5 * (low + high)
        s = capped_sum(mid)
        if abs(s - target) <= tol * max(1.0, target):
            low = high = mid
            break
        if s < target:
            low = mid
        else:
            high = mid

    scale = 0.5 * (low + high)
    pi = np.minimum(cap, scale * base)
    diff = target - float(np.sum(pi))
    if abs(diff) > 1e-6:
        free = (pi > 0.0) & (pi < cap)
        if np.any(free):
            free_sum = float(np.sum(pi[free]))
            if free_sum > 0:
                pi[free] *= (free_sum + diff) / free_sum
                pi = np.clip(pi, 0.0, cap)
    return pi
