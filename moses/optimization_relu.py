"""PDHG with the original one-sided squared tail penalty, in count units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except Exception as exc:  # pragma: no cover
    raise ImportError("MOSES requires JAX.") from exc

from .ot_utils import CdfGraphSystem


@dataclass(frozen=True)
class TailFeatureSystem:
    """Sparse (possibly weighted) feature system Ew for tail exceedance constraints."""

    n_tracks: int
    n_rows: int
    edge_row_ids: np.ndarray  # (n_edges,), int32
    edge_track_ids: np.ndarray  # (n_edges,), int32
    tau: np.ndarray  # (n_rows,), float64 (observed counts in same units as w)
    edge_values: np.ndarray | None = None  # (n_edges,), float32; defaults to 1
    row_meta: Optional[List[dict]] = None


def _project_to_bounded_simplex_jax(
    v: jnp.ndarray,
    *,
    target_sum: float,
    upper: float,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Bisection projection onto {w | sum w = target_sum, 0 <= w <= upper} in JAX."""
    if upper <= 0:
        raise ValueError("Upper bound must be positive.")
    n = v.size
    if target_sum > float(n) * float(upper) + 1e-6:
        raise ValueError("Target sum exceeds feasible region.")
    theta_low = jnp.min(v) - upper
    theta_high = jnp.max(v)

    def cond_fun(state):
        low, high, it = state
        theta = 0.5 * (low + high)
        clipped = jnp.clip(v - theta, 0.0, upper)
        gap = jnp.sum(clipped) - target_sum
        return jnp.logical_and(it < max_iter, jnp.abs(gap) > tol * jnp.maximum(1.0, target_sum))

    def body_fun(state):
        low, high, it = state
        theta = 0.5 * (low + high)
        clipped = jnp.clip(v - theta, 0.0, upper)
        gap = jnp.sum(clipped) - target_sum
        low = jnp.where(gap > 0, theta, low)
        high = jnp.where(gap > 0, high, theta)
        return (low, high, it + 1)

    low, high, _ = lax.while_loop(cond_fun, body_fun, (theta_low, theta_high, 0))
    theta = 0.5 * (low + high)
    return jnp.clip(v - theta, 0.0, upper)


def _make_jax_cdf(cdf_system: CdfGraphSystem) -> Dict[str, object]:
    """Convert CdfGraphSystem to JAX-friendly callable ops."""
    edge_point_ids = jnp.asarray(cdf_system.edge_point_ids, dtype=jnp.int32)
    edge_track_ids = jnp.asarray(cdf_system.edge_track_ids, dtype=jnp.int32)
    edge_cdf_values = jnp.asarray(cdf_system.edge_cdf_values)
    row_weights = jnp.asarray(cdf_system.row_weights)
    b = jnp.asarray(cdf_system.b)
    n_points = int(cdf_system.point_ids.size)
    n_bins = int(cdf_system.n_bins)
    n_rows = int(cdf_system.n_rows)
    n_tracks = int(cdf_system.n_tracks)

    def apply_A(weights: jnp.ndarray) -> jnp.ndarray:
        edge_vals = weights[edge_track_ids, None] * edge_cdf_values  # (n_edges, n_bins)
        accum = jnp.zeros((n_points, n_bins), dtype=edge_vals.dtype).at[edge_point_ids].add(edge_vals)
        return accum.reshape((n_rows,))

    def adjoint(dual: jnp.ndarray) -> jnp.ndarray:
        dual_mat = dual.reshape((n_points, n_bins))
        edge_dual = dual_mat[edge_point_ids]
        edge_grad = jnp.sum(edge_dual * edge_cdf_values, axis=1)
        grad = jnp.zeros((n_tracks,), dtype=edge_grad.dtype).at[edge_track_ids].add(edge_grad)
        return grad

    return {
        "apply_A": apply_A,
        "adjoint": adjoint,
        "row_weights": row_weights,
        "b": b,
        "n_rows": n_rows,
        "n_tracks": n_tracks,
    }


def _jax_estimate_operator_norm(
    apply_A: Callable[[jnp.ndarray], jnp.ndarray],
    adjoint: Callable[[jnp.ndarray], jnp.ndarray],
    row_weights: jnp.ndarray,
    n_tracks: int,
    n_rows: int,
    n_iter: int = 12,
) -> float:
    """Power iteration to estimate ||diag(row_weights) A||_2 in JAX."""
    if n_tracks <= 0 or n_rows <= 0:
        return 0.0
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_tracks,), dtype=jnp.float32)
    x = x / jnp.linalg.norm(x)

    def body(_, vec):
        y = apply_A(vec)
        y = row_weights * y
        y = y / jnp.linalg.norm(y)
        x_new = adjoint(row_weights * y)
        x_new = x_new / jnp.linalg.norm(x_new)
        return x_new

    x = lax.fori_loop(0, max(1, int(n_iter)), body, x)
    y = apply_A(x)
    y = row_weights * y
    return float(jnp.linalg.norm(y))


def _make_jax_tail(tail: TailFeatureSystem) -> Dict[str, object]:
    edge_row_ids = jnp.asarray(tail.edge_row_ids, dtype=jnp.int32)
    edge_track_ids = jnp.asarray(tail.edge_track_ids, dtype=jnp.int32)
    if tail.edge_values is None:
        edge_values = jnp.ones_like(edge_track_ids, dtype=jnp.float32)
    else:
        edge_values = jnp.asarray(tail.edge_values, dtype=jnp.float32)
    tau = jnp.asarray(tail.tau, dtype=jnp.float32)
    n_rows = int(tail.n_rows)
    n_tracks = int(tail.n_tracks)

    def apply_E(weights: jnp.ndarray) -> jnp.ndarray:
        if edge_track_ids.size == 0 or n_rows == 0:
            return jnp.zeros((n_rows,), dtype=weights.dtype)
        return jnp.zeros((n_rows,), dtype=weights.dtype).at[edge_row_ids].add(
            weights[edge_track_ids] * edge_values
        )

    def adjoint(vec: jnp.ndarray) -> jnp.ndarray:
        if edge_track_ids.size == 0 or n_tracks == 0:
            return jnp.zeros((n_tracks,), dtype=vec.dtype)
        return jnp.zeros((n_tracks,), dtype=vec.dtype).at[edge_track_ids].add(vec[edge_row_ids] * edge_values)

    return {
        "apply_E": apply_E,
        "adjoint": adjoint,
        "tau": tau,
        "n_rows": n_rows,
        "n_tracks": n_tracks,
    }


def _jax_estimate_E_norm(tail: Dict[str, object], n_iter: int = 12) -> float:
    n_tracks = int(tail["n_tracks"])  # type: ignore[index]
    n_rows = int(tail["n_rows"])  # type: ignore[index]
    if n_tracks <= 0 or n_rows <= 0:
        return 0.0
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (n_tracks,), dtype=jnp.float32)
    x = x / jnp.linalg.norm(x)
    apply_E = tail["apply_E"]  # type: ignore[assignment]
    adjoint = tail["adjoint"]  # type: ignore[assignment]

    def body(_, vec):
        y = apply_E(vec)
        y = y / jnp.linalg.norm(y)
        x_new = adjoint(y)
        x_new = x_new / jnp.linalg.norm(x_new)
        return x_new

    x = lax.fori_loop(0, max(1, int(n_iter)), body, x)
    y = apply_E(x)
    return float(jnp.linalg.norm(y))


def solve_pdhg_relu(
    *,
    cdf_system: CdfGraphSystem,
    lambda_reg: float,
    lambda_tail: float,
    target_sum: float,
    w_max: float,
    max_iter: int,
    tol: float,
    tail_system: Optional[TailFeatureSystem],
    w_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float], int, str]:
    """Solve the OT+ReLU-tail problem with JAX PDHG."""
    if target_sum <= 0:
        raise ValueError("target_sum must be positive.")
    cdf = _make_jax_cdf(cdf_system)
    apply_A = cdf["apply_A"]  # type: ignore[assignment]
    adjoint = cdf["adjoint"]  # type: ignore[assignment]
    row_weights = cdf["row_weights"]  # type: ignore[assignment]
    b = cdf["b"]  # type: ignore[assignment]
    n_rows = int(cdf["n_rows"])  # type: ignore[index]
    n_tracks = int(cdf["n_tracks"])  # type: ignore[index]

    norm_K = _jax_estimate_operator_norm(
        apply_A=apply_A,
        adjoint=adjoint,
        row_weights=row_weights,
        n_tracks=n_tracks,
        n_rows=n_rows,
        n_iter=12,
    )
    if not np.isfinite(norm_K) or norm_K <= 0:
        raise ValueError("Non-positive operator norm estimate for OT system.")

    use_tail = tail_system is not None and float(lambda_tail) > 0 and int(tail_system.n_rows) > 0
    if use_tail:
        tail = _make_jax_tail(tail_system)
        norm_E = _jax_estimate_E_norm(tail, n_iter=10)
        tau_vec = tail["tau"]  # type: ignore[assignment]
        apply_E = tail["apply_E"]  # type: ignore[assignment]
        adjoint_E = tail["adjoint"]  # type: ignore[assignment]
    else:
        tail = None
        norm_E = 0.0
        tau_vec = None
        apply_E = None
        adjoint_E = None

    # We keep the OT L1 part as the "K" operator and handle:
    # - L2 term via its proximal (closed-form)
    # - Tail ReLU^2 term via a forward gradient step (Lipschitz: 2*lambda_tail*||E||^2)
    L_tail = float(2.0 * float(lambda_tail) * float(norm_E) ** 2) if use_tail else 0.0
    step = 1.0 / (float(norm_K) + float(L_tail) + 1e-8)
    tau = float(step)
    sigma = float(step)
    theta = 1.0

    tau_j = jnp.asarray(tau, dtype=jnp.float32)
    sigma_j = jnp.asarray(sigma, dtype=jnp.float32)
    lambda_reg_j = jnp.asarray(float(lambda_reg), dtype=jnp.float32)
    lambda_tail_j = jnp.asarray(float(lambda_tail), dtype=jnp.float32)
    w_max_j = jnp.asarray(float(w_max), dtype=jnp.float32)

    if w_init is None:
        w0 = jnp.full((n_tracks,), float(target_sum) / float(n_tracks), dtype=jnp.float32)
    else:
        w0_np = np.asarray(w_init, dtype=np.float32).reshape(-1)
        if w0_np.size != n_tracks:
            raise ValueError(f"w_init length {w0_np.size} != n_tracks {n_tracks}")
        w0 = jnp.asarray(w0_np, dtype=jnp.float32)
    wbar0 = w0
    y0 = jnp.zeros((n_rows,), dtype=jnp.float32)

    def pdhg_step(state, _):
        w, w_bar, y = state
        Kw_bar = apply_A(w_bar)
        Kw_scaled = row_weights * (Kw_bar - b)
        y_new = jnp.clip(y + sigma_j * Kw_scaled, -1.0, 1.0)
        gradK = adjoint(row_weights * y_new)

        # Tail gradient (forward step). L2 is handled via prox below.
        grad_tail = jnp.zeros_like(w)
        if use_tail:
            Ew = apply_E(w)  # type: ignore[misc]
            r = jnp.maximum(0.0, tau_vec - Ew)  # type: ignore[operator]
            gradE = adjoint_E(r)  # type: ignore[misc]
            grad_tail = -2.0 * lambda_tail_j * gradE

        v = w - tau_j * (gradK + grad_tail)
        # Prox for L2 term (same as core/_jax_pdhg_solve).
        v = v / (1.0 + 2.0 * tau_j * lambda_reg_j)
        w_new = _project_to_bounded_simplex_jax(v, target_sum=float(target_sum), upper=float(w_max_j))
        w_bar_new = w_new + theta * (w_new - w)
        return (w_new, w_bar_new, y_new), None

    (w_final, _, _), _ = lax.scan(pdhg_step, (w0, wbar0, y0), xs=None, length=int(max_iter))

    # Objective components for monitoring.
    resid = apply_A(w_final) - b
    ot_loss = jnp.dot(row_weights, jnp.abs(resid))
    reg = lambda_reg_j * jnp.dot(w_final, w_final)
    tail_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if use_tail:
        Ew = apply_E(w_final)  # type: ignore[misc]
        r = jnp.maximum(0.0, tau_vec - Ew)  # type: ignore[operator]
        tail_loss = lambda_tail_j * jnp.dot(r, r)
    obj = ot_loss + reg + tail_loss
    history = [float(obj)]
    return np.asarray(w_final, dtype=np.float32), history, 0, "pdhg_relu"


def compute_lambda_tail_autoscale(
    *,
    cdf_system: CdfGraphSystem,
    weights_phys: np.ndarray,
    tail_system: TailFeatureSystem,
    match: str = "ot_l1",
) -> float:
    """Pick lambda_tail so the tail term roughly matches the OT mismatch scale.

    Options
    -------
    - match="ot_l1": match total objective magnitudes at the baseline weights.
    - match="ot_l1_under": like "ot_l1", but scales up by the aggregate underestimation
      factor (tau_under_sum / Ew_under_sum) computed at baseline weights.
    - match="per_row": match average per-row magnitudes (accounts for row counts).
    - match="grad_l2": match L2 norms of (sub)gradients at baseline weights.
    """
    w = np.asarray(weights_phys, dtype=np.float64).reshape(-1)
    if w.size != int(cdf_system.n_tracks):
        raise ValueError("weights_phys length mismatch for autoscale.")
    # OT L1 part (same scale as the PDHG objective)
    resid = cdf_system.forward(w)
    ot_l1 = float(np.dot(cdf_system.row_weights.astype(np.float64), np.abs(resid)))
    # Tail gap sum of squares
    tau = np.asarray(tail_system.tau, dtype=np.float64).reshape(-1)
    if tail_system.edge_track_ids.size == 0 or tail_system.n_rows == 0:
        return 0.0
    if tail_system.edge_values is None:
        edge_vals = np.ones_like(tail_system.edge_track_ids, dtype=np.float64)
    else:
        edge_vals = np.asarray(tail_system.edge_values, dtype=np.float64).reshape(-1)
    Ew = np.zeros(int(tail_system.n_rows), dtype=np.float64)
    np.add.at(
        Ew,
        tail_system.edge_row_ids.astype(np.int64),
        edge_vals * w[tail_system.edge_track_ids.astype(np.int64)],
    )
    gap = np.clip(tau - Ew, 0.0, None)
    tail_ss = float(np.dot(gap, gap))
    if tail_ss <= 0 or not np.isfinite(tail_ss):
        return 0.0
    if match == "ot_l1":
        return max(0.0, ot_l1 / tail_ss)
    if match == "ot_l1_under":
        base = max(0.0, ot_l1 / tail_ss)
        under = gap > 0
        tau_u = float(np.sum(tau[under]))
        Ew_u = float(np.sum(Ew[under]))
        if Ew_u > 0 and np.isfinite(tau_u) and np.isfinite(Ew_u):
            return base * max(1.0, tau_u / Ew_u)
        return base
    if match == "per_row":
        n_ot = max(int(cdf_system.n_rows), 1)
        n_tail = max(int(tail_system.n_rows), 1)
        return max(0.0, (ot_l1 / float(n_ot)) / (tail_ss / float(n_tail)))
    if match == "grad_l2":
        # OT subgradient proxy: A^T (row_weights * sign(resid))
        g_ot = cdf_system.adjoint(cdf_system.row_weights.astype(np.float64) * np.sign(resid))
        # Tail gradient (without lambda): 2 E^T ReLU(tau - Ew)
        # We use the same r used in tail_ss (positive gaps only).
        r = gap.astype(np.float64, copy=False)
        g_tail = np.zeros(int(tail_system.n_tracks), dtype=np.float64)
        np.add.at(
            g_tail,
            tail_system.edge_track_ids.astype(np.int64),
            r[tail_system.edge_row_ids.astype(np.int64)] * edge_vals,
        )
        denom = 2.0 * float(np.linalg.norm(g_tail))
        num = float(np.linalg.norm(g_ot))
        if denom <= 0 or not np.isfinite(denom) or not np.isfinite(num):
            return 0.0
        return max(0.0, num / denom)
    raise ValueError(f"Unknown match mode: {match}")
