"""JAX PDHG weight calibration, retained from the historical workflow."""
from __future__ import annotations

from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .ot_utils import CdfGraphSystem

def _jax_estimate_operator_norm(cdf, n_iter: int = 12) -> float:
    """JAX version of the operator norm estimate to keep everything on device."""
    n = cdf["n_tracks"]
    if n == 0 or cdf["n_rows"] == 0:
        return 0.0
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n,))
    x = x / jnp.linalg.norm(x)

    def body(_, vec):
        y = cdf["apply_A"](vec)
        y = cdf["row_weights"] * y
        y = y / jnp.linalg.norm(y)
        x_new = cdf["adjoint"](cdf["row_weights"] * y)
        x_new = x_new / jnp.linalg.norm(x_new)
        return x_new

    x = lax.fori_loop(0, max(1, n_iter), body, x)
    y = cdf["apply_A"](x)
    y = cdf["row_weights"] * y
    norm_est = jnp.linalg.norm(y)
    return float(norm_est)


def _project_to_bounded_simplex_jax(v: jnp.ndarray, target_sum: float, upper: float, max_iter: int = 50, tol: float = 1e-6) -> jnp.ndarray:
    """Bisection projection onto {w | sum w = target_sum, 0 <= w <= upper} in JAX."""
    if upper <= 0:
        raise ValueError("Upper bound must be positive.")
    n = v.size
    if target_sum > n * upper + 1e-6:
        raise ValueError("Target sum exceeds feasible region.")
    theta_low = jnp.min(v) - upper
    theta_high = jnp.max(v)

    def cond_fun(state):
        low, high, _ = state
        theta = 0.5 * (low + high)
        clipped = jnp.clip(v - theta, 0.0, upper)
        gap = jnp.sum(clipped) - target_sum
        return jnp.abs(gap) > tol * jnp.maximum(1.0, target_sum)

    def body_fun(state):
        low, high, _ = state
        theta = 0.5 * (low + high)
        clipped = jnp.clip(v - theta, 0.0, upper)
        gap = jnp.sum(clipped) - target_sum
        low = jnp.where(gap > 0, theta, low)
        high = jnp.where(gap > 0, high, theta)
        return (low, high, _ + 1)

    low, high, _ = lax.while_loop(cond_fun, body_fun, (theta_low, theta_high, 0))
    theta = 0.5 * (low + high)
    return jnp.clip(v - theta, 0.0, upper)


def _make_jax_cdf(cdf_system: CdfGraphSystem) -> dict:
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


def _jax_pdhg_solve(
    cdf_system: CdfGraphSystem,
    lambda_reg: float,
    target_sum: float,
    w_max: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, List[float], int, str]:
    """
    JAX-accelerated PDHG loop. Keeps all heavy ops on device via JIT and
    scatter/gather primitives.
    """
    cdf = _make_jax_cdf(cdf_system)
    norm_est = _jax_estimate_operator_norm(cdf)
    if norm_est <= 0:
        raise ValueError("Non-positive operator norm estimate.")
    tau = 1.0 / (norm_est + 1e-8)
    sigma = tau
    theta = 1.0

    def pdhg_step(state, _):
        w, w_bar, y = state
        Kw_bar = cdf["apply_A"](w_bar)
        Kw_scaled = cdf["row_weights"] * (Kw_bar - cdf["b"])
        y_new = jnp.clip(y + sigma * Kw_scaled, -1.0, 1.0)
        grad = cdf["adjoint"](cdf["row_weights"] * y_new)
        v = (w - tau * grad) / (1.0 + 2.0 * tau * lambda_reg)
        w_new = _project_to_bounded_simplex_jax(v, target_sum=target_sum, upper=w_max)
        w_bar_new = w_new + theta * (w_new - w)
        return (w_new, w_bar_new, y_new), None

    w0 = jnp.full(cdf["n_tracks"], target_sum / cdf["n_tracks"], dtype=jnp.float32)
    wbar0 = w0
    y0 = jnp.zeros(cdf["n_rows"], dtype=jnp.float32)
    (w_final, _, _), _ = lax.scan(pdhg_step, (w0, wbar0, y0), xs=None, length=max_iter)
    residual = cdf["apply_A"](w_final) - cdf["b"]
    obj = lambda_reg * jnp.dot(w_final, w_final) + jnp.dot(
        cdf["row_weights"], jnp.abs(residual)
    )
    history = [float(obj)]
    return np.asarray(w_final, dtype=np.float32), history, 0, "pdhg_jax"
