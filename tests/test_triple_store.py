"""Tests for momogp.data.build_triple_store (negative subsampling).

Pure-NumPy, no TensorFlow — fast. Run with `pytest tests/` or directly:
    python tests/test_triple_store.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from momogp.data import build_triple_store  # noqa: E402


def _sparse_matrix(I=40, F=60, density=0.1, seed=0):
    """Random matrix with ~`density` non-zeros; return (raw, zscored)."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((I, F))
    mask = rng.random((I, F)) < density
    raw[mask] = rng.standard_normal(mask.sum()) + 3.0  # nonzero, away from 0
    z = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)
    return raw, z


def test_dense_store_is_exact():
    raw, z = _sparse_matrix()
    X, Y, W = build_triple_store(raw, z, neg_sample_ratio=None, shuffle=False)
    I, F = z.shape
    assert X.shape == (I * F, 2)
    assert np.allclose(W, 1.0)
    # Y matches the row-major ravel of the value matrix.
    assert np.allclose(Y, z.reshape(-1))


def test_all_nonzeros_kept_and_weights():
    raw, z = _sparse_matrix()
    I, F = z.shape
    n_nonzero = int((raw != 0).sum())
    n_zeros_total = I * F - n_nonzero
    K = 5
    X, Y, W = build_triple_store(raw, z, neg_sample_ratio=K, seed=7, shuffle=False)

    # Every non-zero is present exactly once, with weight 1.
    nz_lin = set((np.argwhere(raw != 0) @ np.array([F, 1])).tolist())
    kept_lin = X @ np.array([F, 1])
    nonzero_weight_lin = set(kept_lin[W == 1.0].tolist())
    assert nz_lin == nonzero_weight_lin, "all non-zeros must be kept with weight 1"

    # Sampled zeros: count == K * n_nonzero, weight == n_zeros_total / n_sampled.
    n_sampled = int((W != 1.0).sum())
    assert n_sampled == min(K * n_nonzero, n_zeros_total)
    expected_w = n_zeros_total / n_sampled
    assert np.allclose(W[W != 1.0], expected_w)

    # Sampled "zeros" are genuinely zero in raw, and unique.
    zero_rows = X[W != 1.0]
    assert np.all(raw[zero_rows[:, 0], zero_rows[:, 1]] == 0)
    assert len(set((zero_rows @ np.array([F, 1])).tolist())) == n_sampled


def test_estimator_is_unbiased():
    """E[ sum_store w * g(i,j) ] == sum_full g(i,j) for an arbitrary per-triple g."""
    raw, z = _sparse_matrix(I=30, F=50, density=0.15, seed=3)
    I, F = z.shape
    rng = np.random.default_rng(123)
    g = rng.standard_normal((I, F))  # arbitrary per-triple "log-likelihood" surrogate
    full_sum = g.sum()

    K = 4
    n_trials = 400
    ests = []
    for s in range(n_trials):
        X, Y, W = build_triple_store(raw, z, neg_sample_ratio=K, seed=1000 + s, shuffle=False)
        ests.append(float((W * g[X[:, 0], X[:, 1]]).sum()))
    mean_est = np.mean(ests)
    se = np.std(ests, ddof=1) / np.sqrt(n_trials)
    # Mean estimate within ~4 standard errors of the true full sum.
    assert abs(mean_est - full_sum) < 4 * se + 1e-9, (
        f"biased: mean={mean_est:.4f} full={full_sum:.4f} se={se:.4f}")


def test_ratio_zero_keeps_only_nonzeros():
    raw, z = _sparse_matrix()
    n_nonzero = int((raw != 0).sum())
    X, Y, W = build_triple_store(raw, z, neg_sample_ratio=0, seed=1, shuffle=False)
    assert X.shape[0] == n_nonzero
    assert np.allclose(W, 1.0)


if __name__ == "__main__":
    test_dense_store_is_exact()
    test_all_nonzeros_kept_and_weights()
    test_estimator_is_unbiased()
    test_ratio_zero_keeps_only_nonzeros()
    print("all tests passed")
