"""Triple-store construction for MOMO-GP, with sparsity-aware negative subsampling.

The model trains on a "triple store": each observation is ``(i, j, y_ij)`` with
``i`` a sample (cell) index and ``j`` a feature index. Historically the store was
built from the *z-scored* dense matrix (see CLAUDE.md gotcha #1), which materialises
all ``I*J`` entries and processes ~90-98% structural zeros at full cost.

``build_triple_store`` instead keeps the **raw zero-mask** so structural zeros can be
subsampled. With ``neg_sample_ratio=K`` it keeps every non-zero and ``K`` sampled
zeros per non-zero, attaching a per-triple weight ``total_zeros / sampled_zeros`` to
the sampled zeros so the likelihood term remains an **unbiased** estimate of the full
dense ELBO data term (derivation in the PR discussion / CLAUDE.md).

Equivalence target: a structural-zero triple keeps its *z-scored* value (e.g.
``-mean_j / std_j``), reproducing the current objective — "negative" means
*originally-zero entry*, not ``y == 0``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # optional: support scipy sparse inputs without a hard dependency
    from scipy import sparse as _sp
except Exception:  # pragma: no cover
    _sp = None


def _nonzero_indices(raw) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of structurally non-zero entries of ``raw``."""
    if _sp is not None and _sp.issparse(raw):
        coo = raw.tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64)
    raw = np.asarray(raw)
    r, c = np.nonzero(raw)
    return r.astype(np.int64), c.astype(np.int64)


def _sample_zero_linear_ids(
    nz_lin: np.ndarray, I: int, F: int, n_zeros_total: int, n_needed: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly sample ``n_needed`` linear indices of zero entries WITHOUT enumerating
    all zeros (memory-safe for huge, very sparse matrices).

    Uses batched rejection sampling against the non-zero set; acceptance ~= sparsity,
    so this is efficient precisely when the matrix is sparse.

    :param nz_lin: sorted int64 array of linear indices ``i*F + j`` of non-zeros.
    :param n_zeros_total: number of zero entries ``I*F - len(nz_lin)``.
    :returns: int64 array of unique zero linear indices, length ``min(n_needed, n_zeros_total)``.
    """
    n_needed = int(min(n_needed, n_zeros_total))
    if n_needed <= 0:
        return np.empty(0, dtype=np.int64)
    total = I * F
    nz_set = set(nz_lin.tolist())
    chosen: set = set()
    # Oversample per round to amortise rejections; cap rounds defensively.
    while len(chosen) < n_needed:
        draw = rng.integers(0, total, size=2 * (n_needed - len(chosen)) + 16, dtype=np.int64)
        for v in draw.tolist():
            if v not in nz_set and v not in chosen:
                chosen.add(v)
                if len(chosen) == n_needed:
                    break
    return np.fromiter(chosen, dtype=np.int64, count=n_needed)


def build_triple_store(
    raw: "np.ndarray",
    values: Optional["np.ndarray"] = None,
    neg_sample_ratio: Optional[float] = None,
    seed: int = 0,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a ``(X, Y, W)`` triple store, optionally with negative subsampling.

    :param raw: ``(I, F)`` matrix (dense or scipy-sparse) used ONLY for the zero-mask
        (an entry is a "structural zero" iff ``raw[i, j] == 0``).
    :param values: ``(I, F)`` dense matrix of *targets* (e.g. z-scored ``raw``). If
        ``None``, ``raw`` is used as the values too.
    :param neg_sample_ratio: ``K``. If ``None`` -> dense store (all ``I*F`` entries,
        weight 1; reproduces the legacy behaviour). If set -> keep all non-zeros and
        ``K`` sampled zeros per non-zero, weighting zeros by ``n_zeros / n_sampled``.
    :param seed: RNG seed (reproducible sampling).
    :param shuffle: shuffle the returned rows.
    :returns:
        ``X`` ``(n, 2)`` int64 0-based ``(i, j)`` indices (caller adds 1 for the
        mask_zero embedding convention), ``Y`` ``(n,)`` float64 targets, ``W`` ``(n,)``
        float64 per-triple weights.
    """
    if values is None:
        values = raw
    values = np.asarray(values)
    I, F = values.shape

    if neg_sample_ratio is None:
        # Legacy dense store: every entry, unit weight.
        ii, jj = np.meshgrid(np.arange(I), np.arange(F), indexing="ij")
        X = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.int64)
        Y = values.reshape(-1).astype(np.float64)
        W = np.ones(Y.shape[0], dtype=np.float64)
    else:
        if neg_sample_ratio < 0:
            raise ValueError("neg_sample_ratio must be >= 0")
        rng = np.random.default_rng(seed)
        nz_i, nz_j = _nonzero_indices(raw)
        n_nonzero = nz_i.shape[0]
        n_zeros_total = I * F - n_nonzero
        n_needed = int(round(neg_sample_ratio * n_nonzero))

        nz_lin = np.sort(nz_i * np.int64(F) + nz_j)
        z_lin = _sample_zero_linear_ids(nz_lin, I, F, n_zeros_total, n_needed, rng)
        z_i, z_j = np.divmod(z_lin, np.int64(F))
        n_sampled = z_i.shape[0]

        X = np.concatenate(
            [np.stack([nz_i, nz_j], 1), np.stack([z_i, z_j], 1)], axis=0
        ).astype(np.int64)
        Y = np.concatenate(
            [values[nz_i, nz_j], values[z_i, z_j]]
        ).astype(np.float64)
        # Unbiased reweighting: each kept zero stands in for n_zeros_total / n_sampled zeros.
        zero_w = (n_zeros_total / n_sampled) if n_sampled > 0 else 0.0
        W = np.concatenate(
            [np.ones(n_nonzero), np.full(n_sampled, zero_w)]
        ).astype(np.float64)

    if shuffle:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(X.shape[0])
        X, Y, W = X[perm], Y[perm], W[perm]
    return X, Y, W
