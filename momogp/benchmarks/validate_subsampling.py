"""Validate negative subsampling: the reweighted ELBO data-term is an unbiased
estimate of the full dense data-term, on the SAME model parameters.

Implements the sparsity-specific check from CLAUDE.md ("ELBO estimates with and
without negative subsampling, on the same parameters, should agree within Monte-Carlo
noise — run ~20 evaluations, check 95% CI overlap").

Method (single-view, to isolate the data term cleanly):
  * Build a sparse synthetic matrix, z-score it.
  * Build the model ONCE with ``emb_reg=0``; do NOT train. At initialisation GPflux's
    q(u) sits at the prior, so the KL term is 0 and the embedding-reg is 0 — hence the
    Keras ``loss`` equals the mean per-triple data loss exactly.
  * Full data-term  = loss(dense store) * |dense store|.
  * Estimate (per draw) = loss(subsampled store, sample_weight=W) * |subsampled store|,
    which equals sum_store w * l  (an unbiased estimate of the full sum, see derivation).
  * Compare the dense value against the mean +/- 95% CI over N subsampled draws.
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile


def _stub_nvidia_smi():
    d = tempfile.mkdtemp(prefix="momogp_stub_")
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as fh:
        fh.write('#!/usr/bin/env bash\necho "GPU"\necho "        Used : 0 MiB"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=int, default=300)
    ap.add_argument("--F", type=int, default=400)
    ap.add_argument("--density", type=float, default=0.9, help="structural-zero fraction")
    ap.add_argument("--neg_sample_ratio", type=float, default=5.0)
    ap.add_argument("--n_draws", type=int, default=20)
    args = ap.parse_args()

    _stub_nvidia_smi()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import numpy as np
    np.random.seed(0)
    import tensorflow as tf
    tf.random.set_seed(0)
    from momogp import GPD
    from momogp.data import build_triple_store

    I, F = args.I, args.F
    rng = np.random.default_rng(0)
    raw = (rng.standard_normal((I, 3)) @ rng.standard_normal((3, F))) + 5.0
    raw[rng.random((I, F)) < args.density] = 0.0
    z = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)

    Xd, Yd, Wd = build_triple_store(raw, z, neg_sample_ratio=None)  # dense
    n_dense = len(Yd)

    # Build the model once, emb_reg=0, no training -> loss == mean per-triple data loss.
    gp = GPD(I=I, J=F, K=None, M1=128, M2=128, emb_sizes=[2, 2, 2],
             batch_size=4000, obs_mean1=float(Yd.mean()), obs_mean2=None,
             emb_reg=0.0, lr=1e-2, save_path=tempfile.mkdtemp())
    gp.build(kernels=["RBF"])
    gp.num_data1 = n_dense
    gp.set_inducing_points((Xd + 1).astype(int))
    gp.make_model()

    def data_term(X, Y, W):
        loss = gp.model.evaluate([(X[:, 0] + 1), (X[:, 1] + 1)], Y,
                                 sample_weight=W, batch_size=8000,
                                 return_dict=True, verbose=0)["loss"]
        return loss * len(Y)

    full = data_term(Xd, Yd, Wd)

    ests = []
    for s in range(args.n_draws):
        X, Y, W = build_triple_store(raw, z, neg_sample_ratio=args.neg_sample_ratio, seed=100 + s)
        ests.append(data_term(X, Y, W))
    ests = np.array(ests)
    mean, std = ests.mean(), ests.std(ddof=1)
    se = std / np.sqrt(len(ests))
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
    rel = (mean - full) / abs(full)

    print(f"\n=== ELBO data-term equivalence (single-view) ===")
    print(f"density={args.density}  K={args.neg_sample_ratio}  draws={args.n_draws}")
    print(f"dense store size      : {n_dense}")
    print(f"subsampled store size : {len(Y)}  ({100*len(Y)/n_dense:.0f}% of dense)")
    print(f"full data-term (exact): {full:.3f}")
    print(f"subsampled mean       : {mean:.3f}  (std {std:.3f}, se {se:.3f})")
    print(f"95% CI                : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"relative bias         : {100*rel:+.3f}%")
    ok = ci_lo <= full <= ci_hi
    print(f"RESULT                : {'PASS - CI brackets the exact value' if ok else 'CHECK - CI does not bracket'}")


if __name__ == "__main__":
    main()
