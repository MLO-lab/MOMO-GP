"""Single-view subsampling benchmark: one (density, neg_sample_ratio) config per process.

Runs ONE config and prints machine-readable metrics, so an outer loop can launch each
config in a fresh process (isolated wall-clock + peak RSS). Drives at realistic sparsity
(RNA ~90% zero, ATAC ~98% zero) but scaled down so dense is still feasible on a CPU.

    python momogp/benchmarks/bench_subsampling.py --I 1500 --F 2000 --density 0.9 --ratio 5
    python momogp/benchmarks/bench_subsampling.py --I 1500 --F 2000 --density 0.9   # dense
"""
from __future__ import annotations

import argparse
import os
import resource
import stat
import sys
import tempfile
import time


def _stub_nvidia_smi():
    d = tempfile.mkdtemp(prefix="momogp_stub_")
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as fh:
        fh.write('#!/usr/bin/env bash\necho "GPU"\necho "        Used : 0 MiB"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=int, default=1500)
    ap.add_argument("--F", type=int, default=2000)
    ap.add_argument("--density", type=float, default=0.9, help="structural-zero fraction")
    ap.add_argument("--ratio", type=float, default=None, help="neg_sample_ratio K (omit=dense)")
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--likelihood", default="gaussian",
                    help="gaussian | nb | bernoulli (sets the target domain too)")
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
    # Count matrix (Poisson) + extra dropout -> realistic single-cell sparsity. The same
    # `counts` matrix is the zero-mask for all likelihoods, so triple counts match and the
    # per-step cost comparison is apples-to-apples.
    counts = rng.poisson(np.exp(0.3 * (rng.standard_normal((I, 3)) @ rng.standard_normal((3, F))))).astype(float)
    counts[rng.random((I, F)) < args.density] = 0.0
    like = args.likelihood.lower()
    if like == "gaussian":
        values = (counts - counts.mean(0)) / (counts.std(0) + 1e-8)
    elif like in ("nb", "negativebinomial"):
        values = counts
    elif like == "bernoulli":
        values = (counts > 0).astype(float)
    else:
        raise ValueError(like)

    t_build = time.perf_counter()
    X, Y, W = build_triple_store(counts, values, neg_sample_ratio=args.ratio, seed=1, shuffle=False)
    build_s = time.perf_counter() - t_build

    gp = GPD(I=I, J=F, K=None, M1=args.M, M2=args.M, emb_sizes=[2, 2, 2],
             batch_size=args.batch_size, obs_mean1=0.0, obs_mean2=None,
             emb_reg=1e-3, lr=1e-2, save_path=tempfile.mkdtemp(), likelihood=like)
    gp.build(kernels=["RBF"])

    t0 = time.perf_counter()
    gp.train((X + 1).astype(int), Y, n_iter=args.epochs, sample_weight1=W)
    wall = time.perf_counter() - t0

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    dense = I * F
    # Machine-readable line for the outer loop.
    print(f"RESULT like={like} ratio={args.ratio} density={args.density} triples={len(Y)} "
          f"dense={dense} frac={len(Y)/dense:.4f} build_s={build_s:.2f} "
          f"wall_s={wall:.2f} per_epoch_s={wall/args.epochs:.2f} peak_mb={peak_mb:.0f}")


if __name__ == "__main__":
    main()
