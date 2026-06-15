"""Minimal reproducible benchmark for the MOMO-GP (GPD) model.

Builds a small synthetic two-view dataset, runs N training steps, and reports
wall-clock time, peak process memory, and final loss. Optionally runs the
TensorFlow profiler over a window of steps.

This is a NEW artifact and does not modify any model code. It works around two
Mac/CPU realities documented in CLAUDE.md:
  * mwgp.py probes the GPU via `nvidia-smi` at IMPORT time -> we put a stub
    `nvidia-smi` on PATH so the import succeeds on a non-NVIDIA machine.
  * No CUDA memory metrics on Apple Silicon -> we report process RSS instead.

Usage (inside the `momogp-cpu` conda env):
    python benchmarks/synthetic_benchmark.py --epochs 1 --batch_size 1000
    python benchmarks/synthetic_benchmark.py --profile --epochs 1
"""
from __future__ import annotations

import argparse
import os
import resource
import stat
import sys
import tempfile
import time


def _install_nvidia_smi_stub() -> None:
    """Put a fake `nvidia-smi` on PATH so importing mwgp.py does not crash.

    mwgp.get_free_gpu_idx() parses lines containing 'Used' and reads the 3rd
    whitespace token as an int. We emit one such line reporting 0 MiB used.
    """
    d = tempfile.mkdtemp(prefix="momogp_stub_")
    path = os.path.join(d, "nvidia-smi")
    with open(path, "w") as fh:
        fh.write(
            "#!/usr/bin/env bash\n"
            'echo "GPU 00000000:00:00.0"\n'
            'echo "    FB Memory Usage"\n'
            'echo "        Total : 1 MiB"\n'
            'echo "        Used  : 0 MiB"\n'
            'echo "        Free  : 1 MiB"\n'
        )
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def make_synthetic(I: int, J: int, K: int, density1: float = 0.9, density2: float = 0.98,
                   seed: int = 11111986):
    """Build two sparse low-rank raw matrices (raw1, raw2).

    :param density1/2: fraction of STRUCTURAL ZEROS per view (~0.9 RNA, ~0.98 ATAC).
    :returns: (raw1 (I,J), raw2 (I,K)) — call ``make_stores`` to z-score + build triples.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    # Low-rank structure, then a sparsity mask so the data is genuinely sparse like the
    # real RNA (~90% zero) / ATAC (~98% zero) matrices.
    A = rng.standard_normal((I, 3))
    B = rng.standard_normal((J, 3))
    C = rng.standard_normal((K, 3))
    raw1 = (A @ B.T + 0.1 * rng.standard_normal((I, J))) + 5.0  # shift away from 0
    raw2 = (A @ C.T + 0.1 * rng.standard_normal((I, K))) + 5.0
    raw1[rng.random((I, J)) < density1] = 0.0   # structural zeros
    raw2[rng.random((I, K)) < density2] = 0.0
    return raw1, raw2


def make_stores(raw1, raw2, neg_sample_ratio=None, seed=0):
    """z-score, then build (X, Y, W) triple stores (optionally negative-subsampled)."""
    from momogp.data import build_triple_store

    def zscore(M):
        return (M - M.mean(0)) / (M.std(0) + 1e-8)

    z1, z2 = zscore(raw1), zscore(raw2)
    X1, y1, w1 = build_triple_store(raw1, z1, neg_sample_ratio=neg_sample_ratio, seed=seed)
    X2, y2, w2 = build_triple_store(raw2, z2, neg_sample_ratio=neg_sample_ratio, seed=seed + 1)
    # 1-based indices (notebook passes X_tr + 1).
    return (X1 + 1).astype(int), y1, w1, (X2 + 1).astype(int), y2, w2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--I", type=int, default=200, help="cells")
    p.add_argument("--J", type=int, default=100, help="view-1 features (genes)")
    p.add_argument("--K", type=int, default=20, help="view-2 features (proteins)")
    p.add_argument("--dims", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--M1", type=int, default=128)
    p.add_argument("--M2", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--profile", action="store_true", help="run tf.profiler")
    p.add_argument("--logdir", default="benchmarks/tb_logs")
    p.add_argument("--density1", type=float, default=0.9, help="view-1 structural-zero fraction")
    p.add_argument("--density2", type=float, default=0.98, help="view-2 structural-zero fraction")
    p.add_argument("--neg_sample_ratio", type=float, default=None,
                   help="K zeros kept per non-zero (negative subsampling). Omit = dense.")
    p.add_argument("--efficient_multiview", action="store_true",
                   help="batch the two views independently (no pad_sequences)")
    args = p.parse_args()

    _install_nvidia_smi_stub()

    import numpy as np
    np.random.seed(11111986)
    import tensorflow as tf
    tf.random.set_seed(11111986)

    # repo root (two levels up: momogp/benchmarks/ -> momogp/ -> root) so `momogp` imports.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from momogp import GPD  # noqa: E402  (import after stub + path setup)

    raw1, raw2 = make_synthetic(args.I, args.J, args.K, args.density1, args.density2)
    X1, y1, w1, X2, y2, w2 = make_stores(raw1, raw2, neg_sample_ratio=args.neg_sample_ratio)
    dense1 = args.I * args.J
    print(f"view1 triples: {len(y1)} / {dense1} dense ({100*len(y1)/dense1:.0f}%)  "
          f"view2 triples: {len(y2)} / {args.I*args.K} dense")
    n_steps_per_epoch = int(np.ceil(max(len(y1), len(y2)) / args.batch_size))
    print(f"~{n_steps_per_epoch} steps/epoch x {args.epochs} epochs  "
          f"| neg_sample_ratio={args.neg_sample_ratio} efficient_multiview={args.efficient_multiview}")

    save_path = tempfile.mkdtemp(prefix="momogp_ckpt_")
    gp = GPD(
        I=args.I, J=args.J, K=args.K, M1=args.M1, M2=args.M2,
        emb_sizes=args.dims, batch_size=args.batch_size,
        obs_mean1=float(y1.mean()), obs_mean2=float(y2.mean()),
        emb_reg=1e-3, lr=1e-2, save_path=save_path,
    )
    gp.build(kernels=["RBF"])

    if args.profile:
        os.makedirs(args.logdir, exist_ok=True)
        tf.profiler.experimental.start(args.logdir)

    t0 = time.perf_counter()
    hist = gp.train(X1, y1, X_tr2=X2, Y_tr2=y2, n_iter=args.epochs,
                    sample_weight1=w1, sample_weight2=w2,
                    efficient_multiview=args.efficient_multiview)
    wall = time.perf_counter() - t0

    if args.profile:
        tf.profiler.experimental.stop()

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    final_loss = hist.history["loss"][-1]

    print("\n=== RESULTS ===")
    print(f"wall_clock_s        : {wall:.3f}")
    print(f"steps               : {n_steps_per_epoch * args.epochs}")
    print(f"s_per_step          : {wall / max(1, n_steps_per_epoch * args.epochs):.4f}")
    print(f"peak_rss_mb         : {peak_mb:.1f}")
    print(f"final_loss          : {final_loss:.6f}")
    if args.profile:
        print(f"profiler logdir     : {args.logdir}  (open with: tensorboard --logdir {args.logdir})")


if __name__ == "__main__":
    main()
