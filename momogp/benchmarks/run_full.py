"""Full-scale MOMO-GP runner for a GPU box (real multiome RNA + ATAC).

Puts the whole pipeline together: per-view single-cell likelihoods (NB for RNA counts,
Bernoulli for binarised ATAC peaks) + negative subsampling of structural zeros, on the
full-size data where dense is infeasible (PBMC10k ~= 11.9k x 36.6k genes + 134.7k peaks
=> ~2e9 dense triples; subsampling at K=5 cuts ATAC ~8x and RNA ~1.7x).

Run a SMOKE TEST first (no data, CPU or GPU) to confirm the pipeline end-to-end:
    python momogp/benchmarks/run_full.py --synthetic --I 2000 --J 3000 --K 8000 --epochs 3

Then the real run on the GPU box (point at your multiome .h5mu with raw counts):
    python momogp/benchmarks/run_full.py --data /path/pbmc10k.h5mu \
        --rna_mod rna --atac_mod atac --counts_layer counts \
        --neg_sample_ratio 5 --cell_dim 10 --epochs 200 --batch_size 20000 \
        --out_dir ./momogp_run --label_key celltype

Notes
-----
* This module imports `momogp` which runs the repo's real `get_free_gpu_idx()` (picks the
  freest GPU via `nvidia-smi`) at import — intended on a GPU box. On a non-GPU box use
  `--synthetic` (a stub `nvidia-smi` is installed automatically only then).
* The model is float64 throughout (gpflow default); the embeddings are hard-coded float64,
  so mixed precision / float32 is NOT enabled here (would need embedding-dtype changes).
* RNA -> NB needs RAW COUNTS. ATAC is binarised (peak>0). If your `.X` is already
  normalised, pass `--counts_layer` pointing at the raw-count layer.
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
import time


def _maybe_stub_nvidia_smi(synthetic: bool):
    """On a non-GPU box (only used with --synthetic) stub nvidia-smi so import succeeds."""
    if not synthetic:
        return
    import shutil
    if shutil.which("nvidia-smi") is not None:
        return
    d = tempfile.mkdtemp(prefix="momogp_stub_")
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as fh:
        fh.write('#!/usr/bin/env bash\necho "GPU"\necho "        Used : 0 MiB"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------- data


def load_synthetic(I, J, K, d_rna, d_atac, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((I, 3))
    rna = rng.poisson(np.exp(0.3 * (Z @ rng.standard_normal((3, J))))).astype(float)
    rna[rng.random((I, J)) < d_rna] = 0.0
    atac = (rng.poisson(np.exp(0.2 * (Z @ rng.standard_normal((3, K))))) > 0).astype(float)
    atac[rng.random((I, K)) < d_atac] = 0.0
    return rna, atac, None


def load_multiome(path, rna_mod, atac_mod, counts_layer, label_key,
                  n_top_genes, n_top_peaks):
    """Load raw RNA counts + binarised ATAC (+ optional labels) from an .h5mu."""
    import numpy as np
    try:
        import mudata as md
        mdata = md.read(path)
    except Exception:
        import muon as mu
        mdata = mu.read(path)
    import scipy.sparse as sp

    def dense(ad, layer):
        X = ad.layers[layer] if (layer and layer in getattr(ad, "layers", {})) else ad.X
        return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)

    rna = mdata.mod[rna_mod]
    atac = mdata.mod[atac_mod]
    rna_counts = dense(rna, counts_layer).astype(float)
    atac_counts = dense(atac, counts_layer).astype(float)
    atac_bin = (atac_counts > 0).astype(float)

    # Optional feature selection (most variable) to bound size; 0/None keeps all.
    def top_var(M, n):
        if not n or n >= M.shape[1]:
            return M
        ln = np.log1p(M / (M.sum(1, keepdims=True) + 1e-8) * 1e4)
        keep = np.argsort(ln.var(0))[::-1][:n]
        return M[:, keep]

    rna_counts = top_var(rna_counts, n_top_genes)
    if n_top_peaks and n_top_peaks < atac_bin.shape[1]:
        keep = np.argsort(atac_bin.sum(0))[::-1][:n_top_peaks]  # most-detected peaks
        atac_bin = atac_bin[:, keep]

    labels = None
    if label_key and label_key in rna.obs:
        labels = rna.obs[label_key].astype(str).values
    return rna_counts, atac_bin, labels


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source")
    src.add_argument("--synthetic", action="store_true", help="generate fake multiome (smoke test)")
    src.add_argument("--data", help="path to multiome .h5mu (real run)")
    src.add_argument("--rna_mod", default="rna")
    src.add_argument("--atac_mod", default="atac")
    src.add_argument("--counts_layer", default=None, help="layer holding RAW counts (else .X)")
    src.add_argument("--label_key", default=None, help="obs column with cell-type labels (for ARI/NMI)")
    src.add_argument("--n_top_genes", type=int, default=2000, help="HVGs to keep (0=all)")
    src.add_argument("--n_top_peaks", type=int, default=20000, help="top peaks to keep (0=all)")
    # synthetic sizes
    ap.add_argument("--I", type=int, default=2000); ap.add_argument("--J", type=int, default=3000)
    ap.add_argument("--K", type=int, default=8000)
    ap.add_argument("--d_rna", type=float, default=0.9); ap.add_argument("--d_atac", type=float, default=0.97)
    # model / training
    m = ap.add_argument_group("model")
    m.add_argument("--cell_dim", type=int, default=10); m.add_argument("--gene_dim", type=int, default=5)
    m.add_argument("--peak_dim", type=int, default=5)
    m.add_argument("--M1", type=int, default=512); m.add_argument("--M2", type=int, default=512)
    m.add_argument("--neg_sample_ratio", type=float, default=5.0, help="K (None/0 omit = dense)")
    m.add_argument("--likelihood", nargs=2, default=["nb", "bernoulli"], help="per-view: rna atac")
    m.add_argument("--epochs", type=int, default=200); m.add_argument("--batch_size", type=int, default=20000)
    m.add_argument("--lr", type=float, default=1e-2)
    m.add_argument("--efficient_multiview", action="store_true",
                   help="GradientTape per-view batching (benchmark on GPU; can be slower)")
    m.add_argument("--float32", action="store_true",
                   help="run in float32 (halves memory; needs more Cholesky jitter)")
    ap.add_argument("--out_dir", default="./momogp_run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.synthetic and not args.data:
        ap.error("provide --data <h5mu> for a real run, or --synthetic for a smoke test")

    _maybe_stub_nvidia_smi(args.synthetic)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import numpy as np
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):       # avoid grabbing all GPU memory
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPUs visible: {len(gpus)}")

    import gpflow
    if args.float32:
        # Set BEFORE building any gpflow object (kernels/inducing/likelihood) or the
        # embeddings. Halves memory for embeddings, inducing points, kernel Grams and the
        # variational params. float32 Cholesky is less stable -> raise the jitter.
        gpflow.config.set_default_float(np.float32)
        gpflow.config.set_default_jitter(1e-4)
        print("precision: float32 (jitter 1e-4)")
    else:
        print(f"precision: {np.dtype(gpflow.default_float()).name}")

    from momogp import GPD
    from momogp.data import build_triple_store

    # ----- data
    if args.synthetic:
        rna, atac, labels = load_synthetic(args.I, args.J, args.K, args.d_rna, args.d_atac, args.seed)
    else:
        rna, atac, labels = load_multiome(args.data, args.rna_mod, args.atac_mod,
                                          args.counts_layer, args.label_key,
                                          args.n_top_genes, args.n_top_peaks)
    I, J = rna.shape
    _, Kp = atac.shape
    K = None if Kp == 0 else Kp
    ratio = args.neg_sample_ratio or None
    print(f"data: {I} cells | RNA {J} genes ({(rna==0).mean():.2f} zero) | "
          f"ATAC {Kp} peaks ({(atac==0).mean():.2f} zero)")

    # ----- triple stores (raw counts for NB, binary for Bernoulli), negative-subsampled
    X1, Y1, W1 = build_triple_store(rna, rna, neg_sample_ratio=ratio, seed=args.seed)
    X2, Y2, W2 = build_triple_store(atac, atac, neg_sample_ratio=ratio, seed=args.seed + 1)
    print(f"triples: RNA {len(Y1):,} (dense {I*J:,}) | ATAC {len(Y2):,} (dense {I*Kp:,})")

    # ----- model
    os.makedirs(args.out_dir, exist_ok=True)
    gp = GPD(I=I, J=J, K=K, M1=args.M1, M2=args.M2,
             emb_sizes=[args.cell_dim, args.gene_dim, args.peak_dim],
             batch_size=args.batch_size, obs_mean1=0.0, obs_mean2=0.0,
             emb_reg=1e-3, lr=args.lr, save_path=args.out_dir,
             likelihood=tuple(args.likelihood))
    gp.build(kernels=["RBF"])

    class _Timer(tf.keras.callbacks.Callback):
        def __init__(self): self.times = []
        def on_epoch_begin(self, e, logs=None): self._t = time.perf_counter()
        def on_epoch_end(self, e, logs=None): self.times.append(time.perf_counter() - self._t)
    timer = _Timer()

    t0 = time.perf_counter()
    hist = gp.train((X1 + 1).astype(int), Y1, X_tr2=(X2 + 1).astype(int), Y_tr2=Y2,
                    n_iter=args.epochs, sample_weight1=W1, sample_weight2=W2,
                    efficient_multiview=args.efficient_multiview,
                    extra_callbacks=[timer])
    wall = time.perf_counter() - t0

    # ----- report
    print("\n=== RESULTS ===")
    print(f"wall_clock_s   : {wall:.1f}")
    if timer.times:
        med = sorted(timer.times[1:] or timer.times)[len(timer.times[1:] or timer.times)//2]
        print(f"median_epoch_s : {med:.2f}  (epoch1 warmup {timer.times[0]:.1f}s)")
    print(f"final_loss     : {hist.history['loss'][-1]:.4f}")
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        print(f"peak_gpu_mb    : {info['peak'] / 1024**2:.0f}")
    except Exception:
        print("peak_gpu_mb    : n/a (no GPU)")

    # ----- save cell embedding A
    A = np.asarray(gp.emb1((np.arange(I) + 1).astype(int)))
    np.save(os.path.join(args.out_dir, "cell_embedding.npy"), A)
    print(f"saved cell embedding {A.shape} -> {args.out_dir}/cell_embedding.npy")

    # ----- optional downstream quality
    if labels is not None:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        classes = sorted(set(labels))
        lab = np.array([classes.index(s) for s in labels])
        km = KMeans(n_clusters=len(classes), n_init=10, random_state=args.seed).fit_predict(A)
        print(f"ARI={adjusted_rand_score(lab, km):.4f}  NMI={normalized_mutual_info_score(lab, km):.4f}  "
              f"({len(classes)} cell types)")


if __name__ == "__main__":
    main()
