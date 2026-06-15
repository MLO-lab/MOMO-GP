"""NB vs Gaussian on real PBMC data: downstream embedding quality (ARI / NMI).

The bundled CITEseq file (data/CITEseq/pbmc5k_citeseq_sampled.h5mu) stores only *scaled*
RNA (normalize_total -> log1p -> scale; .raw dropped) so it has NO counts and cannot drive
the NB likelihood. We therefore use scanpy's real PBMC3k counts + the annotated cell-type
labels from pbmc3k_processed as an independent reference.

For each likelihood we train MOMO-GP, read the cell embedding A (= emb1 over the cell
indices), cluster it with KMeans(k = #celltypes), and score against the labels:
  - Gaussian : target = z-scored log-normalised expression (its natural domain)
  - NB       : target = raw counts (mu = exp(f), learned dispersion)
Same cells/genes/embedding-dim/epochs for both -> a fair like-for-like comparison.
"""
from __future__ import annotations

import argparse
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


def load_pbmc3k(n_cells, n_genes, seed):
    """Return (counts [C,G] float, lognorm [C,G] float, labels [C] int, n_classes)."""
    import numpy as np
    import scanpy as sc

    proc = sc.datasets.pbmc3k_processed()           # annotated cell types in .obs['louvain']
    counts = sc.datasets.pbmc3k()                   # raw counts
    common = proc.obs_names.intersection(counts.obs_names)
    proc = proc[common].copy()
    counts = counts[common, :].copy()
    # Use the processed (HVG) gene panel, restricted to genes present in the count matrix.
    genes = [g for g in proc.var_names if g in set(counts.var_names)]
    counts = counts[:, genes].copy()

    C = np.asarray(counts.X.todense() if hasattr(counts.X, "todense") else counts.X, dtype=float)
    labels_str = proc.obs["louvain"].astype(str).values
    classes = sorted(set(labels_str))
    lab = np.array([classes.index(s) for s in labels_str])

    rng = np.random.default_rng(seed)
    if n_cells and n_cells < C.shape[0]:
        idx = rng.choice(C.shape[0], n_cells, replace=False)
        C, lab = C[idx], lab[idx]
    if n_genes and n_genes < C.shape[1]:
        # keep the most variable genes (on log-norm) for signal
        ln = np.log1p(C / (C.sum(1, keepdims=True) + 1e-8) * 1e4)
        keep = np.argsort(ln.var(0))[::-1][:n_genes]
        C, ln = C[:, keep], ln[:, keep]
    else:
        ln = np.log1p(C / (C.sum(1, keepdims=True) + 1e-8) * 1e4)
    return C, ln, lab, len(classes)


def cell_embedding(gp, I):
    import numpy as np
    idx = (np.arange(I) + 1).astype(int)        # 1-based (mask_zero) cell indices
    return np.asarray(gp.emb1(idx))             # (I, cell_dim)


def score(A, labels, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(A)
    return adjusted_rand_score(labels, km), normalized_mutual_info_score(labels, km)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cells", type=int, default=1000)
    ap.add_argument("--n_genes", type=int, default=500)
    ap.add_argument("--cell_dim", type=int, default=10)
    ap.add_argument("--feat_dim", type=int, default=5)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _stub_nvidia_smi()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import numpy as np
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    from momogp import GPD
    from momogp.data import build_triple_store

    C, ln, labels, n_classes = load_pbmc3k(args.n_cells, args.n_genes, args.seed)
    I, F = C.shape
    z = (ln - ln.mean(0)) / (ln.std(0) + 1e-8)
    print(f"data: {I} cells x {F} genes, {n_classes} cell types, "
          f"counts zeros={float((C==0).mean()):.2f}")

    def train_and_score(likelihood, values):
        X, Y, W = build_triple_store(C, values, neg_sample_ratio=None, shuffle=False)
        gp = GPD(I=I, J=F, K=None, M1=args.M, M2=args.M,
                 emb_sizes=[args.cell_dim, args.feat_dim, args.feat_dim],
                 batch_size=args.batch_size, obs_mean1=0.0, obs_mean2=None,
                 emb_reg=1e-3, lr=1e-2, save_path=tempfile.mkdtemp(), likelihood=likelihood)
        gp.build(kernels=["RBF"])
        gp.train((X + 1).astype(int), Y, n_iter=args.epochs)
        A = cell_embedding(gp, I)
        ari, nmi = score(A, labels, n_classes, args.seed)
        print(f"RESULT likelihood={likelihood:9s} ARI={ari:.4f} NMI={nmi:.4f}")
        return ari, nmi

    print("\n=== NB vs Gaussian on real PBMC3k (downstream clustering vs cell-type labels) ===")
    train_and_score("gaussian", z)
    train_and_score("nb", C)


if __name__ == "__main__":
    main()
