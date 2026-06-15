"""Tests for the custom NegativeBinomial likelihood (requires TensorFlow/gpflow).

Run in the momogp-cpu env:  python tests/test_likelihoods.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Keep the GPU probe / TF logs quiet for import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def test_nb_logprob_matches_scipy():
    import tensorflow as tf
    from scipy.stats import nbinom
    from momogp.likelihoods import NegativeBinomial

    theta = 2.5
    lik = NegativeBinomial(dispersion=theta)
    F = np.log(np.array([[0.5], [1.0], [3.0], [10.0]]))   # F = log(mean)
    Y = np.array([[0.0], [2.0], [5.0], [0.0]])
    X = np.zeros_like(F)

    lp = lik._scalar_log_prob(tf.constant(X), tf.constant(F), tf.constant(Y)).numpy()

    mu = np.exp(F)
    # scipy nbinom(n, p): n = dispersion theta, p = theta / (theta + mu).
    ref = nbinom.logpmf(Y, n=theta, p=theta / (theta + mu))
    assert np.allclose(lp, ref, atol=1e-6), f"\nNB={lp}\nscipy={ref}"


def test_nb_mean_and_variance():
    import tensorflow as tf
    from momogp.likelihoods import NegativeBinomial

    theta = 3.0
    lik = NegativeBinomial(dispersion=theta)
    F = np.log(np.array([[1.0], [4.0]]))
    X = np.zeros_like(F)
    mu = np.exp(F)
    m = lik._conditional_mean(tf.constant(X), tf.constant(F)).numpy()
    v = lik._conditional_variance(tf.constant(X), tf.constant(F)).numpy()
    assert np.allclose(m, mu, atol=1e-6)
    assert np.allclose(v, mu + mu ** 2 / theta, atol=1e-6)  # NB variance


def test_make_likelihood_resolves():
    from momogp.likelihoods import make_likelihood, NegativeBinomial
    import gpflow
    g, ng = make_likelihood("gaussian")
    assert isinstance(g, gpflow.likelihoods.Gaussian) and ng is None
    nb, nnb = make_likelihood("nb")
    assert isinstance(nb, NegativeBinomial) and nnb == 1
    b, nb2 = make_likelihood("bernoulli")
    assert isinstance(b, gpflow.likelihoods.Bernoulli) and nb2 == 1


if __name__ == "__main__":
    test_nb_logprob_matches_scipy()
    test_nb_mean_and_variance()
    test_make_likelihood_resolves()
    print("all likelihood tests passed")
