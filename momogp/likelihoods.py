"""Custom observation likelihoods for MOMO-GP (count / single-cell aware).

The model's GP latent ``f`` parameterises the likelihood. By swapping the likelihood we
change how observations — especially the many zeros — are interpreted, without touching
the GP machinery or the negative-subsampling speedup (the reweighting is likelihood-
agnostic, so it composes with any of these).

- ``NegativeBinomial`` (here): for raw UMI counts. ``mu = invlink(f)`` (default ``exp``),
  dispersion ``theta > 0`` (scVI/scanpy "inverse-dispersion" convention), so
  ``Var[y] = mu + mu^2 / theta``. NB already absorbs most scRNA-seq zeros; it is the
  substrate for a later zero-inflated (ZINB) extension (add a dropout-logit head).
- For binary ATAC peaks use ``gpflow.likelihoods.Bernoulli`` (built in) on a binarised
  matrix — no custom code needed.

Variational expectations ``E_q[log p(y|f)]`` are non-conjugate; ``ScalarLikelihood``
supplies them via Gauss-Hermite quadrature over the 1-D latent (a small constant cost,
negligible next to the GP linear algebra).
"""
from __future__ import annotations

import tensorflow as tf
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.base import Parameter
from gpflow.likelihoods import ScalarLikelihood
from gpflow.utilities import positive


class NegativeBinomial(ScalarLikelihood):
    r"""Negative-binomial likelihood with mean ``mu = invlink(f)`` and dispersion ``theta``.

    log pmf (mean/inverse-dispersion form)::

        log Gamma(y+theta) - log Gamma(theta) - log Gamma(y+1)
            + theta * (log theta - log(theta+mu)) + y * (log mu - log(theta+mu))

    Computed in log-space (``log(theta+mu) = log theta + softplus(log mu - log theta)``)
    for numerical stability, so ``exp(f)`` is never formed directly.
    """

    def __init__(self, dispersion: float = 1.0, invlink=tf.exp, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dispersion = Parameter(dispersion, transform=positive(), name="nb_dispersion")
        self.invlink = invlink

    @inherit_check_shapes
    def _scalar_log_prob(self, X, F, Y) -> tf.Tensor:
        Y = tf.cast(Y, F.dtype)
        theta = tf.cast(self.dispersion, F.dtype)
        log_theta = tf.math.log(theta)
        log_mu = F if self.invlink is tf.exp else tf.math.log(self.invlink(F))
        # log(theta + mu), stable: = log theta + softplus(log mu - log theta).
        log_theta_plus_mu = log_theta + tf.math.softplus(log_mu - log_theta)
        return (
            tf.math.lgamma(Y + theta) - tf.math.lgamma(theta) - tf.math.lgamma(Y + 1.0)
            + theta * (log_theta - log_theta_plus_mu)
            + Y * (log_mu - log_theta_plus_mu)
        )

    @inherit_check_shapes
    def _conditional_mean(self, X, F) -> tf.Tensor:
        return self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X, F) -> tf.Tensor:
        mu = self.invlink(F)
        return mu + tf.square(mu) / self.dispersion


def make_likelihood(name: str):
    """Resolve a likelihood name to ``(likelihood, num_latent_gps)``.

    - ``"gaussian"``  -> ``gpflow.likelihoods.Gaussian(0.1)`` (continuous, z-scored targets).
    - ``"nb"``        -> :class:`NegativeBinomial` (raw counts).
    - ``"bernoulli"`` -> ``gpflow.likelihoods.Bernoulli`` (binary, e.g. ATAC detected/not).

    Count/binary likelihoods use a single latent GP (``num_latent_gps=1``: f = log-rate /
    logit). Gaussian keeps the legacy ``r1+r2`` latent head for backward compatibility.
    """
    name = (name or "gaussian").lower()
    if name == "gaussian":
        return gpflow.likelihoods.Gaussian(0.1), None  # None -> caller keeps legacy r1+r2
    if name in ("nb", "negativebinomial", "negative_binomial"):
        return NegativeBinomial(), 1
    if name == "bernoulli":
        return gpflow.likelihoods.Bernoulli(), 1
    raise ValueError(f"unknown likelihood {name!r}; use 'gaussian', 'nb', or 'bernoulli'")
