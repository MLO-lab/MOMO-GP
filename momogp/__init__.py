"""MOMO-GP core model package.

The reusable Multi-Output Gaussian Process model, separated from the biological
application code (see the top-level ``analysis/`` package and ``experiments/``).

``GPD`` is imported lazily so that lightweight, TensorFlow-free modules such as
``momogp.data`` can be used (and tested) without importing ``mwgp`` — which pulls in
TensorFlow/GPflow and runs a GPU probe at import time.
"""

__all__ = ["GPD"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "GPD":
        from .mwgp import GPD
        return GPD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
