"""Abstand W1 zwischen zwei Verteilungen auf einem Raster (nur numpy und math), wörtlich aus `verteilungsabstand-demo/vab_abstand.py` kopiert (Kopier-Test `tests/test_zufall.py`).

Hier gebraucht für die Verteilungen der Grundgesamtheit (Normalverteilung gegen die Mischung zweier Gipfel, Experiment 4); die Statistiken der Stichproben
(Referenz 100, Fenster 50) stehen in `aba_statistiken.py`."""

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])


def normal_cdf(z):
    """Verteilungsfunktion der Standardnormalverteilung (math.erf)."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def verteilungsfunktion(p) -> np.ndarray:
    """F_k = p_1 + ... + p_k (an den Zellgrenzen, die letzte ist 1)."""
    return np.cumsum(np.asarray(p, dtype=float))


def w1(p, q, h: float = 1.0) -> float:
    """W1 auf der Geraden: h · Σ_k |F_k − G_k|."""
    return float(h * np.sum(np.abs(verteilungsfunktion(p) - verteilungsfunktion(q))))
