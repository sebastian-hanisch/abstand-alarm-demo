"""Die fünf Statistiken, die Referenz und Fenster vergleichen (nur numpy, ohne Streamlit).

Eine Stichprobe von Tageswerten (Referenz x₁ … x_{n_r}, Fenster y₁ … y_{n_w}) wird gepoolt und aufsteigend sortiert; `ind` kennzeichnet die Fensterwerte (1) und die Referenzwerte (0) in dieser
Reihenfolge. Dann ist an jeder sortierten Stelle i die Differenz der empirischen Verteilungsfunktionen D_i = F(z_i) − G(z_i) = (i + 1 − c_i)/n_r − c_i/n_w mit c_i = Zahl der Fensterwerte unter z₁ … z_i,
und alle Statistiken folgen daraus:
- `W1`: Wasserstein-1-Abstand = Σ |D_i| (z_{i+1} − z_i) (L1-Norm von F − G),
- `KS`: Kolmogorov-Smirnov-Abstand = max |D_i| (L∞-Norm; an Stellen mit gleichen Werten gilt nur die letzte),
- `Energie`: Energie-Distanz = √(2 Σ D_i² (z_{i+1} − z_i)) (L2-Norm, gleich `scipy.stats.energy_distance`),
- `KL`: Kullback-Leibler-Divergenz KL(Referenz ‖ Fenster) über 10 Klassen gleicher Masse der gepoolten Werte (die Klasse eines Werts folgt aus seinem Rang, gleiche Werte teilen eine Klasse), je Klasse +0,5 zum Zähler,
- `|Mittel|`: Betrag der Differenz der Mittelwerte.
Alle Funktionen rechnen über beliebig viele Wiederholungen zugleich (führende Achsen); die letzte Achse sind die N gepoolten Werte."""

import numpy as np

import aba_constants as C


def sortieren(ref: np.ndarray, fenster: np.ndarray):
    """Pool aus Referenz (…, n_r) und Fenster (…, n_w), aufsteigend sortiert: (z, ind) mit ind = 1 für Fensterwerte. Die Reihenfolge innerhalb gleicher Werte ist ohne Einfluss auf die Statistiken."""
    pool = np.concatenate([ref, fenster], axis=-1)
    markiert = np.zeros(pool.shape, dtype=np.float64)
    markiert[..., ref.shape[-1]:] = 1.0
    ordnung = np.argsort(pool, axis=-1)
    return np.take_along_axis(pool, ordnung, axis=-1), np.take_along_axis(markiert, ordnung, axis=-1)


def klassen_enden(z: np.ndarray, klassen: int = C.KLASSEN) -> np.ndarray:
    """Ende jeder Klasse in den sortierten Werten z (…, N): Feld (…, klassen) mit der Zahl der Werte in den Klassen 0 bis k (die Klassen sind zusammenhängende Strecken der sortierten Werte).

    Die Klasse eines Werts ist ⌊Rang · klassen / N⌋ (Klassen gleicher Masse); gleiche Werte kommen in dieselbe Klasse (die des letzten von ihnen), damit die Klassen nur von den Werten
    abhängen und nicht davon, wie ein Gleichstand sortiert wurde (sonst alarmierte KL bei konstanten Werten)."""
    n = z.shape[-1]
    idx = np.broadcast_to(np.arange(n), z.shape)
    neu = np.concatenate([np.ones(z.shape[:-1] + (1,), dtype=bool), z[..., 1:] != z[..., :-1]], axis=-1)
    erster = np.maximum.accumulate(np.where(neu, idx, 0), axis=-1)       # Beginn des Blocks gleicher Werte, zu dem die Stelle gehört
    e = -((-np.arange(1, klassen + 1) * n) // klassen)               # ⌈(k + 1) · N / klassen⌉; Klasse ≤ k heißt: der ganze Block gleicher Werte liegt vor e_k, also beginnt der Block von e_k dort, wo die Klasse k endet
    return np.where(e >= n, n, erster[..., np.minimum(e, n - 1)])


def statistiken(z: np.ndarray, ind: np.ndarray, n_w: int) -> dict:
    """Alle fünf Statistiken für sortierte Pools `z` (…, N) mit Fensterkennung `ind` (…, N); z und ind dürfen führende Achsen per Broadcast teilen (z: (S, 1, N), ind: (S, B, N)).

    Rückgabe: Wörterbuch Statistik → Feld der Form (…) ohne die letzte Achse."""
    n = z.shape[-1]
    n_r = n - n_w
    c = np.cumsum(ind, axis=-1)
    i1 = np.arange(1, n + 1, dtype=np.float64)
    d = (i1 - c) / n_r - c / n_w                                  # D_i; an der letzten Stelle genau 0
    dz = np.diff(z, axis=-1)
    d0 = d[..., :-1]
    w1 = np.sum(np.abs(d0) * dz, axis=-1)
    energie = np.sqrt(2.0 * np.sum(d0 * d0 * dz, axis=-1))
    ks = np.max(np.where(dz > 0, np.abs(d0), 0.0), axis=-1)
    summe_fenster = np.sum(ind * z, axis=-1)
    summe_alle = np.sum(z, axis=-1)
    mittel = np.abs((summe_alle - summe_fenster) / n_r - summe_fenster / n_w)
    ende = klassen_enden(z)
    c_roh = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)
    ende_b = np.broadcast_to(ende, c.shape[:-1] + ende.shape[-1:])
    c_ende = np.take_along_axis(c_roh, ende_b, axis=-1)
    w_k = np.diff(c_ende, axis=-1, prepend=0.0)
    groesse = np.diff(ende, axis=-1, prepend=0).astype(np.float64)
    r_k = groesse - w_k
    g = C.GLAETTUNG_KL
    p = (r_k + g) / (n_r + g * C.KLASSEN)
    q = (w_k + g) / (n_w + g * C.KLASSEN)
    kl = np.sum(p * np.log(p / q), axis=-1)
    return {"W1": w1, "KS": ks, "Energie": energie, "KL": kl, "|Mittel|": mittel}


def statistiken_paare(ref: np.ndarray, fenster: np.ndarray) -> dict:
    """Die fünf Statistiken für Paare (Referenz, Fenster): ref (S, n_r), fenster (S, n_w) → je ein Feld der Länge S."""
    z, ind = sortieren(ref, fenster)
    return statistiken(z, ind, fenster.shape[-1])


def statistiken_matrix(erg: dict) -> np.ndarray:
    """Wörterbuch → Feld (…, 5) in der Reihenfolge `C.STATISTIKEN`."""
    return np.stack([erg[s] for s in C.STATISTIKEN], axis=-1)
