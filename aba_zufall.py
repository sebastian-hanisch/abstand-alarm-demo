"""Zufallszahlen für viele Wiederholungen auf einmal (vektorisierter SplitMix64) und die Tageswerte des Normalzustands.

Jede Wiederholung (Zeile j) hat ihren eigenen Strom (Zufalls-Seed, Zweck, j); `instanz_rng` ist wörtlich aus `verteilungsabstand-demo/vab_profile.py` kopiert (Kopier-Test
`tests/test_zufall.py`). `zufall_gleich_zeilen` rechnet dieselben Zahlen wie `SplitMix64.random()` für die Zeilen ab, bis ab + zeilen, mit uint64 für alle Zeilen zugleich (bitgleich,
Test); darum hängt das Ergebnis einer Wiederholung nie daran, in welchen Blöcken gerechnet wird. Normalzahlen über Box-Muller mit numpy-Funktionen: in der letzten Binärstelle
plattformabhängig, ohne Einfluss auf die Zählungen der Tests (Bänder)."""

import math

import numpy as np

from aba_rng import MASK, SplitMix64


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def zufall_gleich_zeilen(seed: int, zweck: int, ab: int, zeilen: int, spalten: int) -> np.ndarray:
    """Gleichverteilte Zahlen in [0, 1): die Zeilen ab, ab + 1, ..., ab + zeilen − 1 (Zeile j = Strom (seed, zweck, j)) mit je `spalten` Zahlen, bitgleich zu `SplitMix64.random()`."""
    z0 = np.array([instanz_rng(seed, zweck, j).state for j in range(ab, ab + zeilen)], dtype=np.uint64)
    i = np.arange(1, spalten + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z0[:, None] + i[None, :] * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def normal_aus_gleich(u: np.ndarray) -> np.ndarray:
    """Standardnormalzahlen aus gleichverteilten Paaren (Spalten 0, 1 → Wert 0; Spalten 2, 3 → Wert 1; ...): Box-Muller mit numpy."""
    u1 = np.maximum(u[..., 0::2], 1e-12)
    u2 = u[..., 1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def standardwerte(seed: int, zweck: int, ab: int, zeilen: int, spalten: int, familie: str) -> np.ndarray:
    """Werte mit Mittel 0 und Standardabweichung 1 in der Familie: „Normal“ oder „t(3)“ (t mit 3 Freiheitsgraden, auf Standardabweichung 1 normiert: Z / √(Z₁² + Z₂² + Z₃²))."""
    if familie == "Normal":
        return normal_aus_gleich(zufall_gleich_zeilen(seed, zweck, ab, zeilen, 2 * spalten))
    if familie == "t(3)":
        n = normal_aus_gleich(zufall_gleich_zeilen(seed, zweck, ab, zeilen, 8 * spalten)).reshape(zeilen, spalten, 4)
        return n[..., 0] / np.sqrt(np.sum(n[..., 1:] ** 2, axis=-1))
    raise ValueError(f"unbekannte Familie: {familie}")
