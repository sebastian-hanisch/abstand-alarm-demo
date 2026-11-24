"""Alarmentscheidung für feste Fenster: Permutationstest und Schwelle aus dem Normalzustand (nur numpy, ohne Streamlit).

- Permutationstest: die Gruppenzugehörigkeit der 150 gepoolten Werte wird B-mal zufällig vertauscht (n_w Fensterplätze aus N), p = (1 + #{T_π ≥ T_beob}) / (B + 1); Alarm bei p ≤ 0,05.
  Die Fehlalarmrate ist unter Austauschbarkeit höchstens 5 %, ohne Annahme über die Verteilung.
- Schwelle „sauber“: das 95-%-Quantil von T über 20 000 Paare (Referenz 100, Fenster 50) aus dem bekannten Normalzustand der Familie; Alarm bei T > Schwelle.
Jede Wiederholung (Zeile j) hat ihre eigenen Zufallszahlen; die Blockgröße ändert nichts am Ergebnis."""

import numpy as np

import aba_constants as C
import aba_statistiken as S
import aba_szenarien as Z
from aba_zufall import zufall_gleich_zeilen

BLOCK_PERM = 50        # Wiederholungen je Block (Speicher: Block · B · 150 Zahlen)
BLOCK_SCHWELLE = 2000


def vertauschungen(u: np.ndarray, n_w: int) -> np.ndarray:
    """Zufällige Fensterkennungen: aus Schlüsseln u (…, N) werden die n_w kleinsten zum Fenster (1), die übrigen zur Referenz (0)."""
    idx = np.argpartition(u, n_w - 1, axis=-1)[..., :n_w]
    ind = np.zeros(u.shape, dtype=np.float64)
    np.put_along_axis(ind, idx, 1.0, axis=-1)
    return ind


def p_werte(t_beob: np.ndarray, t_perm: np.ndarray) -> np.ndarray:
    """p = (1 + #{T_π ≥ T_beob}) / (B + 1) je Statistik; t_beob (S, 1, 5), t_perm (S, B, 5); Gleichstände gelten als ≥ (Toleranz 1e-12, relativ)."""
    grenze = t_beob - C.TOL_GLEICH * np.maximum(1.0, np.abs(t_beob))
    zaehler = np.sum(t_perm >= grenze, axis=1)
    return (1.0 + zaehler) / (t_perm.shape[1] + 1.0)


def permutationstest(ref: np.ndarray, fenster: np.ndarray, b: int, seed: int, ab: int):
    """Permutationstest für S Paare (Wiederholungen ab, …, ab + S − 1). Rückgabe: (t_beob (S, 5), p (S, 5), gross (S,)); gross ist der Anteil der Vertauschungen, in denen der größte Wert im Fenster liegt."""
    s_n, n_w = ref.shape[0], fenster.shape[-1]
    z, ind = S.sortieren(ref, fenster)
    t_beob = S.statistiken_matrix(S.statistiken(z[:, None, :], ind[:, None, :], n_w))
    u = zufall_gleich_zeilen(seed, C.Z_PERM, ab, s_n, b * z.shape[-1]).reshape(s_n, b, z.shape[-1])
    ind_perm = vertauschungen(u, n_w)
    t_perm = S.statistiken_matrix(S.statistiken(z[:, None, :], ind_perm, n_w))
    gross = ind_perm[..., -1].mean(axis=1) if b > 0 else np.full(s_n, np.nan)
    return t_beob[:, 0, :], p_werte(t_beob, t_perm), gross


def perm_lauf(szen: Z.Szenario, seeds: int, b: int, seed: int) -> dict:
    """Permutationstest für `seeds` Wiederholungen des Szenarios: p-Werte (seeds, 5), Statistiken (seeds, 5), Anteil „größter Wert im Fenster“ (seeds,)."""
    ts, ps, gs = [], [], []
    for ab in range(0, seeds, BLOCK_PERM):
        zeilen = min(BLOCK_PERM, seeds - ab)
        ref, fen = Z.paare(seed, ab, zeilen, szen)
        t, p, g = permutationstest(ref, fen, b, seed, ab)
        ts.append(t)
        ps.append(p)
        gs.append(g)
    return {"t": np.concatenate(ts), "p": np.concatenate(ps), "gross": np.concatenate(gs)}


def alarm_perm(p: np.ndarray, b: int) -> np.ndarray:
    """Alarm bei p ≤ 0,05 (ganzzahlig verglichen: 1 + Zähler ≤ 0,05 · (B + 1)), Form wie p."""
    return (np.rint(p * (b + 1)) <= C.ALPHA * (b + 1) + 1e-9)


def schwelle_sauber(familie: str, seed: int = C.SERIEN_SEED, paare: int = C.N_SCHWELLE) -> dict:
    """95-%-Quantil jeder Statistik über `paare` Paare (Referenz 100, Fenster 50) im Normalzustand der Familie: Wörterbuch Statistik → Schwelle."""
    ts = []
    for ab in range(0, paare, BLOCK_SCHWELLE):
        ref, fen = Z.schwellenpaare(seed, ab, min(BLOCK_SCHWELLE, paare - ab), familie)
        ts.append(S.statistiken_matrix(S.statistiken_paare(ref, fen)))
    t = np.concatenate(ts)
    q = np.quantile(t, 1.0 - C.ALPHA, axis=0)
    return {s: float(q[i]) for i, s in enumerate(C.STATISTIKEN)}


def alarm_schwelle(t: np.ndarray, schwellen: dict) -> np.ndarray:
    """Alarm bei T > Schwelle; t (S, 5) in der Reihenfolge `C.STATISTIKEN`."""
    return t > np.array([schwellen[s] for s in C.STATISTIKEN])[None, :]
