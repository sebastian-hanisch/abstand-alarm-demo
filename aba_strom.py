"""Erkennungsverzögerung im Strom: ein gleitendes Fenster von 50 Werten (Schritt 1) gegen eine feste Referenz von 100 Werten (nur numpy, ohne Streamlit).

Der Strom hat 50 + h Werte; die Änderung beginnt mit dem 51. Wert. Zum Schritt t (t = 0, …, h) gehört das Fenster der Werte t, …, t + 49 (k = t neue Werte; Schritt 0 liegt rein im Normalzustand).
Die Schwelle je Statistik und Horizont ist das 95-%-Quantil des Maximums von T über die Schritte 0 bis h in Normalströmen: höchstens 5 % Fehlalarm über den ganzen Horizont (bei h = 150: je 151 Fensterschritte),
nicht je Fenster und nicht ARL₀ = 370."""

import numpy as np

import aba_constants as C
import aba_statistiken as S
import aba_szenarien as Z

BLOCK_STROM = 250      # Ströme je Block


def verlauf(ref: np.ndarray, strom: np.ndarray, h: int) -> np.ndarray:
    """Die fünf Statistiken zu jedem Schritt t = 0, …, h: Feld (Ströme, h + 1, 5). Je Schritt wird die Referenz mit dem Fenster strom[:, t : t + 50] gepoolt und sortiert."""
    t_werte = np.empty((ref.shape[0], h + 1, len(C.STATISTIKEN)))
    for t in range(h + 1):
        t_werte[:, t, :] = S.statistiken_matrix(S.statistiken_paare(ref, strom[:, t: t + C.N_FENSTER]))
    return t_werte


def strom_verlauf(szen: Z.Szenario, anzahl: int, h: int, seed: int, zweck_basis: int = C.Z_STROM_BASIS) -> np.ndarray:
    """Verlauf für `anzahl` Ströme des Szenarios (Zeilen 0, …, anzahl − 1), blockweise gerechnet; die Blockgröße ändert nichts am Ergebnis."""
    teile = []
    for ab in range(0, anzahl, BLOCK_STROM):
        ref, s = Z.strom(seed, ab, min(BLOCK_STROM, anzahl - ab), szen, h, zweck_basis)
        teile.append(verlauf(ref, s, h))
    return np.concatenate(teile)


def verlauf_schleife(ref: np.ndarray, strom: np.ndarray, h: int) -> np.ndarray:
    """Dasselbe als einfache Schleife über Ströme und Schritte (je ein Paar von Stichproben, ohne Vektorisierung über die Ströme); dient als zweiter Rechenweg in den Tests."""
    t_werte = np.empty((ref.shape[0], h + 1, len(C.STATISTIKEN)))
    for j in range(ref.shape[0]):
        for t in range(h + 1):
            e = S.statistiken_paare(ref[j: j + 1], strom[j: j + 1, t: t + C.N_FENSTER])
            t_werte[j, t, :] = [float(e[s][0]) for s in C.STATISTIKEN]
    return t_werte


def schwellen_je_horizont(t_normal: np.ndarray, horizonte=C.HORIZONT_OPT) -> dict:
    """95-%-Quantil des Maximums über die Schritte 0 bis h in Normalströmen: Wörterbuch h → Wörterbuch Statistik → Schwelle. t_normal: (Ströme, mindestens max(h) + 1, 5)."""
    lauf_max = np.maximum.accumulate(t_normal, axis=1)
    aus = {}
    for h in horizonte:
        q = np.quantile(lauf_max[:, h, :], 1.0 - C.ALPHA, axis=0)
        aus[int(h)] = {s: float(q[i]) for i, s in enumerate(C.STATISTIKEN)}
    return aus


def erste_alarme(t_werte: np.ndarray, schwelle: dict, h: int) -> np.ndarray:
    """Schritt des ersten Alarms je Strom und Statistik (Ströme, 5), bis zum Schritt h; −1, wenn bis h kein Alarm kam. Alarm bei T > Schwelle."""
    s = np.array([schwelle[x] for x in C.STATISTIKEN])
    alarm = t_werte[:, : h + 1, :] > s[None, None, :]
    erster = np.argmax(alarm, axis=1)
    return np.where(alarm.any(axis=1), erster, -1)


def anteil_bis(erste: np.ndarray, h: int) -> np.ndarray:
    """Anteil der Ströme mit Alarm bis zum Schritt k, für k = 0, …, h: Feld (h + 1, 5)."""
    k = np.arange(h + 1)[:, None, None]
    hat = (erste[None, :, :] >= 0) & (erste[None, :, :] <= k)
    return hat.mean(axis=1)


def median_k(erste: np.ndarray) -> np.ndarray:
    """Median des ersten Alarms unter den Strömen mit Alarm (je Statistik, nan ohne Alarm); aussagekräftig nur, wo mehr als die Hälfte alarmiert."""
    aus = np.full(erste.shape[1], np.nan)
    for i in range(erste.shape[1]):
        e = erste[:, i]
        if np.any(e >= 0):
            aus[i] = float(np.median(e[e >= 0]))
    return aus
