"""Orakel-Tests: die Statistiken, der Permutationstest und der Strom gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12). Alle zusammen unter 10 Sekunden.

Orakel:
- W1, KS und Energie-Distanz gegen `scipy.stats.wasserstein_distance`, `ks_2samp`, `energy_distance` (40 Zufallspaare, dazu Paare mit Gleichständen aus ganzen Zahlen; Toleranz 1e-12);
- |Mittel| gegen die Mittelwerte; KL gegen eine zweite Schleife (Klassen nach Rang über `sorted`, ohne `reduceat`);
- die Permutationsverteilung bei N = 10 durch vollständige Aufzählung (210 Aufteilungen) gegen die Zufallsvertauschung der Demo (Band);
- die Fehlalarmprobe im Normalzustand (Alarmrate 5 % ± 3 Standardfehler, beide Familien und beide B);
- der Strom: die vektorisierte Fassung gegen eine einfache Schleife über Ströme und Fensterschritte (gleich auf 1e-12), dazu einzelne Schritte gegen scipy;
- W1 der Grundgesamtheit (Mischung gegen Normal) gegen `scipy.integrate.quad`;
- die Zahlen der unabhängigen Nachrechnung (eigene Zufallszahlen, eigene Vertauschungen als Matrixrechnung) im Band."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

import aba_constants as C
import aba_evaluation as E
import aba_permutation as P
import aba_statistiken as S
import aba_strom as ST
import aba_szenarien as Z

scipy_stats = pytest.importorskip("scipy.stats")
ROOT = Path(__file__).resolve().parent.parent
MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))


def _paare(n=40):
    g = np.random.default_rng(4242)
    for j in range(n):
        nr, nw = int(g.integers(5, 120)), int(g.integers(2, 60))
        yield 100 + 20 * g.normal(size=nr), 100 + 25 * g.normal(size=nw) + g.choice([0.0, 8.0, -15.0])


def test_w1_ks_energie_und_mittel_gegen_scipy_40_paare():
    for x, y in _paare(40):
        e = S.statistiken_paare(x[None, :], y[None, :])
        assert abs(e["W1"][0] - scipy_stats.wasserstein_distance(x, y)) < 1e-12
        assert abs(e["KS"][0] - scipy_stats.ks_2samp(x, y).statistic) < 1e-12
        assert abs(e["Energie"][0] - scipy_stats.energy_distance(x, y)) < 1e-12
        assert abs(e["|Mittel|"][0] - abs(x.mean() - y.mean())) < 1e-12


def test_statistiken_bei_gleichstaenden_gegen_scipy_20_paare_aus_ganzen_zahlen():
    g = np.random.default_rng(99)
    for _ in range(20):
        x = g.integers(0, 12, size=int(g.integers(5, 40))).astype(float)
        y = g.integers(2, 14, size=int(g.integers(3, 25))).astype(float)
        e = S.statistiken_paare(x[None, :], y[None, :])
        assert abs(e["W1"][0] - scipy_stats.wasserstein_distance(x, y)) < 1e-12
        assert abs(e["KS"][0] - scipy_stats.ks_2samp(x, y).statistic) < 1e-12              # KS gilt nur an Stellen, an denen der nächste Wert größer ist
        assert abs(e["Energie"][0] - scipy_stats.energy_distance(x, y)) < 1e-12


def _kl_schleife(x, y, klassen=10, glaettung=0.5):
    """Zweiter Rechenweg für KL(Referenz ‖ Fenster): Rang im sortierten Pool, bei gleichen Werten der Rang des letzten von ihnen; Klasse = ⌊Rang · 10 / N⌋; Zählung in einer Schleife."""
    pool = sorted([(float(v), 0) for v in x] + [(float(v), 1) for v in y])
    n = len(pool)
    letzter = {}
    for rang, (wert, _) in enumerate(pool):
        letzter[wert] = rang
    r, w = [0] * klassen, [0] * klassen
    for wert, gruppe in pool:
        (w if gruppe else r)[letzter[wert] * klassen // n] += 1
    kl = 0.0
    for c in range(klassen):
        p = (r[c] + glaettung) / (len(x) + glaettung * klassen)
        q = (w[c] + glaettung) / (len(y) + glaettung * klassen)
        kl += p * math.log(p / q)
    return kl


def test_kl_gegen_eine_zweite_schleife_40_paare_und_kleine_pools():
    for x, y in _paare(40):
        assert abs(S.statistiken_paare(x[None, :], y[None, :])["KL"][0] - _kl_schleife(x, y)) < 1e-12
    g = np.random.default_rng(5)
    for n in (3, 5, 9, 10, 11):                                    # weniger Werte als Klassen und knapp darüber
        x, y = g.normal(size=n), g.normal(size=2)
        assert abs(S.statistiken_paare(x[None, :], y[None, :])["KL"][0] - _kl_schleife(x, y)) < 1e-12, n


def test_kl_bei_gleichstaenden_gegen_die_zweite_schleife_und_konstante_werte_gegen_den_handwert():
    g = np.random.default_rng(12)
    for _ in range(30):
        x, y = g.integers(0, 6, size=int(g.integers(8, 40))).astype(float), g.integers(2, 8, size=int(g.integers(3, 20))).astype(float)
        assert abs(S.statistiken_paare(x[None, :], y[None, :])["KL"][0] - _kl_schleife(x, y)) < 1e-12
    # konstante Werte: alle in Klasse 0; Referenz 20, Fenster 10 Werte; mit +0,5 je Klasse: p₀ = 20,5/25, q₀ = 10,5/15, übrige neun Klassen 0,5/25 und 0,5/15
    hand = 20.5 / 25 * math.log((20.5 / 25) / (10.5 / 15)) + 9 * (0.5 / 25) * math.log((0.5 / 25) / (0.5 / 15))
    assert abs(S.statistiken_paare(np.full((1, 20), 100.0), np.full((1, 10), 100.0))["KL"][0] - hand) < 1e-14
    assert abs(_kl_schleife(np.full(20, 100.0), np.full(10, 100.0)) - hand) < 1e-14


def test_permutationsverteilung_bei_n_10_durch_aufzaehlung_gegen_die_zufallsvertauschung():
    g = np.random.default_rng(7)
    ref, fen = 100 + 20 * g.normal(size=(30, 6)), 100 + 20 * g.normal(size=(30, 4)) + 15.0
    t_beob, p_zufall, _ = P.permutationstest(ref, fen, 2000, 3, 0)
    for j in range(30):
        pool = np.concatenate([ref[j], fen[j]])
        z = np.sort(pool)
        teile = list(itertools.combinations(range(10), 4))
        ind = np.zeros((len(teile), 10))
        for i, t in enumerate(teile):
            ind[i, list(t)] = 1.0
        t_alle = S.statistiken_matrix(S.statistiken(z[None, :], ind, 4))                    # alle 210 Aufteilungen
        for i in range(5):
            p_exakt = np.mean(t_alle[:, i] >= t_beob[j, i] - 1e-12 * max(1, abs(t_beob[j, i])))
            assert abs(p_exakt - p_zufall[j, i]) < 0.06, (j, i, p_exakt, p_zufall[j, i])      # Standardfehler bei B = 2000 höchstens 0,011: über 5 Standardfehler


@pytest.mark.parametrize("familie,b", [("Normal", 199), ("t(3)", 99)])
def test_fehlalarmprobe_im_normalzustand_im_band_5_prozent_plus_minus_3_standardfehler(familie, b):
    r = P.perm_lauf(Z.Szenario("normal", familie), 1000, b, C.SERIEN_SEED)
    raten = P.alarm_perm(r["p"], b).mean(axis=0)
    se = math.sqrt(0.05 * 0.95 / 1000)
    assert np.all(np.abs(raten - 0.05) < 3 * se), raten               # Verhältnis: V2 4,1 bis 5,8 %, unabhängige Nachrechnung 3,7 bis 5,5 %
    mediane = np.median(r["p"], axis=0)
    assert np.all((mediane > 0.44) & (mediane < 0.62)), mediane       # unter Austauschbarkeit etwa 0,5; die diskreten Werte von KS machen den Test vorsichtig (Median etwas darüber)


def test_strom_vektorisiert_gegen_schleife_gleich_auf_1e12():
    for szen in (Z.Szenario("normal"), Z.Szenario("streuung", "Normal", 1.5), Z.Szenario("ausreisser", "t(3)", 3, 8)):
        ref, s = Z.strom(11, 0, 6, szen, 12)
        assert ref.shape == (6, 100) and s.shape == (6, 62)
        schnell, langsam = ST.verlauf(ref, s, 12), ST.verlauf_schleife(ref, s, 12)
        assert np.max(np.abs(schnell - langsam)) < 1e-12


def test_strom_schritte_gegen_scipy():
    ref, s = Z.strom(11, 0, 3, Z.Szenario("mittel", "Normal", 0.5), 80)
    t = ST.verlauf(ref, s, 80)
    for j, schritt in ((0, 0), (1, 7), (2, 80), (0, 55)):
        x, y = ref[j], s[j, schritt: schritt + 50]
        erwartet = [scipy_stats.wasserstein_distance(x, y), scipy_stats.ks_2samp(x, y).statistic, scipy_stats.energy_distance(x, y)]
        assert np.allclose(t[j, schritt, :3], erwartet, atol=1e-12), (j, schritt)
        assert abs(t[j, schritt, 4] - abs(x.mean() - y.mean())) < 1e-12


def test_w1_der_grundgesamtheit_gegen_quadratur():
    integrate = pytest.importorskip("scipy.integrate")
    norm = scipy_stats.norm
    for d, sg in ((36, 8.72), (20, 8.72), (40, 6.0), (10, 14.0)):
        f = lambda z: abs(norm.cdf(z, 100, 20) - 0.5 * norm.cdf(z, 100 - d / 2, sg) - 0.5 * norm.cdf(z, 100 + d / 2, sg))
        erwartet = integrate.quad(f, -100, 300, limit=800, points=[100 - d / 2, 100, 100 + d / 2])[0]
        assert abs(E.w1_grundgesamtheit(d, sg) - erwartet) < 1e-5, (d, sg)


def test_zahlen_der_unabhaengigen_nachrechnung_liegen_im_band():
    """Unabhängige Nachrechnung (eigene Zufallszahlen, 1 000 Zufalls-Seeds, B = 199) gegen die Zeilen der Messreihe; Band 4 Prozentpunkte (zwei Standardfehler der Differenz zweier Läufe: 2,2)."""
    nachrechnung = {                       # Alarmrate in %, Reihenfolge W1, KS, Energie, KL, |Mittel| (Permutationstest)
        "Normal|normal|0|0": [4.7, 3.7, 4.8, 5.5, 4.9],
        "Normal|mittel|0.4|0": [59.1, 45.7, 57.9, 24.7, 60.7],
        "Normal|streuung|1.5|0": [54.4, 19.1, 41.7, 31.2, 9.1],
        "Normal|ausreisser|3|8": [61.8, 5.9, 24.6, 6.6, 46.3],
        "Normal|ausreisser|1|50": [15.6, 5.0, 9.6, 4.9, 15.1],
        "Normal|zwei|36|8.72": [11.9, 25.0, 16.5, 48.1, 5.7],
        "t(3)|normal|0|0": [3.7, 4.9, 4.6, 4.5, 4.1],
        "t(3)|mittel|0.5|0": [92.5, 94.8, 97.0, 68.8, 83.3],
        "t(3)|streuung|1.5|0": [41.2, 17.1, 34.3, 24.4, 8.6],
    }
    for schluessel, soll in nachrechnung.items():
        ist = [100 * v for v in MESS["perm"][f"{schluessel}|1000|199|{C.SERIEN_SEED}"]["alarm_perm"]]
        assert np.all(np.abs(np.array(ist) - soll) < 4.0), (schluessel, ist, soll)
    strom = {                              # Alarmanteil bis k = 150 in %, Reihenfolge W1, KS, Energie, KL, |Mittel| (600 Ströme in der Nachrechnung: Standardfehler bis 2 Punkte; die Schwellen weichen um bis 4 % ab und Energie und KL reagieren darauf stark, daher 10 Punkte, bei W1 3 Punkte)
        "Normal|mittel|0.4|0": [74.5, 57.3, 67.5, 23.3, 75.7],
        "Normal|streuung|1.5|0": [87.2, 18.0, 47.5, 34.2, 27.0],
        "Normal|ausreisser|3|8": [98.7, 8.2, 60.2, 8.0, 92.3],
        "Normal|ausreisser|1|50": [96.7, 5.0, 65.3, 5.2, 96.7],
        "Normal|zwei|36|8.72": [7.8, 30.5, 15.2, 57.2, 5.7],
    }
    for schluessel, soll in strom.items():
        ist = [100 * v for v in MESS["strom"][f"{schluessel}|1000|150|{C.SERIEN_SEED}"]["marken"]["150"]]
        assert abs(ist[0] - soll[0]) < 3.0 and np.all(np.abs(np.array(ist) - soll) < 10.0), (schluessel, ist, soll)
    # Schwellen aus 5 000 Normalpaaren der Nachrechnung: W1 7,202, KS 0,23, Energie 1,307, KL 0,2593, |Mittel| 6,637 (Schwankung des 95-%-Quantils etwa 2 %); hier 20 000 Paare
    sw = MESS["schwellen_fenster"]["Normal"]
    assert abs(sw["W1"] - 7.202) < 0.3 and abs(sw["KS"] - 0.23) < 0.011 and abs(sw["Energie"] - 1.307) < 0.06 and abs(sw["KL"] - 0.2593) < 0.02 and abs(sw["|Mittel|"] - 6.637) < 0.3
