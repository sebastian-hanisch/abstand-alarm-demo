"""Handbeispiele: kleine Fälle, die von Hand (in Brüchen) gerechnet sind; der Erwartungswert steht im Test als Rechnung, nicht als Abschrift des Codes.

1. Drei Statistiken aus einer Differenz der Verteilungsfunktionen. 2. Exakter Permutationstest. 3. Einzelausreißer: was jede Statistik höchstens sehen kann. 4. Der Ausreißer in der Permutation.
5. Ankunft im Strom. 6. Zwei Gipfel: gleicher Mittelwert, gleiche Streuung."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np

import aba_constants as C
import aba_evaluation as E
import aba_permutation as P
import aba_statistiken as S


def cdf_normen(ref, win):
    """Von Hand: L1 (= W1), L-unendlich (= KS) und L2² der Differenz der Verteilungsfunktionen, exakt in Brüchen (Gewichte 1/n je Wert)."""
    z = sorted(set(ref) | set(win))
    nr, nw = len(ref), len(win)
    l1, l2sq, ksv = Fr(0), Fr(0), Fr(0)
    for i, zi in enumerate(z):
        d = abs(Fr(sum(1 for v in ref if v <= zi), nr) - Fr(sum(1 for v in win if v <= zi), nw))
        ksv = max(ksv, d)
        if i + 1 < len(z):
            l1 += d * (z[i + 1] - zi)
            l2sq += d * d * (z[i + 1] - zi)
    return l1, ksv, l2sq


def stat1(ref, win) -> dict:
    return {k: float(v[0]) for k, v in S.statistiken_paare(np.array([ref], dtype=float), np.array([win], dtype=float)).items()}


# ---- 1. Drei Statistiken, eine Differenz ------------------------------------------------------------------------------------------------------

def test_h1_differenz_der_verteilungsfunktionen_von_hand():
    ref, win = [Fr(v) for v in (0, 1, 2, 3)], [Fr(v) for v in (2, 3, 4)]
    l1, ksv, l2sq = cdf_normen(ref, win)
    assert l1 == Fr(1, 4) + Fr(1, 2) + Fr(5, 12) + Fr(1, 3) == Fr(3, 2)         # Differenzen 1/4, 1/2, 5/12, 1/3 auf vier Intervallen der Breite 1
    assert ksv == Fr(1, 2)
    assert l2sq == Fr(1, 16) + Fr(1, 4) + Fr(25, 144) + Fr(1, 9) == Fr(43, 72)


def test_h1_die_statistiken_der_demo_geben_die_handwerte():
    e = stat1([0, 1, 2, 3], [2, 3, 4])
    assert e["W1"] == 1.5 and e["KS"] == 0.5
    assert abs(e["Energie"] - math.sqrt(2 * 43 / 72)) < 1e-14 and abs(e["Energie"] - math.sqrt(43) / 6) < 1e-14
    assert abs(e["|Mittel|"] - 1.5) < 1e-15                    # Mittel 1,5 gegen 3
    assert e["W1"] == abs(sum([2, 3, 4]) / 3 - sum([0, 1, 2, 3]) / 4)      # F liegt überall über G: W1 = |Mittelwertdifferenz|


def test_h1_verschiebung_einer_stichprobe_um_8_gibt_w1_gleich_8():
    x = 100 + 20 * np.random.default_rng(9611).normal(size=50)
    assert abs(stat1(x, x + 8.0)["W1"] - 8.0) < 1e-12


def test_h1_kl_von_hand_bei_vier_gepoolten_werten():
    # Referenz (0, 1), Fenster (2, 3): vier Werte, 10 Klassen: Rang i liegt in Klasse ⌊10 i / 4⌋ = 0, 2, 5, 7; die Referenz belegt die Klassen 0 und 2, das Fenster 5 und 7.
    # Zähler +0,5, Nenner n + 5 = 7: leere Klassen haben 0,5/7, belegte 1,5/7. KL = Σ p ln(p/q)
    p = [1.5 / 7 if k in (0, 2) else 0.5 / 7 for k in range(10)]
    q = [1.5 / 7 if k in (5, 7) else 0.5 / 7 for k in range(10)]
    erwartet = sum(a * math.log(a / b) for a, b in zip(p, q))
    assert abs(erwartet - (2 * 1.5 - 2 * 0.5) / 7 * math.log(3)) < 1e-15       # Klassen 0 und 2: je (1,5/7) ln 3; Klassen 5 und 7: je (0,5/7) ln(1/3); Summe (2/7) ln 3
    assert abs(stat1([0, 1], [2, 3])["KL"] - erwartet) < 1e-14


# ---- 2. Exakter Permutationstest --------------------------------------------------------------------------------------------------------------

def _alle_aufteilungen(werte, n_w):
    z = np.array(sorted(werte), dtype=float)
    teile = list(itertools.combinations(range(len(z)), n_w))
    ind = np.zeros((len(teile), len(z)))
    for i, t in enumerate(teile):
        ind[i, list(t)] = 1.0
    return z, teile, ind


def test_h2_exakter_permutationstest_pool_0_1_3_7_12():
    z, teile, ind = _alle_aufteilungen([0, 1, 3, 7, 12], 2)
    assert len(teile) == math.comb(5, 2) == 10
    w1 = S.statistiken(z[None, :], ind, 2)["W1"]
    erwartet = sorted([49 / 6, 41 / 6, 31 / 6, 5, 29 / 6, 4, 4, 23 / 6, 19 / 6, 3])      # von Hand gerechnet: W1 aller zehn Aufteilungen
    assert np.allclose(sorted(w1), erwartet, atol=1e-13)
    beob = teile.index((3, 4))                                  # Fenster = {7, 12}
    assert abs(w1[beob] - 49 / 6) < 1e-13 and w1[beob] == w1.max()
    andere = np.delete(np.arange(10), beob)
    t_perm = np.stack([w1[andere]], axis=-1)[None, :, :]
    # p der extremsten Aufteilung = 1/10; die zwei Gleichstände (4 = 4) machen den Test vorsichtig
    p = P.p_werte(np.array([[[w1[beob]]]]), t_perm)
    assert abs(p[0, 0] - 1 / 10) < 1e-15
    pw = []
    for i in range(10):
        rest = np.delete(np.arange(10), i)
        pw.append(float(P.p_werte(np.array([[[w1[i]]]]), w1[rest][None, :, None])[0, 0]))
    assert np.allclose(sorted(pw), [k / 10 for k in (1, 2, 3, 4, 5, 7, 7, 8, 9, 10)]), sorted(pw)
    for alpha in (1 / 20, 1 / 10, 1 / 5, 1 / 2):
        assert np.mean(np.array(pw) <= alpha + 1e-12) <= alpha + 1e-12, alpha                  # Gültigkeit: P(p ≤ α) ≤ α, exakt über alle Aufteilungen


def test_h2_kleinster_p_wert_bei_b_199_ist_ein_zweihundertstel():
    t_perm = np.zeros((1, 199, 5))
    p = P.p_werte(np.full((1, 1, 5), 1.0), t_perm)
    assert np.allclose(p, 1 / 200) and Fr(1, 200) == Fr(1, 199 + 1)


def test_alarm_bei_p_gleich_0_05_zaehlt_als_alarm_in_beiden_b():
    for b in (99, 199):
        p = np.array([[(0.05 * (b + 1)) / (b + 1), (0.05 * (b + 1) + 1) / (b + 1)]])       # genau 5 % (5 von 100 beziehungsweise 10 von 200) und eine Stufe darüber
        assert P.alarm_perm(p, b).tolist() == [[True, False]]


# ---- 3. Einzelausreißer: Schranken ------------------------------------------------------------------------------------------------------------

def test_h3_ein_wert_aendert_ks_um_hoechstens_1_durch_nw_und_w1_um_hoechstens_delta_durch_nw():
    ref = [95, 98, 100, 102, 105, 97, 103, 99, 101, 100]
    win = [96, 99, 100, 101, 104]
    win2 = win[:-1] + [1100]
    a, b = cdf_normen([Fr(v) for v in ref], [Fr(v) for v in win]), cdf_normen([Fr(v) for v in ref], [Fr(v) for v in win2])
    assert a[1] == Fr(1, 10) and b[1] == Fr(1, 5) and b[1] - a[1] <= Fr(1, 5)           # KS 1/10 → 1/5, Änderung höchstens 1/n_w = 1/5
    assert a[0] == Fr(3, 5) and b[0] == Fr(998, 5) and b[0] - a[0] == 199 <= Fr(1000, 5)    # W1 3/5 → 998/5, Änderung 199 ≤ Δ/n_w = 200
    ea, eb = stat1(ref, win), stat1(ref, win2)
    assert abs(ea["KS"] - 0.1) < 1e-15 and abs(eb["KS"] - 0.2) < 1e-15
    assert abs(ea["W1"] - 3 / 5) < 1e-12 and abs(eb["W1"] - 998 / 5) < 1e-9


def test_h3_bei_fenster_50_aendert_ein_wert_ks_um_hoechstens_0_02_in_100_zufallsfaellen():
    g = np.random.default_rng(9613)
    schlimmste = 0.0
    for _ in range(100):
        r = 100 + 20 * g.normal(size=100)
        w = 100 + 20 * g.normal(size=50)
        w2 = w.copy()
        w2[int(g.integers(0, 50))] = float(g.choice([-1000.0, 260.0, 1100.0, 5000.0]))
        schlimmste = max(schlimmste, abs(stat1(r, w2)["KS"] - stat1(r, w)["KS"]))
    assert schlimmste <= 1 / 50 + 1e-12 and schlimmste > 0.0


def test_h3_ein_wert_bei_1100_hebt_w1_um_15_bis_20_und_den_mittelwert_um_etwa_20():
    g = np.random.default_rng(9612)
    r, w = 100 + 20 * g.normal(size=100), 100 + 20 * g.normal(size=50)
    k = int(np.argmin(np.abs(w - 100)))
    w2 = w.copy()
    w2[k] = 1100.0
    dw1 = stat1(r, w2)["W1"] - stat1(r, w)["W1"]
    dmean = w2.mean() - w.mean()
    assert 15.0 < dw1 <= 1000 / 50 + 1e-9
    assert abs(dmean - (1100.0 - w[k]) / 50) < 1e-9 and 19.0 < dmean < 21.0


# ---- 4. Der Ausreißer in der Permutation ------------------------------------------------------------------------------------------------------

def test_h4_p_werte_der_aufteilungen_mit_ausreisser_im_fenster_sind_1_15_bis_5_15():
    werte = [98, 99, 100, 101, 102, 1100]
    z, teile, ind = _alle_aufteilungen(werte, 2)              # Referenz 4, Fenster 2: 15 Aufteilungen
    assert len(teile) == 15
    mit = [i for i, t in enumerate(teile) if 5 in t]
    assert len(mit) == 5 and Fr(len(mit), len(teile)) == Fr(2, 6)               # 5 von 15 = n_w / N = 1/3
    mittel = S.statistiken(z[None, :], ind, 2)["|Mittel|"]
    assert min(mittel[mit]) > max(np.delete(mittel, mit))                         # jede Aufteilung mit Ausreißer im Fenster liegt über allen anderen
    pw = []
    for i in mit:
        rest = np.delete(np.arange(15), i)
        pw.append(float(P.p_werte(np.array([[[mittel[i]]]]), mittel[rest][None, :, None])[0, 0]))
    assert np.allclose(sorted(pw), [k / 15 for k in range(1, 6)])
    anteil = [sum(1 for q in pw if q <= k / 15 + 1e-12) / len(pw) for k in range(1, 6)]
    assert np.allclose(anteil, [k / 5 for k in range(1, 6)])                      # P(p ≤ k/15) = k/5 = α · N/n_w, exakt
    assert abs(0.05 * 150 / 50 - 0.15) < 1e-15
    assert abs(sorted(pw)[2] - 1 / 5) < 1e-15 and Fr(2, 6) / 2 == Fr(1, 6)       # Median der p-Werte 1/5 im Spielzeug, (n_w/N)/2 = 1/6 im Grenzfall


# ---- 5. Ankunft im Strom ----------------------------------------------------------------------------------------------------------------------

def test_h5_ankunft_eines_ausreissers_im_strom():
    q = Fr(2, 100)
    for k, v in {25: 0.3965, 50: 0.6358, 100: 0.8674, 150: 0.9517}.items():
        assert abs(float(1 - (1 - q) ** k) - v) < 5e-5, k
    assert abs(float(1 - (1 - Fr(6, 100)) ** 25) - 0.7871) < 5e-5
    assert 1000 / 50 == 20.0                                    # obere Schranke Δ/n_w für W1


# ---- 6. Zwei Gipfel ---------------------------------------------------------------------------------------------------------------------------

def test_h6_mischung_hat_mittelwert_100_und_streuung_20_und_w1_4_13_unter_der_schwelle():
    var = 18 ** 2 + 8.72 ** 2
    assert abs(var - 400.0384) < 1e-9 and abs(math.sqrt(var) - 20.0) < 1e-3
    assert abs(E.gesamtstreuung_zwei(36, 8.72) - math.sqrt(var)) < 1e-12
    w1 = E.w1_grundgesamtheit(36, 8.72)
    assert 4.0 < w1 < 4.3 and w1 < 7.0                          # unter der Schwelle von W1 für ein Fenster von 50 Werten im Normalzustand (Test der Schwelle: test_claims.py)
    assert E.w1_grundgesamtheit(0, 20.0) < 1e-6                 # gleiche Verteilung: Abstand 0


def test_h4_das_gesetz_gilt_nur_fuer_k_bis_5_und_bedingt_auf_den_ausreisser_im_fenster():
    """P(p ≤ α) = α · N/n_w gilt im Spielzeug für α = k/15 nur mit k ≤ 5 (α ≤ 1/3): für k = 6 wäre es 6/5 > 1; über alle 15 Aufteilungen gemittelt ist P(p ≤ α) = α höchstens α (Austauschbarkeit)."""
    werte = [98, 99, 100, 101, 102, 1100]
    z, teile, ind = _alle_aufteilungen(werte, 2)
    mittel = S.statistiken(z[None, :], ind, 2)["|Mittel|"]
    alle = [float(P.p_werte(np.array([[[mittel[i]]]]), mittel[np.delete(np.arange(15), i)][None, :, None])[0, 0]) for i in range(15)]
    for k in range(1, 16):
        assert sum(1 for q in alle if q <= k / 15 + 1e-12) / 15 <= k / 15 + 1e-12         # über alle Aufteilungen: höchstens α
    assert abs(6 / 15 * 150 / 50 - 6 / 5) < 1e-15 and 6 / 5 > 1                           # die Formel α · N/n_w ist bei k = 6 keine Wahrscheinlichkeit mehr
