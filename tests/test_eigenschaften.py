"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Statistiken und Tests gelten müssen, ohne dass man das richtige Ergebnis kennt (20 bis 100 Instanzen mit festem Zufalls-Seed).

- Skalieren aller Werte mit c > 0 ändert KS, KL und die p-Werte nicht und skaliert W1 und |Mittel| mit c und die Energie-Distanz mit √c (sie ist die Wurzel eines Integrals über z); Verschieben aller Werte ändert nichts.
- Vertauschen von Referenz und Fenster ändert W1, KS, Energie und |Mittel| nicht (KL hängt an der Richtung).
- Ein Fensterwert ändert KS um höchstens 1/n_w und W1 um höchstens Δ/n_w.
- Austauschbarkeit: im Normalzustand ist P(p ≤ α) höchstens α (Band).
- Gleiche Zufalls-Seeds geben gleiche Ergebnisse, auch in einem zweiten Prozess mit anderem Hash-Seed; die Blockgröße ändert nichts; die Basiswerte sind in allen Szenarien dieselben.
- Randfälle: Fenster gleich Referenz, konstante Werte, Fenster von einem Wert, B = 0, Gleichstände."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aba_constants as C
import aba_permutation as P
import aba_statistiken as S
import aba_strom as ST
import aba_szenarien as Z

ROOT = Path(__file__).resolve().parent.parent


def _instanzen(n, nr=40, nw=20, seed=31):
    g = np.random.default_rng(seed)
    for _ in range(n):
        yield 100 + 20 * g.normal(size=nr), 105 + 28 * g.normal(size=nw)


def _s(x, y):
    return {k: float(v[0]) for k, v in S.statistiken_paare(x[None, :], y[None, :]).items()}


def test_skalieren_aendert_ks_und_kl_nicht_und_skaliert_w1_mittel_mit_c_und_energie_mit_wurzel_c():
    for x, y in _instanzen(30):
        a, b = _s(x, y), _s(3.5 * x, 3.5 * y)
        assert abs(a["KS"] - b["KS"]) < 1e-12 and abs(a["KL"] - b["KL"]) < 1e-12
        for k in ("W1", "|Mittel|"):
            assert abs(3.5 * a[k] - b[k]) < 1e-9 * max(1.0, abs(b[k])), k
        assert abs(math.sqrt(3.5) * a["Energie"] - b["Energie"]) < 1e-9 * max(1.0, b["Energie"])         # Energie = √(2 ∫ (F − G)² dz): wächst mit √c, nicht mit c


def test_verschieben_aendert_keine_statistik():
    for x, y in _instanzen(30):
        a, b = _s(x, y), _s(x - 250.0, y - 250.0)
        for k in a:
            assert abs(a[k] - b[k]) < 1e-9, k


def test_skalieren_und_verschieben_aendern_die_p_werte_nicht():
    x = np.random.default_rng(1).normal(size=(10, 12))
    y = np.random.default_rng(2).normal(size=(10, 6)) + 0.8
    _, p1, _ = P.permutationstest(x, y, 99, 5, 0)
    _, p2, _ = P.permutationstest(7.0 * x + 40.0, 7.0 * y + 40.0, 99, 5, 0)
    assert np.allclose(p1, p2, atol=1.0 / 100 * 3)              # derselbe Zufall; nur die Gleichstandstoleranz kann einzelne p-Werte um eine Stufe verschieben
    assert np.mean(np.abs(p1 - p2) < 1e-12) > 0.9


def test_vertauschen_von_referenz_und_fenster_aendert_w1_ks_energie_und_mittel_nicht():
    for x, y in _instanzen(30, nr=35, nw=15):
        a, b = _s(x, y), _s(y, x)
        for k in ("W1", "KS", "Energie", "|Mittel|"):
            assert abs(a[k] - b[k]) < 1e-12, k
        assert a["KL"] >= 0 and b["KL"] >= 0 and abs(a["KL"] - b["KL"]) > 1e-9          # KL hängt an der Richtung


def test_ein_fensterwert_aendert_ks_um_hoechstens_1_durch_nw_und_w1_um_hoechstens_delta_durch_nw():
    g = np.random.default_rng(77)
    for _ in range(100):
        x, y = 100 + 20 * g.normal(size=60), 100 + 20 * g.normal(size=25)
        y2 = y.copy()
        j = int(g.integers(0, 25))
        y2[j] = y[j] + float(g.choice([-300.0, 40.0, 900.0]))
        a, b = _s(x, y), _s(x, y2)
        assert abs(a["KS"] - b["KS"]) <= 1 / 25 + 1e-12
        assert abs(a["W1"] - b["W1"]) <= abs(y2[j] - y[j]) / 25 + 1e-9
        assert abs(a["|Mittel|"] - b["|Mittel|"]) <= abs(y2[j] - y[j]) / 25 + 1e-9


def test_energie_ist_nicht_kleiner_als_null_und_w1_ist_nicht_kleiner_als_der_mittelwertabstand():
    for x, y in _instanzen(50):
        e = _s(x, y)
        assert e["Energie"] >= 0 and e["KS"] >= 0 and e["KL"] >= -1e-15 and e["W1"] >= e["|Mittel|"] - 1e-9      # |∫(F − G)| ≤ ∫|F − G|


@pytest.mark.parametrize("b", [99, 199])
def test_austauschbarkeit_p_wert_ist_im_normalzustand_hoechstens_alpha_im_band(b):
    r = P.perm_lauf(Z.Szenario("normal"), 600, b, 4711)
    for alpha in (0.05, 0.10, 0.25):
        anteil = (r["p"] <= alpha + 1e-9).mean(axis=0)
        assert np.all(anteil <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / 600)), (alpha, anteil)


def test_gleicher_zufalls_seed_gleiches_ergebnis_und_blockgroesse_ohne_einfluss(monkeypatch):
    szen = Z.Szenario("streuung", "Normal", 1.5)
    a = P.perm_lauf(szen, 120, 99, 8)
    b = P.perm_lauf(szen, 120, 99, 8)
    assert np.array_equal(a["p"], b["p"]) and np.array_equal(a["t"], b["t"])
    monkeypatch.setattr(P, "BLOCK_PERM", 7)
    c = P.perm_lauf(szen, 120, 99, 8)
    assert np.array_equal(a["p"], c["p"]) and np.array_equal(a["t"], c["t"])
    monkeypatch.setattr(ST, "BLOCK_STROM", 5)
    d = ST.strom_verlauf(szen, 13, 10, 8)
    monkeypatch.setattr(ST, "BLOCK_STROM", 250)
    assert np.array_equal(d, ST.strom_verlauf(szen, 13, 10, 8))


def test_ein_kuerzerer_lauf_ist_ein_praefix_des_laengeren():
    szen = Z.Szenario("ausreisser", "t(3)", 3, 8)
    kurz, lang = P.perm_lauf(szen, 50, 99, 8), P.perm_lauf(szen, 100, 99, 8)
    assert np.array_equal(kurz["p"], lang["p"][:50])
    assert np.array_equal(ST.strom_verlauf(szen, 20, 30, 8), ST.strom_verlauf(szen, 40, 30, 8)[:20])
    assert np.array_equal(ST.strom_verlauf(szen, 20, 30, 8), ST.strom_verlauf(szen, 20, 60, 8)[:, :31])      # Strom bis Horizont 30 ist ein Präfix des Stroms bis 60


def test_zweiter_prozess_mit_anderem_hash_seed_rechnet_dasselbe():
    code = ("import sys, json; sys.path.insert(0, r'%s');"
            "import numpy as np, aba_permutation as P, aba_szenarien as Z;"
            "r = P.perm_lauf(Z.Szenario('mittel', 'Normal', 0.4), 40, 99, 21);"
            "print(json.dumps([r['p'].sum(), float(r['t'].sum())]))") % ROOT
    aus = []
    for hs in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hs, OMP_NUM_THREADS="1")
        aus.append(json.loads(subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env=env, timeout=120, check=True).stdout))
    assert aus[0][0] == aus[1][0] and abs(aus[0][1] - aus[1][1]) < 1e-9 * abs(aus[0][1])


def test_die_basiswerte_sind_in_allen_szenarien_dieselben_und_nur_das_fenster_aendert_sich():
    ref0, fen0 = Z.paare(5, 0, 8, Z.Szenario("normal"))
    for szen in (Z.Szenario("mittel", "Normal", 0.4), Z.Szenario("streuung", "Normal", 1.5), Z.Szenario("ausreisser", "Normal", 3, 8), Z.Szenario("zwei", "Normal", 36, 8.72)):
        ref, fen = Z.paare(5, 0, 8, szen)
        assert np.array_equal(ref, ref0), szen
        assert not np.array_equal(fen, fen0), szen


# ---- Randfälle ---------------------------------------------------------------------------------------------------------------------------------

def test_fenster_gleich_referenz_gibt_null_und_den_p_wert_eins():
    x = np.random.default_rng(3).normal(size=(4, 30))
    e = S.statistiken_paare(x, x.copy())
    for k in ("W1", "KS", "Energie", "|Mittel|"):
        assert np.allclose(e[k], 0.0, atol=1e-12), k
    assert np.allclose(e["KL"], 0.0, atol=1e-15)                 # gleiche Werte stehen im sortierten Pool paarweise nebeneinander: jede Klasse (6 Werte) hat 3 und 3
    _, p, _ = P.permutationstest(x[:, :20], x[:, :20].copy(), 199, 1, 0)
    assert np.all(p == 1.0)                                      # Statistik 0 bei der Beobachtung: jede Vertauschung ist größer oder gleich


def test_konstante_werte_geben_null_und_keinen_alarm_ohne_fehler():
    ref, fen = np.full((3, 20), 100.0), np.full((3, 10), 100.0)
    e = S.statistiken_paare(ref, fen)
    for k in ("W1", "KS", "Energie", "|Mittel|"):
        assert np.all(e[k] == 0.0), k
    t, p, _ = P.permutationstest(ref, fen, 99, 1, 0)
    assert np.all(p == 1.0) and not P.alarm_perm(p, 99).any()


def test_fenster_von_einem_wert_und_b_gleich_null():
    x = np.random.default_rng(4).normal(size=(5, 25))
    y = np.random.default_rng(5).normal(size=(5, 1)) + 10
    e = S.statistiken_paare(x, y)
    assert np.all(np.isfinite(e["W1"])) and np.all(e["KS"] > 0)
    t, p, gross = P.permutationstest(x, y, 99, 2, 0)
    assert p.min() >= 1 / 100 and np.all(np.isfinite(gross))
    _, p0, _ = P.permutationstest(x, y, 0, 2, 0)
    assert np.all(p0 == 1.0) and not P.alarm_perm(p0, 0).any()              # ohne Vertauschung ist p = 1 und es gibt nie Alarm


def test_gleichstaende_in_den_werten_brechen_nichts():
    g = np.random.default_rng(6)
    x, y = g.integers(0, 5, size=(6, 30)).astype(float), g.integers(1, 6, size=(6, 12)).astype(float)
    t, p, _ = P.permutationstest(x, y, 99, 3, 0)
    assert np.all(np.isfinite(t)) and np.all((p > 0) & (p <= 1))
    z, ind = S.sortieren(x, y)
    assert np.all(np.diff(z, axis=-1) >= 0) and np.all(ind.sum(axis=-1) == 12)


def test_ein_szenario_ohne_aenderung_meldet_keine_wirkung():
    r = P.perm_lauf(Z.Szenario("zwei", "Normal", 0.0, 20.0), 600, 99, 8)         # Mischung zweier gleicher N(100, 20²): keine Änderung
    raten = P.alarm_perm(r["p"], 99).mean(axis=0)
    assert np.all(raten < 0.05 + 3 * math.sqrt(0.05 * 0.95 / 600) + 0.01), raten
