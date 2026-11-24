"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Dieser Test macht vier Dinge:
(1) die Tabellen und Sätze der README werden aus der Datei `tools/messreihen.json` mit denselben Formatfunktionen wie in der App gebildet (`tests/readme_zeilen.py`) und müssen wörtlich in der README stehen;
(2) die Aussagen, die die README über diese Zahlen macht („W1 gewinnt bei Streuung“, „Fehlalarmprobe im Band“), werden an der ganzen Tabelle der Datei gezählt;
(3) die Zeilen der Datei werden neu gerechnet und als Band verglichen (die Datei ist fest; ein Neu-Rechnen auf einer anderen Plattform kann in der letzten Binärstelle abweichen);
(4) Konstanten, Kantenwortlaut, Testzahl, verbotene Wörter."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import aba_constants as C
import aba_evaluation as E
import aba_permutation as P
import aba_statistiken as S
import aba_strom as ST
import aba_szenarien as Z
import readme_zeilen as R

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
SE = math.sqrt(0.05 * 0.95 / 1000)


def _perm(key, seed=C.SERIEN_SEED):
    return R.perm(key, seed)


def test_tabellen_und_saetze_der_readme():
    for name, f in R.STUECKE.items():
        assert f() in README, name


def test_konstanten_der_readme():
    for text in ("Zufalls-Seed 2026", "6 000 Normalströme", "20 000 Paare", "$B = 199$", "1 000 Zufalls-Seeds", "Standardfehler höchstens 1,6 Prozentpunkte", "$x_1, \\dots, x_{100}$", "$y_1, \\dots, y_{50}$"):
        assert text in README, text
    assert C.N_REF == 100 and C.N_FENSTER == 50 and C.N_POOL == 150 and C.MITTEL == 100.0 and C.SIGMA == 20.0 and C.KLASSEN == 10 and C.GLAETTUNG_KL == 0.5
    assert C.SERIEN_SEED == 2026 and C.N_SCHWELLE == 20000 and C.N_STROM_SCHWELLE == 6000 and C.H_STANDARD == 150 and C.B_OPT[-1] == 199 and C.SEEDS_OPT[-1] == 1000 and C.ALPHA == 0.05
    assert abs(math.sqrt(0.25 / 1000) - 0.0158) < 1e-4                           # Standardfehler einer Alarmrate bei 1 000 Zufalls-Seeds: höchstens 1,6 Prozentpunkte
    assert MESS["meta"] == {"seed": 2026, "seeds": 1000, "b": 199, "n_schwelle": 20000, "n_strom_schwelle": 6000, "h_standard": 150, "stroeme": 1000, "wiederholungen": [2027, 2028, 2029]}


def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"- **Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)." in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    for text in (README, APP):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "Plan-", "PLAN-", "wasserstein-demo", "Neuheit ist", "erstmals", "H 3"):
            assert verboten not in text, verboten
        assert not re.search(r"\b(trägt|tragen)\b", text), "„trägt“ und „tragen“ vermeiden"
        assert "Zufallsseed" not in text and "Random Seed" not in text
    assert "TODO " not in APP
    assert "Orakel" not in APP                                                    # die Orakel-Wörter gehören in Tests, nicht in App-Texte


def test_readme_nennt_die_kante_im_wortlaut_und_die_wortregel():
    satz = "Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle."
    assert f"*{satz}*" in README and satz in APP
    assert "**Wortregel.** „Tageswert“ heißt hier ein Wert je Tag" in README and "ein **Tagesprofil** (die Verteilung über 24 Stunden) kommt in dieser Demo nicht vor" in README
    assert "„Tageswert“ heißt hier ein Wert je Tag" in APP and "**Tagesprofil**" in APP


def test_die_zahl_zu_test_tabelle_nennt_vorhandene_tests():
    for datei, name in (("test_claims.py", "test_tabellen_und_saetze_der_readme"), ("test_claims.py", "test_zeilen_der_datei_neu_gerechnet_im_band"),
                        ("test_oracle_abstand_alarm.py", "test_zahlen_der_unabhaengigen_nachrechnung_liegen_im_band")):
        assert f"def {name}" in (ROOT / "tests" / datei).read_text(encoding="utf-8"), name
        assert f"`{name}`" in README, name
    for datei in ("test_handbeispiele.py", "test_oracle_abstand_alarm.py", "test_eigenschaften.py", "test_zufall.py", "test_szenarien.py", "test_evaluation.py", "test_claims.py"):
        assert (ROOT / "tests" / datei).exists() and f"`tests/{datei}`" in README, datei


def test_zaehlungen_der_verifikation_stehen_in_den_tests():
    quelle = {d: (ROOT / "tests" / d).read_text(encoding="utf-8") for d in ("test_oracle_abstand_alarm.py", "test_eigenschaften.py", "test_handbeispiele.py")}
    o, h, e = quelle["test_oracle_abstand_alarm.py"], quelle["test_handbeispiele.py"], quelle["test_eigenschaften.py"]
    assert "_paare(40)" in o and "def _paare(n=40)" in o and "for _ in range(20)" in o and "for j in range(30)" in o and "itertools.combinations(range(10), 4)" in o and "math.comb(5, 2)" in h
    assert "((36, 8.72), (20, 8.72), (40, 6.0), (10, 14.0))" in o
    assert 'Z.Szenario("normal"), Z.Szenario("streuung", "Normal", 1.5), Z.Szenario("ausreisser", "t(3)", 3, 8)' in o and "< 1e-12" in o
    assert "for _ in range(100)" in h and "for _ in range(100)" in e
    for text in ("auf 40 Zufallspaaren", "20 Paaren mit Gleichständen", "(40 Paare, kleine Pools, Gleichstände)", "aller 210 Aufteilungen", "(30 Paare)", "(3 Szenarien, gleich auf 1e-12)", "(4 Mischungen)", "Schranken für einen Wert (100 Fälle)"):
        assert text in README, text


# ---- Aussagen über die ganze Tabelle der Datei --------------------------------------------------------------------------------------------

def test_w1_ist_bei_streuung_und_ausreissern_unter_den_besten_und_in_drei_von_vier_zeilen_allein():
    zeilen = ["Normal|streuung|1.5|0", "Normal|ausreisser|3|8", "Normal|ausreisser|1|50", "t(3)|streuung|1.5|0"]
    for key in zeilen:
        for schluessel in ("nicht_schlechter_perm", "nicht_schlechter_sauber"):
            assert _perm(key)[schluessel][0] is True, (key, schluessel)             # W1 ist nicht schlechter als die beste
    allein = [key for key in zeilen if _perm(key)["nicht_schlechter_perm"] == [True, False, False, False, False]]
    assert allein == ["Normal|streuung|1.5|0", "Normal|ausreisser|3|8", "t(3)|streuung|1.5|0"]
    assert _perm("Normal|ausreisser|1|50")["nicht_schlechter_perm"] == [True, False, False, False, True]           # W1 und |Mittel|


def test_bei_der_gestalt_liegt_kl_vorn_und_bei_t3_mit_mittelwertverschiebung_energie():
    assert _perm("Normal|zwei|36|8.72")["nicht_schlechter_perm"] == [False, False, False, True, False]
    assert _perm("Normal|zwei|36|8.72")["nicht_schlechter_sauber"] == [False, False, False, True, False]
    a = _perm("t(3)|mittel|0.5|0")["alarm_perm"]
    assert _perm("t(3)|mittel|0.5|0")["nicht_schlechter_perm"] == [False, False, True, False, False] and a[2] > a[1] > a[0] > a[4] > a[3]
    m = _perm("Normal|mittel|0.4|0")["nicht_schlechter_perm"]
    assert m == [True, False, False, False, True]


def test_fehlalarmprobe_aller_zehn_zahlen_liegt_im_band_5_prozent_plus_minus_3_standardfehler():
    for key in ("Normal|normal|0|0", "t(3)|normal|0|0"):
        for schluessel in ("alarm_perm", "alarm_sauber"):
            assert all(abs(w - 0.05) < 3 * SE for w in _perm(key)[schluessel]), (key, schluessel, _perm(key)[schluessel])
    for key in ("Normal|normal|0|0", "t(3)|normal|0|0"):
        assert all(abs(w - 0.05) < 3 * SE for w in R.strom(key)["kontrolle"]), key


def test_ks_erkennt_den_einzelausreisser_nicht_ueber_der_fehlalarmrate_in_allen_vier_laeufen_und_beiden_schwellen():
    for seed in (C.SERIEN_SEED, 2027, 2028, 2029):
        z = _perm("Normal|ausreisser|1|50", seed)
        assert z["alarm_perm"][1] < 0.05 + 3 * SE and z["alarm_sauber"][1] < 0.05 + 3 * SE, seed
    assert abs(MESS["schwellen_fenster"]["Normal"]["KS"] - 0.23) < 1e-9 and abs(MESS["schwellen_strom"]["Normal"]["150"]["KS"] - 0.30) < 1e-9          # Schwellen 0,23 und 0,30 gegen die Schranke 1/n_w = 0,02
    assert 1 / C.N_FENSTER == 0.02


def test_einzelausreisser_im_permutationstest_etwa_15_prozent_und_mit_schwelle_100_prozent():
    seeds = (C.SERIEN_SEED, 2027, 2028, 2029)
    w1 = [_perm("Normal|ausreisser|1|50", s)["alarm_perm"][0] for s in seeds]
    assert all(abs(w - 0.15) < 3 * math.sqrt(0.15 * 0.85 / 1000) for w in w1), w1                         # Rechnung 5 % · N/n_w = 15 %, drei Standardfehler 3,4 Prozentpunkte
    z = _perm("Normal|ausreisser|1|50")
    assert z["alarm_sauber"][0] == 1.0 and z["alarm_sauber"][4] == 1.0
    assert 0.15 < z["p_median"][0] < 0.19 and abs(z["gross"] - 1 / 3) < 0.01                               # Median der p-Werte ungefähr 1/6, der größte Wert in einem Drittel der Vertauschungen im Fenster


def test_zweigipfelig_erkennt_keine_statistik_im_strom_bis_k_50_in_mehr_als_einem_sechstel_und_kl_ist_der_groesste_wert():
    z = R.strom("Normal|zwei|36|8.72")["marken"]["50"]
    assert max(z) == z[3] < 1 / 6 and z[0] < 0.05


def test_gestalt_w1_der_verteilungen_liegt_unter_der_schwelle_eines_fensters():
    assert 4.0 < E.w1_grundgesamtheit(36, 8.72) < 4.3 and E.w1_grundgesamtheit(36, 8.72) < MESS["schwellen_fenster"]["Normal"]["W1"]


def test_kontrollzahlen_und_normalzustand_im_strom_liegen_bei_k_0_unter_einem_prozent():
    for key in ("Normal|normal|0|0", "t(3)|normal|0|0"):
        assert max(R.strom(key)["kontrolle_k0"]) < 0.01


def test_schwellenstreuung_der_strom_schwellen_ist_klein_aber_nicht_null():
    d = MESS["schwellenstreuung"]
    w1 = [d[s]["schwelle"]["W1"] for s in d]
    assert len(d) == 3 and 0 < max(w1) / min(w1) - 1 < 0.03
    a = [d[s]["alarm_k150"][0] for s in d]
    assert max(a) - min(a) < 0.03 and all(0.85 < x < 0.92 for x in a)


# ---- Neu-Rechnen im Band ---------------------------------------------------------------------------------------------------------------------

def test_zeilen_der_datei_neu_gerechnet_im_band():
    """Zwei Zeilen des Permutationstests und die Schwelle „sauber“ werden neu gerechnet (Band ±1,5 Prozentpunkte beziehungsweise ±1 %: auf einer anderen Plattform kann ein p-Wert an der Grenze kippen)."""
    schwellen = P.schwelle_sauber("Normal", C.SERIEN_SEED)
    for s in C.STATISTIKEN:
        assert abs(schwellen[s] / MESS["schwellen_fenster"]["Normal"][s] - 1) < 0.01, s
    for szen in (Z.Szenario("streuung", "Normal", 1.5), Z.Szenario("ausreisser", "Normal", 1, 50)):
        r = P.perm_lauf(szen, 1000, 199, C.SERIEN_SEED)
        neu = E.perm_zusammenfassung(r, 199, MESS["schwellen_fenster"]["Normal"])
        alt = MESS["perm"][f"{szen.schluessel()}|1000|199|{C.SERIEN_SEED}"]
        for k in ("alarm_perm", "alarm_sauber"):
            assert np.all(np.abs(np.array(neu[k]) - np.array(alt[k])) <= 0.015), (szen, k, neu[k], alt[k])
        assert neu["nicht_schlechter_perm"] == alt["nicht_schlechter_perm"] and abs(neu["gross"] - alt["gross"]) < 0.002


def test_strom_zeile_der_datei_neu_gerechnet_im_band_und_die_strom_schwelle_aus_zwei_tausend_stroemen():
    szen = Z.Szenario("mittel", "Normal", 0.4)
    thr = MESS["schwellen_strom"]["Normal"]["150"]
    neu = E.strom_zusammenfassung(ST.strom_verlauf(szen, 1000, 150, C.SERIEN_SEED), ST.strom_verlauf(Z.Szenario("normal"), 1000, 150, C.SERIEN_SEED, C.Z_KONTROLLE), thr, 150)
    alt = MESS["strom"][f"{szen.schluessel()}|1000|150|{C.SERIEN_SEED}"]
    assert np.all(np.abs(np.array(neu["marken"]["150"]) - np.array(alt["marken"]["150"])) <= 0.015)
    t2 = ST.strom_verlauf(Z.Szenario("normal"), 2000, 150, 4321, C.Z_STROM_SCHWELLE)               # andere Normalströme, ein Drittel so viele
    thr2 = ST.schwellen_je_horizont(t2, (150,))[150]
    for s in ("W1", "|Mittel|", "Energie", "KL"):
        assert abs(thr2[s] / thr[s] - 1) < 0.05, (s, thr2[s], thr[s])


def test_ein_wert_bei_1100_hebt_w1_um_fast_20_und_ks_um_hoechstens_0_02():
    ref, fen = Z.paare(C.SERIEN_SEED, 0, 1000, Z.Szenario("normal"))
    fen2 = fen.copy()
    fen2[:, 0] = 1100.0
    a, b = S.statistiken_paare(ref, fen), S.statistiken_paare(ref, fen2)
    assert 18.0 < float((b["W1"] - a["W1"]).mean()) <= 20.0 and float((b["KS"] - a["KS"]).max()) <= 0.02 + 1e-12


# ---- Befunde der Aussagenprüfung (2026-11-24) ----------------------------------------------------------------------------------------------

def test_w1_bei_streuung_mit_4000_zufalls_seeds_liegt_unter_der_zeile_mit_1000_und_die_rangfolge_bleibt():
    d = MESS["streuung_lang"]
    laeufe = [k for k in d if k != "zahl"]
    assert d["zahl"] == 4000 and laeufe == ["2027", "2028", "2029"]
    w1 = [d[k]["alarm_perm"][0] for k in laeufe]
    assert all(0.52 < w < 0.58 for w in w1) and max(w1) < _perm("Normal|streuung|1.5|0")["alarm_perm"][0]           # die 57,6 % der Zeile liegen über allen drei Läufen
    for k in laeufe:
        a = d[k]["alarm_perm"]
        assert a[0] > a[2] > a[3] > a[1] > a[4], k                                                                    # W1 vor Energie vor KL vor KS vor |Mittel|
    assert R.streuung_lang_satz() in README


def test_einzelausreisser_bei_kleiner_hoehe_liegt_unter_der_rechnung_15_prozent():
    h = MESS["hoehen"]
    w1 = {k: h[k]["alarm_perm"][0] for k in h}
    mi = {k: h[k]["alarm_perm"][4] for k in h}
    assert w1["4"] < 0.08 and w1["8"] < 0.11 and 0.12 < w1["20"] < 0.17 and w1["4"] < w1["8"] < w1["20"] < _perm("Normal|ausreisser|1|50")["alarm_perm"][0]
    assert mi["4"] < 0.08 and mi["8"] < 0.10 and mi["20"] < 0.15
    assert R.hoehen_satz() in README


def test_einzelausreisser_ks_energie_kl_liegen_mit_beiden_schwellen_unter_22_prozent_und_w1_mittel_bei_100_prozent():
    z = _perm("Normal|ausreisser|1|50")
    assert max(z[k][j] for k in ("alarm_perm", "alarm_sauber") for j in (1, 2, 3)) < 0.22 and z["alarm_sauber"][0] == 1.0 and z["alarm_sauber"][4] == 1.0
    assert "höchstens zu 22 %" in APP and "nur zu etwa 13 % bis 16 %" in APP
    w1 = [_perm("Normal|ausreisser|1|50", s)["alarm_perm"][0] for s in (C.SERIEN_SEED, 2027, 2028, 2029)]
    mi = [_perm("Normal|ausreisser|1|50", s)["alarm_perm"][4] for s in (C.SERIEN_SEED, 2027, 2028, 2029)]
    assert 0.125 < min(w1 + mi) and max(w1 + mi) < 0.165                                                               # „etwa 13 % bis 16 %“ der App
    assert "die Antwort ändert sich mit der Schwelle, nicht mit der Statistik" not in README + APP


def test_ks_fenster_im_normalzustand_knapp_unter_der_schwelle_koennten_darueber_rutschen():
    """Die Schranke 0,02 erklärt allein nicht, dass KS nicht über der Fehlalarmrate liegt: etwa 4 % der Fenster im Normalzustand haben KS in (0,21; 0,23]."""
    ref, fen = Z.paare(C.SERIEN_SEED, 0, 1000, Z.Szenario("normal"))
    ks = S.statistiken_paare(ref, fen)["KS"]
    anteil = float(((ks > 0.23 - 0.02) & (ks <= 0.23)).mean())
    assert 0.02 < anteil < 0.07, anteil
    assert "erklärt allein aber nicht" in README and "erklärt allein aber nicht" in APP and "das erklärt, warum die Alarmrate von KS" not in README + APP


def test_zeilen_ohne_erkennbare_aenderung_haben_in_der_spalte_nicht_schlechter_einen_strich():
    for tabelle in (R.tabelle_a(), R.tabelle_b()):
        for zeile in tabelle.splitlines():
            if zeile.startswith("| Normalzustand") or zeile.startswith("| t(3): Normalzustand"):
                assert zeile.endswith("| – |"), zeile
            elif zeile.startswith("| ") and "Szenario" not in zeile:
                assert not zeile.endswith("| – |"), zeile
    assert "steht „–“: dort gibt es nichts zu erkennen" in README


def test_median_k_steht_nur_bei_mehr_als_der_haelfte_der_stroeme_mit_alarm():
    for name, key in R.ZEILEN:
        z = R.strom(key)
        zeile = next(s for s in R.tabelle_strom().splitlines() if s.startswith(f"| {name} |"))
        if z["marken"]["150"][0] > 0.5:
            assert not zeile.endswith("| – |"), name
        else:
            assert zeile.endswith("| – |"), name
    assert sum(1 for name, key in R.ZEILEN if R.strom(key)["marken"]["150"][0] <= 0.5) == 3          # Normalzustand, zweigipfelig, t(3): Normalzustand


def test_kontrolle_der_strom_schwelle_liegt_zwischen_4_6_und_6_5_prozent_wie_die_readme_sagt():
    werte = [x for key in ("Normal|normal|0|0", "t(3)|normal|0|0") for x in R.strom(key)["kontrolle"]]
    assert round(100 * min(werte), 1) == 4.6 and round(100 * max(werte), 1) == 6.5
    assert "(4,6 % bis 6,5 %)" in README and "auf **5 % Fehlalarm über den Horizont** eingestellt" in README


def test_gestaltsatz_nennt_w1_kaum_und_kl_etwa_die_haelfte():
    z, st = _perm("Normal|zwei|36|8.72"), R.strom("Normal|zwei|36|8.72")
    w1 = [z["alarm_perm"][0], z["alarm_sauber"][0], st["marken"]["150"][0]]
    kl = [z["alarm_perm"][3], z["alarm_sauber"][3], st["marken"]["150"][3]]
    assert max(w1) < 0.13 and 0.45 < min(kl) and max(kl) < 0.65
    assert R.gestalt_deutung() in README


def test_literatur_nennt_nur_was_geprueft_ist():
    for text in (README, APP):
        assert "Text gelesen" not in text and "Auszug gelesen" not in text
        assert "nicht gegen den Text geprüft" in text and "nur die Zusammenfassung gelesen" in text
        assert "braucht eine in der Dimension exponentiell" not in text


def test_zufall_aussagen_behaupten_keinen_linux_lauf():
    for text in (README, APP):
        assert "auf Windows und Linux bitgleich" not in text and "ohne Einfluss auf die" not in text
        assert "ein Lauf auf Linux ist nicht nachgemessen" in text


def test_aussagen_zu_w1_nennen_die_ausnahme_beim_einzelausreisser():
    assert "bei Streuung und Ausreißern" not in README and "bei Streuung und Ausreißern" not in APP
    for text in (README, APP):
        assert "KL ist ohne Glättung bei getrennten Trägern unendlich" in text and "KL wird unendlich" not in text
    assert "mit |Mittel| gleichauf" in README and "nur mit |Mittel| gleichauf" in APP
