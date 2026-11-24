"""Verdikte (reine Funktionen), gepaarter Vergleich, Formatierung und Bezeichnungen, an handgesetzten Werten und an den Zeilen der Messreihe (DEMO-PLAYBOOK §12, §11a Punkt 7: je Preset die erwartete Verdikt-Zeile)."""

import numpy as np
import pytest

import aba_constants as C
import aba_evaluation as E
import aba_szenarien as Z


def _settings(preset: str) -> E.Settings:
    return E.Settings(**C.PRESETS[preset])


def _zeilen(s: E.Settings):
    szen = E.szenario_von(s)
    return szen, E.perm_ergebnis(szen, s.seeds, s.b_perm, s.seed_input), E.perm_ergebnis(szen.als_normal(), s.seeds, s.b_perm, s.seed_input)


# ---- Formatierung --------------------------------------------------------------------------------------------------------------------

def test_formatfunktionen():
    assert E.zahl(0.2549, 2) == "0,25" and E.zahl(-0.0001, 2) == "0,00" and E.zahl(-1.5, 1) == "−1,5" and E.zahl(float("inf")) == "∞" and E.zahl(None) == "–" and E.zahl(float("nan")) == "–"
    assert E.prozent(0.546) == "54,6 %" and E.prozent(1.0, 0) == "100 %" and E.ganz(1000) == "1 000" and E.ganz(20000) == "20 000"
    assert E.md_zelle("a|b") == "a\\|b"
    assert E.tabelle_md(["x|y", "z"], [["|Mittel|", "1"]]) == "| x\\|y | z |\n|---|---|\n| \\|Mittel\\| | 1 |"


def test_bezeichnungen_aller_arten():
    assert E.bezeichnung(Z.Szenario("normal", "t(3)")) == "Normalzustand (t(3))"
    assert E.bezeichnung(Z.Szenario("mittel", "Normal", 0.4)) == "Mittelwert +0,4 s (Normal)"
    assert E.bezeichnung(Z.Szenario("streuung", "Normal", 1.5)) == "Streuung ×1,5 (Normal)"
    assert E.bezeichnung(Z.Szenario("streuung", "Normal", 1.25)) == "Streuung ×1,25 (Normal)"
    assert E.bezeichnung(Z.Szenario("ausreisser", "Normal", 3, 8)) == "3 von 50 Werten bei +8 s (Normal)"
    assert E.bezeichnung(Z.Szenario("ausreisser", "t(3)", 1, 50)) == "1 von 50 Werten bei +50 s (t(3))"
    assert E.bezeichnung(Z.Szenario("zwei", "Normal", 36, 8.72)) == "zweigipfelig (Gipfel bei 82 und 118, Streuung je Gipfel 8,72)"


def test_settings_aus_zustand_und_standard_preset_gleich_standardeinstellung():
    assert E.settings_aus_zustand(dict(C.DEFAULTS)) == E.Settings()
    assert E.szenario_von(E.Settings()) == Z.Szenario("streuung", "Normal", 1.5)       # Standardeinstellung: Streuung ×1,5, Normal
    assert E.Settings(**C.PRESETS["Streuung ×1,5"]) == E.Settings()


# ---- gepaarter Vergleich -----------------------------------------------------------------------------------------------------------------

def test_nicht_schlechter_an_handgesetzten_alarmfolgen():
    # 100 Zufalls-Seeds; Statistik 0 alarmiert immer, die anderen verfehlen j der 100: der Unterschied ist ein Vorsprung von j/100 mit Standardfehler √(j/100 (1 − j/100)/100) (Differenz hat j Einsen)
    def alarm(fehlt):
        a = np.ones((100, 5), dtype=bool)
        for spalte, j in enumerate(fehlt):
            a[:j, spalte] = False
        return a

    ok = E.nicht_schlechter(alarm([0, 0, 3, 4, 20]))
    assert ok == [True, True, True, False, False]           # 3 Fehlende: 0,03 ≤ 2 · 0,0171; 4 Fehlende: 0,04 > 2 · 0,0196
    assert E.nicht_schlechter(alarm([2, 0, 0, 0, 0])) == [True] * 5         # 2 Fehlende: 0,02 ≤ 2 · 0,0141, noch nicht schlechter
    assert E.nicht_schlechter(np.zeros((50, 5), dtype=bool)) == [True] * 5         # nie Alarm: alle gleich


def test_nicht_schlechter_ist_gepaart_nicht_nur_an_den_raten():
    # gleiche Rate 50 %, aber gegenläufig: gepaart ist der Unterschied null (Mittel der Differenz 0), also nicht schlechter in beide Richtungen
    a = np.zeros((100, 2), dtype=bool)
    a[:50, 0] = True
    a[50:, 1] = True
    assert E.nicht_schlechter(a) == [True, True]


# ---- Verdikte je Preset ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("preset,beste,nicht_schlechter", [
    ("Mittelwert +0,4 s", "|Mittel|", "W1 und |Mittel|"),
    ("Streuung ×1,5", "W1", "W1."),
    ("Ausreißerblock", "W1", "W1."),
    ("Einzelausreißer", "W1", "W1 und |Mittel|"),
    ("zweigipfelig", "KL", "KL."),
    ("t(3)", "Energie", "Energie."),
])
def test_verdikt1_je_preset(preset, beste, nicht_schlechter):
    s = _settings(preset)
    szen, z, zn = _zeilen(s)
    kind, text = E.verdikt1(szen, z, zn)
    assert kind == "success"
    assert E.bezeichnung(szen) in text and "Permutationstest (B = 199), 1 000 Zufalls-Seeds" in text
    assert f"Am häufigsten alarmiert {beste};" in text, text
    assert f"(gepaart, zwei Standardfehler): {nicht_schlechter}" in text, text
    assert "Fehlalarmrate im Normalzustand (" in text and "(Soll 5 %)" in text


def test_verdikt1_nennt_alle_fuenf_alarmraten_mit_den_zahlen_der_messreihe():
    szen, z, zn = _zeilen(_settings("Streuung ×1,5"))
    text = E.verdikt1(szen, z, zn)[1]
    for name, wert in zip(C.STATISTIKEN, z["alarm_perm"]):
        assert f"{name} {E.prozent(wert)}" in text


def test_verdikt1_meldet_bei_einem_szenario_ohne_aenderung_keine_wirkung():
    s = E.Settings(szenario="zweigipfelig", gipfel_abstand=0, gipfel_sigma=20.0)       # zwei gleiche Gipfel bei 100 mit Streuung 20: der Normalzustand
    szen, z, zn = _zeilen(s)
    kind, text = E.verdikt1(szen, z, zn)
    assert kind == "warning" and "keine Änderung zu erkennen" in text
    assert E.verdikt2(szen, z, zn)[0] == "warning"
    assert E.verdikt4(0.0, 20.0, E.w1_grundgesamtheit(0, 20.0), 7.3, z, E.strom_ergebnis(szen, 200, 50, 2026), 50)[0] == "warning"


def test_verdikt2_einzelausreisser_nennt_die_rechnung_15_prozent_und_beide_schwellen():
    szen, z, zn = _zeilen(_settings("Einzelausreißer"))
    kind, text = E.verdikt2(szen, z, zn)
    assert kind == "success"
    assert "≈ 5 % · N/n_w = 15 %" in text and "Der größte Wert liegt in 33,4 % der Vertauschungen im Fenster (n_w/N = 33,3 %)" in text
    assert "Schwelle aus dem Normalzustand W1 100,0 %" in text and "Permutationstest W1 16,2 %" in text
    assert "der Normalzustand ist bekannt" in text


def test_verdikt2_bei_anderem_szenario_hat_keine_ausreisserrechnung():
    szen, z, zn = _zeilen(_settings("Streuung ×1,5"))
    assert "N/n_w" not in E.verdikt2(szen, z, zn)[1]
    szen, z, zn = _zeilen(E.Settings(szenario="Ausreißer", m_aus=3, hoehe_aus=8))
    assert "N/n_w" not in E.verdikt2(szen, z, zn)[1]               # die Rechnung gilt nur für einen einzelnen Ausreißer


def test_verdikt3_je_preset_im_strom():
    erwartet = {"Streuung ×1,5": "W1: Median k 54", "zweigipfelig": "W1: Median k nur bei mehr als der Hälfte der Ströme mit Alarm angegeben"}
    for preset, satz in erwartet.items():
        s = _settings(preset)
        szen = E.szenario_von(s)
        kind, text = E.verdikt3(szen, E.strom_ergebnis(szen, s.stroeme, s.horizont, s.seed_input))
        assert kind == "success" and satz in text, text
        assert "nicht ARL₀ = 370" in text and "höchstens 150 neue Werte, 151 Fensterschritte" in text and "(Soll 5 %)" in text


def test_verdikt3_bei_horizont_50_nennt_die_marken_25_und_50():
    szen = Z.Szenario("mittel", "Normal", 0.4)
    st = E.strom_ergebnis(szen, 200, 50, 2026)
    assert sorted(int(k) for k in st["marken"]) == [25, 50]
    kind, text = E.verdikt3(szen, st)
    assert "Bis 50 neuen Werten haben Alarm gegeben" in text and "bis 50:" in text and "151 Fensterschritte" not in text and "51 Fensterschritte" in text


def test_verdikt4_zweigipfelig_standard():
    s = _settings("zweigipfelig")
    szen = E.szenario_von(s)
    z, st = E.perm_ergebnis(szen, 1000, 199, 2026), E.strom_ergebnis(szen, 1000, 150, 2026)
    kind, text = E.verdikt4(36.0, 8.72, E.w1_grundgesamtheit(36, 8.72), 7.29, z, st, 150)
    assert kind == "success"
    assert "Gesamtstreuung 20,00, Mittelwert 100" in text and "W1 der Verteilungen 4,13 gegen die Schwelle 7,29" in text
    assert "W1 12,4 %, KL 51,0 %" in text and "im Strom bis 150 neuen Werten: W1 8,5 %, KL 59,4 %" in text


def test_marken_und_median_aus_der_zusammenfassung_sind_konsistent():
    st = E.strom_ergebnis(Z.Szenario("streuung", "Normal", 1.5), 1000, 150, 2026)
    assert len(st["anteil"]) == 151 and st["marken"]["150"] == st["anteil"][150] or np.allclose(st["marken"]["150"], st["anteil"][150], atol=1e-4)
    assert all(a <= b + 1e-12 for a, b in zip(np.array(st["anteil"])[:-1, 0], np.array(st["anteil"])[1:, 0]))      # der Anteil mit Alarm bis k wächst mit k
    assert all(0.0 <= v <= 1.0 for row in st["anteil"] for v in row)


# ---- Befunde der Aussagenprüfung: Verdikte bei Randfällen ------------------------------------------------------------------------------

def _einzel(hoehe: int, seeds: int = 200, b: int = 99):
    s = E.Settings(szenario="Ausreißer", m_aus=1, hoehe_aus=hoehe, seeds=seeds, b_perm=b)
    return _zeilen(s)


@pytest.mark.parametrize("hoehe", [4, 8])
def test_verdikt2_nennt_bei_niedrigem_einzelausreisser_nicht_die_rechnung_15_prozent(hoehe):
    szen, z, zn = _einzel(hoehe)
    text = E.verdikt2(szen, z, zn)[1]
    assert "= 15 %" not in text and f"Ein einzelner Ausreißer von +{hoehe} s bestimmt die Statistik nicht allein" in text and "gilt erst für sehr hohe Ausreißer (ab +20 s)" in text
    assert "gemessen für W1" in text


@pytest.mark.parametrize("hoehe", [20, 50])
def test_verdikt2_nennt_bei_hohem_einzelausreisser_die_rechnung_15_prozent(hoehe):
    szen, z, zn = _einzel(hoehe)
    text = E.verdikt2(szen, z, zn)[1]
    assert "sehr hohen Ausreißer (ab +20 s) rechnet die Permutation: P(p ≤ 5 %) ≈ 5 % · N/n_w = 15 %" in text and "nicht allein" not in text


def test_die_hoehengrenze_20_stuetzt_sich_auf_die_messung():
    h = E.VG.daten()["hoehen"]
    assert h["4"]["alarm_perm"][0] < 0.08 and h["8"]["alarm_perm"][0] < 0.11 and 0.12 < h["20"]["alarm_perm"][0] < 0.17      # W1 im Permutationstest: unter 11 % bei +4 und +8 s, um 15 % ab +20 s
    assert E.HOEHE_DOMINIERT == 20 and C.HOEHE_OPT == (4, 8, 20, 50)


def test_verdikt1_warnung_beim_einzelausreisser_plus_4_s_nennt_die_grenze_statt_keine_statistik_alarmiert_haeufiger():
    s = E.Settings(szenario="Ausreißer", m_aus=1, hoehe_aus=4)                 # W1 im Permutationstest 6,0 % (gemessen), |Mittel| 5,6 %: leicht über 5 %, unter 5 % + 3 Standardfehlern
    szen, z, zn = _zeilen(s)
    kind, text = E.verdikt1(szen, z, zn)
    assert kind == "warning" and max(z["alarm_perm"]) > 0.05
    assert "Keine Statistik alarmiert deutlich häufiger als die Fehlalarmrate (Grenze: 5 % plus drei Standardfehler = 7,1 %)" in text
    assert "Keine Statistik alarmiert häufiger als die Fehlalarmrate" not in text


def test_spalte_nicht_schlechter_ist_bei_normalzustand_ein_strich_und_sonst_ja_nein():
    szen, z, zn = _zeilen(E.Settings(szenario="zweigipfelig", gipfel_abstand=0, gipfel_sigma=20.0, seeds=200, b_perm=99))
    assert E.nicht_schlechter_spalte(z) == ["–"] * 5 and E.keine_aenderung(z["alarm_perm"], z["seeds"])
    szen, z, zn = _zeilen(_settings("Streuung ×1,5"))
    assert E.nicht_schlechter_spalte(z) == ["ja", "nein", "nein", "nein", "nein"]
    szen, z, zn = _zeilen(_settings("Einzelausreißer"))
    assert E.nicht_schlechter_spalte(z) == ["ja", "nein", "nein", "nein", "ja"]


def test_verdikt4_bei_gleichen_verteilungen_sagt_nur_mit_fenstern_von_50_werten_nicht_zu_unterscheiden():
    szen, z, zn = _zeilen(E.Settings(szenario="zweigipfelig", gipfel_abstand=0, gipfel_sigma=20.0, seeds=200, b_perm=99))
    kind, text = E.verdikt4(0.0, 20.0, E.w1_grundgesamtheit(0, 20.0), 7.3, z, E.strom_ergebnis(szen, 200, 50, 2026), 50)
    assert kind == "warning" and "mit Fenstern von 50 Werten von der Normalverteilung nicht zu unterscheiden" in text and "Grenze: 5 % plus drei Standardfehler" in text
