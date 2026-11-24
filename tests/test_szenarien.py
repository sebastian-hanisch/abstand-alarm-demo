"""Szenarien: jede Art der Änderung wirkt so, wie sie beschrieben ist (feste Fenster und Ströme)."""

import numpy as np
import pytest

import aba_constants as C
import aba_szenarien as Z


def _basis(seed=5, zeilen=400):
    return Z.paare(seed, 0, zeilen, Z.Szenario("normal"))


def test_normalzustand_hat_mittel_100_und_streuung_20_in_beiden_familien():
    for familie in C.FAMILIEN:
        ref, fen = Z.paare(5, 0, 600, Z.Szenario("normal", familie))
        x = np.concatenate([ref, fen], axis=1).ravel()                 # 90 000 Werte
        assert abs(x.mean() - 100) < 0.35, familie
        assert abs(np.median(x) - 100) < 0.3, familie
        if familie == "Normal":
            assert abs(x.std() - 20) < 0.25
        else:                                                          # t(3) auf Standardabweichung 20: das 97,5-%-Quantil liegt bei 100 + 20 · 3,18245/√3
            assert abs(np.quantile(x, 0.975) - (100 + 20 * 3.18245 / np.sqrt(3))) < 1.5


def test_mittelwert_verschiebt_jeden_fensterwert_um_p1_mal_20():
    _, f0 = _basis()
    _, f1 = Z.paare(5, 0, 400, Z.Szenario("mittel", "Normal", 0.4))
    assert np.allclose(f1 - f0, 8.0)


def test_streuung_streckt_die_abweichungen_vom_mittelwert_100():
    _, f0 = _basis()
    _, f1 = Z.paare(5, 0, 400, Z.Szenario("streuung", "Normal", 1.5))
    assert np.allclose(f1 - 100, 1.5 * (f0 - 100))
    assert abs(f1.std() / f0.std() - 1.5) < 1e-9


@pytest.mark.parametrize("m,hoehe", [(1, 50), (3, 8), (6, 4)])
def test_ausreisser_ersetzen_im_festen_fenster_genau_m_werte_bei_100_plus_hoehe_mal_20(m, hoehe):
    _, f0 = _basis()
    _, f1 = Z.paare(5, 0, 400, Z.Szenario("ausreisser", "Normal", m, hoehe))
    geaendert = f1 != f0
    assert np.all(geaendert.sum(axis=1) == m)                          # genau m Ausreißer unter 50 Werten, in jeder Wiederholung
    hoch = f1[geaendert]
    assert abs(hoch.mean() - (100 + 20 * hoehe)) < 0.6                  # Streuung 5 je Ausreißer, bei 400 · m Werten
    assert abs(hoch.std() - 5) < 0.4


def test_ausreisser_im_strom_kommen_mit_dem_anteil_m_durch_50_je_neuem_wert_und_erst_ab_dem_51_wert():
    ref0, s0 = Z.strom(5, 0, 500, Z.Szenario("normal"), 100)
    ref1, s1 = Z.strom(5, 0, 500, Z.Szenario("ausreisser", "Normal", 3, 8), 100)
    assert np.array_equal(ref0, ref1) and np.array_equal(s0[:, :50], s1[:, :50])          # die ersten 50 Werte und die Referenz bleiben im Normalzustand
    geaendert = s1[:, 50:] != s0[:, 50:]
    assert abs(geaendert.mean() - 0.06) < 0.006                         # 6 % je Wert (50 000 Werte: Standardfehler 0,1 Prozentpunkte)
    assert abs((s1[:, 50:][geaendert]).mean() - 260) < 0.6


def test_zweigipfelig_hat_mittelwert_100_und_bei_abstand_36_und_streuung_8_72_die_streuung_20():
    _, f = Z.paare(5, 0, 600, Z.Szenario("zwei", "Normal", 36, 8.72))
    assert abs(f.mean() - 100) < 0.25 and abs(f.std() - 20.0) < 0.3
    assert 0.47 < np.mean(f < 100) < 0.53                              # beide Gipfel gleich oft
    assert abs(np.mean(np.abs(f - 118) < 8.72) - np.mean(np.abs(f - 82) < 8.72)) < 0.02     # Gipfel bei 82 und 118


def test_zweigipfelig_ist_immer_im_normalzustand_normal():
    assert Z.Szenario("zwei", "t(3)", 36, 8.72).familie == "Normal"
    assert Z.Szenario("zwei", "t(3)", 36, 8.72).schluessel() == Z.Szenario("zwei", "Normal", 36, 8.72).schluessel()


def test_strom_erste_50_werte_sind_normalzustand_und_das_szenario_normal_aendert_nichts():
    ref0, s0 = Z.strom(5, 0, 30, Z.Szenario("normal"), 60)
    ref1, s1 = Z.strom(5, 0, 30, Z.Szenario("mittel", "Normal", 0.5), 60)
    assert np.array_equal(s0[:, :50], s1[:, :50]) and np.allclose(s1[:, 50:] - s0[:, 50:], 10.0)
    assert s0.shape == (30, 110)


def test_aus_einstellungen_nimmt_die_zum_szenario_gehoerigen_regler():
    sz = Z.aus_einstellungen("t(3)", "Ausreißer", 0.4, 1.5, 4, 20, 36, 8.72)
    assert (sz.art, sz.familie, sz.p1, sz.p2) == ("ausreisser", "t(3)", 4.0, 20.0)
    sz = Z.aus_einstellungen("Normal", "Streuung", 0.4, 1.75, 4, 20, 36, 8.72)
    assert (sz.art, sz.p1) == ("streuung", 1.75)
    assert set(C.ART_VON_SZENARIO.values()) == set(Z.ARTEN) - {"normal"}


def test_unbekannte_art_und_familie_werden_abgelehnt():
    with pytest.raises(ValueError):
        Z.Szenario("unsinn")
    with pytest.raises(ValueError):
        Z.Szenario("normal", "Laplace")
