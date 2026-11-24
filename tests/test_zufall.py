"""Zufall und Kopien: der vektorisierte SplitMix64 ist bitgleich zum Hausgenerator, die Wiederholungen hängen nicht an der Blockgröße, die Kopien stimmen mit ihren Quellen überein.

Kopier-Tests (DEMO-PLAYBOOK §4): `aba_rng.py` ist die Vorlage aus dem Website-Repo, `instanz_rng` und die Abstandsfunktionen sind aus `verteilungsabstand-demo` kopiert; fehlt der Quellordner
(etwa in der CI), überspringen sich die Tests."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import aba_abstand as A
import aba_zufall as ZU
from aba_rng import SplitMix64

PROJEKTE = Path(__file__).resolve().parents[2]
QUELLE_VAB = PROJEKTE / "verteilungsabstand-demo"
VORLAGE_RNG = PROJEKTE / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def _modul_aus_quelle(ordner: Path, name: str):
    if not (ordner / f"{name}.py").exists():
        pytest.skip(f"Quelle {ordner.name}/{name}.py nicht vorhanden")
    sys.path.insert(0, str(ordner))
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(ordner))


def test_rng_ist_bytegleich_zur_vorlage():
    if not VORLAGE_RNG.exists():
        pytest.skip("Vorlage nicht vorhanden")
    assert (Path(__file__).resolve().parent.parent / "aba_rng.py").read_bytes() == VORLAGE_RNG.read_bytes()


def test_instanz_rng_ist_wortgleich_aus_verteilungsabstand_demo_kopiert():
    quelle = _modul_aus_quelle(QUELLE_VAB, "vab_profile")
    assert inspect.getsource(ZU.instanz_rng) == inspect.getsource(quelle.instanz_rng)
    for seed, zweck, index in ((2026, 11, 0), (7, 13, 999), (0, 0, 0)):
        assert ZU.instanz_rng(seed, zweck, index).state == quelle.instanz_rng(seed, zweck, index).state


def test_abstandsfunktionen_sind_wortgleich_aus_verteilungsabstand_demo_kopiert():
    quelle = _modul_aus_quelle(QUELLE_VAB, "vab_abstand")
    for name in ("normal_cdf", "verteilungsfunktion", "w1"):
        assert inspect.getsource(getattr(A, name)) == inspect.getsource(getattr(quelle, name)), name
    z = np.linspace(-3, 3, 13)
    assert np.array_equal(A.normal_cdf(z), quelle.normal_cdf(z))
    p, q = np.array([0.2, 0.5, 0.3]), np.array([0.1, 0.3, 0.6])
    assert A.w1(p, q, 2.0) == quelle.w1(p, q, 2.0)


def test_vektorisierter_zufall_ist_bitgleich_zum_hausgenerator():
    u = ZU.zufall_gleich_zeilen(2026, 13, 0, 6, 25)
    for j in range(6):
        g = ZU.instanz_rng(2026, 13, j)
        assert np.array_equal(u[j], np.array([g.random() for _ in range(25)]))


def test_zeilen_ab_einem_versatz_gleichen_den_zeilen_eines_groesseren_blocks():
    gross = ZU.zufall_gleich_zeilen(5, 12, 0, 20, 30)
    assert np.array_equal(ZU.zufall_gleich_zeilen(5, 12, 7, 6, 30), gross[7:13])
    assert np.array_equal(ZU.zufall_gleich_zeilen(5, 12, 0, 20, 10), gross[:, :10])      # Spalten sind ein Präfix


@pytest.mark.parametrize("familie", ["Normal", "t(3)"])
def test_standardwerte_haengen_nicht_an_blockgroesse_und_spaltenzahl(familie):
    a = ZU.standardwerte(3, 11, 0, 12, 40, familie)
    assert np.array_equal(ZU.standardwerte(3, 11, 5, 4, 40, familie), a[5:9])
    assert np.array_equal(ZU.standardwerte(3, 11, 0, 12, 25, familie), a[:, :25])         # kürzere Ströme sind ein Präfix längerer


def test_normalwerte_haben_mittel_null_und_streuung_eins_im_band():
    x = ZU.standardwerte(1, 11, 0, 400, 250, "Normal").ravel()      # 100 000 Werte
    assert abs(x.mean()) < 0.015 and abs(x.std() - 1) < 0.01
    assert abs(np.quantile(x, 0.975) - 1.95996) < 0.03


def test_t3_werte_haben_die_quantile_der_t_verteilung_mit_3_freiheitsgraden_auf_streuung_eins():
    x = ZU.standardwerte(1, 11, 0, 400, 250, "t(3)").ravel()
    # t(3): 97,5-%-Quantil 3,18245; auf Standardabweichung 1 normiert (Teilung durch √3): 1,83735; Median des Betrags 0,76489/√3 = 0,44160
    assert abs(np.quantile(x, 0.975) - 3.18245 / np.sqrt(3)) < 0.04
    assert abs(np.median(np.abs(x)) - 0.76489 / np.sqrt(3)) < 0.01
    assert np.abs(x).max() > 6                                      # schwere Ränder: bei 100 000 Werten kommen weit entfernte Werte vor


def test_unbekannte_familie_wird_abgelehnt():
    with pytest.raises(ValueError):
        ZU.standardwerte(1, 11, 0, 2, 5, "Laplace")
