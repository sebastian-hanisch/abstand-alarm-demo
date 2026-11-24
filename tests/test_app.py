"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und prüft, dass kein Python-Fehler
auftritt (insbesondere `StreamlitDuplicateElementId` bei mehreren st.plotly_chart ohne key= und `StreamlitAPIException` bei Reglern mit falschen Grenzen). Dazu: je Preset die erwartete
Aussage, Permalink, Randfälle, Worst-Case am Regler-Maximum, alle vier Experimente für jede Art der Änderung.

Laufzeit: die Standardeinstellung und die Presets lesen ihre Zeilen aus `tools/messreihen.json` (unter einer Sekunde je Lauf); jede andere Einstellung rechnet den Permutationstest (etwa 2 s je Szenario)
und die Ströme (etwa 4 bis 8 s je Szenario und Horizont 150) live; der Zwischenspeicher gilt im Prozess und wärmt spätere Tests."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import aba_constants as C
import aba_evaluation as E
import aba_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")
# Regler, die nur bei einer Art der Änderung wirken (sonst deaktiviert; AppTest kann einen deaktivierten Regler nicht setzen)
NUR_BEI = {"verschiebung": "Mittelwert", "faktor": "Streuung", "m_aus": "Ausreißer", "hoehe_aus": "Ausreißer"}


KLEIN = {"ns": "200", "nstr": "200", "h": "50"}          # 200 Zufalls-Seeds, 200 Ströme, Horizont 50: rechnet jede neue Einstellung in unter einer Sekunde


def _app(**adresse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in adresse.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _setze(at, **werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def _verdikte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning") for e in getattr(at, kind))


def _metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_all_sections():
    at = _app()
    titel = " ".join(s.value for s in at.subheader)
    for abschnitt in ("Experiment 1: Wer sieht was?", "Experiment 2: Woher kommt die Schwelle?", "Experiment 3: Wie lange dauert es?", "Experiment 4: Gestalt ohne Lage und Streuung", "Wo die Annahmen enden"):
        assert abschnitt in titel, abschnitt
    labels = [e.label for e in at.expander]
    assert [l for l in labels if "Mathematische Formulierung" in l] == ["📐 Mathematische Formulierung"] and any("Annahmen, Grenzen und bekannte Probleme" in l for l in labels)
    assert any("So funktioniert der Alarm" in l for l in labels)
    assert len(at.metric) == 8 and len(at.info) == 3 and len(at.success) == 4 and not at.warning          # 5 Alarmraten + 3 Kennzahlen; drei Kernfragen; vier Verdikte
    schluessel = [p.key for p in at.get("plotly_chart")]
    assert len(set(schluessel)) == len(schluessel) == 8          # jedes Diagramm hat einen eigenen Schlüssel
    alles = " ".join(m.value for m in at.markdown)
    assert "Kernfragen" in alles and "Erkennt jede Statistik jede Änderung?" in alles and "Hängt die Antwort an der Schwelle?" in alles and "Wie lange dauert es im Strom?" in alles


def test_every_button_click_does_not_raise():
    at = _app()
    labels = [b.label for b in at.button]
    assert len(labels) == len(C.PRESETS) + 1, labels                      # sechs Presets und der Zufalls-Seed-Button
    for index, label in enumerate(labels[:-1]):
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_der_zufalls_seed_button_setzt_einen_neuen_seed_und_rechnet_neu_ohne_fehler(monkeypatch):
    monkeypatch.setattr(P.random, "randint", lambda lo, hi: 4242)
    at = _app(**KLEIN)
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["seed_input"] == 4242 and at.query_params["seed"] == "4242"


def test_every_slider_at_its_min_and_max_does_not_raise():
    at = _app(**KLEIN)
    schluessel = [s.key for s in at.sidebar.slider]
    assert schluessel == ["gipfel_abstand"], schluessel
    for edge in ("min", "max"):
        regler = _regler(at, "gipfel_abstand")
        regler.set_value(regler.min if edge == "min" else regler.max).run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Auswahlregler und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

def _ecken(optionen):
    """Erste und letzte Option (jede andere Einstellung rechnet Permutationstest und Ströme live, einige Sekunden)."""
    optionen = list(optionen)
    return [optionen[0], optionen[-1]]


@pytest.mark.parametrize("key", [k for k, s in P.SETTING_SPECS.items() if s.options is not None and k not in ("seeds", "horizont", "stroeme", "b_perm", "szenario")])
def test_optionen_der_auswahlregler_laufen_ohne_fehler(key):
    at = _app(**KLEIN)
    if key in NUR_BEI:
        _setze(at, szenario=NUR_BEI[key])
    for option in _ecken(P.SETTING_SPECS[key].options):
        _setze(at, **{key: option})


def test_alle_arten_der_aenderung_in_beiden_familien_laufen_durch_alle_experimente():
    at = _app(**KLEIN)
    for familie in C.FAMILIEN:
        for szenario in C.SZENARIEN:
            _setze(at, familie=familie, szenario=szenario)
            assert len(at.metric) == 8 and len(at.get("plotly_chart")) == 8, (familie, szenario)
            assert not at.warning, (familie, szenario, [w.value for w in at.warning])


@pytest.mark.parametrize("key,wert", [("seeds", 200), ("seeds", 600), ("b_perm", 99), ("horizont", 50), ("horizont", 300), ("stroeme", 200), ("stroeme", 600)])
def test_zahl_der_zufalls_seeds_vertauschungen_horizont_und_stroeme_laufen_und_stehen_im_verdikt(key, wert):
    at = _app()
    _setze(at, **{key: wert})
    text = _verdikte(at)
    if key == "seeds":
        assert f"{E.ganz(wert)} Zufalls-Seeds" in text
    if key == "b_perm":
        assert "Permutationstest (B = 99)" in text
    if key == "horizont":
        assert f"höchstens {wert} neue Werte, {wert + 1} Fensterschritte" in text
    if key == "stroeme":
        assert f"{E.ganz(wert)} Ströme" in text


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_nennt_sein_szenario_und_die_erwarteten_zeilen_im_verdikt(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = E.Settings(**C.PRESETS[name])
    szen = E.szenario_von(s)
    erstes = at.success[0].value
    assert erstes.startswith(f"{E.bezeichnung(szen)}, Referenz 100, Fenster 50, Permutationstest (B = 199), 1 000 Zufalls-Seeds:"), erstes
    assert at.success[2].value.startswith(f"{E.bezeichnung(szen)}, 1 000 Ströme"), at.success[2].value
    if name == "Einzelausreißer":
        assert "≈ 5 % · N/n_w = 15 %" in at.success[1].value
    if name == "t(3)":
        assert "Am häufigsten alarmiert Energie;" in erstes
    if name == "zweigipfelig":
        assert "Am häufigsten alarmiert KL;" in erstes


def test_standardeinstellung_zeigt_die_zahlen_der_messreihe():
    at = _app()
    m = _metrics(at)
    z = E.perm_ergebnis(E.szenario_von(E.Settings()), 1000, 199, C.SERIEN_SEED)
    assert m["Alarmrate W1"] == E.prozent(z["alarm_perm"][0]) == "57,6 %" and m["Alarmrate KS"] == "17,8 %"
    assert m["W1 der Verteilungen"] == "4,13" and m["Gesamtstreuung der Mischung"] == "20,00"


def test_permalink_uebernimmt_werte_und_ignoriert_ungueltiges():
    at = _app(sz="Ausreißer", ma="1", ha="50", fam="t(3)", ns="200", h="100", b="99")
    assert at.session_state["szenario"] == "Ausreißer" and at.session_state["m_aus"] == 1 and at.session_state["hoehe_aus"] == 50
    assert at.session_state["familie"] == "t(3)" and at.session_state["seeds"] == 200 and at.session_state["horizont"] == 100 and at.session_state["b_perm"] == 99
    assert "1 von 50 Werten bei +50 s (t(3))" in _verdikte(at)
    at = _app(sz="Unsinn", ns="abc", vs="nan", fam="Laplace", seed="-5", ga="999", nstr="200", h="50")
    assert at.session_state["szenario"] == C.DEFAULTS["szenario"] and at.session_state["seeds"] == 1000 and at.session_state["familie"] == "Normal"        # ns="abc" ist ungültig: Standard 1 000
    assert at.session_state["verschiebung"] == C.DEFAULTS["verschiebung"] and at.session_state["seed_input"] == 0 and at.session_state["gipfel_abstand"] == 40


def test_die_adresszeile_spiegelt_die_konfiguration_nach_einem_preset():
    at = _app()
    at.sidebar.button(key="preset_Einzelausreißer").click().run(timeout=TIMEOUT)
    qp = at.query_params
    assert qp["sz"] == "Ausreißer" and qp["ma"] == "1" and qp["ha"] == "50" and qp["fam"] == "Normal"


# ----------------------------------------------------------------------------------------------------------------------------------
# Randfälle und Worst-Case
# ----------------------------------------------------------------------------------------------------------------------------------

def test_zweigipfelig_mit_t3_rechnet_im_normalzustand_normal_und_meldet_keine_wirkung_ohne_aenderung():
    at = _app(sz="zweigipfelig", fam="t(3)", ga="0", gs="20.0", **KLEIN)
    text = _verdikte(at)
    assert "zweigipfelig (Gipfel bei 100 und 100, Streuung je Gipfel 20,00)" in text and "keine Änderung zu erkennen" in text
    assert len(at.warning) >= 3


def test_kleinster_und_groesster_wert_aller_regler_zugleich():
    at = _app()
    _setze(at, szenario="Ausreißer")                                  # m_aus und hoehe_aus sind erst bei dieser Art der Änderung aktiv
    _setze(at, familie="t(3)", m_aus=1, hoehe_aus=4, seeds=200, b_perm=99, horizont=50, stroeme=200, gipfel_abstand=0, gipfel_sigma=4.0, stat2="KS")
    _setze(at, m_aus=6, hoehe_aus=50, seeds=1000, b_perm=199, horizont=300, stroeme=1000, gipfel_abstand=40, gipfel_sigma=20.0, stat2="|Mittel|")      # Worst-Case: Horizont 300, 1 000 Ströme
    assert len(at.get("plotly_chart")) == 8 and not at.exception
