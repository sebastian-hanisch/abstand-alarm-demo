"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import aba_constants as C
import aba_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
STANDARD_PRESET = "Streuung ×1,5"
# Regler, die nur bei einer Art der Änderung wirken (sonst deaktiviert): das Preset, bei dem sie wirken
PRESET_FUER = {"verschiebung": "Mittelwert +0,4 s", "faktor": "Streuung ×1,5", "m_aus": "Ausreißerblock", "hoehe_aus": "Ausreißerblock"}


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _anderer_wert(regler, spec):
    if spec.options is not None:
        return next(o for o in spec.options if o != regler.value)
    return spec.hi if regler.value != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [STANDARD_PRESET]


def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf():
    at = _app()
    for name in C.PRESETS:
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert not at.exception, [str(e) for e in at.exception]
        assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name], name
        at.run()
        assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name], name


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "ein Eintrag in SETTING_SPECS hat keinen Regler"


@pytest.mark.parametrize("state_key", [k for k in P.SETTING_SPECS if k != "szenario"])
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key, monkeypatch):
    """An der Funktion `aktives_preset` mit einem Session State aus Wörterbuch (statt je Regler ein AppTest-Lauf): für jedes Preset, bei dem der Regler wirkt, hebt jeder andere Wert die Markierung auf."""
    spec = P.SETTING_SPECS[state_key]
    monkeypatch.setattr(P.st, "session_state", {})
    gesehen = 0
    for name, preset in C.PRESETS.items():
        P.st.session_state.clear()
        P.st.session_state.update({k: s.default for k, s in P.SETTING_SPECS.items()})
        for k, w in preset.items():
            P.st.session_state[k] = w
        assert P.aktives_preset() == name
        alt = P.st.session_state[state_key]
        for neu in (spec.options if spec.options is not None else (spec.lo, spec.hi)):
            if neu != alt:
                P.st.session_state[state_key] = neu
                assert P.aktives_preset() != name, (name, state_key, neu)
                gesehen += 1
        P.st.session_state[state_key] = alt
    assert gesehen >= len(C.PRESETS)                             # mindestens ein anderer Wert je Preset


def test_apply_preset_setzt_alle_regler_und_das_preset_ist_danach_aktiv(monkeypatch):
    monkeypatch.setattr(P.st, "session_state", {k: s.options[-1] if s.options else s.hi for k, s in P.SETTING_SPECS.items()})
    for name in C.PRESETS:
        P.apply_preset(name)
        assert P.aktives_preset() == name
        assert all(P.st.session_state[k] == C.PRESETS[name].get(k, P.SETTING_SPECS[k].default) for k in P.SETTING_SPECS)


@pytest.mark.parametrize("szenario,preset", [("Mittelwert", "Mittelwert +0,4 s"), ("Ausreißer", "Ausreißerblock"), ("zweigipfelig", "zweigipfelig")])
def test_wechsel_der_art_der_aenderung_markiert_das_preset_mit_denselben_werten_und_nicht_mehr_das_alte(szenario, preset):
    """Mit den übrigen Reglern auf Standard ergibt jede Art der Änderung genau eines der Presets: das alte ist nicht mehr hervorgehoben, das passende ist es."""
    at = _app()
    _regler(at, "szenario").set_value(szenario).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [preset]


def test_breite_tabellen_scrollen_waagerecht_statt_den_hauptbereich_zu_verschieben():
    """Auf 375 px Breite schiebt eine breite Markdown-Tabelle sonst den Hauptbereich nach rechts: die Zeile steht im Quelltext von app.py, gleich nach dem Kopf-Stylesheet."""
    quelle = (Path(APP)).read_text(encoding="utf-8")
    zeile = '<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>'
    assert f"st.html('{zeile}')" in quelle
    assert quelle.index("st.html(kopf_css(") < quelle.index(zeile) < quelle.index('st.container(key="kopf")')
