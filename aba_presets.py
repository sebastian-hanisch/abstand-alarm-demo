"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (Name im Session State = Schlüssel in `aba_constants.DEFAULTS`). Reglerwerte aus der Adresszeile werden auf die
Grenzen begrenzt (Zahlen, bei Stufenreglern auf das Raster der Stufe gerundet) beziehungsweise auf den nächsten erlaubten Wert eingerastet (Auswahllisten); ungültige Werte werden ignoriert."""

import random
from dataclasses import dataclass

import streamlit as st

import aba_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahllisten und Stufen)
    step: float | None = None         # Stufe eines Zahlenreglers


def _opt(url, caster, key, options):
    return SettingSpec(url, caster, C.DEFAULTS[key], options=options)


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed_input"], 0, C.SEED_MAX),
    "familie": _opt("fam", str, "familie", C.FAMILIEN),
    "szenario": _opt("sz", str, "szenario", C.SZENARIEN),
    "verschiebung": _opt("vs", float, "verschiebung", C.VERSCHIEBUNG_OPT),
    "faktor": _opt("fk", float, "faktor", C.FAKTOR_OPT),
    "m_aus": _opt("ma", int, "m_aus", C.M_AUS_OPT),
    "hoehe_aus": _opt("ha", int, "hoehe_aus", C.HOEHE_OPT),
    "seeds": _opt("ns", int, "seeds", C.SEEDS_OPT),
    "b_perm": _opt("b", int, "b_perm", C.B_OPT),
    "stat2": _opt("st", str, "stat2", C.STATISTIKEN),
    "horizont": _opt("h", int, "horizont", C.HORIZONT_OPT),
    "stroeme": _opt("nstr", int, "stroeme", C.STROEME_OPT),
    "gipfel_abstand": SettingSpec("ga", int, C.DEFAULTS["gipfel_abstand"], C.GIPFEL_ABSTAND_MIN, C.GIPFEL_ABSTAND_MAX, step=C.GIPFEL_ABSTAND_SCHRITT),
    "gipfel_sigma": _opt("gs", float, "gipfel_sigma", C.GIPFEL_SIGMA_OPT),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _eingerastet(spec: SettingSpec, wert):
    """Wert auf Grenzen begrenzt (und auf die Stufe gerundet) beziehungsweise auf die nächste erlaubte Option eingerastet; None, wenn nichts passt."""
    if spec.options is not None:
        if spec.caster is str:
            return wert if wert in spec.options else None
        if not isinstance(wert, (int, float)) or wert != wert:
            return None
        gewaehlt = min(spec.options, key=lambda o: abs(o - float(wert)))
        return spec.caster(gewaehlt)
    if isinstance(wert, float) and wert != wert:
        return None
    if spec.lo is not None:
        wert = max(spec.lo, wert)
    if spec.hi is not None:
        wert = min(spec.hi, wert)
    if spec.step:
        wert = min(spec.hi, max(spec.lo, round(round((wert - spec.lo) / spec.step) * spec.step + spec.lo, 10)))
        if spec.caster is int:
            wert = int(wert)
    return wert


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt beziehungsweise eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                wert = _eingerastet(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if wert is not None:
                st.session_state[state_key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset legt alle Regler fest: was es nicht nennt, steht danach auf dem Standardwert."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
