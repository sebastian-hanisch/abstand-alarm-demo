"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink (Begrenzen, Einrasten)."""

import aba_constants as C
import aba_evaluation as E
import aba_presets as P


def test_jedes_preset_hat_eine_hilfe_und_nur_bekannte_regler():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen_und_optionen():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_das_preset_streuung_ist_die_standardeinstellung():
    assert E.Settings(**C.PRESETS["Streuung ×1,5"]) == E.Settings()
    assert C.DEFAULTS["szenario"] == "Streuung" and C.DEFAULTS["faktor"] == 1.5 and C.DEFAULTS["familie"] == "Normal"


def test_setting_specs_decken_jeden_standardwert_und_jeden_settings_eintrag_ab():
    assert set(P.SETTING_SPECS) == set(C.DEFAULTS) == {f for f in E.Settings.__dataclass_fields__}
    for key, spec in P.SETTING_SPECS.items():
        assert spec.default == C.DEFAULTS[key], key


def test_die_sechs_presets_aus_dem_plan_und_ihre_szenarien():
    assert list(C.PRESETS) == ["Mittelwert +0,4 s", "Streuung ×1,5", "Ausreißerblock", "Einzelausreißer", "zweigipfelig", "t(3)"]
    sz = {name: E.szenario_von(E.Settings(**preset)) for name, preset in C.PRESETS.items()}
    assert (sz["Mittelwert +0,4 s"].art, sz["Mittelwert +0,4 s"].p1) == ("mittel", 0.4)
    assert (sz["Ausreißerblock"].p1, sz["Ausreißerblock"].p2) == (3, 8) and (sz["Einzelausreißer"].p1, sz["Einzelausreißer"].p2) == (1, 50)
    assert sz["zweigipfelig"].art == "zwei" and (sz["zweigipfelig"].p1, sz["zweigipfelig"].p2) == (36, 8.72)
    assert sz["t(3)"].familie == "t(3)" and sz["t(3)"].p1 == 0.5


def test_einrasten_und_begrenzen_aus_der_adresszeile():
    s = P.SETTING_SPECS
    assert P._eingerastet(s["verschiebung"], 0.43) == 0.4 and P._eingerastet(s["verschiebung"], 99.0) == 1.0
    assert P._eingerastet(s["seeds"], 700) == 600 and P._eingerastet(s["seeds"], 700) in C.SEEDS_OPT and isinstance(P._eingerastet(s["seeds"], 700), int)
    assert P._eingerastet(s["familie"], "t(3)") == "t(3)" and P._eingerastet(s["familie"], "Laplace") is None
    assert P._eingerastet(s["gipfel_abstand"], 37.4) == 38 and P._eingerastet(s["gipfel_abstand"], 500) == 40 and P._eingerastet(s["gipfel_abstand"], -3) == 0
    assert isinstance(P._eingerastet(s["gipfel_abstand"], 37.0), int)
    assert P._eingerastet(s["seed_input"], 10 ** 9) == C.SEED_MAX and P._eingerastet(s["faktor"], float("nan")) is None
