"""Vorgerechnete Messreihen und Schwellen aus `tools/messreihen.json` (erzeugt von `tools/messreihen.py`, Zufalls-Seed 2026).

Die Datei enthält die Schwellen (Fenster: 20 000 Paare je Familie; Strom: Normalströme je Familie und Horizont) und die Zusammenfassungen der Standardzeilen (Alarmraten, Verläufe im Strom).
Eine Einstellung, die dort steht, wird nicht neu gerechnet; jede andere rechnet die App live (mit Zwischenspeicher). Fehlt die Datei oder ein Eintrag, gilt die Rechnung."""

import json
from functools import lru_cache
from pathlib import Path

import aba_constants as C

PFAD = Path(__file__).resolve().parent / "tools" / "messreihen.json"


@lru_cache(maxsize=1)
def daten() -> dict:
    try:
        return json.loads(PFAD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _schluessel(szenario_schluessel: str, a: int, b: int, seed: int) -> str:
    return f"{szenario_schluessel}|{a}|{b}|{seed}"


def schwellen_fenster(familie: str, seed: int = C.SERIEN_SEED) -> dict:
    """Schwellen „sauber“ je Statistik für Paare (Referenz 100, Fenster 50) im Normalzustand der Familie."""
    d = daten().get("schwellen_fenster", {}).get(familie)
    if d is not None and seed == C.SERIEN_SEED:
        return d
    import aba_permutation as P
    return P.schwelle_sauber(familie, seed)


@lru_cache(maxsize=8)
def _schwellen_strom_live(familie: str, h: int, seed: int) -> dict:
    import aba_strom as ST
    import aba_szenarien as Z
    t = ST.strom_verlauf(Z.Szenario("normal", familie), C.N_STROM_SCHWELLE, h, seed, C.Z_STROM_SCHWELLE)
    return ST.schwellen_je_horizont(t, (h,))[h]


def schwelle_strom(familie: str, h: int, seed: int = C.SERIEN_SEED) -> dict:
    """Schwellen je Statistik im Strom: 95-%-Quantil des Maximums über die Schritte 0 bis h in Normalströmen der Familie."""
    d = daten().get("schwellen_strom", {}).get(familie, {}).get(str(h))
    if d is not None and seed == C.SERIEN_SEED:
        return d
    return _schwellen_strom_live(familie, h, seed)


def perm_zeile(szenario_schluessel: str, seeds: int, b: int, seed: int):
    return daten().get("perm", {}).get(_schluessel(szenario_schluessel, seeds, b, seed))


def strom_zeile(szenario_schluessel: str, stroeme: int, h: int, seed: int):
    return daten().get("strom", {}).get(_schluessel(szenario_schluessel, stroeme, h, seed))


def streuung_lang_bereich(j: int):
    """Kleinste und größte Alarmrate der Statistik j (Permutationstest, Streuung ×1,5, Normal) in den Läufen mit je 4 000 Zufalls-Seeds (`tools/messreihen.py streuung_lang`); None, wenn die Datei sie nicht enthält."""
    d = daten().get("streuung_lang")
    if not d:
        return None
    werte = [v["alarm_perm"][j] for k, v in d.items() if k != "zahl"]
    return (min(werte), max(werte), int(d["zahl"])) if werte else None
