"""Erzeugt `tools/messreihen.json`: Schwellen und Standardzeilen (Zufalls-Seed 2026, 1 000 Zufalls-Seeds beziehungsweise Ströme, B = 199, Horizont 150).

Aufruf (Demo-Ordner): C:/Users/sebas/Claude Code Projects/_venvs/test/Scripts/python.exe -X utf8 tools/messreihen.py [Teil]
Teile: schwellen, perm, strom, schwellenstreuung, hoehen, streuung_lang, alles (Standard). Laufzeit insgesamt unter 3 Minuten, ein Prozess, ein Kern. Eine vorhandene Datei wird um den gerechneten Teil ergänzt."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aba_constants as C  # noqa: E402
import aba_evaluation as E  # noqa: E402
import aba_permutation as P  # noqa: E402
import aba_strom as ST  # noqa: E402
import aba_szenarien as Z  # noqa: E402
import aba_vorgerechnet as VG  # noqa: E402

PFAD = ROOT / "tools" / "messreihen.json"
SEED = C.SERIEN_SEED


def standard_szenarien() -> list:
    """Die Zeilen der Messreihe: Normalzustand und fünf Änderungen bei „Normal“, dazu drei Zeilen bei t(3)."""
    n = "Normal"
    return [
        Z.Szenario("normal", n), Z.Szenario("mittel", n, 0.4), Z.Szenario("streuung", n, 1.5), Z.Szenario("ausreisser", n, 3, 8), Z.Szenario("ausreisser", n, 1, 50), Z.Szenario("zwei", n, 36, 8.72),
        Z.Szenario("normal", "t(3)"), Z.Szenario("mittel", "t(3)", 0.5), Z.Szenario("streuung", "t(3)", 1.5),
    ]


WIEDERHOLUNGEN = (2027, 2028, 2029)            # weitere Zufalls-Seeds für den Einzelausreißer (drei zusätzliche Läufe mit je 1 000 Zufalls-Seeds)


def laden() -> dict:
    return json.loads(PFAD.read_text(encoding="utf-8")) if PFAD.exists() else {}


def speichern(d: dict) -> None:
    PFAD.write_text(json.dumps(d, ensure_ascii=False, indent=0, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


def teil_schwellen(d: dict) -> None:
    d["schwellen_fenster"] = {f: P.schwelle_sauber(f, SEED) for f in C.FAMILIEN}
    d["schwellen_strom"] = {}
    for f in C.FAMILIEN:
        t = ST.strom_verlauf(Z.Szenario("normal", f), C.N_STROM_SCHWELLE, C.H_MAX, SEED, C.Z_STROM_SCHWELLE)
        d["schwellen_strom"][f] = {str(h): s for h, s in ST.schwellen_je_horizont(t).items()}
        print("Schwellen Strom", f, d["schwellen_strom"][f][str(C.H_STANDARD)], flush=True)


def teil_perm(d: dict) -> None:
    d["perm"] = {}
    zeilen = [(s, SEED) for s in standard_szenarien()] + [(Z.Szenario("ausreisser", "Normal", 1, 50), w) for w in WIEDERHOLUNGEN]
    for szen, seed in zeilen:
        t0 = time.time()
        r = P.perm_lauf(szen, 1000, 199, seed)
        z = E.perm_zusammenfassung(r, 199, d["schwellen_fenster"][szen.familie])
        d["perm"][f"{szen.schluessel()}|1000|199|{seed}"] = z
        print("perm", szen.schluessel(), seed, [round(x * 100, 1) for x in z["alarm_perm"]], round(time.time() - t0, 1), "s", flush=True)


def teil_strom(d: dict) -> None:
    d["strom"] = {}
    h = C.H_STANDARD
    for szen in standard_szenarien():
        t0 = time.time()
        schwelle = d["schwellen_strom"][szen.familie][str(h)]
        t = ST.strom_verlauf(szen, 1000, h, SEED)
        tk = ST.strom_verlauf(Z.Szenario("normal", szen.familie), 1000, h, SEED, C.Z_KONTROLLE)
        z = E.strom_zusammenfassung(t, tk, schwelle, h)
        d["strom"][f"{szen.schluessel()}|1000|{h}|{SEED}"] = z
        print("strom", szen.schluessel(), [round(x * 100, 1) for x in z["marken"][str(h)]], round(time.time() - t0, 1), "s", flush=True)


SCHWELLEN_SEEDS = (2027, 2028)                 # weitere Zufalls-Seeds für die Strom-Schwelle: wie weit schwankt die Alarmrate mit dem geschätzten Quantil?


def teil_schwellenstreuung(d: dict) -> None:
    """Alarmanteil bis k = 150 bei Streuung ×1,5 im Strom mit Strom-Schwellen aus drei verschiedenen Sätzen von 6 000 Normalströmen (Zufalls-Seed 2026 und zwei weitere)."""
    szen = Z.Szenario("streuung", "Normal", 1.5)
    t = ST.strom_verlauf(szen, 1000, C.H_STANDARD, SEED)
    d["schwellenstreuung"] = {}
    for s in (SEED,) + SCHWELLEN_SEEDS:
        if s == SEED:
            thr = d["schwellen_strom"]["Normal"][str(C.H_STANDARD)]
        else:
            tn = ST.strom_verlauf(Z.Szenario("normal", "Normal"), C.N_STROM_SCHWELLE, C.H_STANDARD, s, C.Z_STROM_SCHWELLE)
            thr = ST.schwellen_je_horizont(tn, (C.H_STANDARD,))[C.H_STANDARD]
        a = ST.anteil_bis(ST.erste_alarme(t, thr, C.H_STANDARD), C.H_STANDARD)[C.H_STANDARD]
        d["schwellenstreuung"][str(s)] = {"schwelle": thr, "alarm_k150": a.tolist()}
        print("Schwellenstreuung", s, [round(x, 3) for x in thr.values()], [round(100 * x, 1) for x in a], flush=True)


LANGE_SEEDS = (2027, 2028, 2029)               # Zufalls-Seeds der Läufe mit 4 000 Zufalls-Seeds (Streuung ×1,5)
LANGE_ZAHL = 4000


def teil_streuung_lang(d: dict) -> None:
    """Streuung ×1,5, Permutationstest mit je 4 000 Zufalls-Seeds in drei Läufen: wie weit liegt die Alarmrate der Zeile mit 1 000 Zufalls-Seeds von dem, was mehr Wiederholungen zeigen?"""
    szen = Z.Szenario("streuung", "Normal", 1.5)
    d["streuung_lang"] = {"zahl": LANGE_ZAHL}
    for seed in LANGE_SEEDS:
        t0 = time.time()
        r = P.perm_lauf(szen, LANGE_ZAHL, 199, seed)
        z = E.perm_zusammenfassung(r, 199, d["schwellen_fenster"]["Normal"])
        d["streuung_lang"][str(seed)] = {"alarm_perm": z["alarm_perm"], "alarm_sauber": z["alarm_sauber"]}
        print("streuung_lang", seed, [round(x * 100, 1) for x in z["alarm_perm"]], round(time.time() - t0, 1), "s", flush=True)


HOEHEN = (4, 8, 20)                            # weitere Höhen des Einzelausreißers (+50 s steht in der Zeile „perm“)


def teil_hoehen(d: dict) -> None:
    """Einzelausreißer (1 von 50 Werten) bei kleineren Höhen: die Rechnung 5 % · N/n_w = 15 % gilt nur, wenn der Ausreißer die Statistik allein bestimmt."""
    d["hoehen"] = {}
    for h in HOEHEN:
        szen = Z.Szenario("ausreisser", "Normal", 1, h)
        r = P.perm_lauf(szen, 1000, 199, SEED)
        z = E.perm_zusammenfassung(r, 199, d["schwellen_fenster"]["Normal"])
        d["hoehen"][str(h)] = {"alarm_perm": z["alarm_perm"], "alarm_sauber": z["alarm_sauber"]}
        print("hoehen", h, [round(x * 100, 1) for x in z["alarm_perm"]], flush=True)


def main(teil: str) -> None:
    d = laden()
    d["meta"] = {"seed": SEED, "seeds": 1000, "b": 199, "n_schwelle": C.N_SCHWELLE, "n_strom_schwelle": C.N_STROM_SCHWELLE, "h_standard": C.H_STANDARD, "stroeme": 1000, "wiederholungen": list(WIEDERHOLUNGEN)}
    if teil in ("schwellen", "alles"):
        teil_schwellen(d)
        speichern(d)
    if teil in ("perm", "alles"):
        if "schwellen_fenster" not in d:
            raise SystemExit("erst den Teil „schwellen“ rechnen")
        teil_perm(d)
        speichern(d)
    if teil in ("strom", "alles"):
        if "schwellen_strom" not in d:
            raise SystemExit("erst den Teil „schwellen“ rechnen")
        teil_strom(d)
        speichern(d)
    if teil in ("schwellenstreuung", "alles"):
        teil_schwellenstreuung(d)
        speichern(d)
    if teil in ("hoehen", "alles"):
        teil_hoehen(d)
        speichern(d)
    if teil in ("streuung_lang", "alles"):
        teil_streuung_lang(d)
        speichern(d)
    VG.daten.cache_clear()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "alles")
