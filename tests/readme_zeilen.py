"""Tabellen und Sätze der README, mit denselben Formatfunktionen wie in der App aus der Datei `tools/messreihen.json` gebildet.

Die README muss jedes dieser Stücke wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aba_constants as C  # noqa: E402
import aba_evaluation as E  # noqa: E402
import aba_statistiken as S  # noqa: E402
import aba_szenarien as Z  # noqa: E402

MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
SEED = C.SERIEN_SEED
STAT = list(C.STATISTIKEN)

# (Beschriftung, Schlüssel des Szenarios in der Datei)
ZEILEN = [
    ("Normalzustand (Fehlalarmprobe)", "Normal|normal|0|0"),
    ("Mittelwert +0,4 s", "Normal|mittel|0.4|0"),
    ("Streuung ×1,5", "Normal|streuung|1.5|0"),
    ("3 von 50 Werten bei +8 s", "Normal|ausreisser|3|8"),
    ("1 von 50 Werten bei +50 s", "Normal|ausreisser|1|50"),
    ("zweigipfelig, gleicher Mittelwert, gleiche Streuung", "Normal|zwei|36|8.72"),
    ("t(3): Normalzustand", "t(3)|normal|0|0"),
    ("t(3): Mittelwert +0,5 s", "t(3)|mittel|0.5|0"),
    ("t(3): Streuung ×1,5", "t(3)|streuung|1.5|0"),
]
AENDERUNGEN = [k for k in ZEILEN if "normal|0|0" not in k[1]]      # die sieben Zeilen mit Änderung


def perm(schluessel: str, seed: int = SEED) -> dict:
    return MESS["perm"][f"{schluessel}|1000|199|{seed}"]


def strom(schluessel: str) -> dict:
    return MESS["strom"][f"{schluessel}|1000|150|{SEED}"]


def md(kopf, zeilen) -> str:
    return E.tabelle_md(kopf, zeilen)


def _namen(z, ok, raten="alarm_perm") -> str:
    """Die Statistiken „nicht schlechter als die beste“; in einer Zeile ohne erkennbare Änderung („–“) gibt es keine beste (die mit der höchsten Fehlalarmrate wäre es)."""
    if E.keine_aenderung(z[raten], z["seeds"]):
        return "–"
    return ", ".join(s for s, o in zip(STAT, ok) if o)


def _raten(werte, stellen=1):
    return [E.prozent(w, stellen) for w in werte]


# ---- Tabellen ----------------------------------------------------------------------------------------------------------------------------

def tabelle_a() -> str:
    zeilen = []
    for name, key in ZEILEN:
        z = perm(key)
        zeilen.append([name] + _raten(z["alarm_perm"]) + [_namen(z, z["nicht_schlechter_perm"])])
    return md(["Szenario"] + STAT + ["nicht schlechter als die beste (gepaart, 2 Standardfehler)"], zeilen)


def tabelle_b() -> str:
    zeilen = []
    for name, key in ZEILEN:
        z = perm(key)
        zeilen.append([name] + _raten(z["alarm_sauber"]) + [_namen(z, z["nicht_schlechter_sauber"], "alarm_sauber")])
    return md(["Szenario"] + STAT + ["nicht schlechter als die beste (gepaart, 2 Standardfehler)"], zeilen)


def schwellen() -> str:
    zeilen = []
    for fam in C.FAMILIEN:
        sf = MESS["schwellen_fenster"][fam]
        ss = MESS["schwellen_strom"][fam][str(C.H_STANDARD)]
        zeilen.append([f"{fam}: Fenster von 50 Werten (20 000 Paare)"] + [E.zahl(sf[s], 3) for s in STAT])
        zeilen.append([f"{fam}: Strom, Horizont 150 (6 000 Normalströme)"] + [E.zahl(ss[s], 3) for s in STAT])
    return md(["Schwelle (95-%-Quantil)"] + STAT, zeilen)


def tabelle_strom() -> str:
    zeilen = []
    for name, key in ZEILEN:
        z = strom(key)
        zeilen.append([name] + [f"{E.prozent(z['marken']['50'][j], 1)} / {E.prozent(z['marken']['150'][j], 1)}" for j in range(5)]
                      + [E.zahl(z["median_k"][0], 0) if z["median_k"][0] is not None and z["marken"]["150"][0] > 0.5 else "–"])
    return md(["Szenario (Alarm bis k = 50 / bis k = 150)"] + STAT + ["Median k von W1"], zeilen)


def kontrolle_strom() -> str:
    zeilen = []
    for fam, key in (("Normal", "Normal|normal|0|0"), ("t(3)", "t(3)|normal|0|0")):
        z = strom(key)
        zeilen.append([f"{fam}: 1 000 Kontroll-Normalströme, Alarm bis k = 150"] + _raten(z["kontrolle"]))
    return md(["Kontrolle (Soll 5 %)"] + STAT, zeilen)


# ---- Sätze -------------------------------------------------------------------------------------------------------------------------------

def fehlalarm_satz() -> str:
    a = [x for key in ("Normal|normal|0|0", "t(3)|normal|0|0") for x in perm(key)["alarm_perm"]]
    b = [x for key in ("Normal|normal|0|0", "t(3)|normal|0|0") for x in perm(key)["alarm_sauber"]]
    return (f"Fehlalarmprobe im Normalzustand (Permutationstest, 1 000 Zufalls-Seeds, beide Familien, zehn Zahlen): {E.prozent(min(a))} bis {E.prozent(max(a))}; "
            f"mit der Schwelle aus dem Normalzustand: {E.prozent(min(b))} bis {E.prozent(max(b))} (Soll 5 %, Standardfehler 0,7 Prozentpunkte).")


def einzel_satz() -> str:
    seeds = (SEED,) + tuple(MESS["meta"]["wiederholungen"])
    w1 = [perm("Normal|ausreisser|1|50", s)["alarm_perm"][0] for s in seeds]
    ks = [perm("Normal|ausreisser|1|50", s)["alarm_perm"][1] for s in seeds]
    mi = [perm("Normal|ausreisser|1|50", s)["alarm_perm"][4] for s in seeds]
    return (f"Einzelausreißer (1 von 50 Werten bei +50 s), Permutationstest, in vier Läufen mit je 1 000 Zufalls-Seeds (Zufalls-Seeds {', '.join(str(s) for s in seeds)}): "
            f"W1 {', '.join(E.prozent(x) for x in w1)} (Bereich {E.prozent(min(w1))} bis {E.prozent(max(w1))}), |Mittel| {E.prozent(min(mi))} bis {E.prozent(max(mi))}, KS {E.prozent(min(ks))} bis {E.prozent(max(ks))}; "
            f"die Rechnung 5 % · 150/50 gibt 15 %.")


def einzel_schwelle_satz() -> str:
    z = perm("Normal|ausreisser|1|50")
    return (f"Einzelausreißer, Zufalls-Seed 2026: Permutationstest W1 {E.prozent(z['alarm_perm'][0])}, |Mittel| {E.prozent(z['alarm_perm'][4])}, KS {E.prozent(z['alarm_perm'][1])}; "
            f"Schwelle aus dem Normalzustand W1 {E.prozent(z['alarm_sauber'][0])}, |Mittel| {E.prozent(z['alarm_sauber'][4])}, KS {E.prozent(z['alarm_sauber'][1])}; "
            f"Median der p-Werte W1 {E.zahl(z['p_median'][0], 3)}, KS {E.zahl(z['p_median'][1], 3)}; der größte Wert liegt in {E.prozent(z['gross'])} der Vertauschungen im Fenster (n_w/N = 33,3 %).")


def gestalt_satz() -> str:
    z, st = perm("Normal|zwei|36|8.72"), strom("Normal|zwei|36|8.72")
    w1 = E.w1_grundgesamtheit(36, 8.72)
    return (f"Zweigipfelig (Gipfel bei 82 und 118, Streuung je Gipfel 8,72, Gesamtstreuung {E.zahl(E.gesamtstreuung_zwei(36, 8.72), 2)}): W1 der Verteilungen {E.zahl(w1, 2)} gegen die Schwelle "
            f"{E.zahl(MESS['schwellen_fenster']['Normal']['W1'], 2)} für ein Fenster von 50 Werten; Alarmrate W1 {E.prozent(z['alarm_perm'][0])} (Permutationstest), {E.prozent(z['alarm_sauber'][0])} (Schwelle aus dem Normalzustand), "
            f"{E.prozent(st['marken']['150'][0])} (Strom bis k = 150); KL {E.prozent(z['alarm_perm'][3])}, {E.prozent(z['alarm_sauber'][3])}, {E.prozent(st['marken']['150'][3])}.")


def negativbefund_a() -> str:
    zw, tm = perm("Normal|zwei|36|8.72"), perm("t(3)|mittel|0.5|0")
    return (f"(a) W1 gewinnt nicht überall: bei zwei Gipfeln W1 {E.prozent(zw['alarm_perm'][0])} gegen KL {E.prozent(zw['alarm_perm'][3])}; bei t(3) mit Mittelwertverschiebung liegen Energie "
            f"({E.prozent(tm['alarm_perm'][2])}) und KS ({E.prozent(tm['alarm_perm'][1])}) über W1 ({E.prozent(tm['alarm_perm'][0])}), Abstand {E.zahl(100 * (tm['alarm_perm'][2] - tm['alarm_perm'][0]), 1)} "
            f"und {E.zahl(100 * (tm['alarm_perm'][1] - tm['alarm_perm'][0]), 1)} Prozentpunkte.")


def negativbefund_b() -> str:
    diffs = [(name, 100 * (perm(key)["alarm_perm"][4] - perm(key)["alarm_perm"][0])) for name, key in AENDERUNGEN]
    ueber = [(n, d) for n, d in diffs if d > 0]
    darunter = [d for n, d in diffs if d <= 0]
    liste = "; ".join(f"{n} {E.zahl(d, 1)}" for n, d in ueber)
    tm = perm("t(3)|mittel|0.5|0")["alarm_perm"]
    abst = [100 * (tm[j] - tm[4]) for j in (0, 1, 2)]
    return (f"(b) Der Mittelwertvergleich liegt in {len(ueber)} von {len(diffs)} Zeilen mit Änderung über W1 (Permutationstest, Differenz in Prozentpunkten: {liste}); in den übrigen {len(darunter)} Zeilen liegt er "
            f"{E.zahl(min(-d for d in darunter), 1)} bis {E.zahl(max(-d for d in darunter), 1)} Prozentpunkte darunter. Bei Normalverteilung und Mittelwertverschiebung leistet W1 also nicht mehr als der Mittelwert; "
            f"bei t(3) mit Mittelwertverschiebung liegen W1, KS und Energie dagegen {E.zahl(min(abst), 1)} bis {E.zahl(max(abst), 1)} Prozentpunkte über |Mittel|.")


def negativbefund_c() -> str:
    z = strom("Normal|zwei|36|8.72")
    i = max(range(5), key=lambda j: z["marken"]["50"][j])
    return (f"(c) Keine der fünf Statistiken erkennt die zweigipfelige Änderung im Strom bis k = 50 in mehr als {E.prozent(max(z['marken']['50']))} der Ströme "
            f"(der größte Wert gehört zu {STAT[i]}, W1 liegt bei {E.prozent(z['marken']['50'][0])}).")


def negativbefund_d() -> str:
    z = perm("Normal|ausreisser|1|50")
    uebrige = [z[k][j] for k in ("alarm_perm", "alarm_sauber") for j in (1, 2, 3)]
    return (f"(d) Ein Einzelausreißer (+50 s) ist im Permutationstest mit W1 zu {E.prozent(z['alarm_perm'][0], 0)} und mit |Mittel| zu {E.prozent(z['alarm_perm'][4], 0)} erkennbar, mit Schwelle aus dem Normalzustand "
            f"zu {E.prozent(z['alarm_sauber'][0], 0)} und {E.prozent(z['alarm_sauber'][4], 0)}: für W1 und |Mittel| ändert sich die Antwort mit der Schwelle. KS, Energie und KL erkennen ihn mit keiner der beiden Schwellen "
            f"zu mehr als {E.prozent(max(uebrige))} (mit der Schwelle aus dem Normalzustand: KS {E.prozent(z['alarm_sauber'][1])}, Energie {E.prozent(z['alarm_sauber'][2])}, KL {E.prozent(z['alarm_sauber'][3])}).")


def hoehen_satz() -> str:
    h = MESS["hoehen"]
    w = {k: h[k]["alarm_perm"] for k in h}
    w["50"] = perm("Normal|ausreisser|1|50")["alarm_perm"]
    teile = [f"+{k} s W1 {E.prozent(w[k][0])}, |Mittel| {E.prozent(w[k][4])}" for k in ("4", "8", "20", "50")]
    return ("Einzelausreißer bei anderer Höhe (Permutationstest, Zufalls-Seed 2026, 1 000 Zufalls-Seeds): " + "; ".join(teile)
            + ". Die Rechnung 5 % · 150/50 = 15 % gilt, wenn der Ausreißer die Statistik allein bestimmt, also für sehr hohe Ausreißer; bei +4 s und +8 s liegen die gemessenen Raten deutlich darunter.")


def gestalt_deutung() -> str:
    z, st = perm("Normal|zwei|36|8.72"), strom("Normal|zwei|36|8.72")
    alle_w1 = [z["alarm_perm"][0], z["alarm_sauber"][0], st["marken"]["150"][0]]
    alle_kl = [z["alarm_perm"][3], z["alarm_sauber"][3], st["marken"]["150"][3]]
    return (f"Der W1-Abstand zwischen der Mischung und der Normalverteilung ({E.zahl(E.w1_grundgesamtheit(36, 8.72), 2)}) liegt unter dem W1-Rauschen eines Fensters von 50 Werten (Schwelle {E.zahl(MESS['schwellen_fenster']['Normal']['W1'], 2)}): "
            f"W1 erkennt ihn kaum ({E.prozent(min(alle_w1))} bis {E.prozent(max(alle_w1))}), KL über 10 Klassen in etwa der Hälfte der Wiederholungen ({E.prozent(min(alle_kl))} bis {E.prozent(max(alle_kl))}), nicht in allen.")


def streuung_lang_satz() -> str:
    d = MESS["streuung_lang"]
    seeds = [k for k in d if k != "zahl"]
    w1 = [d[s]["alarm_perm"][0] for s in seeds]
    en = [d[s]["alarm_perm"][2] for s in seeds]
    return (f"Die {E.prozent(perm('Normal|streuung|1.5|0')['alarm_perm'][0])} für W1 sind eine hohe Ziehung (Standardfehler bei 1 000 Zufalls-Seeds 1,6 Prozentpunkte): in drei weiteren Läufen mit je {E.ganz(d['zahl'])} Zufalls-Seeds (Zufalls-Seeds {', '.join(seeds)}) liegt W1 bei "
            f"{E.prozent(min(w1))} bis {E.prozent(max(w1))}, Energie bei {E.prozent(min(en))} bis {E.prozent(max(en))}; die Rangfolge (W1 vor Energie vor KL vor KS vor |Mittel|) bleibt.")


def streuung_satz() -> str:
    z, s = perm("Normal|streuung|1.5|0"), strom("Normal|streuung|1.5|0")
    return (f"Streuung ×1,5, Normalverteilung: Permutationstest W1 {E.prozent(z['alarm_perm'][0])}, KS {E.prozent(z['alarm_perm'][1])}, Energie {E.prozent(z['alarm_perm'][2])}, KL {E.prozent(z['alarm_perm'][3])}, "
            f"|Mittel| {E.prozent(z['alarm_perm'][4])}; im Strom hat W1 bis k = 50 in {E.prozent(s['marken']['50'][0])} und bis k = 150 in {E.prozent(s['marken']['150'][0])} der Ströme Alarm gegeben (Median k {E.zahl(s['median_k'][0], 0)}), "
            f"KS in {E.prozent(s['marken']['50'][1])} und {E.prozent(s['marken']['150'][1])}.")


def block_satz() -> str:
    z, s = perm("Normal|ausreisser|3|8"), strom("Normal|ausreisser|3|8")
    return (f"Ausreißerblock (3 von 50 Werten bei +8 s; im Strom 6 % je Wert): W1 {E.prozent(z['alarm_perm'][0])} gegen KS {E.prozent(z['alarm_perm'][1])} im Permutationstest, "
            f"{E.prozent(z['alarm_sauber'][0])} mit der Schwelle aus dem Normalzustand; im Strom bis k = 50 / 150: W1 {E.prozent(s['marken']['50'][0], 0)} / {E.prozent(s['marken']['150'][0], 0)}, "
            f"KS {E.prozent(s['marken']['50'][1], 0)} / {E.prozent(s['marken']['150'][1], 0)}.")


def kontrolle_satz() -> str:
    werte = [x for key in ("Normal|normal|0|0", "t(3)|normal|0|0") for x in strom(key)["kontrolle"]]
    se = (0.05 * 0.95 / 1000) ** 0.5
    abw = max(abs(w - 0.05) for w in werte)
    return (f"Die Kontrollzahlen (zehn Zahlen) liegen zwischen {E.prozent(min(werte))} und {E.prozent(max(werte))}; die größte Abweichung von 5 % ist {E.zahl(100 * abw, 1)} Prozentpunkte "
            f"({E.zahl(abw / se, 1)} Standardfehler bei 1 000 Kontrollströmen).")


def schwellenstreuung_satz() -> str:
    d = MESS["schwellenstreuung"]
    seeds = list(d)
    def bereich(j):
        w = [d[s]["alarm_k150"][j] for s in seeds]
        return f"{E.prozent(min(w))} bis {E.prozent(max(w))}"
    def sw(j):
        w = [d[s]["schwelle"][STAT[j]] for s in seeds]
        return f"{E.zahl(min(w), 3)} bis {E.zahl(max(w), 3)}"
    return (f"Wie stark die Alarmrate an der geschätzten Schwelle hängt, zeigt Streuung ×1,5 im Strom mit Strom-Schwellen aus drei Sätzen von je 6 000 Normalströmen (Zufalls-Seeds {', '.join(seeds)}): "
            f"die Schwelle von W1 liegt bei {sw(0)}, die Alarmrate bis k = 150 bei {bereich(0)}; bei Energie liegt die Schwelle bei {sw(2)}, die Alarmrate bei {bereich(2)}; "
            f"bei KL liegt die Schwelle bei {sw(3)}, die Alarmrate bei {bereich(3)}.")


def zuwachs_satz() -> str:
    ref, fen = Z.paare(SEED, 0, 1000, Z.Szenario("normal"))
    fen2 = fen.copy()
    fen2[:, 0] = 1100.0
    a, b = S.statistiken_paare(ref, fen), S.statistiken_paare(ref, fen2)
    zw1 = float((b["W1"] - a["W1"]).mean())
    zks = float((b["KS"] - a["KS"]).max())
    return (f"Ein Fensterwert, der auf 1 100 gesetzt wird (statt eines Werts um 100), hebt W1 in 1 000 Fenstern im Mittel um {E.zahl(zw1, 1)} (obere Schranke Δ/n_w = 20) und KS höchstens um {E.zahl(zks, 3)} "
            f"(Schranke 1/n_w = 0,020); die Schwelle von KS im Strom liegt bei {E.zahl(MESS['schwellen_strom']['Normal'][str(C.H_STANDARD)]['KS'], 2)}, die von W1 bei {E.zahl(MESS['schwellen_strom']['Normal'][str(C.H_STANDARD)]['W1'], 2)}.")


def block_ankunft_satz() -> str:
    z = strom("Normal|ausreisser|3|8")
    ref, fen = Z.paare(SEED, 0, 1000, Z.Szenario("normal"))
    thr = MESS["schwellen_strom"]["Normal"][str(C.H_STANDARD)]["W1"]
    anteile = []
    for m in (1, 2, 3):
        f2 = fen.copy()
        f2[:, :m] = 260.0
        anteile.append(float((S.statistiken_paare(ref, f2)["W1"] > thr).mean()))
    return (f"Ausreißerblock im Strom: mit 6 % je neuem Wert ist bis k = 25 in {E.prozent(1 - 0.94 ** 25)} der Ströme mindestens ein Ausreißer eingetroffen (1 − 0,94^25), "
            f"W1 hat bis k = 25 in {E.prozent(z['marken']['25'][0])} der Ströme Alarm gegeben (KS {E.prozent(z['marken']['25'][1])}); in 1 000 Fenstern im Normalzustand, in denen 1 / 2 / 3 Werte durch 260 ersetzt werden, "
            f"liegt W1 über der Strom-Schwelle {E.zahl(thr, 2)} in {E.prozent(anteile[0])} / {E.prozent(anteile[1])} / {E.prozent(anteile[2])}: ein Block braucht mehrere Werte im Fenster.")


STUECKE = {
    "tabelle_a": tabelle_a, "tabelle_b": tabelle_b, "schwellen": schwellen, "tabelle_strom": tabelle_strom, "kontrolle_strom": kontrolle_strom,
    "fehlalarm_satz": fehlalarm_satz, "einzel_satz": einzel_satz, "einzel_schwelle_satz": einzel_schwelle_satz, "gestalt_satz": gestalt_satz,
    "negativbefund_a": negativbefund_a, "negativbefund_b": negativbefund_b, "negativbefund_c": negativbefund_c, "negativbefund_d": negativbefund_d, "hoehen_satz": hoehen_satz, "gestalt_deutung": gestalt_deutung, "streuung_lang_satz": streuung_lang_satz,
    "streuung_satz": streuung_satz, "block_satz": block_satz, "kontrolle_satz": kontrolle_satz, "schwellenstreuung_satz": schwellenstreuung_satz,
    "zuwachs_satz": zuwachs_satz, "block_ankunft_satz": block_ankunft_satz,
}


if __name__ == "__main__":
    for name, f in STUECKE.items():
        print(f"===== {name} =====")
        print(f())
        print()
