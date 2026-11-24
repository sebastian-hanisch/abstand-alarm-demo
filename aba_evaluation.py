"""Kennzahlen, Zusammenfassungen und Verdikte (reine Funktionen, ohne Streamlit).

Aufbau: `perm_ergebnis` und `strom_ergebnis` rechnen (oder lesen aus `tools/messreihen.json`, wenn die Einstellung die der Messreihe ist) und liefern kleine Wörterbücher; die Verdikte
`verdikt1` bis `verdikt4` sind reine Funktionen dieser Wörterbücher und geben (Art, Text) zurück, die App ruft `getattr(st, Art)(Text)`."""

import math
from dataclasses import dataclass, fields

import numpy as np

import aba_abstand as A
import aba_constants as C
import aba_permutation as P
import aba_strom as ST
import aba_szenarien as Z
import aba_vorgerechnet as VG


@dataclass(frozen=True)
class Settings:
    seed_input: int = C.DEFAULTS["seed_input"]
    familie: str = C.DEFAULTS["familie"]
    szenario: str = C.DEFAULTS["szenario"]
    verschiebung: float = C.DEFAULTS["verschiebung"]
    faktor: float = C.DEFAULTS["faktor"]
    m_aus: int = C.DEFAULTS["m_aus"]
    hoehe_aus: int = C.DEFAULTS["hoehe_aus"]
    seeds: int = C.DEFAULTS["seeds"]
    b_perm: int = C.DEFAULTS["b_perm"]
    stat2: str = C.DEFAULTS["stat2"]
    horizont: int = C.DEFAULTS["horizont"]
    stroeme: int = C.DEFAULTS["stroeme"]
    gipfel_abstand: int = C.DEFAULTS["gipfel_abstand"]
    gipfel_sigma: float = C.DEFAULTS["gipfel_sigma"]


def settings_aus_zustand(werte: dict) -> Settings:
    return Settings(**{f.name: werte[f.name] for f in fields(Settings) if f.name in werte})


def szenario_von(s: Settings) -> Z.Szenario:
    return Z.aus_einstellungen(s.familie, s.szenario, s.verschiebung, s.faktor, s.m_aus, s.hoehe_aus, s.gipfel_abstand, s.gipfel_sigma)


# ---------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------------------------------------------------------------

def zahl(x, stellen: int = 3) -> str:
    """Dezimalkomma; ∞ als Zeichen; −0 wird zu 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    text = f"{x:.{stellen}f}".replace(".", ",")
    return text[1:] if text.startswith("-0") and float(text.replace(",", ".")) == 0 else text.replace("-", "−")


def ganz(n) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


def prozent(x: float, stellen: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma."""
    return zahl(100.0 * x, stellen) + " %"


def md_zelle(text: str) -> str:
    """Ein literales | in einer Markdown-Tabellenzelle bricht die Spalten: als \\| schreiben."""
    return text.replace("|", "\\|")


def tabelle_md(kopf, zeilen) -> str:
    return "\n".join(["| " + " | ".join(md_zelle(k) for k in kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(md_zelle(z) for z in zeile) + " |" for zeile in zeilen])


def bezeichnung(szen: Z.Szenario, kurz: bool = False) -> str:
    """Name des Szenarios mit Stärke, z. B. „Streuung ×1,5“ (s ist die Standardabweichung 20 der Tageswerte)."""
    fam = "" if kurz else f" ({szen.familie})"
    if szen.art == "normal":
        return f"Normalzustand{fam}"
    if szen.art == "mittel":
        return f"Mittelwert +{zahl(szen.p1, 1)} s{fam}"
    if szen.art == "streuung":
        return f"Streuung ×{zahl(szen.p1, 2).rstrip('0').rstrip(',')}{fam}"
    if szen.art == "ausreisser":
        return f"{int(szen.p1)} von 50 Werten bei +{int(szen.p2)} s{fam}"
    return f"zweigipfelig (Gipfel bei {zahl(C.MITTEL - szen.p1 / 2, 0)} und {zahl(C.MITTEL + szen.p1 / 2, 0)}, Streuung je Gipfel {zahl(szen.p2, 2)})"


# ---------------------------------------------------------------------------------------------------------------------------------
# Auswertung der Wiederholungen
# ---------------------------------------------------------------------------------------------------------------------------------

def nicht_schlechter(alarm: np.ndarray) -> list:
    """Welche Statistiken sind nicht schlechter als die beste? alarm: (Wiederholungen, 5) Wahrheitswerte. Gepaart: Differenz der Alarme der besten und der anderen Statistik je Wiederholung;
    „nicht schlechter“, wenn der Mittelwert der Differenz höchstens zwei Standardfehler beträgt."""
    a = alarm.astype(float)
    beste = int(np.argmax(a.mean(axis=0)))
    n = a.shape[0]
    aus = []
    for j in range(a.shape[1]):
        d = a[:, beste] - a[:, j]
        se = float(np.std(d, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
        aus.append(bool(d.mean() <= C.Z_PAAR * se + 1e-15))
    return aus


def perm_zusammenfassung(r: dict, b: int, schwellen: dict) -> dict:
    """Aus den Läufen (p-Werte, Statistiken) die Kennzahlen: Alarmraten für beide Schwellen, gepaarte Vergleiche, p-Wert-Histogramme, Anteil „größter Wert im Fenster“."""
    a_p = P.alarm_perm(r["p"], b)
    a_s = P.alarm_schwelle(r["t"], schwellen)
    hist = [np.histogram(r["p"][:, i] - 1e-9, bins=C.HISTO_KLASSEN, range=(0.0, 1.0))[0].astype(int).tolist() for i in range(len(C.STATISTIKEN))]
    return {
        "seeds": int(r["p"].shape[0]), "b": int(b),
        "alarm_perm": a_p.mean(axis=0).tolist(), "alarm_sauber": a_s.mean(axis=0).tolist(),
        "nicht_schlechter_perm": nicht_schlechter(a_p), "nicht_schlechter_sauber": nicht_schlechter(a_s),
        "p_median": np.median(r["p"], axis=0).tolist(), "p_hist": hist, "gross": float(r["gross"].mean()),
        "p_anteil_klein": (r["p"] <= C.ALPHA + 1e-9).mean(axis=0).tolist(),
    }


def perm_ergebnis(szen: Z.Szenario, seeds: int, b: int, seed: int) -> dict:
    """Alarmraten bei festem Fenster (Permutationstest und Schwelle aus dem Normalzustand) für `seeds` Wiederholungen des Szenarios; aus der Datei, wenn die Einstellung dort steht."""
    vor = VG.perm_zeile(szen.schluessel(), seeds, b, seed)
    if vor is not None:
        return vor
    r = P.perm_lauf(szen, seeds, b, seed)
    return perm_zusammenfassung(r, b, VG.schwellen_fenster(szen.familie))


def strom_zusammenfassung(t_werte: np.ndarray, t_kontrolle: np.ndarray, schwelle: dict, h: int) -> dict:
    """Kennzahlen eines Stroms: Anteil mit Alarm bis k (k = 0, …, h), Median k, Marken bei k = 25, 50, 100, h und die Kontrolle in Normalströmen."""
    erste = ST.erste_alarme(t_werte, schwelle, h)
    anteil = ST.anteil_bis(erste, h)
    ek = ST.erste_alarme(t_kontrolle, schwelle, h)
    ant_k = ST.anteil_bis(ek, h)
    med = ST.median_k(erste)
    marken = sorted({25, 50, 100, h} & set(range(h + 1)))
    return {
        "stroeme": int(t_werte.shape[0]), "h": int(h),
        "anteil": np.round(anteil, 5).tolist(), "median_k": [None if math.isnan(m) else m for m in med],
        "marken": {str(k): anteil[k].tolist() for k in marken},
        "kontrolle": ant_k[h].tolist(), "kontrolle_k0": ant_k[0].tolist(), "schwelle": {s: float(schwelle[s]) for s in C.STATISTIKEN},
    }


def strom_ergebnis(szen: Z.Szenario, stroeme: int, h: int, seed: int) -> dict:
    """Erkennungsverzögerung im Strom für `stroeme` Ströme des Szenarios und `stroeme` Kontroll-Normalströme; aus der Datei, wenn die Einstellung dort steht."""
    vor = VG.strom_zeile(szen.schluessel(), stroeme, h, seed)
    if vor is not None:
        return vor
    schwelle = VG.schwelle_strom(szen.familie, h)
    t = ST.strom_verlauf(szen, stroeme, h, seed)
    tk = ST.strom_verlauf(Z.Szenario("normal", szen.familie), stroeme, h, seed, C.Z_KONTROLLE)
    return strom_zusammenfassung(t, tk, schwelle, h)


def w1_grundgesamtheit(abstand: float, sigma_gipfel: float) -> float:
    """W1 zwischen N(100, 20²) und der Mischung 0,5 N(100 − d/2, s²) + 0,5 N(100 + d/2, s²) (Grundgesamtheit, numerisch auf einem feinen Raster; Schritt 0,02)."""
    x = np.arange(-100.0, 300.0 + 1e-9, 0.02)
    f = A.normal_cdf((x - C.MITTEL) / C.SIGMA)
    g = 0.5 * A.normal_cdf((x - (C.MITTEL - abstand / 2)) / sigma_gipfel) + 0.5 * A.normal_cdf((x - (C.MITTEL + abstand / 2)) / sigma_gipfel)
    return A.w1(np.diff(f), np.diff(g), 0.02)


def w1_kurve(sigma_gipfel: float, abstaende) -> list:
    """Punkte (d, W1 der Grundgesamtheit) für die Kurve in Experiment 4."""
    return [(float(d), w1_grundgesamtheit(float(d), sigma_gipfel)) for d in abstaende]


def gesamtstreuung_zwei(abstand: float, sigma_gipfel: float) -> float:
    """Standardabweichung der Mischung: √(s² + (d/2)²)."""
    return math.sqrt(sigma_gipfel ** 2 + (abstand / 2) ** 2)


def beispiel_paar(s: Settings):
    """Referenz und Fenster der ersten Wiederholung (zur Anschauung in Experiment 1)."""
    ref, fen = Z.paare(s.seed_input, 0, 1, szenario_von(s))
    return ref[0], fen[0]


# ---------------------------------------------------------------------------------------------------------------------------------
# Verdikte
# ---------------------------------------------------------------------------------------------------------------------------------

def _liste(namen) -> str:
    namen = list(namen)
    return ", ".join(namen[:-1]) + " und " + namen[-1] if len(namen) > 1 else (namen[0] if namen else "keine")


def _raten(werte, stellen: int = 1) -> str:
    return ", ".join(f"{s} {prozent(w, stellen)}" for s, w in zip(C.STATISTIKEN, werte))


HOEHE_DOMINIERT = 20          # ab dieser Höhe (in s) bestimmt ein einzelner Ausreißer die Statistik allein (gemessen: W1 im Permutationstest 14,6 % bei +20 s, 6,0 % bei +4 s)


def _grenze_fehlalarm(n: int) -> float:
    """Obergrenze, bis zu der eine Alarmrate noch zur Fehlalarmrate 5 % passt: 5 % plus drei Standardfehler."""
    return C.ALPHA + 3.0 * math.sqrt(C.ALPHA * (1.0 - C.ALPHA) / n)


def _grenze_text(n: int) -> str:
    return f"Grenze: 5 % plus drei Standardfehler = {prozent(_grenze_fehlalarm(n))}"


def keine_aenderung(raten, n: int) -> bool:
    """Keine Alarmrate liegt über der Grenze 5 % plus drei Standardfehler: dann gibt es nichts zu erkennen, und eine Spalte „nicht schlechter als die beste“ hätte keinen Sinn."""
    return max(raten) <= _grenze_fehlalarm(n)


def nicht_schlechter_spalte(z: dict, schluessel: str = "nicht_schlechter_perm", raten: str = "alarm_perm") -> list:
    """Die Spalte „nicht schlechter als die beste“ je Statistik: „ja“ oder „nein“, bei einem Szenario ohne erkennbare Änderung „–“ (dort wäre die Statistik mit der höchsten Fehlalarmrate die „beste“)."""
    if keine_aenderung(z[raten], z["seeds"]):
        return ["–"] * len(z[raten])
    return ["ja" if ok else "nein" for ok in z[schluessel]]


def verdikt1(szen: Z.Szenario, z: dict, z_normal: dict) -> tuple:
    """Experiment 1: Alarmraten je Statistik bei Permutationstest (5 % Fehlalarm), mit Fehlalarmprobe und gepaartem Vergleich."""
    n, b = z["seeds"], z["b"]
    raten = z["alarm_perm"]
    fa = z_normal["alarm_perm"]
    beste = int(np.argmax(raten))
    nb = [s for s, ok in zip(C.STATISTIKEN, z["nicht_schlechter_perm"]) if ok]
    kopf = f"{bezeichnung(szen)}, Referenz 100, Fenster 50, Permutationstest (B = {b}), {ganz(n)} Zufalls-Seeds: {_raten(raten)}."
    fehl = f" Fehlalarmrate im Normalzustand ({szen.familie}): {prozent(min(fa))} bis {prozent(max(fa))} (Soll 5 %)."
    if keine_aenderung(raten, n):
        return "warning", kopf + fehl + f" Keine Statistik alarmiert deutlich häufiger als die Fehlalarmrate ({_grenze_text(n)}): in diesem Szenario ist mit diesen Daten keine Änderung zu erkennen."
    text = kopf + f" Am häufigsten alarmiert {C.STATISTIKEN[beste]}; nicht schlechter als sie (gepaart, zwei Standardfehler): {_liste(nb)}." + fehl
    return "success", text


def verdikt2(szen: Z.Szenario, z: dict, z_normal: dict) -> tuple:
    """Experiment 2: die Schwelle ändert die Antwort: Permutationstest gegen Schwelle aus dem Normalzustand, p-Werte, Anteil der Vertauschungen mit dem größten Wert im Fenster."""
    n = z["seeds"]
    ap, asa = z["alarm_perm"], z["alarm_sauber"]
    gross = z["gross"]
    erwartet = C.N_FENSTER / C.N_POOL
    i_w1, i_ks, i_mi = C.STATISTIKEN.index("W1"), C.STATISTIKEN.index("KS"), C.STATISTIKEN.index("|Mittel|")
    text = (f"{bezeichnung(szen)}, {ganz(n)} Zufalls-Seeds: Permutationstest {_raten(ap)}; Schwelle aus dem Normalzustand {_raten(asa)}. "
            f"Median der p-Werte: W1 {zahl(z['p_median'][i_w1], 3)}, KS {zahl(z['p_median'][i_ks], 3)}. "
            f"Der größte Wert liegt in {prozent(gross)} der Vertauschungen im Fenster (n_w/N = {prozent(erwartet)}).")
    if szen.art == "ausreisser" and int(szen.p1) == 1:
        gemessen = f"gemessen für W1 {prozent(ap[i_w1])} und |Mittel| {prozent(ap[i_mi])}"
        if szen.p2 >= HOEHE_DOMINIERT:
            text += f" Bei einem einzelnen sehr hohen Ausreißer (ab +{HOEHE_DOMINIERT} s) rechnet die Permutation: P(p ≤ 5 %) ≈ 5 % · N/n_w = {prozent(C.ALPHA * C.N_POOL / C.N_FENSTER, 0)}; {gemessen}."
        else:
            text += (f" Ein einzelner Ausreißer von +{zahl(szen.p2, 0)} s bestimmt die Statistik nicht allein: die Rechnung P(p ≤ 5 %) ≈ 5 % · N/n_w gilt erst für sehr hohe Ausreißer (ab +{HOEHE_DOMINIERT} s); "
                     f"{gemessen}.")
    text += " Voraussetzung der zweiten Schwelle: der Normalzustand ist bekannt (hier: " + f"{szen.familie}, Mittel 100, Streuung 20); der Permutationstest braucht nur Austauschbarkeit."
    fehl = max(z_normal["alarm_sauber"] + z_normal["alarm_perm"])
    if keine_aenderung(ap + asa, n):
        return "warning", text + f" Beide Schwellen alarmieren nicht deutlich häufiger als die Fehlalarmrate ({_grenze_text(n)}): keine Änderung zu erkennen."
    return "success", text + f" (Fehlalarmprobe im Normalzustand: höchstens {prozent(fehl)}.)"


def verdikt3(szen: Z.Szenario, st: dict) -> tuple:
    """Experiment 3: Erkennungsverzögerung im Strom: Anteil der Ströme mit Alarm bis k, Median k, Fehlalarm je Horizont."""
    h = st["h"]
    m = st["marken"]
    k_klein = "50" if "50" in m else str(min(int(x) for x in m))
    med = st["median_k"]
    i_w1 = C.STATISTIKEN.index("W1")
    wert_med = f"Median k {zahl(med[i_w1], 0)}" if med[i_w1] is not None and st["marken"][str(h)][i_w1] > 0.5 else "Median k nur bei mehr als der Hälfte der Ströme mit Alarm angegeben"
    text = (f"{bezeichnung(szen)}, {ganz(st['stroeme'])} Ströme, Alarm bei 5 % Fehlalarm je Horizont (höchstens {h} neue Werte, {h + 1} Fensterschritte; nicht ARL₀ = 370). "
            f"Bis {k_klein} neuen Werten haben Alarm gegeben: {_raten(m[k_klein], 0)}; bis {h}: {_raten(m[str(h)], 0)}. W1: {wert_med}. "
            f"Kontrolle in {ganz(st['stroeme'])} Normalströmen, bis {h}: {_raten(st['kontrolle'])} (Soll 5 %).")
    if keine_aenderung(m[str(h)], st["stroeme"]):
        return "warning", text + f" Keine Statistik alarmiert im Strom deutlich häufiger als die Fehlalarmrate ({_grenze_text(st['stroeme'])}): in diesem Szenario ist mit diesen Daten keine Änderung zu erkennen."
    return "success", text


def verdikt4(abstand: float, sigma_gipfel: float, w1_pop: float, schwelle_w1: float, z: dict, st: dict, h: int) -> tuple:
    """Experiment 4: Gestalt ohne Lage und Streuung: W1 der Grundgesamtheit gegen die Schwelle eines Fensters von 50 Werten, Alarmraten fest und im Strom."""
    sg = gesamtstreuung_zwei(abstand, sigma_gipfel)
    i_w1, i_kl = C.STATISTIKEN.index("W1"), C.STATISTIKEN.index("KL")
    text = (f"Zweigipfelig mit Gipfeln bei {zahl(C.MITTEL - abstand / 2, 0)} und {zahl(C.MITTEL + abstand / 2, 0)}, Streuung je Gipfel {zahl(sigma_gipfel, 2)} (Gesamtstreuung {zahl(sg, 2)}, Mittelwert 100): "
            f"W1 der Verteilungen {zahl(w1_pop, 2)} gegen die Schwelle {zahl(schwelle_w1, 2)} für ein Fenster von 50 Werten im Normalzustand. "
            f"Alarmrate Permutationstest ({ganz(z['seeds'])} Zufalls-Seeds): W1 {prozent(z['alarm_perm'][i_w1])}, KL {prozent(z['alarm_perm'][i_kl])}; "
            f"mit Schwelle aus dem Normalzustand: W1 {prozent(z['alarm_sauber'][i_w1])}, KL {prozent(z['alarm_sauber'][i_kl])}; "
            f"im Strom bis {h} neuen Werten: W1 {prozent(st['marken'][str(h)][i_w1])}, KL {prozent(st['marken'][str(h)][i_kl])}.")
    if keine_aenderung(z["alarm_perm"] + z["alarm_sauber"], z["seeds"]):
        return "warning", text + f" Keine Statistik liegt deutlich über der Fehlalarmrate ({_grenze_text(z['seeds'])}): die Mischung ist mit Fenstern von 50 Werten von der Normalverteilung nicht zu unterscheiden."
    return "success", text
