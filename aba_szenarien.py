"""Szenarien der Tageswerte: Normalzustand und fünf Arten der Änderung, für feste Fenster und für Ströme (nur numpy).

Alle Werte sind Tageswerte (ein Wert je Tag) mit Mittel 100 und Standardabweichung 20 im Normalzustand, Familie „Normal“ oder „t(3)“. Eine Wiederholung ist eine Zeile j mit eigenem Zufallsstrom;
die Basiswerte (Referenz und Fenster) einer Zeile sind in allen Szenarien dieselben (gemeinsame Zufallszahlen), die Änderung wird auf die Fensterwerte gelegt. Arten (`art`):
- `normal`: keine Änderung (Fehlalarmprobe),
- `mittel`: Fenster + p1 · 20 (p1 in Standardabweichungen),
- `streuung`: 100 + p1 · (Wert − 100),
- `ausreisser`: im festen Fenster die ersten m = p1 Werte durch 100 + p2 · 20 + 5 · Z ersetzt; im Strom jeder neue Wert mit Wahrscheinlichkeit m/50,
- `zwei`: Mischung 0,5 N(100 − p1/2, p2²) + 0,5 N(100 + p1/2, p2²), unabhängig von der Familie (daher immer Normalzustand „Normal“)."""

from dataclasses import dataclass

import numpy as np

import aba_constants as C
from aba_zufall import standardwerte, zufall_gleich_zeilen

ARTEN = ("normal", "mittel", "streuung", "ausreisser", "zwei")


@dataclass(frozen=True)
class Szenario:
    art: str = "normal"
    familie: str = "Normal"
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self):
        if self.art not in ARTEN:
            raise ValueError(f"unbekannte Art: {self.art}")
        if self.familie not in C.FAMILIEN:
            raise ValueError(f"unbekannte Familie: {self.familie}")
        if self.art == "zwei" and self.familie != "Normal":
            object.__setattr__(self, "familie", "Normal")      # die Mischung ist aus Normalverteilungen; der Normalzustand dazu ist Normal

    def schluessel(self) -> str:
        return f"{self.familie}|{self.art}|{self.p1:g}|{self.p2:g}"

    def als_normal(self) -> "Szenario":
        return Szenario("normal", self.familie)


def aus_einstellungen(familie: str, szenario: str, verschiebung: float, faktor: float, m_aus: int, hoehe_aus: float, gipfel_abstand: float, gipfel_sigma: float) -> Szenario:
    """Das Szenario, das die Regler beschreiben (Namen der Regler wie in `aba_constants.DEFAULTS`)."""
    art = C.ART_VON_SZENARIO[szenario]
    p = {"mittel": (verschiebung, 0.0), "streuung": (faktor, 0.0), "ausreisser": (float(m_aus), float(hoehe_aus)), "zwei": (float(gipfel_abstand), float(gipfel_sigma))}[art]
    return Szenario(art, familie, p[0], p[1])


def _rauschen(u: np.ndarray) -> np.ndarray:
    """Standardnormalzahlen aus den Spalten 1 und 2 von u (…, 3): Box-Muller."""
    return np.sqrt(-2.0 * np.log(np.maximum(u[..., 1], 1e-12))) * np.cos(2.0 * np.pi * u[..., 2])


def anwenden(werte: np.ndarray, szen: Szenario, u: np.ndarray, modus: str) -> np.ndarray:
    """Änderung auf Fensterwerte (…, n) legen; u (…, n, 3) sind gleichverteilte Zahlen (Spalte 0: Auswahl, Spalten 1 und 2: Rauschen). modus: „fenster“ oder „strom“."""
    if szen.art == "normal":
        return werte
    if szen.art == "mittel":
        return werte + szen.p1 * C.SIGMA
    if szen.art == "streuung":
        return C.MITTEL + szen.p1 * (werte - C.MITTEL)
    z = _rauschen(u)
    if szen.art == "ausreisser":
        hoch = C.MITTEL + szen.p2 * C.SIGMA + C.RAUSCHEN_AUSREISSER * z
        if modus == "fenster":
            ist = np.arange(werte.shape[-1]) < int(round(szen.p1))
            ist = np.broadcast_to(ist, werte.shape)
        else:
            ist = u[..., 0] < szen.p1 / C.N_FENSTER
        return np.where(ist, hoch, werte)
    # zwei Gipfel
    vorzeichen = np.where(u[..., 0] < 0.5, -1.0, 1.0)
    return C.MITTEL + vorzeichen * szen.p1 / 2.0 + szen.p2 * z


def paare(seed: int, ab: int, zeilen: int, szen: Szenario):
    """Referenz (zeilen, 100) und Fenster (zeilen, 50) der Wiederholungen ab, ..., ab + zeilen − 1; das Fenster enthält die Änderung des Szenarios."""
    basis = C.MITTEL + C.SIGMA * standardwerte(seed, C.Z_BASIS, ab, zeilen, C.N_POOL, szen.familie)
    u = zufall_gleich_zeilen(seed, C.Z_EXTRA, ab, zeilen, 3 * C.N_FENSTER).reshape(zeilen, C.N_FENSTER, 3)
    return basis[:, : C.N_REF], anwenden(basis[:, C.N_REF:], szen, u, "fenster")


def schwellenpaare(seed: int, ab: int, zeilen: int, familie: str):
    """Paare im Normalzustand für die Schwelle „sauber“ (eigener Zweck, unabhängig von den Wiederholungen der Experimente)."""
    basis = C.MITTEL + C.SIGMA * standardwerte(seed, C.Z_SCHWELLE, ab, zeilen, C.N_POOL, familie)
    return basis[:, : C.N_REF], basis[:, C.N_REF:]


def strom(seed: int, ab: int, zeilen: int, szen: Szenario, h: int, zweck_basis: int = C.Z_STROM_BASIS):
    """Referenz (zeilen, 100) und Strom (zeilen, 50 + h): die ersten 50 Werte im Normalzustand, ab dem 51. Wert gilt die Änderung des Szenarios.

    Das Fenster zum Schritt t (t = 0, …, h) sind die Werte t, …, t + 49; es enthält k = t neue Werte, soweit es die Änderung schon erreicht hat (ab t = 50 nur noch veränderte)."""
    n = C.N_FENSTER + h
    basis = C.MITTEL + C.SIGMA * standardwerte(seed, zweck_basis, ab, zeilen, C.N_REF + n, szen.familie)
    ref, s = basis[:, : C.N_REF], basis[:, C.N_REF:]
    if szen.art != "normal":
        u = zufall_gleich_zeilen(seed, C.Z_STROM_EXTRA, ab, zeilen, 3 * h).reshape(zeilen, h, 3)
        s = np.concatenate([s[:, : C.N_FENSTER], anwenden(s[:, C.N_FENSTER:], szen, u, "strom")], axis=1)
    return ref, s
