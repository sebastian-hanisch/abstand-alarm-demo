"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Regler-Maximum (alles vektorisiert, ein Kern):
- Permutationstest: Zufalls-Seeds (höchstens 1 000) mal B (höchstens 199) Vertauschungen mal 150 Werte, etwa 30 Millionen Zufallszahlen und Statistiken; rund 4 s je Szenario.
- Strom: Ströme (höchstens 1 000) mal Horizont + 1 (höchstens 301) Fensterschritte, je Schritt ein Sortieren von 150 Werten je Strom; rund 6 s je Szenario.
- Alles, was zur Standardeinstellung gehört, liegt vorgerechnet in `tools/messreihen.json`."""

# ---- feste Annahmen (Referenz, Fenster, Normalzustand) ---------------------------------------------------------------------------------------
N_REF, N_FENSTER = 100, 50                 # Referenz 100 Tage, aktuelles Fenster 50 Tage (ein Tageswert je Tag)
N_POOL = N_REF + N_FENSTER
MITTEL, SIGMA = 100.0, 20.0                # Normalzustand: Mittelwert 100, Standardabweichung 20 (Normal und t(3) auf Standardabweichung 20 normiert)
KLASSEN = 10                               # KL über 10 Klassen gleicher Masse der gepoolten Werte
GLAETTUNG_KL = 0.5                         # zu jedem Klassenzähler addiert
ALPHA = 0.05                               # Fehlalarmrate
RAUSCHEN_AUSREISSER = 5.0                  # Streuung der Ausreißerwerte um ihre Höhe
TOL_GLEICH = 1e-12                         # relative Toleranz beim Zählen von T_π ≥ T_beob (Gleichstände der Statistik)
Z_PAAR = 2.0                               # Standardfehler im gepaarten Vergleich „nicht schlechter als der Beste“

STATISTIKEN = ("W1", "KS", "Energie", "KL", "|Mittel|")
STAT_LANG = {
    "W1": "W1 (Wasserstein-1-Abstand)",
    "KS": "KS (Kolmogorov-Smirnov)",
    "Energie": "Energie-Distanz",
    "KL": "KL (Kullback-Leibler)",
    "|Mittel|": "|Mittel| (Mittelwertvergleich)",
}

# ---- Familien und Szenarien ------------------------------------------------------------------------------------------------------------------
FAMILIEN = ("Normal", "t(3)")
SZENARIEN = ("Mittelwert", "Streuung", "Ausreißer", "zweigipfelig")
ART_VON_SZENARIO = {"Mittelwert": "mittel", "Streuung": "streuung", "Ausreißer": "ausreisser", "zweigipfelig": "zwei"}
SZENARIO_VON_ART = {v: k for k, v in ART_VON_SZENARIO.items()}

VERSCHIEBUNG_OPT = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)       # in Standardabweichungen (s)
FAKTOR_OPT = (1.25, 1.5, 1.75, 2.0)
M_AUS_OPT = (1, 2, 3, 4, 5, 6)                               # Zahl der Ausreißer unter 50 Werten (im Strom: Anteil m/50 je neuem Wert)
HOEHE_OPT = (4, 8, 20, 50)                                   # Höhe des Ausreißers über 100, in Standardabweichungen
GIPFEL_ABSTAND_MIN, GIPFEL_ABSTAND_MAX, GIPFEL_ABSTAND_SCHRITT = 0, 40, 2
GIPFEL_SIGMA_OPT = (4.0, 6.0, 8.72, 10.0, 14.0, 20.0)       # Streuung je Gipfel; 8,72 mit Abstand 36 gibt die Streuung 20

SEEDS_OPT = (200, 400, 600, 1000)                            # Zufalls-Seeds (Wiederholungen)
B_OPT = (99, 199)                                            # Vertauschungen
HORIZONT_OPT = (50, 100, 150, 200, 250, 300)
STROEME_OPT = (200, 400, 600, 1000)
H_MAX = max(HORIZONT_OPT)
H_STANDARD = 150

# ---- Zufall ----------------------------------------------------------------------------------------------------------------------------------
SEED_MAX = 999999
SERIEN_SEED = 2026                                           # Zufalls-Seed der festgeschriebenen Messreihen (tools/messreihen.json)
N_SCHWELLE = 20000                                           # Paare (Referenz 100, Fenster 50) im Normalzustand für die Schwelle „sauber“
N_STROM_SCHWELLE = 6000                                      # Normalströme für die Strom-Schwelle (je Familie, alle Horizonte aus denselben Strömen)
N_KONTROLLE = 1000                                           # neue Normalströme zur Kontrolle der Fehlalarmrate

# Zwecknummern der Zufallsströme (Zufalls-Seed, Zweck, Zeile): jede Wiederholung hat ihren eigenen Strom
Z_BASIS, Z_EXTRA, Z_PERM, Z_SCHWELLE, Z_STROM_BASIS, Z_STROM_EXTRA, Z_STROM_SCHWELLE, Z_KONTROLLE = 11, 12, 13, 14, 15, 16, 17, 18

# ---- Darstellung -----------------------------------------------------------------------------------------------------------------------------
FARBEN = {"W1": "#2F6B65", "KS": "#c0504d", "Energie": "#4c78a8", "KL": "#e0a030", "|Mittel|": "#7f7f7f"}
HISTO_KLASSEN = 20

DEFAULTS = {
    "seed_input": SERIEN_SEED,
    "familie": "Normal",
    "szenario": "Streuung",
    "verschiebung": 0.4,
    "faktor": 1.5,
    "m_aus": 3,
    "hoehe_aus": 8,
    "seeds": 1000,
    "b_perm": 199,
    "stat2": "W1",
    "horizont": H_STANDARD,
    "stroeme": 1000,
    "gipfel_abstand": 36,
    "gipfel_sigma": 8.72,
}

PRESETS = {
    "Mittelwert +0,4 s": {"familie": "Normal", "szenario": "Mittelwert", "verschiebung": 0.4},
    "Streuung ×1,5": {"familie": "Normal", "szenario": "Streuung", "faktor": 1.5},
    "Ausreißerblock": {"familie": "Normal", "szenario": "Ausreißer", "m_aus": 3, "hoehe_aus": 8},
    "Einzelausreißer": {"familie": "Normal", "szenario": "Ausreißer", "m_aus": 1, "hoehe_aus": 50},
    "zweigipfelig": {"familie": "Normal", "szenario": "zweigipfelig", "gipfel_abstand": 36, "gipfel_sigma": 8.72},
    "t(3)": {"familie": "t(3)", "szenario": "Mittelwert", "verschiebung": 0.5},
}
PRESET_HELP = {
    "Mittelwert +0,4 s": "Die Tageswerte des Fensters liegen im Mittel 0,4 Standardabweichungen (8 Einheiten) über der Referenz; sonst ändert sich nichts. Normalverteilung, Permutationstest.",
    "Streuung ×1,5": "Das Fenster streut um den Faktor 1,5 stärker als die Referenz, der Mittelwert bleibt. Normalverteilung.",
    "Ausreißerblock": "3 von 50 Werten des Fensters liegen bei 100 + 8 Standardabweichungen (±5); im Strom ist jeder neue Wert mit 6 % ein solcher Ausreißer.",
    "Einzelausreißer": "1 von 50 Werten des Fensters liegt bei 100 + 50 Standardabweichungen (±5); im Strom ist jeder neue Wert mit 2 % ein solcher Ausreißer.",
    "zweigipfelig": "Das Fenster ist eine Mischung zweier Normalverteilungen (Gipfel bei 82 und 118, je Streuung 8,72): gleicher Mittelwert und gleiche Streuung wie die Referenz, andere Gestalt.",
    "t(3)": "Die Tageswerte folgen einer t-Verteilung mit 3 Freiheitsgraden (auf Standardabweichung 20 normiert); das Fenster liegt 0,5 Standardabweichungen höher.",
}
