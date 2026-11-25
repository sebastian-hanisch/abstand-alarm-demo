"""Verteilungsänderung erkennen – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Abstand der Verteilungen, der eine Verschiebung zeigt, soll auch eine Änderung der Tageswerte erkennen: Diese Annahme wird mit einer Referenz von 100 und einem Fenster von 50 Tageswerten geprüft.
Vier Experimente: wer sieht welche Änderung, woher kommt die Schwelle, wie lange dauert es im Strom, und was W1 bei der Gestalt ohne Lage und Streuung nicht sieht.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import aba_constants as C
import aba_evaluation as E
import aba_statistiken as S
import aba_szenarien as Z
from aba_layout import dunkles_design, kopf_css
from aba_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from aba_visualization import (
    build_alarme_gestalt,
    build_alarmraten,
    build_gestalt,
    build_phist,
    build_schwellen,
    build_strom,
    build_verteilungsfunktionen,
    build_w1_kurve,
)

st.set_page_config(page_title="Verteilungsänderung erkennen – Sebastian Hanisch", layout="wide")
KONFIG = {"displayModeBar": False}
T = E.tabelle_md


@st.cache_data(show_spinner="Der Permutationstest wird gerechnet: Zufalls-Seeds × Vertauschungen × 150 Werte (einige Sekunden) ...")
def _perm(art, familie, p1, p2, seeds, b, seed):
    return E.perm_ergebnis(Z.Szenario(art, familie, p1, p2), seeds, b, seed)


@st.cache_data(show_spinner="Die Ströme werden gerechnet: Ströme × Fensterschritte × 150 Werte (einige Sekunden) ...")
def _strom(art, familie, p1, p2, stroeme, h, seed):
    return E.strom_ergebnis(Z.Szenario(art, familie, p1, p2), stroeme, h, seed)


@st.cache_data(show_spinner=False)
def _w1_pop(abstand, sigma):
    return E.w1_grundgesamtheit(abstand, sigma)


@st.cache_data(show_spinner=False)
def _w1_kurve(sigma):
    return E.w1_kurve(sigma, range(C.GIPFEL_ABSTAND_MIN, C.GIPFEL_ABSTAND_MAX + 1, C.GIPFEL_ABSTAND_SCHRITT))


@st.cache_data(show_spinner=False)
def _beispiel(art, familie, p1, p2, seed):
    ref, fen = Z.paare(seed, 0, 1, Z.Szenario(art, familie, p1, p2))
    return ref[0], fen[0], {k: float(v[0]) for k, v in S.statistiken_paare(ref, fen).items()}


def _args(sz: Z.Szenario):
    return sz.art, sz.familie, sz.p1, sz.p2


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in aba_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("🔔 Verteilungsänderung erkennen")
st.markdown(
    """
Ein **Abstand** der Verteilungen soll sagen, ob sich die Verteilung der **Tageswerte** (ein Wert je Tag, etwa die Tagesnachfrage eines Depots) geändert hat: Eine **Referenz** aus 100 Tagen im Normalzustand und ein **Fenster** aus den letzten 50 Tagen
werden verglichen, und eine **Statistik** (englisch test statistic) über der **Schwelle** (englisch threshold) löst den **Alarm** aus. Die Annahme der Demo lautet: *Ein Abstand, der eine Verschiebung zeigt, erkennt auch jede Änderung.*
Geprüft wird sie mit fünf Statistiken, fünf Arten der Änderung (Mittelwert, Streuung, Ausreißer, zweigipfelig, dazu die Familie t(3)) und zwei Herkünften der Schwelle. W1 hat bei Streuung und beim Ausreißerblock die höchste Alarmrate; bei der Gestalt liegt KL vorn,
und beim Einzelausreißer (+50 s) ist W1 im Permutationstest nur mit |Mittel| gleichauf (mit der Schwelle aus dem Normalzustand erkennen beide ihn zu 100 %). Ein Abstand, der eine Verschiebung zeigt, erkennt also nicht jede Änderung. „Tageswert“ heißt hier ein Wert je Tag; ein **Tagesprofil** (die Verteilung über 24 Stunden) kommt in dieser Demo nicht vor. Die Einheit **s** ist die Standardabweichung 20 der Tageswerte.

Weiter unten: **❓ Kernfragen**, die vier **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
Gezählt wird in **Zufalls-Seeds** (Wiederholungen mit eigenen Zufallszahlen) und **Strömen**; eine Alarmrate ist der Anteil davon mit Alarm.
"""
)

with st.expander("So funktioniert der Alarm", expanded=True):
    st.markdown(
        """
1. **Referenz und Fenster.** Die Referenz $x_1, \\dots, x_{100}$ stammt aus dem Normalzustand (Mittel 100, Standardabweichung 20, Normalverteilung oder t-Verteilung mit 3 Freiheitsgraden, auf Standardabweichung 20 normiert), das Fenster $y_1, \\dots, y_{50}$ sind die letzten 50 Tage.
2. **Fünf Statistiken** $T$ messen, wie verschieden die empirischen Verteilungen $F$ (Referenz) und $G$ (Fenster) sind: **W1** (Wasserstein-1-Abstand, $\\int |F - G|$), **KS** (Kolmogorov-Smirnov, $\\max |F - G|$), **Energie-Distanz**
   (englisch energy distance, $\\sqrt{2 \\int (F - G)^2}$), **KL** (Kullback-Leibler-Divergenz über 10 Klassen gleicher Masse der gepoolten Werte, +0,5 je Klasse geglättet, Richtung KL(Referenz ‖ Fenster)) und **|Mittel|** (Mittelwertvergleich, englisch comparison of means).
3. **Schwelle, zwei Wege.** *Permutationstest* (englisch permutation test): die Zugehörigkeit der 150 gepoolten Werte wird $B$-mal zufällig vertauscht, $p = (1 + \\#\\{T_\\pi \\ge T_\\mathrm{beob}\\})/(B + 1)$, Alarm bei $p \\le 0{,}05$; die **Fehlalarmrate** (englisch false alarm rate)
   ist dann ohne Annahme über die Verteilung höchstens 5 % (**Austauschbarkeit**, englisch exchangeability). *Schwelle aus dem Normalzustand:* das 95-%-Quantil von $T$ über 20 000 Paare (Referenz 100, Fenster 50) aus dem bekannten Normalzustand; sie braucht die Familie.
4. **Strom.** Ein gleitendes Fenster von 50 Werten (Schritt 1) läuft gegen **eine** feste Referenz; die Schwelle ist das 95-%-Quantil des Maximums von $T$ über alle Fensterschritte des Horizonts in Normalströmen, also auf 5 % Fehlalarm **über den Horizont** eingestellt, nicht je Fenster; das Quantil ist geschätzt, die gemessene Rate zeigt die Kontrolle in Experiment 3.
   Die **Erkennungsverzögerung** (englisch detection delay) ist die Zahl der neuen Werte $k$ nach der Änderung bis zum ersten Alarm; $k = 0$ ist ein Fenster rein im Normalzustand, ab $k = 50$ enthält das Fenster nur noch veränderte Werte.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

    st.header("⚙️ Szenario (Experimente 1 bis 3)")
    familie = st.radio("Familie der Tageswerte", C.FAMILIEN, key="familie", horizontal=True, help="Normalverteilung oder t-Verteilung mit 3 Freiheitsgraden (schwere Ränder), beide auf Mittel 100 und Standardabweichung 20. Bei „zweigipfelig“ gilt immer die Normalverteilung.")
    szenario = st.selectbox("Art der Änderung", C.SZENARIEN, key="szenario", help="Was sich im Fenster gegenüber der Referenz ändert: der Mittelwert, die Streuung, einzelne Ausreißer oder die Gestalt (zwei Gipfel bei gleichem Mittelwert).")
    verschiebung = st.select_slider("Mittelwert: Verschiebung (s)", C.VERSCHIEBUNG_OPT, key="verschiebung", format_func=lambda x: f"+{x}".replace(".", ","), help="Um wie viele Standardabweichungen (s = 20) das Fenster höher liegt.", disabled=szenario != "Mittelwert")
    faktor = st.select_slider("Streuung: Faktor", C.FAKTOR_OPT, key="faktor", format_func=lambda x: f"×{x}".replace(".", ","), help="Die Abweichungen der Fensterwerte vom Mittelwert 100 werden mit diesem Faktor multipliziert.", disabled=szenario != "Streuung")
    m_aus = st.select_slider("Ausreißer: Zahl unter 50 Werten", C.M_AUS_OPT, key="m_aus", help="Im festen Fenster ersetzen genau so viele Werte durch Ausreißer; im Strom ist jeder neue Wert mit dem Anteil (Zahl / 50) ein Ausreißer.", disabled=szenario != "Ausreißer")
    hoehe_aus = st.select_slider("Ausreißer: Höhe (s über 100)", C.HOEHE_OPT, key="hoehe_aus", format_func=lambda x: f"+{x}", help="Ein Ausreißer liegt bei 100 + Höhe · 20, mit Streuung 5.", disabled=szenario != "Ausreißer")
    seeds = st.select_slider("Zufalls-Seeds (Wiederholungen)", C.SEEDS_OPT, key="seeds", help="Zahl der unabhängigen Wiederholungen mit je einer Referenz und einem Fenster; je mehr, desto kleiner der Standardfehler der Alarmrate (bei 1 000 höchstens 1,6 Prozentpunkte).")
    b_perm = st.radio("Vertauschungen B", C.B_OPT, key="b_perm", horizontal=True, help="Zahl der zufälligen Vertauschungen im Permutationstest; der kleinste mögliche p-Wert ist 1/(B + 1).")

    st.header("🔬 Experiment 2: Schwelle")
    stat2 = st.selectbox("Statistik für die p-Werte", C.STATISTIKEN, key="stat2", help="Für welche Statistik das Diagramm die Verteilung der p-Werte zeigt; die Alarmraten zeigt die Tabelle für alle.")

    st.header("🔬 Experiment 3: Strom")
    horizont = st.select_slider("Horizont (neue Werte)", C.HORIZONT_OPT, key="horizont", help="Bis zu wie vielen neuen Werten der Strom beobachtet wird; die Schwelle wird für diesen Horizont aus Normalströmen bestimmt (auf 5 % Fehlalarm je Horizont eingestellt, das Quantil ist geschätzt).")
    stroeme = st.select_slider("Zahl der Ströme", C.STROEME_OPT, key="stroeme", help="Zahl der unabhängigen Ströme (je Strom eine eigene Referenz); dazu dieselbe Zahl von Normalströmen als Kontrolle.")

    st.header("🔬 Experiment 4: Gestalt")
    gipfel_abstand = st.slider("Abstand der Gipfel", *bounds("gipfel_abstand"), step=C.GIPFEL_ABSTAND_SCHRITT, key="gipfel_abstand", help="Die Mischung hat Gipfel bei 100 − Abstand/2 und 100 + Abstand/2; gilt auch für die Art „zweigipfelig“ in den Experimenten 1 bis 3.")
    gipfel_sigma = st.select_slider("Streuung je Gipfel", C.GIPFEL_SIGMA_OPT, key="gipfel_sigma", format_func=lambda x: f"{x}".replace(".", ","), help="Mit Abstand 36 und Streuung 8,72 hat die Mischung dieselbe Streuung 20 wie der Normalzustand.")

    st.header("🎲 Zufall")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für alle Zufallszahlen der Demo; die vorgerechneten Zeilen gehören zum Zufalls-Seed 2026.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

werte = {k: st.session_state[k] for k in (
    "seed_input", "familie", "szenario", "verschiebung", "faktor", "m_aus", "hoehe_aus", "seeds", "b_perm", "stat2", "horizont", "stroeme", "gipfel_abstand", "gipfel_sigma")}
sync_query_params(werte)
s = E.settings_aus_zustand(werte)
szen = E.szenario_von(s)
normal = szen.als_normal()

# --- Rechnung ------------------------------------------------------------------------------------------------------------------------------

z = _perm(*_args(szen), s.seeds, s.b_perm, s.seed_input)
z_normal = _perm(*_args(normal), s.seeds, s.b_perm, s.seed_input)
st_strom = _strom(*_args(szen), s.stroeme, s.horizont, s.seed_input)
zwei = Z.Szenario("zwei", "Normal", float(s.gipfel_abstand), float(s.gipfel_sigma))
z_zwei = _perm(*_args(zwei), s.seeds, s.b_perm, s.seed_input)
st_zwei = _strom(*_args(zwei), s.stroeme, s.horizont, s.seed_input)
schwelle_fenster = E.VG.schwellen_fenster(szen.familie)
schwelle_fenster_normal = E.VG.schwellen_fenster("Normal")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
fest = {"streuung": Z.Szenario("streuung", "Normal", 1.5), "zwei": Z.Szenario("zwei", "Normal", 36.0, 8.72), "einzel": Z.Szenario("ausreisser", "Normal", 1.0, 50.0)}
k_streu = _perm(*_args(fest["streuung"]), 1000, 199, C.SERIEN_SEED)
k_zwei = _perm(*_args(fest["zwei"]), 1000, 199, C.SERIEN_SEED)
k_einzel = _perm(*_args(fest["einzel"]), 1000, 199, C.SERIEN_SEED)
k_strom = _strom(*_args(fest["streuung"]), 1000, C.H_STANDARD, C.SERIEN_SEED)
i = {x: C.STATISTIKEN.index(x) for x in C.STATISTIKEN}
_lang = E.VG.streuung_lang_bereich(i["W1"])
streu_lang = (f" (bei 1 000 Zufalls-Seeds Standardfehler bis 1,6 Prozentpunkte; in drei weiteren Läufen mit je {E.ganz(_lang[2])} Zufalls-Seeds liegt W1 bei {E.prozent(_lang[0])} bis {E.prozent(_lang[1])})" if _lang else "")
st.markdown("**Erkennt jede Statistik jede Änderung?**")
st.info(
    f"Mit dem Permutationstest (Referenz 100, Fenster 50, 5 % Fehlalarm, 1 000 Zufalls-Seeds, Normalverteilung) erkennt W1 eine um den Faktor 1,5 aufgeweitete Streuung in {E.prozent(k_streu['alarm_perm'][i['W1']])} der Zufalls-Seeds{streu_lang}, KS in {E.prozent(k_streu['alarm_perm'][i['KS']])}; "
    f"bei zwei Gipfeln mit gleichem Mittelwert und gleicher Streuung liegt W1 bei {E.prozent(k_zwei['alarm_perm'][i['W1']])}, KL bei {E.prozent(k_zwei['alarm_perm'][i['KL']])} (Experiment 1 und 4)."
)
st.markdown("**Hängt die Antwort an der Schwelle?**")
st.info(
    f"Ein einzelner Ausreißer (1 von 50 Werten bei +50 s): Der Permutationstest erkennt ihn mit W1 in {E.prozent(k_einzel['alarm_perm'][i['W1']])}, mit der Schwelle aus dem Normalzustand in {E.prozent(k_einzel['alarm_sauber'][i['W1']])}; "
    f"KS liegt bei beiden Schwellen auf dem Niveau der Fehlalarmrate ({E.prozent(k_einzel['alarm_perm'][i['KS']])} und {E.prozent(k_einzel['alarm_sauber'][i['KS']])}). Die zweite Schwelle setzt voraus, dass der Normalzustand bekannt ist (Experiment 2)."
)
st.markdown("**Wie lange dauert es im Strom?**")
st.info(
    f"Streuung ×1,5 im Strom (5 % Fehlalarm je Horizont von 150 neuen Werten, 1 000 Ströme): W1 hat bis 50 neue Werte in {E.prozent(k_strom['marken']['50'][i['W1']], 0)} der Ströme Alarm gegeben, bis 150 in {E.prozent(k_strom['marken']['150'][i['W1']], 0)}; "
    f"KS bis 150 in {E.prozent(k_strom['marken']['150'][i['KS']], 0)} (Experiment 3). Die Zahlen gelten für diese Kalibrierung, nicht für eine Kalibrierung auf ARL₀ = 370."
)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer sieht was?")
st.caption(
    f"Gewähltes Szenario: **{E.bezeichnung(szen)}** (Einstellungen in der Seitenleiste). Je Zufalls-Seed eine Referenz aus 100 und ein Fenster aus 50 Tageswerten; die Zufallszahlen für Referenz und Fenster sind in allen Szenarien dieselben, "
    "nur das Fenster ändert sich. Die Raute zeigt die Fehlalarmrate im Normalzustand derselben Familie (Fehlalarmprobe). Alarm bei $p \\le 0{,}05$ im Permutationstest."
)
m_cols = st.columns(len(C.STATISTIKEN))
for col, name, wert in zip(m_cols, C.STATISTIKEN, z["alarm_perm"]):
    col.metric(f"Alarmrate {name}", E.prozent(wert, 1), help=C.STAT_LANG[name])
c11, c12 = st.columns([3, 2])
c11.plotly_chart(build_alarmraten(z, z_normal, f"Alarmrate bei 5 % Fehlalarm: {E.bezeichnung(szen)}"), width="stretch", key="alarmraten1", config=KONFIG)
ref_b, fen_b, w_b = _beispiel(*_args(szen), s.seed_input)
c12.plotly_chart(build_verteilungsfunktionen(ref_b, fen_b, w_b), width="stretch", key="vf1", config=KONFIG)
c12.caption("Die Verteilungsfunktionen der ersten Wiederholung. W1 ist die Fläche zwischen den Treppen, KS ihr größter senkrechter Abstand.")
zeilen1 = []
spalte1 = E.nicht_schlechter_spalte(z)          # bei einem Szenario ohne erkennbare Änderung „–“: dort gibt es keine „beste“ Statistik
for j, name in enumerate(C.STATISTIKEN):
    zeilen1.append([C.STAT_LANG[name], E.prozent(z["alarm_perm"][j]), E.prozent(z_normal["alarm_perm"][j]), spalte1[j]])
st.markdown(T(["Statistik", "Alarmrate im Szenario", "Fehlalarmrate im Normalzustand", "nicht schlechter als die beste (gepaart, 2 Standardfehler)"], zeilen1))
kind, text = E.verdikt1(szen, z, z_normal)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Woher kommt die Schwelle?")
st.caption(
    "Dieselben Zufalls-Seeds, zwei Schwellen: der Permutationstest (die Schwelle folgt aus den gepoolten Werten, ohne Annahme über die Familie) und die Schwelle aus dem Normalzustand (95-%-Quantil von 20 000 Paaren, setzt die Familie voraus). "
    "Das Diagramm rechts zeigt die Verteilung der p-Werte der gewählten Statistik (Seitenleiste): ohne Änderung wäre sie gleichverteilt."
)
c21, c22 = st.columns(2)
c21.plotly_chart(build_schwellen(z, E.bezeichnung(szen)), width="stretch", key="schwellen2", config=KONFIG)
c22.plotly_chart(build_phist(z, s.stat2), width="stretch", key="phist2", config=KONFIG)
zeilen2 = []
for j, name in enumerate(C.STATISTIKEN):
    zeilen2.append([C.STAT_LANG[name], E.prozent(z["alarm_perm"][j]), E.prozent(z["alarm_sauber"][j]), E.zahl(schwelle_fenster[name], 3), E.zahl(z["p_median"][j], 3), E.prozent(z["p_anteil_klein"][j])])
st.markdown(T(["Statistik", "Alarmrate Permutationstest", "Alarmrate Schwelle aus dem Normalzustand", "Schwelle (95-%-Quantil, 20 000 Paare)", "Median der p-Werte", "Anteil p ≤ 0,05"], zeilen2))
st.markdown(f"**Anteil der Vertauschungen, in denen der größte Wert im Fenster liegt:** {E.prozent(z['gross'])} (bei n_w/N = 50/150 erwartet {E.prozent(C.N_FENSTER / C.N_POOL)}). Fehlalarmprobe der Schwelle aus dem Normalzustand: "
            + ", ".join(f"{n} {E.prozent(w)}" for n, w in zip(C.STATISTIKEN, z_normal["alarm_sauber"])) + ".")
kind, text = E.verdikt2(szen, z, z_normal)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie lange dauert es?")
st.caption(
    f"Ein Strom von 50 + {s.horizont} Werten gegen eine feste Referenz von 100: die ersten 50 Werte im Normalzustand, ab dem 51. Wert gilt das Szenario ({E.bezeichnung(szen)}). Das Fenster gleitet um einen Wert; k ist die Zahl der neuen Werte seit der Änderung. "
    f"Die Schwelle ist das 95-%-Quantil des Maximums über alle {s.horizont + 1} Fensterschritte in {E.ganz(C.N_STROM_SCHWELLE)} Normalströmen: **auf 5 % Fehlalarm je Horizont eingestellt, nicht ARL₀ = 370** (die gemessene Rate steht in der letzten Spalte); mit den Zahlen der Demos zur Mittelwertänderung sind sie nicht vergleichbar. "
    "Ausreißer kommen im Strom mit dem Anteil (Zahl / 50) je neuem Wert, nicht in fester Zahl. k zählt Werte, nicht Kalendertage."
)
st.plotly_chart(build_strom(st_strom, f"Anteil der Ströme mit Alarm bis k neuen Werten: {E.bezeichnung(szen)}"), width="stretch", key="strom3", config=KONFIG)
marken = sorted(int(k) for k in st_strom["marken"])
kopf3 = ["Statistik"] + [f"Alarm bis k = {k}" for k in marken] + ["Median k (Ströme mit Alarm)", f"Fehlalarm in Normalströmen (bis {s.horizont})"]
zeilen3 = []
for j, name in enumerate(C.STATISTIKEN):
    med = st_strom["median_k"][j]
    mehr_als_haelfte = st_strom["marken"][str(s.horizont)][j] > 0.5            # der Median k gilt nur unter den Strömen mit Alarm, aussagekräftig erst bei mehr als der Hälfte
    zeilen3.append([name] + [E.prozent(st_strom["marken"][str(k)][j], 1) for k in marken] + [E.zahl(med, 0) if med is not None and mehr_als_haelfte else "–", E.prozent(st_strom["kontrolle"][j], 1)])
st.markdown(T(kopf3, zeilen3))
st.caption(f"Schwellen im Strom (Horizont {s.horizont}): " + ", ".join(f"{n} {E.zahl(v, 2)}" for n, v in st_strom["schwelle"].items()) + ". Der Median k ist der Median unter den Strömen mit Alarm bis zum Horizont und nur dort aussagekräftig, wo mehr als die Hälfte der Ströme alarmiert; sonst steht „–“.")
kind, text = E.verdikt3(szen, st_strom)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Gestalt ohne Lage und Streuung")
st.caption(
    "Mischung aus zwei Normalverteilungen mit Gipfeln bei 100 ± Abstand/2: derselbe Mittelwert 100 wie im Normalzustand; mit Abstand 36 und Streuung 8,72 je Gipfel auch dieselbe Streuung 20. "
    "Gezeigt wird W1 der Verteilungen (Grundgesamtheit, numerisch integriert) gegen die Schwelle, die ein Fenster von 50 Werten im Normalzustand schon durch Zufall erreicht, und die Alarmraten aller Statistiken."
)
w1_pop = _w1_pop(float(s.gipfel_abstand), float(s.gipfel_sigma))
kurve = _w1_kurve(float(s.gipfel_sigma))
schw_w1 = schwelle_fenster_normal["W1"]
m41, m42, m43 = st.columns(3)
m41.metric("W1 der Verteilungen", E.zahl(w1_pop, 2), help="Wasserstein-1-Abstand zwischen N(100, 20²) und der Mischung (Grundgesamtheit).")
m42.metric("Schwelle für ein Fenster von 50", E.zahl(schw_w1, 2), help="95-%-Quantil von W1 über 20 000 Paare im Normalzustand (Normalverteilung).")
m43.metric("Gesamtstreuung der Mischung", E.zahl(E.gesamtstreuung_zwei(s.gipfel_abstand, s.gipfel_sigma), 2), help="√(Streuung je Gipfel² + (Abstand/2)²); im Normalzustand 20.")
c41, c42 = st.columns(2)
c41.plotly_chart(build_gestalt(float(s.gipfel_abstand), float(s.gipfel_sigma)), width="stretch", key="gestalt4", config=KONFIG)
c42.plotly_chart(build_w1_kurve(kurve, schw_w1, float(s.gipfel_abstand), float(s.gipfel_sigma)), width="stretch", key="w1kurve4", config=KONFIG)
st.plotly_chart(build_alarme_gestalt(z_zwei, st_zwei, s.horizont), width="stretch", key="alarme4", config=KONFIG)
kind, text = E.verdikt4(float(s.gipfel_abstand), float(s.gipfel_sigma), w1_pop, schw_w1, z_zwei, st_zwei, s.horizont)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Vorgänger | Was hier passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| *Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.* | Hier gilt Referenz 100, Fenster 50 und eine Schwelle: das Rauschen der Stichproben entscheidet, ob ein Abstand etwas erkennt (Experiment 1). Die Rangfolge der Abstände aus der Demo zum Verteilungsabstand (W1 sättigt nicht, KL ist ohne Glättung bei getrennten Trägern unendlich) gilt hier nicht ohne Weiteres: bei der Gestalt liegt KL vorn, bei Streuung und beim Ausreißerblock W1 (beim Einzelausreißer ist W1 mit \\|Mittel\\| gleichauf). | – |
| *Der Normalzustand ist bekannt (Familie, Mittel, Streuung).* | Nur die Schwelle aus dem Normalzustand setzt das voraus; der Permutationstest nicht, er zahlt dafür bei Ausreißern (Experiment 2). | – |
| *Es zählt ein Fenster; die Alarme folgen einander nicht.* | Im Strom gilt eine Schwelle je Horizont; die Kalibrierung auf eine mittlere Lauflänge im Normalzustand (ARL₀) ist hier nicht gerechnet. Mittelwertänderung mit Karte, CUSUM und EWMA zeigen die Demos zur Änderungserkennung. | Änderungserkennung (Demos der Reihe zu Karte, CUSUM, EWMA) |
| *Eine Dimension, 100 und 50 Werte, unabhängige Tageswerte.* | Mehrere Depots oder Kanäle, abhängige Werte und andere Größen von Referenz und Fenster sind nicht gemessen; wie der Datenbedarf mit der Dimension wächst, zeigt diese Demo nicht. | Datenbedarf und Dimension (Demo „Wie viele Daten braucht ein Abstand? Dimension, Struktur, Minibatch“ der Reihe) |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Tageswerte sind unabhängig und stammen im Normalzustand aus Normal(100, 20²) oder aus einer t-Verteilung mit 3 Freiheitsgraden, auf Standardabweichung 20 normiert. Die Referenz hat 100, das Fenster 50 Werte.
- Fehlalarmrate 5 %: im Permutationstest je Fenster ($p \\le 0{,}05$, ohne Annahme über die Verteilung außer Austauschbarkeit), bei der Schwelle aus dem Normalzustand je Fenster, im Strom je Horizont. Die Schwellen sind auf 5 % eingestellt; die gemessenen Raten in den Kontrollen weichen davon ab, weil die Quantile geschätzt sind.
- KL wird als KL(Referenz ‖ Fenster) über 10 Klassen gleicher Masse der gepoolten Werte gerechnet (die Klasse eines Werts folgt aus seinem Rang), je Klasse +0,5 zum Zähler.
- Die Zufallszahlen kommen aus dem Hausgenerator SplitMix64 (Ganzzahl-Arithmetik, nicht von der Plattform abhängig); die Normalzahlen entstehen mit Box-Muller und numpy-Funktionen, die in der letzten Binärstelle von der Plattform abhängen können. Die Zahlen der Demo sind auf Windows gemessen; ein Lauf auf Linux ist nicht nachgemessen, eine Abweichung wäre im Rahmen des Standardfehlers der Alarmraten zu erwarten.

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Zufalls-Seeds beziehungsweise Ströme); je Szenario ist eine Stärke gemessen, keine Kurve der Alarmrate gegen die Stärke.
- Die Alarmrate hat bei 1 000 Zufalls-Seeds einen Standardfehler von höchstens 1,6 Prozentpunkten; Unterschiede von wenigen Punkten sind kein Beleg für eine Rangfolge (der gepaarte Vergleich in Experiment 1 gibt sie an).
- KS erkennt einen einzelnen Ausreißer in diesem Modell nicht über der Fehlalarmrate: ein Wert ändert KS höchstens um 1/50 = 0,02, die Schwelle liegt bei etwa 0,23; das begrenzt den Einfluss eines Werts auf KS, erklärt allein aber nicht, dass die Rate nicht steigt (Fenster im Normalzustand mit KS knapp unter der Schwelle könnten darüber rutschen); gemessen ist „nicht über der Fehlalarmrate“.
- Ein einzelner Ausreißer von +50 s: Der Permutationstest erkennt ihn mit W1 und mit |Mittel| nur zu etwa 13 % bis 16 % (Rechnung 5 % · 150/50 = 15 %), die Schwelle aus dem Normalzustand zu 100 %: für diese beiden Statistiken ändert sich die Antwort mit der Schwelle. KS, Energie und KL erkennen ihn mit beiden Schwellen höchstens zu 22 % (Experiment 2). Das gilt für die gemessene Einstellung, nicht für jeden Test; bei kleineren Ausreißern (+4 s, +8 s) liegen auch W1 und |Mittel| im Permutationstest deutlich unter 15 %.
- Im Strom wird gegen eine feste Referenz gerechnet; ein Test mit Permutation je Fenster (199 Vertauschungen je Schritt) ist nicht gemessen.
- Keine Realdaten.

**Bekannte Probleme und Unsicherheiten.**
- Die Schwelle aus 20 000 Paaren beziehungsweise 6 000 Normalströmen ist ein geschätztes Quantil; die Alarmrate hängt an ihr (die README zeigt es an drei Sätzen von Normalströmen), und die Fehlalarmrate in den Kontrollströmen weicht von 5 % ab (die Zahlen stehen in Experiment 3).
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* eine Referenz $x_1, \dots, x_{n_r}$ ($n_r = 100$) und ein Fenster $y_1, \dots, y_{n_w}$ ($n_w = 50$) von Tageswerten. *Gesucht:* eine Entscheidung „Alarm“ oder „kein Alarm“, die bei unveränderter Verteilung höchstens in 5 % der Fälle alarmiert.

| Symbol | Bedeutung |
|---|---|
| $x_i$, $y_j$ | Tageswerte der Referenz und des Fensters, $N = n_r + n_w = 150$ |
| $F$, $G$ | empirische Verteilungsfunktionen der Referenz und des Fensters |
| $T$ | Statistik (W1, KS, Energie, KL oder $\lvert\text{Mittel}\rvert$) |
| $B$, $p$ | Zahl der Vertauschungen und p-Wert des Permutationstests |
| $c$ | Schwelle (95-%-Quantil von $T$ im Normalzustand) |
| $k$ | Zahl neuer Werte seit der Änderung im Strom |
| $H$ | Horizont: größte Zahl neuer Werte, die der Strom beobachtet |

**Fünf Statistiken aus $F$ und $G$.**
$$W_1 = \int \lvert F - G \rvert\,\mathrm{d}z, \qquad \mathrm{KS} = \max_z \lvert F(z) - G(z) \rvert, \qquad \text{Energie} = \sqrt{2 \int (F - G)^2\,\mathrm{d}z}$$
$$\mathrm{KL}(\text{Ref} \,\|\, \text{Fenster}) = \sum_{c=1}^{10} \hat p_c \ln \frac{\hat p_c}{\hat q_c}, \quad \hat p_c = \frac{r_c + 0{,}5}{n_r + 5}, \ \hat q_c = \frac{w_c + 0{,}5}{n_w + 5}, \qquad \lvert\text{Mittel}\rvert = \lvert \bar x - \bar y \rvert$$
Dabei sind $r_c$ und $w_c$ die Zahlen der Referenz- und Fensterwerte in Klasse $c$ (10 Klassen gleicher Masse der gepoolten Werte).

**Permutationstest.** $p = \dfrac{1 + \#\{\pi : T_\pi \ge T_\mathrm{beob}\}}{B + 1}$ über $B$ zufällige Vertauschungen $\pi$ der Gruppenzugehörigkeit; Alarm bei $p \le 0{,}05$. Unter Austauschbarkeit gilt $P(p \le \alpha) \le \alpha$.
**Schwelle aus dem Normalzustand:** $c$ = 95-%-Quantil von $T$ über 20 000 Paare im Normalzustand; Alarm bei $T > c$. **Strom:** $c_H$ = 95-%-Quantil von $\max_{0 \le t \le H} T_t$ über Normalströme; Alarm beim ersten $t$ mit $T_t > c_H$, $k = t$.

**Handbeispiel 1** (Referenz $0, 1, 2, 3$, Fenster $2, 3, 4$): $F - G$ ist auf den vier Intervallen der Breite 1 gleich $\tfrac14, \tfrac12, \tfrac5{12}, \tfrac13$, also
$$W_1 = \tfrac32, \qquad \mathrm{KS} = \tfrac12, \qquad \int (F - G)^2 = \tfrac{43}{72}, \qquad \text{Energie} = \sqrt{2 \cdot \tfrac{43}{72}} = \tfrac{\sqrt{43}}{6} \approx 1{,}0929.$$
Hier liegt $F$ überall über $G$; dann ist $W_1$ gleich dem Betrag der Differenz der Mittelwerte ($1{,}5$ und $3$).
**Handbeispiel 2** (Pool $0, 1, 3, 7, 12$, Fenster 2, Referenz 3): zehn Aufteilungen; die Aufteilung $\{7, 12\}$ hat mit $W_1 = \tfrac{49}{6}$ die größte Statistik, also $p = \tfrac1{10}$. Die kleinste Zahl ist bei $B = 199$ der p-Wert $\tfrac1{200}$.
**Handbeispiel 3** (Ausreißer): Pool aus $98, 99, 100, 101, 102$ und $1100$, Referenz 4, Fenster 2: von 15 Aufteilungen haben 5 den Ausreißer im Fenster ($\tfrac13 = n_w/N$), ihre p-Werte sind $\tfrac1{15}, \tfrac2{15}, \dots, \tfrac5{15}$, also, wenn der Ausreißer im beobachteten Fenster liegt, $P(p \le \tfrac{k}{15}) = \tfrac{k}{5} = \tfrac{k}{15} \cdot \tfrac{N}{n_w}$ für $k = 1, \dots, 5$ (über alle Aufteilungen gemittelt gilt nur $P(p \le \alpha) \le \alpha$). Im Grenzfall $N = 150$, $n_w = 50$ gilt näherungsweise $P(p \le 0{,}05) \approx 0{,}05 \cdot 3 = 0{,}15$.
**Schranken:** ein Fensterwert, der sich um $\Delta$ ändert, ändert KS um höchstens $1/n_w = 0{,}02$ und $W_1$ um höchstens $\Delta / n_w$ (bei $\Delta = 1000$ also höchstens 20).
**Ankunft im Strom:** ist jeder neue Wert mit Anteil $q$ ein Ausreißer, ist bis $k$ neuen Werten mindestens einer eingetroffen mit $1 - (1 - q)^k$; für $q = 0{,}02$ und $k = 25, 50, 100, 150$ sind das $0{,}397;\ 0{,}636;\ 0{,}867;\ 0{,}952$.
**Mischung:** die Mischung $\tfrac12 N(82;\, 8{,}72^2) + \tfrac12 N(118;\, 8{,}72^2)$ hat den Mittelwert 100 und die Varianz $18^2 + 8{,}72^2 = 400{,}04$ (Streuung 20,00); $W_1$ zur Normalverteilung $N(100, 20^2)$ ist numerisch $4{,}13$.

**Literatur** (nur so zitiert, wie gelesen): Zwei-Stichproben-Tests auf Verteilungsfunktionen und Permutationstests sind Lehrstoff. Zur Dimension verweist die Demo auf Chewi, Niles-Weed, Rigollet, *Statistical optimal transport* (Saint-Flour-Skript 2024) als Hintergrund; die Aussage der Quelle über Zwei-Stichproben-Tests mit $W_1$ in höheren Dimensionen ist hier nicht gegen den Text geprüft, nicht selbst gemessen und wird nicht verwendet, und diese Demo rechnet nur eine Dimension.
Resin u. a., *Shift-dispersion decompositions of Wasserstein and Cramér distances*, arXiv 2408.09770 v3 (Preprint, nur die Zusammenfassung gelesen, nicht nachgerechnet). Weitere Arbeiten zu Tests mit Wasserstein-Abständen sind nur aus Zusammenfassungen bekannt und hier nicht zitiert.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html)."
)
