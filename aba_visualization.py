"""Plotly-Figuren: alle Achsen fest (fixedrange), Beschriftungen als Annotationen, eine Farbe je Statistik."""

import numpy as np
import plotly.graph_objects as go

import aba_constants as C

SCHRIFT = dict(size=12)
GRAU = "rgba(128,128,128,0.8)"


def _fest(fig: go.Figure, hoehe: int, titel: str | None = None) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=hoehe, margin=dict(l=55, r=20, t=50 if titel else 30, b=80), title=dict(text=titel, x=0.0, font=dict(size=14)) if titel else None,
                      dragmode=False, legend=dict(orientation="h", y=-0.3, x=0.0), font=SCHRIFT)
    return fig


def _prozent_achse(fig: go.Figure, obergrenze: float = 100.0) -> None:
    fig.update_yaxes(title="Alarmrate (%)", range=[0, obergrenze], ticksuffix=" %")


def build_verteilungsfunktionen(ref: np.ndarray, fen: np.ndarray, werte: dict) -> go.Figure:
    """Die empirischen Verteilungsfunktionen F (Referenz, 100 Werte) und G (Fenster, 50 Werte) einer Wiederholung; die Statistiken stehen als Annotation."""
    fig = go.Figure()
    for name, x, farbe in (("F: Referenz (100 Tage)", ref, "#4c78a8"), ("G: Fenster (50 Tage)", fen, "#c0504d")):
        xs = np.sort(x)
        fig.add_trace(go.Scatter(x=np.concatenate([[xs[0] - 5], xs, [xs[-1] + 5]]), y=np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs), [1]]), mode="lines", line=dict(shape="hv", color=farbe), name=name))
    text = "; ".join(f"{s} {w:.2f}".replace(".", ",") for s, w in werte.items())
    fig.add_annotation(x=0.99, y=0.04, xref="paper", yref="paper", text=text, showarrow=False, xanchor="right", font=dict(size=11), bgcolor="rgba(255,255,255,0.75)", bordercolor=GRAU, borderwidth=1)
    fig.update_xaxes(title="Tageswert")
    fig.update_yaxes(title="Anteil der Tage bis zum Wert", range=[0, 1.02])
    return _fest(fig, 330)


def build_alarmraten(z: dict, z_normal: dict, titel: str, schluessel: str = "alarm_perm") -> go.Figure:
    """Alarmrate je Statistik im Szenario (Balken) und im Normalzustand (Marke, Fehlalarmprobe); Linie bei 5 %."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(C.STATISTIKEN), y=[100 * v for v in z[schluessel]], marker_color=[C.FARBEN[s] for s in C.STATISTIKEN], name="Alarmrate im Szenario", text=[f"{100 * v:.1f} %".replace(".", ",") for v in z[schluessel]], textposition="outside"))
    fig.add_trace(go.Scatter(x=list(C.STATISTIKEN), y=[100 * v for v in z_normal[schluessel]], mode="markers", marker=dict(symbol="diamond", size=10, color="black"), name="Fehlalarmrate im Normalzustand"))
    fig.add_hline(y=100 * C.ALPHA, line_dash="dot", line_color=GRAU, annotation_text="Soll 5 %", annotation_position="top right")
    _prozent_achse(fig, 112)
    return _fest(fig, 360, titel)


def build_schwellen(z: dict, titel: str) -> go.Figure:
    """Alarmrate je Statistik für beide Schwellen: Permutationstest und Schwelle aus dem Normalzustand."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(C.STATISTIKEN), y=[100 * v for v in z["alarm_perm"]], name="Permutationstest (p ≤ 5 %)", marker_color="#4c78a8", text=[f"{100 * v:.0f} %" for v in z["alarm_perm"]], textposition="outside"))
    fig.add_trace(go.Bar(x=list(C.STATISTIKEN), y=[100 * v for v in z["alarm_sauber"]], name="Schwelle aus dem Normalzustand", marker_color="#e0a030", text=[f"{100 * v:.0f} %" for v in z["alarm_sauber"]], textposition="outside"))
    fig.add_hline(y=100 * C.ALPHA, line_dash="dot", line_color=GRAU, annotation_text="Soll 5 %", annotation_position="top right")
    fig.update_layout(barmode="group")
    _prozent_achse(fig, 115)
    return _fest(fig, 360, titel)


def build_phist(z: dict, stat: str) -> go.Figure:
    """Verteilung der p-Werte der gewählten Statistik über die Zufalls-Seeds (20 Klassen); links der Linie p = 0,05 liegt der Alarm. Unter Gleichverteilung wäre jede Klasse gleich hoch."""
    i = C.STATISTIKEN.index(stat)
    h = z["p_hist"][i]
    mitte = [(k + 0.5) / C.HISTO_KLASSEN for k in range(C.HISTO_KLASSEN)]
    fig = go.Figure(go.Bar(x=mitte, y=h, width=1.0 / C.HISTO_KLASSEN * 0.9, marker_color=[C.FARBEN[stat] if m < C.ALPHA else "rgba(150,150,150,0.55)" for m in mitte], name=f"p-Werte von {stat}"))
    fig.add_vline(x=C.ALPHA, line_dash="dot", line_color="black", annotation_text="p = 0,05", annotation_position="top right")
    fig.add_hline(y=z["seeds"] / C.HISTO_KLASSEN, line_dash="dot", line_color=GRAU, annotation_text="gleichverteilt", annotation_position="top right")
    fig.update_xaxes(title=f"p-Wert von {stat}", range=[0, 1])
    fig.update_yaxes(title="Zufalls-Seeds je Klasse")
    return _fest(fig, 330)


def build_strom(st: dict, titel: str) -> go.Figure:
    """Anteil der Ströme mit Alarm bis k neuen Werten, je Statistik; die senkrechte Linie bei k = 50 markiert, ab wann das Fenster nur noch veränderte Werte enthält."""
    anteil = np.array(st["anteil"])
    k = np.arange(anteil.shape[0])
    fig = go.Figure()
    for i, s in enumerate(C.STATISTIKEN):
        fig.add_trace(go.Scatter(x=k, y=100 * anteil[:, i], mode="lines", name=s, line=dict(color=C.FARBEN[s], width=2.5 if s == "W1" else 1.8)))
    fig.add_hline(y=100 * C.ALPHA, line_dash="dot", line_color=GRAU, annotation_text="5 % je Horizont", annotation_position="bottom right")
    if st["h"] >= C.N_FENSTER:
        fig.add_vline(x=C.N_FENSTER, line_dash="dot", line_color="black", annotation_text="k = 50: Fenster ganz verändert", annotation_position="top left")
    fig.update_xaxes(title="neue Werte k seit der Änderung", range=[0, st["h"]])
    _prozent_achse(fig, 102)
    return _fest(fig, 380, titel)


def build_gestalt(abstand: float, sigma_gipfel: float) -> go.Figure:
    """Dichten von N(100, 20²) und der Mischung zweier Gipfel (gleicher Mittelwert; die Streuung steht in der Legende)."""
    x = np.linspace(0, 200, 401)
    dichte = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    mischung = 0.5 * dichte(C.MITTEL - abstand / 2, sigma_gipfel) + 0.5 * dichte(C.MITTEL + abstand / 2, sigma_gipfel)
    gesamt = float(np.sqrt(sigma_gipfel ** 2 + (abstand / 2) ** 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dichte(C.MITTEL, C.SIGMA), mode="lines", name="Normalzustand N(100, 20²)", line=dict(color="#4c78a8")))
    fig.add_trace(go.Scatter(x=x, y=mischung, mode="lines", name=f"Mischung, Gesamtstreuung {gesamt:.1f}".replace(".", ","), line=dict(color="#c0504d")))
    fig.update_xaxes(title="Tageswert")
    fig.update_yaxes(title="Dichte", rangemode="tozero")
    return _fest(fig, 330)


def build_w1_kurve(kurve: list, schwelle: float, d_marker: float, sigma_gipfel: float) -> go.Figure:
    """W1 der Verteilungen (Grundgesamtheit) gegen den Abstand der Gipfel, mit der Schwelle für ein Fenster von 50 Werten (95-%-Quantil im Normalzustand)."""
    d = [p[0] for p in kurve]
    w = [p[1] for p in kurve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=w, mode="lines+markers", name=f"W1 der Verteilungen (Streuung je Gipfel {sigma_gipfel:g})".replace(".", ","), line=dict(color=C.FARBEN["W1"])))
    fig.add_hline(y=schwelle, line_dash="dash", line_color="black", annotation_text=f"Schwelle {schwelle:.2f} (Fenster von 50)".replace(".", ","), annotation_position="top left")
    fig.add_vline(x=d_marker, line_dash="dot", line_color=GRAU)
    fig.update_xaxes(title="Abstand der Gipfel")
    fig.update_yaxes(title="W1 (Einheiten des Tageswerts)", rangemode="tozero")
    return _fest(fig, 330)


def build_alarme_gestalt(z: dict, st: dict, h: int) -> go.Figure:
    """Alarmraten je Statistik für die Mischung: Permutationstest, Schwelle aus dem Normalzustand, Strom bis h neuen Werten."""
    fig = go.Figure()
    quellen = (("Permutationstest", z["alarm_perm"], "#4c78a8"), ("Schwelle aus dem Normalzustand", z["alarm_sauber"], "#e0a030"), (f"Strom bis {h} neue Werte", st["marken"][str(h)], "#7f7f7f"))
    for name, werte, farbe in quellen:
        fig.add_trace(go.Bar(x=list(C.STATISTIKEN), y=[100 * v for v in werte], name=name, marker_color=farbe, text=[f"{100 * v:.0f} %" for v in werte], textposition="outside"))
    fig.add_hline(y=100 * C.ALPHA, line_dash="dot", line_color=GRAU, annotation_text="Soll 5 %", annotation_position="top right")
    fig.update_layout(barmode="group")
    _prozent_achse(fig, 112)
    return _fest(fig, 360)

