# Verteilungsänderung erkennen – wer sieht was, und wann – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-abstand-alarm-demo.streamlit.app/)**

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an einer Annahme an, die die Demo „Abstand zwischen Verteilungen“ stehen lässt: *Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.* Hier gilt dagegen: eine Referenz von 100 und ein Fenster von 50 Tageswerten, ein Rauschen der Stichproben und eine Schwelle, die den Alarm auslöst.

## Warum dieses Problem

Ob sich die Verteilung der Tageswerte geändert hat, etwa der Tagesnachfrage eines Depots, entscheidet man oft mit einem Abstand der Verteilungen von Referenz und aktuellem Fenster: Liegt er über einer Schwelle, gibt es Alarm. Welche Statistik eine Änderung erkennt, hängt an der Art der Änderung und fast ebenso an der Herkunft der Schwelle. Die Demo stellt fünf Statistiken (W1, KS, Energie-Distanz, KL, Mittelwertvergleich), fünf Arten der Änderung und zwei Herkünfte der Schwelle gegenüber und zählt: die Alarmrate (Anteil der Zufalls-Seeds mit Alarm, englisch detection rate, auch Trennschärfe oder power) und die Fehlalarmrate (Alarmanteil im Normalzustand, englisch false alarm rate) stehen in den Experimenten 1 bis 3 getrennt (Experiment 4 zeigt nur die Soll-Linie 5 %). Die Annahme, die geprüft wird, lautet: *Ein Abstand der Verteilungen, der eine Verschiebung zeigt, erkennt auch jede Änderung.* Sie gilt nicht allgemein: W1 hat bei Streuung und beim Ausreißerblock die höchste Alarmrate, bei der Gestalt liegt KL vorn, und beim Einzelausreißer ist W1 im Permutationstest nur mit |Mittel| gleichauf, erkennt ihn mit der Schwelle aus dem Normalzustand aber zu 100 % (Experiment 1 bis 4). Eine Neuheitsaussage ist damit nicht verbunden: Zwei-Stichproben-Tests auf Verteilungsfunktionen sind Lehrstoff, die Demo macht sie vorführbar und mit Zähleinheit nachprüfbar.

**Wortregel.** „Tageswert“ heißt hier ein Wert je Tag (die Verteilung der Tageswerte über die Tage ist eine Variante des Objekts „Zeit“); ein **Tagesprofil** (die Verteilung über 24 Stunden) kommt in dieser Demo nicht vor.

## Modell

Eine **Referenz** $x_1, \dots, x_{100}$ (Tageswerte im Normalzustand, Mittel 100, Standardabweichung 20, Normalverteilung oder t-Verteilung mit 3 Freiheitsgraden, auf Standardabweichung 20 normiert) und ein **Fenster** $y_1, \dots, y_{50}$ (die letzten 50 Tage). Die Einheit **s** ist die Standardabweichung 20. Formeln und Handrechnungen stehen im Expander „📐 Mathematische Formulierung“ der App.

- **Fünf Statistiken** aus den Verteilungsfunktionen $F$ (Referenz) und $G$ (Fenster): **W1** $= \int |F - G|$ (Wasserstein-1-Abstand, L1-Norm der Differenz), **KS** $= \max |F - G|$ (Kolmogorov-Smirnov, L∞), **Energie-Distanz** $= \sqrt{2 \int (F - G)^2}$ (englisch energy distance, L2; gleich `scipy.stats.energy_distance`), **KL** (Kullback-Leibler-Divergenz über 10 Klassen gleicher Masse der gepoolten Werte, +0,5 je Klasse zum Zähler) und **|Mittel|** (Betrag der Differenz der Mittelwerte, englisch comparison of means). **KL-Richtung:** KL(Referenz ‖ Fenster); die Klasse eines Werts folgt aus seinem Rang, gleiche Werte teilen eine Klasse.
- **Schwelle, zwei Wege.** (a) **Permutationstest** (englisch permutation test): die Zugehörigkeit der 150 gepoolten Werte wird $B = 199$-mal zufällig vertauscht, $p = (1 + \#\{T_\pi \ge T_\mathrm{beob}\})/(B + 1)$, Alarm bei $p \le 0{,}05$; die Fehlalarmrate ist dann ohne Annahme über die Verteilung (nur **Austauschbarkeit**, englisch exchangeability) höchstens 5 %. (b) **Schwelle aus dem Normalzustand:** das 95-%-Quantil von $T$ über 20 000 Paare (Referenz 100, Fenster 50) im bekannten Normalzustand der Familie; sie setzt die Familie voraus.
- **Strom:** ein gleitendes Fenster von 50 Werten (Schritt 1) gegen **eine** feste Referenz; die Änderung beginnt mit dem 51. Wert. Die Schwelle je Statistik und Horizont ist das 95-%-Quantil des Maximums von $T$ über alle Fensterschritte des Horizonts in 6 000 Normalströmen: auf **5 % Fehlalarm über den Horizont** eingestellt (bei 150 neuen Werten über 151 Fensterschritte), nicht je Fenster; das Quantil ist geschätzt, die gemessene Rate steht in den Kontrollen von Experiment 3 (4,6 % bis 6,5 %). $k$ ist die Zahl der neuen Werte seit der Änderung (Fenster $k = 0$ rein im Normalzustand, ab $k = 50$ nur veränderte Werte); die **Erkennungsverzögerung** (englisch detection delay) ist das $k$ des ersten Alarms. $k$ zählt Werte, nicht Kalendertage.
- **Arten der Änderung:** Mittelwert +0,4 s (Normal; bei t(3) +0,5 s), Streuung ×1,5, Ausreißerblock (3 von 50 Werten bei 100 + 8 s, ±5), Einzelausreißer (1 von 50 Werten bei 100 + 50 s, ±5), zweigipfelig (Mischung 0,5 N(82; 8,72²) + 0,5 N(118; 8,72²): Mittelwert 100 und Streuung 20,00 wie der Normalzustand). Im Strom kommen Ausreißer mit dem Anteil (Zahl / 50) je neuem Wert statt in fester Zahl: 6 % beim Ausreißerblock, 2 % beim Einzelausreißer.
- **Zählweise:** Zähleinheit ist der **Anteil der Zufalls-Seeds mit Alarm** (je Zeile 1 000 Zufalls-Seeds, Standardfehler höchstens 1,6 Prozentpunkte) und im Strom der **Anteil der Ströme mit Alarm bis k neuen Werten** (1 000 Ströme je Zeile). **Nicht vergleichbar** mit den ARL₁-Zahlen der Demos zur Änderungserkennung: dort ist die Kalibrierung eine mittlere Lauflänge von 370 im Normalzustand (ARL₀), hier 5 % je Horizont; eine Umrechnung ist nicht gemessen.
- **Zufall:** Hausgenerator SplitMix64 (`aba_rng.py`, Ganzzahl-Arithmetik, nicht von der Plattform abhängig), vektorisiert für alle Wiederholungen zugleich (jede Wiederholung hat ihren eigenen Strom aus Zufalls-Seed, Zweck und Zeile; die Blockgröße ändert nichts). Die Normalzahlen entstehen mit Box-Muller und numpy-Funktionen, die in der letzten Binärstelle von der Plattform abhängen können; die Zahlen dieser README sind auf Windows gemessen, ein Lauf auf Linux ist nicht nachgemessen (eine Abweichung wäre im Rahmen des Standardfehlers der Zählungen zu erwarten). Die Basiswerte (Referenz und Fenster) einer Wiederholung sind in allen Szenarien dieselben, nur das Fenster ändert sich (gemeinsame Zufallszahlen). Die Reihen dieser README benutzen den Zufalls-Seed 2026, unabhängig vom Regler.

## Methodik

- **Statistiken** (`aba_statistiken.py`): die fünf Statistiken aus dem sortierten Pool, vektorisiert über Wiederholungen und Vertauschungen.
- **Permutationstest und Schwelle** (`aba_permutation.py`): Vertauschungen aus Zufallsschlüsseln (`argpartition`), p-Werte mit Toleranz für Gleichstände der Statistik, Schwelle aus 20 000 Paaren.
- **Strom** (`aba_strom.py`): Verlauf aller Statistiken über die Fensterschritte, Schwelle je Horizont, erster Alarm, Anteil bis k, Median k.
- **Szenarien und Zufall** (`aba_szenarien.py`, `aba_zufall.py`, `aba_rng.py`): die Arten der Änderung für feste Fenster und Ströme, Normal und t(3).
- **Auswertung** (`aba_evaluation.py`): Zusammenfassungen, gepaarter Vergleich „nicht schlechter als die beste“, vier Verdikte als reine Funktionen, W1 der Grundgesamtheit für Experiment 4.
- **Vorgerechnet** (`tools/messreihen.py` → `tools/messreihen.json`): Schwellen, die neun Standardzeilen, die Läufe mit 4 000 Zufalls-Seeds und die Höhen des Einzelausreißers; jede andere Einstellung rechnet die App live (einige Sekunden).
- **Kopien:** `aba_rng.py` ist die Vorlage des Portfolios (Test auf Gleichheit); `instanz_rng` und die Abstandsfunktionen `normal_cdf`, `verteilungsfunktion`, `w1` sind aus `verteilungsabstand-demo` kopiert (Test auf wörtliche Gleichheit, überspringt sich ohne den Quellordner).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus dem Code der Demo (Zufalls-Seed 2026, 1 000 Zufalls-Seeds beziehungsweise Ströme, $B = 199$, Horizont 150; die Ausnahmen mit anderen Zufalls-Seeds oder 4 000 Zufalls-Seeds sind genannt); jede Tabelle und jeder Satz dieses Abschnitts wird in `tests/test_claims.py` aus der Datei `tools/messreihen.json` gebildet und muss wörtlich hier stehen; die Zahlen sind Messungen dieser Demo und gelten für diese Einstellungen, nicht allgemein.

### Experiment 1: Wer sieht was?

Alarmrate je Statistik im Permutationstest (Referenz 100, Fenster 50, 5 % Fehlalarm). Die letzte Spalte nennt die Statistiken, die nicht schlechter sind als die beste: gepaarter Vergleich je Zufalls-Seed, der Vorsprung der besten Statistik ist höchstens zwei Standardfehler der Differenz. In den Zeilen Normalzustand steht „–“: dort gibt es nichts zu erkennen, und die Statistik mit der höchsten Fehlalarmrate wäre die „beste“.

| Szenario | W1 | KS | Energie | KL | \|Mittel\| | nicht schlechter als die beste (gepaart, 2 Standardfehler) |
|---|---|---|---|---|---|---|
| Normalzustand (Fehlalarmprobe) | 4,3 % | 4,5 % | 5,5 % | 4,0 % | 4,8 % | – |
| Mittelwert +0,4 s | 60,8 % | 47,5 % | 58,7 % | 23,1 % | 61,9 % | W1, \|Mittel\| |
| Streuung ×1,5 | 57,6 % | 17,8 % | 42,3 % | 33,0 % | 8,0 % | W1 |
| 3 von 50 Werten bei +8 s | 63,2 % | 5,5 % | 22,8 % | 7,6 % | 44,4 % | W1 |
| 1 von 50 Werten bei +50 s | 16,2 % | 3,9 % | 8,7 % | 4,6 % | 14,2 % | W1, \|Mittel\| |
| zweigipfelig, gleicher Mittelwert, gleiche Streuung | 12,4 % | 27,1 % | 18,7 % | 51,0 % | 6,7 % | KL |
| t(3): Normalzustand | 4,8 % | 3,7 % | 4,7 % | 5,8 % | 6,2 % | – |
| t(3): Mittelwert +0,5 s | 92,8 % | 94,7 % | 96,0 % | 72,0 % | 83,1 % | Energie |
| t(3): Streuung ×1,5 | 42,0 % | 14,7 % | 34,9 % | 21,9 % | 9,8 % | W1 |

Fehlalarmprobe im Normalzustand (Permutationstest, 1 000 Zufalls-Seeds, beide Familien, zehn Zahlen): 3,7 % bis 6,2 %; mit der Schwelle aus dem Normalzustand: 3,7 % bis 6,0 % (Soll 5 %, Standardfehler 0,7 Prozentpunkte).

Streuung ×1,5, Normalverteilung: Permutationstest W1 57,6 %, KS 17,8 %, Energie 42,3 %, KL 33,0 %, |Mittel| 8,0 %; im Strom hat W1 bis k = 50 in 39,8 % und bis k = 150 in 88,4 % der Ströme Alarm gegeben (Median k 54), KS in 4,9 % und 19,3 %.

Die 57,6 % für W1 sind eine hohe Ziehung (Standardfehler bei 1 000 Zufalls-Seeds 1,6 Prozentpunkte): in drei weiteren Läufen mit je 4 000 Zufalls-Seeds (Zufalls-Seeds 2027, 2028, 2029) liegt W1 bei 54,4 % bis 55,7 %, Energie bei 41,2 % bis 42,1 %; die Rangfolge (W1 vor Energie vor KL vor KS vor |Mittel|) bleibt.

Ausreißerblock (3 von 50 Werten bei +8 s; im Strom 6 % je Wert): W1 63,2 % gegen KS 5,5 % im Permutationstest, 100,0 % mit der Schwelle aus dem Normalzustand; im Strom bis k = 50 / 150: W1 68 % / 98 %, KS 2 % / 8 %.

**Ehrlicher Negativbefund (Zählungen).** (a) W1 gewinnt nicht überall: bei zwei Gipfeln W1 12,4 % gegen KL 51,0 %; bei t(3) mit Mittelwertverschiebung liegen Energie (96,0 %) und KS (94,7 %) über W1 (92,8 %), Abstand 3,2 und 1,9 Prozentpunkte. (b) Der Mittelwertvergleich liegt in 1 von 7 Zeilen mit Änderung über W1 (Permutationstest, Differenz in Prozentpunkten: Mittelwert +0,4 s 1,1); in den übrigen 6 Zeilen liegt er 2,0 bis 49,6 Prozentpunkte darunter. Bei Normalverteilung und Mittelwertverschiebung leistet W1 also nicht mehr als der Mittelwert; bei t(3) mit Mittelwertverschiebung liegen W1, KS und Energie dagegen 9,7 bis 12,9 Prozentpunkte über |Mittel|.

### Experiment 2: Woher kommt die Schwelle?

Dieselben Zufalls-Seeds mit der Schwelle aus dem Normalzustand der jeweiligen Familie (95-%-Quantil von 20 000 Paaren):

| Szenario | W1 | KS | Energie | KL | \|Mittel\| | nicht schlechter als die beste (gepaart, 2 Standardfehler) |
|---|---|---|---|---|---|---|
| Normalzustand (Fehlalarmprobe) | 3,7 % | 4,7 % | 4,9 % | 4,0 % | 4,1 % | – |
| Mittelwert +0,4 s | 60,5 % | 49,5 % | 60,1 % | 23,5 % | 61,6 % | W1, Energie, \|Mittel\| |
| Streuung ×1,5 | 76,7 % | 19,7 % | 54,8 % | 33,6 % | 14,2 % | W1 |
| 3 von 50 Werten bei +8 s | 100,0 % | 6,1 % | 35,0 % | 7,5 % | 78,4 % | W1 |
| 1 von 50 Werten bei +50 s | 100,0 % | 4,1 % | 21,8 % | 4,8 % | 100,0 % | W1, \|Mittel\| |
| zweigipfelig, gleicher Mittelwert, gleiche Streuung | 10,5 % | 30,0 % | 21,4 % | 51,9 % | 5,9 % | KL |
| t(3): Normalzustand | 4,5 % | 4,3 % | 4,7 % | 5,7 % | 6,0 % | – |
| t(3): Mittelwert +0,5 s | 88,2 % | 95,1 % | 96,5 % | 72,1 % | 83,8 % | Energie |
| t(3): Streuung ×1,5 | 49,4 % | 17,6 % | 44,5 % | 22,2 % | 16,0 % | W1 |

Die Schwellen:

| Schwelle (95-%-Quantil) | W1 | KS | Energie | KL | \|Mittel\| |
|---|---|---|---|---|---|
| Normal: Fenster von 50 Werten (20 000 Paare) | 7,286 | 0,230 | 1,309 | 0,254 | 6,810 |
| Normal: Strom, Horizont 150 (6 000 Normalströme) | 10,354 | 0,300 | 1,844 | 0,463 | 10,182 |
| t(3): Fenster von 50 Werten (20 000 Paare) | 7,922 | 0,230 | 1,167 | 0,252 | 6,660 |
| t(3): Strom, Horizont 150 (6 000 Normalströme) | 11,322 | 0,300 | 1,614 | 0,471 | 10,449 |

Einzelausreißer (1 von 50 Werten bei +50 s), Permutationstest, in vier Läufen mit je 1 000 Zufalls-Seeds (Zufalls-Seeds 2026, 2027, 2028, 2029): W1 16,2 %, 15,9 %, 13,5 %, 15,3 % (Bereich 13,5 % bis 16,2 %), |Mittel| 13,2 % bis 14,2 %, KS 3,5 % bis 5,1 %; die Rechnung 5 % · 150/50 gibt 15 %.

Einzelausreißer, Zufalls-Seed 2026: Permutationstest W1 16,2 %, |Mittel| 14,2 %, KS 3,9 %; Schwelle aus dem Normalzustand W1 100,0 %, |Mittel| 100,0 %, KS 4,1 %; Median der p-Werte W1 0,170, KS 0,548; der größte Wert liegt in 33,4 % der Vertauschungen im Fenster (n_w/N = 33,3 %).

Die Rechnung dahinter: Der Ausreißer wandert beim Vertauschen mit; er liegt in $n_w/N = 50/150 = 1/3$ der Vertauschungen im Fenster, und nur dort ist die Statistik so groß wie beobachtet. Liegt er im Fenster, ist $p$ ungefähr $(n_w/N) \cdot U$ mit $U$ gleichverteilt, also $P(p \le 0{,}05) \approx 3 \cdot 0{,}05 = 0{,}15$ und der Median der p-Werte ungefähr $1/6$. Im Spielzeug (Pool aus 6 Werten, einer davon riesig, Referenz 4, Fenster 2) gilt, wenn der Ausreißer im beobachteten Fenster liegt, das Gesetz $P(p \le \alpha) = \alpha \cdot N/n_w$ für $\alpha = k/15$ mit $k = 1, \dots, 5$ exakt (über alle Aufteilungen gemittelt gilt nur $P(p \le \alpha) \le \alpha$) (Handbeispiel in `tests/test_handbeispiele.py`). Das ist eine Deutung, die zu den Zahlen passt, kein Beweis für den Grenzfall. Ein einzelner Wert ändert KS um höchstens $1/n_w = 0{,}02$ (Test mit 100 Zufallsfällen), die Schwelle von KS liegt bei 0,23: das begrenzt den Einfluss eines Werts auf KS, erklärt allein aber nicht, dass die Rate nicht steigt (Fenster im Normalzustand mit KS knapp unter der Schwelle könnten darüber rutschen); gemessen ist „nicht über der Fehlalarmrate“.

(d) Ein Einzelausreißer (+50 s) ist im Permutationstest mit W1 zu 16 % und mit |Mittel| zu 14 % erkennbar, mit Schwelle aus dem Normalzustand zu 100 % und 100 %: für W1 und |Mittel| ändert sich die Antwort mit der Schwelle. KS, Energie und KL erkennen ihn mit keiner der beiden Schwellen zu mehr als 21,8 % (mit der Schwelle aus dem Normalzustand: KS 4,1 %, Energie 21,8 %, KL 4,8 %).

Einzelausreißer bei anderer Höhe (Permutationstest, Zufalls-Seed 2026, 1 000 Zufalls-Seeds): +4 s W1 6,0 %, |Mittel| 5,6 %; +8 s W1 8,5 %, |Mittel| 7,2 %; +20 s W1 14,6 %, |Mittel| 11,4 %; +50 s W1 16,2 %, |Mittel| 14,2 %. Die Rechnung 5 % · 150/50 = 15 % gilt, wenn der Ausreißer die Statistik allein bestimmt, also für sehr hohe Ausreißer; bei +4 s und +8 s liegen die gemessenen Raten deutlich darunter.

### Experiment 3: Wie lange dauert es im Strom?

Anteil der Ströme mit Alarm bis k = 50 / bis k = 150 neuen Werten (1 000 Ströme je Zeile, Schwellen je Horizont 150 aus 6 000 Normalströmen, 5 % Fehlalarm je Horizont). Der Median k ist der Median unter den Strömen mit Alarm bis k = 150 und nur dort aussagekräftig, wo mehr als die Hälfte alarmiert; sonst steht „–“:

| Szenario (Alarm bis k = 50 / bis k = 150) | W1 | KS | Energie | KL | \|Mittel\| | Median k von W1 |
|---|---|---|---|---|---|---|
| Normalzustand (Fehlalarmprobe) | 2,5 % / 5,9 % | 2,2 % / 5,1 % | 2,5 % / 5,8 % | 1,4 % / 4,1 % | 2,6 % / 5,8 % | – |
| Mittelwert +0,4 s | 33,6 % / 72,8 % | 21,0 % / 57,5 % | 31,4 % / 69,6 % | 6,0 % / 27,7 % | 33,7 % / 72,9 % | 53 |
| Streuung ×1,5 | 39,8 % / 88,4 % | 4,9 % / 19,3 % | 18,3 % / 56,8 % | 9,4 % / 39,7 % | 9,2 % / 29,7 % | 54 |
| 3 von 50 Werten bei +8 s | 67,9 % / 98,4 % | 2,5 % / 8,2 % | 21,4 % / 61,8 % | 1,5 % / 6,8 % | 52,7 % / 93,7 % | 38 |
| 1 von 50 Werten bei +50 s | 65,1 % / 96,3 % | 1,9 % / 5,5 % | 28,7 % / 71,1 % | 1,3 % / 4,0 % | 65,1 % / 96,3 % | 33 |
| zweigipfelig, gleicher Mittelwert, gleiche Streuung | 3,1 % / 8,5 % | 7,7 % / 32,8 % | 4,5 % / 16,5 % | 15,4 % / 59,4 % | 2,5 % / 6,8 % | – |
| t(3): Normalzustand | 2,2 % / 4,3 % | 1,9 % / 5,1 % | 1,5 % / 4,5 % | 2,1 % / 5,0 % | 2,5 % / 4,3 % | – |
| t(3): Mittelwert +0,5 s | 51,4 % / 89,7 % | 80,0 % / 99,1 % | 84,3 % / 99,3 % | 30,8 % / 77,9 % | 50,1 % / 87,4 % | 47 |
| t(3): Streuung ×1,5 | 18,4 % / 54,9 % | 3,7 % / 16,9 % | 12,2 % / 46,3 % | 5,4 % / 24,6 % | 6,5 % / 19,9 % | 67 |

Kontrolle der Kalibrierung in 1 000 neuen Normalströmen (Alarm irgendwann bis k = 150):

| Kontrolle (Soll 5 %) | W1 | KS | Energie | KL | \|Mittel\| |
|---|---|---|---|---|---|
| Normal: 1 000 Kontroll-Normalströme, Alarm bis k = 150 | 6,5 % | 5,3 % | 6,0 % | 5,2 % | 5,5 % |
| t(3): 1 000 Kontroll-Normalströme, Alarm bis k = 150 | 4,6 % | 4,9 % | 5,0 % | 5,4 % | 4,9 % |

Die Kontrollzahlen (zehn Zahlen) liegen zwischen 4,6 % und 6,5 %; die größte Abweichung von 5 % ist 1,5 Prozentpunkte (2,2 Standardfehler bei 1 000 Kontrollströmen). Das Quantil der Schwelle ist selbst geschätzt, und die Alarmrate hängt an ihm:

Wie stark die Alarmrate an der geschätzten Schwelle hängt, zeigt Streuung ×1,5 im Strom mit Strom-Schwellen aus drei Sätzen von je 6 000 Normalströmen (Zufalls-Seeds 2026, 2027, 2028): die Schwelle von W1 liegt bei 10,354 bis 10,462, die Alarmrate bis k = 150 bei 87,6 % bis 88,4 %; bei Energie liegt die Schwelle bei 1,844 bis 1,862, die Alarmrate bei 54,4 % bis 56,8 %; bei KL liegt die Schwelle bei 0,463 bis 0,465, die Alarmrate bei 39,3 % bis 39,7 %.

(c) Keine der fünf Statistiken erkennt die zweigipfelige Änderung im Strom bis k = 50 in mehr als 15,4 % der Ströme (der größte Wert gehört zu KL, W1 liegt bei 3,1 %).

Ein Fensterwert, der auf 1 100 gesetzt wird (statt eines Werts um 100), hebt W1 in 1 000 Fenstern im Mittel um 19,1 (obere Schranke Δ/n_w = 20) und KS höchstens um 0,020 (Schranke 1/n_w = 0,020); die Schwelle von KS im Strom liegt bei 0,30, die von W1 bei 10,35.

Ausreißerblock im Strom: mit 6 % je neuem Wert ist bis k = 25 in 78,7 % der Ströme mindestens ein Ausreißer eingetroffen (1 − 0,94^25), W1 hat bis k = 25 in 29,1 % der Ströme Alarm gegeben (KS 1,2 %); in 1 000 Fenstern im Normalzustand, in denen 1 / 2 / 3 Werte durch 260 ersetzt werden, liegt W1 über der Strom-Schwelle 10,35 in 2,2 % / 20,8 % / 80,5 %: ein Block braucht mehrere Werte im Fenster.

Das erklärt, warum W1 einen einzelnen Ausreißer bei +50 s (1100) im Strom beim Eintreffen meldet, einen Ausreißer bei +8 s (260) aber erst mit mehreren Werten im Fenster (der Unterschied liegt an der Höhe, nicht an der Einzelheit); es ist die Rechnung zu den Zahlen, kein Beweis für andere Schwellen.

### Experiment 4: Gestalt ohne Lage und Streuung

Zweigipfelig (Gipfel bei 82 und 118, Streuung je Gipfel 8,72, Gesamtstreuung 20,00): W1 der Verteilungen 4,13 gegen die Schwelle 7,29 für ein Fenster von 50 Werten; Alarmrate W1 12,4 % (Permutationstest), 10,5 % (Schwelle aus dem Normalzustand), 8,5 % (Strom bis k = 150); KL 51,0 %, 51,9 %, 59,4 %.

Der W1-Abstand zwischen der Mischung und der Normalverteilung (4,13) liegt unter dem W1-Rauschen eines Fensters von 50 Werten (Schwelle 7,29): W1 erkennt ihn kaum (8,5 % bis 12,4 %), KL über 10 Klassen in etwa der Hälfte der Wiederholungen (51,0 % bis 59,4 %), nicht in allen. Gilt für diese Mischung (Abstand der Gipfel 36, Streuung je Gipfel 8,72), nicht für jede Gestalt.

## Ehrliche Grenzen

*Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.* Hier gilt Referenz 100, Fenster 50 und eine Schwelle: die Rangfolge der Abstände aus der Demo „Abstand zwischen Verteilungen“ (W1 sättigt nicht, KL ist ohne Glättung bei getrennten Trägern unendlich) gilt hier nicht ohne Weiteres; welche Statistik erkennt, hängt an Art der Änderung, Familie und Schwelle.

- **Aussagen mit Voraussetzungen:** alle Zahlen gelten für Normal(100, 20²) beziehungsweise t(3) auf Standardabweichung 20, eine Referenz von 100 und ein Fenster von 50 Werten, je eine Stärke je Szenario und eine Verteilungsklasse; Zähleinheit Zufalls-Seeds (1 000, Standardfehler höchstens 1,6 Prozentpunkte) beziehungsweise Ströme. Keine Realdaten, keine Stärke-Kurve.
- **Kein „W1 ist der beste Test“:** in den gemessenen Zeilen (Zähleinheit Zufalls-Seeds) hat W1 bei Streuung und beim Ausreißerblock die höchste Alarmrate, beim Einzelausreißer ist es mit |Mittel| gleichauf, bei der Gestalt verliert es, und bei t(3) mit Mittelwertverschiebung liegt es hinter Energie und KS (Experiment 1).
- **Die Schwelle aus dem Normalzustand setzt den Normalzustand voraus** (Familie, Mittel, Streuung); der Permutationstest nicht, er zahlt dafür bei Ausreißern.
- **Strom:** 5 % je Horizont, nicht ARL₀ = 370; kein Vergleich der Verzögerung mit CUSUM, EWMA oder Shewhart bei gleichem ARL₀. Eine feste Referenz; ein Test mit Permutation je Fenster ist nicht gemessen.
- **Nicht gemessen:** andere Größen von Referenz und Fenster, eine verunreinigte oder zu kleine Referenz, mehrere Depots oder Kanäle, abhängige Werte (etwa AR(1)), eine geschätzte statt bekannte Familie, ein Tagesprofil (Verteilung über 24 Stunden) statt der Tageswerte.
- **Literatur** (nur so zitiert, wie gelesen): Zur Dimension verweist die Demo auf Chewi, Niles-Weed, Rigollet, *Statistical optimal transport* (Saint-Flour-Skript 2024) als Hintergrund; die Aussage der Quelle über Zwei-Stichproben-Tests mit W1 in höheren Dimensionen ist hier nicht gegen den Text geprüft, nicht selbst gemessen und wird nicht verwendet, die Demo rechnet nur eine Dimension. Resin u. a., *Shift-dispersion decompositions of Wasserstein and Cramér distances*, arXiv 2408.09770 v3 (Preprint, nur die Zusammenfassung gelesen, nicht nachgerechnet). Weitere Arbeiten zu Tests mit Wasserstein-Abständen sind nur aus Zusammenfassungen bekannt und hier nicht zitiert.
- **Abgrenzung:** Mittelwertänderungen mit Karte, CUSUM und EWMA (Zähleinheit Lauflänge) zeigen die Demos zur Änderungserkennung (`verfahrenswahl-aenderung-demo`, `lastwechsel-gate-demo`), die Zahl der Werte der Referenz (Phase I) `anfangsparameter-demo`, Brüche im Rückblick `segmentierung-offline-demo`, die Wirkung eines Wechsels auf Prognoseintervalle `forecast-interval-demo`. In `regel-verteilungswechsel-demo` heißt „Verteilungswechsel“ etwas anderes (Wechsel der Instanzfamilie).
- **Wo es weitergeht:** Datenbedarf und Dimension (spätere Demo der Reihe); der Name und der Link werden bei der Website-Einbindung ergänzt.

## Verifikation

- **Handbeispiele** (`tests/test_handbeispiele.py`): von Hand in Brüchen gerechnet: drei Statistiken aus einer Differenz der Verteilungsfunktionen (W1 = 3/2, KS = 1/2, ∫(F − G)² = 43/72), exakter Permutationstest (Pool 0, 1, 3, 7, 12: p = 1/10), Einzelausreißer (KS 1/10 → 1/5, W1 3/5 → 998/5), der Ausreißer in der Permutation (p-Werte 1/15 bis 5/15), Ankunft im Strom, Mischung.
- **Orakel** (`tests/test_oracle_abstand_alarm.py`): W1, KS und Energie gegen `scipy.stats` auf 40 Zufallspaaren und 20 Paaren mit Gleichständen; KL gegen eine zweite Schleife (40 Paare, kleine Pools, Gleichstände); die Permutationsverteilung bei N = 10 durch Aufzählung aller 210 Aufteilungen gegen die Zufallsvertauschung (30 Paare); die Fehlalarmprobe (zwei Familien, B = 199 und 99); der Strom vektorisiert gegen eine einfache Schleife (3 Szenarien, gleich auf 1e-12) und einzelne Schritte gegen scipy; W1 der Grundgesamtheit gegen `scipy.integrate.quad` (4 Mischungen); die Zahlen der unabhängigen Nachrechnung im Band.
- **Eigenschaften** (`tests/test_eigenschaften.py`): Skalieren, Verschieben, Vertauschen von Referenz und Fenster, Schranken für einen Wert (100 Fälle), Austauschbarkeit, gleicher Zufalls-Seed gleiches Ergebnis (auch in einem zweiten Prozess mit anderem Hash-Seed), Blockgröße ohne Einfluss, Randfälle (Fenster gleich Referenz, konstante Werte, ein Wert, B = 0, Gleichstände).
- **Zufall und Szenarien** (`tests/test_zufall.py`, `tests/test_szenarien.py`): vektorisierter SplitMix64 bitgleich zum Hausgenerator, Kopier-Tests, jede Art der Änderung wirkt wie beschrieben.
- **Claims** (`tests/test_claims.py`): jede Tabelle und jeder Satz dieser README wird aus der Datei der Messreihe gebildet und muss wörtlich hier stehen; Aussagen über die ganze Tabelle werden gezählt (auch die Läufe mit 4 000 Zufalls-Seeds und die Höhen des Einzelausreißers aus der Datei); zwei Zeilen werden neu gerechnet und im Band verglichen; Konstanten, Kantenwortlaut, Testzahl, verbotene Wörter.
- **Verdikte** (`tests/test_evaluation.py`): je Preset die erwartete Verdikt-Zeile, ein Szenario ohne Änderung meldet keine Wirkung; der Einzelausreißer bei +4 s und +8 s nennt die Rechnung 15 % nicht (erst ab +20 s), seine Warnung nennt die Grenze 5 % plus drei Standardfehler; die Spalte „nicht schlechter als die beste“ ist ohne erkennbare Änderung „–“.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler, jede Art der Änderung in beiden Familien, Permalink, Worst-Case am Regler-Maximum), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Presets, jede Änderung eines Reglers hebt sie auf), Footer-Test, Laufzeit-Importe.
- **Testzahl:** 183 Tests (`pytest --collect-only -q tests`).
- **Zahl zu Test:** Tabellen und Sätze der Befunde: `test_tabellen_und_saetze_der_readme` in `tests/test_claims.py` (wörtlich, aus der Datei `tools/messreihen.json`); Neu-Rechnen der Datei im Band: `test_zeilen_der_datei_neu_gerechnet_im_band`; Handrechnungen: `tests/test_handbeispiele.py`; Alarmraten gegen die unabhängige Nachrechnung: `test_zahlen_der_unabhaengigen_nachrechnung_liegen_im_band` in `tests/test_oracle_abstand_alarm.py`. Was nicht im Test steht, ist als „Messung dieser README, nicht im Test“ gekennzeichnet: die Laufzeiten der Rechnung.
- **Nicht geprüft:** ein Durchklicken der Oberfläche im Browser mit echten Mausklicks; der Linux-Lauf der CI (die Zahlen der Tabellen sind Neu-Rechnen-Tests im Band, die Datei selbst ist fest).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `aba_abstand.py` | W1 auf einem Raster, Normalverteilungsfunktion (Kopie aus `verteilungsabstand-demo`) |
| `aba_constants.py` | Regler-Grenzen, feste Annahmen, Zwecknummern des Zufalls, Presets |
| `aba_evaluation.py` | Zusammenfassungen, gepaarter Vergleich, Verdikte, Formatierung |
| `aba_layout.py` | Fester Kopf (Stylesheet) |
| `aba_permutation.py` | Permutationstest, Schwelle aus dem Normalzustand |
| `aba_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `aba_rng.py` | Hausgenerator SplitMix64 (Kopie der Vorlage) |
| `aba_statistiken.py` | die fünf Statistiken |
| `aba_strom.py` | Verlauf im Strom, Schwelle je Horizont, Erkennungsverzögerung |
| `aba_szenarien.py` | Arten der Änderung für feste Fenster und Ströme |
| `aba_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `aba_vorgerechnet.py` | liest `tools/messreihen.json` |
| `aba_zufall.py` | vektorisierter SplitMix64, Normal- und t(3)-Werte |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/messreihen.py`, `tools/messreihen.json` | die vorgerechneten Schwellen und Standardzeilen |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Zufall, Szenarien, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Eine Kalibrierung des Stroms auf ARL₀ = 370 (Anschluss an die Demos zur Änderungserkennung): erst auf Wunsch.
- Ein Tagesprofil (Verteilung über 24 Stunden) als zweites Vehikel; ein Test mit Permutation je Fenster im Strom; andere Größen von Referenz und Fenster; mehrere Dimensionen; Realdaten.
- MMD (Maximum Mean Discrepancy) wird nicht gerechnet.
- Keine Prognosebewertung (sie steht in der Demo „Abstand zwischen Verteilungen“).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
