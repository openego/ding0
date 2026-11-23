# Testplan ding0: Physik, Geografie, Vollständigkeit

Stand: 2026-11-09, Branch `feature/status2024`. Nur Plan, noch kein Testcode.

Ziel: Tests, die prüfen, ob die erzeugten Netze **physikalisch stimmen**
(Kirchhoff, Ohm, Spannungsband, thermische Grenzen), **geografisch stimmen**
(alles liegt dort, wo es liegen soll, im richtigen CRS) und **vollständig sind**
(alles aus egon-data ist im Netz, nichts doppelt, nichts still verworfen).

---

## 0. Grundsätze

1. **Invarianten statt Referenzzahlen.** Ein Test sagt, *welche Eigenschaft*
   gilt (z. B. „Summe der Lasten im Export = Summe der Gebäudelasten“), nicht
   „Netz hat 1234 Leitungen“. Die bestehenden Golden-Files
   (`synthetic_*_expected.json`) bleiben als Regressionsschutz, sind aber kein
   Beleg für Korrektheit, denn sie enthalten die Upstream-Bugs.
2. **Unabhängige Gegenrechnung.** Physik wird nicht mit ding0-eigenen
   Formeln geprüft, sondern gegen
   - eine eigene Knotenpunkt-Rechnung mit numpy (Admittanzmatrix aus den
     exportierten r, x), um PyPSA *und* den Export zu prüfen,
   - analytische Kleinstfälle (eine Leitung, eine Last: ΔU ≈ (P·R + Q·X)/U),
   - Datenblattwerte (NAYY, NA2XS2Y, Trafos nach Werth 2016).
3. **Eine gemeinsame Invarianten-Bibliothek** `tests/invariants/`
   (`physics.py`, `geo.py`, `completeness.py`). Jede Prüfung gibt *alle*
   Verletzungen als Liste zurück (nicht beim ersten Fehler abbrechen), damit
   ein Testlauf ein vollständiges Bild liefert. Dieselben Funktionen laufen auf
   - den synthetischen Bezirken (ohne DB, in CI),
   - echten egon-data-Bezirken (Marker `db`, Hypothesis zieht Bezirke, wie in
     `test_egon_data_properties.py`),
   - Zwischenständen der Pipeline (nach Schritt 5, 6, 7, 8, 10, 11).
4. **Gefundene Bugs zuerst als Test.** Jeder Verdachtsfall aus Abschnitt 6
   bekommt einen Test, der ihn belegt (`xfail(strict=True)` mit Verweis), bevor
   er behoben wird. So bleibt sichtbar, welche Invariante heute verletzt ist.
5. **Ebenen.** Unit (reine Funktionen, handgebaute Mini-Graphen, Hypothesis) →
   Komponente (ein Pipeline-Schritt auf synthetischen Daten) → E2E.

### Infrastruktur

| Baustein | Zweck |
|---|---|
| `tests/invariants/physics.py` | KCL, Ohm, Energiebilanz, Spannungsband, thermische Grenzen, Trafo-Dimensionierung |
| `tests/invariants/geo.py` | CRS-Bereiche, Enthaltensein, Leitungsgeometrie = Busse, Längen, Distanzgrenzen |
| `tests/invariants/completeness.py` | Abgleich Eingangsdaten ↔ Netz ↔ Export (Anzahl, Summen, IDs) |
| Fixture `staged_synthetic_run` | führt `run_ding0` Schritt für Schritt aus und hält jeden Zwischenstand (Kopie bzw. Re-Run bis Schritt n) |
| Fixture `tiny_grids` | Baukasten für analytische Netze: Station–Leitung–Last, Ring mit Trennstelle, LV-Strang mit n Häusern |
| Erweiterung `synthetic_district.py` | Grenzfälle gezielt: Gebäude genau auf 100/200/5500 kW, Generator mit v_level 3/None, Generator außerhalb jeder Leitungs-Reichweite, zwei Bezirke mit Generatoren in beiden |
| Marker | `db` (egon-data), `slow`, `diagnostic` (meldet Kennzahlen, schlägt nicht fehl) |

---

## 1. Physik-Invarianten (gelten nach jedem Leistungsfluss)

Für jeden Snapshot: `t0` = Lastfall, `t1` = Einspeisefall.

| ID | Invariante | Prüfung |
|---|---|---|
| P1 | **Konvergenz** | `network.pf()` meldet für alle Snapshots `converged`; ding0 prüft das heute nicht |
| P2 | **KCL** an jedem Bus | Σ Einspeisung − Σ Last − Σ abgehende Flüsse (Leitungen p0/p1, Trafos) < 1e-6 MW, für P und Q |
| P3 | **Ohm / KVL** pro Leitung | aus v_mag, v_ang an bus0/bus1 und exportiertem r, x: S0 = V0·conj((V0−V1)/Z) stimmt mit PyPSA `p0, q0` überein. Prüft gleichzeitig Einheiten (Ω, km) im Export |
| P4 | **Verluste ≥ 0** | p0 + p1 ≥ −ε für jede Leitung; Verluste ≈ r·I² |
| P5 | **Energiebilanz** | Slack-P = Σ Last − Σ Erzeugung + Σ Verluste; Verluste im Lastfall < 5 % der Last (Plausibilität) |
| P6 | **Lastfall/Einspeisefall korrekt besetzt** | t0: Lasten = Spitzenlast, Erzeuger = `generation_in_load_case`·P; t1: Erzeuger = P_nom, Lasten = `load_in_generation_case`·Spitzenlast; Q = P·tan φ mit Vorzeichen nach `q_sign` (Last induktiv: Q > 0; Erzeuger kapazitiv) |
| P7 | **Spannungsband MV** nach Verstärkung | \|v − v_Station\| ≤ `mv_max_v_level_lc_diff_normal` (t0) bzw. `…_fc_diff_normal` (t1) an jedem MV-Knoten, sonst Warnung „cannot be reinforced“ im Log (dann Test muss diesen Fall kennen) |
| P8 | **Spannungsband LV** (diagnostisch) | voller Leistungsfluss MV+LV (`only_export_mv=False`): LV-Abweichung ≤ `lv_max_v_level_lc_diff_normal` bzw. `…_fc_diff_normal`. LV-Verstärkung ist in ding0 auskommentiert, deshalb vermutlich verletzt → zuerst `diagnostic`, Kennzahlen berichten |
| P9 | **Thermische Grenze MV** nach Verstärkung | s_res ≤ √3·U_n·I_max_th·num_parallel·load_factor (Leitung/Kabel, lc/fc) |
| P10 | **Thermische Grenze LV** | Strom über jedes Kabel (aus Downstream-Summe im Baum oder aus PF) ≤ I_max_th·`load_factor_lv_cable_*` |
| P11 | **Trafo-Dimensionierung LV** | Σ S_nom der Trafos einer Station · lf ≥ max(Spitzenlast/cos φ_load, Erzeugung/cos φ_gen) der LV-Netze (inkl. angeschlossener LV-Erzeuger und Punktlasten) |
| P12 | **Trafo-Dimensionierung HS/MS (n-1)** | (Σ S_nom − größter Trafo) · lf ≥ max(Last, Erzeugung MV+LV) des Bezirks |
| P13 | **Radialbetrieb** | mit offenen Trennstellen ist das MV-Netz ein Baum, der die HS/MS-Station enthält; jedes LV-Netz ein Baum mit der Station als Wurzel; keine isolierten Knoten |
| P14 | **Ringstruktur** | jeder Ring hat genau eine Trennstelle; Ring geschlossen = Zyklus über die Station; Ringknoten ≤ `load_area_count_per_ring`; Halbringlänge ≤ `max_half_ring_length` (nach Routing; nach Satellitenanschluss diagnostisch) |
| P15 | **Spannungsebenen** | MV-Busse v_nom ∈ {10, 20} kV und = `mv_grid.v_level`; LV-Busse 0,4 kV; Leitungen verbinden Busse gleicher v_nom; Trafo bus0 = höhere Spannung |
| P16 | **Impedanzen konsistent mit Katalog** | r = R'·l/n_par, x = ω·L'·1e-3·l/n_par, s_nom = √3·U·I·n_par (Einheiten MV kV/A → MVA, LV V/A → MVA); Leitungstyp ∈ Katalog des aktiven Planungsprofils |
| P17 | **Trafo-Ersatzdaten** | r_pu = P_k/S_nom, x_pu = √(u_k² − r_pu²), 0 < r_pu < x_pu < u_k |

## 2. Geo-Invarianten

| ID | Invariante | Prüfung |
|---|---|---|
| G1 | **CRS-Bereich** | intern alle Koordinaten EPSG:3035 innerhalb Deutschland-BBox (x ≈ 4.0–4.7e6, y ≈ 2.6–3.6e6); `to_csv`/`to_netcdf` in EPSG:4326 innerhalb lon 5.8–15.1, lat 47.2–55.1; `srid`-Spalte passt zu den Koordinaten (bzw. dokumentierte Ausnahme `to_dataframe`) |
| G2 | **Enthaltensein** | HS/MS-Station ∈ MVGD-Polygon; jede Load Area ⊂ MVGD (Toleranz); jede LV-Station ∈ ihrer Load Area (bzw. aktualisierter `geo_area`); jedes Gebäude ∈ Load Area; Erzeuger und Punktkomponenten ∈ MVGD; LVGD-Polygon (konvexe Hülle) enthält Station und alle seine Gebäude |
| G3 | **Leitungsgeometrie passt zu Bussen** | Anfangs- und Endpunkt der Geometrie liegen auf den Koordinaten von bus0/bus1 (≤ 1 m) |
| G4 | **Leitungslängen** | MV-Luftlinien: `length` = Geometrielänge · `branch_detour_factor`; LV entlang Straßen: `length` = Geometrielänge (±1 %); length ≥ Luftlinie bus0–bus1; Mindestlänge 1 m nur bei zusammenfallenden Punkten |
| G5 | **Distanzgrenzen** | Weg (Straßengraph) Station → jede LV-Last ≤ `ons_dist_threshold`, außer es gibt die zugehörige Meldung in `network.message`; Hausanschluss (Gebäude → nächster Straßenknoten) plausibel kurz (Verteilung berichten) |
| G6 | **Anschlussort** | LV-Erzeuger/-Komponenten Ebene 7 hängen an *ihrem* Gebäude (building_id) bzw. am nächsten; Ebene 6 an der nächsten LV-Station (Brute-Force-Gegenprobe) und **im eigenen MV-Bezirk**; Ebene 4 per eigener Leitung an der HS/MS-Station; Ebene 5 am nächsten Knoten/Leitung |
| G7 | **Trennstelle** | `geo_data` der Trennstelle liegt auf der Geometrie ihrer Leitung (Mittelpunkt) |
| G8 | **Keine Phantombusse** | keine zwei Nicht-Helfer-Busse eines Netzes am selben Ort; keine Busse bei (0, 0) oder NaN |
| G9 | **MV-Netz im Bezirk** (diagnostisch) | Anteil der MV-Leitungslänge außerhalb des MVGD-Polygons berichten |

## 3. Vollständigkeits-Invarianten („nichts vergessen, nichts doppelt“)

| ID | Invariante | Prüfung |
|---|---|---|
| V1 | **Gebäude-Partition** | jedes Gebäude der Load Area landet in genau einer Klasse: LV-Straße (< `lv_threshold_capacity`), LV direkt an Station (< `mv_lv_threshold_capacity`), MV-Last (< `hv_mv_threshold_capacity`), verworfen mit Logeintrag (≥ HS-Grenze). Grenzwerte exakt auf 100/200/5500 kW testen |
| V2 | **Lastsummen** | Σ capacity der Gebäude (inkl. Wärmepumpen·Faktor, Heim-Ladepunkte·Faktor) = Σ LV-Lasten + Σ MV-Lasten + Σ verworfen; Haushaltszahl bleibt erhalten |
| V3 | **Spitzenlast-Kette** | `mvgd.peak_load` = Σ LA-Spitzenlast + Σ MV-Lasten + Σ Punktlasten; `la.peak_load` = Σ `lvgd.peak_load` (Modus `sum_of_loads`); Σ exportierte Last-p_set in t0 = `mvgd.peak_load`/1e3 (mit Komponenten-Faktoren) |
| V4 | **Erzeuger** | jeder Erzeuger aus egon-data (v_level 4–7, ≤ `max_generator_capacity`) genau einmal im Export, gleiche Leistung, Typ, Untertyp, weather_cell_id; Σ je Typ gleich; ungültige v_level führen zu Fehler/Logeintrag, nicht zu stillem Verlust; **in Mehr-Bezirks-Läufen für jeden Bezirk** |
| V5 | **Komponenten** | jede `source_id` (Ladepunkt, Busdepot, Bahn, Großspeicher) genau einmal; Heim-Ladepunkte entweder an Gebäude oder als Punktlast, nie beides; Wärmepumpen/Heimspeicher am richtigen Gebäude |
| V6 | **Load Areas** | jede LA aus der DB ist im Netz oder hat einen Grund im Log (unter Schwelle, keine Gebäude, kein Graph, Clustering gescheitert); keine LA wird übersprungen, weil eine frühere scheiterte |
| V7 | **Keine isolierten Knoten** | nach dem Lauf kein Generator, keine Last, keine Station ohne Kante; der Quickfix #402 (Entfernen von Teilnetzen) greift auf synthetischen Daten nie; greift er, wird die entfernte Last gemeldet |
| V8 | **Aggregierte Load Areas** | Last und Erzeugung erscheinen im Export am `BusBar_lac_*` mit passenden Trafos |
| V9 | **Eindeutige Namen** | Bus-, Leitungs-, Last-, Erzeugernamen eindeutig, auch über mehrere Bezirke |
| V10 | **Export ↔ Objektmodell** | Anzahl LV-Lasten im Graph = Zeilen in `loads` (je Sektor aufgefächert); jede Kante (ohne Helfer) = eine Zeile in `lines`; jede offene Trennstelle = eine Zeile in `switches` mit existierendem `bus_open` |

---

## 4. Plan je Modul und Funktion

Legende: **[P]** Physik, **[G]** Geo, **[V]** Vollständigkeit, **[U]** Unit,
**[H]** Hypothesis/Property, **[S]** nur Smoke. Priorität: ★★★ hoch,
★★ mittel, ★ niedrig.

### 4.1 `tools/egon_data_integration.py` (Datenimport)

- **get_scenario** [U] ★ – Default aus Config, Override greift.
- **check_load_area_peak_load** [U] ★ – NaN/0 stumm; Abweichung > Rundung → Warnung.
- **get_mv_data** [G][V] ★★ (db) – Station ∈ Polygon (G2), genau eine Zeile je bus_id, Polygon gültig, Station in 4326 / Polygon in 3035 wie dokumentiert.
- **get_lv_load_areas** [V] ★★ – MW→kW-Faktor genau einmal angewendet (Summe gegen SQL-Summe); NaN → 0; `peak_load` = Summe der Sektoren.
- **get_egon_ways** [G] ★ – alle Ways schneiden das Abfragepolygon; Geometrie 3035.
- **get_egon_residential/cts/industrial_buildings** [V][G] ★★★ – Gebäude ∈ Load Area; keine Duplikate je building_id und Sektor; Kapazität ≥ 0; Haushalte ≥ 0, Haushalte > 0 ⇔ Wohnlast > 0.
- **get_egon_buildings** [V] ★★★ – Outer-Merge verliert keine building_id (Mengengleichheit mit Vereinigung der drei Abfragen); `capacity` = Summe der Sektoren; Koordinaten-Konsistenz zwischen Sektoren (sonst Fehler im Log); Footprint/Geometrie nicht NaN.
- **func_within / func_intersect** [S] ★.
- **get_res_generators / get_conv_generators** [V][G] ★★★ – jeder Erzeuger ∈ MVGD (G2); kW-Umrechnung; Sortierung deterministisch (zweimal abfragen → gleiche Reihenfolge); Rooftop-PV am Gebäudepunkt; kein Erzeuger doppelt (gens_id eindeutig); `subst_id` überschrieben → prüfen, dass das bewusst ist (Erzeuger mit fremder bus_id im Polygon werden übernommen) und Erzeuger an der Polygongrenze nicht in zwei Bezirken landen (Summe über alle Bezirke = DB-Summe, db, slow).
- **filter_generators_by_capacity** [U][H] ★★ – Profil `none` → nichts gefiltert; sonst exakt die > Grenze entfernt; Warnung nennt Summe.
- **component_enabled** [U] ★ – Schalter aus / Tabelle fehlt → False.
- **voltage_level_from_power** [U][H] ★★ – monoton (mehr Leistung → gleiche oder niedrigere Ebenennummer); Grenzen inklusive (`<=`) exakt auf 100/200/5500/20000 kW.
- **get_building_components** [V] ★★ – Wärmepumpen und Heimspeicher je building_id summiert; Heim-Ladepunkte nur Ebene 7 und nur Use Cases home_*; zurückgegebene `charging_location_ids` = die zugeordneten.
- **assign_to_nearest_buildings** [U][G][H] ★★ – Ergebnis = Brute-Force-Nächster innerhalb max_distance; Gleichstand → kleinste building_id; jede Location höchstens einmal.
- **add_building_components** [U] ★★ – capacity += Faktor·p_set; ohne Komponenten unverändert.
- **get_point_components** [V][G] ★★ – Punkt ∈ MVGD; Ladepunkt-Sektor ∈ eDisGo-Sektoren; Bahn ohne Zeitreihe → Warnung, nicht still; p_nom = max(Zeitreihe)·1e3; BESS-Geometrie korrekt von 4326 nach 3035; ausgeschlossene Ladeorte fehlen.

### 4.2 `core/__init__.py` – `NetworkDing0` (Pipeline)

- **run_ding0** – siehe E2E (Abschnitt 5). Zusätzlich: Rückgabe `message` enthält alle Abbrüche (V6).
- **import_mv_grid_districts** [G][V] ★★ – Typprüfung der IDs; Station wird von 4326 nach 3035 transformiert (G1); jede angefragte ID erzeugt genau einen Bezirk, unbekannte ID → Fehler statt stillem Nichts.
- **build_mv_grid_district** [U] ★ – Objektverweise konsistent (Station.grid, Grid.grid_district).
- **import_lv_load_areas_and_build_new_lv_districts** ★★★ – Herzstück für V1/V2/V3/V6:
  - Partition der Gebäude nach Schwellen (V1), auch exakt auf den Grenzen.
  - Summe `lvgd.peak_load` + MV-Lasten + verworfene = Summe Gebäude (V2).
  - Clustering scheitert → *nur diese* LA übersprungen, Meldung in `message` (heute bricht `return` die restlichen LAs ab, siehe Bug B8).
  - Cluster mit Last 0 → kein LVGD, aber auch kein verlorenes Gebäude.
  - synthetischer Graph (keine Ways) → Knoten im Lastschwerpunkt der Gebäude (G).
  - LVGD-Polygon enthält Station und Gebäude (G2); LA-Zentrum ∈ LA.
- **build_lv_grid_district** (Legacy) [S] ★.
- **import_lv_load_areas / import_lv_grid_districts / import_lv_stations** (Legacy, oedb) – nicht mehr im Pfad; nur Smoke oder entfernen.
- **import_generators** ★★★ – V4 für jeden Bezirk (heute nur erster Bezirk, Bug B1); v_level ∉ {4..7} → Fehler (Bug B2); Ebene 4/5 → `mv_grid.generators()`, 6/7 → `lv_generators_to_connect`; Typen solar/wind → `GeneratorFluctuatingDing0` mit weather_cell_id.
- **import_components** [V] ★★ – PointLoad Ebene 5 → Routingknoten; Speicher nie in Peak-Load; Peak-Load-Zuwachs von `mvgd.peak_load` = Σ Punktlasten.
- **assign_lv_point_loads** [G][V][H] ★★ – Ebene 7: nächstes Gebäude innerhalb `lv_point_load_max_distance` und unter MV-Schwelle, sonst Ebene 6; Ebene 6: nächste Station innerhalb Distanz, sonst Ebene 5 und MV-Routing. Summenerhalt: Zuwachs von lvgd/la-Peak-Load = Σ zugeordnete Last; keine Last zweimal gezählt (Gegenprobe mit Brute-Force-Distanzen).
- **import_config / import_pf_config / import_static_data** [P] ★★ – P16/P17 auf Katalogebene: Trafo r_pu/x_pu aus P_k/u_k (Datenblatt 100 kVA: P_k 1750 W → r_pu 0,0175); Katalog enthält nur im Profil freigegebene Typen; U_n MV in kV, LV in V (Einheitenfalle im Export).
- **import_orm** [S] ★ (db).
- **validate_grid_districts** [V] ★ – leere Bezirke werden gemeldet.
- **mv_routing** [V] ★★ – nach dem Schritt kein Routingknoten (LV-Station, MV-Last, PointLoad) isoliert (heute nur Log, Test soll hart prüfen).
- **build_lv_grids** [P][G] ★★★ – nach dem Schritt: jedes LV-Netz ein Baum (P13), alle Gebäude der LVGD als Last vorhanden (V10), Trafo vorhanden.
- **connect_generators** [V][G] ★★★ – nach dem Schritt keine isolierten Erzeuger/Komponenten (V7), G6.
- **mv_parametrize_grid** – siehe `MVGridDing0.parametrize_grid`.
- **set_circuit_breakers / control_circuit_breakers** [P] ★★ – P13/P14; `mode` ungültig → ValueError.
- **run_powerflow** [P] ★★★ – P1–P6 auf dem Ergebnis.
- **reinforce_grid** [P] ★★★ – P7, P9 nach dem Schritt; Verstärkung nur monoton (Querschnitt/Parallelzahl nie kleiner).
- **to_dataframe / to_csv / to_pypsa / to_netcdf** [G][V] ★★★ – G1 (CRS je Ausgabe), V9, V10; Roundtrip CSV → `components_to_pypsa_network` → PF konvergiert und erfüllt P2/P3.
- **export_mv_grid / export_mv_grid_new / to_dataframe_old** – Altpfade; Smoke oder entfernen.
- **get_mvgd_lvla_lvgd_obj_from_id / list_* / metadata / __repr__** [U] ★ – Abbildungen vollständig und eindeutig.
- **plot_mv_grids / plot_lv_grids** [S] ★.

### 4.3 `core/network/__init__.py`, `grids.py`, `stations.py`, `loads.py`, `components.py`, `cable_distributors.py`

- **GridDing0.add_generator / add_component / graph_add_node** [U] ★ – keine Duplikate; Typprüfung.
- **graph_edges / graph_branches_from_node / graph_nodes_from_branch** [U] ★ – Konsistenz mit nx-Graph (jede Kante hat `branch`).
- **find_path / find_and_union_paths / graph_path_length** [U][P] ★★ – Pfadlänge = Σ `branch.length`; Union ⊇ jeder Einzelpfad.
- **graph_isolated_nodes** [U] ★ – = nx-Grad 0.
- **control_generators** [S] ★.
- **StationDing0.peak_load / add_transformer** [U] ★.
- **TransformerDing0.z** [P][U] ★★ – |z| konsistent mit u_k.
- **BranchDing0** [U] ★ – `length` > 0; `type` gesetzt außer bei Helfern.
- **CircuitBreakerDing0.open/close** [P][U] ★★ – open entfernt genau eine Kante, close stellt sie mit identischem `branch` wieder her; zweimal open/close idempotent.
- **LoadDing0 / LVLoadDing0 / MVLoadDing0 / PointLoadDing0 / StorageUnitDing0** [U] ★★ – `peak_load` = Σ Sektoren (bzw. p_nom·`point_load_dimensioning_factor`); `pypsa_bus_id` eindeutig.
- **MVGridDing0.add_station / add_load / add_cable_distributor / remove_cable_distributor** [U] ★.
- **rings_count / rings_nodes / rings_full_data / get_ring_from_node / graph_nodes_from_subtree** [P][U] ★★ – auf `tiny_grids`-Ring: Ringknoten vollständig, Satelliten optional, Teilbaum hinter Trennstelle korrekt (bestehende Tests erweitern).
- **routing / urban_routing / connect_generators / connect_lv_generators / connect_components** – siehe 4.5/4.6.
- **parametrize_grid** ★★★
  - **set_voltage_level** [P][G][U] – Modus `distance`: 20 kV ⇔ max. Distanz Station→LA-Zentrum ≥ `voltage_per_km_threshold` (Grenzfall exakt). Modus `load_density`: Fläche in km² korrekt (heute Doppeltransformation 4326→3035 auf 3035-Daten, Bug B13).
  - **set_default_branch_type** [P][U][H] – gewählter Typ ist der *kleinste* mit Halbringzahl ≤ `mv_half_ring_count_max`; Strom I = S/(√3·U) in A korrekt; nur `reinforce_only == 0`; Siedlungskabel I_max_th > Freileitung.
  - **set_nodes_aggregation_flag** [P][U] – LA aggregiert ⇔ I_LA > I_max_th·lf; `peak_load_aggregated` = Σ aggregierte LA.
  - **MVStationDing0.select_transformers** [P] – P12 (n-1-Reservetrafo vorhanden, gleich groß wie der größte); Lastfall vs. Einspeisefall korrekt gewählt.
- **export_to_pypsa / run_powerflow / import_powerflow_results** [P] ★★★ – P1–P6; jede MV-Kante hat nach PF `s_res` mit zwei Werten, jeder MV-Knoten `voltage_res`.
- **set_circuit_breakers** [P] ★★ – siehe 4.6 `mv_grid/tools.set_circuit_breakers`.
- **LVGridDing0.add_station / add_load / add_cable_dist / get_cable_dist / loads_sector** [U] ★.
- **LVGridDing0.build_grid** [P][G][V] ★★★ – siehe 4.4.
- **MVStationDing0.peak_generation(mode)** [U] ★★ – MV = Σ MV-Erzeuger; MVLV = + Σ LV-Erzeuger **nach** Anschluss (Reihenfolgeproblem B6).
- **set_operation_voltage_level** [U] ★ – = Faktor·v_level.
- **pypsa_bus_id / pypsa_bus0_id** [U] ★ – eindeutig, MV- und LV-Seite verschieden.

### 4.4 LV-Netzbau (`grid/lv_grid/*`)

`build_grid_on_osm_ways.py`
- **build_branches_on_osm_ways** [P][G][V] ★★★ – Ergebnis ist Baum mit Station als Wurzel (P13); jede LVGD-Last unter `lv_threshold_capacity` hängt an einem Straßenknoten, jede zwischen lv- und mv_lv-Schwelle an eigenem Abgang von der Station (V1); jeder Abgang hat einen Kabeltyp, dessen I_max·lf ≥ Summenstrom des Abgangs (P10); berechneter Spannungsfall ≤ `lv_max_v_level_lc_diff_normal` gegen unabhängige LV-PF (P8, Bug B4/B5 machen das heute falsch); Kabelgeometrien liegen auf Straßen (G3/G4); `feederID` je Abgang zusammenhängend.
- **partition_network / convert_graph_c** [U][H] ★★ – Partition ist vollständig (jeder Knoten genau einmal), Teile zusammenhängend (`contiguous`), Anzahl = n_parts.
- **get_routed_graph / get_shortest_path_tree** [U][H] ★★ – Ergebnis ist Baum; jeder Gebäudeknoten erreichbar; Weg zur Station ist kürzester Weg im Ursprungsgraph.
- **relocate_buildings_with_station_as_nn** [G][U] ★ – Gebäude mit Station als nächstem Knoten werden zum nächsten Nachbarknoten verschoben; `nn_dist` stimmt mit Euklid überein.
- **transform_feeder_to_aggregated_path** [P][U][H] ★★ – Σ Last auf dem Pfad = Σ Last des Abgangs (Lasterhalt); Pfad = längster Weg.
- **allocate_street_load_nodes** [V][U] ★★ – Σ Knotenlast = Σ Gebäudelast (Einheit W, `ceil` → Abweichung < 1 W je Knoten).
- **reinforce_cable_type / cable_type_by_load** [P][U][H] ★★★ – gewähltes Kabel = kleinstes mit I_max·lf > I; monoton in der Last; kein passendes Kabel → `reinforcable=False` und Warnung.
- **check_voltage_drop_in_feeder** [P][U] ★★★ – analytischer Fall (n gleich verteilte Häuser auf einem Strang): ΔU = Σ P_k·R_0k + Q_k·X_0k gegen Formel und gegen PyPSA (≤ 1 % Abweichung). Deckt Bug B4 (L statt ω·L) auf.
- **cable_type_by_voltage** [P][U] ★★ – Ergebnis erfüllt Spannungsfallkriterium oder ist das größte Kabel.
- **get_n_feeder_mandatory** [P][U][H] ★★ – n = ceil(I/I_max(Standardkabel)).

`build_grid.py`
- **select_transformers** [P][U][H] ★★★ – P11: Σ S_nom·lf ≥ S_max; kleinste ausreichende Kombination; monoton in S_max; `case` ungültig → ValueError.
- **transformer** [P] ★★ – Trafos an Station, v_level 0,4 kV.
- **select_grid_model_ria / grid_model_params_ria / build_lv_graph_ria / build_ret_ind_agr_branches / select_grid_model_residential / build_lv_graph_residential / build_residential_branches** – Legacy-Modellnetze (nicht im Pfad, siehe `KNOWN_STALE`); entfernen oder Smoke.

`clustering.py`
- **get_cluster_numbers** [U][H] ★★ – n = ceil(Σ LV-Last / avg_trafo_size)·Faktor, begrenzt auf [1, |Graph|].
- **apply_AgglomerativeClustering** [U] ★ – Labels 0..k−1, Cluster zusammenhängend im Graph (Connectivity).
- **distance_restricted_clustering** [G][U] ★★ – Ergebnis erfüllt `ons_dist_threshold` oder Meldung in `message`; Clusterzahl steigt höchstens um `cluster_increment_counter_threshold`.

`routing.py`
- **assign_nearest_nodes_to_buildings** [G][U][H] ★★ – nn = Brute-Force-Nächster; `nn_dist` = Euklid.
- **identify_street_loads** [V][U] ★★ – Σ Straßenlast = Σ Gebäude < mv_lv-Schwelle; Grenzwert identisch zur Klassifikation (Bug B10: `>` vs. `>=`).
- **get_cluster_graph_and_nodes** [U] ★.
- **loads_in_ons_dist_threshold / get_mvlv_subst_loc_list** [G][U] ★★ – Station = Knoten mit minimaler lastgewichteter Wegsumme (Brute-Force-Gegenprobe); Distanzkriterium.
- **connect_mv_loads_to_graph** [G][U] ★ – Kante Gebäude→nn mit Länge = Euklid.
- **get_lvgd_id** [U][H] ★ – eindeutig für (la_id, cluster), Länge 10 Stellen; Kollisionen bei langen la_ids prüfen.

`graph_processing.py`
- **build_graph_from_ways / update_ways_geo_to_shape** [G][U] ★★ – Kantenlänge = Geometrielänge; Knotenkoordinaten = Way-Stützpunkte.
- **create_simple_synthetic_graph** [G][U] ★ – Knoten ∈ Load Area.
- **create_buffer_polygons / graph_nodes_outside_buffer_polys / truncate_graph_nodes** [G][U] ★ – Puffer wachsend; Knoten korrekt klassiert.
- **compose_graph / nodes_connected_component / connect_graph_components / get_fully_conn_graph / split_conn_graph / get_outer_conn_graph / consolidate_edges** [G][U][H] ★★ – Ergebnis zusammenhängend; alle inneren Knoten bleiben erhalten; synthetische Verbindungskanten haben Geometrie und Länge = Luftlinie.
- **add_edge_geometry_entry / line_shp_from_node_pair** [G][U] ★ – jede Kante hat Geometrie mit Endpunkten = Knoten.
- **simplify_graph_adv (+ _is_endpoint, _build_path, _get_paths_to_simplify)** [G][U][H] ★★ – Lastknoten bleiben erhalten; Gesamtlänge zwischen behaltenen Knoten unverändert; Geometrie der neuen Kante = Verkettung.
- **handle_detour_edges** [G][U] ★ – Umwege > Faktor·Luftlinie ersetzt/entfernt; Zusammenhang bleibt.
- **remove_unloaded_deadends / remove_unloaded_loops / remove_parallels(_and_loops)** [U][H] ★★ – kein Lastknoten entfernt; Zusammenhang bleibt.
- **subdivide_graph_edges** [G][U][H] ★★ – Segmente ≤ `dist_edge_segments`; Σ Segmentlängen = Originallänge.

`geo.py` (lv)
- **get_load_center_node / get_load_center_coords** [G][U] ★★ – Lastschwerpunkt = Σ P·x / Σ P; liegt in konvexer Hülle; LA-Geometrie wird nur erweitert, wenn Zentrum außerhalb.
- **get_points_in_load_area / get_bounding_box_from_points / get_convex_hull_from_points / get_Point_from_xy / convertCoords** [G][U] ★ – Hülle enthält alle Punkte.

`lv_connect.py`
- **lv_connect_generators / lv_connect_components / connect_lv_nodes_by_voltage_level** [G][P][V] ★★★ – G6 (eigener Bezirk, Bug B11), Ebene 6: Kabel aus `grid.tools.cable_type` trägt die Leistung (sonst Warnung, Bug B14); Ebene 7: Gebäude mit gleicher building_id; nie isoliert; v_level ungültig → ValueError.

`parameterization.py`
- **get_peak_load_diversity** [P][U][H] ★★ – Gleichzeitigkeit ≤ Summe; monoton in Haushalten; 0 Haushalte bei Wohnlast > 0 → ValueError.
- **get_peak_load_for_residential** [P][U] ★★ – VDEW-Formel: n=1 → 30 kW, n→∞ → 1,8 kW; streng fallend je Haushalt.
- **get_peak_load / parameterize_by_load_profiles(_IT_DEPRECATED)** – Legacy-OSM-Pfad; Smoke oder entfernen.

`check.py`, `db_conn_load_osm_data.py` – Legacy; Smoke oder entfernen.

### 4.5 MV-Routing (`grid/mv_grid/*`)

`mv_routing.py`
- **ding0_graph_to_routing_specs** [V][U] ★★ – alle Nicht-Satelliten-LA-Zentren und PointLoads als Nachfrage; Nachfrage in kVA = P/cos φ; Distanzmatrix symmetrisch, Diagonale 0, km.
- **routing_solution_to_ding0_graph** [P][G][U] ★★★ – jede Route ist Ring über Depot; jede Route hat genau eine Trennstelle (außer aggregiert); Länge = Luftlinie·Umwegfaktor (G4); **Ausnahmen werden heute mit bare `except` verschluckt** → Test, dass bei Fehler ein Fehler ankommt.
- **solve** [P][H] ★★ – Lösung deckt alle Knoten genau einmal ab; erfüllt `Route.tech_constraints_satisfied` für jede Route; deterministisch.

`models/models.py`
- **Route.tech_constraints_satisfied** [P][U] ★★★ – analytischer Ring: Strom- und Spannungsfallprüfung gegen PyPSA-Ergebnis desselben Rings (Toleranz; Formel nutzt S statt P/Q → konservativ um ≈ 1/cos φ, dokumentieren); n-1-Fall (Ring von einer Seite gespeist) gegen PyPSA.
- **Route.calc_circuit_breaker_position** [P][U][H] ★★ – minimiert |Σ links − Σ rechts| (Brute-Force-Gegenprobe).
- **Route.length / length_from_nodelist / demand / can_allocate / allocate / deallocate / insert / clone** [U][H] ★★ – Länge = Σ Matrixabstände inkl. Depot; allocate/deallocate invers; clone tief.
- **Graph / Node** [U] ★ – Distanz symmetrisch.

`solvers/savings.py`, `solvers/local_search.py`, `solvers/base.py`
- **ClarkeWrightSolver.compute_savings_list / solve** [U][H] ★★ – Savings s_ij = d_0i + d_0j − d_ij; Lösung vollständig und zulässig.
- **LocalSearchSolver.operator_oropt / relocate / exchange / cross / solve** [U][H] ★★ – jede Operation verschlechtert die Gesamtlänge nie und hält Zulässigkeit; Knotenmenge invariant. Benchmark mit Augerat-Instanzen (`mv_grid/tests/testcases`, `.opt` liefert Optimum): Lösung ≤ Optimum·(1+x %).
- **BaseSolution / SavingsSolution / LocalSearchSolution** (`clone, is_complete, length, routes, process, can_process`) [U] ★.
- **draw_network, benchmark_operator_order** [S] ★.

`mv_connect.py`
- **find_nearest_conn_objects** [G][U][H] ★★ – nächster Punkt auf Leitung = Projektion (shapely); Gewichtung Station vs. Leitung nach `conn_dist_weight`.
- **get_lv_load_area_group_from_node_pair** [U] ★ – Tabelle aller Typkombinationen (inkl. PointLoad).
- **find_connection_point / connect_node / disconnect_node** [P][G][U] ★★★ – nach Anschluss bleibt Graph ein Graph mit gleicher Ringzahl; neue Kabelverteiler liegen auf der alten Leitung; alte Leitung ersetzt durch zwei, Σ Länge = alte Länge (G4); Satellitenstrang ≤ `load_area_sat_string_load_threshold` und `…_length_threshold`; disconnect stellt den Ausgangszustand wieder her (Roundtrip).
- **parametrize_lines** [U] ★ – Typ/Art gesetzt.
- **mv_connect_satellites** [V][G] ★★ – nach `mode='isolated'` kein Satellit isoliert.
- **mv_connect_stations** [V][G] ★★ – jede LV-Station einer nicht aggregierten LA mit dem MV-Netz verbunden.
- **mv_connect_generators / mv_connect_components / connect_mv_node_by_voltage_level** [V][G][P] ★★★ – Ebene 4 eigene Leitung zur Station, Länge = Luftlinie·Umweg; Ebene 5 am nächsten Objekt; **nicht verbundener Erzeuger → Fehler/Log statt still isoliert** (Bug B9); Puffer wächst bis Fund.
- **find_nearest_conn_objects_settle / connect_node_settle** [G][U] ★★ – Anschluss entlang Straßengraph; Leitungsgeometrie folgt Straße; Endknoten ohne Straßenknoten werden angebunden (bestehender Test erweitern).

`urban_mv_routing.py`, `urban_mv_connect.py`, `tools.py` (Stichleitungen)
- **osm_graph_to_routing_specs_urban / solve / routing_solution_to_ding0_graph (urban)** [P][G] ★★ – Ringe auf Straßengraph; alle Stationen der aggregierten LA verbunden; Leitungsgeometrie = Straßengeometrie.
- **mv_urban_connect** [V][G] ★★ – jede Station eines Stichs verbunden; Stich hängt an genau einem Ringknoten (bestehender Test).
- **get_core_graph / get_stub_graph / split_graph_by_core / update_graphs / create_stub_dict / check_stub_criterion / update_stub_dict** [U][H] ★★ – Kern ∪ Stiche = Straßengraph (disjunkt in Kanten); Stichlast ≤ Kriterium, sonst in den Kern gezogen.
- **get_edge_tuples_from_path / get_shortest_path_shp_single_target / _multi_target / cut_line_by_distance** [G][U][H] ★★ – Geometrie verbindet Start und Ziel; Länge = Σ Kantenlängen; `cut_line_by_distance(·, 0.5)` liegt auf der Linie.
- **reduce_graph_for_dist_matrix_calc / calc_street_dist_matrix** (auch `tools_for_ways.py`) [U][H] ★ – Distanzmatrix = Dijkstra-Gegenprobe; symmetrisch.
- **conn_ding0_obj_to_osm_graph** [G][U] ★ – Verbindung zum euklidisch nächsten Knoten.
- **relocate_cable_dists_settle / relabel_graph_nodes / update_branch_shps_settle** [G][U] ★★ – Kabelverteiler auf Straßenknoten; Leitungsgeometrie danach = Straßengeometrie, Länge aktualisiert.
- **get_mvgd_ids_for_city** [S] ★ (extern).
- **set_circuit_breakers** [P][U] ★★★ – je Ring genau eine Trennstelle, an LV-Station sofern vorhanden; Position minimiert Lastdifferenz der Halbringe (Brute-Force); Teilbaum-Lasten einschließlich MV-Lasten (Bug B7); `geo_data` auf Leitung (G7).

`analysis.py` – Kennzahlen (Leitungslängen, Strukturdaten); [U] Länge-Summen gegen Export; ★.

`util/data_input.py`, `util/util.py`, `tests/run_test_case.py` – TSPLIB-Parser [U]: Augerat-Dateien lesen, Knotenzahl/Kapazität aus Header; ★.

### 4.6 Leistungsfluss und Export (`tools/pypsa_io.py`, `core/powerflow`)

- **q_sign** [U] ★ – besteht; Tabelle vollständig.
- **PFConfigDing0** [U] ★.
- **initialize_component_dataframes** [U] ★ – Spalten wie eDisGo erwartet.
- **fill_mvgd_component_dataframes / fill_component_dataframes** [V][P] ★★★ – V10; Trennstellen danach im Ausgangszustand (offen bleibt offen); HS/MS-Trafos separat; Populationssumme.
- **nodes_to_dict_of_dataframes** [V][P] ★★★ – jede Last je Sektor einmal; Helfer-Kanten werden zu einem Bus zusammengelegt (Last am Gebäudebus); isolierte Nicht-Stationen → Fehler (heute AttributeError bei Erzeugern, Bug B9); Heimspeicher max_hours = Kapazität/Leistung.
- **select_and_append_load_area_trafos** [P] ★★ – P11 für aggregierte LA.
- **append_generator_pq_set_df / append_load_pq_set_df** [P][U][H] ★★★ – P6, Vorzeichen und Faktoren; kW→MW genau einmal.
- **append_bus_v_mag_set_df** [U] ★ – keine Duplikate.
- **append_load_areas_to_df / append_load_area_to_load_df** [V][P] ★★ – Σ Last am Bus = peak_load der LA/LVGD; Erzeuger der LVGDs vollständig.
- **append_generators_df** [V][U] ★★ – p_nom = capacity·capacity_factor/1e3; source_id = gens_id.
- **building_component_loads / append_component_load_df / append_storage_units_df** [V][U] ★★ – eDisGo-Typen/Sektoren.
- **append_buses_df** [G][P][U] ★★ – v_nom MV in kV, LV in kV (V/1e3); Koordinaten aus geo_data.
- **append_transformers_df** [P][U] ★★ – bus0 MV-Seite; `type_info` mit richtigen Spannungen; s_nom MVA.
- **edges_to_dict_of_dataframes / append_lines_df** [P][U] ★★★ – P16 (Bug B3: num_parallel); Länge km; MV- vs. LV-Einheitenumrechnung; LV-Station auf MV-Seite → pypsa_bus0_id.
- **_check_branch_for_in_building_buses** [U] ★ – Helferleitungen entfallen.
- **circuit_breakers_to_df** [P][U] ★★ – offene Trennstelle: Leitung endet an virtuellem Bus `bus_open`, dieser hat sonst keine Verbindung.
- **run_powerflow_onthefly** [P] ★★★ – P1–P6; heute fehlen Konvergenzprüfung und `_check_integrity_of_pypsa` ist leer.
- **data_integrity / _check_integrity_of_pypsa** [U] ★★ – sollen bei Fehlern scheitern (heute nur Log bzw. leer).
- **process_pf_results / assign_bus_results / assign_line_results** [P][U] ★★ – s_res = max(\|S0\|, \|S1\|) je Snapshot; jeder Knoten bekommt Ergebnis.
- **init_pypsa_network / transform_timeseries4pypsa / create_powerflow_problem** [U] ★ – Snapshots 2, Reihenfolge lc/fc.
- **transform_geodata_to_wgs84** [G][U][H] ★★ – Hin- und Rücktransformation identisch (< 1 mm); Bus und Leitungsgeometrie gleich transformiert.
- **components_to_pypsa_network / _json_records** [V][P] ★★ – Roundtrip: PF auf dem gebauten Netz erfüllt P1–P3; Metadaten vollständig.
- **export_to_dir** [S] ★.

### 4.7 Grenzwertprüfung und Verstärkung (`flexopt/*`)

- **check_load** [P][U] ★★★ – Leitung über Grenze erkannt, darunter nicht; relative Überlastung korrekt; Leitung ohne `s_res` → Fehler statt `except: pass`; num_parallel berücksichtigt (B3).
- **check_voltage** [P][U] ★★★ – Grenzen lc/fc getrennt; Sortierung nach Abweichung; Knoten ohne `voltage_res` → Fehler statt `except: pass`.
- **get_critical_line_loading / peak_load_generation_at_node / get_critical_voltage_at_nodes / get_voltage_at_bus_bar / get_delta_voltage_preceding_line / get_voltage_delta_branch / get_cumulated_conn_gen_load / get_mv_impedance_at_voltage_level** [P][U] ★★ – LV-Pfad (heute nicht aufgerufen): gegen PyPSA-LV-PF auf `tiny_grids`; Impedanz-Umrechnung auf 0,4 kV mit (U_LV/U_MV)².
- **voltage_delta_vde** [P][U] ★★ – Vorzeichenkonvention gegen VDE-AR-N 4105 und PyPSA prüfen (induktive Last: ΔU = S(R cos φ + X sin φ)/U²; heute steht `r·cos φ − x·sin φ` mit `x_sign = +1` für induktive Last – klären).
- **reinforce_grid (MV)** [P] ★★★ – nach Verstärkung P7/P9 oder dokumentierter Abbruch; Schleife terminiert; nur gefährdete Pfade verstärkt.
- **reinforce_branches_current / reinforce_branches_voltage** [P][U] ★★ – neuer Typ ≥ alter Typ; bei fehlendem größeren Typ: Parallelleitung (num_parallel+1) und Export berücksichtigt das.
- **extend_substation / extend_substation_voltage / extend_trafo_power / new_substation / reinforce_lv_branches_overloading** [P][U] ★ – Trafoleistung danach ≥ Bedarf.
- **reinforce_measures_dena.*** – Stubs; Smoke.

### 4.8 Konfiguration, Geo-Hilfen, Sonstiges

- **tools/planning_profiles.*** ★★ – bestehende Tests gut; ergänzen: jede Grenze eines Profils wird im Netz tatsächlich eingehalten (E2E mit Profil `snb`: Halbring ≤ 9 km, max. 15 LA je Ring, Trafo 31,5 MVA statt 25).
- **tools/geo.calc_geo_dist / calc_geo_dist_matrix / calc_edge_geometry / calc_geo_centre_point** [G][U][H] ★★★ – Distanz = Luftlinie·Umwegfaktor; Matrix in km; Mittelpunkt auf Linie; 4326-Zweig = geodätisch gegen 3035-Zweig (< 0,5 % Abweichung in Deutschland); Distanz 0 → 1 m mit Warnung.
- **tools/geo.calc_geo_branches_in_polygon / calc_geo_branches_in_buffer** [G][U] ★ – Ergebnis = Brute-Force-Schnitt; Puffer wächst.
- **grid/tools.cable_type** [P][U][H] ★★ – kleinstes Kabel mit I_max > I; kein passendes → Warnung (heute still).
- **tools/tools.*** (`append_to_dataframe`, `merge_two_dicts(_of_dataframes)`, `get_dest_point`, `get_cart_dest_point`, `create_poly_from_source`) [U] ★ – Merge verliert keine Zeilen; Zielpunkt hat geforderte Distanz.
- **tools/config.load_config / get / set** [U] ★.
- **tools/database.*** – bestehende Tests (`test_egon_data_config.py`).
- **tools/results.*** (Statistiken, Pickle, Export nach oedb) [V] ★ – Pickle-Roundtrip identisch; `calculate_mvgd_stats` Längen-/Leistungssummen = Export.
- **tools/validation.validate_generation / validate_load_areas / validate_lv_districts / compare_grid_impedances** [V] ★★ – auf egon-data umstellen; sie sind inhaltlich V2/V4 und können die neue Bibliothek nutzen.
- **validation/statistics, compare** [S] ★.
- **tools/plots*, animation, logger, debug, write_openego_header, config_db_interfaces, config_lv_grids_osm.get_peak_loads/…** [S] ★ – nur Smoke bzw. Altlast.

---

## 5. End-to-End-Tests

### E2E-1: Synthetische Bezirke, alle Invarianten (CI, ohne DB) ★★★
`run_ding0` auf Bezirk 1, Bezirk 2 und beiden zusammen; danach P1–P17, G1–G9,
V1–V10. Ergänzend die Zwischenstände:

| nach Schritt | Invarianten |
|---|---|
| 1 Import | V1, V2, V3, V6, G2 |
| 2 Erzeuger/Komponenten | V4, V5 (je Bezirk!) |
| 3 Parametrierung | P12 (vorläufig), Spannungsebene, Leitungstyp |
| 5 LV-Netze | P10, P11 (Last), P13 (LV), G3–G5 |
| 6 MV-Routing | P14, keine isolierten Routingknoten |
| 7 Anschluss | V7, G6 |
| 8–9 Trennstellen | P13 (MV), P14, G7 |
| 10 Leistungsfluss | P1–P6 |
| 11 Verstärkung | P7, P9, Monotonie der Typen |
| Export | G1, V9, V10, P15, P16 |

### E2E-2: Metamorphe Tests auf synthetischen Daten ★★
Ohne Referenzzahlen, durch Vergleich zweier Läufe:
- **Verschiebung**: alle Koordinaten um (Δx, Δy) verschieben → gleiche Topologie, gleiche Längen, verschobene Busse.
- **Reihenfolge**: DB-Zeilen permutieren → identisches Netz (Determinismus, Seed).
- **Lastskalierung** ×k → Trafoleistungen und Leitungsquerschnitte monoton nicht kleiner; Spannungsabweichung im Lastfall monoton größer (vor Verstärkung).
- **Erzeuger hinzufügen** → Spannung an seinem Bus im Einspeisefall steigt; kein anderes Objekt verschwindet.
- **Ohne Erzeuger** → Einspeisefall t1: alle Spannungen ≈ v_Station.
- **Profilwechsel** default → snb → Grenzen von snb eingehalten.
- **Bezirk 1 allein vs. Bezirke 1+2** → Netz von Bezirk 1 identisch (heute verletzt durch B1 und B11).

### E2E-3: Unabhängiger Gesamtleistungsfluss MV+LV ★★★
Export mit `only_export_mv=False` → `components_to_pypsa_network` → PF über
alle Ebenen. Prüft P1–P5 auf dem vollständigen Netz, P8 (LV-Spannungsband) und
P10 mit echten Strömen. Ergebnis zuerst als `diagnostic`: Verteilung der
LV-Spannungsabweichungen und Kabelauslastungen berichten, dann Grenzen scharf
schalten, sobald LV-Verstärkung aktiv ist.

### E2E-4: egon-data (Marker `db`) ★★★
Wie `test_egon_data_properties.py`: Hypothesis zieht Bezirke, Lauf, dann die
gesamte Invarianten-Bibliothek. Vollständigkeit gegen die DB (SQL-Summen je
Bezirk: Gebäudelast, Haushalte, Erzeuger je Typ, Ladepunkte, Wärmepumpen,
Speicher). Für eine neue egon-data-Version: `DING0_DB_EXAMPLES=50`.
Zusätzlich ein Mehr-Bezirks-Lauf mit zwei benachbarten Bezirken (B1, B11).

### E2E-5: eDisGo-Roundtrip (optional, `slow`) ★★
`to_csv` → `edisgo.EDisGo(ding0_grid=…)` → eDisGo-Worst-Case-Leistungsfluss
und `check_tech_constraints`: keine Überlastung und kein Spannungsbandverstoß
im MV-Netz; LV diagnostisch. Prüft, dass eDisGo das Netz so versteht wie ding0
(Einheiten, CRS, Schaltzustand der Trennstellen).

### E2E-6: Analytische Kleinstnetze (`tiny_grids`) ★★★
Handgebaute Netze durch den ding0-Export und PF:
- **A1** Station – 10 km NA2XS2Y 3x1x185 – 2 MW, cos φ 0,97, 20 kV: ΔU gegen (P·R + Q·X)/U (≤ 0,5 %).
- **A2** wie A1 mit `num_parallel = 2`: ΔU halbiert, s_nom verdoppelt (B3).
- **A3** LV-Strang mit 5 Häusern im Abstand 30 m: `check_voltage_drop_in_feeder` vs. PyPSA (B4).
- **A4** Ring mit Trennstelle: offen → zwei Radialstränge; geschlossen → Masche; KCL in beiden Zuständen; Strom in jedem Halbring = Σ Last des Halbrings.
- **A5** Trafo 100 kVA: r_pu 0,0175, x_pu = √(0,04² − 0,0175²); Spannungsfall über Trafo gegen Handrechnung.
- **A6** n-1: Ring einseitig gespeist; `Route.tech_constraints_satisfied` vs. PyPSA.

---

## 6. Verdachtsfälle beim Lesen (zuerst per Test bestätigen)

Fundstellen ohne Gewähr. Jeder Punkt bekommt einen Test (xfail strict), bevor
er behoben wird.

| # | Verdacht | Stelle | Folge | Test |
|---|---|---|---|---|
| B1 | Erzeuger werden nur für den **ersten** Bezirk abgefragt (`list(mv_grid_districts_dict.values())[0]`) | `core/__init__.py` `import_generators` | in Mehr-Bezirks-Läufen fehlen Erzeuger aller weiteren Bezirke (im Golden-File bereits als Upstream-Bug vermerkt) | V4 je Bezirk |
| B2 | `ValueError("False voltage level")` wird erzeugt, aber nicht geworfen | ebd. | Erzeuger mit v_level ∉ {4..7} verschwinden still | V4 mit v_level 3 |
| B3 | r, x im Export nicht durch `num_parallel` geteilt, s_nom nicht multipliziert; `check_load` ignoriert `num_parallel` ebenfalls | `pypsa_io.append_lines_df`, `check_tech_constraints.check_load` | Parallelleitungen aus der Verstärkung wirken im PF wie eine; Überlastungen bleiben bestehen | P16, A2 |
| B4 | `x_per_km = cable_type['L_per_km']` – Induktivität in mH/km statt Reaktanz ω·L·1e-3 | `build_grid_on_osm_ways.check_voltage_drop_in_feeder` | LV-Spannungsfall mit falscher Reaktanz (bei 50 Hz ca. Faktor 3 zu groß) | A3 |
| B5 | Für Lasten direkt an der Station wird `load` in kW gesetzt, die Spannungsfallprüfung erwartet W | ebd., Block `lv_loads_to_station` | Spannungsfall für 100–200-kW-Abgänge um Faktor 1000 unterschätzt, Kabel zu klein | A3-Variante, P8 |
| B6 | LV-Trafos werden in Schritt 5 gewählt, die LV-Erzeuger erst in Schritt 7 angeschlossen; HS/MS-Trafos in Schritt 3 ohne LV-Erzeugung | `build_lv_grids`, `parametrize_grid` vs. `connect_generators` | Trafos ignorieren Einspeisung | P11, P12 |
| B7 | `isinstance(node, MVLoadDing0)` statt `node_subtree`; MV-Last im Ring hängt keinen Erzeugungswert an | `mv_grid/tools.set_circuit_breakers` | MV-Lasten in Stichen fehlen bei der Trennstellenwahl; Listenlänge passt im Modus `generation` nicht | Trennstellen-Brute-Force |
| B8 | Gescheitertes Clustering: `return f"…"` mitten in der Schleife, Rückgabewert wird verworfen | `import_lv_load_areas_and_build_new_lv_districts` | alle folgenden Load Areas des Bezirks fehlen still | V6 |
| B9 | Rückgabewert von `connect_mv_node_by_voltage_level` wird bei Erzeugern ignoriert | `mv_connect_generators` | nicht anschließbarer Erzeuger bleibt isoliert, Export scheitert mit AttributeError | V7 |
| B10 | Straßenlasten nutzen `capacity > mv_lv_threshold`, die MV-Klassifikation `>=` | `routing.identify_street_loads` vs. Import | Gebäude mit genau 200 kW wird doppelt gezählt | V1 Grenzwert |
| B11 | LV-Anschluss sucht Stationen/Gebäude in **allen** Bezirken des Netzes; Ebene 7 ohne Distanzgrenze, Erzeuger wird zum Gebäude verschoben | `lv_connect.connect_lv_nodes_by_voltage_level` | Erzeuger können in fremdem Bezirk landen | G6, Mehr-Bezirks-Lauf |
| B12 | Konvergenz des PF nicht geprüft; `_check_integrity_of_pypsa` leer; `check_load`/`check_voltage` mit `except: pass` | `pypsa_io`, `check_tech_constraints` | fehlende Ergebnisse gelten als „in Ordnung“ | P1, 4.7 |
| B13 | Modus `load_density` transformiert 4326→3035, obwohl `geo_data` schon 3035 ist | `MVGridDing0.set_voltage_level` | falsche Fläche (Modus nicht Default) | set_voltage_level |
| B14 | `grid.tools.cable_type` gibt still das größte Kabel zurück, wenn keines reicht | `grid/tools.py` | überlastete Anschlussleitung Ebene 6 ohne Hinweis | cable_type |
| B15 | Vorzeichen in `voltage_delta_vde` (r·cos φ − x·sin φ bei induktiver Last mit x > 0) | `check_tech_constraints` | LV-Spannungsprüfung (heute nicht aktiv) evtl. falsch | A1/A3 gegen PyPSA |
| B16 | Quickfix #402 schreibt `*_graph_debug.html` ins Arbeitsverzeichnis und entfernt LV-Netze ohne Bilanz | `run_ding0` | Lasten verschwinden, nur Log | V7 |
| B17 | Leistungsfluss-Sollwert jeder Sektor-Last eines Gebäudes ist die **gesamte** Gebäude-Spitzenlast (`append_load_pq_set_df` ohne `peak_load`) | `pypsa_io.nodes_to_dict_of_dataframes` | Gebäude mit mehreren Sektoren (und Wärmepumpe/Ladepunkt) zählen in ding0s Leistungsfluss mehrfach; im CSV-`p_set` korrekt | `ding0_set_points_match_export` |
| B18 | Spannungsfall eines NS-Abgangs wird nur entlang des **längsten** Pfads geprüft | `check_voltage_drop_in_feeder` / `transform_feeder_to_aggregated_path` | kürzerer, stark belasteter Abzweig: 12 % statt ≤ 5 % | `test_heavily_loaded_short_branch_keeps_voltage_limit`, `lv_voltage_band` |
| B19 | Betriebsmitteldaten: NAYY 4x1x50 mit R' von 70 mm²; Freileitungen 48/94/122-AL1 mit fast gleichem R' | `data/equipment-parameters_*.csv` | falsche Impedanzen | `test_conductor_resistance_is_inversely_proportional_to_cross_section` |
| B20 | Namen der Trennstellen (`circuit_breaker_1`) wiederholen sich über MV-Bezirke | `CircuitBreakerDing0.__repr__` | doppelte Indizes im Export mehrerer Bezirke | `unique_names` |
| B21 | Der Export schließt offene Trennstellen (`fill_component_dataframes` → `close()`) und öffnet sie nicht wieder; die Leistungsflüsse der Verstärkung laufen danach mit **geschlossenen Ringen** | `pypsa_io.fill_component_dataframes` | ding0 bewertet und verstärkt das vermaschte Netz; im Normalbetrieb (Ringe offen) 2,8 % statt ≤ 2 % Spannungsanhebung, 64 % statt ≤ 60 % Kabelauslastung | `export_keeps_switch_state`, `ding0_mv_results_match_normal_operation`, `mv_voltage_band` |
| B22 | Anschlusskabel der Ebene 6 (eigenes Kabel zur NS-Station) nur nach Strom bemessen, ohne Spannungsprüfung | `lv_connect.add_node_to_station` | 11,9 % Spannungsfall an einem 150-kW-Ladepunkt, 3,6 % Anhebung an PV | `lv_voltage_band` |
| B23 | `calc_geo_branches_in_buffer` erweitert den Suchradius ohne Grenze (`while not branches`) | `tools/geo.py` | MS-Netz ohne Leitungen (z. B. nach B8): Erzeugeranschluss hängt endlos | `test_connecting_to_a_grid_without_lines_ends` |
| B24 | NS-Kabel und -Trafos werden auf I = S/(√3·U_n) mit Lastfaktor 1,0 bemessen, ohne Verluste und ohne die niedrigere Spannung im Strang | `build_grid_on_osm_ways.py`, `build_grid.py` | Strom bis 108 % der Bemessung (default) | `thermal_limits_exact` |
| B25 | NS-Spannungsanhebung im Einspeisefall (3 %) nur in der abgeschalteten NS-Verstärkung | `flexopt/reinforce_measures.py:271` | 3,1–4,0 % | `lv_voltage_band_feedin_case` |
| B26 | HS/MS-Trafos mit Normal-Lastfaktor 0,6 plus Reservetrafo der größten Größe: (n-1) doppelt; `load_factor_mv_trans_lc_malfunc` nie gelesen | `stations.py:79-182` | Bezirk 2: 2×40 statt 2×25 MVA | `hvmv_transformers_minimal` |
| B27 | (n-1) der MS-Ringe nur im Routing geprüft; später angeschlossene Stationen/Satelliten/MS-Lasten nicht | `models.py:355-475` | snb Bezirk 2: 172 %, 17,9 % | `mv_rings_n_minus_1` |
| B28 | snb: Lasten bis 346 kW an der Station, aber kein NS-Kabel über NAYY 150 ausgewählt; `reinforce_cable_type` nimmt still das größte (B14) | `planning_parameters.csv`, `equipment_selection.csv` | 361 A > 275 A (131 %) | `lv_cables_carry_load_and_generation[snb]` |
| B31 | Braucht ein Teilbaum mehr Abgänge als er Knoten hat, wird alles **ein** Abgang; die Meldung sagt das Gegenteil | `build_grid_on_osm_ways.py:528-533` | snb Bezirk 2: bis 149 % | `lv_feeders_within_standard_cable[snb]`, `test_run_reports_no_invalid_districts[district_2_snb]` |
| B29 | MS-Verstärkung bricht still ab, wenn das größte ausgewählte Kabel nicht reicht; keine Parallelleitungen | `flexopt/reinforce_grid.py:89` | snb Bezirk 2: 97 % gegen 50 % | `ding0_mv_limits_after_reinforcement[snb]`, `mv_export_thermal_limits[snb]` |
| B30 | MS-Verstärkung nicht minimal: entscheidet auf geschlossenen Ringen (B21) mit überhöhten Sollwerten (B17), verstärkt alle Zweige auf den Pfaden | `flexopt/reinforce_grid.py:77`, `reinforce_measures.py:116` | Bezirk 1: 3 Kabel unnötig | `mv_reinforcement_necessary` |

Außerdem (ohne eigenen Test): Der Export verliert die `building_id` von Ladepunkten und Wärmepumpen, die einem Gebäude zugeordnet sind.

### 6.1 Stand der Tests (2026-11-09)

Umgesetzt auf Branch `test/physics-invariants`:

- `tests/invariants/` – Prüfbibliothek (`physics`, `geo`, `completeness`,
  `powerflow` mit unabhängigem Leistungsfluss und eigener Admittanzmatrix).
- `tests/test_synthetic_invariants.py` – E2E-1: 43 Invarianten × 3 Läufe
  (Bezirk 1, Bezirk 2 mit Komponenten, beide zusammen), einige direkt nach
  dem Pipeline-Schritt.
- `tests/physics/` – Spannungsfall NS analytisch und gegen PyPSA (E2E-6/A3),
  Betriebsmitteldaten, Export-Impedanzen (A2), Kabel-/Trafoauswahl
  (Hypothesis), Lastformeln.
- `tests/test_completeness.py` – gezielte Fälle für B2, B8, B10, B23.

| Befund | Status |
|---|---|
| B1, B2, B4, B5, B8, B10, B14 | bestätigt (xfail strict) |
| B3 | bestätigt, aber latent: ding0 setzt `num_parallel` nie > 1 |
| B6 | auf synthetischen Daten nicht sichtbar (LV-Erzeugung klein); `lv_transformers_carry_load_and_generation` prüft es, auf egon-data offen |
| B7, B9, B11, B12, B13, B15, B16 | noch ohne Test |
| B17–B23 | neu gefunden durch die Tests, bestätigt (xfail strict) |

Unabhängiger Leistungsfluss MV+NS (synthetisch): Kirchhoff hält
(Restfehler < 1e-6 MVA), Verluste positiv, Energiebilanz geschlossen;
MS-Spannungen ≥ 0,975 p.u. Verletzt sind das NS-Spannungsband (B18, B22) und
nach B21 das MS-Band im Einspeisefall.

Offen aus dem Plan: E2E-2 (metamorph), E2E-4 (Bibliothek auf egon-data),
E2E-5 (eDisGo), Routing/Ringe gegen PyPSA (A6), Unit-Tests der
Graphverarbeitung.

---

## 7. Umsetzungsreihenfolge (Vorschlag)

1. Invarianten-Bibliothek + `staged_synthetic_run` + Erweiterung der
   synthetischen Daten um Grenzfälle.
2. E2E-1 und E2E-6 (analytisch) – liefern sofort Belege für B1–B6, B10.
3. Unit-Tests der Physikfunktionen mit Hypothesis (Kabel-/Trafoauswahl,
   Spannungsfall, pq-Sätze, Export-Impedanzen).
4. E2E-3 (Gesamt-PF) als `diagnostic`, E2E-2 (metamorph).
5. E2E-4 auf egon-data mit der Bibliothek; `test_egon_data_properties.py`
   darauf umstellen.
6. Altpfade (Modellnetze, oedb-Import, `KNOWN_STALE`) entscheiden: löschen
   oder Smoke.
7. E2E-5 eDisGo-Roundtrip.
