# Discover dingo in depth
~~~~~~~~~~~~~~~~~~~~~~~
## Generating MV/LV Substations
Find a method description also at: "The eGo grid model: An open-source and open-data based synthetic medium-voltage grid model for distribution power supply systems" (Amme, J)

### Method
1. Placing LV/MV substations:
The substations are based on a grid of points with an interval of 180m within the load areas.
Derivation of the 180m: There are different lengths of cable found in LV-Grids: 100-1.500m (see Kerber, Scheffler, Mohrmann). As described by Scheffler, length between 200-300m is found most often.
Furthermore, we foud a difference between the cable length and the line over ground is 72% (1.39 Umwegfaktor), see master thesis Jonas Gütter. This seems plausible compared to the value for the MV grid of 77% (1.3).
The chosen value concludes in cable lengths of 250m at the shortest distance and 283m at the longest distance between the middle point of the square and its outer line.

1. Finding LV-Grid districts (LV-GD):
We define Voronoi polygons within the load areas based on a grid of points with an interval of 180m.

1. Assign consumption to the LV-GD:
This works analogously to the methods for the MV-GD, as described in "Allocation of annual electricity consumption and power generation capacities across multi voltage levels in a high spatial resolution" (Huelk)

1. Assign peak load

1. Define transformer

## Generating LV-Grids

### Introduction/Literature
Information on LV-Grids in Germany can be found on several sources:
*  - [Kerber](http://oep.iks.cs.ovgu.de/literature/entry/17/) describes 8 rural and 3 village and 8 suburban LV Grids; each with several branch lines. The exemplary grids are based on 132 real MV/LV Substations data in south Germany.
* [Scheffler] (http://oep.iks.cs.ovgu.de/literature/entry/18/) gives statistical data about technical parameters of LV grids divided on 8 types of settlement areas.
* [Mohrmann] (http://oep.iks.cs.ovgu.de/literature/entry/19/) discribes statistical data about technical parameters of LV grids based on 2700 LV-Grids.
* Demirel
* VNS
However, a method to generate a representative variation of LV-grids, that can be assigned to the modeled LV/MV substations cannot be found.
Given data on MV/LV substations: 
* land use data divided in industry, commercial, agriculture and residential
* population
* --> peak load

### Method

#### Method for residential LV-branches

1. LV-Branches

We are using the LV-Branches of Kerber from the grids. They should be assigned to the most plausible types of settlement areas.

1. Define the type of settlement area

To decide if a LV-grid district is most likely a rural, village or suburban settlement area we are using the population value combined with statistical data. Statisticly, there are 2.3 persons per appartment and 1.5 appartments per house. [see BBR Tabelle B12 http://www.ggr-planung.de/fileadmin/pdf-projekte/SiedEntw_und_InfrastrFolgekosten_Teil_2.pdf] [DEMIREL Seite 37-41, average has been coosen]. (This is not valid for urban areas.) With this we estimate the amount aus house connections (HC).
This value can also be found at the explenation of the database of the "Kerber"-grids and is assinged to the type of settlement area:
Rural: 622 HC at 43 MV/LV substations results in an average amount of 14.5 HC/substation
Village: 2807 HC at 51 MV/LV substations results in an average amount of 55 HC/substation
Suburban: 4856 HC at 38 MV/LV substations results in an average amount of 128 HC/substationTher
With the resulting trendline of this three point,  [the Polynomial degree 2 [ 16.127*(x^2)-7.847*x+6.1848 ] whereas x is the type of of settlement area], we difine the border values for the typ of settlement area at:

*Rural <31 HC/substation
*Village <87 HC/substation
*Suburban >=87 HC/substation

1. Categorising grid branches form "Kerber" model grids

Hinzu kommen auf Basis von kerber interpolierte stränge um Lücken in der Vollständigkeit zu schließen

1. Assinging grid branches to the Substations
Strangzuweisung
    Zu jeder ONS werden in Abhängigkeit von Netztyp und HA, NS-Stränge zugewiesen
   Eine Verteilung des Aufkommens der Stränge anhand von der Gesamtstranglänge geschieht mit Hilfe der Scheffler Angaben (Abbildung      Länge der Netzstrahlen für ausgewählte Siedlungstypen [44])
