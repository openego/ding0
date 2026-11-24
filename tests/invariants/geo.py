"""Geographic invariants of ding0 grids.

ding0 builds grids in EPSG:3035 (metres). Exported coordinates are checked
in that CRS unless stated otherwise.
"""

import networkx as nx
import numpy as np
from shapely.geometry import Point

from ding0.core.network import GeneratorDing0
from ding0.core.network.loads import LVLoadDing0
from ding0.tools import config as cfg_ding0

# Germany in EPSG:3035 and EPSG:4326 (generous bounding boxes)
GERMANY_3035 = (4_000_000, 2_650_000, 4_700_000, 3_600_000)
GERMANY_4326 = (5.5, 47.0, 15.5, 55.2)


def check_coordinates_in_germany(buses, bbox=GERMANY_3035):
    """All bus coordinates lie within Germany in the given CRS."""
    x0, y0, x1, y1 = bbox
    outside = buses.loc[~(buses.x.between(x0, x1) & buses.y.between(y0, y1))]
    return [f"bus {name} at ({row.x}, {row.y}) outside {bbox}" for name, row in outside.iterrows()]


def check_line_endpoints(components, tolerance=1.0):
    """The geometry of every line starts and ends at its buses (in m)."""
    buses = components["Bus"]
    violations = []
    for name, line in components["Line"].iterrows():
        if line.geometry is None:
            violations.append(f"line {name}: no geometry")
            continue
        coords = list(line.geometry.coords)
        ends = [Point(coords[0]), Point(coords[-1])]
        b0 = Point(buses.at[line.bus0, "x"], buses.at[line.bus0, "y"])
        b1 = Point(buses.at[line.bus1, "x"], buses.at[line.bus1, "y"])
        d = min(ends[0].distance(b0) + ends[1].distance(b1),
                ends[0].distance(b1) + ends[1].distance(b0))
        if d > tolerance:
            violations.append(f"line {name}: geometry ends {d:.1f} m away from its buses")
    return violations


def check_line_lengths(components, rtol=0.01):
    """Line lengths match their geometry: either the length of the geometry
    (lines along streets) or the straight distance times the detour factor
    `branch_detour_factor` (lines between two points); never shorter than the
    straight distance between the buses."""
    detour = cfg_ding0.get("assumptions", "branch_detour_factor")
    buses = components["Bus"]
    violations = []
    for name, line in components["Line"].iterrows():
        length_m = line.length * 1e3
        geometry_m = line.geometry.length
        straight = Point(buses.at[line.bus0, "x"], buses.at[line.bus0, "y"]).distance(
            Point(buses.at[line.bus1, "x"], buses.at[line.bus1, "y"]))
        if length_m < 1.0:
            violations.append(f"line {name}: length {length_m:.2f} m below 1 m")
        if length_m + 1e-6 < straight * (1 - rtol):
            violations.append(f"line {name}: length {length_m:.1f} m shorter than the "
                              f"straight distance {straight:.1f} m between its buses")
        if geometry_m > 0.5 and not (
                np.isclose(length_m, geometry_m, rtol=rtol)
                or np.isclose(length_m, geometry_m * detour, rtol=rtol)):
            violations.append(f"line {name}: length {length_m:.1f} m, geometry {geometry_m:.1f} m "
                              f"(ratio {length_m / geometry_m:.3f})")
    return violations


def check_containment(mv_grid_district, tolerance=1.0, building_tolerance=1.0):
    """Objects lie where they belong: the HV/MV station and all generators in
    the MV grid district; load areas in the district; LV stations, their
    buildings and loads in their load area (buildings within
    `building_tolerance` in m); an LV grid district's polygon contains its
    station and buildings."""
    district = mv_grid_district.geo_data.buffer(tolerance)
    mv_grid = mv_grid_district.mv_grid
    violations = []
    if not district.contains(mv_grid.station().geo_data):
        violations.append(f"{mv_grid.station()} outside {mv_grid_district}")
    for generator in mv_grid.generators():
        if not district.contains(generator.geo_data):
            violations.append(f"{generator} outside {mv_grid_district}")
    for la in mv_grid_district.lv_load_areas():
        if not district.contains(la.geo_area):
            share = la.geo_area.difference(district).area / la.geo_area.area
            violations.append(f"{la}: {share:.1%} of its area outside {mv_grid_district}")
        area = la.geo_area.buffer(tolerance)
        for lvgd in la.lv_grid_districts():
            station = lvgd.lv_grid.station()
            if not area.contains(station.geo_data):
                violations.append(f"{station} outside {la}")
            outside = [b for b, g in lvgd.buildings.geometry.items()
                       if la.geo_area.distance(g) > building_tolerance]
            if outside:
                violations.append(f"{lvgd}: buildings {outside[:5]} outside {la}")
            if lvgd.geo_data is not None:
                polygon = lvgd.geo_data.buffer(tolerance)
                if not polygon.contains(station.geo_data):
                    violations.append(f"{station} outside the polygon of {lvgd}")
                outside = [b for b, g in lvgd.buildings.geometry.items() if not polygon.contains(g)]
                if outside:
                    violations.append(f"{lvgd}: buildings {outside[:5]} outside its polygon")
            for generator in lvgd.lv_grid.generators():
                if not district.contains(generator.geo_data):
                    violations.append(f"{generator} in {lvgd} outside {mv_grid_district}")
    return violations


def check_lv_connection_location(network):
    """LV generators and components are connected within their own MV grid
    district; one of voltage level 7 with a building id at that building."""
    violations = []
    for mvgd in network.mv_grid_districts():
        for la in mvgd.lv_load_areas():
            for lvgd in la.lv_grid_districts():
                grid = lvgd.lv_grid
                buildings = {n.building_id: n for n in grid.graph.nodes()
                             if isinstance(n, LVLoadDing0)}
                for node in list(grid.generators()) + list(grid.components()):
                    if node.mv_grid is not mvgd.mv_grid:
                        violations.append(f"{node} of {node.mv_grid} connected to {grid} "
                                          f"in {mvgd}")
                    building_id = getattr(node, "building_id", None)
                    if node.v_level == 7 and building_id and isinstance(node, GeneratorDing0):
                        load = buildings.get(building_id)
                        if load is None:
                            continue
                        if load.geo_data.distance(node.geo_data) > 1.0:
                            violations.append(f"{node}: {load.geo_data.distance(node.geo_data):.0f} m "
                                              f"from its building {building_id}")
    return violations


def check_circuit_breaker_positions(mv_grid, tolerance=1.0):
    """Every switch disconnector lies on the geometry of its branch."""
    violations = []
    for cb in mv_grid.circuit_breakers():
        if cb.branch is None or cb.branch.geometry is None:
            continue
        d = cb.branch.geometry.distance(cb.geo_data)
        if d > tolerance:
            violations.append(f"{cb}: {d:.0f} m from its branch {cb.branch}")
    return violations


def check_no_duplicate_buses(components):
    """No two street nodes (cable distributors that are no building
    connection) of the same voltage and grid at the same position. Buses of
    buildings, generators and loads may coincide when the objects do."""
    buses = components["Bus"]
    buses = buses.loc[buses.index.str.startswith("BranchTee_")
                      & ~buses.index.str.contains("_building_")]
    keys = buses.assign(xr=buses.x.round(2), yr=buses.y.round(2),
                        grid=buses.lv_grid_id.fillna(-1))
    dup = keys.loc[keys.duplicated(["xr", "yr", "v_nom", "grid"], keep=False)]
    return [f"buses at the same position: {list(group.index)}"
            for _, group in dup.groupby(["xr", "yr", "v_nom", "grid"])]


def check_lv_feeder_lengths(components, max_length=None):
    """The cable length from an LV station to every bus of its LV grid is at
    most `ons_dist_threshold` (plus the house connection)."""
    if max_length is None:
        from ding0.config.config_lv_grids_osm import get_config_osm
        max_length = get_config_osm("ons_dist_threshold")
    buses = components["Bus"]
    lines = components["Line"]
    graph = nx.Graph()
    for _, line in lines.iterrows():
        graph.add_edge(line.bus0, line.bus1, length=line.length * 1e3)
    violations = []
    for _, trafo in components["Transformer"].iterrows():
        busbar = trafo.bus1
        lv = buses.lv_grid_id.get(busbar)
        if busbar not in graph:
            continue
        distances = nx.single_source_dijkstra_path_length(graph, busbar, weight="length")
        far = {b: d for b, d in distances.items()
               if buses.at[b, "lv_grid_id"] == lv and d > max_length}
        if far:
            worst = max(far, key=far.get)
            violations.append(f"LV grid {int(lv)}: {len(far)} buses farther than {max_length} m "
                              f"from the station, {worst} at {far[worst]:.0f} m")
    return violations
