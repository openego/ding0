"""Synthetic stand-in for the egon-data database.

Provides the data that `ding0.tools.egon_data_integration` would query, so
that complete ding0 runs can be tested without a database. All geometries
are in EPSG:3035 except the substation points, which egon-data stores in
EPSG:4326. Each load area is a rectangular street grid with buildings along
the streets.

There are two MV grid districts:

* District 1 has three rural load areas, rooftop PV, one wind farm and one
  CHP.
* District 2 covers the code paths district 1 misses: an urban load area
  large enough to be aggregated (street-routed urban MV grid), with dead-end
  streets that become stubs and a large CTS building on the MV level; a
  village with industrial loads of all three LV/MV/HV classes; a satellite
  load area below 100 kW; and generators of voltage levels 4 to 7, including
  a rooftop PV unit assigned to a building.

The getters mirror the queries: load areas and buildings are selected per
grid district, ways by intersection with the queried polygon, generators by
intersection with the grid district polygon.
"""

import pandas as pd
from geoalchemy2.shape import from_shape
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point, box
from shapely.ops import transform, unary_union

SRID = 3035

# Lower left corners of the grid districts in EPSG:3035 (near Berlin), the
# substation position and the district polygon relative to that corner.
DISTRICTS = {
    1: {"origin": (4_550_000.0, 3_280_000.0), "substation": (900.0, 900.0),
        "extent": (-500.0, -500.0, 2_600.0, 2_600.0)},
    2: {"origin": (4_556_000.0, 3_280_000.0), "substation": (1_200.0, -400.0),
        "extent": (-1_000.0, -1_000.0, 3_500.0, 3_200.0)},
}
MV_GRID_ID = 1

# Load areas: grid district, offset of the street grid from the district's
# origin, number of street nodes in x and y direction (spacing 80 m), a factor
# on the household loads, dead-end streets as (start node, direction, number
# of segments) and extra buildings as (node, dx, dy, residential, cts,
# industrial, households).
_SPACING = 80.0
LOAD_AREAS = {
    101: {"mvgd": 1, "offset": (0.0, 0.0), "nodes": (8, 6), "load_factor": 2.5,
          "extra_buildings": [((2, 2), 25.0, 25.0, 0.0, 150.0, 0.0, 0),
                              ((5, 4), 30.0, 30.0, 0.0, 450.0, 0.0, 0)]},
    102: {"mvgd": 1, "offset": (1_600.0, 300.0), "nodes": (5, 5), "load_factor": 1.0},
    103: {"mvgd": 1, "offset": (500.0, 1_700.0), "nodes": (4, 6), "load_factor": 1.0},
    # urban: about 5.6 MW of LV load exceeds what a 10 kV ring can carry.
    # The MV loads at the ends of two dead-end streets make them stubs of the
    # urban MV grid; the one above 1 MVA is pulled into the routed core.
    201: {"mvgd": 2, "offset": (0.0, 0.0), "nodes": (10, 9), "load_factor": 7.5,
          "stubs": [((9, 2), (1, 0), 5), ((0, 6), (-1, 0), 4), ((4, 8), (0, 1), 2)],
          "extra_buildings": [((6, 3), 30.0, 30.0, 0.0, 2_000.0, 0.0, 0),
                              ((14, 2), 10.0, 25.0, 0.0, 600.0, 0.0, 0),
                              ((-4, 6), -10.0, 25.0, 0.0, 1_500.0, 0.0, 0)]},
    # village: industry connected to the street grid's station (150 kW), to
    # the MV grid (1.5 MW) and above the MV threshold (8 MW)
    202: {"mvgd": 2, "offset": (2_000.0, 1_200.0), "nodes": (5, 4), "load_factor": 1.0,
          "extra_buildings": [((1, 1), 25.0, 25.0, 0.0, 0.0, 150.0, 0),
                              ((3, 2), 30.0, 30.0, 0.0, 0.0, 1_500.0, 0),
                              ((4, 0), 40.0, -40.0, 0.0, 0.0, 8_000.0, 0)]},
    # satellite: peak load below 100 kW
    203: {"mvgd": 2, "offset": (2_600.0, 2_400.0), "nodes": (3, 3), "load_factor": 0.4},
}


def _origin(la_id):
    district = DISTRICTS[LOAD_AREAS[la_id]["mvgd"]]
    ox, oy = district["origin"]
    dx, dy = LOAD_AREAS[la_id]["offset"]
    return ox + dx, oy + dy


def _node_id(la_id, i, j):
    return la_id * 10_000 + i * 100 + j


def _node_xy(la_id, i, j):
    # The deterministic jitter of a few metres makes shortest paths unique.
    # On a regular grid many routes tie, and which one is taken depends on
    # the networkx version.
    ox, oy = _origin(la_id)
    dx = ((i * i * 7 + j * j * 13 + i * j * 5 + la_id) % 17 - 8) * 0.91
    dy = ((i * i * 3 + j * j * 11 + i * j * 7 + la_id) % 13 - 6) * 1.27
    return ox + i * _SPACING + dx, oy + j * _SPACING + dy


def _stub_nodes(la_id):
    """Node indices of each dead-end street, starting at the grid node."""
    stubs = []
    for (i, j), (di, dj), length in LOAD_AREAS[la_id].get("stubs", []):
        stubs.append([(i + k * di, j + k * dj) for k in range(length + 1)])
    return stubs


def _area_polygon(la_id):
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    x0, y0 = _node_xy(la_id, 0, 0)
    x1, y1 = _node_xy(la_id, nx_ - 1, ny_ - 1)
    polygon = box(x0 - 20.0, y0 - 20.0, x1 + 20.0, y1 + 20.0)
    stubs = [LineString([_node_xy(la_id, i, j) for i, j in stub]).buffer(40.0)
             for stub in _stub_nodes(la_id)]
    if stubs:
        polygon = unary_union([polygon] + stubs).envelope
    return polygon


def _district_polygon(mvgd_id):
    ox, oy = DISTRICTS[mvgd_id]["origin"]
    x0, y0, x1, y1 = DISTRICTS[mvgd_id]["extent"]
    return box(ox + x0, oy + y0, ox + x1, oy + y1)


def mv_data(mvgd_ids=(MV_GRID_ID,)):
    """Rows as returned by `get_mv_data`."""
    to_wgs84 = Transformer.from_crs("epsg:3035", "epsg:4326", always_xy=True).transform
    rows = {}
    for mvgd_id in mvgd_ids:
        ox, oy = DISTRICTS[mvgd_id]["origin"]
        sx, sy = DISTRICTS[mvgd_id]["substation"]
        station = transform(to_wgs84, Point(ox + sx, oy + sy))
        rows[mvgd_id] = {"poly_geom": _district_polygon(mvgd_id).wkt,
                         "subs_geom": station.wkt}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("bus_id")


def lv_load_areas(mvgd_id=MV_GRID_ID):
    """Rows as returned by `get_lv_load_areas`, before the peak load filter."""
    rows = []
    for la_id, spec in LOAD_AREAS.items():
        if spec["mvgd"] != mvgd_id:
            continue
        polygon = _area_polygon(la_id)
        la_buildings = buildings(la_id)
        peak = la_buildings.capacity.sum()
        industrial = la_buildings.industrial_capacity.sum()
        rows.append({
            "id_db": la_id,
            "population": 100,
            "area": polygon.area / 1e4,
            "geo_area": polygon.wkt,
            "geo_centre": polygon.centroid.wkt,
            "peak_load_residential": peak - industrial,
            "peak_load_cts": 0.0,
            "peak_load_industrial": industrial,
            "peak_load_residential_scaling_factor": 1.0,
            "peak_load_cts_scaling_factor": 1.0,
            "peak_load_industrial_scaling_factor": 1.0,
            "peak_load": peak,
        })
    return pd.DataFrame(rows).set_index("id_db")


def ways(la_id):
    """Rows as returned by `get_egon_ways`: one way per street row/column
    and per dead-end street."""
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    lines = [[(i, j) for i in range(nx_)] for j in range(ny_)]
    lines += [[(i, j) for j in range(ny_)] for i in range(nx_)]
    lines += _stub_nodes(la_id)
    rows = []
    for osm_id, line in enumerate(lines, start=la_id * 1_000):
        coords = [_node_xy(la_id, i, j) for i, j in line]
        rows.append({
            "osm_id": osm_id,
            "nodes": [_node_id(la_id, i, j) for i, j in line],
            "geometry": from_shape(LineString(coords), srid=SRID),
            "highway": "residential",
            "length_segments": [Point(a).distance(Point(b)) for a, b in zip(coords, coords[1:])],
        })
    return pd.DataFrame(rows)


def buildings(la_id):
    """Rows as returned by `get_egon_buildings`.

    Buildings sit 15 m beside the middle of every horizontal street segment
    and of every dead-end street segment, alternating sides. Loads cycle
    through typical household values, scaled by the area's load factor;
    every eleventh building is commercial. The extra buildings of a load area
    follow.
    """
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    household_kw = [4.0, 6.5, 9.0, 12.5, 5.5, 18.0]
    factor = LOAD_AREAS[la_id]["load_factor"]
    segments = [((i, j), (i + 1, j)) for j in range(ny_) for i in range(nx_ - 1)]
    for stub in _stub_nodes(la_id):
        segments += list(zip(stub, stub[1:]))
    rows = []
    n = 0
    for a, b in segments:
        xa, ya = _node_xy(la_id, *a)
        xb, yb = _node_xy(la_id, *b)
        side = 15.0 if n % 2 == 0 else -15.0
        # offset perpendicular to the street: in y for streets along x
        horizontal = a[1] == b[1]
        dx, dy = (0.0, side) if horizontal else (side, 0.0)
        x = xa + _SPACING / 2 if horizontal else xa
        y = ya if horizontal else (ya + yb) / 2
        residential = household_kw[n % len(household_kw)] * factor
        cts = 35.0 if n % 11 == 5 else 0.0
        rows.append(_building(la_id * 1_000 + n, x + dx, y + dy,
                              residential, cts, households=1 + n % 3))
        n += 1
    for (i, j), dx, dy, residential, cts, industrial, households in \
            LOAD_AREAS[la_id].get("extra_buildings", []):
        x, y = _node_xy(la_id, i, j)
        rows.append(_building(la_id * 1_000 + n, x + dx, y + dy,
                              residential, cts, households, industrial))
        n += 1
    return pd.DataFrame(rows).set_index("building_id")


def _building(building_id, x, y, residential, cts, households, industrial=0.0):
    point = Point(x, y)
    return {
        "building_id": building_id,
        "number_households": households,
        "residential_capacity": residential,
        "cts_capacity": cts,
        "industrial_capacity": industrial,
        "capacity": residential + cts + industrial,
        "geometry": point,
        "footprint": point.buffer(6.0, cap_style=3),
    }


def _generator(gens_id, x, y, capacity, voltage_level, generation_type,
               generation_subtype, w_id=None, building_id=None, subst_id=MV_GRID_ID):
    return {
        "subst_id": subst_id,
        "gens_id": gens_id,
        "electrical_capacity": capacity,
        "voltage_level": voltage_level,
        "w_id": w_id,
        "geom": Point(x, y).wkt,
        "generation_type": generation_type,
        "generation_subtype": generation_subtype,
        "building_id": building_id,
    }


def _rel(mvgd_id, dx, dy):
    ox, oy = DISTRICTS[mvgd_id]["origin"]
    return ox + dx, oy + dy


def _in_district(frame, mvgd_id):
    """Generators within the grid district polygon, with its subst_id, as
    the spatial queries of `get_res_generators` and `get_conv_generators`
    return them."""
    polygon = _district_polygon(mvgd_id)
    inside = frame.geom.apply(lambda g: polygon.intersects(wkt.loads(g)))
    frame = frame.loc[inside].reset_index(drop=True)
    frame["subst_id"] = mvgd_id
    return frame


def res_generators(mvgd_id=MV_GRID_ID):
    """Rows as returned by `get_res_generators`."""
    rows = []
    # district 1: rooftop PV in every load area and one wind farm
    for la_id in (101, 102, 103):
        x, y = _node_xy(la_id, 1, 1)
        rows.append(_generator(f"pv_{la_id}", x + 40.0, y + 15.0, 9.8,
                               7, "solar", "pv_rooftop", w_id=1))
        x, y = _node_xy(la_id, 2, 3)
        rows.append(_generator(f"pv_big_{la_id}", x + 40.0, y - 15.0, 60.0,
                               6, "solar", "pv_rooftop", w_id=1))
    rows.append(_generator("wind_1", *_rel(1, 2_400.0, 2_200.0),
                           3_000.0, 5, "wind", "wind_onshore", w_id=1))
    # district 2: one generator per voltage level 4 to 7
    rows.append(_generator("wind_2", *_rel(2, 3_200.0, -800.0),
                           6_000.0, 4, "wind", "wind_onshore", w_id=2))
    rows.append(_generator("pv_park_2", *_rel(2, -700.0, 2_500.0),
                           1_800.0, 5, "solar", "open_space", w_id=2))
    x, y = _node_xy(202, 2, 1)
    rows.append(_generator("pv_barn_202", x + 40.0, y + 20.0,
                           150.0, 6, "solar", "pv_rooftop", w_id=2))
    roof = buildings(202).iloc[3]
    rows.append(_generator("pv_roof_202", roof.geometry.x, roof.geometry.y,
                           12.0, 7, "solar", "pv_rooftop", w_id=2,
                           building_id=int(buildings(202).index[3])))
    x, y = _node_xy(201, 5, 5)
    rows.append(_generator("pv_roof_201", x + 40.0, y + 15.0,
                           20.0, 7, "solar", "pv_rooftop", w_id=2))
    return _in_district(pd.DataFrame(rows), mvgd_id)


def conv_generators(mvgd_id=MV_GRID_ID):
    """Rows as returned by `get_conv_generators`: one CHP per district."""
    return _in_district(pd.DataFrame([
        _generator("chp_1", *_rel(1, 1_200.0, -300.0), 800.0, 5, "gas", "chp"),
        _generator("chp_2", *_rel(2, 1_000.0, 1_800.0), 500.0, 5,
                   "conventional", "combustion"),
    ]), mvgd_id)


# Components of newer egon-data versions, in grid district 2 only. They are
# used if `patch_database` is called with `components=True`.

# heat pumps (kW) by building; the one in load area 201 turns a household
# into a load with a feeder of its own (100-200 kW)
HEAT_PUMPS = {202000: 9.0, 202001: 9.0, 202002: 12.0, 201004: 120.0}
# home batteries (kW, kWh) by building; 202003 has rooftop PV
HOME_BATTERIES = {202003: (8.0, 10.0), 201010: (5.0, 6.0)}


def home_charging_locations():
    """Home charging locations (use case home_detached, voltage level 7): two
    next to buildings of load area 202, one too far from any building."""
    rows = []
    for location_id, building_index in [(1, 5), (2, 9)]:
        building = buildings(202).iloc[building_index]
        rows.append({"location_id": location_id, "p_set": 11.0,
                     "geometry": Point(building.geometry.x + 8.0, building.geometry.y)})
    x, y = _node_xy(203, 1, 1)
    rows.append({"location_id": 3, "p_set": 11.0, "geometry": Point(x, y + 400.0)})
    return pd.DataFrame(rows)


def point_components(mvgd_id, exclude_charging_locations=()):
    """Rows as returned by `get_point_components`."""
    columns = ["component", "type", "sector", "use_case", "p_nom", "v_level",
               "geometry", "source_id", "max_hours"]
    if mvgd_id != 2:
        return pd.DataFrame(columns=columns)
    rows = []
    for location in home_charging_locations().itertuples():
        if location.location_id not in set(exclude_charging_locations):
            rows.append(["load", "charging_point", "home", "home_detached",
                         location.p_set, 7, location.geometry,
                         f"charging_location_{location.location_id}", None])
    x, y = _node_xy(201, 9, 4)
    rows.append(["load", "charging_point", "hpc", "urban_fast", 300.0, 5,
                 Point(x + 30.0, y), "charging_location_10", None])
    rows.append(["load", "charging_point", "hpc", "highway_fast", 2_000.0, 4,
                 Point(*_rel(2, 3_000.0, 2_800.0)), "charging_location_11", None])
    x, y = _node_xy(202, 2, 2)
    rows.append(["load", "charging_point", "public", "street", 22.0, 7,
                 Point(x + 10.0, y + 20.0), "charging_location_12", None])
    x, y = _node_xy(202, 4, 3)
    rows.append(["load", "charging_point", "work", "work", 150.0, 6,
                 Point(x + 20.0, y + 20.0), "charging_location_13", None])
    # within load area 202, so the ring branch to it runs along its streets
    x, y = _node_xy(202, 1, 2)
    rows.append(["load", "charging_point", "public", "retail", 400.0, 5,
                 Point(x + 15.0, y + 10.0), "charging_location_14", None])
    rows.append(["load", "charging_point", "hpc", "bus_depot", 800.0, 5,
                 Point(*_rel(2, 1_500.0, 600.0)), "bus_depot_1", None])
    x, y = _node_xy(201, 3, 0)
    rows.append(["load", "conventional_load", "rail", "rail_transit_dc", 1_200.0, 5,
                 Point(x, y - 60.0), "rail_1", None])
    rows.append(["storage_unit", "large_storage", None, "BESS", 500.0, 5,
                 Point(*_rel(2, 2_200.0, 1_000.0)), "storage_1", 1.0])
    return pd.DataFrame(rows, columns=columns)


def building_components(buildings_df):
    """Components of buildings as returned by `get_building_components`."""
    import ding0.tools.egon_data_integration as db_io

    components = pd.DataFrame(0.0, index=buildings_df.index,
                              columns=db_io.BUILDING_COMPONENT_COLUMNS)
    for building_id, p_set in HEAT_PUMPS.items():
        if building_id in components.index:
            components.loc[building_id, "heat_pump_p_set"] = p_set
    for building_id, (p_nom, capacity) in HOME_BATTERIES.items():
        if building_id in components.index:
            components.loc[building_id, "home_storage_p_nom"] = p_nom
            components.loc[building_id, "home_storage_capacity"] = capacity
    assigned = db_io.assign_to_nearest_buildings(
        home_charging_locations(), buildings_df, max_distance=50.0)
    charging = assigned.groupby("building_id").p_set.sum()
    components.loc[charging.index, "charging_point_p_set"] = charging
    return components, assigned.location_id.tolist()


def patch_database(monkeypatch, components=False):
    """Replace all database getters used by `NetworkDing0.run_ding0`.

    With `components`, grid district 2 has heat pumps, charging points, bus
    depots, a rail substation and battery storage units.
    """
    import ding0.tools.egon_data_integration as db_io

    def get_mv_data(orm, session, mv_grid_districts_no):
        return mv_data(mv_grid_districts_no)

    def get_lv_load_areas(orm, session, mv_grid_id):
        return lv_load_areas(mv_grid_id)

    def get_egon_ways(orm, session, geo_area):
        # ways of the load areas whose polygon the queried buffer intersects
        area = wkt.loads(geo_area)
        frames = [ways(la_id) for la_id in LOAD_AREAS if area.intersects(_area_polygon(la_id))]
        return pd.concat(frames, ignore_index=True)

    def get_egon_buildings(orm, session, subst_id, load_area, scenario=None):
        return buildings(load_area.name)

    def get_res_generators(orm, session, mv_grid_district, scenario=None):
        return res_generators(mv_grid_district.id_db)

    def get_conv_generators(orm, session, mv_grid_district):
        return conv_generators(mv_grid_district.id_db)

    monkeypatch.setattr(db_io, "get_mv_data", get_mv_data)
    monkeypatch.setattr(db_io, "get_lv_load_areas", get_lv_load_areas)
    monkeypatch.setattr(db_io, "get_egon_ways", get_egon_ways)
    monkeypatch.setattr(db_io, "get_egon_buildings", get_egon_buildings)
    monkeypatch.setattr(db_io, "get_res_generators", get_res_generators)
    monkeypatch.setattr(db_io, "get_conv_generators", get_conv_generators)
    if components:
        monkeypatch.setattr(
            db_io, "get_building_components",
            lambda orm, session, buildings_df, load_area, scenario=None:
            building_components(buildings_df))
        monkeypatch.setattr(
            db_io, "get_point_components",
            lambda orm, session, mv_grid_district, scenario=None,
            exclude_charging_locations=(): point_components(
                mv_grid_district.id_db, exclude_charging_locations))
