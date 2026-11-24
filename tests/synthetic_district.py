"""Synthetic stand-in for the egon-data database.

Provides the data that `ding0.tools.egon_data_integration` would query for one
MV grid district, so that a complete ding0 run can be tested without a
database. The district has three load areas, each a rectangular street grid
with buildings along the streets. All geometries are in EPSG:3035 except the
substation point, which egon-data stores in EPSG:4326.
"""

import pandas as pd
from geoalchemy2.shape import from_shape
from pyproj import Transformer
from shapely.geometry import LineString, Point, box
from shapely.ops import transform

MV_GRID_ID = 1
SRID = 3035

# Lower left corners of the load areas in EPSG:3035 (near Berlin) and the
# number of street nodes in x and y direction. Spacing between nodes is 80 m.
_ORIGIN = (4_550_000.0, 3_280_000.0)
_SPACING = 80.0
LOAD_AREAS = {
    101: {"offset": (0.0, 0.0), "nodes": (8, 6), "load_factor": 2.5},
    102: {"offset": (1_600.0, 300.0), "nodes": (5, 5), "load_factor": 1.0},
    103: {"offset": (500.0, 1_700.0), "nodes": (4, 6), "load_factor": 1.0},
}
_SUBSTATION = (_ORIGIN[0] + 900.0, _ORIGIN[1] + 900.0)


def _node_id(la_id, i, j):
    return la_id * 10_000 + i * 100 + j


def _node_xy(la_id, i, j):
    # The deterministic jitter of a few metres makes shortest paths unique.
    # On a regular grid many routes tie, and which one is taken depends on
    # the networkx version.
    ox, oy = LOAD_AREAS[la_id]["offset"]
    dx = ((i * i * 7 + j * j * 13 + i * j * 5 + la_id) % 17 - 8) * 0.91
    dy = ((i * i * 3 + j * j * 11 + i * j * 7 + la_id) % 13 - 6) * 1.27
    return _ORIGIN[0] + ox + i * _SPACING + dx, _ORIGIN[1] + oy + j * _SPACING + dy


def _area_polygon(la_id):
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    x0, y0 = _node_xy(la_id, 0, 0)
    x1, y1 = _node_xy(la_id, nx_ - 1, ny_ - 1)
    return box(x0 - 20.0, y0 - 20.0, x1 + 20.0, y1 + 20.0)


def mv_data():
    """Rows as returned by `get_mv_data`."""
    to_wgs84 = Transformer.from_crs("epsg:3035", "epsg:4326", always_xy=True).transform
    district = box(_ORIGIN[0] - 500.0, _ORIGIN[1] - 500.0,
                   _ORIGIN[0] + 2_600.0, _ORIGIN[1] + 2_600.0)
    station = transform(to_wgs84, Point(_SUBSTATION))
    return pd.DataFrame(
        {"poly_geom": [district.wkt], "subs_geom": [station.wkt]},
        index=pd.Index([MV_GRID_ID], name="bus_id"),
    )


def lv_load_areas():
    """Rows as returned by `get_lv_load_areas`, before the peak load filter."""
    rows = []
    for la_id in LOAD_AREAS:
        polygon = _area_polygon(la_id)
        peak = buildings(la_id).capacity.sum()
        rows.append({
            "id_db": la_id,
            "population": 100,
            "area": polygon.area / 1e4,
            "geo_area": polygon.wkt,
            "geo_centre": polygon.centroid.wkt,
            "peak_load_residential": peak,
            "peak_load_cts": 0.0,
            "peak_load_industrial": 0.0,
            "peak_load_residential_scaling_factor": 1.0,
            "peak_load_cts_scaling_factor": 1.0,
            "peak_load_industrial_scaling_factor": 1.0,
            "peak_load": peak,
        })
    return pd.DataFrame(rows).set_index("id_db")


def ways(la_id):
    """Rows as returned by `get_egon_ways`: one way per street row/column."""
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    lines = [[(i, j) for i in range(nx_)] for j in range(ny_)]
    lines += [[(i, j) for j in range(ny_)] for i in range(nx_)]
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

    Buildings sit 15 m beside the middle of every horizontal street segment,
    alternating sides. Loads cycle through typical household values, scaled
    by the area's load factor; every eleventh building is commercial. Load area 101 additionally has one
    building connected directly to its station (100-200 kW) and one connected
    to the MV grid (>= 200 kW).
    """
    nx_, ny_ = LOAD_AREAS[la_id]["nodes"]
    household_kw = [4.0, 6.5, 9.0, 12.5, 5.5, 18.0]
    factor = LOAD_AREAS[la_id]["load_factor"]
    rows = []
    n = 0
    for j in range(ny_):
        for i in range(nx_ - 1):
            x0, y = _node_xy(la_id, i, j)
            side = 15.0 if n % 2 == 0 else -15.0
            residential = household_kw[n % len(household_kw)] * factor
            cts = 35.0 if n % 11 == 5 else 0.0
            rows.append(_building(la_id * 1_000 + n, x0 + _SPACING / 2, y + side,
                                  residential, cts, households=1 + n % 3))
            n += 1
    if la_id == 101:
        x, y = _node_xy(la_id, 2, 2)
        rows.append(_building(la_id * 1_000 + n, x + 25.0, y + 25.0, 0.0, 150.0, 0))
        x, y = _node_xy(la_id, 5, 4)
        rows.append(_building(la_id * 1_000 + n + 1, x + 30.0, y + 30.0, 0.0, 450.0, 0))
    return pd.DataFrame(rows).set_index("building_id")


def _building(building_id, x, y, residential, cts, households):
    point = Point(x, y)
    return {
        "building_id": building_id,
        "number_households": households,
        "residential_capacity": residential,
        "cts_capacity": cts,
        "industrial_capacity": 0.0,
        "capacity": residential + cts,
        "geometry": point,
        "footprint": point.buffer(6.0, cap_style=3),
    }


def _generator(gens_id, x, y, capacity, voltage_level, generation_type,
               generation_subtype, w_id=None):
    return {
        "subst_id": MV_GRID_ID,
        "gens_id": gens_id,
        "electrical_capacity": capacity,
        "voltage_level": voltage_level,
        "w_id": w_id,
        "geom": Point(x, y).wkt,
        "generation_type": generation_type,
        "generation_subtype": generation_subtype,
        "building_id": None,
    }


def res_generators():
    """Rows as returned by `get_res_generators`: rooftop PV and one wind farm."""
    rows = []
    for k, la_id in enumerate(LOAD_AREAS):
        x, y = _node_xy(la_id, 1, 1)
        rows.append(_generator(f"pv_{la_id}", x + 40.0, y + 15.0, 9.8,
                               7, "solar", "pv_rooftop", w_id=1))
        x, y = _node_xy(la_id, 2, 3)
        rows.append(_generator(f"pv_big_{la_id}", x + 40.0, y - 15.0, 60.0,
                               6, "solar", "pv_rooftop", w_id=1))
    rows.append(_generator("wind_1", _ORIGIN[0] + 2_400.0, _ORIGIN[1] + 2_200.0,
                           3_000.0, 5, "wind", "wind_onshore", w_id=1))
    return pd.DataFrame(rows)


def conv_generators():
    """Rows as returned by `get_conv_generators`: one biogas CHP in the MV grid."""
    return pd.DataFrame([
        _generator("chp_1", _ORIGIN[0] + 1_200.0, _ORIGIN[1] - 300.0, 800.0,
                   5, "gas", "chp"),
    ])


def patch_database(monkeypatch):
    """Replace all database getters used by `NetworkDing0.run_ding0`."""
    import ding0.tools.egon_data_integration as db_io

    def get_lv_load_areas(orm, session, mv_grid_id):
        return lv_load_areas()

    def get_egon_ways(orm, session, geo_area):
        # ways of the load area whose buffer polygon is queried
        from shapely import wkt
        area = wkt.loads(geo_area)
        frames = [ways(la_id) for la_id in LOAD_AREAS if area.intersects(_area_polygon(la_id))]
        return pd.concat(frames, ignore_index=True)

    def get_egon_buildings(orm, session, subst_id, load_area):
        return buildings(load_area.name)

    monkeypatch.setattr(db_io, "get_mv_data", lambda orm, session, ids: mv_data())
    monkeypatch.setattr(db_io, "get_lv_load_areas", get_lv_load_areas)
    monkeypatch.setattr(db_io, "get_egon_ways", get_egon_ways)
    monkeypatch.setattr(db_io, "get_egon_buildings", get_egon_buildings)
    monkeypatch.setattr(db_io, "get_res_generators", lambda orm, session, mvgd: res_generators())
    monkeypatch.setattr(db_io, "get_conv_generators", lambda orm, session, mvgd: conv_generators())
