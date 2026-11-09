"""Properties of ding0 on egon-data that hold for every egon-data database.

The tests check invariants instead of reference numbers, so they run
unchanged on every egon-data database (e.g. after a new egon-data run).

The import tests let Hypothesis draw MV grid districts (and load areas) from
the database. Each test draws `DING0_DB_EXAMPLES` districts (default 3); set
it higher for a thorough check of a new database. Hypothesis reports the
district of a failing example and replays it in the next run (`.hypothesis/`).

The run tests build the grid of one small MV grid district (the one with the
fewest load areas, at least 3) and check the exported grid.
"""

import os
from types import SimpleNamespace

import networkx as nx
import pypsa
import pytest
import shapely.wkt
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform
from sqlalchemy import text

from ding0.core import NetworkDing0
from ding0.tools import database
from ding0.tools import egon_data_integration as db_io
from ding0.tools.planning_profiles import get_limit

EDISGO_CHARGING_POINT_SECTORS = {"home", "work", "public", "hpc"}
VOLTAGE_LEVELS = {4, 5, 6, 7}

db_examples = settings(
    max_examples=int(os.environ.get("DING0_DB_EXAMPLES", 3)),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.fixture(scope="module")
def session():
    with database.session_scope() as session:
        yield session


@pytest.fixture(scope="module")
def orm(session):
    return NetworkDing0(name="network", session=session).orm


@pytest.fixture(scope="module")
def district_ids(session):
    return [int(bus_id) for (bus_id,) in session.execute(text(
        "SELECT bus_id FROM grid.egon_mv_grid_district ORDER BY bus_id"))]


def draw_district(data, orm, session, district_ids):
    """MV grid district as the import functions expect it."""
    district_id = data.draw(st.sampled_from(district_ids), label="mv_grid_district")
    mv_data = db_io.get_mv_data(orm, session, [district_id])
    return SimpleNamespace(
        id_db=district_id,
        geo_data=shapely.wkt.loads(mv_data.poly_geom.iloc[0]),
        substation=shapely.wkt.loads(mv_data.subs_geom.iloc[0]),
    )


def draw_load_area(data, orm, session, district_ids):
    """MV grid district with load areas and one of them."""
    district = draw_district(data, orm, session, district_ids)
    load_areas = db_io.get_lv_load_areas(orm, session, district.id_db)
    if load_areas.empty:
        return district, load_areas, None
    load_area_id = data.draw(st.sampled_from(sorted(load_areas.index)), label="load_area")
    return district, load_areas, load_areas.loc[load_area_id]


# Import of egon-data ########################################################

@db_examples
@given(data=st.data())
def test_substation_lies_in_its_mv_grid_district(data, orm, session, district_ids):
    district = draw_district(data, orm, session, district_ids)
    # substations are in EPSG:4326, districts in EPSG:3035
    to_3035 = Transformer.from_crs("epsg:4326", "epsg:3035", always_xy=True).transform

    assert district.geo_data.is_valid and district.geo_data.area > 0
    assert district.geo_data.contains(transform(to_3035, district.substation))


@db_examples
@given(data=st.data())
def test_load_areas_lie_in_their_mv_grid_district(data, orm, session, district_ids):
    district, load_areas, _ = draw_load_area(data, orm, session, district_ids)

    assert load_areas.index.is_unique
    for load_area in load_areas.itertuples():
        area = shapely.wkt.loads(load_area.geo_area)
        assert area.area > 0, load_area.Index
        assert area.contains(shapely.wkt.loads(load_area.geo_centre)), load_area.Index
        assert area.intersects(district.geo_data), load_area.Index
    peak_loads = ["peak_load_residential", "peak_load_cts", "peak_load_industrial"]
    assert (load_areas[peak_loads] >= 0).all().all()


@db_examples
@given(data=st.data())
def test_buildings_of_a_load_area(data, orm, session, district_ids):
    district, _, load_area = draw_load_area(data, orm, session, district_ids)
    if load_area is None:
        return
    buildings = db_io.get_egon_buildings(orm, session, district.id_db, load_area)
    if buildings.empty:
        return
    components, _ = db_io.get_building_components(orm, session, buildings, load_area)

    assert buildings.index.is_unique
    sectors = ["residential_capacity", "cts_capacity", "industrial_capacity"]
    assert (buildings[sectors] >= 0).all().all()
    assert buildings.capacity.to_numpy() == pytest.approx(
        buildings[sectors].sum(axis="columns").to_numpy())
    assert (buildings.number_households >= 0).all()
    area = shapely.wkt.loads(load_area.geo_area)
    assert all(area.intersects(point) for point in buildings.geometry)
    assert components.index.equals(buildings.index)
    assert (components >= 0).all().all()


@db_examples
@given(data=st.data())
def test_generators_of_an_mv_grid_district(data, orm, session, district_ids):
    district = draw_district(data, orm, session, district_ids)
    max_capacity = get_limit("assumptions", "max_generator_capacity")

    for get_generators in (db_io.get_res_generators, db_io.get_conv_generators):
        generators = get_generators(orm, session, district)

        assert (generators.subst_id == district.id_db).all()
        assert set(generators.voltage_level) <= VOLTAGE_LEVELS
        assert (generators.electrical_capacity > 0).all()
        if max_capacity is not None:
            assert (generators.electrical_capacity <= max_capacity).all()
        assert all(district.geo_data.intersects(shapely.wkt.loads(geom))
                   for geom in generators.geom)
        # same order in every query, so that the grids are reproducible
        assert generators.equals(get_generators(orm, session, district))


@db_examples
@given(data=st.data())
def test_point_components_of_an_mv_grid_district(data, orm, session, district_ids):
    district = draw_district(data, orm, session, district_ids)
    components = db_io.get_point_components(orm, session, district)

    assert components.source_id.is_unique
    assert set(components.v_level) <= VOLTAGE_LEVELS
    assert (components.p_nom > 0).all()
    assert all(district.geo_data.contains(point) for point in components.geometry)
    charging_points = components.loc[components.type == "charging_point"]
    assert set(charging_points.sector) <= EDISGO_CHARGING_POINT_SECTORS


# Grid of one MV grid district ###############################################

@pytest.fixture(scope="module")
def run(session):
    district_id = int(session.execute(text(
        "SELECT bus_id FROM demand.egon_loadarea GROUP BY bus_id "
        "HAVING count(*) >= 3 ORDER BY count(*), bus_id LIMIT 1"
    )).scalar())
    network = NetworkDing0(name="network", session=session)
    messages = network.run_ding0(session=session, mv_grid_districts_no=[district_id])
    (mv_grid_district,) = list(network.mv_grid_districts())
    return SimpleNamespace(
        network=network, messages=messages, mv_grid_district=mv_grid_district,
        components=network.to_dataframe())


def grid_graph(components, close_switches=False):
    """Graph of the buses connected by lines and MV/LV transformers."""
    graph = nx.MultiGraph()
    graph.add_nodes_from(components["Bus"].index)
    for branches in (components["Line"], components["Transformer"]):
        graph.add_edges_from(zip(branches.bus0, branches.bus1))
    if close_switches:
        switches = components["Switch"]
        graph.add_edges_from(zip(switches.bus_closed, switches.bus_open))
    return graph


def test_run_has_no_invalid_mv_grid_districts(run):
    invalid_districts = run.messages[0]
    assert invalid_districts == []


def test_station_lies_in_its_mv_grid_district(run):
    mv_grid_district = run.mv_grid_district

    assert mv_grid_district.geo_data.contains(mv_grid_district.mv_grid.station().geo_data)


def test_components_are_at_buses_of_the_grid(run):
    components = run.components
    buses = components["Bus"].index

    for name in ("Generator", "Load", "StorageUnit"):
        assert components[name].bus.isin(buses).all(), name
    for name in ("Line", "Transformer", "Switch"):
        df = components[name]
        ends = ("bus_closed", "bus_open") if name == "Switch" else ("bus0", "bus1")
        assert df[list(ends)].isin(buses).all().all(), name
    # the HV side of the HV/MV transformers is not part of the grid
    assert components["HVMV_Transformer"].bus1.isin(buses).all()


def test_lines_connect_buses_of_one_voltage_and_transformers_mv_to_lv(run):
    components = run.components
    v_nom = components["Bus"].v_nom
    lines = components["Line"]
    transformers = components["Transformer"]

    assert (v_nom[lines.bus0].values == v_nom[lines.bus1].values).all()
    assert (lines.length > 0).all()
    assert (lines.num_parallel >= 1).all()
    assert (v_nom[transformers.bus0].values > 1).all()
    assert (v_nom[transformers.bus1].values == 0.4).all()


def test_grid_is_connected_and_operated_radially(run):
    open_switches = grid_graph(run.components)
    closed_switches = grid_graph(run.components, close_switches=True)

    assert nx.is_connected(open_switches)
    # parallel lines or transformers between two buses are not a ring
    assert nx.cycle_basis(nx.Graph(open_switches)) == []
    # each switch is the open point of one MV ring
    assert len(nx.cycle_basis(nx.Graph(closed_switches))) == len(run.components["Switch"])


def test_each_lv_grid_has_a_transformer(run):
    buses = run.components["Bus"]
    lv_grids = buses.lv_grid_id.dropna().unique()
    transformers = run.components["Transformer"]

    assert set(buses.lv_grid_id[transformers.bus1]) == set(lv_grids)


def test_all_generators_of_the_database_are_in_the_grid(run, orm, session):
    generators = run.components["Generator"]
    generators = generators.loc[generators.control != "Slack"]
    database_generators = [
        get_generators(orm, session, run.mv_grid_district)
        for get_generators in (db_io.get_res_generators, db_io.get_conv_generators)]

    assert len(generators) == sum(len(df) for df in database_generators)
    assert generators.p_nom.sum() == pytest.approx(
        sum(df.electrical_capacity.sum() for df in database_generators) / 1e3)
    assert (run.components["Generator"].control == "Slack").sum() == 1


def test_all_charging_points_of_the_database_are_in_the_grid(run, orm, session):
    loads = run.components["Load"]
    in_grid = loads.loc[loads.type == "charging_point"].groupby("sector").p_set.sum()
    # charging locations in buildings and at locations of their own
    point_components = db_io.get_point_components(orm, session, run.mv_grid_district)
    charging_points = point_components.loc[point_components.type == "charging_point"]
    in_database = charging_points.groupby("sector").p_nom.sum() / 1e3

    assert in_grid.to_dict() == pytest.approx(in_database.to_dict())


@pytest.fixture(scope="module")
def netcdf_grid(run, tmp_path_factory):
    """The grid of the run, exported as netCDF and read with PyPSA."""
    path = tmp_path_factory.mktemp("netcdf")
    run.network.to_netcdf(str(path))
    return pypsa.Network(str(path / f"{run.mv_grid_district.id_db}.nc"))


def test_power_flow_converges(netcdf_grid):
    netcdf_grid.determine_network_topology()

    assert len(netcdf_grid.sub_networks) == 1
    assert netcdf_grid.pf()["converged"].all().all()


def test_netcdf_coordinates_are_in_the_srid_of_the_network(run, netcdf_grid):
    # the slack is at the MV busbar of the HV/MV station
    generators = netcdf_grid.generators
    (busbar,) = generators.loc[generators.control == "Slack", "bus"]
    busbar = netcdf_grid.buses.loc[busbar]
    to_3035 = Transformer.from_crs(
        f"epsg:{netcdf_grid.srid}", "epsg:3035", always_xy=True).transform

    assert run.mv_grid_district.geo_data.contains(Point(to_3035(busbar.x, busbar.y)))
