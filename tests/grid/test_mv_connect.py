import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from shapely.geometry import Point, box

from ding0.core import MVCableDistributorDing0, NetworkDing0
from ding0.core.network import BranchDing0, RingDing0
from ding0.core.network.components import PointLoadDing0
from ding0.core.network.grids import MVGridDing0
from ding0.core.network.stations import MVStationDing0
from ding0.core.structure.regions import (LVLoadAreaCentreDing0,
                                          LVLoadAreaDing0,
                                          MVGridDistrictDing0)
from ding0.grid.mv_grid.mv_connect import (find_connection_point,
                                           find_nearest_conn_objects)
from ding0.tools import config as cfg_ding0


def test_satellite_connects_to_point_load_in_ring():
    """A satellite whose nearest connection object is a load at a location of
    its own routed in a ring, which belongs to no load area, is connected to
    the load and starts a new satellite string there."""
    NetworkDing0()  # loads the config
    station = MVStationDing0(id_db=1, geo_data=Point(0, 0))
    mv_grid = MVGridDing0(id_db=1, station=station)
    station.grid = mv_grid
    mv_grid_district = MVGridDistrictDing0(
        id_db=1, mv_grid=mv_grid, geo_data=box(-5000, -5000, 5000, 5000))
    mv_grid.grid_district = mv_grid_district
    load = PointLoadDing0(id_db=1, geo_data=Point(1000, 0), p_nom=300.0,
                          type="charging_point", v_level=5, mv_grid=mv_grid)
    ring = RingDing0(grid=mv_grid)
    mv_grid.graph.add_edge(station, load, branch=BranchDing0(
        length=1000, kind="cable", ring=ring, grid=mv_grid))

    satellite = LVLoadAreaDing0(id_db=1, mv_grid_district=mv_grid_district,
                                peak_load=100, is_satellite=True)
    centre = LVLoadAreaCentreDing0(id_db=1, geo_data=Point(1000, 500),
                                   lv_load_area=satellite, grid=mv_grid)
    satellite.lv_load_area_centre = centre
    mv_grid.graph.add_node(centre)

    find_connection_point(
        centre, centre.geo_data, mv_grid.graph, proj=None,
        conn_objects_min_stack=[{"obj": load, "shp": load.geo_data, "dist": 500}],
        conn_dist_ring_mod=100, debug=False)

    assert mv_grid.graph.has_edge(centre, load)
    assert satellite.lv_load_area_group.root_node is load


def test_satellite_string_from_point_load_keeps_its_group():
    """The branch from a load at a location of its own to a satellite belongs
    to the satellite's string, so that further satellites connected to it
    join the string."""
    from types import SimpleNamespace
    from ding0.grid.mv_grid.mv_connect import get_lv_load_area_group_from_node_pair

    group = object()
    satellite = LVLoadAreaCentreDing0(
        lv_load_area=SimpleNamespace(lv_load_area_group=group))
    load = PointLoadDing0(id_db=1, geo_data=Point(0, 0), p_nom=300.0,
                          type="charging_point", v_level=5,
                          mv_grid=SimpleNamespace(id_db=1))
    assert get_lv_load_area_group_from_node_pair(load, satellite) is group
    assert get_lv_load_area_group_from_node_pair(satellite, load) is group


# Property: satellites connect to rings of any node types #####################

@pytest.fixture(scope="module")
def config():
    NetworkDing0()  # loads the config


def ring_node(kind, position, mv_grid, mv_grid_district, number):
    """A node of an MV ring: point load, load area centre or cable distributor."""
    if kind == "point_load":
        return PointLoadDing0(id_db=number, geo_data=position, p_nom=300.0,
                              type="charging_point", v_level=5, mv_grid=mv_grid)
    if kind == "load_area_centre":
        load_area = LVLoadAreaDing0(id_db=number, mv_grid_district=mv_grid_district,
                                    peak_load=5000)
        centre = LVLoadAreaCentreDing0(id_db=number, geo_data=position,
                                       lv_load_area=load_area, grid=mv_grid)
        load_area.lv_load_area_centre = centre
        return centre
    cable_distributor = MVCableDistributorDing0(geo_data=position, grid=mv_grid)
    mv_grid.add_cable_distributor(cable_distributor)
    return cable_distributor


# distinct positions on a 100 m lattice, the first one for the MV station
positions = st.lists(
    st.tuples(st.integers(0, 50), st.integers(0, 50)),
    min_size=4, max_size=14, unique=True,
).map(lambda points: [Point(100 * x, 100 * y) for x, y in points])


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(positions=positions,
       data=st.data())
def test_satellites_connect_to_rings_of_any_node_types(config, positions, data):
    """Satellites (small load areas) are connected to a ring whatever kinds of
    nodes it consists of: every satellite ends up connected to the MV station,
    and every satellite string of several load areas stays within the load
    limit of a string."""
    n_ring = data.draw(st.integers(1, len(positions) - 2), label="ring nodes")
    station_position, ring_positions = positions[0], positions[1:n_ring + 1]
    satellite_positions = positions[n_ring + 1:]
    kinds = data.draw(st.lists(
        st.sampled_from(["point_load", "load_area_centre", "cable_distributor"]),
        min_size=n_ring, max_size=n_ring), label="ring node kinds")
    peak_loads = data.draw(st.lists(
        st.floats(1, 300), min_size=len(satellite_positions),
        max_size=len(satellite_positions)), label="satellite peak loads")

    station = MVStationDing0(id_db=1, geo_data=station_position)
    mv_grid = MVGridDing0(id_db=1, station=station)
    station.grid = mv_grid
    mv_grid_district = MVGridDistrictDing0(
        id_db=1, mv_grid=mv_grid, geo_data=box(-1000, -1000, 6000, 6000))
    mv_grid.grid_district = mv_grid_district
    ring = RingDing0(grid=mv_grid)
    nodes = [station] + [
        ring_node(kind, position, mv_grid, mv_grid_district, number)
        for number, (kind, position) in enumerate(zip(kinds, ring_positions))
    ] + [station]
    for node1, node2 in zip(nodes, nodes[1:]):
        if node1 is not node2:
            mv_grid.graph.add_edge(node1, node2, branch=BranchDing0(
                length=node1.geo_data.distance(node2.geo_data),
                kind="cable", ring=ring, grid=mv_grid))

    conn_dist_weight = cfg_ding0.get("mv_connect", "load_area_sat_conn_dist_weight")
    conn_dist_ring_mod = cfg_ding0.get("mv_connect", "load_area_sat_conn_dist_ring_mod")
    satellites = []
    for number, (position, peak_load) in enumerate(
            zip(satellite_positions, peak_loads), start=1000):
        load_area = LVLoadAreaDing0(id_db=number, mv_grid_district=mv_grid_district,
                                    peak_load=peak_load, is_satellite=True)
        centre = LVLoadAreaCentreDing0(id_db=number, geo_data=position,
                                       lv_load_area=load_area, grid=mv_grid)
        load_area.lv_load_area_centre = centre
        mv_grid.graph.add_node(centre)
        stack = find_nearest_conn_objects(
            position, list(mv_grid.graph_edges()), None, conn_dist_weight,
            debug=False, branches_only=False)
        find_connection_point(centre, position, mv_grid.graph, None, stack,
                              conn_dist_ring_mod, debug=False)
        satellites.append(centre)

    for centre in satellites:
        assert nx.has_path(mv_grid.graph, station, centre), centre
    cos_phi_load = cfg_ding0.get("assumptions", "cos_phi_load")
    for group in mv_grid_district.lv_load_area_groups():
        if len(list(group.lv_load_areas())) > 1:
            assert group.peak_load / cos_phi_load <= group.peak_load_max


def test_station_finds_branch_whose_ends_are_no_street_nodes():
    """A branch crossing a load area may end at nodes outside it that are no
    nodes of the load area's street graph (e.g. a ring branch between a load
    at a location of its own and a cable distributor); an LV station of the
    load area can still be connected to it."""
    from shapely.geometry import LineString

    from ding0.grid.mv_grid.mv_connect import find_nearest_conn_objects_settle

    NetworkDing0()  # loads the config
    mv_grid = MVGridDing0(id_db=1, station=MVStationDing0(id_db=1, geo_data=Point(0, 0)))

    # street from (0, 0) to (300, 0), the LV station at its start
    class Station:
        geo_data = Point(0, 0)

        def __repr__(self):
            return "LVStation_1"

    station = Station()
    street_graph = nx.MultiDiGraph(crs="epsg:3035")
    street_graph.add_node("LVStation_1", x=0, y=0)
    street_graph.add_node(1, x=300, y=0)
    line = LineString([(0, 0), (300, 0)])
    street_graph.add_edge("LVStation_1", 1, geometry=line, length=line.length)
    street_graph.add_edge(1, "LVStation_1", geometry=LineString(line.coords[::-1]),
                          length=line.length)
    # ring branch from a charging point to a cable distributor, both outside
    # the load area, crossing it above the street
    load = PointLoadDing0(id_db=28, geo_data=Point(150, -400), p_nom=300.0,
                          type="charging_point", v_level=5, mv_grid=mv_grid)
    cable_distributor = MVCableDistributorDing0(geo_data=Point(150, 400), grid=mv_grid)
    branch = {"adj_nodes": (load, cable_distributor),
              "branch": BranchDing0(geometry=LineString([(150, -400), (150, 400)]),
                                    length=800, kind="cable", grid=mv_grid)}

    stack, _ = find_nearest_conn_objects_settle(
        station, [branch], [station], street_graph, {}, conn_dist_weight=1.0,
        debug=False)

    assert stack
    assert street_graph.has_node(str(load)) and street_graph.has_node(str(cable_distributor))
