import networkx as nx
import pandas as pd
from hypothesis import given, strategies as st
from shapely.geometry import Point

from ding0.grid.lv_grid.build_grid_on_osm_ways import (
    allocate_street_load_nodes,
    get_shortest_path_tree,
    relocate_buildings_with_station_as_nn,
)
from ding0.grid.lv_grid.routing import get_lvgd_id

STATION = 1


def street_graph():
    """Station 1 with a loop to itself, street 1 - 2 - 3."""
    G = nx.MultiDiGraph(crs="epsg:3035")
    for node, x in [(1, 0), (2, 100), (3, 200)]:
        G.add_node(node, x=x, y=0)
    for u, v in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]:
        G.add_edge(u, v, length=100 if u != v else 10)
    return G


def buildings_at_station(n):
    return pd.DataFrame(
        {"nn": [STATION] * n, "nn_dist": [5.0] * n,
         "nn_coords": [Point(0, 0)] * n, "geometry": [Point(10, 5)] * n,
         "capacity": [10.0] * n, "number_households": [1] * n},
        index=[f"building_{i}" for i in range(n)])


def test_tree_of_several_buildings_at_the_station_holds_the_station():
    tree = get_shortest_path_tree(street_graph(), STATION, [STATION, STATION])

    allocate_street_load_nodes(buildings_at_station(2), tree.copy(), STATION)
    assert STATION in tree


def test_no_cable_from_the_station_to_itself():
    # issue #406
    tree = get_shortest_path_tree(street_graph(), STATION, [STATION])

    assert list(nx.selfloop_edges(tree)) == []


def test_buildings_at_the_station_move_to_a_street_node_next_to_it():
    buildings = relocate_buildings_with_station_as_nn(
        street_graph(), STATION, buildings_at_station(2))

    assert set(buildings.nn) == {2}


@given(st.integers(1, 224497), st.integers(0, 9999),
       st.integers(1, 224497), st.integers(0, 9999))
def test_lv_grid_districts_of_different_clusters_get_different_ids(la1, c1, la2, c2):
    if (la1, c1) != (la2, c2):
        assert get_lvgd_id(la1, c1) != get_lvgd_id(la2, c2)


def test_lv_grid_districts_of_load_areas_92_and_920_get_different_ids():
    assert get_lvgd_id(92, 0) != get_lvgd_id(920, 0)
