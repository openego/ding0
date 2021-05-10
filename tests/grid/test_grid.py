from tests.core.network.test_grids import TestMVGridDing0
from ding0.grid.mv_grid import mv_connect
from networkx.readwrite.graphml import read_graphml
import networkx as nx
from networkx.algorithms.isomorphism import categorical_edge_match
import os


def test_mv_connect_generators():
    """Test connection of MV generators

    Verifies that lines are chosen and connected correctly by testing the
    graph's isomorphism. This include the lines attributes.
    """

    # Ding0's first steps
    nd, mv_grid, lv_stations = \
        TestMVGridDing0().minimal_unrouted_testgrid()
    nd.mv_routing()
    graph = mv_connect.mv_connect_generators(
        nd._mv_grid_districts[0],
        nd._mv_grid_districts[0].mv_grid._graph)

    # Reorganize edges attributes
    edges = nx.get_edge_attributes(graph, 'branch')
    for k, v in edges.items():
        edge_attrs = v.__dict__
        edge_attrs.update(**edge_attrs["type"].to_dict())
        for kk in ["type", "ring", "branch", "circuit_breaker"]:
            edge_attrs.pop(kk, None)
        edge_attrs = {k1: v1 for k1, v1 in edge_attrs.items() if v1}
        edges[k] = edge_attrs
        edges[k].pop("branch", None)
    for n1, n2, d in graph.edges(data=True):
        d.pop("branch", None)
    nx.set_edge_attributes(graph, edges)

    # Get comparison data (expected) and compare all attributes
    expected_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))
    expected_graph = read_graphml(os.path.join(
        expected_file_path,
        "grid_mv_connect_generators_expected.graphml"))
    attr_names = list(edge_attrs.keys())
    em = categorical_edge_match(attr_names, [0] * len(attr_names))
    assert nx.is_isomorphic(graph, expected_graph, edge_match=em)
