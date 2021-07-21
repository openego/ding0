from tests.core.network.test_grids import TestMVGridDing0
from ding0.grid.mv_grid import mv_connect
from networkx.readwrite.graphml import read_graphml
import networkx as nx
from networkx.algorithms.isomorphism import categorical_edge_match, numerical_edge_match
import os
import numbers


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
        for kk in ["type", "ring", "branch", "circuit_breaker", "grid"]:
            edge_attrs.pop(kk, None)
        edge_attrs = {k1: v1 for k1, v1 in edge_attrs.items() if v1}
        edges[k] = edge_attrs
        edges[k].pop("branch", None)
    for n1, n2, d in graph.edges(data=True):
        d.pop("branch", None)
    nx.set_edge_attributes(graph, edges)

    expected_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))

    # Snippet to export new comparison graph:
    # graph_new = nx.Graph()
    # graph_new.add_nodes_from([repr(n) for n in graph.nodes()])
    # for n1, n2, a in graph.edges(data=True):
    #     a['grid'] = repr(a['grid'])
    #     graph_new.add_edge(repr(n1), repr(n2), **a)
    # nx.write_graphml(graph_new, os.path.join(expected_file_path, "grid_mv_connect_generators_expected.graphml"))

    # Get comparison data (expected) and compare all attributes
    expected_graph = read_graphml(os.path.join(
        expected_file_path,
        "grid_mv_connect_generators_expected.graphml"))

    # distinct categorical and numeric attrs
    edge_attrs_num = [a for a in edge_attrs if isinstance(list(edges.values())[0].get(a, None), numbers.Number)]
    edge_attrs_cat = [a for a in edge_attrs if a not in edge_attrs_num]

    em_cat = categorical_edge_match(edge_attrs_cat, [0] * len(edge_attrs_cat))
    em_num = numerical_edge_match(edge_attrs_num, [0] * len(edge_attrs_num), rtol=1e-3)

    assert nx.is_isomorphic(graph, expected_graph, edge_match=em_cat)
    assert nx.is_isomorphic(graph, expected_graph, edge_match=em_num)
