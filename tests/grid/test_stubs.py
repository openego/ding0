"""Stubs of the urban MV routing (create_stub_dict).

A stub is a part of the street graph that hangs off the routed MV rings and
leads to loads. It is connected to the rings at its root.
"""
from types import SimpleNamespace

import networkx as nx
from hypothesis import given
from hypothesis import strategies as st

from ding0.core import NetworkDing0
from ding0.grid.mv_grid.tools import create_stub_dict

NetworkDing0()  # loads the config


def street_graph(edges):
    """Street graph with both directions of the given edges, 100 m each."""
    graph = nx.MultiDiGraph()
    for u, v in edges:
        graph.add_edge(u, v, length=100)
        graph.add_edge(v, u, length=100)
    return graph


def test_root_of_stub_touching_the_rings_twice_is_next_to_its_load():
    """A stub may touch the routed rings at a station and at a street node of
    a ring branch; its root is the one the load hangs off (district 36315)."""
    stub_graph = street_graph([("LVStation_36", "250961441"),
                               ("250961441", "LVStation_83")])
    node_list = {"LVStation_36": SimpleNamespace(peak_load=300),
                 "LVStation_83": SimpleNamespace(peak_load=180)}

    (stub,) = create_stub_dict(
        stub_graph, {"LVStation_36", "250961441"}, node_list).values()

    assert stub["root"] == "250961441"
    assert set(stub["load"]) == {"LVStation_83"}


@given(st.data())
def test_stubs_hang_off_a_single_ring_node(data):
    """Whatever nodes of a component of the stub graph lie on the routed
    rings, it is split into stubs whose loads hang off one of them, their
    root, and are reached from it within the stub; every load is in exactly
    one stub."""
    n_nodes = data.draw(st.integers(2, 9), label="nodes")
    # a tree: each node hangs off one of the nodes before it
    parents = [data.draw(st.integers(0, i - 1), label=f"parent of {i}")
               for i in range(1, n_nodes)]
    names = data.draw(st.permutations([f"n{i}" for i in range(n_nodes)]),
                      label="names")
    stub_graph = street_graph(
        [(names[parent], names[i]) for i, parent in enumerate(parents, start=1)])
    roots = set(data.draw(st.lists(st.sampled_from(names), min_size=1, unique=True),
                          label="nodes on the rings"))
    # ding0 nodes (LV stations, MV loads), some of them on the rings
    ding0_nodes = data.draw(st.lists(st.sampled_from(names), unique=True),
                            label="ding0 nodes")
    node_list = {name: SimpleNamespace(peak_load=100) for name in ding0_nodes}
    loads = set(ding0_nodes) - roots

    stubs = list(create_stub_dict(stub_graph, roots, node_list).values())

    # stubs without loads are dropped later (check_stub_criterion)
    for stub in (stub for stub in stubs if stub["load"]):
        assert stub["comp"] & roots == {stub["root"]}
        comp_graph = nx.Graph(stub_graph.subgraph(stub["comp"]))
        for load in stub["load"]:
            assert nx.has_path(comp_graph, stub["root"], load)
    stub_loads = [load for stub in stubs for load in stub["load"]]
    assert sorted(stub_loads) == sorted(loads)


def test_split_stubs_are_added_when_all_stubs_were_removed():
    """check_stub_criterion may remove every stub and split them into new
    ones (district 36348); these are the stubs then."""
    from ding0.grid.mv_grid.tools import update_stub_dict

    node_list = {"LVStation_1": SimpleNamespace(peak_load=500),
                 "LVStation_2": SimpleNamespace(peak_load=500)}

    stub_dict = update_stub_dict({}, [["1", "LVStation_1"], ["2", "LVStation_2"]],
                                 node_list)

    assert sorted(stub["root"] for stub in stub_dict.values()) == ["1", "2"]
