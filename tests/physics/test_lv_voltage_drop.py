"""Voltage drop in LV feeders: ding0's check against the analytic formula.

ding0 sizes the cable of an LV feeder by current and then upsizes it until
`check_voltage_drop_in_feeder` reports a voltage drop within
`lv_max_v_level_lc_diff_normal` (`cable_type_by_voltage`). The reference
here is the usual approximation for radial feeders,

    dU = sum_k P_k * (R_0k + X_0k * tan(phi)) / U_n,

with R_0k and X_0k the resistance and reactance (X = 2 pi f L) of the path
from the station to the point where load k leaves the path to the node
considered. The first test checks this reference against a power flow.
"""

from math import acos, pi, tan

import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import pytest

from ding0.grid.lv_grid.build_grid_on_osm_ways import (
    cable_type_by_load,
    cable_type_by_voltage,
    check_voltage_drop_in_feeder,
    transform_feeder_to_aggregated_path,
)
from ding0.tools import config as cfg_ding0

OMEGA = 2 * pi * 50


def tan_phi():
    return tan(acos(cfg_ding0.get("assumptions", "cos_phi_load")))


def analytic_drop(graph, root, cable, u_n=400.0):
    """Relative voltage drop at every node of a radial feeder (tree with edge
    attribute `length` in m, node attribute `load` in W)."""
    tree = nx.bfs_tree(graph, root)
    z = (cable.R_per_km + OMEGA * cable.L_per_km * 1e-3 * tan_phi()) / 1e3  # ohm per m
    downstream = {n: graph.nodes[n]["load"] + sum(graph.nodes[d]["load"] for d in nx.descendants(tree, n))
                  for n in tree}
    drop = {root: 0.0}
    for parent, child in nx.bfs_edges(tree, root):
        drop[child] = drop[parent] + downstream[child] * z * graph.edges[parent, child]["length"] / u_n
    return {n: d / u_n for n, d in drop.items()}


def chain(lengths, loads):
    """Path station (node 0) - 1 - 2 ... with edge lengths (m) and node loads (W)."""
    graph = nx.Graph()
    graph.add_node(0, load=0)
    for i, (length, load) in enumerate(zip(lengths, loads), start=1):
        graph.add_node(i, load=load)
        graph.add_edge(i - 1, i, length=length)
    return graph


def as_ding0_feeder(graph):
    """The feeder graph ding0 builds (MultiDiGraph from the street graph)."""
    feeder = nx.MultiDiGraph()
    feeder.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        feeder.add_edge(u, v, 0, **data)
    return feeder


def power_flow_drop(graph, root, cable):
    network = pypsa.Network()
    for node in graph.nodes:
        network.add("Bus", str(node), v_nom=0.4)
    network.add("Generator", "slack", bus=str(root), control="Slack")
    for u, v, data in graph.edges(data=True):
        km = data["length"] / 1e3
        network.add("Line", f"{u}-{v}", bus0=str(u), bus1=str(v), r=cable.R_per_km * km,
                    x=OMEGA * cable.L_per_km * 1e-3 * km, s_nom=1)
    for node, data in graph.nodes(data=True):
        if data["load"]:
            p = data["load"] / 1e6
            network.add("Load", f"load_{node}", bus=str(node), p_set=p, q_set=p * tan_phi())
    network.pf()
    v = network.buses_t.v_mag_pu.iloc[0]
    return {n: 1 - v[str(n)] for n in graph.nodes}


def test_analytic_reference_matches_power_flow(static_data):
    """The analytic voltage drop agrees with a power flow within 0.3
    percentage points for a typical LV feeder (six houses, 5 % drop)."""
    cable = static_data["LV_cables"].loc["NAYY 4x1x150"]
    graph = chain([60, 40, 40, 40, 40, 40], [15e3, 20e3, 15e3, 25e3, 20e3, 30e3])
    analytic = analytic_drop(graph, 0, cable)
    flow = power_flow_drop(graph, 0, cable)
    assert max(analytic.values()) == pytest.approx(0.05, abs=0.02)
    for node in graph.nodes:
        assert analytic[node] == pytest.approx(flow[node], abs=0.003)


@pytest.mark.xfail(strict=True, reason=(
    "B4: check_voltage_drop_in_feeder takes the inductance L' (mH/km) as "
    "reactance (ohm/km) instead of 2 pi f L'"))
def test_feeder_below_the_limit_is_not_critical(static_data, lvgd_cfg):
    """A feeder with a voltage drop of 4.5 % (limit 5 %) is not critical."""
    cable = static_data["LV_cables"].loc["NAYY 4x1x150"]
    z = cable.R_per_km + OMEGA * cable.L_per_km * 1e-3 * tan_phi()  # ohm/km
    load = 0.045 * 400 ** 2 / (z * 0.3)  # W at 300 m for 4.5 %
    graph = chain([300], [load])
    assert analytic_drop(graph, 0, cable)[1] == pytest.approx(0.045)
    feeder_agg, path = transform_feeder_to_aggregated_path(as_ding0_feeder(graph), 0)
    assert not check_voltage_drop_in_feeder(feeder_agg, path, cable, lvgd_cfg)


def test_feeder_above_the_limit_is_critical(static_data, lvgd_cfg):
    """A feeder with a voltage drop of 5.5 % (limit 5 %) is critical."""
    cable = static_data["LV_cables"].loc["NAYY 4x1x150"]
    z = cable.R_per_km + OMEGA * cable.L_per_km * 1e-3 * tan_phi()
    load = 0.055 * 400 ** 2 / (z * 0.3)
    feeder_agg, path = transform_feeder_to_aggregated_path(as_ding0_feeder(chain([300], [load])), 0)
    assert check_voltage_drop_in_feeder(feeder_agg, path, cable, lvgd_cfg)


@pytest.mark.xfail(strict=True, reason=(
    "B5: for loads connected directly to the LV station "
    "build_branches_on_osm_ways sets the load in kW, the voltage check "
    "expects W, so the cable is not upsized"))
def test_direct_station_connection_keeps_voltage_limit(static_data, lvgd_cfg):
    """A 150 kW building 300 m from the station, connected by a feeder of
    its own (lv_threshold_capacity <= capacity < mv_lv_threshold_capacity),
    gets a cable with a voltage drop within the limit.

    The feeder graph is built as in `build_branches_on_osm_ways` (block
    `lv_loads_to_station`): two nodes, the building's load set to its
    capacity in kW."""
    capacity_kw = 150.0
    graph = nx.MultiDiGraph()
    graph.add_edge("station", "building", 0, length=300.0)
    nx.set_node_attributes(graph, {"station": {"load": 0}, "building": {"load": capacity_kw}})
    cable = cable_type_by_load(capacity_kw, lvgd_cfg)
    cable = cable_type_by_voltage(graph, "station", cable, lvgd_cfg)

    reference = nx.Graph()
    reference.add_node("station", load=0)
    reference.add_node("building", load=capacity_kw * 1e3)
    reference.add_edge("station", "building", length=300.0)
    drop = analytic_drop(reference, "station", cable)["building"]
    assert drop <= lvgd_cfg["v_diff_max"], f"{cable.name}: {drop:.1%}"


@pytest.mark.xfail(strict=True, reason=(
    "B18: check_voltage_drop_in_feeder only checks the longest path of a "
    "feeder; a shorter but heavily loaded branch is not checked"))
def test_heavily_loaded_short_branch_keeps_voltage_limit(static_data, lvgd_cfg):
    """A feeder forks 10 m from the station: a long branch (4 x 100 m) with
    small loads and a shorter one (350 m) with a 60 kW load at its end. The
    cable ding0 chooses keeps the voltage drop at every node within the
    limit."""
    graph = nx.Graph()
    graph.add_node("S", load=0)
    graph.add_node("A", load=0)
    graph.add_edge("S", "A", length=10.0)
    previous = "A"
    for i in range(4):
        graph.add_node(f"B{i}", load=1e3)
        graph.add_edge(previous, f"B{i}", length=100.0)
        previous = f"B{i}"
    graph.add_node("C", load=60e3)
    graph.add_edge("A", "C", length=350.0)

    feeder = as_ding0_feeder(nx.bfs_tree(graph, "S").to_undirected())
    for u, v in graph.edges:
        key = (u, v) if feeder.has_edge(u, v) else (v, u)
        feeder.edges[key[0], key[1], 0]["length"] = graph.edges[u, v]["length"]
    for n, data in graph.nodes(data=True):
        feeder.nodes[n]["load"] = int(data["load"])
    total_kw = sum(d["load"] for _, d in graph.nodes(data=True)) / 1e3
    cable = cable_type_by_voltage(feeder, "S", cable_type_by_load(total_kw, lvgd_cfg), lvgd_cfg)

    drops = analytic_drop(graph, "S", cable)
    worst = max(drops, key=drops.get)
    assert drops[worst] <= lvgd_cfg["v_diff_max"], f"{cable.name}: {drops[worst]:.1%} at {worst}"
