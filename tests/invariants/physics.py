"""Physical invariants of ding0 grids.

Exported grids (`components`, see `powerflow.export_mv_grid_district`),
solved networks (`powerflow.solve`) and ding0's object model are checked.
Units: exported power in MW/MVA, line r and x in ohm, length in km;
ding0's objects use kW/kVA and m.
"""

import re
from math import pi, sqrt

import networkx as nx
import numpy as np
import pandas as pd

from ding0.core.network import (
    CircuitBreakerDing0,
    GeneratorDing0,
    LoadDing0,
)
from ding0.core.network.components import GridComponentDing0, PointLoadDing0
from ding0.core.network.loads import LVLoadDing0
from ding0.core.network.stations import LVStationDing0
from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.tools import config as cfg_ding0

from . import powerflow

KCL_TOLERANCE = 1e-6  # MVA


def _get(section, key):
    return cfg_ding0.get(section, key)


# Equipment data ##############################################################

def cross_section(name):
    """Conductor cross-section in mm2 from a cable or line type name, e.g.
    185 from 'NA2XS2Y 3x1x185 RM/25', 48 from '48-AL1/8-ST1A'."""
    match = re.search(r"x1x(\d+)", name) or re.match(r"(\d+)-AL", name)
    return int(match.group(1))


def check_equipment_catalog(static_data, tolerance=0.15):
    """Equipment data are physically plausible.

    Within a family of conductors (same material and voltage) the resistance
    per km is inversely proportional to the cross-section, R' * A = rho
    (within `tolerance` of the family's median), and resistance falls and
    ampacity rises with the cross-section. Transformers: short-circuit
    losses 0.5 % to 2.5 % of the rated power, u_k 3.5 % to 8 %.
    """
    violations = []
    families = [("LV_cables", static_data["LV_cables"].reset_index())]
    for key in ["MV_cables", "MV_overhead_lines"]:
        for u_n, df in static_data[key].groupby("U_n"):
            families.append((f"{key} {u_n} kV", df))
    for family, df in families:
        df = df.assign(A=df["name"].apply(cross_section)).sort_values("A")
        rho = df.R_per_km * df.A
        median = rho.median()
        for _, row in df.assign(rho=rho).iterrows():
            if abs(row.rho / median - 1) > tolerance:
                violations.append(
                    f"{family} {row['name']}: R'*A = {row.rho:.1f} ohm*mm2/km, "
                    f"family median {median:.1f} (R' = {row.R_per_km} ohm/km)")
        if not df.R_per_km.is_monotonic_decreasing:
            violations.append(f"{family}: R' does not fall with the cross-section")
        if not df.I_max_th.is_monotonic_increasing:
            violations.append(f"{family}: I_max_th does not rise with the cross-section")
    for key in ["LV_trafos"]:
        df = static_data[key].sort_values("S_nom")
        p_k_share = df.P_k / (df.S_nom * 1e3)
        for (_, row), share in zip(df.iterrows(), p_k_share):
            if not 0.005 <= share <= 0.025:
                violations.append(f"{key} {row.S_nom} kVA: P_k is {share:.2%} of S_nom")
            if not 3.5 <= row.u_kr <= 8:
                violations.append(f"{key} {row.S_nom} kVA: u_k = {row.u_kr} %")
    return violations


def check_transformer_data(static_data):
    """r_pu = P_k / S_nom and x_pu = sqrt(u_k^2 - r_pu^2) of LV transformers."""
    violations = []
    for _, t in static_data["LV_trafos"].iterrows():
        r_pu = t.P_k / (t.S_nom * 1e3)
        x_pu = sqrt((t.u_kr / 100) ** 2 - r_pu ** 2)
        if not (np.isclose(t.r_pu, r_pu) and np.isclose(t.x_pu, x_pu)):
            violations.append(f"LV transformer {t.S_nom} kVA: r_pu {t.r_pu}, x_pu {t.x_pu}, "
                              f"expected {r_pu}, {x_pu}")
    return violations


# Exported grid ###############################################################

def _catalog_row(static_data, kind, type_info, v_nom):
    if v_nom < 1:
        catalog = static_data["LV_cables"]
        return catalog.loc[type_info] if type_info in catalog.index else None
    catalog = static_data["MV_cables" if kind == "cable" else "MV_overhead_lines"]
    rows = catalog.loc[(catalog["name"] == type_info) & (catalog.U_n == round(v_nom))]
    return rows.iloc[0] if len(rows) else None


def check_line_impedances(components, static_data, rtol=1e-4):
    """r, x and s_nom of every exported line follow from its type and length:
    r = R' * l / n, x = 2 pi f L' * l / n, s_nom = sqrt(3) * U_n * I_max * n
    with n parallel lines (PyPSA uses r, x and s_nom as given when no
    standard type is set)."""
    omega = 2 * pi * _get("assumptions", "frequency")
    buses = components["Bus"]
    violations = []
    for name, line in components["Line"].iterrows():
        v_nom = buses.at[line.bus0, "v_nom"]
        row = _catalog_row(static_data, line.kind, line.type_info, v_nom)
        if row is None:
            violations.append(f"{name}: type {line.type_info!r} ({line.kind}, {v_nom} kV) "
                              f"not in the equipment catalog")
            continue
        n = line.num_parallel
        u_n_kv = row.U_n / 1e3 if v_nom < 1 else row.U_n
        expected = {"r": row.R_per_km * line.length / n,
                    "x": omega * row.L_per_km * 1e-3 * line.length / n,
                    "s_nom": sqrt(3) * u_n_kv * row.I_max_th / 1e3 * n}
        for attr, value in expected.items():
            if not np.isclose(line[attr], value, rtol=rtol):
                violations.append(f"{name} ({line.type_info}, {n} parallel, "
                                  f"{line.length:.3f} km): {attr} = {line[attr]:.6g}, "
                                  f"expected {value:.6g}")
    return violations


def check_voltage_levels(components, mv_v_level):
    """MV buses at the grid's voltage level, LV buses at 0.4 kV, lines
    between buses of one voltage, transformers from the higher to the lower
    voltage."""
    buses = components["Bus"]
    lv_v_nom = _get("assumptions", "lv_nominal_voltage") / 1e3
    violations = [f"bus {name}: v_nom {v} kV"
                  for name, v in buses.v_nom.items()
                  if not (np.isclose(v, mv_v_level) or np.isclose(v, lv_v_nom))]
    for name, line in components["Line"].iterrows():
        v0, v1 = buses.at[line.bus0, "v_nom"], buses.at[line.bus1, "v_nom"]
        if v0 != v1:
            violations.append(f"line {name} connects {v0} kV and {v1} kV")
    for name, trafo in components["Transformer"].iterrows():
        v0, v1 = buses.at[trafo.bus0, "v_nom"], buses.at[trafo.bus1, "v_nom"]
        if not (np.isclose(v0, mv_v_level) and np.isclose(v1, lv_v_nom)):
            violations.append(f"transformer {name}: bus0 {v0} kV, bus1 {v1} kV")
    return violations


def export_graph(components):
    """Graph of the exported buses, lines and MV/LV transformers."""
    graph = nx.MultiGraph()
    graph.add_nodes_from(components["Bus"].index)
    for key in ["Line", "Transformer"]:
        for name, row in components[key].iterrows():
            graph.add_edge(row.bus0, row.bus1, key=name)
    return graph


def check_radial(components):
    """With the switch disconnectors open (normal operation) the grid is a
    single tree: connected, without meshes and without parallel edges."""
    graph = export_graph(components)
    violations = []
    parts = list(nx.connected_components(graph))
    if len(parts) > 1:
        small = sorted(parts, key=len)[:-1]
        violations.append(f"{len(parts)} connected components; buses outside the "
                          f"largest: {sorted(b for part in small for b in part)[:10]}")
    cycles = nx.cycle_basis(nx.Graph(graph))
    if cycles:
        violations.append(f"{len(cycles)} meshes, e.g. {cycles[0][:6]}")
    for u, v, k in graph.edges(keys=True):
        if graph.number_of_edges(u, v) > 1:
            violations.append(f"parallel edges between {u} and {v}")
            break
    return violations


def check_switches(components):
    """Every open switch disconnector separates its line at a bus of its own
    (`bus_open`), which only that line reaches."""
    lines = components["Line"]
    buses = components["Bus"]
    violations = []
    for name, switch in components["Switch"].iterrows():
        at_open = lines.loc[(lines.bus0 == switch.bus_open) | (lines.bus1 == switch.bus_open)]
        if switch.bus_open not in buses.index or switch.bus_closed not in buses.index:
            violations.append(f"switch {name}: bus missing")
        elif list(at_open.index) != [switch.branch]:
            violations.append(f"switch {name}: open bus reached by {list(at_open.index)}")
    return violations


# Solved network ##############################################################

def check_converged(converged):
    return [f"power flow did not converge in snapshot {s}"
            for s, ok in converged.items() if not ok]


def check_kirchhoff(network, tolerance=KCL_TOLERANCE):
    """Kirchhoff's current law and Ohm's law: at every bus the power flowing
    into the lines and transformers, computed from the solved voltages and
    the exported impedances (S = V conj(Y V)), equals the net injection of
    the loads and generators."""
    y = powerflow.admittance_matrix(network).values
    violations = []
    for snapshot in network.snapshots:
        v = powerflow.complex_voltages(network, snapshot)
        s_flow = v.values * np.conj(y @ v.values)
        residual = pd.Series(s_flow, index=v.index) - powerflow.nodal_injections(network, snapshot)
        for bus, value in residual[residual.abs() > tolerance].items():
            violations.append(f"snapshot {snapshot}, bus {bus}: KCL residual {abs(value):.3g} MVA")
    return violations


def check_losses(network, max_share=0.1):
    """Line and transformer losses are not negative and below `max_share` of
    the power drawn at the slack."""
    violations = []
    for snapshot in network.snapshots:
        branch_losses = pd.concat([
            network.lines_t.p0.loc[snapshot] + network.lines_t.p1.loc[snapshot],
            network.transformers_t.p0.loc[snapshot] + network.transformers_t.p1.loc[snapshot]])
        for name, loss in branch_losses[branch_losses < -1e-9].items():
            violations.append(f"snapshot {snapshot}, {name}: negative losses {loss:.3g} MW")
        slack = network.generators.index[network.generators.control == "Slack"]
        throughput = abs(network.generators_t.p.loc[snapshot, slack].sum())
        if throughput > 1e-6 and branch_losses.sum() > max_share * throughput:
            violations.append(f"snapshot {snapshot}: losses {branch_losses.sum():.3f} MW are more "
                              f"than {max_share:.0%} of {throughput:.3f} MW at the slack")
    return violations


def check_energy_balance(network, tolerance=1e-6):
    """Slack power = loads - generation + losses (active power)."""
    violations = []
    for snapshot in network.snapshots:
        p = network.generators_t.p.loc[snapshot]
        slack = network.generators.control == "Slack"
        losses = (network.lines_t.p0.loc[snapshot] + network.lines_t.p1.loc[snapshot]).sum() + \
            (network.transformers_t.p0.loc[snapshot] + network.transformers_t.p1.loc[snapshot]).sum()
        loads = network.loads_t.p.loc[snapshot].sum() if not network.loads.empty else 0.0
        balance = p[slack].sum() - (loads - p[~slack].sum() + losses)
        if abs(balance) > tolerance:
            violations.append(f"snapshot {snapshot}: energy balance off by {balance:.3g} MW")
    return violations


def _lv_busbars(network):
    """LV busbar of each LV grid (bus1 of its station transformers)."""
    buses = network.buses
    return {buses.at[t.bus1, "lv_grid_id"]: t.bus1 for _, t in network.transformers.iterrows()}


def check_voltage_band(network, levels=("MV", "LV")):
    """Voltage deviations within the planning limits: MV buses against the
    HV/MV station (load case `mv_max_v_level_lc_diff_normal`, feed-in case
    `mv_max_v_level_fc_diff_normal`), LV buses against their station's LV
    busbar (drop `lv_max_v_level_lc_diff_normal`, rise
    `lv_max_v_level_fc_diff_normal`)."""
    limits = {
        "MV": [_get("mv_routing_tech_constraints", "mv_max_v_level_lc_diff_normal"),
               _get("mv_routing_tech_constraints", "mv_max_v_level_fc_diff_normal")],
        "LV": [_get("assumptions", "lv_max_v_level_lc_diff_normal"),
               _get("assumptions", "lv_max_v_level_fc_diff_normal")],
    }
    buses = network.buses
    slack_bus = network.generators.loc[network.generators.control == "Slack", "bus"].iloc[0]
    busbars = _lv_busbars(network)
    violations = []
    for snapshot, case in zip(network.snapshots, ["load case", "feed-in case"]):
        v = network.buses_t.v_mag_pu.loc[snapshot]
        if "MV" in levels:
            mv = buses.index[buses.lv_grid_id.isna()]
            dv = (v[mv] - v[slack_bus]).abs()
            limit = limits["MV"][snapshot]
            for bus, d in dv[dv > limit + 1e-9].items():
                violations.append(f"{case}, MV bus {bus}: deviation {d:.2%} > {limit:.0%}")
        if "LV" in levels:
            limit = limits["LV"][snapshot]
            for lv_grid_id, group in buses.loc[buses.lv_grid_id.notna()].groupby("lv_grid_id"):
                ref = v[busbars[lv_grid_id]] if lv_grid_id in busbars else None
                if ref is None:
                    continue
                dv = (ref - v[group.index]) if snapshot == powerflow.LOAD_CASE \
                    else (v[group.index] - ref)
                worst = dv.idxmax()
                if dv[worst] > limit + 1e-9:
                    violations.append(f"{case}, LV grid {int(lv_grid_id)}: deviation "
                                      f"{dv[worst]:.2%} > {limit:.0%} at {worst}")
    return violations


def check_thermal_limits(network, tolerance=0.05):
    """Apparent power of every line and MV/LV transformer below its rating
    times the load factor of the planning profile (per kind, voltage and
    case). `tolerance` allows for the higher current at voltages below
    nominal within the permitted band."""
    get = _get
    buses = network.buses
    factor = {
        ("MV", "line"): [get("assumptions", "load_factor_mv_line_lc_normal"),
                         get("assumptions", "load_factor_mv_line_fc_normal")],
        ("MV", "cable"): [get("assumptions", "load_factor_mv_cable_lc_normal"),
                          get("assumptions", "load_factor_mv_cable_fc_normal")],
        ("LV", "cable"): [get("assumptions", "load_factor_lv_cable_lc_normal"),
                          get("assumptions", "load_factor_lv_cable_fc_normal")],
        ("LV", "trafo"): [get("assumptions", "load_factor_lv_trans_lc_normal"),
                          get("assumptions", "load_factor_lv_trans_fc_normal")],
    }
    violations = []
    for snapshot, case in zip(network.snapshots, ["load case", "feed-in case"]):
        for key, df, df_t in [("Line", network.lines, network.lines_t),
                              ("Transformer", network.transformers, network.transformers_t)]:
            s = np.maximum(np.hypot(df_t.p0.loc[snapshot], df_t.q0.loc[snapshot]),
                           np.hypot(df_t.p1.loc[snapshot], df_t.q1.loc[snapshot]))
            for name, row in df.iterrows():
                level = "MV" if buses.at[row.bus0, "v_nom"] > 1 else "LV"
                kind = "trafo" if key == "Transformer" else row.kind
                lf = factor.get((level, kind))
                if lf is None:
                    continue
                loading = s[name] / row.s_nom
                if loading > lf[snapshot] * (1 + tolerance):
                    violations.append(f"{case}, {name}: loading {loading:.0%} > "
                                      f"{lf[snapshot]:.0%}")
    return violations


def check_set_points(components, data, rtol=1e-6):
    """ding0's own power flow set points equal the ones derived from the
    exported nominal values (every load and generator once, same p and q)."""
    expected = powerflow.set_points(components)
    actual = powerflow.ding0_set_points(data)
    violations = []
    for key in ["p_load", "q_load", "p_gen", "q_gen"]:
        exp, act = expected[key], actual[key]
        missing = sorted(set(exp.columns) - set(act.columns))
        extra = sorted(set(act.columns) - set(exp.columns))
        if missing:
            violations.append(f"{key}: no ding0 set point for {missing[:5]} ({len(missing)})")
        if extra:
            violations.append(f"{key}: ding0 set points without component {extra[:5]}")
        common = exp.columns.intersection(act.columns)
        diff = ~np.isclose(act[common].astype(float), exp[common].astype(float), rtol=rtol, atol=1e-9)
        for name in common[diff.any(axis=0)]:
            violations.append(f"{key} {name}: ding0 {list(act[name].round(6))}, "
                              f"expected {list(exp[name].round(6))}")
    return violations


# Object model ################################################################

def check_ding0_powerflow_results(mv_grid):
    """After ding0's power flow every MV node has a voltage and every MV
    branch an apparent power for both cases."""
    violations = []
    for node in mv_grid.graph.nodes():
        if isinstance(node, (CircuitBreakerDing0, LVLoadAreaCentreDing0)) or \
                mv_grid.graph.degree(node) == 0:
            continue
        v = getattr(node, "voltage_res", None)
        if v is None or len(v) != 2:
            violations.append(f"{node}: no voltage result")
    for edge in mv_grid.graph_edges():
        s = getattr(edge["branch"], "s_res", None)
        if s is None or len(s) != 2:
            violations.append(f"{edge['branch']}: no power result")
    return violations


def check_export_keeps_switch_state(mv_grid_district):
    """Exporting a grid does not change it: switch disconnectors open before
    the export are open afterwards (ding0 runs its power flows on exports)."""
    mv_grid = mv_grid_district.mv_grid
    mv_grid.open_circuit_breakers()
    edges_before = mv_grid.graph.number_of_edges()
    powerflow.export_mv_grid_district(mv_grid_district, only_export_mv=True)
    violations = [f"{cb} is {cb.status} after the export"
                  for cb in mv_grid.circuit_breakers() if cb.status != "open"]
    if mv_grid.graph.number_of_edges() != edges_before:
        violations.append(f"{mv_grid}: {mv_grid.graph.number_of_edges() - edges_before} "
                          f"edges more after the export")
    return violations


def check_ding0_results_match(mv_grid, network, v_tol=0.002, s_rtol=0.02, s_atol=0.01):
    """ding0's own MV power flow results (`voltage_res`, `s_res`), on which
    its reinforcement decides, agree with an independent power flow of the
    same MV grid in normal operation (open switch disconnectors, LV grids
    aggregated at their stations: `powerflow.solve(..., only_export_mv=True)`).
    ding0 stores the voltage of the LV busbar at MV/LV stations."""
    v = network.buses_t.v_mag_pu
    s = np.maximum(np.hypot(network.lines_t.p0, network.lines_t.q0),
                   np.hypot(network.lines_t.p1, network.lines_t.q1))
    violations = []
    for node in mv_grid.graph.nodes():
        result = getattr(node, "voltage_res", None)
        bus = getattr(node, "pypsa_bus_id", None)
        if result is None or bus not in v.columns:
            continue
        if np.abs(np.array(result) - v[bus].values).max() > v_tol:
            violations.append(f"{node}: ding0 voltage {np.round(result, 4).tolist()}, "
                              f"independent {np.round(v[bus].values, 4).tolist()}")
    for edge in mv_grid.graph_edges():
        branch = edge["branch"]
        result = getattr(branch, "s_res", None)
        if result is None or repr(branch) not in s.columns:
            continue
        mine = s[repr(branch)].values
        if not np.allclose(result, mine, rtol=s_rtol, atol=s_atol):
            violations.append(f"{branch}: ding0 {np.round(result, 3).tolist()} MVA, "
                              f"independent {np.round(mine, 3).tolist()} MVA")
    return violations


def check_ding0_mv_limits(mv_grid):
    """ding0's own MV power flow results after reinforcement within the
    voltage band and the thermal limits (with parallel lines)."""
    get = _get
    violations = []
    v_station = mv_grid.station().voltage_res
    limits = [get("mv_routing_tech_constraints", "mv_max_v_level_lc_diff_normal"),
              get("mv_routing_tech_constraints", "mv_max_v_level_fc_diff_normal")]
    for node in mv_grid.graph.nodes():
        v = getattr(node, "voltage_res", None)
        if v is None:
            continue
        for case, limit in enumerate(limits):
            if abs(v[case] - v_station[case]) > limit + 1e-9:
                violations.append(f"{node}: voltage deviation {abs(v[case] - v_station[case]):.2%} "
                                  f"> {limit:.0%} (case {case})")
    for edge in mv_grid.graph_edges():
        branch = edge["branch"]
        if getattr(branch, "s_res", None) is None:
            continue
        lf = [get("assumptions", f"load_factor_mv_{branch.kind}_lc_normal"),
              get("assumptions", f"load_factor_mv_{branch.kind}_fc_normal")]
        s_max = sqrt(3) * branch.type["U_n"] * branch.type["I_max_th"] * branch.num_parallel / 1e3
        for case in range(2):
            if branch.s_res[case] > s_max * lf[case] + 1e-9:
                violations.append(f"{branch}: {branch.s_res[case]:.3f} MVA > "
                                  f"{lf[case]:.0%} of {s_max:.3f} MVA (case {case})")
    return violations


def _node_load_kw(node):
    """Peak load (kW) a node draws for grid dimensioning."""
    if isinstance(node, LVLoadDing0):
        return node.peak_load
    if isinstance(node, PointLoadDing0) and not node.building_id:
        # point loads assigned to a building are part of its peak load
        return node.peak_load
    return 0.0


def _node_generation_kw(node):
    if isinstance(node, GeneratorDing0):
        return node.capacity
    return 0.0


def lv_downstream(lv_grid):
    """Tree of an LV grid directed away from the station with the peak load
    and generation (kW) below every node (including the node)."""
    tree = nx.bfs_tree(lv_grid.graph, lv_grid.station())
    load, generation = {}, {}
    for node in reversed(list(nx.topological_sort(tree))):
        load[node] = _node_load_kw(node) + sum(load[c] for c in tree.successors(node))
        generation[node] = _node_generation_kw(node) + sum(generation[c] for c in tree.successors(node))
    return tree, load, generation


def check_lv_tree(lv_grid):
    """An LV grid is a tree containing its station and every node."""
    graph = lv_grid.graph
    violations = []
    if not nx.is_connected(graph):
        parts = sorted(nx.connected_components(graph), key=len)
        violations.append(f"{lv_grid}: {len(parts)} parts, e.g. {sorted(map(repr, parts[0]))[:5]}")
    elif not nx.is_tree(graph):
        violations.append(f"{lv_grid}: has meshes")
    return violations


def check_lv_cable_sizing(lv_grid):
    """Every LV cable carries the peak load and the generation below it
    (I = S / (sqrt(3) U_n)) within its ampacity times the load factor."""
    get = _get
    v_nom = get("assumptions", "lv_nominal_voltage") / 1e3
    cos_load, cos_gen = get("assumptions", "cos_phi_load"), get("assumptions", "cos_phi_gen")
    lf = [get("assumptions", "load_factor_lv_cable_lc_normal"),
          get("assumptions", "load_factor_lv_cable_fc_normal")]
    tree, load, generation = lv_downstream(lv_grid)
    violations = []
    for parent, child in tree.edges():
        branch = lv_grid.graph.edges[parent, child]["branch"]
        if branch.helper_component:
            continue
        i_max = branch.type["I_max_th"] * branch.num_parallel
        for case, s in enumerate([load[child] / cos_load, generation[child] / cos_gen]):
            current = s / (sqrt(3) * v_nom)
            if current > i_max * lf[case] + 1e-6:
                violations.append(
                    f"{lv_grid}, {branch} ({branch.type.name}): {current:.0f} A > "
                    f"{lf[case]:.0%} of {i_max:.0f} A ({['load', 'feed-in'][case]} case, "
                    f"{s:.1f} kVA below)")
    return violations


def check_lv_transformer_sizing(lv_grid):
    """The MV/LV transformers of an LV grid carry its peak load and its
    generation (rated power times the load factor)."""
    get = _get
    station = lv_grid.station()
    s_trafos = sum(t.s_max_a for t in station.transformers())
    _, load, generation = lv_downstream(lv_grid)
    s_load = max(load[station], lv_grid.grid_district.peak_load) / get("assumptions", "cos_phi_load")
    s_gen = generation[station] / get("assumptions", "cos_phi_gen")
    violations = []
    for case, s, lf in [("load", s_load, get("assumptions", "load_factor_lv_trans_lc_normal")),
                        ("feed-in", s_gen, get("assumptions", "load_factor_lv_trans_fc_normal"))]:
        if s > s_trafos * lf + 1e-6:
            violations.append(f"{lv_grid}: {case} case needs {s:.1f} kVA, transformers "
                              f"{s_trafos:.0f} kVA at {lf:.0%}")
    return violations


def check_hvmv_transformer_sizing(mv_grid_district):
    """The HV/MV station is (n-1)-secure: without its largest transformer the
    others carry the peak load and the generation of the whole district."""
    get = _get
    mv_grid = mv_grid_district.mv_grid
    trafos = sorted(t.s_max_a for t in mv_grid.station().transformers())
    if not trafos:
        return [f"{mv_grid}: no HV/MV transformer"]
    remaining = sum(trafos[:-1])
    generation = sum(g.capacity for g in mv_grid.generators()) + sum(
        g.capacity for la in mv_grid_district.lv_load_areas()
        for lvgd in la.lv_grid_districts() for g in lvgd.lv_grid.generators())
    violations = []
    for case, s, lf in [
            ("load", mv_grid_district.peak_load / get("assumptions", "cos_phi_load"),
             get("assumptions", "load_factor_mv_trans_lc_malfunc")),
            ("feed-in", generation / get("assumptions", "cos_phi_gen"),
             get("assumptions", "load_factor_mv_trans_lc_malfunc"))]:
        if s > remaining * lf + 1e-6:
            violations.append(f"{mv_grid}: {case} case needs {s:.0f} kVA, (n-1) leaves "
                              f"{remaining:.0f} kVA of {trafos} kVA")
    return violations


def check_rings(mv_grid):
    """Every MV ring has exactly one switch disconnector, on one of its own
    branches, and at most `load_area_count_per_ring` load areas."""
    violations = []
    rings = list(mv_grid.rings_nodes(include_root_node=False))
    breakers = list(mv_grid.circuit_breakers())
    if len(rings) != len(breakers):
        violations.append(f"{mv_grid}: {len(rings)} rings, {len(breakers)} switch disconnectors")
    for cb in breakers:
        if cb.branch is None or cb.branch.circuit_breaker is not cb:
            violations.append(f"{cb}: not on its branch")
    max_count = _get("mv_routing", "load_area_count_per_ring")
    for ring in rings:
        stations = [n for n in ring if isinstance(n, LVStationDing0)]
        load_areas = {s.lv_load_area for s in stations}
        if len(load_areas) > max_count:
            violations.append(f"{mv_grid}: ring with {len(load_areas)} load areas > {max_count}")
    return violations


def check_no_isolated_nodes(grid):
    """No load, generator, storage or station without a connection."""
    return [f"{grid}: {node} is isolated" for node in grid.graph.nodes()
            if grid.graph.degree(node) == 0
            and isinstance(node, (LoadDing0, GeneratorDing0, GridComponentDing0, LVStationDing0))]
