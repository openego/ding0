"""Planning principles of the active planning profile, checked on the result
grid, and whether the grid is no larger than these principles require.

The values come from ding0's config, into which `NetworkDing0.run_ding0`
writes the parameters of its planning profile
(:func:`ding0.tools.planning_profiles.apply_profile`); `static_data` of the
network holds the equipment the profile selects.

Minimality checks ("not planned too large") ask for every piece of
equipment chosen above the minimum whether something smaller would violate
a planning limit. A downgrade counts as needed if it exceeds a limit that
held before or exceeds an already exceeded limit further (`needed`), so
violations the grid already has (other findings) do not decide the result.
"""

from math import ceil, sqrt

import networkx as nx
import numpy as np

from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.tools import config as cfg_ding0
from ding0.tools.geo import calc_geo_dist

from . import physics, powerflow


def _get(section, key):
    return cfg_ding0.get(section, key)


def _lv_grids(mv_grid_district):
    return [lvgd.lv_grid for la in mv_grid_district.lv_load_areas()
            for lvgd in la.lv_grid_districts()]


# Equipment and structure #####################################################

def check_equipment_in_selection(mv_grid_district, static_data):
    """Every line, cable and transformer is of a type the planning profile
    selects (`equipment_selection.csv`, filtered into `static_data`)."""
    mv_grid = mv_grid_district.mv_grid
    mv_types = set(static_data["MV_cables"]["name"]) | set(static_data["MV_overhead_lines"]["name"])
    violations = []
    for edge in mv_grid.graph_edges():
        name = edge["branch"].type["name"]
        if name not in mv_types:
            violations.append(f"{edge['branch']}: MV type {name} not selected")
    hvmv = set(static_data["MV_trafos"]["S_nom"])
    for trafo in mv_grid.station().transformers():
        if trafo.s_max_a not in hvmv:
            violations.append(f"{mv_grid}: HV/MV transformer {trafo.s_max_a} kVA not selected")
    lv_cables = set(static_data["LV_cables"].index)
    lv_trafos = set(static_data["LV_trafos"]["S_nom"])
    for lv_grid in _lv_grids(mv_grid_district):
        for edge in lv_grid.graph_edges():
            branch = edge["branch"]
            if branch.helper_component:
                continue
            if branch.type.name not in lv_cables:
                violations.append(f"{branch}: LV cable {branch.type.name} not selected")
        for trafo in lv_grid.station().transformers():
            if trafo.s_max_a not in lv_trafos:
                violations.append(f"{lv_grid}: transformer {trafo.s_max_a} kVA not selected")
    return violations


def check_voltage_level_choice(mv_grid_district):
    """The MV grid runs at 20 kV if a load area centre lies at least
    `voltage_per_km_threshold` km from the HV/MV station, else at 10 kV."""
    mv_grid = mv_grid_district.mv_grid
    threshold = float(_get("assumptions", "voltage_per_km_threshold"))
    distances = [calc_geo_dist(mv_grid.station(), node) / 1e3
                 for node in mv_grid.graph.nodes() if isinstance(node, LVLoadAreaCentreDing0)]
    expected = 20 if distances and max(distances) >= threshold else 10
    if mv_grid.v_level != expected:
        return [f"{mv_grid}: {mv_grid.v_level} kV, farthest load area centre "
                f"{max(distances, default=0):.1f} km (threshold {threshold} km)"]
    return []


def half_rings(mv_grid):
    """The rings of an MV grid, each as a dict with `halves` (the two lists
    of branches from the HV/MV station to the switch disconnector),
    `switch` (the branch with the switch disconnector) and the export names
    of these branches (`half_names`, `switch_name`). Leaves the switch
    disconnectors open."""
    station = mv_grid.station()
    rings = []
    for _, branches, _ in mv_grid.rings_full_data():
        switch_branches = [b for b in branches if b.circuit_breaker is not None]
        if len(switch_branches) != 1:
            continue
        switch = switch_branches[0]
        graph = nx.Graph()
        for branch in branches:
            if branch is not switch:
                graph.add_edge(*mv_grid.graph_nodes_from_branch(branch), branch=branch)
        halves = []
        for end in mv_grid.graph_nodes_from_branch(switch):
            path = nx.shortest_path(graph, station, end) if end in graph else [station]
            halves.append([graph.edges[u, v]["branch"] for u, v in zip(path, path[1:])])
        # branch names need the branch in the graph, i.e. closed rings
        rings.append({"halves": halves, "switch": switch,
                      "half_names": [[repr(b) for b in half] for half in halves],
                      "switch_name": repr(switch)})
    mv_grid.open_circuit_breakers()
    return rings


def check_half_ring_lengths(mv_grid):
    """No half ring is longer than `max_half_ring_length` (km)."""
    limit = float(_get("mv_routing", "max_half_ring_length"))
    violations = []
    for ring in half_rings(mv_grid):
        for half in ring["halves"]:
            length = sum(b.length for b in half) / 1e3
            if length > limit + 1e-3:
                violations.append(f"{mv_grid}: half ring to {ring['switch'].circuit_breaker} is "
                                  f"{length:.2f} km > {limit} km")
    return violations


def check_satellites(mv_grid_district):
    """A load area is a satellite if and only if its peak load is below
    `load_area_sat_load_threshold`."""
    threshold = _get("mv_connect", "load_area_sat_load_threshold")
    return [f"{la}: peak load {la.peak_load:.0f} kW, satellite {la.is_satellite} "
            f"(threshold {threshold} kW)"
            for la in mv_grid_district.lv_load_areas()
            if bool(la.is_satellite) != (la.peak_load < threshold)]


def check_generator_levels(mv_grid_district):
    """Generators of voltage levels 4 and 5 are in the MV grid, of levels 6
    and 7 in an LV grid; level 6 at the LV busbar of its station."""
    violations = [f"{g}: level {g.v_level} in the MV grid"
                  for g in mv_grid_district.mv_grid.generators() if g.v_level not in (4, 5)]
    for lv_grid in _lv_grids(mv_grid_district):
        station = lv_grid.station()
        for g in lv_grid.generators():
            if g.v_level not in (6, 7):
                violations.append(f"{g}: level {g.v_level} in {lv_grid}")
            elif g.v_level == 6 and station not in lv_grid.graph.neighbors(g):
                violations.append(f"{g}: level 6 not connected to the station of {lv_grid}")
    return violations


def check_lv_feeder_load(lv_grid, tolerance=0.1):
    """Every feeder leaving an LV station carries at most the rated current
    of the standard cable `lv_standard_line` (ding0 chooses the number of
    feeders by this rule), within `tolerance` because loads are split into
    whole buildings."""
    get = _get
    standard = get("assumptions", "lv_standard_line")
    i_max = lv_grid.network.static_data["LV_cables"].at[standard, "I_max_th"]
    v_nom = get("assumptions", "lv_nominal_voltage") / 1e3
    cos_phi = get("assumptions", "cos_phi_load")
    tree, load, _ = physics.lv_downstream(lv_grid)
    station = lv_grid.station()
    violations = []
    for feeder in tree.successors(station):
        current = load[feeder] / cos_phi / (sqrt(3) * v_nom)
        if current > i_max * (1 + tolerance):
            violations.append(f"{lv_grid}, feeder to {feeder}: {current:.0f} A > "
                              f"{i_max:.0f} A of {standard}")
    return violations


# (n-1) of MV rings ###########################################################

def check_rings_n_minus_1(mv_grid, network, components):
    """Every MV ring stays within its (n-1) limits in the load case: with
    its switch disconnector closed and one of its two lines at the HV/MV
    station out of service, no ring line carries more than
    `load_factor_mv_{kind}_lc_malfunc` of its rated current and no MV node
    drops more than `mv_max_v_level_lc_diff_malfunc` below the station.

    `network` is the solved MV export (LV grids aggregated at their
    stations), `components` its export."""
    get = _get
    v_limit = get("mv_routing_tech_constraints", "mv_max_v_level_lc_diff_malfunc")
    lf = {kind: get("assumptions", f"load_factor_mv_{kind}_lc_malfunc") for kind in ["line", "cable"]}
    slack_bus = network.generators.loc[network.generators.control == "Slack", "bus"].iloc[0]
    switches = components["Switch"].set_index("branch")
    violations = []
    for ring in half_rings(mv_grid):
        switch_name = ring["switch_name"]
        ring_lines = [name for half in ring["half_names"] for name in half] + [switch_name]
        for half in ring["half_names"]:
            if not half:
                continue
            outage = half[0]
            case = network.copy()
            switch = switches.loc[switch_name]
            for end in ["bus0", "bus1"]:
                if case.lines.at[switch_name, end] == switch.bus_open:
                    case.lines.at[switch_name, end] = switch.bus_closed
            case.remove("Bus", switch.bus_open)
            case.remove("Line", outage)
            converged = powerflow.run_pf(case)
            if not converged[powerflow.LOAD_CASE]:
                violations.append(f"{mv_grid}: no power flow without {outage}")
                continue
            v = case.buses_t.v_mag_pu.loc[powerflow.LOAD_CASE]
            drop = (v[slack_bus] - v).max()
            if drop > v_limit + 1e-9:
                violations.append(f"{mv_grid}, without {outage}: voltage drop {drop:.2%} "
                                  f"> {v_limit:.0%}")
            lines = case.lines.loc[[name for name in ring_lines if name != outage]]
            t = case.lines_t
            lc = powerflow.LOAD_CASE
            i = np.maximum(np.hypot(t.p0.loc[lc, lines.index], t.q0.loc[lc, lines.index])
                           / v[lines.bus0].values,
                           np.hypot(t.p1.loc[lc, lines.index], t.q1.loc[lc, lines.index])
                           / v[lines.bus1].values) / lines.s_nom
            for name, loading in i.items():
                limit = lf[lines.at[name, "kind"]]
                if loading > limit + 1e-9:
                    violations.append(f"{mv_grid}, without {outage}: {name} at {loading:.0%} "
                                      f"> {limit:.0%}")
    return violations


# Not planned too large ########################################################

def check_hvmv_transformers_minimal(mv_grid_district):
    """The HV/MV station is not larger than (n-1) requires: no set of equal
    transformers of one selected size with less rated power in total would
    still carry the peak load and the generation without one transformer at
    `load_factor_mv_trans_lc_malfunc`."""
    get = _get
    mv_grid = mv_grid_district.mv_grid
    sizes = sorted(mv_grid.network.static_data["MV_trafos"]["S_nom"])
    installed = sorted(t.s_max_a for t in mv_grid.station().transformers())
    generation = sum(g.capacity for g in mv_grid.generators()) + sum(
        g.capacity for lv_grid in _lv_grids(mv_grid_district) for g in lv_grid.generators())
    need = max(mv_grid_district.peak_load / get("assumptions", "cos_phi_load"),
               generation / get("assumptions", "cos_phi_gen")) \
        / get("assumptions", "load_factor_mv_trans_lc_malfunc")
    # n transformers of size s carry `need` without one: (n - 1) s >= need, n >= 2
    options = {s: (max(ceil(need / s), 1) + 1) * s for s in sizes}
    smallest = min(options.values())
    if sum(installed) > smallest + 1e-6:
        best = min(options, key=options.get)
        return [f"{mv_grid}: HV/MV transformers {installed} kVA for {need:.0f} kVA in (n-1); "
                f"{options[best] // best:.0f} x {best} kVA would do"]
    return []


def check_lv_transformers_minimal(lv_grid):
    """The transformers of an LV station are not larger than needed: with
    the next smaller selected size, or with one transformer less, they would
    no longer carry the LV grid's peak load and generation at the load
    factors `load_factor_lv_trans_{lc,fc}_normal`."""
    get = _get
    sizes = sorted(lv_grid.network.static_data["LV_trafos"]["S_nom"])
    station = lv_grid.station()
    trafos = [t.s_max_a for t in station.transformers()]
    _, load, generation = physics.lv_downstream(lv_grid)
    s_load = max(load[station], lv_grid.grid_district.peak_load) / get("assumptions", "cos_phi_load")
    s_gen = generation[station] / get("assumptions", "cos_phi_gen")
    need = max(s_load / get("assumptions", "load_factor_lv_trans_lc_normal"),
               s_gen / get("assumptions", "load_factor_lv_trans_fc_normal"))
    violations = []
    size, count = max(trafos), len(trafos)
    smaller = [s for s in sizes if s < size]
    if smaller and count * smaller[-1] >= need:
        violations.append(f"{lv_grid}: {count} x {size:.0f} kVA for {need:.0f} kVA; "
                          f"{count} x {smaller[-1]:.0f} kVA would do")
    if count > 1 and (count - 1) * size >= need:
        violations.append(f"{lv_grid}: {count} x {size:.0f} kVA for {need:.0f} kVA; "
                          f"{count - 1} would do")
    return violations


def _smaller_mv_type(catalog, current_name, u_n):
    """Next smaller MV type (by rated current) of the same voltage."""
    catalog = catalog.loc[catalog.U_n == u_n]
    current = catalog.loc[catalog["name"] == current_name]
    smaller = catalog.loc[catalog.I_max_th < current.I_max_th.iloc[0]]
    return None if smaller.empty else smaller.loc[smaller.I_max_th.idxmax()]


def _set_type(network, line, row, u_n_kv):
    omega = 2 * np.pi * _get("assumptions", "frequency")
    n = network.lines.at[line, "num_parallel"]
    length = network.lines.at[line, "length"]
    network.lines.at[line, "r"] = row.R_per_km * length / n
    network.lines.at[line, "x"] = omega * row.L_per_km * 1e-3 * length / n
    network.lines.at[line, "s_nom"] = sqrt(3) * u_n_kv * row.I_max_th / 1e3 * n


def needed(before, after):
    """Whether a downgrade is needed: a limit that held is exceeded, or one
    that was exceeded is exceeded further (`physics.limit_usage`)."""
    return any(u > 1 + 1e-9 and u > before.get(key, 0) + 1e-9 for key, u in after.items())


def check_mv_reinforcement_necessary(mv_grid, network):
    """Every MV cable of a larger type than the grid's standard type is
    needed: one size smaller, on its own, it would make the MV voltage band
    or a thermal limit in normal operation fail (`needed`).

    `network` is the solved MV export in normal operation (open switch
    disconnectors, LV grids aggregated)."""
    catalog = mv_grid.network.static_data["MV_cables"]
    standard = mv_grid.default_branch_type
    before = physics.limit_usage(network, levels=("MV",))
    violations = []
    for edge in mv_grid.graph_edges():
        branch = edge["branch"]
        line = repr(branch)
        if branch.kind != "cable" or line not in network.lines.index or \
                branch.type["I_max_th"] <= standard["I_max_th"]:
            continue
        smaller = _smaller_mv_type(catalog, branch.type["name"], branch.type["U_n"])
        if smaller is None or smaller.I_max_th < standard["I_max_th"]:
            continue
        case = network.copy()
        _set_type(case, line, smaller, branch.type["U_n"])
        if not powerflow.run_pf(case).all():
            continue
        if not needed(before, physics.limit_usage(case, levels=("MV",))):
            violations.append(f"{branch} ({branch.type['name']}, {branch.length:.0f} m): "
                              f"{smaller['name']} would also keep all limits")
    return violations


def check_lv_feeders_minimal(lv_grid, network):
    """The cable type of every LV feeder is needed: with the cables of the
    feeder's type (the type of its first cable at the station) one size
    smaller, the LV grid would fail the LV voltage band or a thermal limit
    (`needed`). House connections keep their type.

    `network` is the solved export including the LV grids; the LV grid is
    solved on its own with its busbar voltage from `network`."""
    catalog = lv_grid.network.static_data["LV_cables"]
    tree = nx.bfs_tree(lv_grid.graph, lv_grid.station())
    subnet = powerflow.lv_subnetwork(network, lv_grid.id_db)
    before = physics.limit_usage(subnet, levels=("LV",))
    violations = []
    for head in tree.successors(lv_grid.station()):
        first = lv_grid.graph.edges[lv_grid.station(), head]["branch"]
        cable = first.type
        smaller = catalog.loc[catalog.I_max_th < cable.I_max_th]
        if first.helper_component or smaller.empty:
            continue
        row = smaller.loc[smaller.I_max_th.idxmax()]
        case = subnet.copy()
        edges = [(lv_grid.station(), head)] + list(nx.bfs_edges(tree, head))
        changed = 0
        for u, v in edges:
            branch = lv_grid.graph.edges[u, v]["branch"]
            if branch.helper_component or branch.type.name != cable.name or \
                    repr(branch) not in case.lines.index:
                continue
            _set_type(case, repr(branch), row, catalog.at[cable.name, "U_n"] / 1e3)
            changed += 1
        if changed and powerflow.run_pf(case).all() and \
                not needed(before, physics.limit_usage(case, levels=("LV",))):
            violations.append(f"{lv_grid}, feeder {first} ({changed} cables {cable.name}): "
                              f"{row.name} would also keep all limits")
    return violations
