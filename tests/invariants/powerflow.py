"""Independent power flow on the exported grid of an MV grid district.

ding0 checks its MV grids with a power flow of its own (MV only, LV grids
aggregated at their stations). Here the exported grid, including all LV
grids, is solved with PyPSA as eDisGo would read it, for the two cases ding0
dimensions grids for:

* snapshot 0, load case: loads at peak load, generators at
  `generation_in_load_case` of their capacity
* snapshot 1, feed-in case: generators at capacity, loads at
  `load_in_generation_case` of their peak load

The set points are derived here from the exported nominal values (load
`p_set`, generator `p_nom`) and the planning parameters, as eDisGo's worst
cases would derive them, not taken from ding0's own power flow data. Whether
ding0's own data agree is checked separately (`ding0_set_points`).
"""

from math import acos, tan

import numpy as np
import pandas as pd

from ding0.core.powerflow import q_sign
from ding0.tools import config as cfg_ding0
from ding0.tools.pypsa_io import (
    components_to_pypsa_network,
    fill_mvgd_component_dataframes,
    initialize_component_dataframes,
)

LOAD_CASE, FEEDIN_CASE = 0, 1


def case_factors():
    """Share of the nominal value of loads and generators per snapshot and
    their reactive power factors Q/P (generator sign convention for
    generators, load sign convention for loads)."""
    get = cfg_ding0.get
    return {
        "load": [1.0, get("assumptions", "load_in_generation_case")],
        "generation": [get("assumptions", "generation_in_load_case"), 1.0],
        "q_load": q_sign(get("assumptions", "cos_phi_load_mode"), "load")
        * tan(acos(get("assumptions", "cos_phi_load"))),
        "q_generation": q_sign(get("assumptions", "cos_phi_gen_mode"), "generator")
        * tan(acos(get("assumptions", "cos_phi_gen"))),
    }


def set_points(components):
    """Load and generator set points (MW, Mvar) of both snapshots from the
    exported nominal values, as dataframes (snapshots x names)."""
    f = case_factors()
    snapshots = [LOAD_CASE, FEEDIN_CASE]
    loads = components["Load"].p_set.astype(float)
    p_load = pd.DataFrame({s: loads * f["load"][s] for s in snapshots}).T
    generators = components["Generator"]
    generators = generators.loc[generators.control != "Slack", "p_nom"].astype(float)
    p_gen = pd.DataFrame({s: generators * f["generation"][s] for s in snapshots}).T
    return {"p_load": p_load, "q_load": p_load * f["q_load"],
            "p_gen": p_gen, "q_gen": p_gen * f["q_generation"]}


def ding0_set_points(data):
    """ding0's own power flow set points (`return_time_varying_data`) as
    dataframes (snapshots x names), MW and Mvar."""
    def frame(df, attr):
        return pd.DataFrame({name: list(values) for name, values in df[attr].items()},
                            index=[LOAD_CASE, FEEDIN_CASE])
    return {"p_load": frame(data["Load"], "p_set"), "q_load": frame(data["Load"], "q_set"),
            "p_gen": frame(data["Generator"], "p_set"),
            "q_gen": frame(data["Generator"], "q_set")}


def export_mv_grid_district(mv_grid_district, only_export_mv=False):
    """Components, network and grids dataframes and power flow set points of
    an MV grid district, with the switch disconnectors open (normal
    operation). The coordinates stay in EPSG:3035."""
    mv_grid_district.mv_grid.open_circuit_breakers()
    components, network_df, grids_df, data = fill_mvgd_component_dataframes(
        mv_grid_district, *initialize_component_dataframes(),
        only_export_mv=only_export_mv, return_time_varying_data=True)
    return components, network_df, grids_df, data


def solve(mv_grid_district, only_export_mv=False):
    """Runs a power flow on the exported grid.

    Returns
    -------
    network : :pypsa:`pypsa.Network`
        Solved network with snapshots 0 (load case) and 1 (feed-in case)
    converged : :pandas:`pandas.Series`
        Whether the power flow converged, per snapshot
    export : :obj:`tuple`
        As returned by :func:`export_mv_grid_district`
    """
    export = export_mv_grid_district(mv_grid_district, only_export_mv)
    components, network_df, grids_df, data = export
    network = components_to_pypsa_network(components, network_df, grids_df)
    network.set_snapshots([LOAD_CASE, FEEDIN_CASE])

    points = set_points(components)
    network.loads_t.p_set = points["p_load"]
    network.loads_t.q_set = points["q_load"]
    network.generators_t.p_set = points["p_gen"]
    network.generators_t.q_set = points["q_gen"]
    slack_bus = network.generators.loc[network.generators.control == "Slack", "bus"]
    v_slack = cfg_ding0.get("mv_routing_tech_constraints", "mv_station_v_level_operation")
    network.buses_t.v_mag_pu_set = pd.DataFrame(
        v_slack, index=network.snapshots, columns=list(slack_bus))

    info = network.pf()
    converged = info["converged"].iloc[:, 0] if isinstance(
        info["converged"], pd.DataFrame) else info["converged"]
    return network, converged, export


def complex_voltages(network, snapshot):
    """Complex bus voltages in per unit."""
    v_mag = network.buses_t.v_mag_pu.loc[snapshot]
    v_ang = network.buses_t.v_ang.loc[snapshot]
    return v_mag * np.exp(1j * v_ang)


def admittance_matrix(network):
    """Bus admittance matrix in per unit (base power 1 MVA, base voltage
    `v_nom` of each bus), built from the exported line and transformer data
    without using PyPSA's own matrices.

    Lines: r and x in ohm, base impedance v_nom^2 / 1 MVA. Transformers: r
    and x in per unit of s_nom, so r / s_nom in the system base. No shunts
    (ding0 exports none).
    """
    buses = network.buses.index
    index = {bus: i for i, bus in enumerate(buses)}
    y = np.zeros((len(buses), len(buses)), dtype=complex)

    def add(bus0, bus1, z):
        i, j = index[bus0], index[bus1]
        y[i, i] += 1 / z
        y[j, j] += 1 / z
        y[i, j] -= 1 / z
        y[j, i] -= 1 / z

    for _, line in network.lines.iterrows():
        v_nom = network.buses.at[line.bus0, "v_nom"]
        add(line.bus0, line.bus1, complex(line.r, line.x) / v_nom ** 2)
    for _, trafo in network.transformers.iterrows():
        add(trafo.bus0, trafo.bus1, complex(trafo.r, trafo.x) / trafo.s_nom)
    return pd.DataFrame(y, index=buses, columns=buses)


def nodal_injections(network, snapshot):
    """Expected net injection (MW, Mvar) per bus from the set points of loads
    and generators (generator convention), the slack as solved."""
    injection = pd.Series(0j, index=network.buses.index)
    p_load = network.loads_t.p_set.loc[snapshot] if not network.loads_t.p_set.empty else {}
    q_load = network.loads_t.q_set.loc[snapshot] if not network.loads_t.q_set.empty else {}
    for name, load in network.loads.iterrows():
        injection[load.bus] -= complex(p_load.get(name, 0.0), q_load.get(name, 0.0))
    for name, gen in network.generators.iterrows():
        if gen.control == "Slack":
            p = network.generators_t.p.at[snapshot, name]
            q = network.generators_t.q.at[snapshot, name]
        else:
            p = network.generators_t.p_set.at[snapshot, name] \
                if name in network.generators_t.p_set else 0.0
            q = network.generators_t.q_set.at[snapshot, name] \
                if name in network.generators_t.q_set else 0.0
        injection[gen.bus] += complex(p, q)
    return injection
