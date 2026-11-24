"""Plausibility ranges from German distribution grid planning practice.

Unlike the invariants, these are not rules a grid must follow: a grid
outside a range is unusual, not wrong. The checks return messages that the
tests report as warnings (`PlausibilityWarning`) without failing.

Ranges (domain knowledge of DSO practice; Kerber's LV reference grids,
TU Munich 2011, for transformer sizes and feeder lengths):

* MV/LV transformers are loaded with at least half their rating at the
  planning load; a station loaded less is larger than needed.
* LV feeders leave the station with a trunk cable of at least 95 mm2
  (standard NAYY 4x150 or 4x240; 35 and 50 mm2 are house connections).
* LV feeders reach at most 1 km from the station, and a station has one to
  twelve feeders.
"""

import networkx as nx

from ding0.tools import config as cfg_ding0

from . import physics


class PlausibilityWarning(UserWarning):
    """A grid property outside the range usual in DSO planning."""


MIN_TRANSFORMER_UTILISATION = 0.5
MIN_TRUNK_CROSS_SECTION = 95  # mm2
MAX_FEEDER_REACH = 1000  # m
MAX_FEEDERS = 12


def lv_grid_messages(lv_grid):
    """Plausibility messages of one LV grid."""
    get = cfg_ding0.get
    station = lv_grid.station()
    messages = []
    installed = sum(t.s_max_a for t in station.transformers())
    _, load, generation = physics.lv_downstream(lv_grid)
    need = max(max(load[station], lv_grid.grid_district.peak_load) / get("assumptions", "cos_phi_load"),
               generation[station] / get("assumptions", "cos_phi_gen"))
    if installed and need / installed < MIN_TRANSFORMER_UTILISATION:
        messages.append(f"{lv_grid}: transformers {installed:.0f} kVA loaded with "
                        f"{need / installed:.0%} at the planning load")
    graph = lv_grid.graph
    feeders = list(graph.neighbors(station))
    if len(feeders) > MAX_FEEDERS:
        messages.append(f"{lv_grid}: {len(feeders)} feeders")
    distances = nx.single_source_dijkstra_path_length(
        graph, station, weight=lambda u, v, d: d["branch"].length)
    if distances and max(distances.values()) > MAX_FEEDER_REACH:
        messages.append(f"{lv_grid}: feeders reach {max(distances.values()):.0f} m")
    # street feeders; a direct connection of a single load is a house connection
    trunks = [graph.edges[station, node]["branch"] for node in feeders
              if graph.degree(node) > 1 and not graph.edges[station, node]["branch"].helper_component]
    thin = sorted(b.type.name for b in trunks
                  if physics.cross_section(b.type.name) < MIN_TRUNK_CROSS_SECTION)
    if thin:
        messages.append(f"{lv_grid}: {len(thin)} of {len(trunks)} street feeders start with "
                        f"less than {MIN_TRUNK_CROSS_SECTION} mm2 ({', '.join(thin)})")
    return messages


def mv_grid_messages(network, share=0.5, loading=0.2):
    """A message if more than `share` of the MV cable length is loaded below
    `loading` of its rating in both cases (lightly used equipment)."""
    lines = network.lines.loc[network.buses.loc[network.lines.bus0, "v_nom"].values > 1]
    if lines.empty:
        return []
    usage = physics.loadings(network)
    peak = lines.index.map(lambda name: max(usage[s, name][0] for s in network.snapshots))
    light = lines.length[peak < loading].sum() / lines.length.sum()
    if light > share:
        return [f"{light:.0%} of the MV cable length loaded below {loading:.0%}"]
    return []
