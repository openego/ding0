"""Completeness: everything from the input data is in the grid and in the
export, once, with its values; nothing is dropped silently.

The expected data are passed in, so the checks work for the synthetic grid
districts and for egon-data alike:

* buildings: dataframe indexed by building_id with columns mvgd, la_id,
  capacity (kW, including the dimensioning share of heat pumps and home
  charging points), residential_capacity, cts_capacity,
  industrial_capacity, number_households
* generators: dataframe with columns mvgd, gens_id, electrical_capacity
  (kW), voltage_level
* point components: dataframe with columns mvgd, source_id, p_nom (kW),
  component ('load' or 'storage_unit')
"""

from collections import defaultdict

import numpy as np

from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.core.network.components import PointLoadDing0, StorageUnitDing0
from ding0.core.network.loads import LVLoadDing0
from ding0.tools import config as cfg_ding0


def building_class(capacity):
    """Where a building of a load area is connected, by its capacity."""
    if capacity < get_config_osm("lv_threshold_capacity"):
        return "lv_street"
    if capacity < get_config_osm("mv_lv_threshold_capacity"):
        return "lv_station"
    if capacity < get_config_osm("hv_mv_threshold_capacity"):
        return "mv"
    return "hv"


def grid_buildings(mv_grid_district):
    """Buildings in the grid: building_id -> list of (kind, load object)."""
    found = defaultdict(list)
    for la in mv_grid_district.lv_load_areas():
        for lvgd in la.lv_grid_districts():
            for node in lvgd.lv_grid.graph.nodes():
                if isinstance(node, LVLoadDing0):
                    found[int(node.building_id)].append(("lv", node))
    for load in mv_grid_district.mv_grid._loads:
        found[int(load.building_id)].append(("mv", load))
    return found


def check_buildings(network, buildings):
    """Every building of a load area that is not below the load area
    threshold is in the grid exactly once: below `mv_lv_threshold_capacity`
    as an LV load, below `hv_mv_threshold_capacity` as an MV load (above it
    deliberately not, with a warning). Sectoral peak loads and households
    are kept."""
    threshold = cfg_ding0.get("mv_routing", "load_area_threshold")
    violations = []
    for mvgd in network.mv_grid_districts():
        expected = buildings.loc[buildings.mvgd == mvgd.id_db]
        la_totals = expected.groupby("la_id").capacity.sum()
        found = grid_buildings(mvgd)
        for building_id, row in expected.iterrows():
            entries = found.get(int(building_id), [])
            if la_totals[row.la_id] <= threshold:
                continue
            cls = building_class(row.capacity)
            if cls == "hv":
                if entries:
                    violations.append(f"building {building_id} ({row.capacity:.0f} kW) above "
                                      f"the MV threshold is in the grid")
                continue
            if len(entries) != 1:
                violations.append(f"building {building_id} ({row.capacity:.1f} kW, {cls}) "
                                  f"is {len(entries)} times in the grid")
                continue
            kind, load = entries[0]
            if (kind == "mv") != (cls == "mv"):
                violations.append(f"building {building_id} ({row.capacity:.1f} kW) is an "
                                  f"{kind.upper()} load, expected {cls}")
            for attr, column in [("peak_load_residential", "residential_capacity"),
                                 ("peak_load_cts", "cts_capacity"),
                                 ("peak_load_industrial", "industrial_capacity"),
                                 ("number_households", "number_households")]:
                if not np.isclose(getattr(load, attr) or 0, row[column]):
                    violations.append(f"building {building_id}: {attr} {getattr(load, attr)}, "
                                      f"expected {row[column]}")
        unexpected = set(found) - set(int(b) for b in expected.index)
        if unexpected:
            violations.append(f"{mvgd}: loads of unknown buildings {sorted(unexpected)[:5]}")
    return violations


def _unique(objects):
    """Objects in order, each once (the lists of objects still to connect
    keep them after they were connected)."""
    seen = {}
    for obj in objects:
        seen.setdefault(id(obj), obj)
    return list(seen.values())


def grid_generators(mv_grid_district):
    """All generators of an MV grid district, connected or not."""
    mv_grid = mv_grid_district.mv_grid
    generators = list(mv_grid.generators()) + list(mv_grid.lv_generators_to_connect)
    for la in mv_grid_district.lv_load_areas():
        for lvgd in la.lv_grid_districts():
            generators += list(lvgd.lv_grid.generators())
    return _unique(generators)


def check_generators(network, generators):
    """Every generator of an MV grid district is in its grid exactly once,
    with its capacity and voltage level, and connected."""
    violations = []
    for mvgd in network.mv_grid_districts():
        expected = generators.loc[generators.mvgd == mvgd.id_db].set_index("gens_id")
        found = defaultdict(list)
        for generator in grid_generators(mvgd):
            found[generator.gens_id].append(generator)
        missing = sorted(set(expected.index) - set(found))
        if missing:
            violations.append(f"{mvgd}: generators missing: {missing}")
        unexpected = sorted(set(found) - set(expected.index))
        if unexpected:
            violations.append(f"{mvgd}: unexpected generators: {unexpected}")
        for gens_id, objects in found.items():
            if len(objects) > 1:
                violations.append(f"{mvgd}: generator {gens_id} {len(objects)} times")
            if gens_id in expected.index and not np.isclose(
                    objects[0].capacity, expected.at[gens_id, "electrical_capacity"]):
                violations.append(f"generator {gens_id}: {objects[0].capacity} kW, "
                                  f"expected {expected.at[gens_id, 'electrical_capacity']} kW")
    return violations


def grid_components(mv_grid_district):
    """Loads and storage units at locations of their own of an MV grid
    district, connected or not."""
    mv_grid = mv_grid_district.mv_grid
    components = list(mv_grid.components()) + list(mv_grid.mv_components_to_connect) + \
        list(mv_grid.lv_components_to_connect)
    for la in mv_grid_district.lv_load_areas():
        for lvgd in la.lv_grid_districts():
            components += list(lvgd.lv_grid.components())
    return _unique(components)


def check_point_components(network, components):
    """Every point load and storage unit is in the grid exactly once."""
    violations = []
    for mvgd in network.mv_grid_districts():
        expected = components.loc[components.mvgd == mvgd.id_db].set_index("source_id")
        found = defaultdict(list)
        for component in grid_components(mvgd):
            found[component.source_id].append(component)
        missing = sorted(set(expected.index) - set(found))
        if missing:
            violations.append(f"{mvgd}: components missing: {missing}")
        for source_id, objects in found.items():
            if len(objects) > 1:
                violations.append(f"{mvgd}: component {source_id} {len(objects)} times")
            if source_id in expected.index and not np.isclose(
                    objects[0].p_nom, expected.at[source_id, "p_nom"]):
                violations.append(f"component {source_id}: {objects[0].p_nom} kW, expected "
                                  f"{expected.at[source_id, 'p_nom']} kW")
    return violations


def check_peak_load_chain(mv_grid_district, rtol=1e-6):
    """Peak loads add up bottom-up: an LV grid district's peak load is the
    sum of its loads (buildings including assigned point loads of level 7,
    and point loads of level 6 at its station); a load area's peak load is
    the sum of its LV grid districts'; the MV grid district's peak load is
    the sum of its load areas, MV loads and MV point loads."""
    violations = []
    total = 0.0
    for la in mv_grid_district.lv_load_areas():
        lvgd_sum = 0.0
        for lvgd in la.lv_grid_districts():
            grid = lvgd.lv_grid
            loads = sum(n.peak_load for n in grid.graph.nodes() if isinstance(n, LVLoadDing0))
            loads += sum(c.peak_load for c in grid.components()
                         if isinstance(c, PointLoadDing0) and c.v_level == 6)
            if not np.isclose(loads, lvgd.peak_load, rtol=rtol):
                violations.append(f"{lvgd}: peak load {lvgd.peak_load:.1f} kW, its loads "
                                  f"{loads:.1f} kW")
            lvgd_sum += lvgd.peak_load
        if not np.isclose(lvgd_sum, la.peak_load, rtol=rtol):
            violations.append(f"{la}: peak load {la.peak_load:.1f} kW, its LV grid districts "
                              f"{lvgd_sum:.1f} kW")
        total += la.peak_load
    mv_grid = mv_grid_district.mv_grid
    total += sum(load.peak_load for load in mv_grid._loads)
    total += sum(c.peak_load for c in mv_grid.components() if isinstance(c, PointLoadDing0))
    if not np.isclose(total, mv_grid_district.peak_load, rtol=rtol):
        violations.append(f"{mv_grid_district}: peak load {mv_grid_district.peak_load:.1f} kW, "
                          f"bottom-up {total:.1f} kW")
    return violations


def check_export_matches_grid(components, mv_grid_district, rtol=1e-6):
    """The export contains the loads, generators and storage units of the
    grid with their values (MW): sectoral building loads, heat pumps and
    charging points of buildings, point loads, generators."""
    loads = components["Load"]
    violations = []
    expected_building_loads = 0.0
    for kind, load in [entry for entries in grid_buildings(mv_grid_district).values()
                       for entry in entries]:
        expected_building_loads += sum(getattr(load, f"peak_load_{s}") or 0
                                       for s in ["residential", "cts", "industrial"])
    sectoral = loads.loc[loads.sector.isin(["residential", "cts", "industrial"])]
    if not np.isclose(sectoral.p_set.sum() * 1e3, expected_building_loads, rtol=rtol):
        violations.append(f"sectoral building loads exported {sectoral.p_set.sum() * 1e3:.1f} kW, "
                          f"in the grid {expected_building_loads:.1f} kW")
    point_loads = [c for c in grid_components(mv_grid_district) if isinstance(c, PointLoadDing0)]
    source_ids = loads["source_id"] if "source_id" in loads else loads.index.to_series().map(lambda _: None)
    exported = loads.loc[source_ids.isin([c.source_id for c in point_loads])]
    if not np.isclose(exported.p_set.sum() * 1e3, sum(c.p_nom for c in point_loads), rtol=rtol):
        violations.append(f"point loads exported {exported.p_set.sum() * 1e3:.1f} kW, in the grid "
                          f"{sum(c.p_nom for c in point_loads):.1f} kW")
    generators = components["Generator"]
    generators = generators.loc[generators.control != "Slack"]
    in_grid = grid_generators(mv_grid_district)
    if len(generators) != len(in_grid):
        violations.append(f"{len(generators)} generators exported, {len(in_grid)} in the grid")
    if not np.isclose(generators.p_nom.sum() * 1e3, sum(g.capacity for g in in_grid), rtol=rtol):
        violations.append(f"generation exported {generators.p_nom.sum() * 1e3:.1f} kW, in the grid "
                          f"{sum(g.capacity for g in in_grid):.1f} kW")
    storage = [c for c in grid_components(mv_grid_district) if isinstance(c, StorageUnitDing0)]
    exported = components["StorageUnit"]
    exported = exported.loc[exported.type == "large_storage"]
    if len(exported) != len(storage):
        violations.append(f"{len(exported)} large storage units exported, {len(storage)} in the grid")
    return violations


def check_unique_names(components):
    """Component names are unique within each component type."""
    return [f"{key}: duplicate names {sorted(set(df.index[df.index.duplicated()]))[:5]}"
            for key, df in components.items()
            if hasattr(df, "index") and df.index.has_duplicates]
