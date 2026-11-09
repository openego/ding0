"""Physical, geographic and completeness invariants of complete ding0 runs on
the synthetic grid districts (no database needed).

Each invariant of `tests/invariants` is checked on three runs: grid district
1, grid district 2 with all components of newer egon-data versions (heat
pumps, charging points, rail, storage) and both districts in one run. Some
invariants are also checked right after the pipeline step that establishes
them (`staged`), since later steps change the grid.

The power flow checks use an independent power flow on the exported grid
including all LV grids (`tests/invariants/powerflow.py`), not ding0's own
MV power flow.

Invariants ding0 violates today are marked as expected failures with the
reason; `strict` reports them as soon as they hold.
"""

import functools
import logging

import pytest

from ding0.core import NetworkDing0
from tests import synthetic_district
from tests.invariants import assert_no_violations, completeness, geo, physics, powerflow

# run name: (MV grid districts, with components of newer egon-data versions)
RUNS = {
    "district_1": ([1], False),
    "district_2": ([2], True),
    "districts_1_2": ([1, 2], True),
}


class Run:
    def __init__(self, name):
        self.name = name
        self.mvgd_ids, self.components = RUNS[name]
        self.staged = {}
        self._solved = {}

    def mv_grid_districts(self):
        return list(self.network.mv_grid_districts())

    def solved(self, mvgd):
        """Independent power flow on the full export (MV and LV)."""
        if mvgd.id_db not in self._solved:
            self._solved[mvgd.id_db] = powerflow.solve(mvgd)
        return self._solved[mvgd.id_db]

    def export(self, mvgd):
        return self.solved(mvgd)[2]

    def solved_mv(self, mvgd):
        """Independent power flow on the MV export (LV grids aggregated at
        their stations, as in ding0's own power flow)."""
        key = ("mv", mvgd.id_db)
        if key not in self._solved:
            self._solved[key] = powerflow.solve(mvgd, only_export_mv=True)
        return self._solved[key]


def _record_after(monkeypatch, run, method, check):
    """Wrap a pipeline step of NetworkDing0 to check invariants right after it."""
    original = getattr(NetworkDing0, method)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        run.staged[method] = check(self)
        return result

    monkeypatch.setattr(NetworkDing0, method, wrapper)


def _lv_grids(network):
    return [lvgd.lv_grid for mvgd in network.mv_grid_districts()
            for la in mvgd.lv_load_areas() for lvgd in la.lv_grid_districts()]


def _after_build_lv_grids(network):
    violations = []
    for grid in _lv_grids(network):
        violations += physics.check_lv_tree(grid)
    return violations


def _after_mv_routing(network):
    from ding0.core.network.components import PointLoadDing0
    from ding0.core.network.loads import MVLoadDing0
    from ding0.core.network.stations import LVStationDing0
    return [f"{mvgd}: {node} not connected by the MV routing"
            for mvgd in network.mv_grid_districts()
            for node in mvgd.mv_grid.graph_isolated_nodes()
            if isinstance(node, (LVStationDing0, MVLoadDing0, PointLoadDing0))]


def _after_connect_generators(network):
    violations = []
    for mvgd in network.mv_grid_districts():
        violations += physics.check_no_isolated_nodes(mvgd.mv_grid)
        for grid in _lv_grids(network):
            violations += physics.check_no_isolated_nodes(grid)
    return violations


@pytest.fixture(scope="module", params=list(RUNS))
def run(request):
    run = Run(request.param)
    monkeypatch = pytest.MonkeyPatch()
    synthetic_district.patch_database(monkeypatch, components=run.components)
    _record_after(monkeypatch, run, "build_lv_grids", _after_build_lv_grids)
    _record_after(monkeypatch, run, "mv_routing", _after_mv_routing)
    _record_after(monkeypatch, run, "connect_generators", _after_connect_generators)
    try:
        logging.disable(logging.WARNING)
        network = NetworkDing0(name="synthetic")
        network._orm = {}
        run.messages = network.run_ding0(session=None, mv_grid_districts_no=run.mvgd_ids)
        run.network = network
        for mvgd in network.mv_grid_districts():
            run.solved_mv(mvgd)
            run.solved(mvgd)
    finally:
        logging.disable(logging.NOTSET)
        monkeypatch.undo()
    return run


def per_district(check):
    """Applies a check to every MV grid district of a run."""
    def apply(run):
        violations = []
        for mvgd in run.mv_grid_districts():
            violations += [f"{mvgd}: {v}" for v in check(run, mvgd)]
        return violations
    return apply


def per_lv_grid(check):
    def apply(run):
        return [v for grid in _lv_grids(run.network) for v in check(grid)]
    return apply


CHECKS = {
    # physics: independent power flow on the exported MV and LV grids
    "power_flow_converges": per_district(lambda r, g: physics.check_converged(r.solved(g)[1])),
    "kirchhoff_and_ohm": per_district(lambda r, g: physics.check_kirchhoff(r.solved(g)[0])),
    "losses_positive_and_small": per_district(lambda r, g: physics.check_losses(r.solved(g)[0])),
    "energy_balance": per_district(lambda r, g: physics.check_energy_balance(r.solved(g)[0])),
    "mv_voltage_band": per_district(
        lambda r, g: physics.check_voltage_band(r.solved(g)[0], levels=("MV",))),
    "lv_voltage_band": per_district(
        lambda r, g: physics.check_voltage_band(r.solved(g)[0], levels=("LV",))),
    "thermal_limits": per_district(lambda r, g: physics.check_thermal_limits(r.solved(g)[0])),
    # physics: exported data and ding0's own results
    "line_impedances_follow_type_and_length": per_district(
        lambda r, g: physics.check_line_impedances(r.export(g)[0], r.network.static_data)),
    "voltage_levels": per_district(
        lambda r, g: physics.check_voltage_levels(r.export(g)[0], g.mv_grid.v_level)),
    "radial_operation": per_district(lambda r, g: physics.check_radial(r.export(g)[0])),
    "open_switches_separate_rings": per_district(lambda r, g: physics.check_switches(r.export(g)[0])),
    "ding0_set_points_match_export": per_district(
        lambda r, g: physics.check_set_points(r.export(g)[0], r.export(g)[3])),
    "ding0_power_flow_results_complete": per_district(
        lambda r, g: physics.check_ding0_powerflow_results(g.mv_grid)),
    "ding0_mv_limits_after_reinforcement": per_district(
        lambda r, g: physics.check_ding0_mv_limits(g.mv_grid)),
    "ding0_mv_results_match_normal_operation": per_district(
        lambda r, g: physics.check_ding0_results_match(g.mv_grid, r.solved_mv(g)[0])),
    "mv_export_power_flow_converges": per_district(
        lambda r, g: physics.check_converged(r.solved_mv(g)[1])),
    "mv_export_voltage_band": per_district(
        lambda r, g: physics.check_voltage_band(r.solved_mv(g)[0], levels=("MV",))),
    "mv_export_thermal_limits": per_district(
        lambda r, g: physics.check_thermal_limits(r.solved_mv(g)[0], tolerance=0.0)),
    "export_keeps_switch_state": per_district(
        lambda r, g: physics.check_export_keeps_switch_state(g)),
    "rings_and_switch_disconnectors": per_district(lambda r, g: physics.check_rings(g.mv_grid)),
    "hvmv_transformers_n_minus_1": per_district(
        lambda r, g: physics.check_hvmv_transformer_sizing(g)),
    "lv_grids_are_trees": per_lv_grid(physics.check_lv_tree),
    "lv_cables_carry_load_and_generation": per_lv_grid(physics.check_lv_cable_sizing),
    "lv_transformers_carry_load_and_generation": per_lv_grid(physics.check_lv_transformer_sizing),
    "no_isolated_nodes": lambda r: [v for g in r.mv_grid_districts()
                                    for v in physics.check_no_isolated_nodes(g.mv_grid)]
    + [v for grid in _lv_grids(r.network) for v in physics.check_no_isolated_nodes(grid)],
    # geography
    "coordinates_in_germany": per_district(
        lambda r, g: geo.check_coordinates_in_germany(r.export(g)[0]["Bus"])),
    "line_geometries_end_at_buses": per_district(lambda r, g: geo.check_line_endpoints(r.export(g)[0])),
    "line_lengths_match_geometries": per_district(lambda r, g: geo.check_line_lengths(r.export(g)[0])),
    # the street nodes of the synthetic load areas are jittered by up to
    # 7.6 m and the polygon is drawn around the corner nodes, so buildings
    # beside the outer streets lie up to 15 m outside it
    "objects_lie_in_their_areas": per_district(
        lambda r, g: geo.check_containment(g, building_tolerance=20.0)),
    "lv_components_connected_in_own_district": lambda r: geo.check_lv_connection_location(r.network),
    "switch_disconnectors_on_their_lines": per_district(
        lambda r, g: geo.check_circuit_breaker_positions(g.mv_grid)),
    "no_duplicate_buses": per_district(lambda r, g: geo.check_no_duplicate_buses(r.export(g)[0])),
    "lv_feeders_within_distance": per_district(lambda r, g: geo.check_lv_feeder_lengths(r.export(g)[0])),
    # completeness
    "all_buildings_once": lambda r: completeness.check_buildings(
        r.network, synthetic_district.expected_buildings(r.components)),
    "all_generators_once": lambda r: completeness.check_generators(
        r.network, synthetic_district.expected_generators(r.mvgd_ids)),
    "all_point_components_once": lambda r: completeness.check_point_components(
        r.network, synthetic_district.expected_point_components(r.mvgd_ids)
        if r.components else synthetic_district.expected_point_components(r.mvgd_ids).iloc[0:0]),
    "peak_loads_add_up": per_district(lambda r, g: completeness.check_peak_load_chain(g)),
    "export_matches_grid": per_district(
        lambda r, g: completeness.check_export_matches_grid(r.export(g)[0], g)),
    "unique_names": lambda r: completeness.check_unique_names(r.network.to_dataframe()),
    # right after the pipeline step that establishes them
    "staged_lv_grids_are_trees": lambda r: r.staged["build_lv_grids"],
    "staged_mv_routing_connects_all_nodes": lambda r: r.staged["mv_routing"],
    "staged_all_generators_connected": lambda r: r.staged["connect_generators"],
}

# (check, run) -> why ding0 violates it today; run None for all runs
B1 = ("B1: generators are only imported for the first MV grid district of a run")
B17 = ("B17: ding0's power flow set point of every sectoral load of a building is "
       "the building's total peak load")
B18_B22 = ("B18/B22: LV voltage limits are exceeded; ding0 checks the voltage drop only "
           "along the longest path of a feeder and not at all on cables of voltage level 6")
B20 = "B20: switch disconnector names repeat across MV grid districts"
B21 = ("B21: the export closes open switch disconnectors and leaves them closed, so "
       "the power flows of the reinforcement run on closed rings")
KNOWN = {
    ("ding0_set_points_match_export", None): B17,
    ("lv_voltage_band", None): B18_B22,
    ("export_keeps_switch_state", None): B21,
    ("ding0_mv_results_match_normal_operation", None): B21,
    ("mv_voltage_band", "district_2"): B21,
    ("mv_export_voltage_band", "district_2"): B21,
    ("thermal_limits", "districts_1_2"): B21,
    ("mv_export_thermal_limits", "districts_1_2"): B21,
    ("all_generators_once", "districts_1_2"): B1,
    ("unique_names", "districts_1_2"): B20,
}


def _known(check, run_name):
    return KNOWN.get((check, run_name)) or KNOWN.get((check, None))


@pytest.mark.parametrize("check", list(CHECKS))
def test_invariant(run, check, request):
    reason = _known(check, run.name)
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    assert_no_violations(CHECKS[check](run))


def test_run_reports_no_invalid_districts(run):
    assert run.messages == [[]]
