"""Physical, geographic and completeness invariants of complete ding0 runs on
the synthetic grid districts (no database needed).

Each invariant of `tests/invariants` is checked on five runs: grid district
1, grid district 2 with all components of newer egon-data versions (heat
pumps, charging points, rail, storage), both districts in one run, and the
two districts again with the planning profile `snb`. Some invariants are
also checked right after the pipeline step that establishes them
(`staged`), since later steps change the grid.

The planning checks (`tests/invariants/planning.py`) read the parameters of
the run's planning profile, which is applied again before each check.

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
from ding0.tools import planning_profiles
from tests import synthetic_district
import warnings

from tests.invariants import (
    assert_no_violations,
    completeness,
    geo,
    physics,
    planning,
    plausibility,
    powerflow,
)

# run name: (MV grid districts, with components of newer egon-data versions,
# planning profile)
RUNS = {
    "district_1": ([1], False, "default"),
    "district_2": ([2], True, "default"),
    "districts_1_2": ([1, 2], True, "default"),
    "district_1_snb": ([1], False, "snb"),
    "district_2_snb": ([2], True, "snb"),
}


class Run:
    def __init__(self, name):
        self.name = name
        self.mvgd_ids, self.components, self.profile = RUNS[name]
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

    def solved_mv(self, mvgd, closed_rings=False):
        """Independent power flow on the MV export (LV grids aggregated at
        their stations, as in ding0's own power flow), in normal operation
        or with closed rings."""
        key = ("mv", closed_rings, mvgd.id_db)
        if key not in self._solved:
            self._solved[key] = powerflow.solve(mvgd, only_export_mv=True,
                                                closed_rings=closed_rings)
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
        network = NetworkDing0(name="synthetic", planning_profile=run.profile)
        network._orm = {}
        run.messages = network.run_ding0(session=None, mv_grid_districts_no=run.mvgd_ids)
        run.network = network
        for mvgd in network.mv_grid_districts():
            run.solved_mv(mvgd)
            run.solved_mv(mvgd, closed_rings=True)
            run.solved(mvgd)
    finally:
        logging.disable(logging.NOTSET)
        monkeypatch.undo()
        planning_profiles.apply_profile(planning_profiles.DEFAULT_PROFILE)
    return run


@pytest.fixture
def profile_of(run):
    """Applies the run's planning profile for a check, the default after."""
    planning_profiles.apply_profile(run.profile)
    yield
    planning_profiles.apply_profile(planning_profiles.DEFAULT_PROFILE)


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


def per_lv_grid_solved(check):
    """Applies a check to every LV grid with the solved export of its MV
    grid district."""
    def apply(run):
        return [v for mvgd in run.mv_grid_districts()
                for la in mvgd.lv_load_areas() for lvgd in la.lv_grid_districts()
                for v in check(lvgd.lv_grid, run.solved(mvgd)[0])]
    return apply


CHECKS = {
    # physics: independent power flow on the exported MV and LV grids
    "power_flow_converges": per_district(lambda r, g: physics.check_converged(r.solved(g)[1])),
    "kirchhoff_and_ohm": per_district(lambda r, g: physics.check_kirchhoff(r.solved(g)[0])),
    "losses_positive_and_small": per_district(lambda r, g: physics.check_losses(r.solved(g)[0])),
    "energy_balance": per_district(lambda r, g: physics.check_energy_balance(r.solved(g)[0])),
    "mv_voltage_band": per_district(
        lambda r, g: physics.check_voltage_band(r.solved(g)[0], levels=("MV",))),
    "lv_voltage_band_load_case": per_district(lambda r, g: physics.check_voltage_band(
        r.solved(g)[0], levels=("LV",), snapshots=[powerflow.LOAD_CASE])),
    "lv_voltage_band_feedin_case": per_district(lambda r, g: physics.check_voltage_band(
        r.solved(g)[0], levels=("LV",), snapshots=[powerflow.FEEDIN_CASE])),
    # within 10 % of the limit: gross overloads
    "thermal_limits": per_district(
        lambda r, g: physics.check_thermal_limits(r.solved(g)[0], tolerance=0.1)),
    "thermal_limits_exact": per_district(lambda r, g: physics.check_thermal_limits(r.solved(g)[0])),
    # physics: exported data and ding0's own results
    "line_impedances_follow_type_and_length": per_district(
        lambda r, g: physics.check_line_impedances(r.export(g)[0], r.network.static_data)),
    "voltage_levels": per_district(
        lambda r, g: physics.check_voltage_levels(r.export(g)[0], g.mv_grid.v_level)),
    "radial_operation": per_district(lambda r, g: physics.check_radial(r.export(g)[0])),
    "open_switches_separate_rings": per_district(lambda r, g: physics.check_switches(r.export(g)[0])),
    "ding0_set_points_match_export": per_district(
        lambda r, g: physics.check_set_points(r.export(g)[0], r.export(g)[3])),
    "ding0_set_points_match_export_single_load_buses": per_district(
        lambda r, g: physics.check_set_points(r.export(g)[0], r.export(g)[3],
                                              only_single_load_buildings=True)),
    "ding0_power_flow_results_complete": per_district(
        lambda r, g: physics.check_ding0_powerflow_results(g.mv_grid)),
    "ding0_mv_limits_after_reinforcement": per_district(
        lambda r, g: physics.check_ding0_mv_limits(g.mv_grid)),
    "ding0_mv_results_match_normal_operation": per_district(
        lambda r, g: physics.check_ding0_results_match(g.mv_grid, r.solved_mv(g)[0])),
    "ding0_mv_results_match_closed_rings": per_district(lambda r, g: physics.check_ding0_results_match(
        g.mv_grid, r.solved_mv(g, closed_rings=True)[0])),
    "mv_export_power_flow_converges": per_district(
        lambda r, g: physics.check_converged(r.solved_mv(g)[1])),
    "mv_export_voltage_band": per_district(
        lambda r, g: physics.check_voltage_band(r.solved_mv(g)[0], levels=("MV",))),
    "mv_export_thermal_limits": per_district(
        lambda r, g: physics.check_thermal_limits(r.solved_mv(g)[0], levels=("MV",))),
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
    # planning principles of the run's profile
    "equipment_in_profile_selection": per_district(
        lambda r, g: planning.check_equipment_in_selection(g, r.network.static_data)),
    "voltage_level_by_distance": per_district(lambda r, g: planning.check_voltage_level_choice(g)),
    "half_ring_lengths": per_district(lambda r, g: planning.check_half_ring_lengths(g.mv_grid)),
    "satellites_by_peak_load": per_district(lambda r, g: planning.check_satellites(g)),
    "generators_at_their_voltage_level": per_district(
        lambda r, g: planning.check_generator_levels(g)),
    "lv_feeders_within_standard_cable": per_lv_grid(planning.check_lv_feeder_load),
    "mv_rings_n_minus_1": per_district(lambda r, g: planning.check_rings_n_minus_1(
        g.mv_grid, r.solved_mv(g)[0], r.solved_mv(g)[2][0])),
    # not planned larger than the planning principles require
    "hvmv_transformers_minimal": per_district(
        lambda r, g: planning.check_hvmv_transformers_minimal(g)),
    "lv_transformers_minimal": per_lv_grid(planning.check_lv_transformers_minimal),
    "mv_reinforcement_necessary": per_district(lambda r, g: planning.check_mv_reinforcement_necessary(
        g.mv_grid, r.solved_mv(g)[0])),
    "lv_feeder_cables_minimal": per_lv_grid_solved(planning.check_lv_feeders_minimal),
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

# why ding0 violates an invariant today, per finding
B1 = "B1: generators are only imported for the first MV grid district of a run"
B17 = ("B17: ding0's power flow set point of every sectoral load of a building is "
       "the building's total peak load")
B18_B22 = ("B18/B22: LV voltage limits are exceeded; ding0 checks the voltage drop only "
           "along the longest path of a feeder and not at all on cables of voltage level 6")
B20 = "B20: switch disconnector names repeat across MV grid districts"
B21 = ("B21: the export closes open switch disconnectors and leaves them closed, so "
       "the power flows of the reinforcement run on closed rings")
B24 = ("B24: ding0 sizes LV cables and transformers for I = S / (sqrt(3) U_n) of the load "
       "below at load factor 1.0, without losses and without the lower voltage along the "
       "feeder; the current reaches up to 108 % of the rating")
B25 = ("B25: the LV voltage rise limit of the feed-in case (lv_max_v_level_fc_diff_normal, "
       "3 %) is only applied by the LV reinforcement, which is switched off; rises reach "
       "3.1 to 4.0 %")
B26 = ("B26: HV/MV transformers are sized with the normal load factor (0.6) and then get a "
       "redundant transformer of the largest size, so (n-1) is covered twice; "
       "load_factor_mv_trans_lc_malfunc is never read (district 2: 2 x 40 MVA where "
       "2 x 25 MVA are (n-1)-secure)")
B27 = ("B27: (n-1) of MV rings is only checked while routing; stations, satellites and MV "
       "loads connected to the rings afterwards are not (snb district 2: 172 % loading, "
       "17.9 % voltage drop)")
B28_B31 = ("B28/B31: LV feeders above the rating of their cable: with profile snb loads up to "
           "346 kW connect at the station, but no LV cable above NAYY 4x1x150 is selected "
           "(reinforce_cable_type silently returns the largest, B14); and when a subtree "
           "needs more feeders than it has nodes, all its nodes become one feeder although "
           "the message says every node becomes a feeder (up to 149 %)")
B29 = ("B29: the MV reinforcement stops silently when the largest selected cable is too "
       "small and builds no parallel lines (snb district 2: 97 % against 50 % with an "
       "8 MW MV load at 10 kV)")
B30 = ("B30: the MV reinforcement upgrades cables that are not needed in normal operation: "
       "it decides on closed rings (B21) with overstated set points (B17) and upgrades every "
       "branch on the paths to critical nodes (district 1: 3 cables)")

ALL = list(RUNS)
DEFAULT = ["district_1", "district_2", "districts_1_2"]
SNB = ["district_1_snb", "district_2_snb"]
# check -> {run: reason}
KNOWN = {
    "ding0_set_points_match_export": dict.fromkeys(ALL, B17),
    "lv_voltage_band_load_case": dict.fromkeys(DEFAULT + ["district_2_snb"], B18_B22),
    "lv_voltage_band_feedin_case": dict.fromkeys(ALL, B25),
    "thermal_limits_exact": dict.fromkeys(ALL, B24),
    "thermal_limits": dict.fromkeys(SNB, B28_B31),
    "lv_cables_carry_load_and_generation": dict.fromkeys(SNB, B28_B31),
    "lv_feeders_within_standard_cable": dict.fromkeys(SNB, B28_B31),
    "export_keeps_switch_state": dict.fromkeys(ALL, B21),
    "ding0_mv_results_match_normal_operation": dict.fromkeys(ALL, B21),
    "mv_voltage_band": dict.fromkeys(["district_2"] + SNB, B21),
    "mv_export_voltage_band": dict.fromkeys(["district_2"] + SNB, B21),
    "mv_export_thermal_limits": {"districts_1_2": B21, "district_2_snb": B29},
    "ding0_mv_limits_after_reinforcement": {"district_2_snb": B29},
    "mv_rings_n_minus_1": {"district_2_snb": B27},
    "hvmv_transformers_minimal": dict.fromkeys(["district_2", "districts_1_2", "district_2_snb"], B26),
    "mv_reinforcement_necessary": dict.fromkeys(["district_1", "districts_1_2", "district_2_snb"], B30),
    "all_generators_once": {"districts_1_2": B1},
    "unique_names": {"districts_1_2": B20},
}


def _known(check, run_name):
    return KNOWN.get(check, {}).get(run_name)


@pytest.mark.parametrize("check", list(CHECKS))
def test_invariant(run, profile_of, check, request):
    reason = _known(check, run.name)
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    assert_no_violations(CHECKS[check](run))


def test_plausibility(run, profile_of):
    """Grid properties outside the ranges usual in DSO planning are reported
    as warnings, not failures (`tests/invariants/plausibility.py`)."""
    for mvgd in run.mv_grid_districts():
        messages = plausibility.mv_grid_messages(run.solved_mv(mvgd)[0])
        messages += [m for la in mvgd.lv_load_areas() for lvgd in la.lv_grid_districts()
                     for m in plausibility.lv_grid_messages(lvgd.lv_grid)]
        for message in messages:
            warnings.warn(f"{run.name}, {mvgd}: {message}", plausibility.PlausibilityWarning)


def test_run_reports_no_invalid_districts(run, request):
    if run.name == "district_2_snb":
        request.applymarker(pytest.mark.xfail(reason=B28_B31, strict=True))
    assert run.messages == [[]]
