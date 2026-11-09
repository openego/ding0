"""Equipment data and the selection of cables, lines and transformers."""

from math import ceil, sqrt
from types import SimpleNamespace

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import LineString, Point

from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.core.network import BranchDing0
from ding0.core.network.cable_distributors import MVCableDistributorDing0
from ding0.core.network.grids import MVGridDing0
from ding0.core.network.stations import MVStationDing0
from ding0.grid.lv_grid.build_grid import select_transformers
from ding0.grid.lv_grid.build_grid_on_osm_ways import get_n_feeder_mandatory, reinforce_cable_type
from ding0.grid.lv_grid.parameterization import get_peak_load_diversity, get_peak_load_for_residential
from ding0.grid.tools import cable_type
from ding0.tools import config as cfg_ding0
from ding0.tools.egon_data_integration import voltage_level_from_power
from ding0.tools.pypsa_io import append_lines_df, initialize_component_dataframes
from tests.invariants import assert_no_violations, physics


# Equipment data ##############################################################

@pytest.mark.xfail(strict=True, reason=(
    "B19: NAYY 4x1x50 has the resistance of 70 mm2 (0.449 instead of about "
    "0.641 ohm/km) and the overhead lines 48/94/122-AL1 almost the same "
    "resistance (0.35/0.33/0.31 instead of about 0.59/0.31/0.24 ohm/km)"))
def test_conductor_resistance_is_inversely_proportional_to_cross_section(static_data):
    assert_no_violations(physics.check_equipment_catalog(static_data))


def test_lv_transformer_equivalent_circuit_follows_datasheet(static_data):
    """r_pu = P_k / S_nom, x_pu = sqrt(u_k^2 - r_pu^2), e.g. 100 kVA with
    P_k 1750 W and u_k 4 %: r_pu 0.0175."""
    assert_no_violations(physics.check_transformer_data(static_data))
    trafo = static_data["LV_trafos"].set_index("S_nom").loc[100]
    assert trafo.r_pu == pytest.approx(0.0175)
    assert trafo.x_pu == pytest.approx(sqrt(0.04 ** 2 - 0.0175 ** 2))


# Export of line impedances ###################################################

def exported_line(static_data, num_parallel):
    """A 2 km MV cable between two cable distributors, exported."""
    station = MVStationDing0(id_db=0, geo_data=Point(0, 0))
    grid = MVGridDing0(id_db=0, station=station)
    grid.v_level = 10
    a = MVCableDistributorDing0(grid=grid, geo_data=Point(0, 0))
    grid.add_cable_distributor(a)
    b = MVCableDistributorDing0(grid=grid, geo_data=Point(2000, 0))
    grid.add_cable_distributor(b)
    cables = static_data["MV_cables"]
    branch = BranchDing0(grid=grid, length=2000.0, kind="cable",
                         type=cables.loc[cables["name"] == "NA2XS2Y 3x1x240 RM/25"].iloc[0],
                         geometry=LineString([(0, 0), (2000, 0)]))
    branch.num_parallel = num_parallel
    grid.graph.add_edge(a, b, branch=branch)  # a branch is named after its nodes
    buses =pd.DataFrame({"v_nom": [10.0, 10.0], "in_building": [False, False]},
                         index=[a.pypsa_bus_id, b.pypsa_bus_id])
    lines = append_lines_df({"branch": branch, "adj_nodes": (a, b)},
                            initialize_component_dataframes()[2], buses)
    return {"Line": lines.set_index("name"), "Bus": buses}


def test_exported_impedance_of_a_single_line(static_data):
    """r = R' l, x = 2 pi f L' l, s_nom = sqrt(3) U I_max."""
    components = exported_line(static_data, 1)
    assert_no_violations(physics.check_line_impedances(components, static_data))
    line = components["Line"].iloc[0]
    assert line.r == pytest.approx(0.125 * 2)
    assert line.s_nom == pytest.approx(sqrt(3) * 10 * 417 / 1e3)


@pytest.mark.xfail(strict=True, reason=(
    "B3 (latent, ding0 never sets num_parallel > 1 itself): the export does "
    "not divide r and x by num_parallel nor multiply s_nom"))
def test_exported_impedance_of_parallel_lines(static_data):
    assert_no_violations(physics.check_line_impedances(exported_line(static_data, 2), static_data))


# Selection of cables and transformers ########################################

currents = st.floats(min_value=1.0, max_value=600.0)


@settings(max_examples=200, deadline=None)
@given(power=st.floats(min_value=1.0, max_value=280.0))
def test_cable_type_is_the_smallest_sufficient_cable(static_data, power):
    """`grid.tools.cable_type` chooses the cable with the lowest ampacity
    above the current I = S / (sqrt(3) U)."""
    cables = static_data["LV_cables"]
    current = power / (sqrt(3) * 0.4)
    chosen = cable_type(power, 0.4, cables)
    sufficient = cables.loc[cables.I_max_th > current]
    assert chosen.I_max_th > current
    assert chosen.I_max_th == sufficient.I_max_th.min()


@pytest.mark.xfail(strict=True, reason=(
    "B14: grid.tools.cable_type returns the largest cable without a warning "
    "when no cable carries the power"))
def test_cable_type_warns_when_no_cable_suffices(static_data, caplog):
    cables = static_data["LV_cables"]
    chosen = cable_type(400.0, 0.4, cables)  # 577 A
    assert chosen.I_max_th == cables.I_max_th.max()
    assert any(r.levelname == "WARNING" for r in caplog.records)


@settings(max_examples=200, deadline=None)
@given(current=currents)
def test_reinforce_cable_type(lvgd_cfg, current):
    """The smallest cable with I_max * load factor above the current, or the
    largest one flagged as not sufficient."""
    cables = lvgd_cfg["lv_cables_df"]
    lf = lvgd_cfg["lv_cable_lf"]
    chosen, reinforcable = reinforce_cable_type(current, lvgd_cfg)
    sufficient = cables.loc[cables.I_max_th * lf > current]
    if len(sufficient):
        assert reinforcable
        assert chosen.I_max_th == sufficient.I_max_th.min()
    else:
        assert not reinforcable
        assert chosen.I_max_th == cables.I_max_th.max()


@settings(max_examples=100, deadline=None)
@given(capacity=st.floats(min_value=1.0, max_value=2000.0))
def test_number_of_feeders_carries_the_load(lvgd_cfg, capacity):
    """n feeders of the standard cable carry the current of the load."""
    standard = lvgd_cfg["lv_cables_df"].loc[cfg_ding0.get("assumptions", "lv_standard_line")]
    current = capacity / (sqrt(3) * 0.4) / lvgd_cfg["cos_phi_load"]
    n = get_n_feeder_mandatory(capacity, lvgd_cfg)
    assert n == ceil(current / standard.I_max_th)
    assert n * standard.I_max_th >= current


@settings(max_examples=200, deadline=None)
@given(s_max=st.floats(min_value=1.0, max_value=5000.0), case=st.sampled_from(["load", "gen"]))
def test_lv_transformers_carry_the_power(static_data, s_max, case):
    """Count times rated power times load factor exceeds the power, with the
    smallest transformer type that does so for this count, and one
    transformer less of the largest type would not."""
    grid = SimpleNamespace(network=SimpleNamespace(static_data=static_data))
    lf = cfg_ding0.get("assumptions",
                       "load_factor_lv_trans_lc_normal" if case == "load"
                       else "load_factor_lv_trans_fc_normal")
    trafo, count = select_transformers(grid, s_max={"s_max": s_max, "case": case})
    sizes = static_data["LV_trafos"].S_nom
    assert count * trafo.S_nom * lf > s_max
    assert not any((s < trafo.S_nom) and (count * s * lf > s_max) for s in sizes)
    if count > 1:
        assert (count - 1) * sizes.max() * lf <= s_max


# Loads and voltage levels ####################################################

def test_residential_peak_load_vdew():
    """VDEW: 30 kW for one household, falling towards 0.06 * 30 = 1.8 kW."""
    assert get_peak_load_for_residential(1) == pytest.approx(30.0)
    values = [get_peak_load_for_residential(n) for n in range(1, 500)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] > 1.8


@settings(max_examples=100, deadline=None)
@given(households=st.lists(st.integers(1, 20), min_size=1, max_size=30),
       cts=st.floats(0, 500))
def test_diversity_peak_load_not_above_the_sum(households, cts):
    """The peak load with simultaneity is at most the sum of the buildings'
    peak loads, as given by the VDEW formula per household."""
    buildings = pd.DataFrame({
        "number_households": households,
        "residential_capacity": [30.0 * n for n in households],
        "cts_capacity": [cts] + [0.0] * (len(households) - 1),
        "industrial_capacity": 0.0,
    })
    diversity = get_peak_load_diversity(buildings)
    assert diversity <= buildings[["residential_capacity", "cts_capacity"]].sum().sum() + 1e-9


def test_voltage_level_from_power_limits():
    """Upper limits (kW) as in eDisGo, limits inclusive."""
    limits = {7: 100, 6: 200, 5: 5500, 4: 20000}
    for level, limit in limits.items():
        assert voltage_level_from_power(limit) == level
        assert voltage_level_from_power(limit + 0.1) == max(level - 1, 4)


@settings(max_examples=200, deadline=None)
@given(a=st.floats(0, 50000), b=st.floats(0, 50000))
def test_voltage_level_from_power_is_monotonic(a, b):
    low, high = sorted([a, b])
    assert voltage_level_from_power(low) >= voltage_level_from_power(high)


def test_building_thresholds_partition_the_capacities():
    """LV street, LV station, MV and above-MV classes adjoin without gap or
    overlap."""
    lv = get_config_osm("lv_threshold_capacity")
    mv_lv = get_config_osm("mv_lv_threshold_capacity")
    hv_mv = get_config_osm("hv_mv_threshold_capacity")
    assert 0 < lv < mv_lv < hv_mv
